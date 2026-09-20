from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np


class BackpropModel(Protocol):
    weights: list[np.ndarray]
    biases: list[np.ndarray]

    def forward(self, values: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]: ...

    def backward(
        self,
        activations: list[np.ndarray],
        pre_activations: list[np.ndarray],
        targets: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]: ...

    def error_func(self, targets: np.ndarray, predictions: np.ndarray) -> float: ...


@dataclass(frozen=True)
class GradientCheckPolicy:
    epsilon: float = 1e-5
    absolute_tolerance: float = 1e-7
    relative_tolerance: float = 1e-4
    max_parameters: int | None = 500
    seed: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("epsilon", self.epsilon),
            ("absolute_tolerance", self.absolute_tolerance),
            ("relative_tolerance", self.relative_tolerance),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_parameters is not None and self.max_parameters < 1:
            raise ValueError("max_parameters must be positive or None")


@dataclass(frozen=True)
class GradientError:
    parameter: str
    analytic: float
    numerical: float
    absolute_error: float
    relative_error: float
    passed: bool


@dataclass(frozen=True)
class GradientCheckReport:
    passed: bool
    checked_parameters: int
    total_parameters: int
    max_absolute_error: float
    max_relative_error: float
    mean_relative_error: float
    failures: tuple[GradientError, ...]
    policy: GradientCheckPolicy

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checked_parameters": self.checked_parameters,
            "total_parameters": self.total_parameters,
            "max_absolute_error": self.max_absolute_error,
            "max_relative_error": self.max_relative_error,
            "mean_relative_error": self.mean_relative_error,
            "failures": [asdict(item) for item in self.failures],
            "policy": asdict(self.policy),
        }


def _validate_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def check_gradients(
    model: BackpropModel,
    values: np.ndarray,
    targets: np.ndarray,
    policy: GradientCheckPolicy | None = None,
) -> GradientCheckReport:
    """Compare backprop gradients with central finite differences."""

    active_policy = policy or GradientCheckPolicy()
    values = _validate_array("values", values)
    targets = _validate_array("targets", targets)
    activations, pre_activations = model.forward(values)
    analytic_weights, analytic_biases = model.backward(activations, pre_activations, targets)

    parameters: list[tuple[str, np.ndarray, np.ndarray, tuple[int, ...]]] = []
    for kind, tensors, gradients in (
        ("weight", model.weights, analytic_weights),
        ("bias", model.biases, analytic_biases),
    ):
        if len(tensors) != len(gradients):
            raise ValueError(f"{kind} gradient layer count mismatch")
        for layer, (tensor, gradient) in enumerate(zip(tensors, gradients, strict=True)):
            if tensor.shape != gradient.shape:
                raise ValueError(f"{kind} gradient shape mismatch at layer {layer}")
            if not np.all(np.isfinite(gradient)):
                raise ValueError(f"{kind} gradient is non-finite at layer {layer}")
            for index in np.ndindex(tensor.shape):
                parameters.append((f"{kind}[{layer}]{index}", tensor, gradient, index))

    if not parameters:
        raise ValueError("model has no parameters")
    selected = np.arange(len(parameters))
    if active_policy.max_parameters is not None and len(selected) > active_policy.max_parameters:
        selected = np.sort(
            np.random.default_rng(active_policy.seed).choice(
                selected, size=active_policy.max_parameters, replace=False
            )
        )

    results: list[GradientError] = []
    for position in selected:
        name, tensor, analytic_tensor, index = parameters[int(position)]
        original = float(tensor[index])
        try:
            tensor[index] = original + active_policy.epsilon
            plus = float(model.error_func(targets, model.forward(values)[0][-1]))
            tensor[index] = original - active_policy.epsilon
            minus = float(model.error_func(targets, model.forward(values)[0][-1]))
        finally:
            tensor[index] = original
        if not math.isfinite(plus) or not math.isfinite(minus):
            raise ValueError(f"non-finite perturbed loss for {name}")

        numerical = (plus - minus) / (2 * active_policy.epsilon)
        analytic = float(analytic_tensor[index])
        absolute_error = abs(analytic - numerical)
        scale = max(abs(analytic), abs(numerical), active_policy.absolute_tolerance)
        relative_error = absolute_error / scale
        passed = absolute_error <= (
            active_policy.absolute_tolerance + active_policy.relative_tolerance * scale
        )
        results.append(
            GradientError(name, analytic, numerical, absolute_error, relative_error, passed)
        )

    failures = tuple(
        sorted((item for item in results if not item.passed), key=lambda x: -x.relative_error)
    )
    return GradientCheckReport(
        passed=not failures,
        checked_parameters=len(results),
        total_parameters=len(parameters),
        max_absolute_error=max(item.absolute_error for item in results),
        max_relative_error=max(item.relative_error for item in results),
        mean_relative_error=float(np.mean([item.relative_error for item in results])),
        failures=failures,
        policy=active_policy,
    )
