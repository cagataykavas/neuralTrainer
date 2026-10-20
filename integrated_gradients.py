from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from neural_trainer import NeuralNetwork, get_activation_funcs

MAX_INTEGRATION_STEPS = 4096
SUPPORTED_HIDDEN_ACTIVATIONS = {
    "leakyrelu",
    "leaky_relu",
    "relu",
    "sigmoid",
    "softplus",
    "tanh",
}


@dataclass(frozen=True)
class IntegratedGradientsReport:
    target_index: int
    steps: int
    attributions: tuple[float, ...]
    baseline_output: float
    input_output: float
    output_difference: float
    attribution_sum: float
    completeness_delta: float
    model_digest: str
    input_digest: str
    baseline_digest: str

    def passes_completeness(
        self,
        *,
        absolute_tolerance: float = 1e-5,
        relative_tolerance: float = 1e-4,
    ) -> bool:
        """Check the Integrated Gradients completeness property."""

        _validate_tolerance(absolute_tolerance, "absolute_tolerance")
        _validate_tolerance(relative_tolerance, "relative_tolerance")
        allowed_error = absolute_tolerance + relative_tolerance * abs(
            self.output_difference
        )
        return abs(self.completeness_delta) <= allowed_error

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        payload["explanation_id"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        return payload


def _validate_tolerance(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _single_column(values: Any, *, rows: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape != (rows, 1):
        raise ValueError(f"{name} must have shape ({rows}, 1); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _array_digest(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(json.dumps(canonical.shape, separators=(",", ":")).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _network_digest(network: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(network.input_size).encode("ascii"))
    digest.update(b"|")
    digest.update(str(network.output_size).encode("ascii"))
    for activation in network.activations:
        digest.update(b"|")
        digest.update(activation.lower().encode("utf-8"))
    for weights, biases in zip(network.weights, network.biases, strict=True):
        digest.update(b"|")
        digest.update(_array_digest(weights).encode("ascii"))
        digest.update(b"|")
        digest.update(_array_digest(biases).encode("ascii"))
    return digest.hexdigest()


def _validate_network(network: Any) -> None:
    for attribute in ("input_size", "output_size", "weights", "biases", "activations"):
        if not hasattr(network, attribute):
            raise TypeError(f"network is missing required attribute {attribute!r}")
    if not callable(getattr(network, "forward", None)) or not callable(
        getattr(network, "predict", None)
    ):
        raise TypeError("network must provide callable forward() and predict() methods")
    if isinstance(network.input_size, bool) or not isinstance(network.input_size, int):
        raise TypeError("network.input_size must be an integer")
    if isinstance(network.output_size, bool) or not isinstance(
        network.output_size, int
    ):
        raise TypeError("network.output_size must be an integer")
    if network.input_size <= 0 or network.output_size <= 0:
        raise ValueError("network input and output sizes must be positive")
    if not network.weights or len(network.weights) != len(network.biases):
        raise ValueError("network must have matching, non-empty weights and biases")
    if len(network.activations) != len(network.weights) - 1:
        raise ValueError("network must define one activation per hidden layer")

    expected_inputs = network.input_size
    for layer, (weights, biases) in enumerate(
        zip(network.weights, network.biases, strict=True)
    ):
        if not isinstance(weights, np.ndarray) or not isinstance(biases, np.ndarray):
            raise TypeError(f"layer {layer} parameters must be NumPy arrays")
        if (
            weights.ndim != 2
            or weights.shape[1] != expected_inputs
            or weights.shape[0] == 0
        ):
            raise ValueError(
                f"layer {layer} has an invalid weight shape {weights.shape}"
            )
        if biases.shape != (weights.shape[0], 1):
            raise ValueError(f"layer {layer} has an invalid bias shape {biases.shape}")
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(biases)):
            raise ValueError(f"layer {layer} parameters must be finite")
        expected_inputs = weights.shape[0]
    if expected_inputs != network.output_size:
        raise ValueError("final layer size does not match network.output_size")

    for layer, activation in enumerate(network.activations):
        if not isinstance(activation, str):
            raise TypeError(f"hidden layer {layer} activation must be a string")
        normalized = activation.lower()
        if normalized == "softmax":
            raise ValueError(
                f"hidden layer {layer} uses softmax, whose element-wise derivative is not valid for attribution"
            )
        if normalized not in SUPPORTED_HIDDEN_ACTIVATIONS:
            raise ValueError(
                f"hidden layer {layer} uses unsupported activation {activation!r}"
            )


def _validate_target(network: Any, target_index: int) -> None:
    if isinstance(target_index, bool) or not isinstance(target_index, int):
        raise TypeError("target_index must be an integer")
    if not 0 <= target_index < network.output_size:
        raise ValueError(f"target_index must be in [0, {network.output_size})")


def _validate_steps(steps: int) -> None:
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be an integer")
    if not 2 <= steps <= MAX_INTEGRATION_STEPS:
        raise ValueError(f"steps must be between 2 and {MAX_INTEGRATION_STEPS}")


def input_gradient(
    network: NeuralNetwork,
    sample: Any,
    *,
    target_index: int = 0,
) -> np.ndarray:
    """Return the exact input gradient for one linear-output network sample."""

    _validate_network(network)
    _validate_target(network, target_index)
    point = _single_column(sample, rows=network.input_size, name="sample")
    return _input_gradient_validated(network, point, target_index=target_index)


def _input_gradient_validated(
    network: NeuralNetwork,
    point: np.ndarray,
    *,
    target_index: int,
) -> np.ndarray:
    _, pre_activations = network.forward(point)
    if len(pre_activations) != len(network.weights):
        raise ValueError("network forward pass returned the wrong number of layers")
    for layer, values in enumerate(pre_activations):
        expected_shape = (network.weights[layer].shape[0], 1)
        if not isinstance(values, np.ndarray) or values.shape != expected_shape:
            raise ValueError(
                f"hidden layer {layer} returned an invalid pre-activation shape"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"hidden layer {layer} returned non-finite pre-activations"
            )

    gradient = network.weights[-1][target_index : target_index + 1, :].copy()
    for layer in range(len(network.weights) - 2, -1, -1):
        _, derivative = get_activation_funcs(network.activations[layer])
        activation_gradient = np.asarray(
            derivative(pre_activations[layer]), dtype=np.float64
        )
        if activation_gradient.shape != pre_activations[layer].shape:
            raise ValueError(
                f"hidden layer {layer} derivative returned an invalid shape"
            )
        if not np.all(np.isfinite(activation_gradient)):
            raise ValueError(
                f"hidden layer {layer} derivative returned non-finite values"
            )
        gradient = (gradient * activation_gradient.T) @ network.weights[layer]

    result = gradient.T
    if result.shape != point.shape or not np.all(np.isfinite(result)):
        raise ValueError("input gradient is invalid or non-finite")
    return result


def integrated_gradients(
    network: NeuralNetwork,
    sample: Any,
    *,
    baseline: Any | None = None,
    target_index: int = 0,
    steps: int = 128,
) -> IntegratedGradientsReport:
    """Attribute one selected output using bounded trapezoidal integration."""

    _validate_network(network)
    _validate_target(network, target_index)
    _validate_steps(steps)
    model_digest = _network_digest(network)
    point = _single_column(sample, rows=network.input_size, name="sample")
    reference = (
        np.zeros_like(point)
        if baseline is None
        else _single_column(baseline, rows=network.input_size, name="baseline")
    )
    difference = point - reference

    gradient_sum = np.zeros_like(point)
    for index in range(steps + 1):
        alpha = index / steps
        weight = 0.5 if index in {0, steps} else 1.0
        interpolated = reference + alpha * difference
        gradient_sum += weight * _input_gradient_validated(
            network,
            interpolated,
            target_index=target_index,
        )
    attributions = difference * (gradient_sum / steps)

    baseline_prediction = np.asarray(network.predict(reference), dtype=np.float64)
    input_prediction = np.asarray(network.predict(point), dtype=np.float64)
    expected_shape = (network.output_size, 1)
    if (
        baseline_prediction.shape != expected_shape
        or input_prediction.shape != expected_shape
    ):
        raise ValueError(f"network predictions must have shape {expected_shape}")
    if not np.all(np.isfinite(baseline_prediction)) or not np.all(
        np.isfinite(input_prediction)
    ):
        raise ValueError("network predictions must be finite")

    baseline_output = float(baseline_prediction[target_index, 0])
    input_output = float(input_prediction[target_index, 0])
    output_difference = input_output - baseline_output
    attribution_sum = float(np.sum(attributions))
    completeness_delta = output_difference - attribution_sum
    if not all(
        math.isfinite(value)
        for value in (
            baseline_output,
            input_output,
            output_difference,
            attribution_sum,
            completeness_delta,
        )
    ):
        raise ValueError("attribution report contains non-finite values")
    if _network_digest(network) != model_digest:
        raise RuntimeError("network parameters changed during attribution")

    return IntegratedGradientsReport(
        target_index=target_index,
        steps=steps,
        attributions=tuple(float(value) for value in attributions[:, 0]),
        baseline_output=baseline_output,
        input_output=input_output,
        output_difference=output_difference,
        attribution_sum=attribution_sum,
        completeness_delta=completeness_delta,
        model_digest=model_digest,
        input_digest=_array_digest(point),
        baseline_digest=_array_digest(reference),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a deterministic Integrated Gradients demo."
    )
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()

    network = NeuralNetwork(
        input_size=3,
        hidden_layers=[4],
        output_size=1,
        activations=["tanh"],
        seed=42,
    )
    report = integrated_gradients(
        network,
        np.array([[1.0], [2.0], [3.0]]),
        baseline=np.zeros((3, 1)),
        steps=args.steps,
    )
    payload = report.to_dict()
    payload["passes_completeness"] = report.passes_completeness(
        absolute_tolerance=1e-4,
        relative_tolerance=1e-3,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
