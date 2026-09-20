from __future__ import annotations

import json

import numpy as np
import pytest

from gradient_check import GradientCheckPolicy, check_gradients


class LinearModel:
    def __init__(self, *, wrong_gradient: bool = False) -> None:
        self.weights = [np.array([[0.4, -0.2]], dtype=float)]
        self.biases = [np.array([[0.1]], dtype=float)]
        self.wrong_gradient = wrong_gradient

    def forward(self, values):
        predictions = self.weights[0] @ values + self.biases[0]
        return [values, predictions], [predictions]

    def error_func(self, targets, predictions):
        return np.mean((targets - predictions) ** 2)

    def backward(self, activations, _pre_activations, targets):
        derivative = 2 * (activations[-1] - targets) / targets.size
        weight_gradient = derivative @ activations[0].T
        if self.wrong_gradient:
            weight_gradient = -weight_gradient
        return [weight_gradient], [np.sum(derivative, axis=1, keepdims=True)]


VALUES = np.array([[1.0, 2.0], [0.5, -1.0]])
TARGETS = np.array([[0.7, -0.1]])


def test_accepts_correct_backprop_and_preserves_parameters():
    model = LinearModel()
    original_weights = model.weights[0].copy()
    original_biases = model.biases[0].copy()

    report = check_gradients(model, VALUES, TARGETS)

    assert report.passed
    assert report.checked_parameters == 3
    assert report.max_relative_error < 1e-6
    np.testing.assert_array_equal(model.weights[0], original_weights)
    np.testing.assert_array_equal(model.biases[0], original_biases)
    assert json.loads(json.dumps(report.to_dict()))["passed"] is True


def test_reports_parameter_level_backprop_failure():
    report = check_gradients(LinearModel(wrong_gradient=True), VALUES, TARGETS)

    assert not report.passed
    assert {failure.parameter for failure in report.failures} == {
        "weight[0](0, 0)",
        "weight[0](0, 1)",
    }


def test_parameter_sampling_is_deterministic_and_reported():
    policy = GradientCheckPolicy(max_parameters=2, seed=7)

    first = check_gradients(LinearModel(), VALUES, TARGETS, policy)
    second = check_gradients(LinearModel(), VALUES, TARGETS, policy)

    assert first.to_dict() == second.to_dict()
    assert first.checked_parameters == 2
    assert first.total_parameters == 3


def test_rejects_shape_mismatch_and_non_finite_evidence():
    model = LinearModel()
    model.backward = lambda *_args: ([np.zeros((2, 2))], [np.zeros((1, 1))])
    with pytest.raises(ValueError, match="shape mismatch"):
        check_gradients(model, VALUES, TARGETS)
    with pytest.raises(ValueError, match="finite"):
        check_gradients(LinearModel(), np.array([[np.nan]]), TARGETS)


def test_restores_parameter_when_perturbed_loss_fails():
    model = LinearModel()
    original = model.weights[0].copy()
    model.error_func = lambda *_args: float("nan")

    with pytest.raises(ValueError, match="non-finite perturbed loss"):
        check_gradients(model, VALUES, TARGETS)

    np.testing.assert_array_equal(model.weights[0], original)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epsilon": 0},
        {"absolute_tolerance": -1},
        {"relative_tolerance": float("inf")},
        {"max_parameters": 0},
    ],
)
def test_rejects_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        GradientCheckPolicy(**kwargs)
