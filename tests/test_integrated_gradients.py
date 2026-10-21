import copy

import numpy as np
import pytest

from integrated_gradients import (
    audit_baseline_stability,
    input_gradient,
    integrated_gradients,
)
from neural_trainer import NeuralNetwork


def linear_network(weights: np.ndarray, biases: np.ndarray) -> NeuralNetwork:
    return NeuralNetwork(
        input_size=weights.shape[1],
        hidden_layers=[],
        output_size=weights.shape[0],
        activations=[],
        manual_params=[(weights, biases)],
    )


def test_linear_model_has_exact_feature_attributions() -> None:
    network = linear_network(np.array([[2.0, -3.0]]), np.array([[0.5]]))

    report = integrated_gradients(
        network,
        np.array([4.0, 5.0]),
        baseline=np.zeros(2),
        steps=2,
    )

    assert report.attributions == pytest.approx((8.0, -15.0))
    assert report.output_difference == pytest.approx(-7.0)
    assert report.attribution_sum == pytest.approx(-7.0)
    assert report.completeness_delta == pytest.approx(0.0)
    assert report.passes_completeness()


def test_target_index_selects_one_output() -> None:
    network = linear_network(
        np.array([[1.0, 2.0], [-3.0, 4.0]]),
        np.array([[0.1], [-0.2]]),
    )

    report = integrated_gradients(network, [2.0, 3.0], target_index=1, steps=2)

    assert report.attributions == pytest.approx((-6.0, 12.0))
    assert report.output_difference == pytest.approx(6.0)


def test_input_gradient_matches_linear_coefficients() -> None:
    network = linear_network(np.array([[1.5, -2.5, 0.25]]), np.array([[8.0]]))

    gradient = input_gradient(network, [10.0, 20.0, 30.0])

    np.testing.assert_allclose(gradient[:, 0], [1.5, -2.5, 0.25])


def test_tanh_network_satisfies_completeness_with_bounded_error() -> None:
    network = NeuralNetwork(
        input_size=2,
        hidden_layers=[2],
        output_size=1,
        activations=["tanh"],
        manual_params=[
            (np.array([[0.6, -0.2], [0.1, 0.5]]), np.array([[0.1], [-0.2]])),
            (np.array([[1.2, -0.7]]), np.array([[0.3]])),
        ],
    )

    report = integrated_gradients(
        network,
        [1.4, -0.8],
        baseline=[0.0, 0.0],
        steps=512,
    )

    assert report.passes_completeness(absolute_tolerance=1e-6, relative_tolerance=1e-5)
    assert abs(report.completeness_delta) < 1e-6


def test_relu_path_attribution_is_directionally_correct() -> None:
    network = NeuralNetwork(
        input_size=1,
        hidden_layers=[1],
        output_size=1,
        activations=["relu"],
        manual_params=[
            (np.array([[1.0]]), np.array([[0.0]])),
            (np.array([[3.0]]), np.array([[0.0]])),
        ],
    )

    report = integrated_gradients(network, [2.0], baseline=[-1.0], steps=300)

    assert report.attributions[0] == pytest.approx(6.0, abs=0.02)
    assert report.output_difference == pytest.approx(6.0)
    assert report.passes_completeness(absolute_tolerance=0.02, relative_tolerance=0.0)


def test_report_identity_binds_model_input_and_baseline() -> None:
    network = linear_network(np.array([[2.0, -3.0]]), np.array([[0.5]]))
    first = integrated_gradients(network, [4.0, 5.0], steps=8).to_dict()
    second = integrated_gradients(network, [4.0, 5.0], steps=8).to_dict()
    changed_input = integrated_gradients(network, [4.0, 6.0], steps=8).to_dict()

    assert first == second
    assert len(first["explanation_id"]) == 64
    assert first["explanation_id"] != changed_input["explanation_id"]
    assert first["input_digest"] != changed_input["input_digest"]


def test_attribution_does_not_mutate_model_parameters() -> None:
    network = NeuralNetwork(2, [3], 1, activations=["softplus"], seed=17)
    weights_before = copy.deepcopy(network.weights)
    biases_before = copy.deepcopy(network.biases)

    integrated_gradients(network, [0.5, -0.25], steps=32)

    for actual, expected in zip(network.weights, weights_before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(network.biases, biases_before, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_model_mutation_during_attribution_is_rejected() -> None:
    class MutatingNetwork(NeuralNetwork):
        def forward(self, values):
            result = super().forward(values)
            self.weights[0][0, 0] += 1e-6
            return result

    network = MutatingNetwork(1, [], 1, activations=[], seed=3)

    with pytest.raises(RuntimeError, match="changed during attribution"):
        integrated_gradients(network, [1.0], steps=2)


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        ([1.0], "shape"),
        ([[1.0, 2.0], [3.0, 4.0]], "shape"),
        ([1.0, float("nan")], "finite"),
        ([1.0, float("inf")], "finite"),
    ],
)
def test_invalid_samples_fail_closed(sample: object, message: str) -> None:
    network = linear_network(np.array([[1.0, 1.0]]), np.array([[0.0]]))

    with pytest.raises(ValueError, match=message):
        integrated_gradients(network, sample)


def test_invalid_baseline_shape_fails_closed() -> None:
    network = linear_network(np.array([[1.0, 1.0]]), np.array([[0.0]]))

    with pytest.raises(ValueError, match="baseline must have shape"):
        integrated_gradients(network, [1.0, 2.0], baseline=[0.0])


@pytest.mark.parametrize("steps", [1, 4097, True, 2.5])
def test_invalid_integration_budget_fails_closed(steps: object) -> None:
    network = linear_network(np.array([[1.0]]), np.array([[0.0]]))
    error = TypeError if isinstance(steps, (bool, float)) else ValueError

    with pytest.raises(error, match="steps"):
        integrated_gradients(network, [1.0], steps=steps)


@pytest.mark.parametrize("target", [-1, 2, True, 0.5])
def test_invalid_target_fails_closed(target: object) -> None:
    network = linear_network(np.eye(2), np.zeros((2, 1)))
    error = TypeError if isinstance(target, (bool, float)) else ValueError

    with pytest.raises(error, match="target_index"):
        integrated_gradients(network, [1.0, 2.0], target_index=target)


def test_hidden_softmax_is_rejected_until_jacobian_is_implemented() -> None:
    network = NeuralNetwork(2, [2], 1, activations=["softmax"], seed=3)

    with pytest.raises(ValueError, match="softmax"):
        integrated_gradients(network, [1.0, 2.0])


def test_non_finite_parameters_fail_closed() -> None:
    network = linear_network(np.array([[1.0, 2.0]]), np.array([[0.0]]))
    network.weights[0][0, 0] = np.inf

    with pytest.raises(ValueError, match="parameters must be finite"):
        integrated_gradients(network, [1.0, 2.0])


def test_non_finite_intermediate_activation_fails_closed() -> None:
    network = NeuralNetwork(
        input_size=1,
        hidden_layers=[1],
        output_size=1,
        activations=["relu"],
        manual_params=[
            (np.array([[1e308]]), np.array([[0.0]])),
            (np.array([[1.0]]), np.array([[0.0]])),
        ],
    )

    with (
        np.errstate(over="ignore"),
        pytest.raises(ValueError, match="non-finite pre-activations"),
    ):
        integrated_gradients(network, [1e308], steps=2)


def test_non_string_activation_fails_closed() -> None:
    network = NeuralNetwork(1, [1], 1, activations=["relu"], seed=3)
    network.activations[0] = None

    with pytest.raises(TypeError, match="activation must be a string"):
        integrated_gradients(network, [1.0])


@pytest.mark.parametrize("tolerance", [-1.0, float("nan"), float("inf")])
def test_invalid_completeness_tolerance_fails_closed(tolerance: float) -> None:
    network = linear_network(np.array([[1.0]]), np.array([[0.0]]))
    report = integrated_gradients(network, [1.0], steps=2)

    with pytest.raises(ValueError, match="finite and non-negative"):
        report.passes_completeness(absolute_tolerance=tolerance)


def test_baseline_stability_reports_sign_and_top_k_agreement() -> None:
    network = linear_network(np.array([[3.0, 2.0, 0.5]]), np.array([[0.0]]))

    report = audit_baseline_stability(
        network,
        [4.0, 3.0, 2.0],
        baselines=[[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 1.0, 0.0]],
        steps=2,
        top_k=2,
    )

    assert report.baseline_count == 3
    assert report.median_attributions == pytest.approx((9.0, 5.0, 1.0))
    assert report.sign_agreement_ratios == pytest.approx((1.0, 1.0, 1.0))
    assert report.min_sign_agreement == pytest.approx(1.0)
    assert report.min_top_k_overlap == pytest.approx(1.0)
    assert report.max_abs_completeness_delta == pytest.approx(0.0)
    assert report.passes_stability()


def test_baseline_stability_detects_rank_and_sign_instability() -> None:
    network = linear_network(np.array([[1.0, 1.0]]), np.array([[0.0]]))

    report = audit_baseline_stability(
        network,
        [2.0, 2.0],
        baselines=[[0.0, 1.0], [1.0, 0.0], [3.0, 0.0]],
        steps=2,
        top_k=1,
    )

    assert report.min_sign_agreement == pytest.approx(2 / 3)
    assert report.min_top_k_overlap == pytest.approx(0.0)
    assert not report.passes_stability(min_sign_agreement=0.8, min_top_k_overlap=0.5)


def test_baseline_set_is_order_independent() -> None:
    network = linear_network(np.array([[2.0, -1.0]]), np.array([[0.0]]))
    baselines = [[0.0, 0.0], [0.5, -0.5], [1.0, -1.0]]

    first = audit_baseline_stability(
        network, [2.0, -2.0], baselines=baselines, steps=2, top_k=1
    ).to_dict()
    second = audit_baseline_stability(
        network,
        [2.0, -2.0],
        baselines=list(reversed(baselines)),
        steps=2,
        top_k=1,
    ).to_dict()

    assert first == second
    assert len(first["stability_id"]) == 64


def test_duplicate_baselines_are_rejected() -> None:
    network = linear_network(np.array([[1.0]]), np.array([[0.0]]))

    with pytest.raises(ValueError, match="distinct"):
        audit_baseline_stability(
            network, [1.0], baselines=[[0.0], [0.0]], steps=2, top_k=1
        )


@pytest.mark.parametrize("baselines", [[], [[0.0]], [[0.0]] * 17])
def test_baseline_count_is_bounded(baselines: list[list[float]]) -> None:
    network = linear_network(np.array([[1.0]]), np.array([[0.0]]))

    with pytest.raises(ValueError, match="between 2 and 16"):
        audit_baseline_stability(network, [1.0], baselines=baselines, steps=2, top_k=1)


def test_baseline_gradient_budget_is_bounded() -> None:
    network = linear_network(np.array([[1.0]]), np.array([[0.0]]))

    with pytest.raises(ValueError, match="path-gradient evaluation budget"):
        audit_baseline_stability(
            network,
            [1.0],
            baselines=[[float(value)] for value in range(16)],
            steps=4096,
            top_k=1,
        )


@pytest.mark.parametrize("top_k", [0, 3, True, 1.5])
def test_invalid_top_k_fails_closed(top_k: object) -> None:
    network = linear_network(np.array([[1.0, 1.0]]), np.array([[0.0]]))
    error = TypeError if isinstance(top_k, (bool, float)) else ValueError

    with pytest.raises(error, match="top_k"):
        audit_baseline_stability(
            network,
            [1.0, 1.0],
            baselines=[[0.0, 0.0], [0.5, 0.5]],
            steps=2,
            top_k=top_k,
        )


def test_incomplete_baseline_explanation_is_rejected() -> None:
    network = NeuralNetwork(
        1,
        [1],
        1,
        activations=["tanh"],
        manual_params=[
            (np.array([[3.0]]), np.array([[0.2]])),
            (np.array([[2.0]]), np.array([[0.0]])),
        ],
    )

    with pytest.raises(ValueError, match="failed the completeness gate"):
        audit_baseline_stability(
            network,
            [2.0],
            baselines=[[-2.0], [0.0]],
            steps=2,
            top_k=1,
            completeness_absolute_tolerance=0.0,
            completeness_relative_tolerance=0.0,
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), True])
def test_invalid_stability_threshold_fails_closed(threshold: object) -> None:
    network = linear_network(np.array([[1.0]]), np.array([[0.0]]))
    report = audit_baseline_stability(
        network, [1.0], baselines=[[0.0], [0.5]], steps=2, top_k=1
    )
    error = TypeError if isinstance(threshold, bool) else ValueError

    with pytest.raises(error, match="min_sign_agreement"):
        report.passes_stability(min_sign_agreement=threshold)
