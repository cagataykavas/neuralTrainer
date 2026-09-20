# Numerical gradient audit

`gradient_check.py` compares explicit backpropagation against central finite
differences before training logic is trusted. It works with the repository's
`NeuralNetwork` interface and other models exposing weights, biases, forward,
backward, and loss functions.

```python
report = check_gradients(
    nn,
    values,
    targets,
    GradientCheckPolicy(epsilon=1e-5, relative_tolerance=1e-4),
)
assert report.passed, report.to_dict()
```

Every checked weight and bias receives an analytic gradient, numerical gradient,
absolute error, relative error, and pass/fail decision. Large models can use a
seeded parameter sample; the report still exposes checked and total counts.
Parameters are restored with `finally`, including when perturbed loss evaluation
fails. Shape mismatches and non-finite inputs, gradients, or losses fail closed.

## Interpretation and limits

Finite differences are a correctness audit, not an explanation of model
behavior. Run them on small deterministic batches, away from non-differentiable
activation boundaries such as ReLU at zero. Epsilon and tolerances depend on
floating-point scale. Passing sampled parameters does not prove every gradient is
correct; use `max_parameters=None` for complete small-network checks. The
softmax derivative caveat documented by the project remains intentionally
visible and should be tested with a proper Jacobian-vector product before that
activation is treated as production-ready.
