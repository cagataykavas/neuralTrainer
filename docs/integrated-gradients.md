# Integrated Gradients for the NumPy network

`integrated_gradients.py` attributes one selected linear output to one input
sample without PyTorch, TensorFlow or an autograd engine. It differentiates the
existing NumPy network with the chain rule and integrates those gradients along
the straight path from a caller-selected baseline to the input.

```bash
python integrated_gradients.py --steps 128
```

The JSON result includes per-feature attribution, the prediction difference
from the baseline, the sum of attributions and the completeness residual:

```text
completeness_delta = (f(input) - f(baseline)) - sum(attributions)
```

It also includes SHA-256 identities for the model parameters, input, baseline
and complete explanation record. The digests make two experiment artifacts
comparable; they are not signatures and do not prove who produced an artifact.

## Baseline stability audit

`audit_baseline_stability()` evaluates one input against 2–16 distinct,
domain-valid baselines. Before comparing explanations it requires every
Integrated Gradients result to pass the configured completeness gate. It then
reports:

- per-feature median attribution and sign-agreement ratio;
- minimum and mean sign agreement across features;
- minimum and mean pairwise overlap among the top-k absolute attributions;
- the largest absolute completeness residual;
- order-independent SHA-256 identities for the baseline set and full audit.

Near-zero attributions are assigned a separate neutral sign using the explicit
`zero_tolerance`. `passes_stability()` applies caller-owned sign and top-k
thresholds; the defaults are examples, not universal release criteria.

Duplicate baselines are rejected because they would inflate agreement. The
audit also limits the baseline count to 16 and the total path-gradient
evaluations to 65,536, so untrusted configuration cannot create unbounded work.
The completeness tolerances are included in the audit identity.

## Safety and numerical contract

- Exactly one sample is accepted per explanation so feature axes cannot be
  silently mixed across a batch.
- Input, baseline, parameters, gradients and predictions must be finite.
- Inputs are copied on admission, and the model digest is checked again after
  integration; a parameter update during explanation causes rejection instead
  of emitting evidence bound to the wrong model state.
- The target output index is explicit and bounds checked.
- Integration uses a streaming trapezoidal sum, so memory does not scale with
  the number of steps.
- The integration budget is restricted to 2–4096 steps.
- Hidden softmax is rejected because the recovered trainer intentionally uses
  an element-wise approximation instead of the required Jacobian-vector
  product. Producing an explanation through that derivative would be
  misleading.
- `passes_completeness()` provides an explicit absolute/relative numerical
  gate rather than claiming every approximation is exact.
- Stability evidence is emitted only after every baseline explanation passes
  that completeness gate.

## Limitations

Integrated Gradients explains the implemented network along one baseline path;
it does not establish causality, feature independence or fairness. Baseline
choice can materially change the attribution. Sign agreement and top-k overlap
measure sensitivity to the supplied baseline set; they do not prove that the
set is representative or that stable features are causal. ReLU paths may cross
non-differentiable points, while curved sigmoid/tanh/softplus paths require
enough integration steps for the chosen tolerance. This implementation handles
one input at a time and a linear output layer, matching the current trainer.

The next extension should add the full softmax Jacobian-vector product and
support versioned domain policies for baseline selection.
