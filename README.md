# Parking RL Lab — Experimental

This branch contains the research-heavy version of the autonomous parking environment: **Double DQN, PPO, SAC, curriculum learning, residual RL, hierarchical state and harder scenario experiments**.

For a cleaner portfolio/demo baseline, use `parking-rl-stable`. This branch is intentionally allowed to be more aggressive and less conservative because its purpose is ablation and research exploration.

## Algorithms

- **Double DQN + Dueling Network** for fixed `DISCRETE_9` and `DISCRETE_43` control
- **PPO** for categorical discrete policies and squashed-Gaussian continuous policies
- **SAC** for native continuous steering/throttle with twin critics and entropy tuning

## Main experimental idea: action-space curriculum

The core curriculum is:

```text
9 discrete actions
        ↓
43 discrete actions
        ↓
annealed 43-action quantization → continuous motor command
        ↓
fully continuous steering/throttle
```

For PPO, the curriculum keeps a stable two-dimensional latent motor policy while the action adapter changes the executed control resolution. This avoids replacing a 9-way policy head with a 43-way head and then a continuous head mid-training.

The default phases are configured through `CURRICULUM_PHASES` in `parking_rl.py`.

## Residual RL

The environment includes a geometric parking controller that can provide a base command. With residual RL enabled, the policy learns a correction:

```text
executed_action = base_controller + residual_scale * policy_action
```

This provides a useful experiment against pure end-to-end RL: instead of learning basic approach geometry from zero, the neural policy can focus on correcting the hand-designed controller.

Enable it in `CONFIG`:

```python
"USE_RESIDUAL_RL": True,
"RESIDUAL_SCALE": 0.35,
```

## Hierarchical state

The state can include an explicit three-stage phase indicator:

```text
approach → align → settle
```

This is intentionally simple and inspectable. It lets the policy condition on the qualitative parking phase without introducing a second learned manager that would make debugging and attribution harder.

## Training examples

### Curriculum PPO

```bash
python parking_rl.py \
  --mode TRAIN \
  --algorithm PPO \
  --action-mode CURRICULUM \
  --scenario RANDOM \
  --episodes 3000 \
  --headless
```

### Continuous SAC

```bash
python parking_rl.py \
  --mode TRAIN \
  --algorithm SAC \
  --action-mode CONTINUOUS \
  --scenario WALLS \
  --episodes 3000 \
  --headless
```

### Double DQN baseline

```bash
python dqn_agent.py --action-mode DISCRETE_43 --episodes 1600 --seed 42
```

DQN remains fixed-discrete by design. The curriculum experiment is mainly intended for PPO/SAC-style motor policies rather than forcing value-based DQN into an artificial continuous approximation.

## Experimental matrix

Useful ablations include:

| Experiment | Variable |
|---|---|
| discrete resolution | 9 vs 43 commands |
| action representation | fixed discrete vs continuous |
| curriculum | fixed mode vs `9 → 43 → continuous` |
| residual control | pure RL vs controller + residual |
| state hierarchy | 13 geometric values vs + phase one-hot |
| scenario difficulty | SIMPLE vs WALLS vs RANDOM vs CUSTOM |
| policy family | Double DQN vs PPO vs SAC |
| number of cars | single-car vs multi-car |

## Environment details

- oriented-rectangle car, lot and wall geometry;
- separating-axis-theorem collision checks;
- configurable reward shaping for progress, bearing, smooth control, collisions and parking success;
- custom JSON layouts;
- reproducible seeds;
- checkpoint boundaries at curriculum transitions;
- CSV experiment logs;
- asynchronous replay rendering in a separate process;
- manual keyboard PLAY mode for sanity-checking dynamics.

## Hard scenario

`examples/u_shape_trap.json` provides a U-shaped parking scenario designed to expose weaknesses in greedy/direct-to-goal behavior. It is useful for comparing pure RL, residual control and curriculum policies.

## Repository map

```text
parking_rl.py                 environment, PPO, SAC, curriculum, residual RL, rendering
dqn_agent.py                  Double-DQN / dueling discrete baseline
examples/u_shape_trap.json    harder custom scenario
tests/test_parking_rl.py      geometry, curriculum, reproducibility and integration tests
docs/ARCHITECTURE.md          state/action/reward architecture
docs/EXPERIMENT_GUIDE.md      multi-seed evaluation protocol
```

## Stable vs experimental

- `parking-rl-stable`: recruiter/demo-facing fixed-action baselines, DQN/PPO/SAC, conservative defaults.
- `parking-rl-experimental`: curriculum, residual control, hierarchy and harder ablations.

## Evidence boundary

This branch contains the runnable research machinery, not fabricated performance claims. Any success-rate or reward table should come from real multi-seed training runs with the exact config/checkpoint recorded.
