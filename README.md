# Parking RL Lab — Stable

A self-contained reinforcement-learning environment for autonomous parking with **Double DQN, PPO and SAC**, oriented-rectangle collision geometry, configurable parking scenarios, deterministic evaluation, checkpoints and replay visualization.

This branch is the **stable portfolio release**. Its recommended experiments use fixed action spaces so algorithm comparisons are easy to interpret and reproduce. More aggressive curriculum/residual-control experiments live on `parking-rl-experimental`.

## Algorithm suite

| Algorithm | Action modes | Main idea | Best use in this project |
|---|---|---|---|
| **Double DQN + Dueling Network** | `DISCRETE_9`, `DISCRETE_43` | value-based off-policy learning with replay + target network | clean discrete baseline |
| **PPO** | `DISCRETE_9`, `DISCRETE_43`, `CONTINUOUS` | clipped on-policy actor-critic | robust general baseline |
| **SAC** | `CONTINUOUS` | entropy-regularized off-policy actor-critic with twin critics | continuous steering/throttle |

The point is not to claim one algorithm is universally superior. The repository makes the action-space trade-offs explicit so the algorithms can be compared under the same parking physics and reward function.

## Highlights

- 9-command and 43-command discrete motor spaces
- native continuous steering/throttle control
- Double DQN with dueling network, replay buffer, Huber loss and Double-DQN target selection
- PPO categorical and squashed-Gaussian policies
- SAC twin critics and automatic entropy tuning
- separating-axis-theorem collision detection for rotated rectangles
- multi-car environment support
- SIMPLE, WALLS, RANDOM and custom JSON scenarios
- hierarchical approach / align / settle state features
- model checkpoints and CSV experiment logs
- asynchronous replay visualization in a separate process
- deterministic seed control and machine-readable environment contract

## Quick verification

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"

python parking_rl.py --smoke-test
python parking_rl.py --describe-json
pytest
```

## Training

### Double DQN — 9 actions

```bash
python dqn_agent.py --action-mode DISCRETE_9 --episodes 1200 --seed 42
```

### Double DQN — 43 actions

```bash
python dqn_agent.py --action-mode DISCRETE_43 --episodes 1600 --seed 42
```

### PPO — discrete

```bash
python parking_rl.py \
  --mode TRAIN \
  --algorithm PPO \
  --action-mode DISCRETE_43 \
  --scenario RANDOM \
  --episodes 1500 \
  --headless
```

### PPO — continuous

```bash
python parking_rl.py --mode TRAIN --algorithm PPO --action-mode CONTINUOUS --episodes 1500 --headless
```

### SAC — continuous

```bash
python parking_rl.py --mode TRAIN --algorithm SAC --action-mode CONTINUOUS --episodes 1500 --headless
```

## DQN implementation

`dqn_agent.py` deliberately reuses the same `CarParkingEnvMulti` environment and discrete action tables as PPO/SAC instead of creating a second toy environment.

The baseline includes:

- online and target Q networks;
- dueling value / advantage heads;
- replay memory;
- epsilon-greedy exploration;
- Double-DQN target action selection;
- Huber TD loss;
- gradient clipping;
- periodic hard target updates;
- best/latest checkpoints;
- deterministic evaluation mode.

DQN is **not** forced into continuous control. That would make the comparison less meaningful; native continuous experiments belong to PPO/SAC.

## Environment

The observation contains geometric state for the controlled car and target parking lot. With hierarchical state enabled, the base 13-dimensional representation is extended with a three-value one-hot phase indicator:

```text
approach → align → settle
```

Actions update heading and longitudinal speed, after which the environment computes progress, bearing/alignment shaping, time penalties, collision penalties and terminal parking success.

Collision checks are performed using the separating axis theorem on oriented rectangles rather than axis-aligned boxes.

## Scenarios

Built-in scenarios include:

- `SIMPLE` — baseline parking geometry
- `WALLS` — obstacle-aware parking
- `RANDOM` — randomized scenario selection
- `CUSTOM` — JSON-defined cars, parking lots and walls

`examples/u_shape_trap.json` provides a harder U-shaped custom environment.

## Evaluation protocol

A useful comparison is not “which run had the highest reward once.” Use multiple fixed seeds and report:

- parking success rate;
- collision rate;
- timeout rate;
- mean/median episode return;
- final position error;
- final alignment error;
- episode length.

See `docs/EXPERIMENT_GUIDE.md` for the reproducible evaluation protocol.

## Repository map

```text
parking_rl.py                 environment, PPO, SAC, curriculum engine, rendering, CLI
dqn_agent.py                  Double-DQN / dueling discrete baseline
examples/u_shape_trap.json    harder custom parking scenario
tests/test_parking_rl.py      geometry, reset, curriculum and integration tests
docs/ARCHITECTURE.md          state/action/reward architecture
docs/EXPERIMENT_GUIDE.md      multi-seed experiment protocol
```

## Stable vs experimental

### `parking-rl-stable`

Use this branch for portfolio demos and algorithm comparisons. Recommended configurations use fixed action spaces and conservative defaults.

### `parking-rl-experimental`

Contains the same core algorithms plus research-oriented configurations for:

- `9 → 43 → continuous` curriculum learning;
- annealed discrete-to-continuous execution;
- residual RL on top of a geometric controller;
- harder scenario curricula;
- multi-car / ablation experiments.

## Evidence boundary

The code, smoke tests and experiment harness are committed. The README intentionally does **not** invent convergence numbers. Trained-policy performance should be attached to exact configs, seeds and checkpoints after real runs.
