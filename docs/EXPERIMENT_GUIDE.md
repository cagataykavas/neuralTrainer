# Reproducible experiment guide

The goal is to compare decisions, not produce one lucky training curve.

## 1. Lock the protocol

Use at least five training seeds. Keep a disjoint list of validation seeds for model selection and test seeds for final reporting. Store the command, git commit, resolved JSON configuration, and hardware with every run.

Start with the same scenario distribution and episode budget for each algorithm:

```bash
python parking_rl.py --describe-json > resolved_config.json
python parking_rl.py --mode TRAIN --algorithm PPO --action-mode CONTINUOUS --scenario RANDOM --seed 11 --episodes 3000 --headless
python parking_rl.py --mode TRAIN --algorithm SAC --action-mode CONTINUOUS --scenario RANDOM --seed 11 --episodes 3000 --headless
```

Run seeds `11, 23, 37, 41, 53` before drawing a conclusion. Keep training output directories separate so checkpoints and logs cannot overwrite each other.

## 2. Compare a small, useful matrix

| Experiment | Algorithm | Control | Residual | Question |
|---|---|---|---:|---|
| A | PPO | continuous | no | Plain on-policy baseline |
| B | SAC | continuous | no | Does replay improve sample efficiency? |
| C | PPO | curriculum | no | Does coarse-to-fine control stabilize learning? |
| D | PPO | curriculum | yes | Does the geometric prior improve early success? |

Change one factor at a time. For residual runs, set `USE_RESIDUAL_RL=True` and record `RESIDUAL_SCALE`.

## 3. Report task metrics

For each held-out seed suite, report mean and a confidence interval for:

- parking success rate;
- wall/car collision rate;
- out-of-bounds and timeout rate;
- final center distance and angular error;
- episode length among successful attempts;
- environment steps to a fixed success threshold.

Reward curves are diagnostic. They are not a substitute for those metrics because reward shaping can improve while task success does not.

## 4. Required ablations

1. Remove bearing shaping.
2. Remove straight-motion bonus.
3. Remove idle and waggle penalties.
4. Compare fixed continuous control with the curriculum.
5. Compare direct control with residual control.

If a component does not improve held-out success or sample efficiency, remove it or explain why it remains.

## 5. Acceptance criteria for published results

- Every result points to a commit and resolved config.
- Test seeds were not used for model selection.
- Failures are broken down by collision, boundary, and timeout.
- Curves aggregate multiple seeds and show dispersion.
- The README distinguishes measured results from planned experiments.

