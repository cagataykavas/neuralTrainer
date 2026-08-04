.PHONY: describe smoke test train-short

describe:
	python parking_rl.py --describe-json

smoke:
	python parking_rl.py --smoke-test

test:
	pytest

train-short:
	python parking_rl.py --mode TRAIN --algorithm PPO --action-mode CURRICULUM --scenario RANDOM --episodes 25 --headless

