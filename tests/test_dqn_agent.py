import numpy as np
import torch

from dqn_agent import DQNConfig, DoubleDQNAgent, DuelingQNetwork


def test_dueling_network_shape():
    model = DuelingQNetwork(state_dim=16, action_dim=9, hidden_size=64)
    x = torch.zeros((4, 16), dtype=torch.float32)
    q = model(x)
    assert q.shape == (4, 9)
    assert torch.isfinite(q).all()


def test_double_dqn_update_smoke():
    cfg = DQNConfig(
        action_mode="DISCRETE_9",
        batch_size=8,
        warmup_steps=8,
        replay_size=64,
        hidden_size=64,
        target_update_every=2,
        device="cpu",
    )
    agent = DoubleDQNAgent(state_dim=16, action_dim=9, cfg=cfg)

    rng = np.random.default_rng(42)
    for i in range(16):
        state = rng.normal(size=16).astype(np.float32)
        next_state = rng.normal(size=16).astype(np.float32)
        agent.replay.add(state, i % 9, float(i % 3 - 1), next_state, bool(i % 5 == 0))

    agent.environment_steps = 8
    stats = agent.update()
    assert stats is not None
    assert np.isfinite(stats["loss"])
    assert np.isfinite(stats["mean_abs_td_error"])


def test_epsilon_schedule_is_bounded():
    cfg = DQNConfig(device="cpu", epsilon_decay_steps=100)
    agent = DoubleDQNAgent(state_dim=16, action_dim=9, cfg=cfg)
    assert cfg.epsilon_end <= agent.epsilon() <= cfg.epsilon_start
    agent.environment_steps = 10_000
    assert abs(agent.epsilon() - cfg.epsilon_end) < 1e-9
