"""Double-DQN baseline for the Parking RL Lab.

This module intentionally lives beside ``parking_rl.py`` instead of modifying the
large original experiment file. It reuses the same environment, reward function,
action tables, seed handling and checkpoint directory.

DQN is defined only for fixed discrete action spaces:
- DISCRETE_9
- DISCRETE_43

PPO and SAC remain available through ``parking_rl.py``. This gives the project
three complementary algorithm families without pretending that value-based DQN
is appropriate for native continuous control.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from parking_rl import (
    CONFIG,
    CarParkingEnvMulti,
    DISCRETE_9_TABLE,
    DISCRETE_43_TABLE,
    ensure_output_dirs,
    set_seed,
)


@dataclass
class DQNConfig:
    action_mode: str = "DISCRETE_9"
    episodes: int = 1200
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 256
    replay_size: int = 100_000
    warmup_steps: int = 1_000
    target_update_every: int = 1_000
    train_every: int = 1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 40_000
    hidden_size: int = 256
    max_grad_norm: float = 1.0
    checkpoint_every: int = 100
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def normalized_action_mode(self) -> str:
        mode = self.action_mode.strip().upper()
        aliases = {"9": "DISCRETE_9", "43": "DISCRETE_43", "DISCRETE9": "DISCRETE_9", "DISCRETE43": "DISCRETE_43"}
        mode = aliases.get(mode, mode)
        if mode not in {"DISCRETE_9", "DISCRETE_43"}:
            raise ValueError("DQN supports only DISCRETE_9 and DISCRETE_43.")
        return mode


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.data: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=int(capacity))

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.data.append(
            (
                np.asarray(state, dtype=np.float32).copy(),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32).copy(),
                float(done),
            )
        )

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.data, int(batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.stack(states), dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.long, device=device),
            torch.as_tensor(rewards, dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device),
            torch.as_tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self) -> int:
        return len(self.data)


class DuelingQNetwork(nn.Module):
    """Dueling MLP used by the Double-DQN agent."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class DoubleDQNAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: DQNConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.action_dim = int(action_dim)
        self.online = DuelingQNetwork(state_dim, action_dim, cfg.hidden_size).to(self.device)
        self.target = DuelingQNetwork(state_dim, action_dim, cfg.hidden_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = optim.AdamW(self.online.parameters(), lr=cfg.learning_rate)
        self.replay = ReplayBuffer(cfg.replay_size)
        self.environment_steps = 0
        self.gradient_steps = 0

    def epsilon(self) -> float:
        fraction = min(1.0, self.environment_steps / max(1, self.cfg.epsilon_decay_steps))
        return self.cfg.epsilon_start + fraction * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    @torch.no_grad()
    def act(self, states: Sequence[np.ndarray], deterministic: bool = False) -> List[int]:
        eps = 0.0 if deterministic else self.epsilon()
        tensor = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
        q_values = self.online(tensor)
        greedy = torch.argmax(q_values, dim=-1).cpu().numpy().astype(np.int64)
        actions: List[int] = []
        for greedy_action in greedy:
            if not deterministic and random.random() < eps:
                actions.append(random.randrange(self.action_dim))
            else:
                actions.append(int(greedy_action))
        return actions

    def observe(
        self,
        states: Sequence[np.ndarray],
        actions: Sequence[int],
        rewards: Sequence[float],
        next_states: Sequence[np.ndarray],
        dones: Sequence[bool],
    ) -> None:
        for item in zip(states, actions, rewards, next_states, dones):
            self.replay.add(*item)
        self.environment_steps += 1

    def update(self) -> Optional[Dict[str, float]]:
        if len(self.replay) < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return None
        if self.environment_steps % max(1, self.cfg.train_every) != 0:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.cfg.batch_size, self.device)
        q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: action selection from online network, value from target network.
            next_actions = torch.argmax(self.online(next_states), dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + self.cfg.gamma * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        self.gradient_steps += 1

        if self.gradient_steps % max(1, self.cfg.target_update_every) == 0:
            self.target.load_state_dict(self.online.state_dict())

        with torch.no_grad():
            td_abs = torch.mean(torch.abs(target - q)).item()
        return {
            "loss": float(loss.detach().cpu()),
            "mean_abs_td_error": float(td_abs),
            "grad_norm": float(grad_norm),
            "epsilon": float(self.epsilon()),
        }

    def save(self, path: Path, episode: int, best_reward: float) -> None:
        torch.save(
            {
                "algorithm": "DOUBLE_DQN",
                "episode": int(episode),
                "best_reward": float(best_reward),
                "environment_steps": int(self.environment_steps),
                "gradient_steps": int(self.gradient_steps),
                "config": self.cfg.__dict__,
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: Path, load_optimizer: bool = True) -> Dict[str, object]:
        checkpoint = torch.load(path, map_location=self.device)
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["online"]))
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.environment_steps = int(checkpoint.get("environment_steps", 0))
        self.gradient_steps = int(checkpoint.get("gradient_steps", 0))
        return checkpoint


def action_table(mode: str) -> np.ndarray:
    return DISCRETE_9_TABLE if mode == "DISCRETE_9" else DISCRETE_43_TABLE


def train(cfg: DQNConfig) -> Path:
    mode = cfg.normalized_action_mode()
    set_seed(cfg.seed)
    CONFIG["SEED"] = cfg.seed
    CONFIG["ACTION_MODE"] = mode
    CONFIG["USE_RESIDUAL_RL"] = False

    env = CarParkingEnvMulti(seed=cfg.seed)
    table = action_table(mode)
    agent = DoubleDQNAgent(env.state_dim, len(table), cfg)
    dirs = ensure_output_dirs()
    checkpoints = dirs["checkpoints"]
    best_path = checkpoints / f"dqn_{mode.lower()}_best.pth"
    latest_path = checkpoints / f"dqn_{mode.lower()}_latest.pth"

    best_reward = -math.inf
    recent_rewards: Deque[float] = deque(maxlen=50)

    for episode in range(1, cfg.episodes + 1):
        states = env.reset()
        total_reward = 0.0
        episode_steps = 0
        success_count = 0
        latest_stats: Optional[Dict[str, float]] = None

        while True:
            action_indices = agent.act(states)
            env_actions = [table[index].copy() for index in action_indices]
            next_states, rewards, dones = env.step(env_actions)
            agent.observe(states, action_indices, rewards, next_states, dones)
            latest_stats = agent.update() or latest_stats

            total_reward += float(sum(rewards))
            episode_steps += 1
            states = next_states

            if all(dones):
                success_count = sum(
                    1
                    for car in env.cars
                    if env.done_flags.get(car.uid, False)
                    and any(
                        lot.uid == car.target_id and abs(car.x - lot.x) < lot.l and abs(car.y - lot.y) < lot.w
                        for lot in env.lots
                    )
                )
                break

        recent_rewards.append(total_reward)
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(best_path, episode, best_reward)

        if episode % cfg.checkpoint_every == 0 or episode == cfg.episodes:
            agent.save(latest_path, episode, best_reward)

        if episode == 1 or episode % 10 == 0:
            mean_reward = float(np.mean(recent_rewards)) if recent_rewards else total_reward
            stat_text = "" if latest_stats is None else f" loss={latest_stats['loss']:.4f}"
            print(
                f"[DQN] ep={episode:5d} reward={total_reward:10.2f} mean50={mean_reward:10.2f} "
                f"steps={episode_steps:4d} eps={agent.epsilon():.3f} success={success_count}{stat_text}"
            )

    return latest_path


def evaluate(cfg: DQNConfig, checkpoint: Path, episodes: int = 20) -> Dict[str, float]:
    mode = cfg.normalized_action_mode()
    set_seed(cfg.seed)
    CONFIG["SEED"] = cfg.seed
    CONFIG["ACTION_MODE"] = mode
    CONFIG["USE_RESIDUAL_RL"] = False

    env = CarParkingEnvMulti(seed=cfg.seed)
    table = action_table(mode)
    agent = DoubleDQNAgent(env.state_dim, len(table), cfg)
    agent.load(checkpoint, load_optimizer=False)

    rewards: List[float] = []
    lengths: List[int] = []
    for _ in range(int(episodes)):
        states = env.reset()
        total = 0.0
        steps = 0
        while True:
            indices = agent.act(states, deterministic=True)
            states, step_rewards, dones = env.step([table[index].copy() for index in indices])
            total += float(sum(step_rewards))
            steps += 1
            if all(dones):
                break
        rewards.append(total)
        lengths.append(steps)

    result = {
        "episodes": float(episodes),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
    }
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Double-DQN baseline for Parking RL Lab")
    parser.add_argument("--action-mode", default="DISCRETE_9", choices=["DISCRETE_9", "DISCRETE_43"])
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eval-episodes", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DQNConfig(action_mode=args.action_mode, episodes=args.episodes, seed=args.seed)
    if args.checkpoint:
        evaluate(cfg, args.checkpoint, args.eval_episodes)
    else:
        path = train(cfg)
        print(f"Saved latest checkpoint to {path}")


if __name__ == "__main__":
    main()
