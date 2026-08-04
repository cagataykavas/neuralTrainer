"""
parking_rl.py
=============

One self-contained parking-RL project. The shortest verification path is:

    python parking_rl.py --smoke-test

Everything can be controlled from CONFIG or the command line.

Included
--------
- Modes: TRAIN, TRANSFER, INFERENCE, PLAY.
- Algorithms: PPO and SAC. "SOC" is accepted as an alias for SAC because the
  original project/screenshots used SAC and the name was later typed as SOC.
- Control modes:
    * DISCRETE_9
    * DISCRETE_43 (7 steering levels x 6 throttle levels + neutral)
    * CONTINUOUS
    * CURRICULUM: DISCRETE_9 -> DISCRETE_43 -> gradual 43-to-continuous blend
      -> CONTINUOUS.
- Multi-car environment, parking lots, walls, collision checks, custom JSON
  scenarios, shaped rewards, model saving/loading, CSV work logs.
- Optional hierarchical state, optional residual RL over a simple controller.
- Non-blocking episode visualization: selected episodes are recorded and sent
  to a separate renderer process. Training never waits for plotting.
- Manual PLAY mode with matplotlib keyboard controls.

Dependencies
------------
- numpy
- torch
- matplotlib

The code intentionally avoids gym/gymnasium, pygame, keyboard, OpenCV, and
project-local imports. It is a single file and requires no edits outside CONFIG.
"""

from __future__ import annotations

import csv
import argparse
import json
import math
import os
import queue
import random
import signal
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =============================================================================
# CONFIG — change values here and run: python parking_rl_final.py
# =============================================================================
CONFIG: Dict[str, Any] = {
    # -------------------------------------------------------------------------
    # Main switchboard
    # -------------------------------------------------------------------------
    "MODE": "TRAIN",                    # TRAIN | TRANSFER | INFERENCE | PLAY
    "ALGORITHM": "PPO",                # PPO | SAC | SOC (SOC aliases SAC)
    "ACTION_MODE": "CURRICULUM",       # DISCRETE_9 | DISCRETE_43 | CONTINUOUS | CURRICULUM
    "ACTION_DIMENSION": "AUTO",          # derived: 9, 43, or 2; kept visible for original-style configs
    "SCENARIO": "SIMPLE",              # SIMPLE | WALLS | CUSTOM | RANDOM
    "CUSTOM_LEVEL_FILE": "examples/u_shape_trap.json",

    # -------------------------------------------------------------------------
    # Curriculum: actual discrete motor commands -> finer discrete -> continuous
    # -------------------------------------------------------------------------
    # For PPO, CURRICULUM deliberately uses a two-dimensional latent motor policy
    # throughout. The environment first quantizes that policy to 9 actual actions,
    # then 43 actual actions, then gradually removes quantization. This keeps the
    # policy head shape constant and avoids 9->43->continuous checkpoint mismatch.
    "CURRICULUM_PHASES": [
        {"name": "coarse_discrete_9", "kind": "DISCRETE_9", "episodes": 400},
        {"name": "fine_discrete_43", "kind": "DISCRETE_43", "episodes": 600},
        {"name": "anneal_43_to_continuous", "kind": "ANNEAL_43_TO_CONTINUOUS", "episodes": 600},
        {"name": "continuous", "kind": "CONTINUOUS", "episodes": 10_000_000},
    ],

    # -------------------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------------------
    "SEED": 42,
    "NUM_CARS": 1,
    "AREA_SIZE": 100.0,
    "MAX_STEPS_PER_EP": 300,
    "DEFAULT_CAR_LENGTH": 8.0,
    "DEFAULT_CAR_WIDTH": 4.0,
    "DEFAULT_LOT_LENGTH": 12.0,
    "DEFAULT_LOT_WIDTH": 6.0,
    "TURN_DEG_PER_STEP": 5.0,
    "SPEED_OFFSET": 0.6,
    "SPEED_SCALE": 3.0,
    "SUCCESS_ALIGNMENT_DEG": 30.0,
    "SUCCESS_MAX_SPEED": 0.75,

    # Hierarchical phase information is appended to the state:
    # approach / align / settle. This is a practical, robust hierarchy rather
    # than a fragile learned manager.
    "USE_HIERARCHICAL_STATE": True,

    # Residual RL: final command = base_controller + scale * policy_command.
    # It can be used with discrete or continuous execution.
    "USE_RESIDUAL_RL": False,
    "RESIDUAL_SCALE": 0.35,

    # -------------------------------------------------------------------------
    # Reward terms (close to the original screenshot logic, but configurable)
    # -------------------------------------------------------------------------
    "REWARD_PROGRESS_SCALE": 100.0,
    "REWARD_PROGRESS_CLIP": 2.0,
    "REWARD_BEARING_SCALE": 10.0,
    "REWARD_STRAIGHT_BONUS": 10.0,
    "REWARD_TIME_PENALTY": 0.05,
    "REWARD_DIRECTION_CHANGE_PENALTY": 2.0,
    "REWARD_IDLE_PENALTY": 100.0,
    "REWARD_STEER_WAGGLE_PENALTY": 20.0,
    "REWARD_SPEED_DIVISOR": 10.0,
    "REWARD_OUT_OF_BOUNDS_PENALTY": 5000.0,
    "REWARD_WALL_COLLISION_PENALTY": 2000.0,
    "REWARD_CAR_COLLISION_PENALTY": 1000.0,
    "REWARD_SUCCESS": 2500.0,

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "EPISODES": 3000,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "LEARNING_RATE": 3e-4,
    "BATCH_SIZE": 256,
    "PPO_EPOCHS": 8,
    "PPO_CLIP_EPS": 0.2,
    "PPO_ENTROPY_COEF": 0.01,
    "PPO_VALUE_COEF": 0.5,
    "MAX_GRAD_NORM": 0.8,

    # SAC / SOC
    "SAC_REPLAY_SIZE": 200_000,
    "SAC_WARMUP_STEPS": 2_000,
    "SAC_UPDATES_PER_STEP": 1,
    "SAC_TAU": 0.005,
    "SAC_TARGET_ENTROPY": -2.0,
    "SAC_ALPHA_LR": 3e-4,

    # -------------------------------------------------------------------------
    # Files/checkpoints
    # -------------------------------------------------------------------------
    "OUTPUT_DIR": "parking_rl_output",
    "CHECKPOINT_EVERY": 100,
    "TRANSFER_FROM": "AUTO",            # AUTO or explicit .pth path
    "INFERENCE_FROM": "AUTO",           # AUTO or explicit .pth path
    "INFERENCE_EPISODES": 10,
    "LOG_EVERY": 10,

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------
    "ASYNC_RENDER": True,
    "RENDER_INTERVAL": 100,              # episode 100, 200, 300, ...
    "RENDER_EPISODES": [],               # explicit additions, e.g. [1, 25, 75]
    "RENDER_QUEUE_SIZE": 2,
    "RENDER_FPS": 30,
    "RENDER_TRAIL": True,
    "RENDER_WINDOW_SIZE": (9, 7),
    "INFERENCE_RENDER": True,
    "PLAY_FPS": 30,

    # PLAY controls: W/S throttle, A/D steering, R reset, Q/Escape quit.
    # Only the first car is human-controlled; other cars use the base controller.
}


# =============================================================================
# Generic utilities
# =============================================================================
Number = Union[int, float, np.number]


def canonical_algorithm(name: str) -> str:
    value = str(name).strip().upper()
    if value == "SOC":
        return "SAC"
    if value not in {"PPO", "SAC"}:
        raise ValueError("ALGORITHM must be PPO, SAC, or SOC (SOC aliases SAC).")
    return value


def canonical_action_mode(name: str) -> str:
    value = str(name).strip().upper()
    aliases = {
        "9": "DISCRETE_9",
        "43": "DISCRETE_43",
        "DISCRETE9": "DISCRETE_9",
        "DISCRETE43": "DISCRETE_43",
        "DISCRETE": "DISCRETE_9",
        "CONT": "CONTINUOUS",
    }
    value = aliases.get(value, value)
    allowed = {"DISCRETE_9", "DISCRETE_43", "CONTINUOUS", "CURRICULUM"}
    if value not in allowed:
        raise ValueError(f"ACTION_MODE must be one of {sorted(allowed)}")
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def angle_diff_deg(a: float, b: float) -> float:
    return normalize_angle_deg(a - b)


def bearing_reward(relative_bearing_deg: float) -> float:
    diff = normalize_angle_deg(relative_bearing_deg)
    return 2.0 * math.exp(-abs(diff) / 10.0) - 1.0


def ensure_output_dirs() -> Dict[str, Path]:
    root = Path(str(CONFIG["OUTPUT_DIR"]))
    checkpoints = root / "checkpoints"
    replays = root / "replays"
    root.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    replays.mkdir(parents=True, exist_ok=True)
    return {"root": root, "checkpoints": checkpoints, "replays": replays}


# =============================================================================
# Geometry and entities
# =============================================================================
@dataclass
class Entity:
    uid: int
    team: int
    x: float
    y: float
    angle: float
    l: float
    w: float
    ent_type: str
    target_id: Optional[int] = None
    speed: float = 0.0

    def copy(self) -> "Entity":
        return Entity(**self.__dict__)


class PhysicsUtils:
    @staticmethod
    def get_corners(entity: Entity) -> np.ndarray:
        angle = math.radians(entity.angle)
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
        local = np.array(
            [
                [entity.l / 2.0, entity.w / 2.0],
                [-entity.l / 2.0, entity.w / 2.0],
                [-entity.l / 2.0, -entity.w / 2.0],
                [entity.l / 2.0, -entity.w / 2.0],
            ],
            dtype=np.float32,
        )
        return local @ rotation.T + np.array([entity.x, entity.y], dtype=np.float32)

    @staticmethod
    def _axes(corners: np.ndarray) -> List[np.ndarray]:
        axes: List[np.ndarray] = []
        for index in range(4):
            edge = corners[(index + 1) % 4] - corners[index]
            axis = np.array([-edge[1], edge[0]], dtype=np.float32)
            norm = float(np.linalg.norm(axis))
            if norm > 1e-8:
                axes.append(axis / norm)
        return axes

    @staticmethod
    def _projection(corners: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
        values = corners @ axis
        return float(values.min()), float(values.max())

    @staticmethod
    def check_collision(first: Entity, second: Entity) -> bool:
        first_corners = PhysicsUtils.get_corners(first)
        second_corners = PhysicsUtils.get_corners(second)
        for axis in PhysicsUtils._axes(first_corners) + PhysicsUtils._axes(second_corners):
            first_min, first_max = PhysicsUtils._projection(first_corners, axis)
            second_min, second_max = PhysicsUtils._projection(second_corners, axis)
            if first_max < second_min or second_max < first_min:
                return False
        return True

    @staticmethod
    def is_car_inside_lot(car: Entity, lot: Entity) -> bool:
        corners = PhysicsUtils.get_corners(car)
        relative = corners - np.array([lot.x, lot.y], dtype=np.float32)
        angle = math.radians(-lot.angle)
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
        local = relative @ rotation.T
        inside_x = np.abs(local[:, 0]) <= lot.l / 2.0
        inside_y = np.abs(local[:, 1]) <= lot.w / 2.0
        return bool(np.all(inside_x & inside_y))


# =============================================================================
# Scenarios
# =============================================================================
def _boundary_walls(area_size: float) -> List[Entity]:
    half = area_size / 2.0
    thickness = 2.0
    return [
        Entity(900, 0, 0.0, half + thickness / 2.0, 0.0, area_size + 4.0, thickness, "WALL"),
        Entity(901, 0, 0.0, -half - thickness / 2.0, 0.0, area_size + 4.0, thickness, "WALL"),
        Entity(902, 0, half + thickness / 2.0, 0.0, 90.0, area_size + 4.0, thickness, "WALL"),
        Entity(903, 0, -half - thickness / 2.0, 0.0, 90.0, area_size + 4.0, thickness, "WALL"),
    ]


def setup_simple_scenario(rng: random.Random, num_cars: int, area_size: float) -> Tuple[List[Entity], List[Entity], List[Entity]]:
    cars: List[Entity] = []
    lots: List[Entity] = []
    walls = _boundary_walls(area_size)
    margin = 14.0

    for index in range(num_cars):
        lot = Entity(
            uid=100 + index,
            team=0,
            x=rng.uniform(-area_size / 2.0 + margin, area_size / 2.0 - margin),
            y=rng.uniform(-area_size / 2.0 + margin, area_size / 2.0 - margin),
            angle=rng.uniform(0.0, 360.0),
            l=float(CONFIG["DEFAULT_LOT_LENGTH"]),
            w=float(CONFIG["DEFAULT_LOT_WIDTH"]),
            ent_type="PARKING",
        )
        lots.append(lot)

    for index, lot in enumerate(lots):
        for _ in range(200):
            x = rng.uniform(-area_size / 2.0 + margin, area_size / 2.0 - margin)
            y = rng.uniform(-area_size / 2.0 + margin, area_size / 2.0 - margin)
            if math.dist((x, y), (lot.x, lot.y)) >= 25.0:
                break
        car = Entity(
            uid=index,
            team=1,
            x=x,
            y=y,
            angle=rng.uniform(0.0, 360.0),
            l=float(CONFIG["DEFAULT_CAR_LENGTH"]),
            w=float(CONFIG["DEFAULT_CAR_WIDTH"]),
            ent_type="CAR",
            target_id=lot.uid,
        )
        cars.append(car)

    return cars, lots, walls


def setup_walls_scenario(rng: random.Random, num_cars: int, area_size: float) -> Tuple[List[Entity], List[Entity], List[Entity]]:
    cars, lots, walls = setup_simple_scenario(rng, num_cars, area_size)
    # A simple U-like obstacle layout, leaving enough room to maneuver.
    walls.extend(
        [
            Entity(300, 0, 0.0, 12.0, 0.0, 28.0, 2.0, "WALL"),
            Entity(301, 0, -14.0, 0.0, 90.0, 24.0, 2.0, "WALL"),
            Entity(302, 0, 14.0, 0.0, 90.0, 24.0, 2.0, "WALL"),
        ]
    )
    return cars, lots, walls


def load_custom_scenario(path: str) -> Tuple[List[Entity], List[Entity], List[Entity]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    def make_entity(raw: Dict[str, Any], ent_type: str, default_uid: int) -> Entity:
        return Entity(
            uid=int(raw.get("uid", default_uid)),
            team=int(raw.get("team", 1 if ent_type == "CAR" else 0)),
            x=float(raw["x"]),
            y=float(raw["y"]),
            angle=float(raw.get("angle", 0.0)),
            l=float(raw.get("l", raw.get("length", 8.0 if ent_type == "CAR" else 12.0))),
            w=float(raw.get("w", raw.get("width", 4.0 if ent_type == "CAR" else 6.0))),
            ent_type=ent_type,
            target_id=raw.get("target_id"),
            speed=float(raw.get("speed", 0.0)),
        )

    cars = [make_entity(item, "CAR", index) for index, item in enumerate(data.get("cars", []))]
    lots = [make_entity(item, "PARKING", 100 + index) for index, item in enumerate(data.get("lots", []))]
    walls = [make_entity(item, "WALL", 300 + index) for index, item in enumerate(data.get("walls", []))]

    if not cars or not lots:
        raise ValueError("Custom scenario JSON must contain at least one car and one lot.")

    for index, car in enumerate(cars):
        if car.target_id is None:
            car.target_id = lots[index % len(lots)].uid

    return cars, lots, walls


def build_scenario(rng: random.Random, scenario: str, num_cars: int, area_size: float) -> Tuple[List[Entity], List[Entity], List[Entity]]:
    selected = str(scenario).upper()
    custom_path = str(CONFIG.get("CUSTOM_LEVEL_FILE", ""))

    if selected == "RANDOM":
        candidates = ["SIMPLE", "WALLS"]
        if custom_path and os.path.exists(custom_path):
            candidates.append("CUSTOM")
        selected = rng.choice(candidates)

    if selected == "CUSTOM":
        if not custom_path or not os.path.exists(custom_path):
            print(f"[SCENARIO] Custom file not found: {custom_path}. Falling back to SIMPLE.")
            return setup_simple_scenario(rng, num_cars, area_size)
        return load_custom_scenario(custom_path)

    if selected == "WALLS":
        return setup_walls_scenario(rng, num_cars, area_size)

    return setup_simple_scenario(rng, num_cars, area_size)


# =============================================================================
# Control stages, quantization, curriculum, and residual decoding
# =============================================================================
DISCRETE_9_TABLE = np.array(
    [
        [-1.0, 1.0],
        [-0.5, 1.0],
        [0.0, 1.0],
        [0.5, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, -1.0],
        [-1.0, -1.0],
        [1.0, -1.0],
    ],
    dtype=np.float32,
)


def build_discrete_43_table() -> np.ndarray:
    steer_values = np.array([-1.0, -0.75, -0.5, 0.0, 0.5, 0.75, 1.0], dtype=np.float32)
    throttle_values = np.array([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0], dtype=np.float32)
    table = [[float(steer), float(throttle)] for steer in steer_values for throttle in throttle_values]
    table.append([0.0, 0.0])
    return np.asarray(table, dtype=np.float32)


DISCRETE_43_TABLE = build_discrete_43_table()


@dataclass(frozen=True)
class ControlStage:
    name: str
    kind: str
    continuous_mix: float
    action_count: Optional[int]
    phase_index: int


class Curriculum:
    def __init__(self, action_mode: str, phases: Sequence[Dict[str, Any]]) -> None:
        self.action_mode = canonical_action_mode(action_mode)
        self.phases = list(phases)

    def stage_for_episode(self, episode: int) -> ControlStage:
        if self.action_mode == "DISCRETE_9":
            return ControlStage("fixed_discrete_9", "DISCRETE_9", 0.0, 9, 0)
        if self.action_mode == "DISCRETE_43":
            return ControlStage("fixed_discrete_43", "DISCRETE_43", 0.0, 43, 0)
        if self.action_mode == "CONTINUOUS":
            return ControlStage("fixed_continuous", "CONTINUOUS", 1.0, None, 0)

        remaining = max(1, int(episode))
        for index, phase in enumerate(self.phases):
            count = int(phase["episodes"])
            if remaining <= count:
                kind = str(phase["kind"]).upper()
                if kind == "ANNEAL_43_TO_CONTINUOUS":
                    progress = (remaining - 1) / max(1, count - 1)
                    return ControlStage(str(phase["name"]), kind, float(progress), 43, index)
                if kind == "DISCRETE_9":
                    return ControlStage(str(phase["name"]), kind, 0.0, 9, index)
                if kind == "DISCRETE_43":
                    return ControlStage(str(phase["name"]), kind, 0.0, 43, index)
                return ControlStage(str(phase["name"]), "CONTINUOUS", 1.0, None, index)
            remaining -= count

        return ControlStage("continuous", "CONTINUOUS", 1.0, None, len(self.phases) - 1)


class ActionAdapter:
    def __init__(self, action_mode: str, algorithm: str) -> None:
        self.action_mode = canonical_action_mode(action_mode)
        self.algorithm = canonical_algorithm(algorithm)
        self.curriculum = Curriculum(self.action_mode, CONFIG["CURRICULUM_PHASES"])

    def ppo_policy_kind(self) -> str:
        # Fixed PPO discrete modes retain a true categorical 9/43-action head.
        # Curriculum uses a stable two-dimensional latent motor policy so its
        # checkpoint transfers cleanly through all phases.
        if self.algorithm == "PPO" and self.action_mode in {"DISCRETE_9", "DISCRETE_43"}:
            return "CATEGORICAL"
        return "GAUSSIAN"

    def categorical_size(self) -> int:
        if self.action_mode == "DISCRETE_9":
            return 9
        if self.action_mode == "DISCRETE_43":
            return 43
        raise RuntimeError("Categorical size is only defined for fixed discrete modes.")

    @staticmethod
    def nearest_table_action(command: np.ndarray, table: np.ndarray) -> np.ndarray:
        command = np.asarray(command, dtype=np.float32).reshape(2)
        distances = np.sum((table - command[None, :]) ** 2, axis=1)
        return table[int(np.argmin(distances))].copy()

    def quantize_numpy(self, command: np.ndarray, stage: ControlStage) -> np.ndarray:
        command = np.clip(np.asarray(command, dtype=np.float32).reshape(2), -1.0, 1.0)
        if stage.kind == "DISCRETE_9":
            return self.nearest_table_action(command, DISCRETE_9_TABLE)
        if stage.kind in {"DISCRETE_43", "ANNEAL_43_TO_CONTINUOUS"}:
            discrete = self.nearest_table_action(command, DISCRETE_43_TABLE)
            if stage.kind == "ANNEAL_43_TO_CONTINUOUS":
                return np.clip((1.0 - stage.continuous_mix) * discrete + stage.continuous_mix * command, -1.0, 1.0)
            return discrete
        return command

    def decode_numpy(
        self,
        policy_action: Union[int, Sequence[float], np.ndarray],
        stage: ControlStage,
        base_action: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if np.isscalar(policy_action):
            index = int(policy_action)
            if stage.kind == "DISCRETE_9":
                policy_command = DISCRETE_9_TABLE[index].copy()
            else:
                policy_command = DISCRETE_43_TABLE[index].copy()
        else:
            policy_command = np.clip(np.asarray(policy_action, dtype=np.float32).reshape(2), -1.0, 1.0)

        if bool(CONFIG["USE_RESIDUAL_RL"]) and base_action is not None:
            candidate = np.clip(
                np.asarray(base_action, dtype=np.float32).reshape(2)
                + float(CONFIG["RESIDUAL_SCALE"]) * policy_command,
                -1.0,
                1.0,
            )
        else:
            candidate = policy_command

        return self.quantize_numpy(candidate, stage)

    @staticmethod
    def _nearest_torch(command: torch.Tensor, table: np.ndarray) -> torch.Tensor:
        table_tensor = torch.as_tensor(table, dtype=command.dtype, device=command.device)
        distances = ((command.unsqueeze(1) - table_tensor.unsqueeze(0)) ** 2).sum(dim=-1)
        nearest = table_tensor[torch.argmin(distances, dim=1)]
        # Straight-through estimator: forward uses discrete command, backward uses identity.
        return command + (nearest - command).detach()

    def decode_torch(
        self,
        policy_action: torch.Tensor,
        stage: ControlStage,
        base_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        command = torch.clamp(policy_action, -1.0, 1.0)
        if bool(CONFIG["USE_RESIDUAL_RL"]) and base_action is not None:
            command = torch.clamp(base_action + float(CONFIG["RESIDUAL_SCALE"]) * command, -1.0, 1.0)

        if stage.kind == "DISCRETE_9":
            return self._nearest_torch(command, DISCRETE_9_TABLE)
        if stage.kind in {"DISCRETE_43", "ANNEAL_43_TO_CONTINUOUS"}:
            discrete = self._nearest_torch(command, DISCRETE_43_TABLE)
            if stage.kind == "ANNEAL_43_TO_CONTINUOUS":
                return torch.clamp((1.0 - stage.continuous_mix) * discrete + stage.continuous_mix * command, -1.0, 1.0)
            return discrete
        return command


# =============================================================================
# Environment
# =============================================================================
class CarParkingEnvMulti:
    def __init__(self, render_mode: bool = False, seed: Optional[int] = None) -> None:
        self.render_mode = bool(render_mode)
        self.area_size = float(CONFIG["AREA_SIZE"])
        self.num_cars = int(CONFIG["NUM_CARS"])
        self.max_steps_per_ep = int(CONFIG["MAX_STEPS_PER_EP"])
        self.rng = random.Random(int(CONFIG["SEED"] if seed is None else seed))

        self.cars: List[Entity] = []
        self.lots: List[Entity] = []
        self.walls: List[Entity] = []
        self.time_steps: Dict[int, int] = {}
        self.prev_dists: Dict[int, float] = {}
        self.prev_speeds: Dict[int, float] = {}
        self.prev_steers: Dict[int, float] = {}
        self.done_flags: Dict[int, bool] = {}

    @property
    def state_dim(self) -> int:
        return 13 + (3 if bool(CONFIG["USE_HIERARCHICAL_STATE"]) else 0)

    def _get_lot_for_car(self, car: Entity) -> Entity:
        for lot in self.lots:
            if lot.uid == car.target_id:
                return lot
        if not self.lots:
            raise RuntimeError("Environment has no parking lots.")
        return self.lots[car.uid % len(self.lots)]

    def hierarchical_phase(self, car: Entity, lot: Entity) -> int:
        distance = math.dist((car.x, car.y), (lot.x, lot.y))
        alignment = abs(angle_diff_deg(car.angle, lot.angle))
        if distance > 12.0:
            return 0  # approach
        if distance > 3.5 or alignment > 35.0:
            return 1  # align
        return 2      # settle

    def base_controller_action(self, car: Entity) -> np.ndarray:
        lot = self._get_lot_for_car(car)
        dx = lot.x - car.x
        dy = lot.y - car.y
        distance = math.hypot(dx, dy)
        desired_heading = math.degrees(math.atan2(dy, dx))
        heading_error = angle_diff_deg(desired_heading, car.angle)
        alignment_error = angle_diff_deg(lot.angle, car.angle)
        phase = self.hierarchical_phase(car, lot)

        if phase == 0:
            steer = clamp(heading_error / 60.0, -1.0, 1.0)
            throttle = 1.0 if abs(heading_error) < 75.0 else 0.35
        elif phase == 1:
            steer = clamp((0.65 * heading_error + 0.35 * alignment_error) / 60.0, -1.0, 1.0)
            throttle = 0.35 if distance > 5.0 else 0.15
        else:
            steer = clamp(alignment_error / 45.0, -1.0, 1.0)
            throttle = clamp((distance - 1.0) / 4.0, -0.35, 0.25)

        return np.array([steer, throttle], dtype=np.float32)

    def _build_state(self, car: Entity) -> np.ndarray:
        lot = self._get_lot_for_car(car)
        dx = lot.x - car.x
        dy = lot.y - car.y
        distance = math.hypot(dx, dy)
        bearing = math.degrees(math.atan2(dy, dx))
        relative_bearing = angle_diff_deg(bearing, car.angle)
        alignment = angle_diff_deg(car.angle, lot.angle)

        values = [
            car.x / self.area_size,
            car.y / self.area_size,
            math.cos(math.radians(car.angle)),
            math.sin(math.radians(car.angle)),
            car.l / self.area_size,
            car.w / self.area_size,
            dx / self.area_size,
            dy / self.area_size,
            distance / self.area_size,
            relative_bearing / 180.0,
            alignment / 180.0,
            lot.l / self.area_size,
            lot.w / self.area_size,
        ]

        if bool(CONFIG["USE_HIERARCHICAL_STATE"]):
            phase = self.hierarchical_phase(car, lot)
            values.extend([1.0 if phase == index else 0.0 for index in range(3)])

        return np.asarray(values, dtype=np.float32)

    def reset(self) -> List[np.ndarray]:
        self.cars, self.lots, self.walls = build_scenario(
            self.rng,
            str(CONFIG["SCENARIO"]),
            self.num_cars,
            self.area_size,
        )
        self.time_steps = {}
        self.prev_dists = {}
        self.prev_speeds = {}
        self.prev_steers = {}
        self.done_flags = {}

        for car in self.cars:
            lot = self._get_lot_for_car(car)
            self.time_steps[car.uid] = 0
            self.prev_dists[car.uid] = math.dist((car.x, car.y), (lot.x, lot.y))
            self.prev_speeds[car.uid] = 0.0
            self.prev_steers[car.uid] = 0.0
            self.done_flags[car.uid] = False

        return [self._build_state(car) for car in self.cars]

    def _decode_env_action(self, action: Union[int, Sequence[float], np.ndarray]) -> np.ndarray:
        if np.isscalar(action):
            index = int(action)
            if index < 0:
                raise ValueError("Discrete action cannot be negative.")
            if index < 9:
                return DISCRETE_9_TABLE[index].copy()
            if index < 43:
                return DISCRETE_43_TABLE[index].copy()
            raise ValueError(f"Discrete action index out of range: {index}")

        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.size != 2:
            raise ValueError(f"Continuous motor action must have exactly two values, got shape {values.shape}")
        return np.clip(values, -1.0, 1.0)

    def step(
        self,
        actions: Sequence[Union[int, Sequence[float], np.ndarray]],
        breaker: Optional[float] = None,
        play_mode: bool = False,
    ) -> Tuple[List[np.ndarray], List[float], List[bool]]:
        if len(actions) != len(self.cars):
            raise ValueError(f"Expected {len(self.cars)} actions, received {len(actions)}.")

        rewards: List[float] = []
        step_dones: List[bool] = []

        for index, car in enumerate(self.cars):
            if self.done_flags[car.uid]:
                rewards.append(0.0)
                step_dones.append(True)
                continue

            command = self._decode_env_action(actions[index])
            steer = float(command[0])
            throttle = float(command[1])
            self.time_steps[car.uid] += 1

            car.angle = (car.angle + float(CONFIG["TURN_DEG_PER_STEP"]) * steer) % 360.0
            speed = float(CONFIG["SPEED_OFFSET"]) + float(CONFIG["SPEED_SCALE"]) * throttle
            car.speed = speed
            radians = math.radians(car.angle)
            car.x += speed * math.cos(radians)
            car.y += speed * math.sin(radians)

            lot = self._get_lot_for_car(car)
            dx = lot.x - car.x
            dy = lot.y - car.y
            distance = math.hypot(dx, dy)
            previous_distance = self.prev_dists[car.uid]
            progress = clamp(
                previous_distance - distance,
                -float(CONFIG["REWARD_PROGRESS_CLIP"]),
                float(CONFIG["REWARD_PROGRESS_CLIP"]),
            )

            reward = 0.0
            done = False

            if not play_mode:
                reward += progress * float(CONFIG["REWARD_PROGRESS_SCALE"])
                bearing = math.degrees(math.atan2(dy, dx))
                relative_bearing = angle_diff_deg(bearing, car.angle)
                reward += bearing_reward(relative_bearing) * float(CONFIG["REWARD_BEARING_SCALE"])

                if abs(speed) > 0.0 and abs(steer) <= 0.3:
                    reward += float(CONFIG["REWARD_STRAIGHT_BONUS"])

                reward -= float(CONFIG["REWARD_TIME_PENALTY"])

                if self.prev_speeds[car.uid] * speed <= 0.0:
                    reward -= float(CONFIG["REWARD_DIRECTION_CHANGE_PENALTY"])

                if abs(self.prev_speeds[car.uid]) <= 0.2 and abs(speed) <= 0.2:
                    reward -= float(CONFIG["REWARD_IDLE_PENALTY"])

                if self.prev_steers[car.uid] * steer < -1.0:
                    reward -= float(CONFIG["REWARD_STEER_WAGGLE_PENALTY"])

                reward += abs(speed) / max(1e-6, float(CONFIG["REWARD_SPEED_DIVISOR"]))

            half = self.area_size / 2.0
            if abs(car.x) > half or abs(car.y) > half:
                if not play_mode:
                    reward -= float(CONFIG["REWARD_OUT_OF_BOUNDS_PENALTY"])
                done = True

            if not done:
                for wall in self.walls:
                    if PhysicsUtils.check_collision(car, wall):
                        if not play_mode:
                            reward -= float(CONFIG["REWARD_WALL_COLLISION_PENALTY"])
                        done = True
                        break

            if not done:
                for other_index, other in enumerate(self.cars):
                    if other_index == index:
                        continue
                    if PhysicsUtils.check_collision(car, other):
                        if not play_mode:
                            reward -= float(CONFIG["REWARD_CAR_COLLISION_PENALTY"])
                        done = True
                        break

            alignment = angle_diff_deg(car.angle, lot.angle)
            success = (
                PhysicsUtils.is_car_inside_lot(car, lot)
                and abs(alignment) < float(CONFIG["SUCCESS_ALIGNMENT_DEG"])
                and abs(car.speed) <= float(CONFIG["SUCCESS_MAX_SPEED"])
            )
            if success:
                if not play_mode:
                    reward += float(CONFIG["REWARD_SUCCESS"]) - self.time_steps[car.uid]
                done = True

            if self.time_steps[car.uid] >= self.max_steps_per_ep:
                done = True
            if breaker is not None and breaker <= -800:
                done = True

            self.prev_dists[car.uid] = distance
            self.prev_speeds[car.uid] = speed
            self.prev_steers[car.uid] = steer
            self.done_flags[car.uid] = done

            rewards.append(float(reward))
            step_dones.append(bool(done))

        return [self._build_state(car) for car in self.cars], rewards, step_dones

    def snapshot(self) -> Dict[str, Any]:
        def entity_data(entity: Entity) -> Dict[str, Any]:
            return {
                "uid": entity.uid,
                "x": entity.x,
                "y": entity.y,
                "angle": entity.angle,
                "l": entity.l,
                "w": entity.w,
                "type": entity.ent_type,
                "target_id": entity.target_id,
            }

        return {
            "area_size": self.area_size,
            "cars": [entity_data(entity) for entity in self.cars],
            "lots": [entity_data(entity) for entity in self.lots],
            "walls": [entity_data(entity) for entity in self.walls],
        }


# =============================================================================
# Rendering
# =============================================================================
def _snapshot_corners(raw: Dict[str, Any]) -> np.ndarray:
    entity = Entity(
        uid=int(raw.get("uid", 0)),
        team=0,
        x=float(raw["x"]),
        y=float(raw["y"]),
        angle=float(raw["angle"]),
        l=float(raw["l"]),
        w=float(raw["w"]),
        ent_type=str(raw.get("type", "ENTITY")),
    )
    return PhysicsUtils.get_corners(entity)


class SnapshotRenderer:
    def __init__(self, title: str) -> None:
        self.title = title
        self.fig, self.ax = plt.subplots(figsize=tuple(CONFIG["RENDER_WINDOW_SIZE"]))
        self.closed = False
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _on_close(self, _event: Any) -> None:
        self.closed = True

    def draw(self, snapshot: Dict[str, Any], subtitle: str = "", trails: Optional[List[List[Tuple[float, float]]]] = None) -> None:
        if self.closed:
            return
        area_size = float(snapshot["area_size"])
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-area_size / 2.0 - 2.0, area_size / 2.0 + 2.0)
        self.ax.set_ylim(-area_size / 2.0 - 2.0, area_size / 2.0 + 2.0)
        self.ax.grid(True, alpha=0.25)
        self.ax.set_title(f"{self.title}\n{subtitle}")

        for wall in snapshot["walls"]:
            self.ax.add_patch(Polygon(_snapshot_corners(wall), closed=True, facecolor="black", alpha=0.65))
        for lot in snapshot["lots"]:
            self.ax.add_patch(Polygon(_snapshot_corners(lot), closed=True, facecolor="tab:blue", alpha=0.35))
        for car in snapshot["cars"]:
            self.ax.add_patch(Polygon(_snapshot_corners(car), closed=True, facecolor="tab:red", alpha=0.75))
            heading = math.radians(float(car["angle"]))
            self.ax.arrow(
                float(car["x"]),
                float(car["y"]),
                math.cos(heading) * 3.0,
                math.sin(heading) * 3.0,
                width=0.15,
                head_width=0.8,
                length_includes_head=True,
                color="darkred",
            )

        if trails is not None and bool(CONFIG["RENDER_TRAIL"]):
            for trail in trails:
                if len(trail) > 1:
                    xs = [point[0] for point in trail]
                    ys = [point[1] for point in trail]
                    self.ax.plot(xs, ys, linewidth=1.0, alpha=0.35)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def close(self) -> None:
        if not self.closed:
            plt.close(self.fig)
        self.closed = True


def async_renderer_main(render_queue: mp.Queue) -> None:
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    renderer: Optional[SnapshotRenderer] = None
    frame_delay = 1.0 / max(1, int(CONFIG["RENDER_FPS"]))

    while True:
        try:
            message = render_queue.get(timeout=0.25)
        except queue.Empty:
            continue

        if message is None:
            break

        episode = int(message["episode"])
        stage = str(message["stage"])
        reward = float(message["reward"])
        snapshots = message["snapshots"]
        print(f"[ASYNC RENDER] Replaying episode {episode} ({stage}), reward={reward:.1f}")

        if renderer is None or renderer.closed:
            renderer = SnapshotRenderer("Training replay — separate process")

        trails: List[List[Tuple[float, float]]] = [[] for _ in snapshots[0]["cars"]]
        for frame_index, snapshot in enumerate(snapshots):
            for car_index, car in enumerate(snapshot["cars"]):
                trails[car_index].append((float(car["x"]), float(car["y"])))
            renderer.draw(
                snapshot,
                subtitle=f"Episode {episode} | {stage} | frame {frame_index + 1}/{len(snapshots)} | reward {reward:.1f}",
                trails=trails,
            )
            if renderer.closed:
                break
            time.sleep(frame_delay)

    if renderer is not None:
        renderer.close()


class AsyncRenderManager:
    def __init__(self) -> None:
        self.process: Optional[mp.Process] = None
        self.queue: Optional[mp.Queue] = None

    def start(self) -> None:
        if not bool(CONFIG["ASYNC_RENDER"]):
            return
        context = mp.get_context("spawn")
        self.queue = context.Queue(maxsize=int(CONFIG["RENDER_QUEUE_SIZE"]))
        self.process = context.Process(target=async_renderer_main, args=(self.queue,), daemon=True)
        self.process.start()

    def submit(self, payload: Dict[str, Any]) -> bool:
        if self.queue is None:
            return False
        try:
            self.queue.put_nowait(payload)
            return True
        except queue.Full:
            return False

    def close(self) -> None:
        if self.queue is not None:
            try:
                self.queue.put_nowait(None)
            except Exception:
                pass
        if self.process is not None:
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                self.process.terminate()


# =============================================================================
# PPO
# =============================================================================
class PPOActorCritic(nn.Module):
    def __init__(self, state_dim: int, policy_kind: str, discrete_size: int = 0) -> None:
        super().__init__()
        self.policy_kind = policy_kind
        hidden = 256
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        if self.policy_kind == "CATEGORICAL":
            self.policy_head = nn.Linear(hidden, discrete_size)
        else:
            self.mean_head = nn.Linear(hidden, 2)
            self.log_std = nn.Parameter(torch.full((2,), -0.5))
        self.value_head = nn.Linear(hidden, 1)

    def distribution_and_value(self, states: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        features = self.backbone(states)
        value = self.value_head(features).squeeze(-1)
        if self.policy_kind == "CATEGORICAL":
            distribution = torch.distributions.Categorical(logits=self.policy_head(features))
        else:
            mean = self.mean_head(features)
            std = torch.exp(torch.clamp(self.log_std, -5.0, 1.0)).expand_as(mean)
            distribution = torch.distributions.Normal(mean, std)
        return distribution, value

    @staticmethod
    def tanh_log_prob(distribution: torch.distributions.Normal, raw_action: torch.Tensor) -> torch.Tensor:
        squashed = torch.tanh(raw_action)
        return distribution.log_prob(raw_action).sum(dim=-1) - torch.log(1.0 - squashed.pow(2) + 1e-6).sum(dim=-1)

    @torch.no_grad()
    def act(self, states: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.distribution_and_value(states)
        if self.policy_kind == "CATEGORICAL":
            action = torch.argmax(distribution.logits, dim=-1) if deterministic else distribution.sample()
            log_prob = distribution.log_prob(action)
            return action, log_prob, value

        raw_action = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = self.tanh_log_prob(distribution, raw_action)
        return action, log_prob, value

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.distribution_and_value(states)
        if self.policy_kind == "CATEGORICAL":
            log_prob = distribution.log_prob(actions.long())
            entropy = distribution.entropy()
            return log_prob, entropy, value

        actions = torch.clamp(actions, -0.999, 0.999)
        raw_action = 0.5 * torch.log((1.0 + actions) / (1.0 - actions))
        log_prob = self.tanh_log_prob(distribution, raw_action)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy, value


@dataclass
class PPOTransition:
    state: np.ndarray
    action: Union[int, np.ndarray]
    old_log_prob: float
    reward: float
    done: bool
    value: float


class PPOAgent:
    def __init__(self, state_dim: int, adapter: ActionAdapter) -> None:
        self.device = torch.device(str(CONFIG["DEVICE"]))
        self.adapter = adapter
        self.policy_kind = adapter.ppo_policy_kind()
        discrete_size = adapter.categorical_size() if self.policy_kind == "CATEGORICAL" else 0
        self.model = PPOActorCritic(state_dim, self.policy_kind, discrete_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(CONFIG["LEARNING_RATE"]))

    @torch.no_grad()
    def act(self, states: List[np.ndarray], deterministic: bool = False) -> Tuple[List[Any], List[float], List[float]]:
        tensor = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
        actions, log_probs, values = self.model.act(tensor, deterministic=deterministic)
        if self.policy_kind == "CATEGORICAL":
            action_values: List[Any] = actions.cpu().numpy().astype(np.int64).tolist()
        else:
            action_values = [row.astype(np.float32) for row in actions.cpu().numpy()]
        return (
            action_values,
            log_probs.cpu().numpy().astype(np.float32).tolist(),
            values.cpu().numpy().astype(np.float32).tolist(),
        )

    @torch.no_grad()
    def values(self, states: List[np.ndarray]) -> List[float]:
        tensor = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
        _, values = self.model.distribution_and_value(tensor)
        return values.cpu().numpy().astype(np.float32).tolist()

    def update(self, trajectories: List[List[PPOTransition]], next_states: List[np.ndarray]) -> Dict[str, float]:
        gamma = float(CONFIG["GAMMA"])
        gae_lambda = float(CONFIG["GAE_LAMBDA"])
        next_values = self.values(next_states)

        states_all: List[np.ndarray] = []
        actions_all: List[Any] = []
        log_probs_all: List[float] = []
        returns_all: List[float] = []
        advantages_all: List[float] = []

        for car_index, trajectory in enumerate(trajectories):
            if not trajectory:
                continue
            advantage = 0.0
            next_value = 0.0 if trajectory[-1].done else float(next_values[car_index])
            local_returns = [0.0] * len(trajectory)
            local_advantages = [0.0] * len(trajectory)

            for index in reversed(range(len(trajectory))):
                transition = trajectory[index]
                nonterminal = 0.0 if transition.done else 1.0
                delta = transition.reward + gamma * next_value * nonterminal - transition.value
                advantage = delta + gamma * gae_lambda * nonterminal * advantage
                local_advantages[index] = advantage
                local_returns[index] = advantage + transition.value
                next_value = transition.value

            for transition, return_value, advantage_value in zip(trajectory, local_returns, local_advantages):
                states_all.append(transition.state)
                actions_all.append(transition.action)
                log_probs_all.append(transition.old_log_prob)
                returns_all.append(return_value)
                advantages_all.append(advantage_value)

        if not states_all:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        states = torch.as_tensor(np.stack(states_all), dtype=torch.float32, device=self.device)
        if self.policy_kind == "CATEGORICAL":
            actions = torch.as_tensor(np.asarray(actions_all), dtype=torch.long, device=self.device)
        else:
            actions = torch.as_tensor(np.stack(actions_all), dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(log_probs_all, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns_all, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages_all, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(states.shape[0])
        batch_size = int(CONFIG["BATCH_SIZE"])
        stats = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        updates = 0

        for _ in range(int(CONFIG["PPO_EPOCHS"])):
            np.random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                log_probs, entropy, values = self.model.evaluate(states[batch], actions[batch])
                ratio = torch.exp(log_probs - old_log_probs[batch])
                unclipped = ratio * advantages[batch]
                clipped = torch.clamp(
                    ratio,
                    1.0 - float(CONFIG["PPO_CLIP_EPS"]),
                    1.0 + float(CONFIG["PPO_CLIP_EPS"]),
                ) * advantages[batch]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, returns[batch])
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + float(CONFIG["PPO_VALUE_COEF"]) * value_loss
                    - float(CONFIG["PPO_ENTROPY_COEF"]) * entropy_mean
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), float(CONFIG["MAX_GRAD_NORM"]))
                self.optimizer.step()

                stats["loss"] += float(loss.detach().cpu())
                stats["policy_loss"] += float(policy_loss.detach().cpu())
                stats["value_loss"] += float(value_loss.detach().cpu())
                stats["entropy"] += float(entropy_mean.detach().cpu())
                updates += 1

        for key in stats:
            stats[key] /= max(1, updates)
        return stats

    def save(self, path: Path, episode: int, best_reward: float, stage: ControlStage) -> None:
        torch.save(
            {
                "algorithm": "PPO",
                "episode": episode,
                "best_reward": best_reward,
                "stage": stage.__dict__,
                "policy_kind": self.policy_kind,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": dict(CONFIG),
            },
            path,
        )

    def load(self, path: Path, load_optimizer: bool = True) -> Dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device)
        current = self.model.state_dict()
        compatible = {
            key: value
            for key, value in checkpoint["model"].items()
            if key in current and current[key].shape == value.shape
        }
        missing, unexpected = self.model.load_state_dict(compatible, strict=False)
        if missing:
            print(f"[LOAD] Reinitialized incompatible PPO parameters: {len(missing)}")
        if unexpected:
            print(f"[LOAD] Ignored unexpected PPO parameters: {len(unexpected)}")
        if load_optimizer and "optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception:
                print("[LOAD] Optimizer state did not match; using a fresh optimizer.")
        return checkpoint


# =============================================================================
# SAC / SOC
# =============================================================================
class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.data: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float, np.ndarray, np.ndarray]] = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        base_action: np.ndarray,
        next_base_action: np.ndarray,
    ) -> None:
        self.data.append(
            (
                state.astype(np.float32),
                action.astype(np.float32),
                float(reward),
                next_state.astype(np.float32),
                float(done),
                base_action.astype(np.float32),
                next_base_action.astype(np.float32),
            )
        )

    def sample(self, size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.data, size)
        fields = list(zip(*batch))
        return tuple(np.stack(field) if index not in {2, 4} else np.asarray(field, dtype=np.float32) for index, field in enumerate(fields))

    def __len__(self) -> int:
        return len(self.data)


class SACActor(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, 2)
        self.log_std = nn.Linear(256, 2)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(states)
        return self.mean(features), torch.clamp(self.log_std(features), -20.0, 2.0)

    def sample(self, states: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(states)
        std = torch.exp(log_std)
        distribution = torch.distributions.Normal(mean, std)
        raw = mean if deterministic else distribution.rsample()
        action = torch.tanh(raw)
        log_prob = distribution.log_prob(raw).sum(dim=-1) - torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob


class SACCritic(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([states, actions], dim=-1)).squeeze(-1)


class SACAgent:
    def __init__(self, state_dim: int, adapter: ActionAdapter) -> None:
        self.device = torch.device(str(CONFIG["DEVICE"]))
        self.adapter = adapter
        self.actor = SACActor(state_dim).to(self.device)
        self.critic1 = SACCritic(state_dim).to(self.device)
        self.critic2 = SACCritic(state_dim).to(self.device)
        self.target1 = SACCritic(state_dim).to(self.device)
        self.target2 = SACCritic(state_dim).to(self.device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        lr = float(CONFIG["LEARNING_RATE"])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=float(CONFIG["SAC_ALPHA_LR"]))
        self.replay = ReplayBuffer(int(CONFIG["SAC_REPLAY_SIZE"]))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def latent_actions(self, states: List[np.ndarray], deterministic: bool = False) -> List[np.ndarray]:
        tensor = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
        actions, _ = self.actor.sample(tensor, deterministic=deterministic)
        return [row.astype(np.float32) for row in actions.cpu().numpy()]

    def update(self, stage: ControlStage) -> Dict[str, float]:
        if len(self.replay) < int(CONFIG["BATCH_SIZE"]):
            return {"critic_loss": 0.0, "actor_loss": 0.0, "alpha": float(self.alpha.detach().cpu())}

        states_np, actions_np, rewards_np, next_states_np, dones_np, base_np, next_base_np = self.replay.sample(int(CONFIG["BATCH_SIZE"]))
        states = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions_np, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states_np, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones_np, dtype=torch.float32, device=self.device)
        base_actions = torch.as_tensor(base_np, dtype=torch.float32, device=self.device)
        next_base_actions = torch.as_tensor(next_base_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_latent, next_log_prob = self.actor.sample(next_states)
            next_executed = self.adapter.decode_torch(next_latent, stage, next_base_actions)
            target_q = torch.min(self.target1(next_states, next_executed), self.target2(next_states, next_executed))
            target = rewards + float(CONFIG["GAMMA"]) * (1.0 - dones) * (target_q - self.alpha.detach() * next_log_prob)

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(q1, target)
        critic2_loss = F.mse_loss(q2, target)

        self.critic1_optimizer.zero_grad(set_to_none=True)
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad(set_to_none=True)
        critic2_loss.backward()
        self.critic2_optimizer.step()

        latent, log_prob = self.actor.sample(states)
        executed = self.adapter.decode_torch(latent, stage, base_actions)
        actor_loss = (self.alpha.detach() * log_prob - torch.min(self.critic1(states, executed), self.critic2(states, executed))).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + float(CONFIG["SAC_TARGET_ENTROPY"])).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        tau = float(CONFIG["SAC_TAU"])
        with torch.no_grad():
            for source, target_network in [(self.critic1, self.target1), (self.critic2, self.target2)]:
                for source_parameter, target_parameter in zip(source.parameters(), target_network.parameters()):
                    target_parameter.data.mul_(1.0 - tau).add_(tau * source_parameter.data)

        return {
            "critic_loss": float((critic1_loss + critic2_loss).detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
        }

    def save(self, path: Path, episode: int, best_reward: float, stage: ControlStage) -> None:
        torch.save(
            {
                "algorithm": "SAC",
                "episode": episode,
                "best_reward": best_reward,
                "stage": stage.__dict__,
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "target1": self.target1.state_dict(),
                "target2": self.target2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic1_optimizer": self.critic1_optimizer.state_dict(),
                "critic2_optimizer": self.critic2_optimizer.state_dict(),
                "log_alpha": float(self.log_alpha.detach().cpu()),
                "config": dict(CONFIG),
            },
            path,
        )

    def load(self, path: Path, load_optimizer: bool = True) -> Dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"], strict=False)
        self.critic1.load_state_dict(checkpoint["critic1"], strict=False)
        self.critic2.load_state_dict(checkpoint["critic2"], strict=False)
        self.target1.load_state_dict(checkpoint.get("target1", checkpoint["critic1"]), strict=False)
        self.target2.load_state_dict(checkpoint.get("target2", checkpoint["critic2"]), strict=False)
        if load_optimizer:
            for optimizer, key in [
                (self.actor_optimizer, "actor_optimizer"),
                (self.critic1_optimizer, "critic1_optimizer"),
                (self.critic2_optimizer, "critic2_optimizer"),
            ]:
                try:
                    optimizer.load_state_dict(checkpoint[key])
                except Exception:
                    pass
        if "log_alpha" in checkpoint:
            self.log_alpha.data.fill_(float(checkpoint["log_alpha"]))
        return checkpoint


# =============================================================================
# Checkpoints and logging
# =============================================================================
def checkpoint_stem(algorithm: str, action_mode: str) -> str:
    return f"{algorithm.lower()}_{action_mode.lower()}"


def checkpoint_paths(algorithm: str, action_mode: str) -> Dict[str, Path]:
    directories = ensure_output_dirs()
    stem = checkpoint_stem(algorithm, action_mode)
    return {
        "latest": directories["checkpoints"] / f"{stem}_latest.pth",
        "best": directories["checkpoints"] / f"{stem}_best.pth",
    }


def resolve_checkpoint(value: str, algorithm: str, action_mode: str) -> Path:
    if str(value).upper() != "AUTO":
        return Path(value)
    paths = checkpoint_paths(algorithm, action_mode)
    if paths["latest"].exists():
        return paths["latest"]
    if paths["best"].exists():
        return paths["best"]
    raise FileNotFoundError(f"No checkpoint found for {algorithm}/{action_mode} in {paths['latest'].parent}")


def append_training_log(row: Dict[str, Any]) -> None:
    path = ensure_output_dirs()["root"] / "training_log.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def should_record_episode(episode: int) -> bool:
    interval = int(CONFIG["RENDER_INTERVAL"])
    explicit = {int(value) for value in CONFIG.get("RENDER_EPISODES", [])}
    return (interval > 0 and episode % interval == 0) or episode in explicit


# =============================================================================
# Training helpers
# =============================================================================
def policy_to_env_actions(
    env: CarParkingEnvMulti,
    adapter: ActionAdapter,
    policy_actions: Sequence[Any],
    stage: ControlStage,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    env_actions: List[np.ndarray] = []
    base_actions: List[np.ndarray] = []
    for car, policy_action in zip(env.cars, policy_actions):
        base = env.base_controller_action(car) if bool(CONFIG["USE_RESIDUAL_RL"]) else np.zeros(2, dtype=np.float32)
        decoded = adapter.decode_numpy(policy_action, stage, base)
        env_actions.append(decoded)
        base_actions.append(base)
    return env_actions, base_actions


def create_agent(state_dim: int, adapter: ActionAdapter) -> Union[PPOAgent, SACAgent]:
    if adapter.algorithm == "PPO":
        return PPOAgent(state_dim, adapter)
    return SACAgent(state_dim, adapter)


def train(start_from_checkpoint: Optional[Path] = None) -> None:
    set_seed(int(CONFIG["SEED"]))
    algorithm = canonical_algorithm(str(CONFIG["ALGORITHM"]))
    action_mode = canonical_action_mode(str(CONFIG["ACTION_MODE"]))
    adapter = ActionAdapter(action_mode, algorithm)
    env = CarParkingEnvMulti(render_mode=False)
    states = env.reset()
    agent = create_agent(env.state_dim, adapter)

    start_episode = 1
    best_reward = -float("inf")
    if start_from_checkpoint is not None:
        checkpoint = agent.load(start_from_checkpoint, load_optimizer=True)
        start_episode = int(checkpoint.get("episode", 0)) + 1
        best_reward = float(checkpoint.get("best_reward", -float("inf")))
        print(f"[TRANSFER] Loaded {start_from_checkpoint} and will continue at episode {start_episode}.")

    render_manager = AsyncRenderManager()
    render_manager.start()
    paths = checkpoint_paths(algorithm, action_mode)
    previous_stage_name: Optional[str] = None
    total_environment_steps = 0

    try:
        for episode in range(start_episode, int(CONFIG["EPISODES"]) + 1):
            stage = adapter.curriculum.stage_for_episode(episode)
            if stage.name != previous_stage_name:
                print(
                    f"[CURRICULUM] Episode {episode}: {stage.name} | "
                    f"execution={stage.kind} | actions={stage.action_count or 'continuous'} | "
                    f"continuous_mix={stage.continuous_mix:.3f}"
                )
                if previous_stage_name is not None:
                    phase_path = paths["latest"].with_name(
                        f"{checkpoint_stem(algorithm, action_mode)}_phase_{stage.phase_index}_{stage.name}.pth"
                    )
                    agent.save(phase_path, episode - 1, best_reward, stage)
                previous_stage_name = stage.name

            states = env.reset()
            done_flags = [False] * env.num_cars
            episode_reward = 0.0
            steps = 0
            successes = 0
            snapshots: List[Dict[str, Any]] = []
            record_episode = should_record_episode(episode)

            ppo_trajectories: List[List[PPOTransition]] = [[] for _ in range(env.num_cars)]
            update_stats: Dict[str, float] = (
                {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
                if algorithm == "PPO"
                else {"critic_loss": 0.0, "actor_loss": 0.0, "alpha": 1.0}
            )

            while not all(done_flags) and steps < env.max_steps_per_ep:
                steps += 1
                total_environment_steps += env.num_cars

                if algorithm == "PPO":
                    assert isinstance(agent, PPOAgent)
                    policy_actions, log_probs, values = agent.act(states, deterministic=False)
                else:
                    assert isinstance(agent, SACAgent)
                    if total_environment_steps < int(CONFIG["SAC_WARMUP_STEPS"]):
                        policy_actions = [np.random.uniform(-1.0, 1.0, size=2).astype(np.float32) for _ in states]
                    else:
                        policy_actions = agent.latent_actions(states, deterministic=False)
                    log_probs = [0.0] * len(states)
                    values = [0.0] * len(states)

                env_actions, base_actions = policy_to_env_actions(env, adapter, policy_actions, stage)
                next_states, rewards, step_dones = env.step(env_actions, None, play_mode=False)

                if algorithm == "PPO":
                    assert isinstance(agent, PPOAgent)
                    for car_index in range(env.num_cars):
                        if done_flags[car_index]:
                            continue
                        stored_action: Union[int, np.ndarray]
                        if agent.policy_kind == "CATEGORICAL":
                            stored_action = int(policy_actions[car_index])
                        else:
                            stored_action = np.asarray(policy_actions[car_index], dtype=np.float32)
                        ppo_trajectories[car_index].append(
                            PPOTransition(
                                state=states[car_index],
                                action=stored_action,
                                old_log_prob=float(log_probs[car_index]),
                                reward=float(rewards[car_index]),
                                done=bool(step_dones[car_index]),
                                value=float(values[car_index]),
                            )
                        )
                else:
                    assert isinstance(agent, SACAgent)
                    next_base_actions = [
                        env.base_controller_action(car) if bool(CONFIG["USE_RESIDUAL_RL"]) else np.zeros(2, dtype=np.float32)
                        for car in env.cars
                    ]
                    for car_index in range(env.num_cars):
                        if done_flags[car_index]:
                            continue
                        agent.replay.add(
                            states[car_index],
                            env_actions[car_index],
                            rewards[car_index],
                            next_states[car_index],
                            step_dones[car_index],
                            base_actions[car_index],
                            next_base_actions[car_index],
                        )
                    if total_environment_steps >= int(CONFIG["SAC_WARMUP_STEPS"]):
                        for _ in range(int(CONFIG["SAC_UPDATES_PER_STEP"])):
                            update_stats = agent.update(stage)

                states = next_states
                for index, new_done in enumerate(step_dones):
                    if not done_flags[index] and new_done:
                        lot = env._get_lot_for_car(env.cars[index])
                        success = PhysicsUtils.is_car_inside_lot(env.cars[index], lot)
                        successes += int(success)
                    done_flags[index] = done_flags[index] or bool(new_done)
                episode_reward += float(np.sum(rewards))

                if record_episode:
                    snapshots.append(env.snapshot())

            if algorithm == "PPO":
                assert isinstance(agent, PPOAgent)
                update_stats = agent.update(ppo_trajectories, states)

            best_reward = max(best_reward, episode_reward)
            queued = False
            if record_episode and snapshots:
                queued = render_manager.submit(
                    {
                        "episode": episode,
                        "stage": stage.name,
                        "reward": episode_reward,
                        "snapshots": snapshots,
                    }
                )

            if episode % int(CONFIG["LOG_EVERY"]) == 0 or episode == start_episode:
                if algorithm == "PPO":
                    print(
                        f"[TRAIN][PPO] ep={episode:5d} stage={stage.name:<28} "
                        f"reward={episode_reward:10.1f} best={best_reward:10.1f} steps={steps:3d} "
                        f"success={successes}/{env.num_cars} render={'queued' if queued else 'no'} "
                        f"loss={update_stats['loss']:.4f}"
                    )
                else:
                    print(
                        f"[TRAIN][SAC/SOC] ep={episode:5d} stage={stage.name:<28} "
                        f"reward={episode_reward:10.1f} best={best_reward:10.1f} steps={steps:3d} "
                        f"success={successes}/{env.num_cars} render={'queued' if queued else 'no'} "
                        f"critic={update_stats['critic_loss']:.4f} actor={update_stats['actor_loss']:.4f}"
                    )

            append_training_log(
                {
                    "episode": episode,
                    "algorithm": algorithm,
                    "configured_action_mode": action_mode,
                    "stage": stage.name,
                    "execution_kind": stage.kind,
                    "action_count": stage.action_count if stage.action_count is not None else "continuous",
                    "continuous_mix": round(stage.continuous_mix, 6),
                    "reward": round(episode_reward, 6),
                    "best_reward": round(best_reward, 6),
                    "steps": steps,
                    "successes": successes,
                    "num_cars": env.num_cars,
                    "render_queued": queued,
                }
            )

            if episode_reward >= best_reward:
                agent.save(paths["best"], episode, best_reward, stage)
            if episode % int(CONFIG["CHECKPOINT_EVERY"]) == 0 or episode == int(CONFIG["EPISODES"]):
                agent.save(paths["latest"], episode, best_reward, stage)

    finally:
        render_manager.close()


# =============================================================================
# Inference and play
# =============================================================================
def inference() -> None:
    set_seed(int(CONFIG["SEED"]))
    algorithm = canonical_algorithm(str(CONFIG["ALGORITHM"]))
    action_mode = canonical_action_mode(str(CONFIG["ACTION_MODE"]))
    adapter = ActionAdapter(action_mode, algorithm)
    env = CarParkingEnvMulti(render_mode=True)
    states = env.reset()
    agent = create_agent(env.state_dim, adapter)
    checkpoint = resolve_checkpoint(str(CONFIG["INFERENCE_FROM"]), algorithm, action_mode)
    agent.load(checkpoint, load_optimizer=False)
    print(f"[INFERENCE] Loaded {checkpoint}")

    renderer = SnapshotRenderer("Inference") if bool(CONFIG["INFERENCE_RENDER"]) else None
    try:
        for episode in range(1, int(CONFIG["INFERENCE_EPISODES"]) + 1):
            stage = adapter.curriculum.stage_for_episode(10_000_000)
            states = env.reset()
            done_flags = [False] * env.num_cars
            reward_total = 0.0
            trails: List[List[Tuple[float, float]]] = [[] for _ in range(env.num_cars)]

            for step in range(1, env.max_steps_per_ep + 1):
                if algorithm == "PPO":
                    assert isinstance(agent, PPOAgent)
                    policy_actions, _, _ = agent.act(states, deterministic=True)
                else:
                    assert isinstance(agent, SACAgent)
                    policy_actions = agent.latent_actions(states, deterministic=True)

                env_actions, _ = policy_to_env_actions(env, adapter, policy_actions, stage)
                states, rewards, step_dones = env.step(env_actions, None, play_mode=True)
                reward_total += float(np.sum(rewards))
                done_flags = [old or new for old, new in zip(done_flags, step_dones)]

                if renderer is not None:
                    for car_index, car in enumerate(env.cars):
                        trails[car_index].append((car.x, car.y))
                    renderer.draw(
                        env.snapshot(),
                        subtitle=f"Episode {episode} | step {step} | reward {reward_total:.1f}",
                        trails=trails,
                    )
                    if renderer.closed:
                        return
                    time.sleep(1.0 / max(1, int(CONFIG["RENDER_FPS"])))

                if all(done_flags):
                    break

            print(f"[INFERENCE] Episode {episode}: reward={reward_total:.1f}")
    finally:
        if renderer is not None:
            renderer.close()


class ManualPlaySession:
    def __init__(self, env: CarParkingEnvMulti, adapter: ActionAdapter) -> None:
        self.env = env
        self.adapter = adapter
        self.renderer = SnapshotRenderer("PLAY — W/S throttle, A/D steer, R reset, Q quit")
        self.keys = {"w": False, "a": False, "s": False, "d": False}
        self.quit = False
        self.reset_requested = False
        self.renderer.fig.canvas.mpl_connect("key_press_event", self.on_press)
        self.renderer.fig.canvas.mpl_connect("key_release_event", self.on_release)

    def on_press(self, event: Any) -> None:
        key = str(event.key).lower()
        if key in self.keys:
            self.keys[key] = True
        elif key == "r":
            self.reset_requested = True
        elif key in {"q", "escape"}:
            self.quit = True

    def on_release(self, event: Any) -> None:
        key = str(event.key).lower()
        if key in self.keys:
            self.keys[key] = False

    def human_command(self) -> np.ndarray:
        steer = float(self.keys["d"]) - float(self.keys["a"])
        throttle = float(self.keys["w"]) - float(self.keys["s"])
        if steer != 0.0 and throttle != 0.0:
            steer *= 0.7
            throttle *= 0.7
        return np.array([steer, throttle], dtype=np.float32)

    def run(self) -> None:
        states = self.env.reset()
        episode = 1
        reward_total = 0.0
        step = 0
        trails: List[List[Tuple[float, float]]] = [[] for _ in range(self.env.num_cars)]
        stage = self.adapter.curriculum.stage_for_episode(10_000_000)
        frame_delay = 1.0 / max(1, int(CONFIG["PLAY_FPS"]))

        while not self.quit and not self.renderer.closed:
            if self.reset_requested:
                states = self.env.reset()
                reward_total = 0.0
                step = 0
                episode += 1
                trails = [[] for _ in range(self.env.num_cars)]
                self.reset_requested = False

            command = self.human_command()
            actions: List[np.ndarray] = []
            for car_index, car in enumerate(self.env.cars):
                if car_index == 0:
                    actions.append(self.adapter.quantize_numpy(command, stage))
                else:
                    actions.append(self.env.base_controller_action(car))

            states, rewards, dones = self.env.step(actions, None, play_mode=True)
            reward_total += float(np.sum(rewards))
            step += 1
            for index, car in enumerate(self.env.cars):
                trails[index].append((car.x, car.y))

            self.renderer.draw(
                self.env.snapshot(),
                subtitle=f"Episode {episode} | step {step} | reward {reward_total:.1f} | command {command.tolist()}",
                trails=trails,
            )

            if all(dones) or step >= self.env.max_steps_per_ep:
                self.reset_requested = True

            time.sleep(frame_delay)

        self.renderer.close()


def play() -> None:
    set_seed(int(CONFIG["SEED"]))
    algorithm = canonical_algorithm(str(CONFIG["ALGORITHM"]))
    action_mode = canonical_action_mode(str(CONFIG["ACTION_MODE"]))
    env = CarParkingEnvMulti(render_mode=True)
    adapter = ActionAdapter(action_mode, algorithm)
    ManualPlaySession(env, adapter).run()


def transfer() -> None:
    algorithm = canonical_algorithm(str(CONFIG["ALGORITHM"]))
    action_mode = canonical_action_mode(str(CONFIG["ACTION_MODE"]))
    checkpoint = resolve_checkpoint(str(CONFIG["TRANSFER_FROM"]), algorithm, action_mode)
    train(start_from_checkpoint=checkpoint)


# =============================================================================
# Portfolio-friendly CLI and machine-readable project description
# =============================================================================
def project_description() -> Dict[str, Any]:
    """Return a compact, machine-readable explanation of the experiment."""
    base_observation = [
        "car_x_normalized",
        "car_y_normalized",
        "heading_cos",
        "heading_sin",
        "car_length_normalized",
        "car_width_normalized",
        "target_dx_normalized",
        "target_dy_normalized",
        "target_distance_normalized",
        "relative_bearing_normalized",
        "parking_alignment_normalized",
        "lot_length_normalized",
        "lot_width_normalized",
    ]
    if bool(CONFIG["USE_HIERARCHICAL_STATE"]):
        base_observation.extend(["phase_approach", "phase_align", "phase_settle"])

    return {
        "project": "Parking RL Lab",
        "purpose": "Train and evaluate parking policies across discrete-to-continuous control regimes.",
        "algorithms": {
            "PPO": "On-policy actor-critic with categorical or squashed-Gaussian actions.",
            "SAC": "Off-policy twin-critic actor-critic with entropy tuning.",
        },
        "action_spaces": {
            "DISCRETE_9": {"commands": 9, "shape": ["steering", "throttle"]},
            "DISCRETE_43": {"commands": 43, "construction": "7 steering x 6 throttle + neutral"},
            "CONTINUOUS": {"bounds": [-1.0, 1.0], "shape": ["steering", "throttle"]},
            "CURRICULUM": [
                "DISCRETE_9",
                "DISCRETE_43",
                "ANNEAL_43_TO_CONTINUOUS",
                "CONTINUOUS",
            ],
        },
        "observation": {"dimensions": len(base_observation), "fields": base_observation},
        "scenarios": ["SIMPLE", "WALLS", "CUSTOM", "RANDOM"],
        "termination": ["parked", "collision", "out_of_bounds", "step_limit"],
        "artifacts": ["checkpoints", "training_log.csv", "optional asynchronous replays"],
        "resolved_config": dict(CONFIG),
    }


def smoke_test_report() -> Dict[str, Any]:
    """Exercise physics, action decoding, both policy families, and one env step."""
    set_seed(int(CONFIG["SEED"]))
    CONFIG["ASYNC_RENDER"] = False
    CONFIG["INFERENCE_RENDER"] = False
    CONFIG["DEVICE"] = "cpu"

    assert DISCRETE_9_TABLE.shape == (9, 2)
    assert DISCRETE_43_TABLE.shape == (43, 2)
    assert np.allclose(DISCRETE_43_TABLE[-1], np.zeros(2, dtype=np.float32))

    env = CarParkingEnvMulti(render_mode=False, seed=int(CONFIG["SEED"]))
    states = env.reset()
    assert len(states) == env.num_cars
    assert all(state.shape == (env.state_dim,) for state in states)

    first = Entity(1, 0, 0.0, 0.0, 0.0, 4.0, 2.0, "CAR")
    overlapping = Entity(2, 0, 1.0, 0.0, 15.0, 4.0, 2.0, "CAR")
    separate = Entity(3, 0, 20.0, 20.0, 0.0, 4.0, 2.0, "CAR")
    assert PhysicsUtils.check_collision(first, overlapping)
    assert not PhysicsUtils.check_collision(first, separate)

    ppo_adapter = ActionAdapter(str(CONFIG["ACTION_MODE"]), "PPO")
    ppo_agent = PPOAgent(env.state_dim, ppo_adapter)
    ppo_actions, _, _ = ppo_agent.act(states, deterministic=True)
    stage = ppo_adapter.curriculum.stage_for_episode(1)
    env_actions, _ = policy_to_env_actions(env, ppo_adapter, ppo_actions, stage)
    next_states, rewards, dones = env.step(env_actions)

    sac_adapter = ActionAdapter("CONTINUOUS", "SAC")
    sac_agent = SACAgent(env.state_dim, sac_adapter)
    sac_actions = sac_agent.latent_actions(states, deterministic=True)

    assert len(next_states) == env.num_cars
    assert len(rewards) == env.num_cars
    assert len(dones) == env.num_cars
    assert np.isfinite(np.asarray(rewards, dtype=np.float32)).all()
    assert all(np.asarray(action).shape == (2,) for action in sac_actions)

    return {
        "status": "ok",
        "seed": int(CONFIG["SEED"]),
        "checks": {
            "sat_collision": "passed",
            "discrete_9_table": list(DISCRETE_9_TABLE.shape),
            "discrete_43_table": list(DISCRETE_43_TABLE.shape),
            "ppo_forward_pass": "passed",
            "sac_forward_pass": "passed",
            "environment_step": "passed",
        },
        "environment": {
            "cars": env.num_cars,
            "state_dimension": env.state_dim,
            "reward_finite": True,
            "terminal_after_first_step": bool(all(dones)),
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, or inspect the self-contained parking RL lab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", type=str.upper, choices=["TRAIN", "TRANSFER", "INFERENCE", "PLAY"])
    parser.add_argument("--algorithm", type=str.upper, choices=["PPO", "SAC", "SOC"])
    parser.add_argument(
        "--action-mode",
        type=str.upper,
        choices=["DISCRETE_9", "DISCRETE_43", "CONTINUOUS", "CURRICULUM"],
    )
    parser.add_argument("--scenario", type=str.upper, choices=["SIMPLE", "WALLS", "CUSTOM", "RANDOM"])
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cars", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--headless", action="store_true", help="Disable replay and inference windows.")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast integration checks and emit JSON.")
    parser.add_argument("--describe-json", action="store_true", help="Print the resolved experiment contract as JSON.")
    return parser.parse_args(argv)


def apply_cli_overrides(args: argparse.Namespace) -> None:
    mapping = {
        "mode": "MODE",
        "algorithm": "ALGORITHM",
        "action_mode": "ACTION_MODE",
        "scenario": "SCENARIO",
        "episodes": "EPISODES",
        "seed": "SEED",
        "cars": "NUM_CARS",
        "output_dir": "OUTPUT_DIR",
    }
    for argument, config_key in mapping.items():
        value = getattr(args, argument)
        if value is not None:
            CONFIG[config_key] = value
    if args.headless:
        CONFIG["ASYNC_RENDER"] = False
        CONFIG["INFERENCE_RENDER"] = False
        plt.switch_backend("Agg")


# =============================================================================
# Validation and entry point
# =============================================================================
def validate_configuration() -> None:
    algorithm = canonical_algorithm(str(CONFIG["ALGORITHM"]))
    action_mode = canonical_action_mode(str(CONFIG["ACTION_MODE"]))
    mode = str(CONFIG["MODE"]).upper()
    if mode not in {"TRAIN", "TRANSFER", "INFERENCE", "PLAY"}:
        raise ValueError("MODE must be TRAIN, TRANSFER, INFERENCE, or PLAY.")
    if int(CONFIG["NUM_CARS"]) < 1:
        raise ValueError("NUM_CARS must be at least 1.")
    if action_mode == "CURRICULUM" and not CONFIG["CURRICULUM_PHASES"]:
        raise ValueError("CURRICULUM_PHASES cannot be empty when ACTION_MODE is CURRICULUM.")
    print(
        f"[CONFIG] mode={mode} algorithm={algorithm} action_mode={action_mode} "
        f"cars={CONFIG['NUM_CARS']} device={CONFIG['DEVICE']} scenario={CONFIG['SCENARIO']}"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    apply_cli_overrides(args)
    if args.describe_json:
        print(json.dumps(project_description(), indent=2, sort_keys=True))
        return
    if args.smoke_test:
        print(json.dumps(smoke_test_report(), indent=2, sort_keys=True))
        return

    validate_configuration()
    mode = str(CONFIG["MODE"]).upper()
    if mode == "TRAIN":
        train()
    elif mode == "TRANSFER":
        transfer()
    elif mode == "INFERENCE":
        inference()
    else:
        play()


if __name__ == "__main__":
    # Windows multiprocessing requires the main guard. This is already in place,
    # so asynchronous rendering works when launched as: python parking_rl.py
    mp.freeze_support()
    main()
