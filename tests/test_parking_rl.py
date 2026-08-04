import copy
import json
import os

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

import parking_rl as parking


@pytest.fixture(autouse=True)
def restore_config():
    original = copy.deepcopy(parking.CONFIG)
    yield
    parking.CONFIG.clear()
    parking.CONFIG.update(original)


def test_action_tables_are_complete_and_bounded():
    assert parking.DISCRETE_9_TABLE.shape == (9, 2)
    assert parking.DISCRETE_43_TABLE.shape == (43, 2)
    assert np.all(np.abs(parking.DISCRETE_9_TABLE) <= 1.0)
    assert np.all(np.abs(parking.DISCRETE_43_TABLE) <= 1.0)
    np.testing.assert_array_equal(parking.DISCRETE_43_TABLE[-1], [0.0, 0.0])


def test_curriculum_boundaries():
    curriculum = parking.Curriculum("CURRICULUM", parking.CONFIG["CURRICULUM_PHASES"])
    assert curriculum.stage_for_episode(1).kind == "DISCRETE_9"
    assert curriculum.stage_for_episode(400).kind == "DISCRETE_9"
    assert curriculum.stage_for_episode(401).kind == "DISCRETE_43"
    assert curriculum.stage_for_episode(1000).kind == "DISCRETE_43"
    assert curriculum.stage_for_episode(1001).kind == "ANNEAL_43_TO_CONTINUOUS"
    assert curriculum.stage_for_episode(1601).kind == "CONTINUOUS"


def test_rotated_rectangle_collision_and_containment():
    car = parking.Entity(1, 0, 0.0, 0.0, 30.0, 4.0, 2.0, "CAR")
    overlap = parking.Entity(2, 0, 1.0, 0.0, -15.0, 4.0, 2.0, "CAR")
    far = parking.Entity(3, 0, 20.0, 20.0, 0.0, 4.0, 2.0, "CAR")
    lot = parking.Entity(100, 0, 0.0, 0.0, 30.0, 8.0, 5.0, "PARKING")

    assert parking.PhysicsUtils.check_collision(car, overlap)
    assert not parking.PhysicsUtils.check_collision(car, far)
    assert parking.PhysicsUtils.is_car_inside_lot(car, lot)


def test_seeded_reset_is_reproducible():
    first = parking.CarParkingEnvMulti(seed=123)
    second = parking.CarParkingEnvMulti(seed=123)

    first_states = first.reset()
    second_states = second.reset()

    assert first.state_dim == 16
    np.testing.assert_allclose(first_states, second_states)


def test_custom_scenario_schema(tmp_path):
    scenario = {
        "cars": [{"uid": 7, "x": -10, "y": 0, "target_id": 107}],
        "lots": [{"uid": 107, "x": 10, "y": 0}],
        "walls": [{"uid": 300, "x": 0, "y": 12, "length": 20, "width": 2}],
    }
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario), encoding="utf-8")

    cars, lots, walls = parking.load_custom_scenario(str(path))

    assert cars[0].target_id == lots[0].uid == 107
    assert walls[0].ent_type == "WALL"


def test_project_description_matches_resolved_state():
    description = parking.project_description()

    assert description["project"] == "Parking RL Lab"
    assert description["observation"]["dimensions"] == 16
    assert description["resolved_config"]["SEED"] == parking.CONFIG["SEED"]

