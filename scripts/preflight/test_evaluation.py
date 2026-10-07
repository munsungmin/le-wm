import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from scripts.eval.protocol import (
    pusht_success,
    sample_final_goal_states,
    sample_final_windows,
)


class FakeDataset:
    column_names = ["episode_idx", "step_idx", "state"]

    def __init__(self):
        self.data = {
            "episode_idx": np.repeat([10, 11, 12], [5, 7, 9]),
            "step_idx": np.concatenate([np.arange(5), np.arange(7), np.arange(9)]),
            "state": np.arange(21 * 7).reshape(21, 7),
        }

    def get_col_data(self, name):
        return self.data[name]

    def get_row_data(self, rows):
        return {name: values[rows] for name, values in self.data.items()}


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_path = Path(__file__).parents[2] / "config" / "eval" / "pusht.yaml"
        cfg = OmegaConf.load(config_path)
        cls.position_tolerance_px = cfg.eval.success.position_tolerance_px
        cls.angle_tolerance_deg = cfg.eval.success.angle_tolerance_deg

    def success(self, goal, current):
        return pusht_success(
            goal,
            current,
            self.position_tolerance_px,
            self.angle_tolerance_deg,
        )[0]

    def test_success_uses_only_block_pose_and_five_degrees(self):
        goal = np.zeros(7)
        current = np.zeros(7)
        current[:2] = 1000  # Agent location must not affect success.
        current[2:4] = [19, 0]
        current[4] = np.deg2rad(4.9)
        self.assertTrue(self.success(goal, current))

        current[4] = np.deg2rad(5.1)
        self.assertFalse(self.success(goal, current))
        current[4] = 0
        current[2:4] = [20.1, 0]
        self.assertFalse(self.success(goal, current))

        current[2:4] = [20.0, 0]
        current[4] = np.deg2rad(5.0)
        self.assertTrue(self.success(goal, current))

    def test_success_wraps_angles_at_two_pi(self):
        goal = np.zeros(7)
        current = np.zeros(7)
        current[4] = 2 * np.pi - np.deg2rad(1)
        self.assertTrue(self.success(goal, current))

    def test_final_window_starts_exactly_goal_offset_before_last_frame(self):
        dataset = FakeDataset()
        episodes, starts = sample_final_windows(dataset, 2, goal_offset=4, seed=7)
        lengths = {10: 5, 11: 7, 12: 9}
        self.assertTrue(
            all(
                start + 4 == lengths[episode] - 1
                for episode, start in zip(episodes, starts)
            )
        )

    def test_random_goals_are_final_demonstration_states(self):
        dataset = FakeDataset()
        goals = sample_final_goal_states(dataset, 3, seed=7)
        expected_rows = dataset.data["state"][[4, 11, 20]]
        self.assertEqual(
            {tuple(row) for row in goals}, {tuple(row) for row in expected_rows}
        )
