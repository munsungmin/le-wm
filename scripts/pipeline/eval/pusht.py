"""PushT evaluation protocols used by LeWM and FF-JEPA.

Samplers return episode IDs and starting steps without changing the dataset.
Criteria match ``PushT.eval_state`` and can be attached with ``MethodType``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _episode_columns(dataset, ep_indices):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episodes = np.asarray(dataset.get_col_data(col_name)).reshape(-1)
    steps = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    _, inverse = np.unique(episodes, return_inverse=True)
    last_steps = np.full(inverse.max(initial=-1) + 1, -1, dtype=np.int64)
    np.maximum.at(last_steps, inverse, steps)
    selected = np.isin(episodes, ep_indices)
    return episodes, steps, last_steps[inverse], selected


def _sample_rows(rows, num_eval, seed):
    if num_eval < 1:
        raise ValueError("num_eval must be positive.")
    if len(rows) < num_eval:
        raise ValueError(
            f"Requested {num_eval} evaluation starts, but only {len(rows)} "
            "satisfy the sampling protocol."
        )
    return np.sort(np.random.default_rng(seed).choice(rows, size=num_eval, replace=False))


def lewm_sample(
    dataset,
    ep_indices,
    *,
    goal_offset_steps: int,
    num_eval: int,
    seed: int,
    criterion: Callable | None = None,
):
    """Sample valid starting frames uniformly, without filtering for success.

    The goal is ``goal_offset_steps`` later in the same episode (LeWM, App. F.1).
    Different starts may belong to the same episode, as in the original evaluator.
    ``criterion`` keeps the interface shared with FF-JEPA; it does not restrict
    the LeWM candidate pool.
    """
    if goal_offset_steps < 1:
        raise ValueError("goal_offset_steps must be positive.")
    episodes, steps, last_steps, selected = _episode_columns(dataset, ep_indices)
    rows = np.flatnonzero(selected & (steps + goal_offset_steps <= last_steps))
    chosen = _sample_rows(rows, num_eval, seed)
    return episodes[chosen], steps[chosen]


def ffjepa_sample(
    dataset,
    ep_indices,
    *,
    goal_offset_steps: int,
    num_eval: int,
    seed: int,
    criterion: Callable | None = None,
):
    """Sample the final window of successful episodes (FF-JEPA, Sec. III-A).

    Short and long use this implementation with offsets of 25 and 75.
    Check raw terminal states using the configured criterion, leaving the
    dataset and the normalization statistics intact.
    """
    if goal_offset_steps < 1:
        raise ValueError("goal_offset_steps must be positive.")
    episodes, steps, last_steps, selected = _episode_columns(dataset, ep_indices)
    rows = np.flatnonzero(
        selected & (steps == last_steps) & (steps >= goal_offset_steps)
    )
    if "state" not in dataset.column_names:
        raise ValueError("FF-JEPA PushT sampling requires the raw 'state' column.")
    terminal_states = np.asarray(dataset.get_col_data("state"))[rows]
    criterion = criterion or ffjepa_criterion
    successful = np.array(
        [criterion(None, state, state)[0] for state in terminal_states], dtype=bool
    )
    chosen = _sample_rows(rows[successful], num_eval, seed)
    return episodes[chosen], steps[chosen] - goal_offset_steps


def _angle_distance(first, second):
    return abs((first - second + np.pi) % (2 * np.pi) - np.pi)


def lewm_criterion(self, goal_state, cur_state):
    """SWM PushT success: joint agent/block xy <20 px and angle <20 degrees."""
    goal_state = np.asarray(goal_state)
    cur_state = np.asarray(cur_state)
    pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])
    angle_diff = _angle_distance(goal_state[4], cur_state[4])
    success = pos_diff < 20 and angle_diff < np.pi / 9
    return bool(success), float(np.linalg.norm(goal_state - cur_state))


def ffjepa_criterion(
    self,
    goal_state,
    cur_state,
    *,
    goal_pose=(256.0, 256.0, np.pi / 4),
    position_threshold: float = 20.0,
    angle_threshold: float = np.pi / 36,
):
    """Check the block against the fixed PushT target, ignoring agent position.

    FF-JEPA Sec. III-A specifies 20-pixel and 5-degree tolerances. The target
    pose uses original 512x512 environment coordinates, not resized model pixels.
    Preserve SWM's distance/reward calculation.
    """
    goal_state = np.asarray(goal_state)
    cur_state = np.asarray(cur_state)
    goal_pose = np.asarray(goal_pose)
    pos_diff = np.linalg.norm(goal_pose[:2] - cur_state[2:4])
    angle_diff = _angle_distance(goal_pose[2], cur_state[4])
    success = pos_diff < position_threshold and angle_diff < angle_threshold
    return bool(success), float(np.linalg.norm(goal_state - cur_state))
