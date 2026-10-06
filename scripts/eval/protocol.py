"""PushT evaluation protocol helpers, kept independent of the world model."""

from types import MethodType

import numpy as np


POSITION_TOLERANCE_PX = 20.0
ANGLE_TOLERANCE_DEG = 5.0


def pusht_success(goal_state, current_state):
    """Return the FF-JEPA PushT success decision and a diagnostic distance."""
    goal_state = np.asarray(goal_state)
    current_state = np.asarray(current_state)

    # PushT state layout: agent (x, y), block (x, y, angle), agent velocity.
    position_error = np.linalg.norm(goal_state[2:4] - current_state[2:4])
    angle_error = abs(float(goal_state[4] - current_state[4]))
    angle_error = min(angle_error, 2 * np.pi - angle_error)
    success = (
        position_error < POSITION_TOLERANCE_PX
        and angle_error < np.deg2rad(ANGLE_TOLERANCE_DEG)
    )
    return bool(success), float(np.linalg.norm(goal_state - current_state))


def configure_pusht_success(world):
    """Install the paper's success predicate on every PushT environment."""
    for wrapped_env in world.envs.envs:
        env = wrapped_env.unwrapped
        env.eval_state = MethodType(
            lambda self, goal, current: pusht_success(goal, current), env
        )


def episode_lengths(dataset, episode_ids):
    episode_column = (
        "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    )
    rows_episode = dataset.get_col_data(episode_column)
    rows_step = dataset.get_col_data("step_idx")
    all_ids, first_rows = np.unique(rows_episode, return_index=True)
    all_lengths = np.maximum.reduceat(rows_step, first_rows) + 1
    positions = np.searchsorted(all_ids, episode_ids)
    if np.any(all_ids[positions] != episode_ids):
        raise ValueError("Unknown episode id requested.")
    return all_lengths[positions]


def sample_final_windows(dataset, num_eval, goal_offset, seed):
    """Sample episodes whose start is exactly N steps before the final frame."""
    episode_column = (
        "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    )
    episode_ids = np.unique(dataset.get_col_data(episode_column))
    lengths = episode_lengths(dataset, episode_ids)
    eligible = np.flatnonzero(lengths > goal_offset)
    if len(eligible) < num_eval:
        raise ValueError(
            f"Need {num_eval} episodes longer than {goal_offset} steps; "
            f"found {len(eligible)}."
        )

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(eligible, size=num_eval, replace=False))
    return episode_ids[chosen], lengths[chosen] - goal_offset - 1


def sample_final_goal_states(dataset, num_eval, seed):
    """Sample final states from demonstrations for random-initialization goals."""
    episode_column = (
        "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    )
    rows_episode = dataset.get_col_data(episode_column)
    episode_ids, first_rows = np.unique(rows_episode, return_index=True)
    lengths = episode_lengths(dataset, episode_ids)
    if len(episode_ids) < num_eval:
        raise ValueError(f"Need {num_eval} goal episodes; found {len(episode_ids)}.")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(episode_ids), size=num_eval, replace=False)
    goal_rows = first_rows[chosen] + lengths[chosen] - 1
    return np.asarray(dataset.get_row_data(np.asarray(goal_rows))["state"])
