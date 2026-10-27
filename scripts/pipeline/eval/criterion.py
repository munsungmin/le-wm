"""Vectorized PushT success criteria for evaluation and episode selection."""

from functools import partial

import numpy as np
from hydra.utils import instantiate


def _state_array(state):
    state = np.asarray(state, dtype=np.float64)
    if state.ndim < 1 or state.shape[-1] < 5:
        raise ValueError("PushT state must contain agent xy, block xy and block angle.")
    return state


def ffjepa_success(state, goal_state=None, position_threshold=20.0, angle_threshold_deg=5.0,
                   goal_position=(256.0, 256.0), goal_angle=np.pi / 4):
    """FF-JEPA: block xy/orientation only, using the expert task's fixed goal.

    goal_state is accepted for the shared evaluation interface. The FF-JEPA task
    goal remains the configured block pose, independent of the agent's pose.
    """
    state = _state_array(state)
    distance = np.linalg.norm(state[..., 2:4] - goal_position, axis=-1)
    angle = np.abs((state[..., 4] - goal_angle + np.pi) % (2 * np.pi) - np.pi)
    return (np.isfinite(state[..., 2:5]).all(axis=-1)
            & (distance < position_threshold) & (angle < np.deg2rad(angle_threshold_deg)))


def lewm_success(state, goal_state=None, position_threshold=20.0, angle_threshold_deg=20.0):
    """Original PushT eval_state: joint agent/block xy error and block angle.

    A full goal state is required, as in LeWM's goal-conditioned evaluation.
    Agent velocity is excluded from success and remains part of reward distance.
    """
    state = _state_array(state)
    if goal_state is None:
        raise ValueError("LEWM success requires goal_state; set sampling goal_column or goal_state.")
    goal = _state_array(goal_state)
    distance = np.linalg.norm(state[..., :4] - goal[..., :4], axis=-1)
    angle = np.abs((state[..., 4] - goal[..., 4] + np.pi) % (2 * np.pi) - np.pi)
    finite = np.isfinite(state[..., :5]).all(axis=-1) & np.isfinite(goal[..., :5]).all(axis=-1)
    return finite & (distance < position_threshold) & (angle < np.deg2rad(angle_threshold_deg))


def _eval_state(goal_state, cur_state, criterion):
    success = bool(criterion(cur_state, goal_state=goal_state))
    return success, np.linalg.norm(np.asarray(goal_state) - np.asarray(cur_state))


def configure_criterion(world, config):
    """Use the selected criterion for PushT termination while preserving reward."""
    if config is not None:
        criterion = instantiate(config)
        for env in world.envs.envs:
            if not hasattr(env.unwrapped, "eval_state"):
                raise TypeError("The selected criterion requires an env.eval_state method.")
            env.unwrapped.eval_state = partial(_eval_state, criterion=criterion)
