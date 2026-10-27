"""Sequential clip selection shared by training and evaluation."""

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from scripts.utils.runs import _is_primary


def select_episodes(dataset, criterion, state_column="state", goal_column=None, goal_state=None):
    """Keep every existing clip of episodes whose terminal state passes criterion.

    Episode IDs, offsets and raw columns stay intact. A later selector receives
    only the clips retained by its predecessors, so selection is an intersection.
    """
    lengths = np.asarray(dataset.lengths)
    offsets = np.asarray(dataset.offsets)
    episodes = np.unique([ep for ep, _ in dataset.clip_indices]).astype(int)
    episodes = episodes[lengths[episodes] > 0]
    if not len(episodes):
        dataset.clip_indices = []
        return dataset
    terminal_rows = offsets[episodes] + lengths[episodes] - 1
    states = np.asarray(dataset.get_col_data(state_column))[terminal_rows]
    if goal_column is not None:
        if goal_state is not None:
            raise ValueError("Choose either goal_column or goal_state.")
        goal_state = np.asarray(dataset.get_col_data(goal_column))[terminal_rows]
    accepted = np.asarray(criterion(states, goal_state=goal_state))
    if accepted.shape != (len(episodes),) or accepted.dtype != np.bool_:
        raise ValueError("Episode criterion must return one boolean per terminal state.")
    retained = set(episodes[accepted].tolist())
    dataset.clip_indices = [(ep, start) for ep, start in dataset.clip_indices if ep in retained]
    return dataset


def select_episode_length(dataset, min_steps=1, max_steps=None):
    """Optional independent selector, measured in raw environment timesteps."""
    if min_steps < 0 or (max_steps is not None and max_steps < min_steps):
        raise ValueError("Episode length bounds must satisfy 0 <= min_steps <= max_steps.")
    lengths = np.asarray(dataset.lengths)
    dataset.clip_indices = [
        (ep, start) for ep, start in dataset.clip_indices
        if lengths[ep] >= min_steps and (max_steps is None or lengths[ep] <= max_steps)
    ]
    return dataset


def select_terminal_window(dataset, goal_offset_steps):
    """Keep the start exactly goal_offset_steps before each episode's last state.

    Indices are raw environment timesteps: a 25-step window needs 26 states.
    Only existing clips are retained, preserving preceding selector results.
    """
    if goal_offset_steps < 0:
        raise ValueError("goal_offset_steps must be non-negative.")
    lengths = np.asarray(dataset.lengths)
    dataset.clip_indices = [
        (ep, start) for ep, start in dataset.clip_indices
        if lengths[ep] > goal_offset_steps
        and start == lengths[ep] - 1 - goal_offset_steps
    ]
    return dataset


def preprocess_dataset(dataset, config):
    """Apply the configured selector list in order; an empty list is a no-op."""
    sampling = config.get("sampling", [])
    if not OmegaConf.is_list(sampling) and not isinstance(sampling, list):
        raise TypeError("preprocessing.sampling must be a list of callable configs.")
    for index, spec in enumerate(sampling):
        before = set(dataset.clip_indices)
        selector = instantiate(spec)
        dataset = selector(dataset)
        after = list(dataset.clip_indices)
        if len(after) != len(set(after)) or not set(after).issubset(before):
            raise ValueError("Sampling modules may only remove existing clips.")
        if _is_primary():
            print(f"[sampling {index + 1}] {spec['_target_']}: {len(after)}/{len(before)} clips retained", flush=True)
        if not after:
            raise ValueError(f"Sampling module {index + 1} retained no full training clips.")
    return dataset
