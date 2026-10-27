"""Training samplers selected as one callable in the model's data config."""

import numpy as np

from scripts.utils.runs import _is_primary


def lewm_sample(dataset):
    """Use every clip constructed by the dataset loader."""
    return dataset


def ffjepa_sample(
    dataset,
    *,
    criterion,
    state_column="state",
    goal_column=None,
    goal_state=None,
):
    """Keep all existing clips of episodes whose terminal state succeeds.

    Only clip_indices are filtered; raw rows, episode metadata and the data
    used for normalization stay intact. Clip length/stride come from the loader.
    """
    before = len(dataset.clip_indices)
    lengths = np.asarray(dataset.lengths)
    offsets = np.asarray(dataset.offsets)
    episodes = np.unique([ep for ep, _ in dataset.clip_indices]).astype(int)
    episodes = episodes[lengths[episodes] > 0]
    if not len(episodes):
        dataset.clip_indices = []
    else:
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
        dataset.clip_indices = [
            (ep, start) for ep, start in dataset.clip_indices if ep in retained
        ]

    if len(dataset.clip_indices) != len(set(dataset.clip_indices)):
        raise ValueError("Sampling must not duplicate existing clips.")
    if _is_primary():
        print(f"[sampling] ffjepa_sample: {len(dataset.clip_indices)}/{before} clips retained", flush=True)
    if not dataset.clip_indices:
        raise ValueError("FF-JEPA sampling retained no full training clips.")
    return dataset
