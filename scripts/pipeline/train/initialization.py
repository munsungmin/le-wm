"""One initializer shared by random, LEWM and FF-JEPA checkpoint configs."""

from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict
from stable_worldmodel.wm.utils import _resolve

from scripts.utils.runs import _is_primary


def initialize_model(cfg, model=None, path=None, module=(), freeze=(), resume=False, saved_model=None):
    """Prepare module configs, then load/freeze modules after model construction.

    Calling without a model restores saved dimensions before instantiate(). On
    resume those dimensions come from the run snapshot; weights are restored by
    Trainer.fit, without requiring the original pretrained file.
    """
    names = list(module)
    frozen = list(freeze) if freeze else [False] * len(names)
    if len(names) != len(set(names)) or len(names) != len(frozen):
        raise ValueError("init.module must be unique and match init.freeze in length.")
    if names and path is None and not resume:
        raise ValueError("init.path is required when selecting pretrained modules.")
    if model is None:
        if not names:
            return []
        if resume:
            if saved_model is None:
                raise ValueError("Resume initialization requires the saved model config.")
            saved = saved_model
        else:
            init_path = Path(path).expanduser().resolve()
            _, saved = _resolve(init_path.name, init_path.parent)
        with open_dict(cfg):
            for name in names:
                requested = OmegaConf.to_container(cfg.model[name], resolve=True)
                restored = dict(saved[name])
                restored.pop("_target_", None)
                cfg.model[name] = OmegaConf.merge(requested, restored)
        return []

    state = None
    if names and not resume:
        init_path = Path(path).expanduser().resolve()
        checkpoint, _ = _resolve(init_path.name, init_path.parent)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    frozen_modules = []
    for name, should_freeze in zip(names, frozen, strict=True):
        target = getattr(model, name)
        if state is not None:
            prefix = f"{name}."
            weights = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            target.load_state_dict(weights, strict=True)
        target.requires_grad_(not should_freeze)
        target.train(not should_freeze)
        if should_freeze:
            frozen_modules.append(name)
        if _is_primary():
            print(f"[init] {name}: {'frozen' if should_freeze else 'trainable'}", flush=True)
    return frozen_modules
