import os

os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

from scripts.eval.protocol import (
    configure_pusht_success,
    sample_final_goal_states,
    sample_final_windows,
)
from scripts.utils.records import (
    append_jsonl,
    make_run_id,
    prepare_run,
    validate_result_row,
)

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset

def run(cfg: DictConfig):
    """Evaluate a flat LeWM policy under an FF-JEPA PushT scenario."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    world = swm.World(**cfg.world, image_shape=(224, 224))
    configure_pusht_success(
        world,
        position_tolerance_px=cfg.eval.success.position_tolerance_px,
        angle_tolerance_deg=cfg.eval.success.angle_tolerance_deg,
    )

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    world.set_policy(policy)

    run_id = make_run_id(cfg.record.env, cfg.record.model, cfg.seed)
    run_dir, metadata = prepare_run(cfg, run_id, cfg.output.root)

    if cfg.eval.init_mode == "dataset_final":
        eval_episodes, eval_start_idx = sample_final_windows(
            dataset,
            num_eval=cfg.eval.num_eval,
            goal_offset=cfg.eval.goal_offset_steps,
            seed=cfg.seed,
        )
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
            video=run_dir if cfg.eval.save_video else None,
        )
    elif cfg.eval.init_mode == "random":
        goal_states = sample_final_goal_states(dataset, cfg.eval.num_eval, cfg.seed)
        options = [{"goal_state": goal_state} for goal_state in goal_states]
        metrics = world.evaluate(
            episodes=cfg.eval.num_eval,
            seed=cfg.seed,
            options=options,
            reset_mode="wait",
            video=run_dir if cfg.eval.save_video else None,
        )
    else:
        raise ValueError(f"Unknown eval.init_mode: {cfg.eval.init_mode}")

    metrics_serializable = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in metrics.items()
    }
    result_row = {
        "env": cfg.record.env,
        "model": cfg.record.model,
        "seed": int(cfg.seed),
        "depth": 0,
        "horizon": int(cfg.plan_config.horizon * cfg.plan_config.action_block),
        "intervention": "none",
        "interference_k": 0,
        "checkpoint_step": cfg.record.checkpoint_step,
        "goal_offset": cfg.eval.goal_offset_steps,
        "budget": int(cfg.eval.eval_budget),
        "scenario": cfg.eval.name,
        "run_id": run_id,
        "git_commit": metadata["git_commit"],
        "git_dirty": metadata["git_dirty"],
        "config_hash": metadata["config_hash"],
        "environment_version": metadata["environment_version"],
        **metrics_serializable,
    }
    validate_result_row(result_row)
    append_jsonl(run_dir / "results.jsonl", result_row)
    print(result_row)

    if "oracle" in world.infos:
        raise RuntimeError("PushT info must not contain the MiniGrid-only 'oracle' key.")

    return result_row
