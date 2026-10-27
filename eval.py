import os

os.environ["MUJOCO_GL"] = "egl"

import json
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
from types import MethodType
import pytorch_lightning as pl 
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


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


def _json_default(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON")


def _write_json(path, data):
    path.write_text(
        json.dumps(data, default=_json_default, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def prepare_eval_output(cfg):
    """Use the checkpoint directory for both single runs and Hydra sweeps."""
    choices = HydraConfig.get().runtime.choices if HydraConfig.initialized() else {}
    pipeline = choices.get("pipeline", cfg.pipeline.name)
    output = cfg.output
    checkpoint_path = None
    if cfg.policy != "random":
        cache_dir = swm.data.utils.get_cache_dir(sub_folder="checkpoints")
        checkpoint_path = Path(cache_dir, cfg.policy).expanduser()
        if not checkpoint_path.is_file():
            if not checkpoint_path.is_dir():
                checkpoint_path = cache_dir / f"models--{cfg.policy.replace('/', '--')}"
            # load_pretrained already requires an unambiguous checkpoint folder.
            checkpoint_path, = checkpoint_path.glob("*.pt")
        checkpoint_path = checkpoint_path.resolve()

    checkpoint_dir = (
        checkpoint_path.parent if checkpoint_path is not None else Path(__file__).parent
    )
    seed_dir = checkpoint_dir / "evaluations" / pipeline / f"seed_{cfg.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(ZoneInfo("Asia/Seoul"))
    results_path = Path(tempfile.mkdtemp(
        prefix=started_at.strftime("%Y%m%dT%H%M%S%f%z") + "_", dir=seed_dir
    ))
    epoch = (
        re.search(r"(?:^|_)epoch_(\d+)(?:_|$)", checkpoint_path.stem)
        if checkpoint_path is not None else None
    )
    metadata = {
        "env": cfg.world_name,
        "model": output.get("model") or choices.get("policy", cfg.policy),
        "seed": int(cfg.seed),
        "depth": int(output.get("depth", 0)),
        # CEM lookahead in environment steps; independent of the goal distance.
        "horizon": int(cfg.plan_config.horizon * cfg.plan_config.action_block),
        "horizon_unit": "environment_steps",
        "intervention": output.get("intervention", "none"),
        "interference_k": int(output.get("interference_k", 0)),
        "checkpoint_step": checkpoint_path.name if checkpoint_path else None,
        "checkpoint_epoch": int(epoch.group(1)) if epoch else None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "pipeline": pipeline,
        "goal_offset": int(cfg.eval.goal_offset_steps),
        "budget": int(cfg.eval.eval_budget),
        "num_eval": int(cfg.eval.num_eval),
        "plan_horizon": int(cfg.plan_config.horizon),
        "action_block": int(cfg.plan_config.action_block),
        "solver": cfg.solver._target_ if checkpoint_path else None,
        "dataset": cfg.eval.dataset_name,
        "run_id": results_path.name,
        "started_at": started_at.isoformat(),
    }
    OmegaConf.save(cfg, results_path / "config.yaml", resolve=True)
    _write_json(results_path / "metadata.json", metadata)
    return results_path, metadata


def save_eval_results(results_path, metadata, metrics, elapsed, episodes, start_steps):
    record = {
        **metadata,
        "evaluation_time_seconds": elapsed,
        "episode_indices": episodes,
        "start_steps": start_steps,
        "metrics": metrics,
    }
    _write_json(results_path / "metrics.json", record)
    return record


@hydra.main(version_base=None, config_path="./config/eval", config_name="eval")
def run(cfg: DictConfig):
    
    pl.seed_everything(cfg.seed, workers=True)
    """Run evaluation of dinowm vs random policy."""

    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # add : pipeline 에 따른 eval_state 기준 변화 대응 
    eval_state = hydra.utils.instantiate(cfg.criterion)
    for wrapped_env in world.envs.envs:
        env = wrapped_env.unwrapped  # 실제 PushT 객체
        env.eval_state = MethodType(eval_state, env)

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

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

        # add : 현재 config 의 frameskip 과 predictor의 frameskip 및 action_dim 검사 
        env_action_dim = int(np.prod(world.envs.single_action_space.shape))
        action_block = cfg.plan_config.action_block
        encoder_action_dim = model.action_encoder.patch_embed.in_channels
        assert env_action_dim * action_block == encoder_action_dim, (
            f"Action dimension mismatch: environment action_dim={env_action_dim} "
            f"x plan_config.action_block={action_block} gives "
            f"{env_action_dim * action_block}, but the checkpoint's action encoder "
            f"expects {encoder_action_dim}. Check plan_config.action_block."
        )
        predictor_frameskip = getattr(model.predictor, "frameskip", None)
        assert predictor_frameskip is None or predictor_frameskip == action_block, (
            f"plan_config.action_block={action_block} does not match the "
            f"checkpoint's predictor.frameskip={predictor_frameskip}."
        )

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
        policy = swm.policy.RandomPolicy(seed=cfg.seed)

    results_path, run_metadata = prepare_eval_output(cfg)

    # add : pipeline 에 따른 episode 선정 기준 대응  
    sampler = hydra.utils.instantiate(cfg.sampler)
    eval_episodes, eval_start_idx = sampler(dataset, ep_indices)

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path / "videos",
    )
    end_time = time.time()
    
    print(metrics)

    record = save_eval_results(
        results_path, run_metadata, metrics, end_time - start_time,
        eval_episodes, eval_start_idx,
    )
    print(f"Evaluation results saved to {results_path / 'metrics.json'}")

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(json.dumps(record, default=_json_default, ensure_ascii=False, indent=2))
        f.write("\n")


if __name__ == "__main__":
    run()
