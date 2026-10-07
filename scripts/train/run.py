import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from scripts.train.module import SIGReg
from scripts.train.utils import (
    JsonlMetricsCallback,
    SaveFinalWeightsCallback,
    get_column_normalizer,
    get_img_preprocessor,
)
from scripts.utils.records import make_run_id, prepare_run


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

def run(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    spt.set(
        default_callbacks={
            "sklearn_checkpoint": False,
            "wandb_checkpoint": False,
            "trackio_checkpoint": False,
            "swanlab_checkpoint": False,
            "hf_checkpoint": False,
        }
    )

    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or make_run_id(
        cfg.record.env, cfg.record.model, cfg.seed
    )
    with open_dict(cfg):
        cfg.subdir = run_id
    storage_root = Path(cfg.storage_root)
    checkpoints_root = storage_root / "checkpoints"
    run_dir, _ = prepare_run(cfg, run_id, checkpoints_root)

    logger = False
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    checkpoint_interval = int(cfg.checkpoint.every_n_epochs)
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint.every_n_epochs must be positive")
    if int(cfg.trainer.max_epochs) % checkpoint_interval:
        raise ValueError(
            "trainer.max_epochs must be divisible by checkpoint.every_n_epochs "
            "so lewm_retrain.ckpt contains the final epoch"
        )

    resume_checkpoint = run_dir / f"{cfg.checkpoint.final_name}.ckpt"
    lightning_checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        filename=cfg.checkpoint.final_name,
        every_n_epochs=checkpoint_interval,
        save_on_train_epoch_end=True,
        save_top_k=1,
        save_last=False,
        enable_version_counter=False,
    )
    final_weights_callback = SaveFinalWeightsCallback(
        run_name=run_id,
        cfg=cfg.model,
        cache_dir=storage_root,
        final_name=cfg.checkpoint.final_name,
    )
    metrics_callback = JsonlMetricsCallback(run_dir / "metrics.jsonl")

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            lightning_checkpoint,
            final_weights_callback,
            metrics_callback,
        ],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    # stable_pretraining.Module optimizes manually. Preserve Trainer-level
    # clipping for its training_step, while bypassing Lightning's automatic-
    # optimization-only validation.
    if trainer.gradient_clip_val is not None and trainer.gradient_clip_val > 0:
        trainer.gradient_clip_val_ = trainer.gradient_clip_val
        trainer.gradient_clip_algorithm_ = trainer.gradient_clip_algorithm
        trainer.gradient_clip_val = None

    trainer.fit(
        world_model,
        datamodule=data_module,
        ckpt_path=str(resume_checkpoint) if resume_checkpoint.exists() else None,
    )
    return
