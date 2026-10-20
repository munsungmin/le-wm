import os
from functools import partial

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf, open_dict

from scripts.lewm.train.loss import Loss
from scripts.utils.callbacks import SaveCkptCallback
from scripts.utils.loggers import JsonLogger
from scripts.utils.runs import finish_run, prepare_run
from scripts.utils.utils import get_column_normalizer, get_img_preprocessor


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    cfg, run_dir, ckpt_path = prepare_run(cfg)
    try:
        trainer = train(cfg, run_dir, ckpt_path)
    except BaseException as exc:
        finish_run(run_dir, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
        raise
    else:
        finish_run(run_dir, "interrupted" if trainer.interrupted else "completed")
    finally:
        os.environ.pop("LEWM_RUN_TIMESTAMP", None)


def train(cfg, run_dir, ckpt_path):
    pl.seed_everything(cfg.seed, workers=True)

    ##############################
    ##          model           ##
    ##############################
    world_model = hydra.utils.instantiate(cfg.model)
    load_cfg = OmegaConf.to_container(cfg.init, resolve=True) if ckpt_path is None else None
    if load_cfg:
        pretrained = swm.wm.utils.load_pretrained(
            load_cfg["path"], cache_dir=load_cfg["cache_dir"]
        )
        for name in load_cfg["module"]:
            getattr(world_model, name).load_state_dict(
                getattr(pretrained, name).state_dict()
            )
        del pretrained

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


    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }
    forward_func = hydra.utils.instantiate(cfg.forward)
    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        loss=Loss(cfg.loss),
        forward=partial(forward_func, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    loggers = [JsonLogger(save_dir=run_dir, epoch_only=True)]
    if cfg.wandb.enabled:
        wandb_config = OmegaConf.to_container(cfg.wandb.config, resolve=True)
        wandb_config.update(name=cfg.run_id, id=cfg.run_id, save_dir=str(run_dir))
        loggers.append(WandbLogger(**wandb_config))
    for logger in loggers:
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    object_dump_callback = SaveCkptCallback(
        run_name="", cfg=cfg.model, epoch_interval=1, cache_dir=str(run_dir),
    )
    resume_callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="last",
        save_top_k=1,
        save_last=False,
        save_weights_only=False,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        enable_version_counter=False,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        default_root_dir=str(run_dir),
        callbacks=[object_dump_callback, resume_callback],
        num_sanity_val_steps=1,
        logger=loggers,
        enable_checkpointing=True,
    )

    trainer.fit(world_model, datamodule=data_module, ckpt_path=ckpt_path)
    return trainer


if __name__ == "__main__":
    run()
