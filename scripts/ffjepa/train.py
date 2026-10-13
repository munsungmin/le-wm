import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback

def ffjepa_forward(self, batch, stage, cfg): 

    predictor_cfg = OmegaConf.to_container(cfg.predictor, resolve=True)
    for name, predictor_sub_cfg in predictor_cfg.items():
        if "ff" in name:  
            ctx_len = cfg.predictor.ctx_len
            n_preds = cfg.predictor.n_preds

            img_batch = batch['pixel']
            output = self.model.encode(img_batch)

            emb = output["emb"]

            ctx_emb = emb[:, :ctx_len]

            tgt_emb = emb[:, ctx_len:]
            pred_emb = self.model.plan(emb, )

            if "det" in name: 
                output["ffjepa_det_pred_loss"] = (pred_emb - tgt_emb).pow(2).mean() 

            if "DM" in name:
               # output["ffjepa_DM_pred_loss"] = 
               pass 

            losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
            self.log_dict(losses_dict, on_step=True, sync_dist=True)
            return output