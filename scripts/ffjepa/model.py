import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from scripts.lewm.model import JEPA
def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class ffJEPA(JEPA):
    def __init__(
            self,
            encoder,
            predictor,
            action_encoder,
            planner,
            projector=None,
            pred_proj=None,
            plan_proj=None,

        ):
            super().__init__(encoder, predictor, action_encoder, projector, pred_proj)
            self.encoder = encoder
            self.predictor = predictor
            self.action_encoder = action_encoder
            self.planner = planner
            self.projector = projector or nn.Identity()
            self.pred_proj = pred_proj or nn.Identity()
            self.plan_proj = plan_proj or nn.Identity()
            

    def plan(self, emb):
        preds = self.planner(emb)
        preds = self.plan_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds