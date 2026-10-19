import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from scripts.lewm.model.model import JEPA
def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class ffJEPA(JEPA):
    def __init__(
            self,
            encoder,
            predictor,
            action_encoder,
            planner,
            loss=None,
            projector=None,
            pred_proj=None,
            plan_proj=None,

        ):
            super().__init__(
                encoder=encoder,
                predictor=predictor,
                action_encoder=action_encoder,
                projector=projector,
                pred_proj=pred_proj,
                loss=loss,
            )
            self.planner = planner
            self.plan_proj = plan_proj or nn.Identity() 
            self.emb_history = None
            self.len_history = planner.num_frames
            
    ####################
    ## train only ##
    ####################

    def step(self, batch, cfg):

        loss_info={}
        out = self.encode(batch, loss_info)
        self.planner.step(out["emb"], loss_info)
        output = self._compute_loss(loss_info)
        return output

    ####################
    ## Inference only ##
    ####################

    def plan(self, emb):

        preds = self.planner.sample(emb)
        preds = self.plan_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))

        return preds

    def update_history(self, info: dict, start_idx, end_idx):
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)

        if self.emb_history is None:
            self.emb_history = _init["emb"]

        elif self.emb_history.size(0) <= start_idx:
            self.emb_history = torch.concat((self.emb_history, _init["emb"]),dim=0)

        elif start_idx == 0:
            buffer = self.emb_history.new_zeros((self.emb_history.shape[0], 1, self.emb_history.shape[2]))
            self.emb_history = torch.concat((self.emb_history, buffer), dim=1)
            self.emb_history[start_idx:end_idx, -1:, :] = _init["emb"]

        else:
            self.emb_history[start_idx:end_idx, -1:, :] = _init["emb"]

    def _get_sub_goal(self, info, start_idx=0, end_idx=None):

        B, S = info['pixels'].shape[:2]
        context = self.emb_history[start_idx:end_idx, -self.len_history:, :]
        seq = self.plan(context)
        sub_goal = seq[:, -1:].unsqueeze(1).expand(B,S,-1,-1)
        return sub_goal
      
    def _criterion(self, info_dict: dict, start_idx=0, end_idx=None):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B, S, T-1, dim)
        goal_emb = self._get_sub_goal(info_dict, start_idx, end_idx) # (B, S, 1, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost
