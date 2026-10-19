import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from scripts.lewm.model.model import JEPA
def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

from collections import deque

class ffJEPA_Det(JEPA):
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
            self.planner = planner
            self.plan_proj = plan_proj or nn.Identity() 
            self.emb_history = None
            self.len_history = torch.planner.context_size()
            
    ####################
    ## train only ##
    ####################
    
    def _plan(self, emb):
        
        preds = self.planner(emb)
        preds = self.plan_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds


    ####################
    ## Inference only ##
    ####################

    def update_history(self, info : dict , start_idx, end_idx):

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init) # (B, 1, D)
        #emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        #_init = {k: detach_clone(v) for k, v in _init.items()}
        
        if self.emb_history is None:
            self.emb_history = _init["emb"]

        elif self.emb_history.size(0) < start_idx:
            self.emb_history = torch.concat(self.emb_history, _init["emb"], dim=0)

        elif start_idx == 0 : 
            buffer = self.emb_history.new_zeros((self.emb_history.shape(0),1,self.emb_history.shape(2),self.emb_history.shape(3)))
            self.emb_history = torch.concat(self.emb_history, buffer)
            self.emb_history[start_idx:end_idx, -1:, :] = _init["emb"]

        else:
            self.emb_history[start_idx:end_idx, -1:, :] = _init["emb"]

    def _get_sub_goal(self, info):

        B, S = info['pixels'].shape[:2]
        context = self.emb_history[...,-self.len_history:,:] 
        seq = self._plan(context) # (B, 3, D)
        sub_goal = seq[...,-1:,:].unsqueeze(1).expand(B,S,-1,-1) # (B, S, 1 ,D)
        return sub_goal
      
    def _criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B, S, T-1, dim)
        goal_emb = self._get_sub_goal() # (B, S, 1, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost


class ffJEPA_DM(ffJEPA_Det):

    ####################
    ## Inference only ##
    ####################   


    def _get_sub_goal(self, info):

        B, S = info['pixel'].shape[:2]
        context = self.emb_history[...,-self.len_history:,] 
        
        noise = self.planner(context) # (B, 1, D)
        
        
        #sub_goal = seq[...,-1:,:].unsqueeze(1).expand(B,S,-1,-1) # (B, S, 1 ,D)
    