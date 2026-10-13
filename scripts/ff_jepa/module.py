    
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn 
from scripts.lewm.module import Transformer, ARPredictor, Block
def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class Deterministic_planner(ARPredictor):
    
    def __init__(
            self,
            *,
            num_frames,
            depth,
            heads,
            mlp_dim,
            input_dim,
            hidden_dim,
            output_dim=None,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
        ):
            super().__init__(num_frames,depth,heads,mlp_dim,input_dim,hidden_dim,output_dim,dim_head,dropout,emb_dropout)
            self.transformer = Transformer(
                        input_dim,
                        hidden_dim,
                        output_dim or input_dim,
                        depth,
                        heads,
                        dim_head,
                        mlp_dim,
                        dropout,
                        block_class=Block,
                    )
    

    def forward(self, x): 
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x)
        pass 

class Diffusion_planner(nn.Module):
    def __init__(self):
        pass 

    def forward(self, x):
        pass 
