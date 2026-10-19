    
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn 
from scripts.lewm.module.module import Transformer, ARPredictor
from scripts.lewm.module.utils.transformer import Block
from .diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class Deterministic_planner(ARPredictor):
    
    def __init__(
            self,
            *,
            num_frames,
            n_preds,
            depth,
            heads,
            mlp_dim,
            input_dim,
            hidden_dim,
            output_dim=None,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
            block_class=Block,
        ):
            super().__init__(
                num_frames=num_frames,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dim_head=dim_head,
                dropout=dropout,
                emb_dropout=emb_dropout,
            )
            self.transformer = Transformer(
                        input_dim,
                        hidden_dim,
                        output_dim or input_dim,
                        depth,
                        heads,
                        dim_head,
                        mlp_dim,
                        dropout,
                        block_class=block_class,
                    )
            self.num_frames=num_frames
            self.n_preds=n_preds
    
    def step(self, emb, info):
        if emb.size(1) < self.num_frames + self.n_preds:
            raise ValueError("Not enough embeddings for the planner context and targets.")
        ctx_emb = emb[:,:self.num_frames]
        tgt_emb = emb[:,self.n_preds:self.n_preds + self.num_frames]
        info["planner.pred_emb"] = self._forward(ctx_emb)
        info["planner.target_emb"] = tgt_emb
        return info

    def _forward(self, x):
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x)
        return x 

    def sample(self, x):
        return self._forward(x)

class _DiffusionBlock(Block):
    def forward(self, x):
        x = x + self.attn(self.norm1(x), causal=False)
        return x + self.mlp(self.norm2(x))


class DM_planner(Deterministic_planner):
    """Conditional epsilon prediction over a future embedding sequence."""

    def __init__(
        self,
        *,
        diffusion_steps=1000,
        beta_schedule="squaredcos_cap_v2",
        **kwargs,
    ):
        super().__init__(block_class=_DiffusionBlock, **kwargs)
        input_dim = kwargs["input_dim"]
        if kwargs.get("output_dim") not in (None, input_dim):
            raise ValueError("Diffusion output_dim must match the embedding input_dim.")
        if diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least 2.")

        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1 + self.num_frames + self.n_preds, input_dim)
        )
        self.timestep_embedding = nn.Embedding(diffusion_steps, input_dim)
        self.diffusion = GaussianDiffusion(
            betas=get_named_beta_schedule(beta_schedule, diffusion_steps),
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
        )

    def step(self, emb, info):
        if emb.size(1) < self.num_frames + self.n_preds:
            raise ValueError("Not enough embeddings for the planner context and targets.")
        context = emb[:, :self.num_frames]
        target = emb[:, self.num_frames:self.num_frames + self.n_preds]
        noise = torch.randn_like(target)
        timesteps = torch.randint(
            self.diffusion.num_timesteps, (emb.size(0),), device=emb.device
        )
        noisy_target = self.diffusion.q_sample(target, timesteps, noise=noise)
        info["planner.pred_noise"] = self._forward(noisy_target, timesteps, context)
        info["planner.target_noise"] = noise
        return info

    def _forward(self, x, timesteps, context):
        """Predict noise (B, H, D) from noisy future, timestep, and context."""
        timesteps = torch.as_tensor(timesteps, device=x.device, dtype=torch.long)
        timesteps = timesteps.reshape(-1).expand(x.size(0))
        time_token = self.timestep_embedding(timesteps).unsqueeze(1)
        tokens = torch.cat((time_token, context, x.to(context.dtype)), dim=1)
        tokens = self.dropout(tokens + self.pos_embedding[:, :tokens.size(1)])
        return self.transformer(tokens)[:, -self.n_preds:]

    @torch.no_grad()
    def sample(self, x, noise=None):
        """Denoise Gaussian noise into future embeddings (B, H, D)."""
        context = x[:, -self.num_frames:]
        if context.size(1) == 0:
            raise ValueError("At least one context embedding is required.")
        if context.size(1) < self.num_frames:
            padding = context[:, :1].expand(-1, self.num_frames - context.size(1), -1)
            context = torch.cat((padding, context), dim=1)
        shape = (x.size(0), self.n_preds, x.size(-1))
        if noise is None:
            noise = torch.randn(shape, device=x.device, dtype=x.dtype)
        elif noise.shape != shape:
            raise ValueError(f"Expected initial noise shape {shape}, got {tuple(noise.shape)}.")
        return self.diffusion.p_sample_loop(
            self._forward,
            shape,
            noise=noise,
            model_kwargs={"context": context},
            device=x.device,
            clip_denoised=False,
        )
