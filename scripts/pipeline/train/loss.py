import hydra
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class Loss(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fns = torch.nn.ModuleDict()
        self.inputs = {}
        self.weights = {}
        for (name, spec), weight in zip(cfg.term.items(), cfg.weights, strict=True):
            self.fns[name] = hydra.utils.instantiate(spec.fn)
            self.inputs[name] = dict(spec.inputs)
            self.weights[name] = float(weight)

    def forward(self, info):
        losses = {}
        total = None
        for name, fn in self.fns.items():
            kwargs = {
                arg: info.get(key)
                for arg, key in self.inputs[name].items()
            }
            if not all(torch.is_tensor(value) for value in kwargs.values()):
                continue
            if torch.is_grad_enabled() and not any(
                value.requires_grad for value in kwargs.values()
            ):
                continue
            value = fn(**kwargs)
            losses[f"{name}_loss"] = value
            weighted = self.weights[name] * value
            total = weighted if total is None else total + weighted

        if total is None:
            raise RuntimeError(
                "No applicable loss terms: required tensors are missing, "
                "or disconnected from gradients."
            )
        losses["loss"] = total
        return losses


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time

class MSE(torch.nn.Module):

    def __init__(self):
        pass 

    def forward(self, x , target): 
        return (x-target).pow(2).mean()
