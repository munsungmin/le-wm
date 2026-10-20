import torch 

def lejepa_forward(self, batch, stage, cfg):
    """Run the model's training step and log the configured objectives."""

    # Replace NaN values with 0 (occurs at sequence boundaries)
    if "action" in batch:
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    info = self.model.step(batch, cfg)
    output = self.loss(info)

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output
