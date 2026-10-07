from pathlib import Path

import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

from scripts.utils.records import append_jsonl

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveFinalWeightsCallback(Callback):
    """Export one eval-ready state dict after Lightning training completes."""

    def __init__(
        self,
        run_name,
        cfg,
        cache_dir,
        final_name="lewm_retrain",
    ):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.cache_dir = Path(cache_dir)
        self.final_name = final_name

    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        from stable_worldmodel.wm.utils import save_pretrained

        final_filename = f"{self.final_name}.pt"
        save_pretrained(
            pl_module.model,
            run_name=self.run_name,
            config=self.cfg,
            filename=final_filename,
            cache_dir=str(self.cache_dir),
        )


class JsonlMetricsCallback(Callback):
    """Append scalar training metrics once per epoch."""

    def __init__(self, path):
        super().__init__()
        self.path = path

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        row = {"epoch": trainer.current_epoch + 1, "step": trainer.global_step}
        for name, value in trainer.callback_metrics.items():
            if torch.is_tensor(value) and value.numel() == 1:
                row[name] = value.detach().cpu().item()
            elif isinstance(value, (int, float)):
                row[name] = value
        append_jsonl(self.path, row)
