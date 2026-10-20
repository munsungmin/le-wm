import json
import math
from pathlib import Path

from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class JsonLogger(Logger):
    """Append scalar metrics to a JSONL file on rank zero."""

    def __init__(
        self, save_dir: str | Path, filename: str = "metrics.jsonl", epoch_only: bool = False
    ):
        super().__init__()
        self._save_dir = Path(save_dir).expanduser().resolve()
        self._path = self._save_dir / filename
        self._epoch_only = epoch_only
        self._logged_epochs = set()
        if epoch_only and self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                self._logged_epochs.add((record.get("stage"), record.get("epoch")))

    @property
    def name(self):
        return ""

    @property
    def version(self):
        return ""

    @property
    def save_dir(self):
        return str(self._save_dir)

    @property
    def root_dir(self):
        return self.save_dir

    @property
    def log_dir(self):
        return self.save_dir

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        # The training config is saved separately; this logger records metrics.
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        record = {}
        for key, value in metrics.items():
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            record[key] = value
        record["step"] = step

        if self._epoch_only:
            stages = {"fit": "train", "validate": "validation", "test": "test"}
            for prefix, stage in stages.items():
                values = {k: v for k, v in record.items() if k.startswith(f"{prefix}/")}
                key = (stage, record.get("epoch"))
                if not values or key[1] is None or key in self._logged_epochs:
                    continue
                self._append(dict(values, stage=stage, epoch=key[1], step=step))
                self._logged_epochs.add(key)
            return

        self._append(record)

    def _append(self, record):
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
