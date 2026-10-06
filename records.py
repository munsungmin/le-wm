"""Small, dependency-light helpers for the team's experiment record contract."""

import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf


REQUIRED_RESULT_FIELDS = (
    "env",
    "model",
    "seed",
    "depth",
    "horizon",
    "intervention",
    "interference_k",
    "checkpoint_step",
)


def make_run_id(env, model, seed, now=None):
    now = now or datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    return f"{env}_{model}_{seed}_{timestamp}"


def config_hash(cfg):
    payload = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty():
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def prepare_run(cfg, run_id, root):
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    metadata = {
        "run_id": run_id,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "config_hash": config_hash(cfg),
        "seed": int(cfg.seed),
        "environment_version": f"stable-worldmodel=={package_version('stable-worldmodel')}",
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return run_dir, metadata


def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def validate_result_row(row):
    missing = [field for field in REQUIRED_RESULT_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Missing required result fields: {', '.join(missing)}")
