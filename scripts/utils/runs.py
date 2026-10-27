"""Experiment directories, config snapshots, and resume metadata."""

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from omegaconf import OmegaConf


def _timestamp():
    return os.environ.get("LEWM_RUN_TIMESTAMP") or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )


def _experiment_dir(root, run_id, resume_dir, resume=False):
    if resume and not resume_dir:
        raise ValueError("resume=true requires resume_dir=/absolute/run/path.")
    path = resume_dir if resume else Path(root) / run_id
    return str(Path(path).expanduser().resolve())


OmegaConf.register_new_resolver("utc_timestamp", _timestamp, use_cache=True)
OmegaConf.register_new_resolver("experiment_dir", _experiment_dir)


def _is_primary():
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def _git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=Path(__file__).resolve().parents[2], text=True
    ).strip()


def _config_hash(config):
    content = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _write_metadata(run_dir, metadata):
    path = run_dir / "metadata.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_run(cfg):
    """Create a run, or restore its saved config after validating the commit."""
    if cfg.resume and not cfg.resume_dir:
        raise ValueError("resume=true requires resume_dir=/absolute/run/path.")
    run_dir = Path(cfg.run_dir)
    ckpt_path = None
    if cfg.resume:
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        current_commit = _git("rev-parse", "HEAD")
        if metadata["git_commit"] != current_commit:
            raise ValueError(
                f"Resume Git commit mismatch: saved={metadata['git_commit']}, "
                f"current={current_commit}."
            )
        cfg = OmegaConf.load(run_dir / "config.yaml")
        if _config_hash(OmegaConf.to_container(cfg, resolve=True)) != metadata["config_hash"]:
            raise ValueError("Saved config.yaml does not match metadata.config_hash.")
        ckpt_path = run_dir / "checkpoints" / "last.ckpt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        cfg.resume = True
        cfg.resume_dir = str(run_dir)
        cfg.run_dir = str(run_dir)
        if _is_primary():
            resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
            metadata["config_hash"] = _config_hash(resolved)
            metadata["status"] = "running"
            _write_metadata(run_dir, metadata)
    else:
        resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        cfg = OmegaConf.create(resolved)
        if _is_primary():
            if (run_dir / "metadata.json").exists() or (run_dir / "config.yaml").exists():
                raise FileExistsError(f"Run already exists; use resume=true resume_dir={run_dir}")
            run_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "schema_version": 1,
                "run_id": cfg.run_id,
                "env": cfg.data.env,
                "model": cfg.output_model_name,
                "seed": cfg.seed,
                "git_commit": _git("rev-parse", "HEAD"),
                "git_dirty": bool(_git("status", "--porcelain")),
                "config_hash": _config_hash(resolved),
                "environment_id": cfg.data.environment_id,
                "environment_package_version": version(cfg.data.environment_package),
                "python_version": platform.python_version(),
                "torch_version": version("torch"),
                "lightning_version": version("lightning"),
                "started_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                "status": "running",
            }
            OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
            _write_metadata(run_dir, metadata)
    if _is_primary():
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "wandb").mkdir(exist_ok=True)
    os.environ["LEWM_RUN_TIMESTAMP"] = cfg.run_timestamp
    return cfg, run_dir, ckpt_path


def finish_run(run_dir, status):
    if _is_primary():
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        metadata["status"] = status
        _write_metadata(run_dir, metadata)
