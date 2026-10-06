"""Unified Hydra entry point for LeWM PushT training and evaluation."""

import contextlib
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


CONFIG_ROOT = Path(__file__).resolve().parent / "config"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _translate_config_flag():
    """Accept the requested ``--config NAME`` alias for Hydra config names."""

    def config_name(value):
        value = value.removesuffix(".yaml")
        return value if value.startswith("run/") else f"run/{value}"

    for index, arg in enumerate(sys.argv):
        if arg == "--config":
            sys.argv[index] = "--config-name"
            if index + 1 < len(sys.argv):
                sys.argv[index + 1] = config_name(sys.argv[index + 1])
        elif arg.startswith("--config="):
            sys.argv[index] = "--config-name=" + config_name(arg.split("=", 1)[1])


def _run_train(cfg: DictConfig):
    from train import run

    return run(cfg)


def _run_preflight(cfg: DictConfig):
    if not cfg.preflight.enabled:
        print("preflight: skipped (preflight.enabled=false)")
        return

    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(cfg.preflight.test_dir),
        "-p",
        str(cfg.preflight.pattern),
        "-v" if int(cfg.preflight.verbosity) >= 2 else "-q",
    ]
    print("preflight:", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=HydraConfig.get().runtime.cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout, end="")
    if completed.returncode:
        raise RuntimeError(
            f"Preflight tests failed with exit code {completed.returncode}"
        )
    print("preflight: passed")


def _run_eval(cfg: DictConfig):
    from eval import run

    results = []
    for scenario_name in cfg.scenarios:
        scenario_path = CONFIG_ROOT / "eval" / "scenario" / f"{scenario_name}.yaml"
        if not scenario_path.is_file():
            raise FileNotFoundError(f"Unknown evaluation scenario: {scenario_name}")
        scenario = OmegaConf.load(scenario_path)
        job_cfg = OmegaConf.merge(cfg, {"eval": scenario})
        results.append(run(job_cfg))
    return results


def _print_dry_run(cfg: DictConfig):
    # Importing registers the same OmegaConf resolvers used by the real job,
    # but does not load data, a checkpoint, or allocate a CUDA model.
    if cfg.task == "train":
        import train  # noqa: F401
    else:
        import eval  # noqa: F401
    print(OmegaConf.to_yaml(cfg, resolve=True))


@hydra.main(version_base=None, config_path="config", config_name="run/pusht_train")
def main(cfg: DictConfig):
    if cfg.task not in {"train", "eval"}:
        raise ValueError(f"Unknown task: {cfg.task}")

    # Set GPU visibility before importing train/eval (and therefore torch).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "console.log"

    metadata = {
        "project": str(cfg.project),
        "task": str(cfg.task),
        "seed": int(cfg.seed),
        "gpu": int(cfg.gpu),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
    }
    OmegaConf.save(OmegaConf.create(metadata), run_dir / "run.yaml")

    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        stdout = _Tee(sys.__stdout__, log_file)
        stderr = _Tee(sys.__stderr__, log_file)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print(OmegaConf.to_yaml(OmegaConf.create(metadata), resolve=True))
            try:
                _run_preflight(cfg)
                if cfg.dry_run:
                    _print_dry_run(cfg)
                    return None
                if cfg.task == "train":
                    return _run_train(cfg)
                return _run_eval(cfg)
            except BaseException:
                traceback.print_exc()
                raise


if __name__ == "__main__":
    _translate_config_flag()
    main()
