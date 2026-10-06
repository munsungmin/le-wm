"""Thin command-line entry point for Hydra-managed PushT jobs."""

import hydra
from omegaconf import DictConfig

from scripts.runner import run, translate_config_flag


@hydra.main(version_base=None, config_path="config", config_name="run/pusht_train")
def main(cfg: DictConfig):
    return run(cfg)


if __name__ == "__main__":
    translate_config_flag()
    main()
