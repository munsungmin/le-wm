# LeWorldModel PushT

Minimal PushT training and FF-JEPA flat-baseline evaluation built from the
[LeWorldModel](https://github.com/lucas-maes/le-wm) implementation.

## Setup

The configured environment uses Python 3.10 with `stable-worldmodel[train,env]`.
The local data/checkpoint root is `/data/sungmin/lewm`.

Required artifacts:

- Dataset: `/data/sungmin/lewm/datasets/pusht_expert_train.h5`
- Public checkpoint: `/data/sungmin/lewm/checkpoints/pusht/lewm/`

## Commands

Hydra manages GPU selection, seeds, scenarios, project names, preflight tests,
and timestamped output directories.

```text
python main.py --config pusht_train gpu=0 seed=3072
python main.py --config pusht_train gpu=1 seed=3073
python main.py --config pusht_eval gpu=2 seed=42
python main.py --config pusht_eval gpu=3 seed=43
```

`pusht_eval` runs short and then long evaluation, each over 256 episodes.
Every command first runs the preflight regression tests. A failure stops the
job before model or dataset loading. To inspect the resolved setup without
starting a job:

```text
python main.py --config pusht_train gpu=0 seed=3072 dry_run=true
python main.py --config pusht_eval gpu=2 seed=42 dry_run=true
```

## Outputs

Each command writes to:

```text
/data/sungmin/lewm/runs/<project>/<timestamp>_<task>_seed<seed>/
```

The directory contains `console.log`, `run.yaml`, and Hydra's resolved config
and override files. Evaluation also writes `results/<run_id>/results.jsonl`,
`config.yaml`, and `metadata.json`.

## Source layout

```text
main.py                    CLI entry point
scripts/runner.py          Hydra orchestration, GPU isolation, logging
scripts/train/             LeWM model, training loop, callbacks
scripts/eval/              PushT loop and evaluation protocol
scripts/preflight/         regression tests run before every job
scripts/utils/             shared result and metadata helpers
config/run/                top-level train/eval job configs
config/train/              PushT LeWM training config
config/eval/               PushT short/long evaluation config
config/preflight/          preflight configuration
```

Only the PushT paths used by these commands are kept in this branch.

## Reference

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```
