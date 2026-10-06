#!/usr/bin/env python3
"""Summarize completed checkpoint-evaluation JSONL rows by scenario."""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


root = Path(
    sys.argv[1] if len(sys.argv) > 1 else "/data/sungmin/lewm/runs/pusht_4gpu_latest/results"
).resolve()
rows = []
for path in sorted(root.glob("*/results.jsonl")):
    rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)

if not rows:
    raise SystemExit(f"No completed result rows under {root}")

groups = defaultdict(list)
for row in rows:
    groups[(row["model"], row["scenario"])].append(row)

for (model, scenario), items in sorted(groups.items()):
    rates = [float(item["success_rate"]) for item in items]
    seeds = [item["seed"] for item in items]
    mean = statistics.fmean(rates)
    std = statistics.stdev(rates) if len(rates) > 1 else 0.0
    print(
        f"model={model} scenario={scenario} seeds={seeds} "
        f"rates={rates} mean={mean:.4f} std={std:.4f}"
    )

