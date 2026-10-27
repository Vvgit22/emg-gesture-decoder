"""Collecting every training run into one results file.

What goes in
    `artifacts/runs/<model>/<config>/<kind>/<name>.json`, one file per
    training run, written by the training scripts. Each holds the run's
    confusion matrix plus bookkeeping (sizes, seconds, parameters). The
    schema is `RunRecord` below; `save_run` is the only writer.

What comes out
    `artifacts/results.json`: for every (model, channel config, split kind)
    the mean and spread of accuracy and macro-F1 across its splits, the
    per-class F1, the summed confusion matrix, and the per-split scores.
    The website reads this file and nothing else.

Why it is separate from training
    Training is slow and happens in many pieces, sometimes days apart. If
    the summary were produced inside the training script, re-running one
    fold would mean re-running everything to get a consistent table. Here
    the table is rebuilt from the run files in a second, and the run files
    are the audit trail.

What would go wrong without it
    Numbers would be copied by hand from logs into a web page, and a stale
    or mistyped number would be indistinguishable from a real one.

CLI: `uv run python -m emgdec.evaluate [--runs DIR] [--out FILE]`
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from emgdec import metrics as M
from emgdec.config import GESTURE_NAMES, N_GESTURES

REPO = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO / "artifacts" / "runs"
RESULTS = REPO / "artifacts" / "results.json"

MODELS = ("lda", "cnn", "cnn_w128", "cnn_w64")
CONFIGS = ("forearm", "wrist", "both")
KINDS = ("within_subject", "cross_day", "cross_subject")


@dataclass
class RunRecord:
    """Everything one training run leaves behind. Labels in `confusion` are 0-based (gesture-1)."""

    model: str
    config: str
    kind: str
    name: str
    confusion: list[list[int]]
    n_train: int
    n_val: int
    n_test: int
    train_seconds: float
    seed: int
    params: dict = field(default_factory=dict)
    val_accuracy: float | None = None
    epochs_run: int | None = None

    def path(self, runs_dir: Path = RUNS_DIR) -> Path:
        return runs_dir / self.model / self.config / self.kind / f"{self.name}.json"


def save_run(rec: RunRecord, runs_dir: Path = RUNS_DIR) -> Path:
    cm = np.asarray(rec.confusion)
    if cm.shape != (N_GESTURES, N_GESTURES):
        raise ValueError(f"confusion must be {N_GESTURES}x{N_GESTURES}, got {cm.shape}")
    if cm.sum() != rec.n_test:
        raise ValueError(f"confusion total {cm.sum()} != n_test {rec.n_test}")
    if rec.model not in MODELS or rec.config not in CONFIGS or rec.kind not in KINDS:
        raise ValueError(f"unknown model/config/kind: {rec.model}/{rec.config}/{rec.kind}")
    p = rec.path(runs_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(rec), indent=1))
    return p


def load_runs(runs_dir: Path = RUNS_DIR) -> list[RunRecord]:
    recs = []
    for p in sorted(runs_dir.rglob("*.json")):
        d = json.loads(p.read_text())
        recs.append(RunRecord(**d))
    return recs


def _summarise(group: list[RunRecord]) -> dict:
    accs = np.array([M.accuracy(r.confusion) for r in group])
    f1s = np.array([M.macro_f1(r.confusion) for r in group])
    total_cm = np.sum([np.asarray(r.confusion) for r in group], axis=0)
    return {
        "n_splits": len(group),
        "n_test_windows": int(total_cm.sum()),
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        "accuracy_min": float(accs.min()),
        "accuracy_max": float(accs.max()),
        "macro_f1_mean": float(f1s.mean()),
        "macro_f1_std": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
        "chance_accuracy": M.chance_accuracy(total_cm),
        "per_class_f1": [None if np.isnan(v) else float(v) for v in M.per_class_f1(total_cm)],
        "confusion": total_cm.tolist(),
        "per_split": {r.name: {"accuracy": M.accuracy(r.confusion), "macro_f1": M.macro_f1(r.confusion)} for r in group},
        "train_seconds_total": float(sum(r.train_seconds for r in group)),
        "params": group[0].params,
    }


def build(runs_dir: Path = RUNS_DIR) -> dict:
    recs = load_runs(runs_dir)
    results: dict = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": platform.platform(),
        "classes": GESTURE_NAMES,
        "n_runs": len(recs),
        "table": [],
        "cells": {},
    }
    for model in MODELS:
        for config in CONFIGS:
            for kind in KINDS:
                group = [r for r in recs if (r.model, r.config, r.kind) == (model, config, kind)]
                if not group:
                    continue
                cell = _summarise(group)
                results["cells"].setdefault(model, {}).setdefault(config, {})[kind] = cell
                results["table"].append(
                    {
                        "model": model,
                        "config": config,
                        "kind": kind,
                        "n_splits": cell["n_splits"],
                        "accuracy_mean": cell["accuracy_mean"],
                        "accuracy_std": cell["accuracy_std"],
                        "macro_f1_mean": cell["macro_f1_mean"],
                        "macro_f1_std": cell["macro_f1_std"],
                        "chance_accuracy": cell["chance_accuracy"],
                    }
                )
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=RUNS_DIR)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    res = build(a.runs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"{res['n_runs']} runs -> {a.out}")
    for row in res["table"]:
        print(
            f"{row['model']:>4} {row['config']:>8} {row['kind']:>15}  "
            f"acc {100*row['accuracy_mean']:5.1f} ± {100*row['accuracy_std']:4.1f}  "
            f"F1 {100*row['macro_f1_mean']:5.1f}  (n={row['n_splits']}, chance {100*row['chance_accuracy']:.1f})"
        )


if __name__ == "__main__":
    main()
