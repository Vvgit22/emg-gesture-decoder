"""Figures: turn the saved results into pictures.

What goes in: ``artifacts/results.json``, which ``evaluate.py`` built from all
the per-split run files. What comes out: SVG pictures in
``artifacts/figures/`` — a confusion-matrix heatmap for each split kind, a bar
chart comparing accuracies, and a strip of per-subject accuracies.

Why this step exists: a table of numbers is hard to feel. These pictures answer
"which gestures get confused?", "how much does the wrist ring lose?", and "is
every subject equally easy?" at a glance.

What would go wrong if skipped: nothing breaks, but you would have to eyeball a
JSON file instead of a picture.

Everything here only *reads* results — nothing is re-fitted or recomputed, so
the figures always match the audit trail.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen needed; deterministic SVG output
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "artifacts" / "results.json"
FIG_DIR = REPO / "artifacts" / "figures"

KINDS = ["within_subject", "cross_day", "cross_subject"]
CONFIGS = ["forearm", "wrist", "both"]
KIND_LABEL = {
    "within_subject": "within-subject",
    "cross_day": "cross-day",
    "cross_subject": "cross-subject",
}


def _load() -> dict:
    return json.loads(RESULTS.read_text())


def _confusion_fig(cell: dict, classes: list[str], title: str, out: Path) -> None:
    """Row-normalised confusion heatmap: each row says where that true
    gesture's windows were sent."""
    cm = np.asarray(cell["confusion"], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(np.nan_to_num(norm), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("predicted gesture")
    ax.set_ylabel("true gesture")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def _accuracy_bar_fig(results: dict, out: Path) -> None:
    """Grouped bars: mean accuracy for each split kind × channel config,
    with ±1 std error bars and a dashed line at chance (1/17 ≈ 5.9%)."""
    cells = results["cells"]["lda"]
    x = np.arange(len(KINDS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, cfg in enumerate(CONFIGS):
        means = [cells[cfg][k]["accuracy_mean"] * 100 for k in KINDS]
        stds = [cells[cfg][k]["accuracy_std"] * 100 for k in KINDS]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3,
               label=cfg)
    chance = results["table"][0]["chance_accuracy"] * 100
    ax.axhline(chance, ls="--", color="grey", lw=1,
               label=f"chance ({chance:.1f}%)")
    ax.set_xticks(x, [KIND_LABEL[k] for k in KINDS])
    ax.set_ylabel("accuracy (%)")
    ax.set_title("LDA accuracy by split kind and channel set")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def _per_subject_fig(results: dict, out: Path) -> None:
    """Strip plot: one dot per (subject, session) within-subject split for
    the 'both' config — 129 dots showing whether some people are harder
    to decode than others."""
    per_split = results["cells"]["lda"]["both"]["within_subject"]["per_split"]
    subjects = sorted({int(re.match(r"s(\d+)", n).group(1)) for n in per_split})
    xs, ys = [], []
    for name, v in per_split.items():
        s = int(re.match(r"s(\d+)", name).group(1))
        xs.append(subjects.index(s))
        ys.append(v["accuracy"] * 100)
    fig, ax = plt.subplots(figsize=(12, 4))
    rng = np.random.default_rng(0)  # fixed seed: deterministic jitter
    ax.scatter(np.asarray(xs) + rng.uniform(-0.25, 0.25, len(xs)), ys,
               s=14, alpha=0.7, edgecolors="none")
    ax.set_xticks(range(len(subjects)), [f"s{s:02d}" for s in subjects],
                  rotation=90, fontsize=6)
    ax.set_ylabel("accuracy (%)")
    ax.set_xlabel("subject")
    ax.set_title("Within-subject LDA accuracy per subject (both channels)")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = _load()
    classes = results["classes"]
    for kind in KINDS:
        cell = results["cells"]["lda"]["both"][kind]
        _confusion_fig(cell, classes,
                       f"LDA, all channels — {KIND_LABEL[kind]}",
                       FIG_DIR / f"confusion_lda_both_{kind}.svg")
    _accuracy_bar_fig(results, FIG_DIR / "accuracy_by_split_lda.svg")
    _per_subject_fig(results,
                     FIG_DIR / "within_subject_per_subject_lda.svg")
    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
