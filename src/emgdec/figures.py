"""Figures: turn the saved results into pictures.

What goes in: ``artifacts/results.json``, which ``evaluate.py`` built from all
the per-split run files. What comes out: SVG pictures in
``artifacts/figures/`` — confusion-matrix heatmaps, an accuracy bar chart
comparing models and channel sets, a per-subject strip, a per-class F1 chart,
and the window-length ablation curve.

Why this step exists: a table of numbers is hard to feel. These pictures answer
"which gestures get confused?", "how much does the wrist ring lose?", "does the
neural net beat LDA?", and "does a shorter window cost accuracy?" at a glance.

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
MODELS = ["lda", "cnn", "cnn_w128", "cnn_w64"]
KIND_LABEL = {
    "within_subject": "within-subject",
    "cross_day": "cross-day",
    "cross_subject": "cross-subject",
}
MODEL_COLOR = {"lda": "#8c8c8c", "cnn": "#1f77b4",
               "cnn_w128": "#7fb8dd", "cnn_w64": "#b9d7ee"}


def _load() -> dict:
    return json.loads(RESULTS.read_text())


def _cells(results: dict) -> dict:
    return results.get("cells", {})


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
    """Grouped bars: mean accuracy for each model × channel config per
    split kind, with ±1 std error bars and a dashed chance line."""
    cells = _cells(results)
    combos = [(m, c) for m in MODELS for c in CONFIGS
              if any(k in cells.get(m, {}).get(c, {}) for k in KINDS)]
    x = np.arange(len(KINDS))
    width = min(0.8 / max(len(combos), 1), 0.22)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (m, c) in enumerate(combos):
        offs = x + (i - (len(combos) - 1) / 2) * width
        for xi, k in zip(offs, KINDS):
            cell = cells.get(m, {}).get(c, {}).get(k)
            if cell is None:
                continue
            ax.bar(xi, cell["accuracy_mean"] * 100, width * 0.9,
                   yerr=cell["accuracy_std"] * 100, capsize=2,
                   color=MODEL_COLOR.get(m), alpha=0.55 + 0.45
                   * (CONFIGS.index(c) / 2), edgecolor="none",
                   label=f"{m} {c}" if xi == offs[0] else None)
    chance = results["table"][0]["chance_accuracy"] * 100
    ax.axhline(chance, ls="--", color="grey", lw=1,
               label=f"chance ({chance:.1f}%)")
    ax.set_xticks(x, [KIND_LABEL[k] for k in KINDS])
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Accuracy by split kind, channel set and model")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def _per_subject_fig(results: dict, out: Path) -> None:
    """Strip plot: one dot per (subject, session) within-subject split for
    the 'both' config — whether some people are harder to decode, and how
    the CNN compares to LDA on the same splits."""
    cells = _cells(results)
    rng = np.random.default_rng(0)  # fixed seed: deterministic jitter
    fig, ax = plt.subplots(figsize=(12, 4))
    handles = []
    for mi, model in enumerate(MODELS):
        cell = cells.get(model, {}).get("both", {}).get("within_subject")
        if cell is None:
            continue
        per_split = cell["per_split"]
        subjects = sorted({int(re.match(r"s(\d+)", n).group(1))
                           for n in per_split})
        xs = np.array([subjects.index(int(re.match(r"s(\d+)", n).group(1)))
                       for n in per_split], dtype=float)
        xs += (mi - 0.5) * 0.3 + rng.uniform(-0.1, 0.1, len(xs))
        ys = [v["accuracy"] * 100 for v in per_split.values()]
        handles.append(ax.scatter(xs, ys, s=14, alpha=0.7,
                                  color=MODEL_COLOR.get(model),
                                  edgecolors="none", label=model))
    ax.set_xticks(range(43), [f"s{s:02d}" for s in range(1, 44)],
                  rotation=90, fontsize=6)
    ax.set_ylabel("accuracy (%)")
    ax.set_xlabel("subject")
    ax.set_title("Within-subject accuracy per subject (both channels)")
    if handles:
        ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def _per_class_f1_fig(results: dict, model: str, config: str, kind: str,
                      out: Path) -> None:
    """Per-class F1 for one cell: which gestures the model tells apart
    well and which it confuses."""
    cell = _cells(results)[model][config][kind]
    f1 = np.asarray(cell["per_class_f1"], dtype=float) * 100
    classes = results["classes"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(classes)), f1, color=MODEL_COLOR.get(model, "#1f77b4"))
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_ylabel("F1 (%)")
    ax.set_title(f"Per-class F1 — {model} {config} {KIND_LABEL[kind]}")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def _ablation_fig(results: dict, out: Path) -> None:
    """Accuracy vs window length (256/128/64 samples = 250/125/62.5 ms)
    for the 'both' config on cross_day and cross_subject."""
    cells = _cells(results)
    crops = [("cnn", 256), ("cnn_w128", 128), ("cnn_w64", 64)]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for kind in ("cross_day", "cross_subject"):
        xs, ys, es = [], [], []
        for model, crop in crops:
            cell = cells.get(model, {}).get("both", {}).get(kind)
            if cell is None:
                continue
            xs.append(crop / 1.024)  # samples at 1024 Hz -> ms
            ys.append(cell["accuracy_mean"] * 100)
            es.append(cell["accuracy_std"] * 100)
        if xs:
            order = np.argsort(xs)
            ax.errorbar(np.asarray(xs)[order], np.asarray(ys)[order],
                        yerr=np.asarray(es)[order], marker="o", capsize=3,
                        label=KIND_LABEL[kind])
    ax.set_xlabel("window length (ms)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Window-length ablation (both channels)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = _load()
    classes = results["classes"]
    cells = _cells(results)
    # A confusion heatmap per model that has a 'both' cell, per kind.
    for model in MODELS:
        for kind in KINDS:
            cell = cells.get(model, {}).get("both", {}).get(kind)
            if cell is None:
                continue
            _confusion_fig(cell, classes,
                           f"{model}, all channels — {KIND_LABEL[kind]}",
                           FIG_DIR / f"confusion_{model}_both_{kind}.svg")
    _accuracy_bar_fig(results, FIG_DIR / "accuracy_by_split.svg")
    _per_subject_fig(results,
                     FIG_DIR / "within_subject_per_subject.svg")
    for kind in KINDS:
        cell = cells.get("cnn", {}).get("both", {}).get(kind)
        if cell is not None:
            _per_class_f1_fig(results, "cnn", "both", kind,
                              FIG_DIR / f"per_class_f1_cnn_both_{kind}.svg")
    _ablation_fig(results, FIG_DIR / "ablation_window_length.svg")
    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
