"""Turning predictions into numbers, in one place.

What goes in
    A confusion matrix: a 17×17 table of counts where row i, column j says
    "the true gesture was i and the model said j". Every training run saves
    exactly this table and nothing else about its predictions.

What comes out
    Accuracy, per-class F1 and macro-F1, all computed from that table.

Why only the table is saved
    If each training script computed its own accuracy, a bug in one of them
    would give a wrong number that looks like any other number. Saving the
    raw counts and computing every metric here means there is one
    definition of "accuracy" in the whole project, it is tested against
    hand-worked examples below, and anyone can recompute it from the
    committed run files.

What would go wrong without it
    Numbers on the website could disagree with each other, and nobody could
    tell which one was right.

Definitions (for one class c):
    precision_c = correct_c / predicted_as_c
    recall_c    = correct_c / actually_c
    F1_c        = 2 · precision · recall / (precision + recall)   (0 if both are 0)
    macro-F1    = mean of F1_c over classes that actually occur in the test set
    accuracy    = sum of the diagonal / sum of everything
"""

from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Count (true, predicted) pairs. Labels must be 0-based integers in [0, n_classes)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred have different shapes")
    if y_true.min() < 0 or y_true.max() >= n_classes or y_pred.min() < 0 or y_pred.max() >= n_classes:
        raise ValueError("labels outside [0, n_classes)")
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def accuracy(cm: np.ndarray) -> float:
    cm = np.asarray(cm)
    total = cm.sum()
    return float(np.trace(cm) / total) if total else float("nan")


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    """F1 for every class. Classes absent from the test set (empty row) get NaN."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    actually = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(actually > 0, tp / actually, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    f1[actually == 0] = np.nan
    return f1


def macro_f1(cm: np.ndarray) -> float:
    f1 = per_class_f1(cm)
    return float(np.nanmean(f1)) if np.isfinite(f1).any() else float("nan")


def chance_accuracy(cm: np.ndarray) -> float:
    """What a model that always guesses the most common class would score."""
    cm = np.asarray(cm)
    total = cm.sum()
    return float(cm.sum(axis=1).max() / total) if total else float("nan")
