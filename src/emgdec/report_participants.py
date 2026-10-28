"""Participant-level report: people, not windows, are the unit.

Reads the predictions npz from protocol_v1_analysis/predictions and
aggregates accuracy / macro-F1 per participant (all of that person's test
windows pooled within a kind), then across participants: mean, median, SD,
min, max, 5th percentile, counts below 70% and 80%. Also per-class
precision/recall/F1, a paired CNN-minus-LDA difference per participant
with a 95% CI from a participant-level bootstrap (10k resamples, seeded —
windows within a trial are correlated, so the participant is the
resampling unit), rest<->active error rates, and trial-level accuracy by
majority vote over a trial's 35 windows (clearly labelled).

Outputs artifacts/experiments/protocol_v1_analysis/participants.{json,md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from emgdec import metrics
from emgdec.config import GESTURE_NAMES, N_GESTURES

REPO = Path(__file__).resolve().parents[2]
PRED = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    REPO / "artifacts" / "experiments" / "protocol_v1_analysis" / "predictions"
OUTD = (Path(sys.argv[2]) if len(sys.argv) > 2 else
        REPO / "artifacts" / "experiments" / "protocol_v1_analysis")
REST = 16  # 0-based gesture 17 = Rest


def _cm_for(kind: str, model: str, config: str) -> tuple[np.ndarray, dict]:
    """Pooled test confusion + per-(name,subject) record list."""
    files = sorted((PRED / model / config / kind).glob("*.npz"))
    total = np.zeros((N_GESTURES, N_GESTURES), int)
    parts = {}
    for f in files:
        z = np.load(f)
        te = z["flag"] == 1
        y = z["gesture"][te].astype(int) - 1
        p = z["pred"][te].astype(int)
        total += metrics.confusion(y, p, N_GESTURES)
        for s in np.unique(z["subject"][te]):
            m = (z["subject"][te] == s)
            parts.setdefault(int(s), []).append(
                (y[m], p[m], z["trial"][te][m], z["win_in_trial"][te][m],
                 z["session"][te][m], z["gesture"][te][m]))
    return total, parts


def _per_participant(parts: dict) -> dict[int, dict]:
    out = {}
    for s, chunks in parts.items():
        y = np.concatenate([c[0] for c in chunks])
        p = np.concatenate([c[1] for c in chunks])
        cm = metrics.confusion(y, p, N_GESTURES)
        out[s] = {"acc": metrics.accuracy(cm), "f1": metrics.macro_f1(cm),
                  "n": int((p == p).sum()),
                  "rest_to_active": float(cm[REST, :REST].sum() / cm[REST].sum())
                  if cm[REST].sum() else None,
                  "active_to_rest": float(cm[:REST, REST].sum() / cm[:REST].sum())
                  if cm[:REST].sum() else None}
    return out


def _trial_accuracy(parts: dict) -> float:
    """Majority vote per trial = (subject, session, gesture, trial):
    the trial's gesture is compared to the modal prediction of its windows."""
    correct = tot = 0
    for s, chunks in parts.items():
        for y, p, t, w, ses, g in chunks:
            keys = set(zip(ses.tolist(), g.tolist(), t.tolist()))
            for se, gg, tr in keys:
                m = (ses == se) & (g == gg) & (t == tr)
                votes = np.bincount(p[m], minlength=N_GESTURES).argmax()
                correct += votes == y[m][0]
                tot += 1
    return correct / tot if tot else float("nan")


def _summary(per: dict) -> dict:
    a = np.array([v["acc"] for v in per.values()])
    return {"n_participants": len(a), "mean": a.mean(), "median": np.median(a),
            "sd": a.std(ddof=1), "min": a.min(), "max": a.max(),
            "p5": np.percentile(a, 5),
            "below70": int((a < 0.70).sum()), "below80": int((a < 0.80).sum())}


def _bootstrap_diff(cnn: dict, lda: dict, seed: int = 0) -> dict:
    subs = sorted(set(cnn) & set(lda))
    d = np.array([cnn[s]["acc"] - lda[s]["acc"] for s in subs])
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, (10000, len(d)), replace=True).mean(1)
    return {"n_pairs": len(d), "mean_diff_pp": d.mean() * 100,
            "sd_diff_pp": d.std(ddof=1) * 100,
            "ci95_pp": [float(np.percentile(boots, 2.5) * 100),
                        float(np.percentile(boots, 97.5) * 100)]}


def main() -> None:
    report = {"note": "windows within a trial are correlated; the "
                      "participant is the resampling/reporting unit",
              "kinds": {}}
    kinds = sys.argv[3].split(",") if len(sys.argv) > 3 else \
        ["cross_subject", "cross_day"]
    models_l = sys.argv[4].split(",") if len(sys.argv) > 4 else \
        ["cnn", "lda"]
    for kind in kinds:
        for model in models_l:
            if not (PRED / model / "both" / kind).exists():
                continue
            total, parts = _cm_for(kind, model, "both")
            per = _per_participant(parts)
            tp = np.diag(total).astype(float)
            prec = np.divide(tp, total.sum(0), out=np.zeros_like(tp),
                             where=total.sum(0) > 0)
            rec = np.divide(tp, total.sum(1), out=np.zeros_like(tp),
                            where=total.sum(1) > 0)
            entry = {"per_participant": {str(s): v for s, v in per.items()},
                     "summary": _summary(per),
                     "trial_level_majority_accuracy": _trial_accuracy(parts),
                     "per_class": {
                         GESTURE_NAMES[i]: {
                             "precision": float(prec[i]),
                             "recall": float(rec[i]),
                             "f1": float(metrics.per_class_f1(total)[i])}
                         for i in range(N_GESTURES)},
                     "rest_errors_pooled": {
                         "rest_to_active": float(
                             total[REST, :REST].sum() / total[REST].sum()),
                         "active_to_rest": float(
                             total[:REST, REST].sum() / total[:REST].sum())}}
            report["kinds"].setdefault(kind, {})[model] = entry
        if "cnn" in report["kinds"][kind] and \
                "lda" in report["kinds"][kind]:
            cnn = report["kinds"][kind]["cnn"]["per_participant"]
            lda = report["kinds"][kind]["lda"]["per_participant"]
            c = {int(k): v for k, v in cnn.items()}
            lda_p = {int(k): v for k, v in lda.items()}
            report["kinds"][kind]["paired_cnn_minus_lda"] = \
                _bootstrap_diff(c, lda_p)
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "participants.json").write_text(json.dumps(report, indent=1))
    md = ["# Participant-level results\n",
          "Windows within a trial are correlated; the participant is the unit.\n"]
    for kind, models in report["kinds"].items():
        for model in ("cnn", "lda"):
            if model not in models:
                continue
            s = models[model]["summary"]
            md.append(f"## {model} / both / {kind}\n"
                      f"- n={s['n_participants']} participants; mean "
                      f"{100*s['mean']:.2f}%, median {100*s['median']:.2f}%, "
                      f"SD {100*s['sd']:.2f} pp, range "
                      f"{100*s['min']:.2f}–{100*s['max']:.2f}%, "
                      f"p5 {100*s['p5']:.2f}%\n"
                      f"- below 70%: {s['below70']}; below 80%: {s['below80']}\n"
                      f"- trial-level (majority vote): "
                      f"{100*models[model]['trial_level_majority_accuracy']:.2f}%\n"
                      f"- rest→active {100*models[model]['rest_errors_pooled']['rest_to_active']:.2f}%, "
                      f"active→rest {100*models[model]['rest_errors_pooled']['active_to_rest']:.2f}%\n")
        if "paired_cnn_minus_lda" not in models:
            continue
        d = models["paired_cnn_minus_lda"]
        md.append(f"### paired CNN−LDA ({kind})\n"
                  f"mean {d['mean_diff_pp']:.2f} pp, SD {d['sd_diff_pp']:.2f} pp, "
                  f"95% CI [{d['ci95_pp'][0]:.2f}, {d['ci95_pp'][1]:.2f}] "
                  f"(participant bootstrap, 10k)\n")
    (OUTD / "participants.md").write_text("\n".join(md))
    print((OUTD / "participants.md").read_text())


if __name__ == "__main__":
    main()
