"""Calibration: how honest are the network's confidences?

For cnn/both on each split kind, on the saved logits:
  - reliability diagram data (10 equal-width bins over max softmax prob)
  - ECE = sum_b (n_b/N) * |acc_b - conf_b| over those bins
  - NLL and multi-class Brier score
  - temperature scaling: T fit by minimising NLL on the VAL windows of each
    fold (scipy L-BFGS over log T), applied to test; before/after metrics
  - rejection: pick a threshold on calibrated max-prob ON VAL to hit target
    coverage {100,90,80,70}%, then report achieved test coverage, accuracy
    on accepted, and rejected fraction.

Writes calibration.json + calibration.md. The two demo models get their
temperature appended into their export .meta.json (`temperature` key).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_P = (sys.argv[sys.argv.index("--pred") + 1] if "--pred" in sys.argv
      else None)
_E = (sys.argv[sys.argv.index("--experiment") + 1]
      if "--experiment" in sys.argv else "protocol_v1_analysis")
PRED = (Path(_P) if _P else
        REPO / "artifacts" / "experiments" / _E / "predictions")
OUTD = REPO / "artifacts" / "experiments" / _E
ONNX_DIR = REPO / "artifacts" / "onnx"
BINS = 10
COVERAGE_TARGETS = [1.0, 0.9, 0.8, 0.7]


def _softmax(logits, T=1.0):
    z = logits.astype(np.float64) / T
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def _brier(p, y):
    one = np.zeros_like(p)
    one[np.arange(len(y)), y] = 1
    return float(((p - one) ** 2).sum(1).mean())


def _ece(p, y, bins=BINS):
    conf = p.max(1)
    pred = p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if not m.sum():
            continue
        acc, cf = (pred[m] == y[m]).mean(), conf[m].mean()
        ece += m.mean() * abs(acc - cf)
        rows.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                     "acc": float(acc), "conf": float(cf)})
    return float(ece), rows


def _fit_T(val_logits, y):
    from scipy.optimize import minimize
    def f(lt):
        return _nll(_softmax(val_logits, np.exp(lt[0])), y)
    r = minimize(f, np.zeros(1), method="L-BFGS-B",
                 bounds=[(np.log(0.05), np.log(10))])
    return float(np.exp(r.x[0]))


def _load(kind: str, name: str):
    z = np.load(PRED / "cnn" / "both" / kind / f"{name}.npz")
    y = z["gesture"].astype(int) - 1
    return z, y


def main() -> None:
    kinds = ["cross_subject", "cross_day", "within_subject"]
    report = {"ece_definition": "sum_b (n_b/N)|acc_b-conf_b|, "
              "10 equal-width bins on max softmax prob",
              "temperature": "NLL-min on VAL logits, L-BFGS on log T in "
                             "[0.05,10]",
              "models": {}}
    temps = {}
    for kind in kinds:
        splits_dir = PRED / "cnn" / "both" / kind
        for f in sorted(splits_dir.glob("*.npz")):
            z, y = _load(kind, f.stem)
            vl, vt = z["flag"] == 0, z["flag"] == 1
            logits_v, logits_t = z["logits"][vl], z["logits"][vt]
            yv, yt = y[vl], y[vt]
            p_raw = _softmax(logits_t)
            T = _fit_T(logits_v, yv)
            p_cal = _softmax(logits_t, T)
            ece_r, bins_r = _ece(p_raw, yt)
            ece_c, bins_c = _ece(p_cal, yt)
            # rejection thresholds chosen on VAL
            conf_v = _softmax(logits_v, T).max(1)
            conf_t = p_cal.max(1)
            corr_t = p_cal.argmax(1) == yt
            rej = []
            for cov in COVERAGE_TARGETS:
                thr = np.quantile(conf_v, 1 - cov)
                acc_mask = conf_t >= thr
                rej.append({"target_coverage": cov,
                            "test_coverage": float(acc_mask.mean()),
                            "accepted_accuracy": float(
                                corr_t[acc_mask].mean()) if acc_mask.any()
                            else None,
                            "rejected_fraction": float((~acc_mask).mean())})
            report["models"].setdefault(kind, {})[f.stem] = {
                "temperature": T,
                "raw": {"ece": ece_r, "nll": _nll(p_raw, yt),
                        "brier": _brier(p_raw, yt), "bins": bins_r},
                "calibrated": {"ece": ece_c, "nll": _nll(p_cal, yt),
                               "brier": _brier(p_cal, yt), "bins": bins_c},
                "rejection": rej}
            temps.setdefault(kind, {})[f.stem] = T
        print(f"{kind}: done", flush=True)
    # pooled cross_subject raw bins for the audit comparison
    outs = report["models"]["cross_subject"]
    for nm, cell in outs.items():
        print(f"{nm}: T={cell['temperature']:.3f} "
              f"ece {100*cell['raw']['ece']:.2f}->{100*cell['calibrated']['ece']:.2f}pp")
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "calibration.json").write_text(json.dumps(report, indent=1))
    md = ["# Calibration\n",
          "ECE = sum over 10 equal-width bins of (n_b/N)|acc_b-conf_b|; "
          "temperature fit on val NLL (L-BFGS).\n"]
    for kind, cells in report["models"].items():
        md.append(f"## {kind}")
        for nm, c in cells.items():
            md.append(f"- {nm}: T={c['temperature']:.3f}, "
                      f"raw ece {100*c['raw']['ece']:.2f}pp nll "
                      f"{c['raw']['nll']:.3f} brier {c['raw']['brier']:.3f} -> "
                      f"cal ece {100*c['calibrated']['ece']:.2f}pp")
            md.append("  rejection " + "; ".join(
                f"c{int(r['target_coverage']*100)}: cov "
                f"{100*r['test_coverage']:.1f} acc "
                f"{100*(r['accepted_accuracy'] or 0):.1f}"
                for r in c["rejection"]))
    (OUTD / "calibration.md").write_text("\n".join(md))
    # demo model temperatures into their meta.json
    for stem, kind in (("cnn_both_within_subject_s01_d1", "within_subject"),
                       ("cnn_both_cross_subject_fold3", "cross_subject")):
        mp = ONNX_DIR / f"{stem}.meta.json"
        if not mp.exists():
            continue
        mj = json.loads(mp.read_text())
        key = stem.split(kind + "_")[1]
        if key in temps.get(kind, {}):
            mj["temperature"] = temps[kind][key]
            mp.write_text(json.dumps(mj, indent=1))
            print(f"temperature -> {mp.name}: {mj['temperature']:.3f}")


if __name__ == "__main__":
    main()
