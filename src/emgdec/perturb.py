"""Synthetic perturbations: controlled stress tests on the eval side.

Eval-only: loads the v1 cnn/both/cross_subject checkpoints and re-scores
their test windows under a PREDEFINED perturbation grid (recorded in
DECISIONS): channel dropout zeroing k of 28 channels (3 seeded draws
averaged), per-channel random gain ~U(0.8,1.25) and ~U(0.5,2.0), and
additive white Gaussian noise at 20/10/0 dB SNR relative to each window's
RMS. These are controlled synthetic diagnostics, NOT evidence of real
electrode re-donning robustness.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from emgdec import splits
from emgdec.baselines import _load_meta
from emgdec.models import TinyCNN
from emgdec.train import MODELS_DIR, _open_X

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "experiments" / "protocol_v1_analysis" / "perturbations.json"
LABEL = ("controlled synthetic diagnostics, not evidence of real "
         "electrode re-donning robustness")
SEED = 1234


def _perturb(x: np.ndarray, kind: str, param, rng) -> np.ndarray:
    x = x.copy()
    if kind == "channel_dropout":
        k = param
        drop = rng.choice(x.shape[1], size=(len(x), k), replace=True)
        for i in range(len(x)):
            x[i, drop[i]] = 0
        return x
    if kind == "gain":
        lo, hi = param
        g = rng.uniform(lo, hi, (x.shape[0], x.shape[1], 1))
        return x * g
    if kind == "noise":
        snr_db = param
        rms = np.sqrt((x ** 2).mean(axis=(1, 2), keepdims=True)) + 1e-9
        n = rng.normal(0, 1, x.shape).astype(np.float32)
        n /= np.sqrt((n ** 2).mean(axis=(1, 2), keepdims=True)) + 1e-9
        return x + rms * n * 10 ** (-snr_db / 20)
    raise ValueError(kind)


GRID = [("channel_dropout", k) for k in (1, 2, 4, 8)] + \
       [("gain", (0.8, 1.25)), ("gain", (0.5, 2.0)),
        ("noise", 20), ("noise", 10), ("noise", 0)]
DRAWS = 3


def main() -> None:
    meta = _load_meta()
    X, _ = _open_X()
    out = {"label": LABEL, "grid": [f"{k}:{p}" for k, p in GRID],
           "folds": {}}
    for fold in range(5):
        name = f"fold{fold}"
        pt = MODELS_DIR / "cnn" / "both" / "cross_subject" / f"{name}.pt"
        split = next(s for s in splits.make("cross_subject", meta)
                     if s.name == name)
        net = TinyCNN(28, 17)
        net.load_state_dict(torch.load(pt, map_location="cpu",
                                       weights_only=True))
        net.eval()
        idx = np.sort(np.asarray(split.test))
        x = np.asarray(X[idx], dtype=np.float32)
        y = (meta["gesture"][idx].astype(int) - 1)

        def acc(xx):
            with torch.no_grad():
                pr = net(torch.from_numpy(
                    xx.astype(np.float32))).argmax(1).numpy()
            return float((pr == y).mean())

        res = {"clean": acc(x)}
        rng = np.random.default_rng(SEED + fold)
        for kind, param in GRID:
            accs = [acc(_perturb(x, kind, param, rng))
                    for _ in range(DRAWS)]
            res[f"{kind}:{param}"] = float(np.mean(accs))
        out["folds"][name] = res
        print(name, {k: round(v, 3) for k, v in res.items()}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
