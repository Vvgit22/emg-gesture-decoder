"""Synthetic perturbations: controlled stress tests on the eval side.

Eval-only: loads the v1 cnn/both/cross_subject checkpoints and re-scores
their test windows under a PREDEFINED perturbation grid (recorded in
DECISIONS): channel dropout zeroing k of 28 channels (3 seeded draws
averaged), per-channel random gain ~U(0.8,1.25) and ~U(0.5,2.0), and
additive white Gaussian noise at 20/10/0 dB SNR relative to each window's
RMS. These are controlled synthetic diagnostics, NOT evidence of real
electrode re-donning robustness.

Speed: windows load once per fold via the chunked float16 loader;
inference runs on the accelerator in batches of 2048; each perturbation
is applied to the whole array at once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from emgdec import splits
from emgdec.baselines import _load_meta
from emgdec.models import TinyCNN
from emgdec.train import MODELS_DIR, _load_windows, _open_X

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "experiments" / "protocol_v1_analysis" / "perturbations.json"
LABEL = ("controlled synthetic diagnostics, not evidence of real "
         "electrode re-donning robustness")
SEED = 1234
BATCH = 2048


def _perturb_all(x: np.ndarray, kind: str, param, rng) -> np.ndarray:
    """Vectorised perturbation of [N, C, W]."""
    x = x.copy()
    if kind == "channel_dropout":
        k = param
        # a fixed random channel set per draw (same dropout everywhere),
        # seeded — matches "k randomly chosen channels, 3 draws averaged"
        drop = rng.choice(x.shape[1], size=k, replace=False)
        x[:, drop] = 0
        return x
    if kind == "gain":
        lo, hi = param
        g = rng.uniform(lo, hi, (1, x.shape[1], 1)).astype(np.float32)
        return x * g
    if kind == "noise":
        snr_db = param
        # float32 noise generated per 20k-window block — a full float64
        # allocation is ~6.3 GB and swaps this machine
        scale = 10 ** (-snr_db / 20)
        out = np.empty_like(x, dtype=np.float32)
        for i in range(0, len(x), 20000):
            b = x[i:i + 20000].astype(np.float32)
            rms = np.sqrt((b ** 2).mean(axis=(1, 2), keepdims=True)) + 1e-9
            n = rng.standard_normal(b.shape, dtype=np.float32)
            n /= np.sqrt((n ** 2).mean(axis=(1, 2), keepdims=True)) + 1e-9
            out[i:i + 20000] = b + rms * n * scale
        return out
    raise ValueError(kind)


GRID = [("channel_dropout", k) for k in (1, 2, 4, 8)] + \
       [("gain", (0.8, 1.25)), ("gain", (0.5, 2.0)),
        ("noise", 20), ("noise", 10), ("noise", 0)]
DRAWS = 3


def _acc(net, x, y, device) -> float:
    n = 0
    for i in range(0, len(x), BATCH):
        xb = torch.from_numpy(x[i:i + BATCH].astype(np.float32)).to(device)
        with torch.no_grad():
            pr = net(xb).argmax(1).cpu().numpy()
        n += (pr == y[i:i + BATCH]).sum()
    return n / len(y)


def main(experiment: str = "offline_v1b") -> None:
    meta = _load_meta()
    X, _ = _open_X()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    chans = np.arange(28)
    out = {"label": LABEL + ("; checkpoints are offline_v1b seed-0 (the v1"
           " .pt were overwritten by later experiments — same split"
           " protocol, same architecture, new code)"),
           "grid": [f"{k}:{p}" for k, p in GRID], "folds": {}}
    for fold in range(5):
        name = f"fold{fold}"
        pt = (MODELS_DIR / experiment / "cnn" / "both" / "cross_subject"
              / f"{name}.pt")
        split = next(s for s in splits.make("cross_subject", meta)
                     if s.name == name)
        net = TinyCNN(28, 17).to(device)
        net.load_state_dict(torch.load(pt, map_location="cpu",
                                       weights_only=True))
        net.eval()
        x = _load_windows(X, np.asarray(split.test), chans, 256)
        y = (meta["gesture"][split.test].astype(int) - 1)

        res = {"clean": _acc(net, x, y, device)}
        for kind, param in GRID:
            accs = [_acc(net, _perturb_all(x, kind, param,
                                          np.random.default_rng(SEED + fold * 100 + d)),
                         y, device) for d in range(DRAWS)]
            res[f"{kind}:{param}"] = float(np.mean(accs))
        out["folds"][name] = res
        print(name, {k: round(v, 3) for k, v in res.items()}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
