"""Classical EMG features: 11 numbers per channel, per window.

What goes in
    A batch of windows `X` shaped [N, C, W] (as stored in data/memmap/X.f16,
    already band-passed, notched and decimated — the per-channel
    standardisation happens later, at fit time, on training data only).

What comes out
    A float32 array [N, C*11], channel-major: for channel c the block
    cols[c*11:(c+1)*11] holds, in order
      RMS, MAV, WL, ZC, SSC, Hjorth mobility, Hjorth complexity,
      band power 20-50, 50-100, 100-200, 200-450 Hz.

What each feature measures, in words
    RMS       how strong the signal is on average (root-mean-square —
              like loudness).
    MAV       mean absolute value — the average deflection from zero.
    WL        waveform length — how far the pen would travel drawing the
              signal; more wiggle, more length.
    ZC        zero crossings — how often the signal crosses zero with a
              real step, not just noise. Needs a threshold because a
              signal hovering near zero would otherwise count thousands
              of meaningless crossings.
    SSC       slope sign changes — how often the signal turns around:
              peaks and troughs, same threshold idea as ZC.
    Hjorth mobility      like an average frequency estimate: if the
              derivative is strong relative to the signal, the signal is
              fast-moving.
    Hjorth complexity    how much the "average frequency" itself varies —
              a clean sine gives ~1, a rich signal gives more.
    Band powers          how much of the signal's energy sits in each
              frequency band (Welch PSD, nperseg=128).

The ZC/SSC threshold is a fixed 1e-3 in whatever units arrive (the input
is roughly z-scored per channel later; a fixed threshold is used instead
of a per-channel fraction of std so that a genuinely silent channel does
not manufacture crossings out of its own noise floor).

Skipped at your peril
    These features are the whole reason the LDA baseline exists: without
    them the only model comparison would be "CNN vs nothing", and any
    claim that the CNN earns its complexity would be unverifiable.

CLI: `uv run python -m emgdec.features --cache` — streams the whole
     memmap in chunks, writes data/memmap/features.f32 [N, 308] and
     features_names.json. Chunked so peak RSS stays far below 6 GB.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import signal

from .config import FS_OUT, WIN

FEATURES = ("rms", "mav", "wl", "zc", "ssc", "hj_mob", "hj_cplx",
            "bp_20_50", "bp_50_100", "bp_100_200", "bp_200_450")
N_FEATURES = len(FEATURES)  # 11
ZC_SSC_THRESHOLD = 1e-3
BANDS = ((20, 50), (50, 100), (100, 200), (200, 450))
CHUNK = 20_000  # windows per block; ~0.6 GB float32 intermediate

REPO = Path(__file__).resolve().parents[2]
MEMMAP_DIR = REPO / "data" / "memmap"
FEATURES_F32 = MEMMAP_DIR / "features.f32"
FEATURES_NAMES = MEMMAP_DIR / "features_names.json"


def _band_powers(x: np.ndarray) -> np.ndarray:
    """Welch PSD band sums for x [N, C, W] → [N, C, 4]."""
    f, p = signal.welch(x, fs=FS_OUT, nperseg=min(128, WIN), axis=-1)
    out = np.stack(
        [p[..., (f >= lo) & (f <= hi)].sum(axis=-1) for lo, hi in BANDS],
        axis=-1,
    )
    return out


def compute(x: np.ndarray) -> np.ndarray:
    """x [N, C, W] (any dtype) → float32 features [N, C*11]."""
    x = np.asarray(x, dtype=np.float32)
    n, c, w = x.shape
    dx = np.diff(x, axis=-1)

    rms = np.sqrt(np.mean(x * x, axis=-1))
    mav = np.mean(np.abs(x), axis=-1)
    wl = np.sum(np.abs(dx), axis=-1)

    thr = ZC_SSC_THRESHOLD
    # ZC: sign flip across zero with a real step.
    s = np.signbit(x)
    zc = ((s[..., 1:] != s[..., :-1]) & (np.abs(dx) >= thr)).sum(axis=-1)
    # SSC: the first difference changes sign where both steps are real.
    d1, d2 = dx[..., :-1], dx[..., 1:]
    ssc = ((d1 * d2 < 0) & ((np.abs(d1) >= thr) | (np.abs(d2) >= thr))).sum(axis=-1)

    var0 = x.var(axis=-1)
    var1 = dx.var(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mob1 = np.sqrt(np.where(var0 > 0, var1 / var0, 0.0))
        var2 = np.diff(dx, axis=-1).var(axis=-1)
        mob2 = np.sqrt(np.where(var1 > 0, var2 / var1, 0.0))
        hj_cplx = np.where(mob1 > 0, mob2 / mob1, 0.0)

    feats = np.concatenate(
        [
            rms[..., None], mav[..., None], wl[..., None],
            zc[..., None].astype(np.float32), ssc[..., None].astype(np.float32),
            mob1[..., None], hj_cplx[..., None],
            _band_powers(x),
        ],
        axis=-1,
    )
    return feats.reshape(n, c * feats.shape[-1]).astype(np.float32)


def cache(chunk: int = CHUNK, memmap_dir: Path = MEMMAP_DIR) -> dict:
    """Stream X.f16 → features.f32 one chunk at a time."""
    shape = json.loads((memmap_dir / "shape.json").read_text())
    n, c, w = shape["N"], shape["C"], shape["W"]
    X = np.memmap(memmap_dir / "X.f16", dtype=np.float16, mode="r",
                  shape=(n, c, w))
    F = np.memmap(FEATURES_F32, dtype=np.float32, mode="w+",
                  shape=(n, c * N_FEATURES))
    t0 = time.time()
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        F[lo:hi] = compute(np.asarray(X[lo:hi]))
        F.flush()
        print(f"features: {hi}/{n}  ({time.time() - t0:.0f}s)", flush=True)
    names = [f"ch{ch}_{name}" for ch in range(c) for name in FEATURES]
    FEATURES_NAMES.write_text(json.dumps(
        {"names": names, "features_per_channel": N_FEATURES,
         "order": "channel-major", "feature_names": FEATURES}))
    secs = time.time() - t0
    print(f"features: N={n} cols={c * N_FEATURES} wall={secs:.0f}s")
    return {"N": n, "cols": c * N_FEATURES, "seconds": secs}


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    args = ap.parse_args()
    if args.cache:
        cache()


if __name__ == "__main__":
    _main()
