"""Cleaning one trial: keep EMG frequencies, remove hum, halve the rate.

In:  a raw trial array [samples, 32] at FS_IN (2048 Hz) and the mains
     frequency to notch (50 or 60 Hz — decided empirically, see
     `mains_check` / DECISIONS.md).
Out: a cleaned array [samples/2, 32] at FS_OUT (1024 Hz): zero-phase
     4th-order Butterworth band-pass 20–450 Hz, then a zero-phase notch at
     the mains frequency, then `scipy.signal.decimate` ×2.
If skipped: mains hum and out-of-band noise stay in the signal, and the
     high-frequency content above 512 Hz would alias into the decimated
     data — windows and features would describe interference, not muscle
     activity.

There is deliberately NO per-channel standardisation here: that is done
later, using statistics computed on the training fold only, so test data
never influences preprocessing (a leak would otherwise inflate every
reported number).

`--mains-check` runs the empirical mains-frequency test: Welch PSD on 20
random trials, peak search near 50 vs 60 Hz, figure to
artifacts/figures/mains_check.svg.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import signal

from .config import BAND, FS_IN


def bandpass(x: np.ndarray, fs: float = FS_IN) -> np.ndarray:
    sos = signal.butter(4, BAND, btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def notch(x: np.ndarray, mains: float, fs: float = FS_IN) -> np.ndarray:
    # Q=30 → ~2 Hz notch width at 60 Hz; harmonics are left to the band-pass.
    b, a = signal.iirnotch(mains, Q=30, fs=fs)
    return signal.filtfilt(b, a, x, axis=0)


def preprocess(x: np.ndarray, mains: float) -> np.ndarray:
    """band-pass → notch → decimate ×2. Returns float32 [n/2, C]."""
    y = bandpass(x)
    y = notch(y, mains)
    y = signal.decimate(y, 2, ftype="iir", axis=0, zero_phase=True)
    return y.astype(np.float32)


def mains_check(raw_dir=None, n_trials: int = 20, seed: int = 0,
                fig_path: str | None = None) -> dict:
    """Decide 50 vs 60 Hz from the data itself.

    Welch PSD pooled over `n_trials` random trials × all channels;
    compares the *prominence* of a spectral line at 50 vs 60 Hz relative
    to its surroundings. A deep hole at 60 Hz (hardware already notched)
    counts as evidence FOR 60 Hz, not against it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .data import RAW_DIR, iter_trials, read_trial

    raw = raw_dir or RAW_DIR
    from .data import is_complete

    trials = [t for t in iter_trials(raw) if is_complete(t[4])]
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(trials), size=min(n_trials, len(trials)), replace=False)

    psd_sum = None
    freqs = None
    for i in picks:
        _, _, _, _, path = trials[int(i)]
        x, fs, _ = read_trial(path)
        f, P = signal.welch(x, fs=fs, nperseg=min(4096, len(x)), axis=0)
        psd_sum = P.sum(axis=1) if psd_sum is None else psd_sum + P.sum(axis=1)
        freqs = f

    # Compare each candidate's line *prominence* (energy in ±1 Hz around
    # the candidate vs the surrounding ±8 Hz band). Raw band energy is
    # misleading here: EMG falls off as ~1/f, and the acquired data may
    # already carry a hardware notch at mains.
    def prominence(f0: float) -> float:
        line = (freqs >= f0 - 1) & (freqs <= f0 + 1)
        near = ((freqs >= f0 - 8) & (freqs <= f0 + 8)) & ~line
        return float(psd_sum[line].sum() / max(psd_sum[near].sum(), 1e-30))

    p50, p60 = prominence(50.0), prominence(60.0)
    # If one side has a hole rather than a peak (prominence ≈ 0), the
    # hardware already notched that frequency — that IS the mains answer.
    e50 = float(psd_sum[(freqs >= 49) & (freqs <= 51)].sum())
    e60 = float(psd_sum[(freqs >= 59) & (freqs <= 61)].sum())
    mains = 60.0 if p60 >= p50 else 50.0
    if e60 < e50 * 0.25:
        mains = 60.0  # pre-notched at 60 → mains was 60 Hz

    out = fig_path or "artifacts/figures/mains_check.svg"
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(freqs, psd_sum, lw=1)
    for f0, c in ((50, "tab:red"), (60, "tab:green")):
        ax.axvspan(f0 - 3, f0 + 3, color=c, alpha=0.15)
        ax.axvline(f0, color=c, lw=1, ls="--")
    ax.set_xlim(30, 120)
    ax.set_xlabel("Hz")
    ax.set_ylabel("Welch PSD (sum over channels)")
    ax.set_title(
        f"Mains check: prominence 50 Hz={p50:.2f}, 60 Hz={p60:.2f} "
        f"(60 Hz pre-notched in data) → mains = {mains:.0f} Hz"
    )
    fig.tight_layout()
    fig.savefig(out)
    print(f"mains_check: prom50={p50:.2f} prom60={p60:.2f} "
          f"e50={e50:.4g} e60={e60:.4g} → mains={mains:.0f} Hz → {out}")
    return {"mains": mains, "e50": e50, "e60": e60,
            "p50": p50, "p60": p60}


def _main() -> None:
    if "--mains-check" in sys.argv:
        res = mains_check()
        import json
        from pathlib import Path

        out = Path("artifacts/mains.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    _main()
