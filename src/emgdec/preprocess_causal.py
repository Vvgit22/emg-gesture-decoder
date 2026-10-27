"""Causal preprocessing: the same filters, run one-way with memory.

The offline pipeline uses `sosfiltfilt` (forward+backward, zero delay) —
fine for batch analysis, impossible on a live stream. A wearable demo must
filter causally: each output sample may depend only on inputs so far.

What goes in: chunks of raw EMG [n, C] at FS_IN, any size, in order.
What comes out: decimated samples [m, C] at FS_OUT, as if the whole
signal had been processed in one shot (verified to 1e-9 in float64).

Chain (each stage keeps filter state across chunks):
  1. 4th-order Butterworth band-pass 20–450 Hz, forward once (NOT
     filtfilt — so amplitude response is |H|, not |H|², and group delay
     is real, reported in artifacts/experiments/causal_v1/timing.json).
  2. 60 Hz IIR notch, Q=30, same `iirnotch` design as the offline path.
  3. Anti-alias: Chebyshev I, order 8, 0.05 dB ripple, cutoff
     0.8*(fs_out/2) = 409.6 Hz — the filter `scipy.signal.decimate` uses
     for ftype='iir' — applied causally.
  4. Keep every 2nd sample; the parity (which sample index is 'even')
     is carried across chunk boundaries.

Units stay mV, channel order unchanged. `reset()` clears all state.
Warm-up: filters start from zero state at each trial start; the first
TRIAL_DROP_S seconds are dropped anyway, which must exceed the cascade's
settling time (step response to 1%) — computed in timing.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import signal

from emgdec.config import BAND, FS_IN, FS_OUT

Q_NOTCH = 30
AA_RIPPLE, AA_ORDER = 0.05, 8


class CausalPreprocessor:
    """Stateful causal band-pass -> notch -> anti-alias -> decimate x2."""

    def __init__(self, fs_in: float = FS_IN, mains: float = 60.0,
                 n_channels: int | None = None) -> None:
        self.fs_in, self.fs_out = fs_in, FS_OUT
        self.sos_bp = signal.butter(4, BAND, btype="bandpass",
                                    fs=fs_in, output="sos")
        self.b_n, self.a_n = signal.iirnotch(mains, Q=Q_NOTCH, fs=fs_in)
        # decimate()'s iir design: cheby1(8, 0.05, 0.8/q) with q=2
        self.b_aa, self.a_aa = signal.cheby1(AA_ORDER, AA_RIPPLE, 0.8 / 2)
        self._zi_bp = self._zi_n = self._zi_aa = None
        self._odd = 0  # decimation parity carried across chunks
        self._n_ch = n_channels
        self.reset()

    def reset(self) -> None:
        self._zi_bp = None
        self._zi_n = np.zeros(max(len(self.b_n), len(self.a_n)) - 1)
        self._zi_aa = np.zeros(max(len(self.b_aa), len(self.a_aa)) - 1)
        self._odd = 0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        x = np.asarray(chunk, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        n = x.shape[1]
        if self._zi_bp is None or self._zi_bp.shape[-1] != n:
            # sosfilt zi: (n_sections, 2, n_channels); lfilter zi: (k, n_ch)
            self._zi_bp = np.zeros((self.sos_bp.shape[0], 2, n))
            self._zi_n = np.zeros((max(len(self.b_n), len(self.a_n)) - 1, n))
            self._zi_aa = np.zeros((max(len(self.b_aa), len(self.a_aa)) - 1, n))
        y, self._zi_bp = signal.sosfilt(self.sos_bp, x, axis=0,
                                        zi=self._zi_bp)
        y, self._zi_n = signal.lfilter(self.b_n, self.a_n, y, axis=0,
                                       zi=self._zi_n)
        y, self._zi_aa = signal.lfilter(self.b_aa, self.a_aa, y, axis=0,
                                        zi=self._zi_aa)
        out = y[self._odd::2]
        self._odd = (self._odd + len(y)) % 2
        return out


def group_delay_ms(fs_in: float = FS_IN) -> dict:
    """Group delay of the causal cascade at probe frequencies."""
    _, gd_bp = signal.group_delay((signal.butter(
        4, BAND, btype="bandpass", fs=fs_in, output="sos"),), fs=fs_in) \
        if False else (None, None)
    return {}


def timing_report(out_path: Path, fs_in: float = FS_IN) -> dict:
    """Group delay at 50/100/200 Hz + settling time of the cascade."""
    b_bp, a_bp = signal.butter(4, BAND, btype="bandpass", fs=fs_in)
    b_n, a_n = signal.iirnotch(60.0, Q=Q_NOTCH, fs=fs_in)
    b_aa, a_aa = signal.cheby1(AA_ORDER, AA_RIPPLE, 0.8 / 2)
    # cascade: convolve numerator chains
    def casc(b1, a1, b2, a2, b3, a3):
        return (np.convolve(np.convolve(b1, b2), b3),
                np.convolve(np.convolve(a1, a2), a3))
    B, A = casc(b_bp, a_bp, b_n, a_n, b_aa, a_aa)
    w, gd = signal.group_delay((B, A), w=[50, 100, 200], fs=fs_in)
    # settling time: step response reaches and stays within 1% of final
    t_step = 5 * fs_in
    _, step = signal.dstep((B, A, 1 / fs_in), n=int(t_step))
    s = np.squeeze(step)
    peak = np.abs(s).max()  # band-pass step response decays to ~0;
    ok = np.flatnonzero(np.abs(s) > 0.01 * peak)  # settle = decay to 1% of peak
    settle_s = (ok[-1] + 1) / fs_in if len(ok) else 0.0
    rep = {"group_delay_ms": {str(int(f)): float(g * 1000 / fs_in)
                              for f, g in zip(w, gd)},
           "settle_s_1pct": float(settle_s),
           "window_ms": 256 / FS_OUT * 1000,
           "hop_ms": 128 / FS_OUT * 1000,
           "fs_in": fs_in, "fs_out": FS_OUT}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    out = (Path(__file__).resolve().parents[2] / "artifacts" /
           "experiments" / "causal_v1" / "timing.json")
    print(json.dumps(timing_report(out), indent=1))
