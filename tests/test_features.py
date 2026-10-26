"""features.compute against cases you can verify by hand."""

import numpy as np

from emgdec.config import FS_OUT, WIN
from emgdec.features import FEATURES, N_FEATURES, compute

# column offsets inside one channel's 11-feature block
COL = {name: i for i, name in enumerate(FEATURES)}


def _one(x):
    """1 window × 1 channel → the 11 features as a flat row."""
    return compute(x.reshape(1, 1, WIN))[0]


def test_constant_window():
    c = 0.5
    f = _one(np.full(WIN, c, dtype=np.float32))
    assert f[COL["rms"]] == np.float32(abs(c))
    assert f[COL["mav"]] == np.float32(abs(c))
    assert f[COL["wl"]] == 0.0
    assert f[COL["zc"]] == 0.0
    assert f[COL["ssc"]] == 0.0
    assert f[COL["hj_mob"]] == 0.0


def test_sine_rms_and_zc():
    freq, amp = 10.0, 1.0
    t = np.arange(WIN) / FS_OUT
    f = _one((amp * np.sin(2 * np.pi * freq * t)).astype(np.float32))
    assert abs(f[COL["rms"]] - amp / np.sqrt(2)) < 0.01 * (amp / np.sqrt(2))
    cycles = freq * WIN / FS_OUT
    assert abs(f[COL["zc"]] - 2 * cycles) <= 1


def test_band_power_concentrates():
    freq = 150.0
    t = np.arange(WIN) / FS_OUT
    f = _one(np.sin(2 * np.pi * freq * t).astype(np.float32))
    bp = np.array([f[COL[k]] for k in FEATURES[7:]])
    assert bp[2] > 0.8 * bp.sum()


def test_batch_shape_and_finite():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 28, WIN)).astype(np.float32)
    f = compute(x)
    assert f.shape == (64, 28 * N_FEATURES)
    assert f.dtype == np.float32
    assert np.isfinite(f).all()
