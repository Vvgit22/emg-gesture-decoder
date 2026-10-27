"""Tests for the site-asset packing on fake data — no memmap needed."""

import numpy as np
import pytest

from emgdec.site_assets import _quantize, stitch, STEP, WIN


def _windows(c=28):
    """35 fake windows of one trial with honest 128-sample overlaps."""
    rng = np.random.default_rng(0)
    full = rng.normal(0, 1, (c, 4608)).astype(np.float16)
    return np.stack([full[:, k * STEP:k * STEP + WIN] for k in range(35)])


def test_stitch_recovers_trace():
    w = _windows()
    sig = stitch(w)
    assert sig.shape == (28, 4608)
    full = np.concatenate([w[0]] + [x[:, STEP:] for x in w[1:]], axis=1)
    assert np.array_equal(sig, full)


def test_stitch_fires_on_corrupted_overlap():
    w = _windows()
    w = w.copy()
    w[3][:, STEP:] += np.float16(0.5)  # break overlap between w[2], w[3]
    with pytest.raises(AssertionError, match="overlap"):
        stitch(w)


def test_int16_roundtrip_within_half_scale():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.05, (17, 28, 2048)).astype(np.float16)
    scale = float(np.abs(x).max() / 32000)
    q = _quantize(x, scale)
    err = np.abs(q.astype(np.float32) * scale - x).max()
    assert err <= scale / 2 + 1e-9  # quantisation error bounded by half step
    assert q.dtype == np.dtype("<i2")


def test_demo_bin_size_formula():
    assert 17 * 28 * 2048 * 2 == 1_949_696
