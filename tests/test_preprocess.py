"""preprocess must halve the length, keep channels, and stay finite."""

import numpy as np

from emgdec.config import FS_IN
from emgdec.preprocess import preprocess


def test_preprocess_shapes():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2048, 32)).astype(np.float32)
    y = preprocess(x, mains=60.0)
    assert y.shape == (1024, 32)
    assert np.isfinite(y).all()


def test_preprocess_removes_mains():
    # A strong 60 Hz tone plus small EMG-like noise; after processing the
    # 60 Hz component should be nearly gone.
    t = np.arange(FS_IN) / FS_IN
    x = (5.0 * np.sin(2 * np.pi * 60 * t))[:, None] + 0.1 * np.random.default_rng(
        1
    ).standard_normal((FS_IN, 32))
    y = preprocess(x.astype(np.float32), mains=60.0)
    t_out = np.arange(len(y)) / (FS_IN // 2)
    tone_in = abs(np.mean(x[:, 0] * np.exp(-2j * np.pi * 60 * t)))
    tone_out = abs(np.mean(y[:, 0] * np.exp(-2j * np.pi * 60 * t_out)))
    assert tone_out < tone_in * 0.15
