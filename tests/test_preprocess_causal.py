"""Causal preprocessor: chunked == whole to 1e-9; strict causality."""

import numpy as np
import pytest

from emgdec.preprocess_causal import CausalPreprocessor, timing_report


def _signal(n=4096, c=6):
    rng = np.random.default_rng(0)
    t = np.arange(n) / 2048
    return rng.normal(0, 0.05, (n, c)) + np.sin(
        2 * np.pi * np.outer(t, [80, 120, 60, 200, 40, 150]))


@pytest.mark.parametrize("chunk", [1, 7, 64, 333])
def test_chunked_equals_whole(chunk):
    x = _signal()
    whole = CausalPreprocessor().process(x)
    p = CausalPreprocessor()
    outs = [p.process(x[i:i + chunk]) for i in range(0, len(x), chunk)]
    got = np.concatenate(outs)
    assert got.shape == whole.shape
    np.testing.assert_allclose(got, whole, atol=1e-9, rtol=0)


def test_causality_future_independent():
    x = _signal()
    p = CausalPreprocessor()
    a = p.process(x[:2000])
    x2 = x.copy(); x2[1500:] *= 10
    p2 = CausalPreprocessor()
    b = p2.process(x2[:1000])
    np.testing.assert_array_equal(a[:500], b[:500])


def test_timing_report(tmp_path):
    rep = timing_report(tmp_path / "timing.json")
    assert 50 in {int(k) for k in rep["group_delay_ms"]}
    assert rep["settle_s_1pct"] < 0.5  # must be under the dropped head
