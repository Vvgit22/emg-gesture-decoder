"""Adaptation: leak guards, recipe runs, n=0 returns base predictions."""
import numpy as np
import torch

from emgdec.adapt import assert_adapt_leak, fine_tune
from emgdec.models import TinyCNN


def test_adapt_leak_fires_on_overlap():
    meta = {"subject": np.array([1, 1, 1, 1]),
            "session": np.array([1, 1, 2, 2]),
            "trial": np.array([1, 2, 1, 2])}
    import pytest
    with pytest.raises(AssertionError):
        assert_adapt_leak(np.array([0]), np.array([0]), meta, 1, {1})
    # no shared (session,trial) group -> passes
    assert_adapt_leak(np.array([0, 1]), np.array([2, 3]), meta, 1, {1})


def test_fine_tune_changes_weights_and_runs_8_epochs():
    net = TinyCNN(28, 17)
    w0 = net.net[0].weight.detach().clone() if hasattr(net, 'net') else \
        next(net.parameters()).detach().clone()
    x = np.random.default_rng(0).normal(0, 1, (64, 28, 16)).astype(np.float32)
    y = np.random.default_rng(0).integers(0, 17, 64)
    n = fine_tune(net, x, y, seed=0, device=torch.device("cpu"))
    assert n == 8
    w1 = next(net.parameters()).detach().clone()
    assert not torch.equal(next(iter([w0])), w1)


def test_acc_on_noncpu_device():
    """_acc works when the model lives on mps (the mps/cpu crash)."""
    import pytest
    from emgdec.adapt import _acc
    if not torch.backends.mps.is_available():
        pytest.skip("no mps")
    net = TinyCNN(28, 17).to("mps")
    x = np.random.default_rng(0).normal(0, 1, (5000, 28, 16)).astype(np.float16)
    y = np.random.default_rng(0).integers(1, 18, 5000)
    cm = _acc(net, x, y, torch.device("mps"))
    assert cm.shape == (17, 17) and cm.sum() == 5000
