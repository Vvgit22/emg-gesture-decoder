"""Fast CPU tests for the TinyCNN and the training loop — fake data only."""

import numpy as np
import torch

from emgdec import splits
from emgdec.models import TinyCNN, n_params
from emgdec import train as T


def test_param_count_and_lengths():
    m = TinyCNN(28, 17)
    assert 40_000 <= n_params(m) <= 80_000
    for w in (256, 128, 64):
        out = m(torch.randn(4, 28, w))
        assert out.shape == (4, 17)


def test_normalisation_is_inside_the_model():
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(3.0, 2.0, (8, 12, 256)).astype(np.float32))
    mean = x.mean(dim=(0, 2))
    std = x.std(dim=(0, 2))
    normed = TinyCNN(12, 17, mean=mean.clone(), std=std.clone())
    plain = TinyCNN(12, 17)
    plain.load_state_dict(
        {k: v for k, v in normed.state_dict().items()
         if k not in ("mean", "std")}, strict=False)
    normed.eval()
    plain.eval()
    with torch.no_grad():
        a = normed(x)
        b = plain((x - mean.reshape(12, 1)) / std.reshape(12, 1))
    assert torch.max(torch.abs(a - b)) < 1e-5


def test_overfit_smoke_learns():
    """Three classes differing only in per-channel amplitude: 30 steps
    of the real training loop must get most of the training set right."""
    rng = np.random.default_rng(1)
    n, c, w = 200, 8, 64
    X = rng.normal(0, 0.1, (n, c, w)).astype(np.float32)
    y = rng.integers(0, 3, n)
    for cls in range(3):
        X[y == cls] += (cls + 1) * np.sin(
            np.linspace(0, 6 * np.pi, w))[None, :] * np.linspace(1, 2, c)[:, None]
    model = TinyCNN(c, 3)
    dev = torch.device("cpu")
    state, _, epochs = T.fit(model, X, y, None, None, device=dev,
                             max_epochs=30, patience=30, seed=0)
    model.load_state_dict(state)
    acc = T._accuracy(model, X, y, dev)
    assert epochs == 30
    assert acc > 0.9


def _fake_meta(n_per: int = 2):
    """2 subjects x 1 session x 17 gestures x 7 trials x n_per windows."""
    rows = []
    for s in (1, 2):
        for g in range(1, 18):
            for t in range(1, 8):
                rows += [(s, 1, g, t)] * n_per
    a = np.array(rows, dtype=np.int64)
    return {"subject": a[:, 0].astype(np.int16), "session": a[:, 1].astype(np.int8),
            "gesture": a[:, 2].astype(np.int8), "trial": a[:, 3].astype(np.int8)}


def test_train_split_end_to_end_and_leak_checked(monkeypatch):
    meta = _fake_meta()
    n = len(meta["subject"])
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (n, 28, 16)).astype(np.float32)
    called = []
    real = splits.assert_no_leak
    def spy(split, m):
        called.append(split.name)
        return real(split, m)
    monkeypatch.setattr(splits, "assert_no_leak", spy)
    sp = splits.make("within_subject", meta)[0]
    rec = T.train_split(X, meta, sp, "both", device="cpu", max_epochs=2,
                        patience=1, save_weights=False)
    assert called == [sp.name]
    assert rec.model == "cnn"
    assert rec.n_train == 17 * 4 * 2 and rec.n_test == 17 * 2 * 2
    assert np.asarray(rec.confusion).shape == (17, 17)
    assert np.asarray(rec.confusion).sum() == rec.n_test
    assert rec.epochs_run <= 2
