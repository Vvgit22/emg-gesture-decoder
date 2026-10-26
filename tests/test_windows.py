"""windows.build on a tiny synthetic tree: counts, memmap, meta alignment."""

import numpy as np
import pyarrow.parquet as pq
import pytest

from emgdec.config import FS_OUT, KEEP, TRIAL_DROP_S, WIN
from emgdec.windows import build, windows_per_trial

pytestmark = pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("wfdb") is None,
    reason="wfdb not installed",
)


def _write_trial(root, subject, session, gesture, trial, seed=0):
    """Create a fake WFDB trial: 5 s, 32 channels, 2048 Hz, small amplitude."""
    import wfdb

    d = root / f"Session{session}" / f"session{session}_participant{subject}"
    d.mkdir(parents=True, exist_ok=True)
    name = f"session{session}_participant{subject}_gesture{gesture}_trial{trial}"
    rng = np.random.default_rng(seed + subject * 100 + gesture * 10 + trial)
    sig = rng.standard_normal((10240, 32)).astype(np.float32)
    # Write as digital samples like the real dataset (fmt 16, gain 200).
    wfdb.wrsamp(name, fs=2048, units=["uV"] * 32,
                sig_name=[f"ch{c}" for c in range(1, 33)],
                d_signal=np.round(sig * 200).astype(np.int32),
                fmt=["16"] * 32, adc_gain=[200] * 32, baseline=[0] * 32,
                write_dir=str(d))
    return sig


def test_windows_small(tmp_path):
    raw = tmp_path / "raw"
    _write_trial(raw, subject=1, session=1, gesture=1, trial=1, seed=7)
    _write_trial(raw, subject=1, session=1, gesture=17, trial=2, seed=8)
    out = tmp_path / "memmap"

    res = build(subjects=[1], out_dir=out, mains=60.0, raw_dir=raw)

    n_expected = 2 * windows_per_trial(10240 // 2)
    assert res["N"] == n_expected

    X = np.memmap(out / "X.f16", dtype=np.float16, mode="r",
                  shape=(res["N"], res["C"], res["W"]))
    meta = pq.read_table(out / "meta.parquet").to_pandas()
    assert list(meta.columns) == [
        "idx", "subject", "session", "gesture", "trial", "start_sample"
    ]
    assert len(meta) == res["N"]
    assert (meta["subject"] == 1).all()
    assert set(meta["gesture"]) == {1, 17}
    assert meta["idx"].tolist() == list(range(res["N"]))
    assert meta["start_sample"].min() >= int(TRIAL_DROP_S * FS_OUT)

    # Window 0 must equal the re-read trial, processed the same way.
    from emgdec.data import iter_trials, read_trial
    from emgdec.preprocess import preprocess

    _, _, _, _, path = next(iter_trials(raw))
    raw_sig, _, _ = read_trial(path)
    expected = preprocess(raw_sig, mains=60.0)[:, KEEP]
    w0 = meta.iloc[0]
    ref = expected[w0["start_sample"]:w0["start_sample"] + WIN].T
    np.testing.assert_allclose(np.asarray(X[0]), ref, atol=5e-2)
