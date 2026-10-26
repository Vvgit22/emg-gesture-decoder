"""The most important test in the repository.

If this fails, every accuracy number is suspect. It builds a small fake
metadata table shaped like the real one (several subjects, three sessions,
all gestures, seven trials, several windows per trial), generates every
split, and checks that nothing from the test side can be seen in training.

It also checks the coverage properties promised in the write-up: every
subject is tested exactly once in cross_subject; within_subject tests on
trials 6-7 and nothing else; cross_day tests on session 3 only.

Finally, if the real meta.parquet exists, the same leak checks run on it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from emgdec import splits as S

REAL_META = Path(os.environ.get("EMGDEC_MEMMAP", "data/memmap")) / "meta.parquet"


def fake_meta(n_subjects=9, n_sessions=3, n_gestures=17, n_trials=7, windows_per_trial=5, seed=0):
    rows = np.array(
        [
            (s, d, g, t, w)
            for s in range(1, n_subjects + 1)
            for d in range(1, n_sessions + 1)
            for g in range(1, n_gestures + 1)
            for t in range(1, n_trials + 1)
            for w in range(windows_per_trial)
        ],
        dtype=np.int64,
    )
    np.random.default_rng(seed).shuffle(rows)  # row order must not matter
    return {
        "subject": rows[:, 0].astype(np.int16),
        "session": rows[:, 1].astype(np.int8),
        "gesture": rows[:, 2].astype(np.int8),
        "trial": rows[:, 3].astype(np.int8),
        "start_sample": rows[:, 4].astype(np.int32),
    }


@pytest.fixture(scope="module")
def meta():
    return fake_meta()


@pytest.mark.parametrize("kind", list(S.ALL))
def test_no_leak_on_fake_data(meta, kind):
    made = S.make(kind, meta)
    assert made, f"{kind} produced no splits"
    for sp in made:
        S.assert_no_leak(sp, meta)
        assert len(sp.train) and len(sp.test), f"{sp.name} has an empty side"


def test_within_subject_tests_only_trials_6_and_7(meta):
    for sp in S.within_subject(meta):
        assert set(meta["trial"][sp.test].tolist()) == set(S.TEST_TRIALS)
        assert set(meta["trial"][sp.val].tolist()) == {S.VAL_TRIAL_WITHIN}
        assert not set(meta["trial"][sp.train].tolist()) & {*S.TEST_TRIALS, S.VAL_TRIAL_WITHIN}
        assert len(set(meta["subject"][sp.test].tolist())) == 1
        assert len(set(meta["session"][sp.test].tolist())) == 1


def test_cross_day_tests_only_session_3(meta):
    for sp in S.cross_day(meta):
        assert set(meta["session"][sp.test].tolist()) == {S.TEST_SESSION}
        assert set(meta["session"][sp.train].tolist()) <= set(S.TRAIN_SESSIONS)
        assert set(meta["trial"][sp.val].tolist()) == {S.VAL_TRIAL_CROSS_DAY}


def test_cross_subject_every_subject_tested_exactly_once(meta):
    made = S.cross_subject(meta)
    seen: list[int] = []
    for sp in made:
        seen += np.unique(meta["subject"][sp.test]).tolist()
    assert sorted(seen) == np.unique(meta["subject"]).tolist()
    sizes = [len(np.unique(meta["subject"][sp.test])) for sp in made]
    assert max(sizes) - min(sizes) <= 1, "folds are unbalanced"


def test_cross_subject_is_deterministic(meta):
    for x, y in zip(S.cross_subject(meta), S.cross_subject(meta), strict=True):
        assert np.array_equal(x.test, y.test)


def test_assert_no_leak_catches_a_leak(meta):
    sp = S.within_subject(meta)[0]
    # Same window index on both sides.
    leaky = S.Split(sp.kind, sp.name, np.concatenate([sp.train, sp.test[:1]]), sp.val, sp.test)
    with pytest.raises(AssertionError):
        S.assert_no_leak(leaky, meta)
    # Different window index, but from a trial that is otherwise in the test set.
    s0, d0 = meta["subject"][sp.test[0]], meta["session"][sp.test[0]]
    trial6 = np.flatnonzero((meta["subject"] == s0) & (meta["session"] == d0) & (meta["trial"] == 6))
    sneaky = S.Split(
        sp.kind, sp.name,
        np.concatenate([sp.train, trial6[:1]]), sp.val, np.setdiff1d(sp.test, trial6[:1]),
    )
    with pytest.raises(AssertionError):
        S.assert_no_leak(sneaky, meta)


@pytest.mark.skipif(not REAL_META.exists(), reason="real metadata not built yet")
@pytest.mark.parametrize("kind", list(S.ALL))
def test_no_leak_on_real_data(kind):
    import pyarrow.parquet as pq

    table = pq.read_table(REAL_META)
    meta = {c: table[c].to_numpy() for c in ("subject", "session", "gesture", "trial")}
    made = S.make(kind, meta)
    assert made
    for sp in made:
        S.assert_no_leak(sp, meta)
