"""How we divide the data into "learn from this" and "test on that".

What goes in
    The metadata table that windows.py produced: one row per window, with the
    columns subject, session, gesture, trial. The signal itself is not needed
    here: deciding which rows are the exam and which are the textbook only
    needs the labels on the rows, not their contents.

What comes out
    A list of Split objects. Each one says: these window indices are for
    training, these for validation (checking progress while training), and
    these for the final test. The three sets never overlap, and the test set
    is chosen by a rule the training set cannot cheat on.

Why this file exists at all
    EMG windows overlap in time and come from the same person doing the same
    movement seven times in a row. If you shuffled all windows and took 20%
    at random as the test, almost every test window would have a near-twin in
    the training set: same person, same day, same repetition, a few
    milliseconds apart. A model that simply memorised would score 99% and the
    number would mean nothing. So we never split at the window level. We
    split by a *group* the model has not seen:

    within_subject   learns from a person's repetitions 1-4 (5 = validation)
                     and is tested on repetitions 6-7, same person, same day.
                     "Can it recognise your gestures once calibrated to you?"
    cross_day        learns from days 1 and 2, tested on day 3 of the same
                     person. "Does it still work when you put the sensor
                     back on tomorrow?"
    cross_subject    learns from some people, tested on people it has never
                     seen. "Does it work out of the box for a stranger?"

    Validation sets are carved out of the *training* groups, never the test
    groups, so choosing when to stop training cannot peek at the exam.

What would go wrong without it
    Every result in this project would be inflated and unrepeatable, and
    anyone who knows EMG would spot it in a minute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emgdec.config import SEED

# The seven repetitions of each gesture, in recording order.
TRAIN_TRIALS = (1, 2, 3, 4, 5)
VAL_TRIAL_WITHIN = 5  # held back from TRAIN_TRIALS for early stopping
TEST_TRIALS = (6, 7)

TRAIN_SESSIONS = (1, 2)
VAL_TRIAL_CROSS_DAY = 7  # trial 7 of the training sessions is validation
TEST_SESSION = 3

N_FOLDS_CROSS_SUBJECT = 5
N_VAL_SUBJECTS = 4

Meta = dict[str, np.ndarray]  # column name -> 1-D array; all the same length


@dataclass(frozen=True)
class Split:
    """One train / validation / test division.

    kind   'within_subject' | 'cross_day' | 'cross_subject'
    name   human-readable id, e.g. 's07_d2' or 'fold3'
    train, val, test   int64 arrays of window indices into X / meta
    """

    kind: str
    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    split_protocol: str = "v1"

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _require(meta: Meta) -> None:
    for col in ("subject", "session", "gesture", "trial"):
        if col not in meta:
            raise KeyError(f"meta is missing column {col!r}")
    n = len(meta["subject"])
    if any(len(meta[c]) != n for c in ("session", "gesture", "trial")):
        raise ValueError("meta columns have different lengths")


def within_subject(meta: Meta) -> list[Split]:
    """One split per (subject, session): trials 1-4 train, 5 val, 6-7 test."""
    _require(meta)
    subj, sess, trial = meta["subject"], meta["session"], meta["trial"]
    out: list[Split] = []
    for s in np.unique(subj):
        for d in np.unique(sess[subj == s]):
            here = (subj == s) & (sess == d)
            train = np.flatnonzero(here & np.isin(trial, TRAIN_TRIALS) & (trial != VAL_TRIAL_WITHIN))
            val = np.flatnonzero(here & (trial == VAL_TRIAL_WITHIN))
            test = np.flatnonzero(here & np.isin(trial, TEST_TRIALS))
            if len(train) and len(test):
                out.append(Split("within_subject", f"s{int(s):02d}_d{int(d)}", train, val, test))
    return out


def cross_day(meta: Meta) -> list[Split]:
    """One split per subject: sessions 1-2 train (their trial 7 = val), session 3 test."""
    _require(meta)
    subj, sess, trial = meta["subject"], meta["session"], meta["trial"]
    out: list[Split] = []
    for s in np.unique(subj):
        here = subj == s
        in_train_days = here & np.isin(sess, TRAIN_SESSIONS)
        train = np.flatnonzero(in_train_days & (trial != VAL_TRIAL_CROSS_DAY))
        val = np.flatnonzero(in_train_days & (trial == VAL_TRIAL_CROSS_DAY))
        test = np.flatnonzero(here & (sess == TEST_SESSION))
        if len(train) and len(test):
            out.append(Split("cross_day", f"s{int(s):02d}", train, val, test))
    return out


def cross_subject(meta: Meta, n_folds: int = N_FOLDS_CROSS_SUBJECT, seed: int = SEED,
                  protocol: str = "v1") -> list[Split]:
    """K folds over *people*. Each person is a test subject in exactly one fold.

    Subjects are shuffled once with a fixed seed and dealt into folds like
    cards, so fold membership is reproducible and balanced (±1 subject).
    Within each fold, N_VAL_SUBJECTS of the training subjects are held out
    whole as validation.
    """
    _require(meta)
    subj = meta["subject"]
    subjects = np.unique(subj)
    order = np.random.default_rng(seed).permutation(subjects)
    fold_of = {int(s): i % n_folds for i, s in enumerate(order)}

    out: list[Split] = []
    used_val: list[set] = []
    for k in range(n_folds):
        test_subjects = [s for s in order if fold_of[int(s)] == k]
        pool = [s for s in order if fold_of[int(s)] != k]
        if protocol == "v2":
            # v2: seeded-per-fold draw with the constraint that no subject is
            # a validation subject in more than one fold (v1 always took the
            # first four of the pool — the fixed-val-subject defect).
            while True:
                r = np.random.default_rng(seed + 1000 + k)
                cand = list(r.choice(pool, size=N_VAL_SUBJECTS, replace=False))
                if not any(c in v for v in used_val for c in cand):
                    break
            used_val.append(set(cand))
            val_subjects = cand
            train_subjects = [s for s in pool if s not in val_subjects]
        else:
            val_subjects, train_subjects = pool[:N_VAL_SUBJECTS], pool[N_VAL_SUBJECTS:]
        out.append(
            Split(
                "cross_subject",
                f"fold{k}",
                np.flatnonzero(np.isin(subj, train_subjects)),
                np.flatnonzero(np.isin(subj, val_subjects)),
                np.flatnonzero(np.isin(subj, test_subjects)),
                split_protocol=protocol,
            )
        )
    return out


ALL = {"within_subject": within_subject, "cross_day": cross_day, "cross_subject": cross_subject}

# The columns that define "the same group" for each split kind. Two windows
# sharing all of these columns are near-twins and must stay on one side.
GROUP_COLS = {
    "within_subject": ("subject", "session", "gesture", "trial"),
    "cross_day": ("subject", "session"),
    "cross_subject": ("subject",),
}


def make(kind: str, meta: Meta, protocol: str = "v1") -> list[Split]:
    """Dispatch by name. Raises KeyError for an unknown kind."""
    if kind == "cross_subject" and protocol == "v2":
        return cross_subject(meta, protocol="v2")
    return ALL[kind](meta)


# ---------------------------------------------------------------------------
# Leak checks. They are also run as tests, but keeping them next to the split
# code lets a training script assert them cheaply before spending an hour on
# the GPU.
# ---------------------------------------------------------------------------


def _groups(meta: Meta, idx: np.ndarray, cols: tuple[str, ...]) -> set[tuple[int, ...]]:
    return set(zip(*(meta[c][idx].astype(int).tolist() for c in cols), strict=True))


def assert_no_leak(split: Split, meta: Meta) -> None:
    """Raise AssertionError if training information can touch the test set.

    1. train / val / test index sets are pairwise disjoint.
    2. The group that defines the split kind (GROUP_COLS) never appears on
       both the train side and the test side, nor on val and test.
    3. For cross_subject, train and val are also disjoint by subject.
    """
    _require(meta)
    tr, va, te = set(split.train.tolist()), set(split.val.tolist()), set(split.test.tolist())
    assert not (tr & te), f"{split.name}: train and test share window indices"
    assert not (tr & va), f"{split.name}: train and val share window indices"
    assert not (va & te), f"{split.name}: val and test share window indices"

    cols = GROUP_COLS[split.kind]
    g_train, g_val, g_test = (_groups(meta, ix, cols) for ix in (split.train, split.val, split.test))
    assert not (g_train & g_test), f"{split.name}: a {cols} group is in both train and test"
    assert not (g_val & g_test), f"{split.name}: a {cols} group is in both val and test"
    if split.kind == "cross_subject":
        assert not (g_train & g_val), f"{split.name}: a subject is in both train and val"
