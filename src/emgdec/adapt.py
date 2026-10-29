"""Per-user adaptation: how much of a new person's data helps?

Question: a model trained on 42 strangers gets ~80% on person 43.
How many repetitions of each gesture from that person, recorded once on
day 1, lift them toward the personalised level?

Protocol (fixed a priori, see DECISIONS):
  base     = offline_v1b seed-0 cross_subject fold model whose TEST set
             contains the subject (the base never saw them)
  adapt    = subject s, session 1, trials 1..n (all 17 gestures)
             n in {0,1,2,3}; n=0 is the unadapted control
  test     = subject s, sessions 2+3 all trials (calibrate day 1, use on
             days 8 and 29 — the realistic case); same-day control =
             session 1 trials 6,7
  recipe   = base weights, normalisation buffers kept, AdamW lr 5e-4
             wd 1e-2, batch 64, exactly 8 epochs, no val (with n=1 there
             is nothing to validate on — fixed, not tuned), all layers,
             seed = SEED + s

Leak guards before every fit: adapt/test disjoint at (session,trial)
group level and subject s sits in the fold's test subjects.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from emgdec import metrics, splits
from emgdec.baselines import _load_meta
from emgdec.config import N_GESTURES, SEED
from emgdec.evaluate import RunRecord, save_run
from emgdec.models import TinyCNN
from emgdec.provenance import collect as prov_collect
from emgdec.train import (MODELS_DIR, _load_windows, _open_X)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "experiments" / "adapt_v1"
ADAPT_SEED_OFFSET = 0
EPOCHS, BATCH, LR, WD = 8, 64, 5e-4, 1e-2


def subject_fold(subject: int, meta) -> int:
    """Fold index whose test set contains this subject (asserted)."""
    for k, sp in enumerate(splits.make("cross_subject", meta)):
        subs = np.unique(meta["subject"][sp.test])
        if subject in subs:
            return k
    raise AssertionError(f"subject {subject} in no fold's test set")


def assert_adapt_leak(adapt_idx, test_idx, meta, subject, fold_subs_test):
    """Adaptation and test must share no (session, trial) group; the
    subject must be a test subject of the fold used."""
    assert subject in fold_subs_test, "subject not a fold test subject"
    a = set(zip(meta["session"][adapt_idx].tolist(),
                meta["trial"][adapt_idx].tolist()))
    t = set(zip(meta["session"][test_idx].tolist(),
                meta["trial"][test_idx].tolist()))
    assert not (a & t), "adapt/test share a (session,trial) group"
    assert set(adapt_idx).isdisjoint(set(test_idx)), "index overlap"


def fine_tune(model: TinyCNN, Xa, ya, *, epochs: int = EPOCHS,
              batch: int = BATCH, lr: float = LR, wd: float = WD,
              seed: int, device: torch.device) -> int:
    """Fixed recipe: all layers, no val, exactly `epochs` epochs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    x = torch.from_numpy(Xa.astype(np.float32))
    y = torch.from_numpy(np.asarray(ya))
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=gen).tolist()
        for lo in range(0, len(x), batch):
            sel = order[lo:lo + batch]
            opt.zero_grad()
            loss = lossf(model(x[sel].to(device)), y[sel].to(device))
            loss.backward()
            opt.step()
    return epochs


def _acc(model, X, y, device, batch: int = 2048):
    prs = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch].astype(np.float32)).to(device)
        with torch.no_grad():
            prs.append(model(xb).argmax(1).cpu().numpy())
    pr = np.concatenate(prs)
    return metrics.confusion(y.astype(int) - 1, pr, N_GESTURES)


def run_subject(X, meta, subject: int, *, ns=(0, 1, 2, 3),
                device=None) -> list[RunRecord]:
    device = device or torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    k = subject_fold(subject, meta)
    sp = splits.make("cross_subject", meta)[k]
    subs_test = np.unique(meta["subject"][sp.test])
    chans = np.arange(28)
    m = (meta["subject"] == subject)
    adapt_sets = {n: np.flatnonzero(
        m & (meta["session"] == 1) & np.isin(meta["trial"], range(1, n + 1))
    ) for n in ns if n > 0}
    test_late = np.flatnonzero(m & np.isin(meta["session"], [2, 3]))
    test_same = np.flatnonzero(
        m & (meta["session"] == 1) & np.isin(meta["trial"], [6, 7]))
    pt = (MODELS_DIR / "offline_v1b" / "cnn" / "both" / "cross_subject"
          / f"fold{k}.pt")
    recs = []
    for n in ns:
        net = TinyCNN(28, N_GESTURES)
        net.load_state_dict(torch.load(pt, map_location="cpu",
                                       weights_only=True))
        net.to(device).eval()
        t0 = time.time()
        if n > 0:
            ai = adapt_sets[n]
            assert_adapt_leak(ai, np.concatenate([test_late, test_same]),
                              meta, subject, subs_test)
            Xa = _load_windows(X, ai, chans, 256)
            ya = meta["gesture"][ai].astype(int) - 1
            net.train()
            fine_tune(net, Xa, ya, seed=SEED + subject, device=device)
            net.eval()
        else:
            assert subject in subs_test
        X_late = _load_windows(X, test_late, chans, 256)
        cm = _acc(net, X_late, meta["gesture"][test_late], device)
        X_same = _load_windows(X, test_same, chans, 256)
        cm_same = _acc(net, X_same, meta["gesture"][test_same], device)
        elapsed = time.time() - t0
        prov = prov_collect(seed=SEED + subject,
                            opt={"lr": LR, "wd": WD, "batch": BATCH,
                                 "epochs": EPOCHS},
                            device=str(device), split=sp,
                            extra={"base_fold": k, "n_adapt_trials": n,
                                   "protocol": "adapt_v1"})
        recs.append(RunRecord(
            model="adapt", config="both", kind="cross_subject",
            name=f"s{subject:02d}_n{n}", confusion=cm.tolist(),
            n_train=0 if n == 0 else len(adapt_sets[n]),
            n_val=0, n_test=len(test_late),
            train_seconds=elapsed, seed=SEED + subject,
            params={"n": n, "epochs": EPOCHS,
                    "confusion_same_day": cm_same.tolist(),
                    "n_same_day": len(test_same)},
            provenance=prov))
    return recs


def main() -> None:
    meta = _load_meta()
    X, _ = _open_X()
    runs_dir = OUT / "runs"
    for s in sorted(np.unique(meta["subject"]).tolist()):
        for rec in run_subject(X, meta, s):
            save_run(rec, runs_dir)
        print(f"s{s}: done", flush=True)
    print("adapt done")


if __name__ == "__main__":
    main()
