"""Predictions dump: every saved checkpoint's logits, frozen to disk.

For each .pt under artifacts/models (and each re-fit LDA for protocol_v1)
writes predictions/<model>/<config>/<kind>/<name>.npz holding, for BOTH
val and test windows: window index, subject/session/gesture/trial,
window-in-trial, predicted class, logits (float16). Before writing, the
reconstructed test confusion matrix must equal the committed run JSON's
exactly — a checkpoint that cannot reproduce its own confusion is a bug,
not a result, and we stop.

Output root: artifacts/experiments/protocol_v1_analysis/predictions/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from emgdec import metrics, splits
from emgdec.baselines import CONFIG_COLS, _load_meta
from emgdec.config import N_GESTURES
from emgdec.evaluate import exp_runs
from emgdec.models import CHANNELS, TinyCNN
from emgdec.train import MODELS_DIR, _open_X

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "experiments" / "protocol_v1_analysis" / "predictions"


def _dump_one(X, meta, split, chans, model, out: Path, committed_cm):
    recs = []
    for flag, idx in ((0, split.val), (1, split.test)):
        idx = np.sort(np.asarray(idx))
        if not len(idx):
            continue
        x = torch.from_numpy(
            np.asarray(X[idx][:, chans, :], dtype=np.float32))
        with torch.no_grad():
            logits = model(x).numpy()
        recs.append((flag, idx, logits))
    all_idx = np.concatenate([r[1] for r in recs])
    logits = np.concatenate([r[2] for r in recs])
    flags = np.concatenate([np.full(len(r[1]), r[0]) for r in recs])
    pred = logits.argmax(1)
    cm = metrics.confusion(
        meta["gesture"][all_idx[flags == 1]].astype(int) - 1,
        pred[flags == 1], N_GESTURES)
    assert np.array_equal(cm, committed_cm), \
        f"reconstructed confusion != committed for {out.stem}"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, idx=all_idx, flag=flags,
        subject=meta["subject"][all_idx], session=meta["session"][all_idx],
        gesture=meta["gesture"][all_idx], trial=meta["trial"][all_idx],
        win_in_trial=(all_idx % 35).astype(np.int8),
        pred=pred, logits=logits.astype(np.float16))


def main() -> None:
    meta = _load_meta()
    X, _ = _open_X()
    n = 0
    for pt in sorted(MODELS_DIR.rglob("*.pt")):
        rel = pt.relative_to(MODELS_DIR).with_suffix("")
        model, config, kind, name = rel.parts[0], rel.parts[1], rel.parts[2], rel.parts[3]
        committed = json.loads(
            (exp_runs("protocol_v1") / model / config / kind
             / f"{name}.json").read_text())
        split = next(s for s in splits.make(kind, meta) if s.name == name)
        chans = np.asarray(list(CHANNELS[config]))
        net = TinyCNN(len(chans), N_GESTURES)
        net.load_state_dict(torch.load(pt, map_location="cpu",
                                       weights_only=True))
        net.eval()
        _dump_one(X, meta, split, chans, net, OUT / f"{rel}.npz",
                  np.asarray(committed["confusion"]))
        n += 1
        print(f"{rel}: ok", flush=True)
    print(f"done: {n} npz (confusion equality asserted for all)")


def dump_lda() -> int:
    """Re-fit v1 LDA (both, all kinds) and dump predict_proba; the v1
    commits are reproduced by construction (same seeds, deterministic
    solver) — asserted per split before writing."""
    import pyarrow.parquet as pq
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis)
    from emgdec.features import FEATURES_F32, MEMMAP_DIR, N_FEATURES
    from emgdec.config import N_CHANNELS, SEED

    meta = _load_meta()
    shape = json.loads((MEMMAP_DIR / "shape.json").read_text())
    F = np.memmap(FEATURES_F32, dtype=np.float32, mode="r",
                  shape=(shape["N"], N_CHANNELS * N_FEATURES))
    cols = (np.arange(N_CHANNELS)[:, None] * N_FEATURES
            + np.arange(N_FEATURES)[None, :]).ravel()
    n = 0
    for kind in splits.ALL:
        for split in splits.make(kind, meta):
            committed = json.loads(
                (exp_runs("protocol_v1") / "lda" / "both" / kind
                 / f"{split.name}.json").read_text())
            Xtr = np.asarray(F[np.asarray(split.train)][:, cols],
                             dtype=np.float32)
            mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
            clf = LinearDiscriminantAnalysis(solver="lsqr",
                                             shrinkage="auto")
            clf.fit((Xtr - mu) / sd, meta["gesture"][split.train] - 1)
            del Xtr
            recs = []
            for flag, idx in ((0, split.val), (1, split.test)):
                idx = np.sort(np.asarray(idx))
                if not len(idx):
                    continue
                x = np.asarray(F[idx][:, cols], dtype=np.float32)
                recs.append((flag, idx, clf.predict_proba((x - mu) / sd)))
            all_idx = np.concatenate([r[1] for r in recs])
            proba = np.concatenate([r[2] for r in recs])
            flags = np.concatenate([np.full(len(r[1]), r[0]) for r in recs])
            pred = proba.argmax(1)
            cm = metrics.confusion(
                meta["gesture"][all_idx[flags == 1]].astype(int) - 1,
                pred[flags == 1], N_GESTURES)
            assert np.array_equal(cm, np.asarray(committed["confusion"])), \
                f"LDA {kind}/{split.name}: confusion mismatch — stopping"
            out = OUT / "lda" / "both" / kind / f"{split.name}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out, idx=all_idx, flag=flags,
                subject=meta["subject"][all_idx],
                session=meta["session"][all_idx],
                gesture=meta["gesture"][all_idx],
                trial=meta["trial"][all_idx],
                win_in_trial=(all_idx % 35).astype(np.int8),
                pred=pred, logits=np.log(proba + 1e-12).astype(np.float16))
            n += 1
        print(f"lda/{kind}: done", flush=True)
    print(f"done: {n} lda npz; all confusions match committed v1")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--lda":
        dump_lda()
    else:
        main()
