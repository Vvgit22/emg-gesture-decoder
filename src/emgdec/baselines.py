"""The classical baseline: LDA on hand-crafted EMG features.

What goes in
    `data/memmap/features.f32` (built by `python -m emgdec.features --cache`),
    the meta table, and a split kind + channel config from the CLI.

What comes out
    One `RunRecord` per split under `artifacts/runs/lda/<config>/<kind>/`,
    written through `evaluate.save_run` — the only writer of results.

How it works
    For each split the features are standardised (z-scored per column)
    using means and standard deviations computed **on the training indices
    only** — this is the per-channel standardisation that preprocess.py
    deliberately defers, so statistics from test windows can never leak
    into the pipeline. Then `LinearDiscriminantAnalysis(solver='lsqr',
    shrinkage='auto')` (Ledoit-Wolf shrinkage, which regularises the
    covariance and is the standard choice when features outnumber
    examples) is fit on train and scored on test.

    LDA needs no validation set — it has no epochs to early-stop — so
    `split.val` is unused here; n_val is still recorded for the audit
    trail. `splits.assert_no_leak` runs before every fit.

Channel configs select feature columns, channel-major:
    forearm = channels 0-15 → cols 0..175;  wrist = channels 16-27 →
    cols 176..307;  both = all 308.

CLI: `uv run python -m emgdec.baselines
      --kind within_subject|cross_day|cross_subject|all
      --config forearm|wrist|both|all`
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pyarrow.parquet as pq
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from emgdec import metrics, splits
from emgdec.config import N_CHANNELS, N_GESTURES, SEED
from emgdec.evaluate import RUNS_DIR, RunRecord, save_run
from emgdec.features import FEATURES_F32, MEMMAP_DIR, N_FEATURES

CONFIG_COLS = {
    "forearm": list(range(0, 16)),
    "wrist": list(range(16, N_CHANNELS)),
    "both": list(range(N_CHANNELS)),
}

MODEL = "lda"
LDA_PARAMS = {"solver": "lsqr", "shrinkage": "auto", "features": N_FEATURES}


def _load_meta() -> dict[str, np.ndarray]:
    df = pq.read_table(MEMMAP_DIR / "meta.parquet").to_pandas()
    return {c: df[c].to_numpy() for c in ("subject", "session", "gesture", "trial")}


def _feature_columns(config: str) -> np.ndarray:
    """Channel-major layout: channel c occupies cols c*11 … c*11+10."""
    chans = np.asarray(CONFIG_COLS[config])
    return (chans[:, None] * N_FEATURES + np.arange(N_FEATURES)[None, :]).ravel()


def run_split(F: np.memmap, meta: dict[str, np.ndarray], split, cols: np.ndarray,
              config: str) -> RunRecord:
    splits.assert_no_leak(split, meta)
    Xtr = np.asarray(F[np.asarray(split.train)][:, cols], dtype=np.float32)
    Xte = np.asarray(F[np.asarray(split.test)][:, cols], dtype=np.float32)
    ytr = meta["gesture"][split.train].astype(int) - 1
    yte = meta["gesture"][split.test].astype(int) - 1

    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        try:
            clf.fit(Xtr, ytr)
        except UserWarning as w:
            print(f"  WARN {split.name}: {w}", flush=True)
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            clf.fit(Xtr, ytr)
    train_seconds = time.time() - t0
    cm = metrics.confusion(yte, clf.predict(Xte), N_GESTURES)
    return RunRecord(
        model=MODEL, config=config, kind=split.kind, name=split.name,
        confusion=cm.tolist(),
        n_train=len(split.train), n_val=len(split.val), n_test=len(split.test),
        train_seconds=train_seconds, seed=SEED, params=dict(LDA_PARAMS))


def run(kind: str, config: str) -> int:
    meta = _load_meta()
    shape = json.loads((MEMMAP_DIR / "shape.json").read_text())
    F = np.memmap(FEATURES_F32, dtype=np.float32, mode="r",
                  shape=(shape["N"], N_CHANNELS * N_FEATURES))
    kinds = list(splits.ALL) if kind == "all" else [kind]
    configs = list(CONFIG_COLS) if config == "all" else [config]
    n = 0
    for k in kinds:
        for cfg in configs:
            cols = _feature_columns(cfg)
            for split in splits.make(k, meta):
                rec = run_split(F, meta, split, cols, cfg)
                save_run(rec, RUNS_DIR)
                n += 1
            print(f"{cfg}/{k}: {n} runs so far", flush=True)
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="all",
                    choices=["within_subject", "cross_day", "cross_subject", "all"])
    ap.add_argument("--config", default="all",
                    choices=["forearm", "wrist", "both", "all"])
    a = ap.parse_args(argv)
    n = run(a.kind, a.config)
    print(f"done: {n} runs")


if __name__ == "__main__":
    main()
