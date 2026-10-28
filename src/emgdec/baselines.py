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
from sklearn.ensemble import HistGradientBoostingClassifier

from emgdec import metrics, splits
from emgdec.config import N_CHANNELS, N_GESTURES, SEED
from emgdec.evaluate import RunRecord, exp_runs, save_run
from emgdec.provenance import collect as prov_collect
from emgdec.features import FEATURES_F32, MEMMAP_DIR, N_FEATURES

CONFIG_COLS = {
    "forearm": list(range(0, 16)),
    "wrist": list(range(16, N_CHANNELS)),
    "both": list(range(N_CHANNELS)),
}

MODEL = "lda"
LDA_PARAMS = {"solver": "lsqr", "shrinkage": "auto", "features": N_FEATURES}
HGB_PARAMS = {"max_iter": 200, "learning_rate": 0.1,
              "early_stopping": False, "features": N_FEATURES}


def _fit(model: str, Xtr, ytr, seed: int):
    """Construct + fit the requested classifier."""
    if model == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=HGB_PARAMS["max_iter"],
            learning_rate=HGB_PARAMS["learning_rate"],
            early_stopping=False, random_state=seed).fit(Xtr, ytr)
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, ytr)


def _load_meta() -> dict[str, np.ndarray]:
    df = pq.read_table(MEMMAP_DIR / "meta.parquet").to_pandas()
    return {c: df[c].to_numpy() for c in ("subject", "session", "gesture", "trial")}


def _feature_columns(config: str) -> np.ndarray:
    """Channel-major layout: channel c occupies cols c*11 … c*11+10."""
    chans = np.asarray(CONFIG_COLS[config])
    return (chans[:, None] * N_FEATURES + np.arange(N_FEATURES)[None, :]).ravel()


def run_split(F: np.memmap, meta: dict[str, np.ndarray], split, cols: np.ndarray,
              config: str, model: str = MODEL, seed: int = SEED) -> RunRecord:
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
        try:
            clf = _fit(model, Xtr, ytr, seed)
        except UserWarning as w:
            print(f"  WARN {split.name}: {w}", flush=True)
            clf = _fit(model, Xtr, ytr, seed)
    train_seconds = time.time() - t0
    cm = metrics.confusion(yte, clf.predict(Xte), N_GESTURES)
    params = dict(HGB_PARAMS if model == "hgb" else LDA_PARAMS)
    prov = prov_collect(seed=seed, opt=params, device="cpu", split=split)
    return RunRecord(
        model=model, config=config, kind=split.kind, name=split.name,
        confusion=cm.tolist(),
        n_train=len(split.train), n_val=len(split.val), n_test=len(split.test),
        train_seconds=train_seconds, seed=seed, params=params,
        provenance=prov)


def run(kind: str, config: str, model: str = MODEL,
        experiment: str = "dev", limit: int | None = None) -> int:
    meta = _load_meta()
    shape = json.loads((MEMMAP_DIR / "shape.json").read_text())
    F = np.memmap(FEATURES_F32, dtype=np.float32, mode="r",
                  shape=(shape["N"], N_CHANNELS * N_FEATURES))
    runs_dir = exp_runs(experiment)
    kinds = list(splits.ALL) if kind == "all" else [kind]
    configs = list(CONFIG_COLS) if config == "all" else [config]
    n = 0
    for k in kinds:
        for cfg in configs:
            cols = _feature_columns(cfg)
            for i, split in enumerate(splits.make(k, meta)):
                if limit is not None and i >= limit:
                    break
                rec = run_split(F, meta, split, cols, cfg, model=model)
                save_run(rec, runs_dir)
                n += 1
            print(f"{model}/{cfg}/{k}: {n} runs so far", flush=True)
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="all",
                    choices=["within_subject", "cross_day", "cross_subject", "all"])
    ap.add_argument("--config", default="all",
                    choices=["forearm", "wrist", "both", "all"])
    ap.add_argument("--model", default="lda", choices=["lda", "hgb"])
    ap.add_argument("--experiment", default="dev")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    n = run(a.kind, a.config, model=a.model, experiment=a.experiment,
            limit=a.limit)
    print(f"done: {n} runs")


if __name__ == "__main__":
    main()
