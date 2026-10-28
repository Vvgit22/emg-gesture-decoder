"""Cutting every trial into overlapping windows, stored on disk.

In:  the raw mirror (via `data.iter_trials`) and the cleaning step
     (`preprocess.preprocess` at the measured mains frequency).
Out: `data/memmap/X.f16` — np.memmap float16, shape [N, 28, WIN], channels
     ordered forearm-16 then wrist-12 (see config.KEEP); and
     `data/memmap/meta.parquet` — one row per window with columns
     idx, subject, session, gesture, trial, start_sample. Also
     `data/memmap/shape.json` {N, C, W}.
If skipped: models would have to re-clean ~9 GB of raw data every run and
     window/label alignment would be re-derived each time — the parquet
     metadata is what makes splits, leakage tests and reproducibility
     cheap.

Each 5 s trial drops its first TRIAL_DROP_S seconds (reaction ramp), then
contributes floor((kept - WIN)/STEP) + 1 windows. Processing is streamed
subject-by-subject so peak memory stays far below the dataset size.

CLI: `uv run python -m emgdec.windows [--subjects 1,2] [--out DIR]
      [--mains 60]` — `--subjects` selects a subset for quick tests.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import FS_OUT, KEEP, N_CHANNELS, STEP, TRIAL_DROP_S, WIN
from .data import RAW_DIR, is_complete, iter_trials, read_trial

REPO = Path(__file__).resolve().parents[2]
MEMMAP_DIR = REPO / "data" / "memmap"


def _load_mains() -> float:
    """Mains frequency decided by preprocess.mains_check (DECISIONS.md)."""
    cfg = REPO / "artifacts" / "mains.json"
    if cfg.exists():
        return float(json.loads(cfg.read_text())["mains"])
    return 60.0  # default until the check runs; recorded in DECISIONS


def windows_per_trial(n_out_samples: int, drop_s: float = TRIAL_DROP_S) -> int:
    kept = n_out_samples - int(drop_s * FS_OUT)
    return max(0, (kept - WIN) // STEP + 1)


def build(subjects: list[int] | None = None, out_dir: Path = MEMMAP_DIR,
          mains: float | None = None, raw_dir: Path = RAW_DIR,
          preprocess: str = "offline") -> dict:
    """Build the memmap. Returns summary dict {N, C, W, bytes, seconds}."""
    from .preprocess import preprocess as _offline
    from .preprocess_causal import CausalPreprocessor

    mains = _load_mains() if mains is None else mains
    import wfdb
    causal = CausalPreprocessor() if preprocess == "causal" else None

    trials = [t for t in iter_trials(raw_dir)
              if (subjects is None or t[0] in subjects)
              and is_complete(t[4])]
    # Two passes: count windows first (memmap needs its final size up
    # front). Headers alone give sample counts — no need to read 9 GB twice.
    counts: list[int] = []
    total = 0
    for *_, path in trials:
        sig_len = wfdb.rdheader(str(path)).sig_len
        n = windows_per_trial(sig_len // 2)
        counts.append(n)
        total += n
    out_dir.mkdir(parents=True, exist_ok=True)
    X = np.memmap(out_dir / "X.f16", dtype=np.float16, mode="w+",
                  shape=(total, N_CHANNELS, WIN))
    meta_rows = {k: [] for k in
                 ("idx", "subject", "session", "gesture", "trial", "start_sample")}
    t0 = time.time()
    idx = 0
    for (subject, session, gesture, trial, path), n_expected in zip(trials, counts):
        if n_expected == 0:
            continue
        x, _, _ = read_trial(path)
        x = causal.reset() or causal.process(x) if causal else _offline(x, mains)          # [n_out, 32] at FS_OUT
        x = x[:, KEEP]                    # [n_out, 28]
        drop = int(TRIAL_DROP_S * FS_OUT)
        for i in range(n_expected):
            start = drop + i * STEP
            X[idx] = x[start:start + WIN].T.astype(np.float16)  # [28, WIN]
            meta_rows["idx"].append(idx)
            meta_rows["subject"].append(subject)
            meta_rows["session"].append(session)
            meta_rows["gesture"].append(gesture)
            meta_rows["trial"].append(trial)
            meta_rows["start_sample"].append(start)
            idx += 1
    X.flush()
    table = pa.table({
        "idx": pa.array(meta_rows["idx"], type=pa.int64()),
        "subject": pa.array(meta_rows["subject"], type=pa.int16()),
        "session": pa.array(meta_rows["session"], type=pa.int8()),
        "gesture": pa.array(meta_rows["gesture"], type=pa.int8()),
        "trial": pa.array(meta_rows["trial"], type=pa.int8()),
        "start_sample": pa.array(meta_rows["start_sample"], type=pa.int32()),
    })
    pq.write_table(table, out_dir / "meta.parquet")
    shape = {"N": total, "C": N_CHANNELS, "W": WIN}
    (out_dir / "shape.json").write_text(json.dumps(shape))
    secs = time.time() - t0
    nbytes = total * N_CHANNELS * WIN * 2
    print(f"build: N={total} C={N_CHANNELS} W={WIN} "
          f"bytes={nbytes / 1e9:.2f} GB wall={secs:.0f}s "
          f"mains={mains} subjects={subjects or 'all'}")
    return {**shape, "bytes": nbytes, "seconds": secs}


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=str, default=None,
                    help="comma list, e.g. 1,2 — default: all")
    ap.add_argument("--out", type=str, default=str(MEMMAP_DIR))
    ap.add_argument("--mains", type=float, default=None)
    ap.add_argument("--preprocess", choices=["offline", "causal"],
                    default="offline")
    args = ap.parse_args()
    subs = [int(s) for s in args.subjects.split(",")] if args.subjects else None
    build(subjects=subs, out_dir=Path(args.out), mains=args.mains,
          preprocess=args.preprocess)


if __name__ == "__main__":
    _main()
