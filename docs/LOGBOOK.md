# LOGBOOK

Dated entries of what was run, how long it took, what came out, and what
broke. Written as the work happens.

## 2026-10-26 — Scaffold + download started

- `wget` was not installed on this Mac (first two launch attempts failed
  silently via nohup — log line `nohup: wget: No such file or directory`).
  Installed with `brew install wget`. Real download started **15:26:55**,
  `wget -r -N -c -np -nH --cut-dirs=3 -P data/raw
  https://physionet.org/files/grabmyo/1.1.0/` (~9.1 GB expected).
- Repo location changed mid-task (lead instruction): from
  `~/Downloads/Cover Letters and CVs/` to `~/Projects/`. The first
  scaffold attempt (uv init + `uv add` at the old location) was deleted;
  nothing was committed.
- `uv init --python 3.12 --package`, pinned deps (all releases several
  weeks old or more): numpy 2.3.2, scipy 1.16.1, scikit-learn 1.7.1,
  torch 2.8.0, wfdb 4.3.0, pyarrow 21.0.0, onnx 1.19.0, onnxruntime
  1.22.0, matplotlib 3.10.3, pytest 8.4.1, ruff 0.12.9. `uv sync` clean.
- `python -c "import torch; print(torch.backends.mps.is_available())"`
  → **True** (torch 2.8.0, Apple M5).
- File layout confirmed against the first downloaded records:
  `Session{s}/session{s}_participant{p}/session{s}_participant{p}_gesture{g}_trial{t}.{dat,hea}`;
  each trial 10240 samples × 32 ch at 2048 Hz (5 s), units mV.
  Header channel names `F1..F16, U1, W1..W6, U2, U3, W7..W12, U4` verify
  the channel map in `config.py` (recorded in DECISIONS).
  `MotionSequence.txt` confirms gesture17 = Rest (no remapping needed).
  RECORDS has 1806 records per gesture (= 43 subjects × 3 sessions ×
  7 trials × .dat/.hea halves of 903 actual records per gesture).
- `uv run pytest` — 3 passed (preprocess shapes/notch, windows small
  build with meta↔X alignment). One fix on the way: `wfdb.wrsamp` rejects
  float signals outside digital bounds, so the test writes digital
  samples (fmt 16, gain 200) like the real files. `ruff check` clean.

## 2026-10-26 — Download switched to parallel fetch

- `wget -r` mirrored at ~30–40 KB/s per connection; measured the zip
  endpoint at ~28 KB/s too — per-connection throttle on PhysioNet's side.
  Replaced `make download` with `scripts/download.sh`: reads the file
  list from SHA256SUMS.txt and fetches with `xargs -P` + `curl -C -`
  (resumable). Measured ~19–20 MB/min aggregate at both JOBS=16 and
  JOBS=32 — the server caps aggregate bandwidth, so ~9.8 GB ⇒ ~8 h.
  Running in background (`/tmp/emg-download2.log`).
- Added `data.is_complete()`: the .hea lands before its .dat, so a pair
  looks present mid-download; checks .dat size ≥ header sig_len×n_sig×bps.
  `mains_check` and `windows.build` skip incomplete trials.
- **Mains check done** on the first 120 complete trials: PSD shows a
  deep hole at exactly 60 Hz (hardware pre-notched) → mains = 60 Hz.
  Recorded in DECISIONS + `artifacts/mains.json`; figure
  `artifacts/figures/mains_check.svg`.
- Smoke build on subject 1's completed trials (`--subjects 1`):
  N=3990 windows, C=28, W=256, 55 MB, wall 2 s, peak RSS 12.9 MB —
  pipeline end-to-end works on real data (test output deleted).
- `make verify` mid-download: manifest-line parse bug found and fixed
  (single-space separator) → now reports `386 ok, 29 failed, 30300
  missing`; the 29 "failed" are files still being fetched by `curl -C -`
  and will verify once the download completes.
- Repo pushed to github.com/Vvgit22/emg-gesture-decoder (public, branch
  `master`): commits `2c41e21` Scaffold, `7223181` data pipeline,
  `25cd43f` verify fix.
- Full `--subjects` build is scheduled once the ~9.8 GB download
  finishes (~8 h at the server's ~330 KB/s aggregate cap).

## 2026-10-26 — Switched download to the S3 mirror

- Lead verified PhysioNet's open-data S3 bucket
  (`physionet-open.s3.amazonaws.com`) serves the same tree unthrottled.
  Stopped the parallel physionet.org run at ~156 MB, pruned 27
  incomplete .dat/.hea records via `is_complete`, rewrote
  `scripts/download.sh` (S3 base, `SRC=physionet` fallback; file list
  still built from SHA256SUMS.txt — an S3 listing paginated attempt
  returned HTTP 400 on continuation tokens, and the manifest is
  authoritative anyway). Relaunched 15:54:45 — measured **521 MB/min**
  (~8.7 MB/s, 31 active curls) ⇒ ~18 min for the remaining ~9.7 GB.
  DECISIONS updated.

## 2026-10-26 — Download finished, verified, full memmap built

- S3 download finished ~16:20 (9.59 GB in ~22 min from relaunch at
  15:54; aggregate ~7 MB/s with bursts to ~500 MB/min). First verify
  pass: 30,465 ok / 14 failed / 236 missing — the stragglers were curls
  that had silently errored mid-file; the 14 corrupt were deleted and a
  second `download.sh` run filled all gaps. Final
  `uv run python -m emgdec.data --verify`: **30,715 ok, 0 failed, 0
  missing** (97 s for the full re-hash).
- Full `python -m emgdec.windows` (`/usr/bin/time -l`):
  **N=537,285 windows, C=28, W=256, X.f16 = 7.70 GB, wall 434 s,
  peak RSS 2.75 GB** (< 6 GB bound). N equals exactly 15,351 trials ×
  35 windows/trial — complete coverage.
- Trial audit on `meta.parquet`: all 43 subjects × 3 sessions ×
  17 gestures × 7 trials present, exactly 35 windows each, **0 missing,
  0 extra**; 15,351 .hea files on disk match.
- `uv run pytest`: **14 passed** — includes the lead's
  `test_splits_no_leak.py` (fake-data + real-data leak checks on all
  three splits, all clean). `uv run ruff check`: clean.
- Note: `splits.py` + `test_splits_no_leak.py` were authored by the lead
  and left uncommitted in the tree; all my files are committed.
