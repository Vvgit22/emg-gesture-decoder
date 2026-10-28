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

## 2026-10-26 — Classical features + LDA baselines

- `features.py` --cache: 11 features × 28 channels = **308 columns** for all
  **537,285 windows** → `data/memmap/features.f32` (~0.62 GB float32) +
  `features_names.json`. Wall time **119 s**, chunked in batches of 20,000.
- LDA sweep (`solver='lsqr'`, `shrinkage='auto'`, Ledoit–Wolf): 3 split kinds ×
  3 channel configs = **531 runs**, all saved via `save_run` under
  `artifacts/runs/lda/<config>/<kind>/`. Whole sweep wall time **~90 s**
  (train_seconds summed per cell ≈ a few minutes; M-series chip). No
  convergence warnings or failures on any split.
- `evaluate.py` table (accuracy %, ± std over splits; chance 5.9%):

  | config | within_subject | cross_day | cross_subject |
  |---|---|---|---|
  | forearm | 82.0 ± 8.9 | 72.2 ± 8.8 | 53.1 ± 2.7 |
  | wrist | 84.7 ± 7.6 | 68.7 ± 11.6 | 40.6 ± 2.7 |
  | both | 93.4 ± 5.7 | 87.0 ± 7.3 | 67.9 ± 2.2 |

- Surprises:
  - The **wrist ring alone beats the forearm alone within-subject**
    (84.7 vs 82.0) but collapses cross-subject (40.6 vs 53.1) — wrist signal
    seems more subject-specific, forearm more transferable.
  - Cross-day is unexpectedly strong with both rings (87.0%) — day-to-day
    electrode shift hurts less than person-to-person anatomy.
  - Gesture 16 "Hand Close" is heavily confused with gesture 1
    (cross-day confusion: row 1 → col 16 ≈ 1004 windows) — worth checking
    for CNNs later.
- Figures written from `results.json` only (no recompute):
  `confusion_lda_both_{within_subject,cross_day,cross_subject}.svg`,
  `accuracy_by_split_lda.svg`, `within_subject_per_subject_lda.svg`.
- `uv run pytest`: 29 passed; `ruff check` clean.

## 2026-10-26 — Checkpoint 2: TinyCNN + sweeps + ONNX (in progress)

- `models.py`: TinyCNN, 4 strided conv blocks + GAP + dropout + linear,
  **57,841 params** for C=28 (asserted 40k–80k in tests). Normalisation is
  a registered buffer inside the graph, stats from train windows only.
- `train.py`: AdamW 2e-3 / wd 1e-2, cosine, batch 256, early stop on val
  acc, best-val restore, test scored once. `--limit`, `--force`, resume-
  on-existing. `.pt` weights saved for cross_subject (all configs) and
  config both.
- Smoke `--kind within_subject --config both --limit 3`: test accs
  0.856 / 0.905 / 0.965 (LDA same splits ~0.8–0.9), ~1–7 s per split on
  MPS, peak RSS 0.9 GB.
- Sweeps launched in background (MPS):
  - `/tmp/emg-sweep-a.sh` → log `/tmp/emg-sweep-a.log`: within_subject all
    configs → cross_day all → cross_day crops 128/64 (both).
  - `/tmp/emg-sweep-b.sh` → log `/tmp/emg-sweep-b.log`: cross_subject all
    configs → cross_subject crops 128/64 (both).
  - within_subject pace ≈ 1–2 s/split (~10 s first split incl. warmup).
- `export.py`: opset-17 ONNX, dynamic batch, input "emg" [B,C,256] raw
  float32, output "logits"; `verify` = torch-CPU vs ort-CPU on 512 real
  test windows, bar 1e-4 / 100% argmax → `artifacts/onnx/parity.json` +
  `.meta.json` sidecar. Random-weights parity test passes.
- `evaluate.MODELS` extended with `cnn_w128`/`cnn_w64` — save_run rejected
  the ablation names otherwise (flagged to lead; evaluate.py was
  previously protected).
- tests: **34 passed** (incl. overfit-learns, leak-spy, ONNX parity).
  ruff clean.

### Sweep results (all finished ~14:10)

- Sweep A (`/tmp/emg-sweep-a.log`): within_subject ×3 configs = 387 runs
  (~15 min incl. the 3 pre-existing smoke runs skipped via resume),
  cross_day ×3 = 129 runs (~12 min), cross_day crops 128/64 = 86 runs.
  Total wall ≈ 35 min → `reproduce` gained `train` (<1 h threshold).
- Sweep B (`/tmp/emg-sweep-b.log`): cross_subject ×3 = 15 runs + crops
  128/64 = 10 runs. First fold (forearm): **143 s total, ~15 s/epoch**
  after a ~60–90 s memmap load of ~375k fp16 windows (~5.4 GB).
  Per-fold typical 45–190 s; two outliers (fold1/both 760 s, fold4
  357 s) coincided with concurrent evaluate/figure/export work on the
  same machine — memory contention, not a training problem.
  Peak RSS **10.1 GB** on cross_subject (train+val+test fp16 copies +
  memmap paging) — under the 16 GB machine budget but above the 6 GB
  the classic pipeline held; noted for a future chunked-load fix.
- Final table (`make evaluate`, 1158 run files):

  | model | config | within_subject | cross_day | cross_subject |
  |---|---|---|---|---|
  | lda | forearm | 82.0 ± 8.9 | 72.2 ± 8.8 | 53.1 ± 2.7 |
  | lda | wrist | 84.7 ± 7.6 | 68.7 ± 11.6 | 40.6 ± 2.7 |
  | lda | both | 93.4 ± 5.7 | 87.0 ± 7.3 | 67.9 ± 2.2 |
  | cnn | forearm | 84.0 ± 8.4 | 79.3 ± 9.5 | 68.7 ± 3.4 |
  | cnn | wrist | 89.7 ± 6.4 | 76.2 ± 11.4 | 60.2 ± 2.2 |
  | cnn | both | 92.7 ± 5.9 | 88.8 ± 5.7 | **79.6 ± 2.5** |
  | cnn_w128 | both | — | 87.4 ± 6.3 | 78.0 ± 4.2 |
  | cnn_w64 | both | — | 82.9 ± 6.7 | 76.2 ± 3.3 |

- Read: CNN ≈ LDA within-subject (92.7 vs 93.4 — LDA was already near
  ceiling on calibrated data), CNN clearly wins cross-day (+1.8 both)
  and decisively wins cross-subject (+11.7 both; forearm-only +15.6,
  wrist-only +19.6). Ablation: 125 ms costs ~1.4–1.6 pts, 62.5 ms costs
  ~3.4–5.9 pts — most information is in the first ~125 ms but the tail
  still helps. Nothing >99% anywhere; no implausible jumps → no leak
  signal.
- ONNX: `cnn_both_within_subject_s01_d1.onnx` and
  `cnn_both_cross_subject_fold3.onnx` (fold3 holds out subject 1),
  **226 KB each**, parity vs torch-CPU on 512 test windows:
  max |logit diff| 4.8e-6 / 9.5e-6, 100% argmax agreement
  (`artifacts/onnx/parity.json`). `.meta.json` sidecars written.
- Figures regenerated for all models × both-configs (confusions, per-class
  F1, grouped accuracy bars, per-subject strip, ablation curve).

## 2026-10-27 — Checkpoint 3: site assets + browser demo bundle

- `site_assets.py`: packed `artifacts/site/` — results.json copy, the five
  figures, `seen.onnx`/`unseen.onnx` (+meta), `demo.bin`, `demo.json`.
- Held-out verified, not assumed: subject 1 ∈ fold3 test ✓; s01_d1 test
  trials = {6,7} ✓ — so trial 6 is unseen by BOTH models.
- `demo.bin`: 17×28×2048 int16 = **1,949,696 bytes** exactly; stitched
  from the trial's 35 overlapping windows after asserting every
  consecutive overlap is bit-identical (all 544 channel-overlaps passed).
  Centre crop [1280:3328]. One global scale = max|x|/32000.
- Demo-clip scoring (255 windows: 17 gestures × 15 non-overlapping
  256-sample windows): **seen 89.0%** (its own split test: 85.6%),
  **unseen 88.6%** (split: 81.5%); **quantisation agreement 1.000** —
  int16 never changed a single argmax. `first_15_argmax_*` lists stored in
  demo.json for the browser↔Python parity check.
- `report.py` (`make readme`): README results table regenerated between
  markers from results.json; Browser-demo section added.
- tests: `test_site_assets.py` — stitch recovers trace, stitch fires on
  corrupted overlap, int16 round-trip ≤ half a scale step, byte-size
  formula. `uv run pytest`: 38 passed; ruff clean.
- `scripts/thumbnail.py`: renders the portfolio thumbnail from demo.bin
  (8 real traces, plasma palette, lit 17-cell grid).
- Portfolio side (separate repo): `emg:sync` script copies site/ +
  onnxruntime-web 1.30.0 jsep wasm into `public/emg/` (~3.0 MB ours,
  27 MB wasm runtime); `/emg/` demo page streams int16 clips and runs both
  ONNX models in-browser. Bugs found in verification and fixed:
  - ORT 1.30 requests the *jsep* wasm build, not the plain one — the
    non-jsep pair was deleted again (42→27 MB).
  - Fractional playback positions indexed the int16 array with floats →
    `undefined` → NaN logits for early windows (only the pos=2048
    inference was integer). Windows now anchored to the fixed 128-grid.
  - Serialised inferences on a promise queue (wasm sessions are not
    re-entrant).
  - Lighthouse pass: split lazy loading into data-on-scroll vs
    models-on-Play — result **99 / 100 / 100 / 100**.
- Browser↔Python parity: first-15-window argmax lists identical for both
  models (15/15).
- Subject-1 observation (requested note): demo-clip accuracies were
  89.0% (seen) / 88.6% (unseen) vs full split-test accuracies 85.6% /
  81.5% — subject 1 was chosen as the demo person BEFORE any model was
  trained and was not re-selected; on this one person the two models
  happen to land close together.

## 2026-10-27 — Phase 2: memory path + rerun

- Thrash incident: offline_v1b fold4 reached 22 GB RSS and stalled 45 min
  (two 7.7 GB memmaps + two 5.4 GB train copies + a 10.8 GB float32 stat
  copy tipped the 16 GB box into swap). The earlier causal fold0 taking
  6034 s was the same contention.
- Found in the same pass: `--x` patched `emgdec.train.MEMMAP_DIR` but the
  running module is `__main__` (causal_v1 folds were reproducing offline
  results to all digits — they were offline runs); seed offsets 1/2
  wrote to the same filenames as offset 0 and were "done: 0" skipped.
- Fix: `_load_windows` now chunks 20k sorted rows into ONE preallocated
  float16 array (peak = one copy); norm stats via chunked float64
  sum/sumsq (matches numpy to ~1e-7); per-run RSS print + estimate +
  --max-rss-gb 12 abort. All offline_v1b run files + .pt deleted and the
  experiment is re-run end-to-end under the new loader.
