# DECISIONS

Every non-obvious choice in the pipeline, with the reason, the alternative
rejected, and what we would expect to change if we had chosen otherwise.
Newest entries at the bottom; do not rewrite history — append.

## 2026-10-26 — Dataset: GrabMyo v1.1.0

- **Decision:** GrabMyo (PhysioNet) — 43 subjects × 3 sessions/days ×
  (16 gestures + rest) × 7 trials, 5 s at 2048 Hz, 16 forearm + 12 wrist
  bipolar channels plus 4 unused columns.
- **Reason:** open access, no account, real multi-day design (the
  interesting question — does a model still work tomorrow / on someone
  else — is built into the data).
- **Alternative rejected:** Ninapro DB2 (much larger community usage but
  heavier, more complex protocol) and a self-recorded set (no
  cross-subject validity at all).
- **What would change:** smaller dataset → faster pipeline but weaker
  claims; self-recorded → no credible cross-subject number.

## 2026-10-26 — Preprocessing: band-pass 20–450 Hz, mains notch, ×2 decimate

- **Decision:** zero-phase 4th-order Butterworth band-pass 20–450 Hz at
  the native 2048 Hz, then a notch at mains frequency, then
  `scipy.signal.decimate` ×2 to 1024 Hz.
- **Reason:** most EMG energy sits in 20–450 Hz; filtering before
  decimating keeps the design honest (no aliased high-frequency content)
  and 1024 Hz halves storage for negligible information loss (Nyquist
  512 Hz > 450 Hz band edge).
- **Alternative rejected:** decimate first then filter (simpler, but the
  450 Hz edge sits too close to the 512 Hz Nyquist of the decimated
  signal) and no notch (mains hum would dominate some channels).
- **What would change:** a wider band keeps more noise; a narrow band
  discards high-frequency gesture detail.

## 2026-10-26 — Windows: 250 ms with 50% overlap, drop first 0.5 s

- **Decision:** WIN=256 samples at FS_OUT=1024 (250 ms), STEP=128 (50%
  overlap); drop the first 0.5 s of every 5 s trial.
- **Reason:** 250 ms is a standard EMG segment length — long enough for
  stable features, short enough that the gesture inside one window is
  roughly constant. The first 0.5 s is the subject's reaction ramp, not
  the held gesture.
- **Alternative rejected:** 150/400 ms windows (kept as an ablation for
  checkpoint 2) and keeping the ramp (labels would be wrong at trial
  starts).
- **What would change:** shorter windows → more windows, noisier
  features; longer → fewer, smoother.

## 2026-10-26 — Channel order and layout

- **Decision:** store windows as `[N, 28, WIN]` — channels 1–16
  (forearm, 0-based 0–15) then channels 18–23 and 26–31 (wrist,
  0-based 17–22 and 25–30). The four unused columns (17, 24, 25, 32 —
  1-based; 0-based 16, 23, 24, 31) are dropped at windowing time.
- **Verified 2026-10-26:** header signal names are exactly
  `F1..F16, U1, W1..W6, U2, U3, W7..W12, U4` — forearm at 1–16, unused
  (U1–U4) at 17/24/25/32, wrist at 18–23 and 26–31 (1-based), matching
  the PhysioNet description and `Electrodelocation.pdf`.
- **Reason:** keeps forearm and wrist contiguous so the forearm/wrist
  channel configs are slices `X[:, :16]` and `X[:, 16:]`.
- **Alternative rejected:** keep all 32 columns and mask at training
  time — wastes ~14% storage and invites accidental leakage of dead
  channels into features.
- **What would change:** a different order only changes indexing, not
  results; it must be consistent everywhere.

## 2026-10-26 — Storage: memmap float16 + parquet metadata

- **Decision:** windows live in `data/memmap/X.f16` (np.memmap float16),
  metadata in `data/memmap/meta.parquet`, one row per window.
- **Reason:** ~550k windows × 28 × 256 × 2 B ≈ 8 GB — fits on disk,
  streams to the trainer; float32 would double that and float16 keeps
  plenty of precision for EMG amplitudes.
- **Alternative rejected:** float32 (cleaner, twice the size) and HDF5
  (extra dependency for no benefit at this scale).
- **What would change:** float32 would cost disk, not accuracy; HDF5
  would cost a dependency, not accuracy.

## 2026-10-26 — Rest class label

- **Decision:** rest is gesture index 17 (classes are 1–17, rest = 17).
- **Verified:** `data/raw/MotionSequence.txt` lists 17 motion names in
  order 1–17; the 17th is 'Rest'. Filenames carry `gesture1…gesture17`
  directly (1806 records each in RECORDS), so no remapping is needed.

## 2026-10-26 — Mains frequency: 60 Hz, already notched in the data

- **Decision:** notch at 60 Hz.
- **Evidence (not a guess):** pooled Welch PSD over 20 random trials shows
  essentially zero energy in a ±1 Hz band at 60 Hz while the surrounding
  52–68 Hz band is populated — a deep hole at exactly 60 Hz, i.e. the
  acquisition hardware already applied a 60 Hz mains notch. A plain
  energy comparison would wrongly pick 50 Hz (the 1/f tail is higher at
  lower frequencies); `mains_check` therefore measures line *prominence*
  and treats a pre-notched hole as evidence for that frequency.
- **Alternative rejected:** skip the notch because the hardware already
  did it — rejected as fragile (a second zero-phase notch on an already
  empty band is a cheap no-op; if any trial lacks the hardware notch it
  still protects the features).
- **Figure:** `artifacts/figures/mains_check.svg`; result stored in
  `artifacts/mains.json` and read by `windows.build`.

## 2026-10-26 — Download mirror: PhysioNet S3 open-data bucket

- **Decision:** fetch dataset bytes from `physionet-open.s3.amazonaws.com`
  (PhysioNet's official open-data AWS mirror), keeping
  `physionet.org/files/...` as a documented `SRC=physionet` fallback in
  `scripts/download.sh`.
- **Reason:** physionet.org throttles to ~30 KB/s per connection and caps
  aggregate at ~330 KB/s (measured at 16 and 32 workers) → ~8 h for
  9.8 GB. The S3 mirror measured ~8.7 MB/s sustained → ~20 min.
- **Alternative rejected:** keep waiting on the throttled mirror; also
  rejected: trusting S3 for *correctness* — the file list and integrity
  come from `SHA256SUMS.txt` served by physionet.org, and `make verify`
  re-hashes every file. The mirror is trusted for bytes, not for truth.
- **What would change:** nothing in the pipeline — only provenance of the
  bytes; every file is still checksum-verified.
