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

## 2026-10-26 — Splits: group-wise, fixed, leak-tested

- **Decision:** three split families, all defined on the metadata table,
  never on the signal:
  - `within_subject` — per (subject, day): trials 1–4 train, trial 5
    validation, trials 6–7 test.
  - `cross_day` — per subject: days 1–2 train (their trial 7 =
    validation), day 3 test.
  - `cross_subject` — 5 folds over people, seeded shuffle, each person
    tested exactly once; 4 whole training subjects held out as
    validation per fold.
  `assert_no_leak` checks index disjointness and that the defining group
  (trial / day / person) never appears on both sides; it runs in the
  test suite and can be called before any training run.
- **Reason:** overlapping windows from one 5 s repetition are near-copies
  of each other. Any split that puts two windows of the same repetition on
  opposite sides measures memory, not recognition. Validation comes from
  the training side so early stopping cannot see the exam.
- **Alternative rejected:** random window-level split (inflated to ~99%,
  meaningless); leave-one-trial-out cross-validation (7× the compute for
  the same question); a single random subject split (one fold = one noisy
  number, and some subjects never tested).
- **What would change:** fewer folds → cheaper, noisier cross-subject
  estimate; testing on trial 7 only → half the test windows.
- **Seed:** `config.SEED = 20261026` (the date the splits were fixed).
  Changing it changes fold membership and requires a note here.

## 2026-10-26 — Feature set: 11 hand-crafted features per channel

- **Decision:** per window, per channel: RMS, MAV, waveform length, zero
  crossings, slope sign changes, Hjorth mobility + complexity, and four
  Welch band powers (20–50, 50–100, 100–200, 200–450 Hz, nperseg=128) —
  11 × 28 = 308 features per window.
- **Reason:** this is the classic sEMG feature set (Hudgins' time-domain
  features + frequency-domain staples). Every one is cheap, interpretable,
  and decades of literature says it works. It gives the deep model a hard
  baseline to beat instead of a straw man.
- **Alternative rejected:** raw windows into LDA (too many dimensions,
  hopeless); a bigger kitchen-sink set (marginal gains, slower, harder to
  explain).
- **What would change:** adding features changes `features.f32` width and
  every downstream model's input size — bump the cache and re-run.

## 2026-10-26 — ZC/SSC threshold: fixed 1e-3 in standardised units

- **Decision:** zero crossings and slope sign changes count a crossing only
  when the signal moves by more than 0.001 in the *train-standardised*
  units used downstream; for the raw float16 windows in the cache we apply
  the same 1e-3 on z-scored per-channel data.
- **Reason:** without a dead-zone, noise wiggle around zero inflates the
  count; scaling the threshold per channel (e.g. 0.01×std) makes the count
  depend on amplitude, which defeats the point — the threshold exists to
  ignore *quantisation/hiss*, not to adapt to the gesture.
- **Alternative rejected:** threshold = 0.01 of channel std (varies with
  amplitude; rest windows get absurdly low thresholds); zero threshold
  (noise-dominated counts).
- **What would change:** a different threshold only shifts ZC/SSC
  distributions; all models would need re-caching.

## 2026-10-26 — LDA solver: lsqr + shrinkage='auto'

- **Decision:** `LinearDiscriminantAnalysis(solver='lsqr',
  shrinkage='auto')` — Ledoit–Wolf shrinkage estimated from the training
  fold each time.
- **Reason:** with 176–308 features and ~20k–400k training windows the
  covariance is fine but shrinkage still guards against collinear features
  (e.g. RMS vs MAV). 'auto' needs no hyperparameter tuning, so the
  validation split is genuinely unused — there is no early stopping or
  model selection for LDA.
- **Alternative rejected:** solver='svd' (no shrinkage, unstable when
  features correlate); eigen with a hand-picked shrinkage (one more knob
  to justify and tune).
- **What would change:** a different solver changes all 531 baselines;
  re-run the sweep.

## 2026-10-26 — Standardisation statistics from the training fold only

- **Decision:** mean and std for feature standardisation are computed on
  the training indices of each split only; the same transform is applied
  to validation and test.
- **Reason:** test-fold statistics leak information about the test
  distribution into the fitted pipeline — the model would implicitly
  "peek". It is also the honest version of the per-channel standardisation
  deferred from `preprocess.py`.
- **Alternative rejected:** global standardisation computed once over all
  windows (leaks test data, inflates every split's score); per-channel
  standardisation inside preprocessing (same problem, earlier).
- **What would change:** refitting scalars on all data would raise
  reported accuracies and invalidate the numbers as honest estimates.

## 2026-10-26 — CNN architecture: ~57k-parameter 1-D conv stack

- **Decision:** four strided Conv1d+BatchNorm+ReLU blocks (C→32→64→64→96,
  kernels 7/5/5/3, stride 2), `AdaptiveAvgPool1d(1)`, Dropout(0.3),
  Linear(96→17). ~57k parameters for C=28.
- **Reason:** a small temporal-conv net is the standard strong baseline
  for sEMG: enough capacity to learn gesture features, small enough that
  one split trains in seconds and the ONNX file ships in ~200 KB.
- **Alternative rejected:** a bigger ResNet-style net (slower, more
  overfitting surface, no evidence it is needed at 17 classes); a
  recurrent net (harder to export, slower per epoch); averaging-pool
  replaced by flatten (would tie the model to exactly 256 samples and
  break the window-length ablation).
- **What would change:** capacity changes alter every CNN run and the
  exported graph — full retrain.
- **No augmentation, no label smoothing:** kept out deliberately so the
  base model is simple and honest; augmentation is a named future
  ablation, not done here.

## 2026-10-26 — Normalisation lives inside the model

- **Decision:** `TinyCNN` registers `mean`/`std` buffers [C,1] and does
  `(x - mean)/std` as the first op; stats come from the split's TRAIN
  windows only (`set_normalisation`, std clamped at 1e-6).
- **Reason:** the browser demo feeds raw electrode signal; if the stats
  were a sidecar, the exported graph would not be self-contained and the
  demo could silently apply the wrong scaling. Train-only stats are the
  same no-peek rule as the LDA standardisation.
- **Alternative rejected:** a `normalisation.json` sidecar (two artifacts
  to keep in sync, easy to mismatch); preprocessing-time standardisation
  (leaks test distribution — already rejected for LDA).
- **What would change:** moving normalisation out would change every
  exported model's expected input and the demo code.

## 2026-10-26 — Optimiser, schedule and early stopping

- **Decision:** AdamW(lr=2e-3, weight_decay=1e-2), cosine LR over
  max_epochs, batch 256, plain cross-entropy. Early stopping on
  *validation* accuracy only (patience 6 within/cross-day, 3
  cross-subject), best-val weights restored before scoring test once.
- **Reason:** standard, boring, works. Early stopping on val is the only
  stopping rule that cannot see the exam; cosine decay removes the "when
  to drop the LR" hyperparameter.
- **Alternative rejected:** stopping on a fixed epoch count (would either
  under- or over-fit per split); any test-informed stopping (invalidates
  every number); SGD+momentum (needs more tuning for the same result).
- **What would change:** different budget → different weights; retrain
  and re-aggregate.

## 2026-10-26 — Epoch budget per split kind

- **Decision:** within_subject / cross_day: max_epochs=40, patience=6.
  cross_subject: max_epochs=15, patience=3.
- **Reason:** a cross-subject fold trains on ~35x more windows
  (~375k vs ~2–7k); 40 epochs there would burn GPU-hours per fold for
  gains early stopping rarely uses. Patience 3 still lets it stop early.
- **Alternative rejected:** same 40/6 everywhere (wasted compute);
  subsampling the cross-subject train set (throws away data instead of
  budgeting epochs — information is what we want more of there).
- **What would change:** budgets are in `train.BUDGET`; raising them
  needs a LOGBOOK note with the new timings.

## 2026-10-26 — Reproducibility: seeded per split; MPS is not bitwise

- **Decision:** `torch.manual_seed(SEED + fold_index)`, same for numpy
  and the shuffle Generator; `device` and `torch.__version__` stored in
  each run's params.
- **Reason:** fold_index makes each split's seed independent and stable
  if the sweep is resumed. CPU runs are bitwise reproducible; MPS runs
  are reproducible up to float-order noise (same code, same seed, tiny
  last-digit differences possible — noted, accepted, never silently
  claimed).
- **Alternative rejected:** one global seed for all splits (resume order
  would change results); forcing CPU for bitwise determinism (hours
  slower; accuracy claims don't depend on float order).
- **What would change:** changing SEED or device changes exact weights;
  accuracies should move only within float noise on MPS.

## 2026-10-26 — Window-length ablation by centre-crop, not rebuild

- **Decision:** ablation uses `crop=128` and `crop=64` — a centre slice of
  the stored 256-sample windows — recorded as models `cnn_w128`,
  `cnn_w64` (rejected save_run names until `evaluate.MODELS` listed them).
- **Reason:** the global-average-pool head accepts any length ≥16, so one
  memmap serves all three lengths; rebuilding memmaps at other lengths
  would produce *different* windows (different alignment) and confound
  length with phase.
- **What it measures:** accuracy vs analysis-window duration on centre
  crops of the already-filtered windows (after the 0.5 s trial-start
  drop). It says nothing about gesture-onset latency, and nothing about
  windows LONGER than 250 ms — we only store 250 ms; a longer-window
  answer needs a rebuild.
- **Alternative rejected:** rebuilding X.f16 at W=128/64 (changes window
  positions, not just length — measures two things at once).
- **What would change:** longer windows → new memmap, new pipeline.

## 2026-10-26 — Which models are exported, and the parity bar

- **Decision:** `train_split` saves .pt weights for every cross_subject
  fold (15 files) and every `both` config split (129 within + 43 cross-day
  + 5 cross-subject). The demo exports exactly two: within_subject
  `s01_d1` (calibrated-to-you story) and the cross_subject fold holding
  out subject 1 (works-on-a-stranger story). Both .onnx are committed via
  force-add; the rest stay gitignored under artifacts/models/.
- **Reason:** two small files (~200 KB each) carry the whole demo story;
  committing hundreds of megabytes of checkpoints would bloat the repo.
- **Parity bar:** max |logit diff| < 1e-4 and 100% argmax agreement
  between torch CPU and onnxruntime CPU on 512 real test windows —
  recorded in artifacts/onnx/parity.json. A browser demo that disagrees
  with training is worse than no demo.
- **Alternative rejected:** exporting every split's model (storage for no
  demo value); skipping parity (silent browser-vs-training mismatch).
- **What would change:** a different demo story → different named exports.

## 2026-10-27 — Demo clip: trial 6 of subject 1, quantised to int16

- **Decision:** the browser demo plays the central 2.0 s of subject 1,
  session 1, trial 6 — a trial in the TEST set of both exported models
  (asserted at build time, not assumed). Stored as int16 with one global
  scale (max|x|/32000); the page dequantises with the scale in demo.json.
- **Reason:** 2 s × 17 gestures is long enough to watch the decoder work
  and short enough to ship (~1.9 MB). int16 halves the file vs float32
  and cost nothing: measured quantisation agreement on all 255 windows
  was 1.000 — not one argmax changed.
- **Alternative rejected:** float32 bundle (3.8 MB for zero measured
  gain); per-gesture scaling (nicer SNR but breaks "one file, one scale"
  and the parity story); a train trial (would demo memorisation, not
  generalisation).
- **What would change:** a different clip choice changes demo.bin and
  demo.json's source block; re-run `make site`.

## 2026-10-27 — ORT web: version 1.30.0, jsep wasm, single thread

- **Decision:** `onnxruntime-web@1.30.0` (latest stable published ≥7 days
  before integration; 1.31.x were dev builds), the
  `ort-wasm-simd-threaded.jsep` pair copied from node_modules into
  `public/emg/ort/`, `wasm.numThreads = 1`.
- **Reason:** ORT 1.30's wasm backend loads the jsep build — the plain
  `ort-wasm-simd-threaded` pair is never fetched (verified in headless
  Chrome). numThreads=1 because GitHub Pages sends no COOP/COEP headers,
  so SharedArrayBuffer is unavailable and a threaded build would fall
  back with warnings.
- **Alternative rejected:** loading wasm from a CDN (breaks offline,
  unpinned, third-party fetch on a page about honest numbers); bundling
  wasm through Vite (public/ keeps it out of the JS graph entirely).
- **What would change:** an ORT version bump changes which wasm filenames
  to copy — the sync script is the single place that lists them.
