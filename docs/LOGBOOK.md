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
