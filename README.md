# emg-gesture-decoder

Hand-gesture recognition from forearm and wrist EMG recorded by the open
GrabMyo dataset: a fully reproducible pipeline (download → windows → splits
→ models → ONNX) with leak-tested train/test splits and a live browser demo
on the project page.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
```

## Download

```bash
make download   # mirrors ~9.1 GB into data/raw/ (resumable)
make verify     # checks every file against the dataset's SHA256SUMS manifest
```

## Reproduce

Targets to come: `make windows` (windowed dataset), `make reproduce`
(baselines + within-subject CNN), `make reproduce-full` (all splits,
channels configs, ablations). Each regenerates `artifacts/results.json`,
the figures, and the ONNX models with fixed seeds.

## Data licence and citations

The GrabMyo dataset is distributed by PhysioNet under **CC BY 4.0**; the
code here is MIT (see LICENSE). Required citations:

- Pradhan A, He J, Jiang N. *Multi-channel surface electromyography signals
  for hand gesture recognition.* **Sci Data** 9, 733 (2022).
  doi:10.1038/s41597-022-01836-y
- GrabMyo dataset, PhysioNet. doi:10.13026/89dm-f662
- Pollard J, Gawlikowska A, et al. *Resources available in PhysioNet.*
  **Nature Health** (2026). doi:10.1038/s44360-026-00096-z
