# Mirrors the GrabMyo dataset from PhysioNet into data/raw/ (resumable, ~9.1 GB; JOBS=N for parallelism)
download:
	sh scripts/download.sh

# Checks every downloaded file against the dataset's SHA256SUMS manifest
verify:
	uv run python -m emgdec.data --verify

# Builds the windowed dataset (memmap X + meta.parquet); pass SUBJECTS="1,2" for a quick subset
windows:
	uv run python -m emgdec.windows $(if $(SUBJECTS),--subjects $(SUBJECTS))

# Runs the test suite
test:
	uv run pytest

# Lint + tests, the same as CI
check: lint test

# Lints the source and tests
lint:
	uv run ruff check src tests

# Computes all 11 features/channel for every window -> data/memmap/features.f32 (~100 s)
features:
	uv run python -m emgdec.features --cache

# Runs the LDA baseline sweep: 3 split kinds x 3 channel configs = 531 run files
baselines:
	uv run python -m emgdec.baselines --kind all --config all

# Aggregates $(EXP) run JSONs into its results.json and prints the table
evaluate:
	uv run python -m emgdec.evaluate --experiment $(EXP)

# Draws the five result figures (SVG) from artifacts/results.json
figures:
	uv run python -m emgdec.figures

# End-to-end replay of checkpoint 1C: features -> baselines -> evaluate -> figures
reproduce: features baselines train evaluate figures

# Experiment dir used by training/evaluate targets; a NEW id by default so
# committed JSONs can never cause silent skipping
EXP ?= repro_$(shell date +%Y%m%d)

# Trains the CNN on within_subject + cross_day for all channel configs (resumable)
train:
	uv run python -m emgdec.train --kind within_subject --config all --experiment $(EXP) --mode resume
	uv run python -m emgdec.train --kind cross_day --config all --experiment $(EXP) --mode resume

# Trains the CNN on cross_subject for all configs — the long pole, hours
train-cross-subject:
	uv run python -m emgdec.train --kind cross_subject --config all --experiment $(EXP) --mode resume

# Window-length ablation: centre-crops of 125 ms and 62.5 ms, both config
ablation:
	uv run python -m emgdec.train --kind cross_day --config both --crop 128 --experiment $(EXP) --mode resume
	uv run python -m emgdec.train --kind cross_day --config both --crop 64 --experiment $(EXP) --mode resume
	uv run python -m emgdec.train --kind cross_subject --config both --crop 128 --experiment $(EXP) --mode resume
	uv run python -m emgdec.train --kind cross_subject --config both --crop 64 --experiment $(EXP) --mode resume

# Full pipeline from features through CNNs to figures (hours; sweeps are resumable)
reproduce-full: reproduce train train-cross-subject ablation evaluate figures

# Packs artifacts/site/ (demo.bin, demo.json, the two ONNX models, figures)
site: export
	uv run python -m emgdec.site_assets

# Rewrites the README results table from artifacts/results.json
readme:
	uv run python -m emgdec.report

# Tiny smoke run: 2 within-subject splits, both config, CPU-safe
smoke:
	uv run python -m emgdec.train --kind within_subject --config both --limit 2 --experiment smoke --mode resume --device cpu

# Diffs the mirror's SHA256SUMS.txt against PhysioNet's (mirror trusted for bytes only)
verify-manifest:
	curl -fsSL https://physionet.org/files/grabmyo/1.1.0/SHA256SUMS.txt -o /tmp/SHA256SUMS.physionet && diff -q /tmp/SHA256SUMS.physionet data/raw/SHA256SUMS.txt && echo "manifests identical"

# Lint + tests, same as CI
check: lint test

# Exports the two demo ONNX models + parity for EXP (default causal_v1)
export:
	uv run python -m emgdec.export --model cnn --config both --kind within_subject --name s01_d1 --experiment $(if $(EXP),$(EXP),causal_v1) --x data/memmap_causal
	uv run python -m emgdec.export --model cnn --config both --kind cross_subject --name fold3 --experiment $(if $(EXP),$(EXP),causal_v1) --x data/memmap_causal
