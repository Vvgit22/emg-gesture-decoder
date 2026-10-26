# Mirrors the GrabMyo dataset from PhysioNet into data/raw/ (resumable, ~9.1 GB; JOBS=N for parallelism)
download:
	sh scripts/download.sh

# Checks every downloaded file against the dataset's SHA256SUMS manifest
verify:
	uv run python -m emgdec.data --verify

# Builds the windowed dataset (memmap X + meta.parquet); pass SUBJECTS="1,2" for a quick subset
windows:
	uv run python -m emgdec.windows --subjects $(SUBJECTS)

# Runs the test suite
test:
	uv run pytest

# Lints the source and tests
lint:
	uv run ruff check src tests

# Computes all 11 features/channel for every window -> data/memmap/features.f32 (~100 s)
features:
	uv run python -m emgdec.features --cache

# Runs the LDA baseline sweep: 3 split kinds x 3 channel configs = 531 run files
baselines:
	uv run python -m emgdec.baselines --kind all --config all

# Aggregates artifacts/runs/**/*.json into artifacts/results.json and prints the table
evaluate:
	uv run python -m emgdec.evaluate

# Draws the five result figures (SVG) from artifacts/results.json
figures:
	uv run python -m emgdec.figures

# End-to-end replay of checkpoint 1C: features -> baselines -> evaluate -> figures
reproduce: features baselines evaluate figures
