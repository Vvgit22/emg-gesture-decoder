# Mirrors the GrabMyo dataset from PhysioNet into data/raw/ (resumable, ~9.1 GB)
download:
	wget -r -N -c -np -nH --cut-dirs=3 -P data/raw https://physionet.org/files/grabmyo/1.1.0/

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
