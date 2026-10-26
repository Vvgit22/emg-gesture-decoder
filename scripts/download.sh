#!/bin/sh
# Parallel mirror of GrabMyo into data/raw/.
#
# Default source is PhysioNet's open AWS S3 mirror
# (physionet-open.s3.amazonaws.com) — same tree, no auth, no per-connection
# throttle (~960 KB/s+ measured vs ~30 KB/s from physionet.org, which made
# a 9.8 GB mirror take ~8 h). physionet.org remains the fallback:
#   SRC=physionet sh scripts/download.sh
# The file list is built from the S3 bucket listing (paginated). Byte
# integrity is still proven by SHA256SUMS.txt from PhysioNet via
# `make verify` — the mirror is trusted for bytes, not for truth.
# `curl -C -` resumes; re-running is safe.
set -u
cd "$(dirname "$0")/.."
JOBS="${JOBS:-16}"
SRC="${SRC:-s3}"

if [ "$SRC" = "s3" ]; then
  BASE="https://physionet-open.s3.amazonaws.com/grabmyo/1.1.0"
else
  BASE="https://physionet.org/files/grabmyo/1.1.0"
fi
mkdir -p data/raw

# Always refresh the authoritative manifest first (small file).
curl -sfL -o data/raw/SHA256SUMS.txt "$BASE/SHA256SUMS.txt"

if [ "$SRC" = "s3" ]; then
  # Drop files still mid-download: .dat/.hea pairs are checked against
  # the header's claimed size by emgdec.is_complete; anything else we
  # cannot size-check is left for `make verify` to flag after the run.
  uv run python - <<'PY'
from pathlib import Path
from emgdec.data import is_complete
removed = 0
for hea in Path("data/raw").rglob("*.hea"):
    if not is_complete(hea.with_suffix("")):
        for ext in (".hea", ".dat"):
            f = hea.with_suffix(ext)
            if f.exists():
                f.unlink()
        removed += 1
print(f"pruned {removed} incomplete trial records")
PY
fi
awk '{print $2}' data/raw/SHA256SUMS.txt | grep -v '^SHA256SUMS.txt$' \
  | xargs -P "$JOBS" -I {} sh -c '
    f="$1"
    mkdir -p "data/raw/$(dirname "$f")"
    curl -sfL -C - --retry 3 -o "data/raw/$f" "'"$BASE"'/$f"
  ' _ {}
echo "download: done"
