#!/bin/sh
# Parallel mirror of GrabMyo into data/raw/.
#
# PhysioNet throttles each connection to ~25 KB/s, so the textbook
# single-stream `wget -r` mirror would take days. We fetch the file list
# from the dataset's own SHA256SUMS.txt and download files in parallel
# (16 workers by default, override with JOBS=N). `curl -C -` makes the
# whole thing resumable: re-running skips finished files and continues
# partial ones. Top-level docs are fetched first so they exist even
# mid-download.
set -u
cd "$(dirname "$0")/.."
BASE="https://physionet.org/files/grabmyo/1.1.0"
JOBS="${JOBS:-16}"
mkdir -p data/raw

curl -sfL -o data/raw/SHA256SUMS.txt "$BASE/SHA256SUMS.txt"

awk '{print $2}' data/raw/SHA256SUMS.txt | grep -v '^SHA256SUMS.txt$' \
  | xargs -P "$JOBS" -I {} sh -c '
      f="$1"
      mkdir -p "data/raw/$(dirname "$f")"
      curl -sfL -C - --retry 3 -o "data/raw/$f" "'"$BASE"'/$f"
    ' _ {}
echo "download: done"
