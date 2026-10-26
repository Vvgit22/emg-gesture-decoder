"""Reading GrabMyo files: one trial at a time, and a walk over the whole tree.

In:  the `data/raw/` mirror produced by `make download` (WFDB .dat + .hea
     pairs named `session{s}_participant{p}_gesture{g}_trial{t}`).
Out: `read_trial()` returns a float32 array [samples, 32] for one trial
     plus the sampling rate and unit strings from its header;
     `iter_trials()` yields one record per trial so nothing ever holds the
     whole dataset in memory.
If skipped: everything downstream would either load the wrong files or
     mislabel gestures — the (subject, session, gesture, trial) tuple parsed
     from each filename is what the metadata table and the leak-tested
     splits are built on.

`--verify` mode re-checks every downloaded file against the dataset's
SHA256SUMS.txt manifest (same check as `shasum -a 256 -c`).
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import wfdb

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

_TRIAL_RE = re.compile(
    r"session(?P<session>\d+)_participant(?P<subject>\d+)"
    r"_gesture(?P<gesture>\d+)_trial(?P<trial>\d+)$"
)


def read_trial(path: str | Path) -> tuple[np.ndarray, float, list[str]]:
    """Read one WFDB record (path without extension) → (signal, fs, units).

    `signal` is float32, shape [samples, 32], in the header's physical
    units (microvolts for GrabMyo).
    """
    rec = wfdb.rdrecord(str(path))
    sig = np.asarray(rec.p_signal, dtype=np.float32)
    units = list(rec.units or [])
    return sig, float(rec.fs), units


def parse_trial_name(name: str) -> tuple[int, int, int, int] | None:
    """'session1_participant2_gesture5_trial3' → (1, 2, 5, 3) or None."""
    m = _TRIAL_RE.match(name)
    if not m:
        return None
    return tuple(int(m.group(k)) for k in ("session", "subject", "gesture", "trial"))


def is_complete(path: str | Path) -> bool:
    """True when the .dat file holds as many bytes as the header claims.

    Guards against reading a file that is still mid-download: the .hea is
    tiny and arrives before its (large) .dat, so a pair can look present
    while the data file is short.
    """
    path = Path(path)
    try:
        hdr = wfdb.rdheader(str(path))
    except Exception:
        return False
    bps = {"16": 2, "24": 4, "32": 4, "80": 1}.get(str(hdr.fmt[0]), 2)
    expected = hdr.sig_len * hdr.n_sig * bps
    dat = path.with_suffix(".dat")
    return dat.exists() and dat.stat().st_size >= expected


def iter_trials(raw_dir: str | Path = RAW_DIR):
    """Yield (subject, session, gesture, trial, path) for every .hea record.

    `path` has no extension — pass it straight to `read_trial`. Sorted so
    output order is deterministic: session, then subject, then gesture,
    then trial.
    """
    raw = Path(raw_dir)
    items = []
    for hea in raw.rglob("*.hea"):
        parsed = parse_trial_name(hea.stem)
        if parsed is None:
            continue
        session, subject, gesture, trial = parsed
        items.append((subject, session, gesture, trial, hea.with_suffix("")))
    items.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    yield from items


def verify(raw_dir: str | Path = RAW_DIR) -> tuple[int, int, list[str]]:
    """Check files against SHA256SUMS.txt → (ok, failed, missing_names)."""
    raw = Path(raw_dir)
    manifest = raw / "SHA256SUMS.txt"
    ok = failed = 0
    missing: list[str] = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition("  ") or line.split(" ", 1)
        name = name.strip()
        f = raw / name
        if not f.exists():
            missing.append(name)
            continue
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        if h == digest:
            ok += 1
        else:
            failed += 1
    return ok, failed, missing


def _main() -> None:
    if "--verify" in sys.argv:
        ok, failed, missing = verify()
        print(f"verify: {ok} ok, {failed} failed, {len(missing)} missing")
        for name in missing[:10]:
            print(f"  missing: {name}")
        sys.exit(1 if failed or (not missing and ok == 0) else 0)


if __name__ == "__main__":
    _main()
