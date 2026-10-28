"""Provenance: what exactly produced this run, recorded not asserted.

Collects the facts a result file needs to be reproducible and comparable:
git state, dataset version + manifest hash, preprocessing config, the exact
split fingerprint, seeds actually used, optimiser kwargs, library versions.
Old v1 records have none of this; loaders default to {} and evaluate flags
them as historical.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from emgdec.config import BAND, FS_IN, FS_OUT, STEP, TRIAL_DROP_S, WIN

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "data" / "raw" / "SHA256SUMS.txt"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), *args], text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def split_fingerprint(split) -> str:
    """Hash of the sorted index arrays — two runs with identical splits
    share it regardless of array order."""
    h = hashlib.sha256()
    for idx in (split.train, split.val, split.test):
        h.update(np.sort(np.asarray(idx)).tobytes())
    return h.hexdigest()


def preprocessing_config() -> dict:
    return {"fs_in": FS_IN, "fs_out": FS_OUT, "band_hz": list(BAND),
            "notch_hz": 60.0, "win": WIN, "step": STEP,
            "trial_drop_s": TRIAL_DROP_S}


def dataset_subset() -> bool:
    return (REPO / "data" / "raw" / "SUBSET.txt").exists()


def collect(*, seed: int, opt: dict, device: str, split,
            extra: dict | None = None) -> dict:
    prov = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "dataset": "grabmyo/1.1.0",
        "dataset_manifest_sha256": (sha256_file(MANIFEST)
                                  if MANIFEST.exists() else None),
        "dataset_subset": dataset_subset(),
        "preprocessing": preprocessing_config(),
        "split_fingerprint": split_fingerprint(split),
        "split_protocol": "v1",
        "seed": seed,
        "optimiser": opt,
        "device": device,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import sklearn
        prov["sklearn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import torch
        prov["torch"] = torch.__version__
    except Exception:
        pass
    if extra:
        prov.update(extra)
    return prov
