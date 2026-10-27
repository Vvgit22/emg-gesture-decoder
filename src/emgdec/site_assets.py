"""Site assets: pack everything the browser demo needs into artifacts/site/.

What goes in
    The float16 window memmap, the meta table, the two demo ONNX models
    (`cnn_both_within_subject_s01_d1`, `cnn_both_cross_subject_fold3`),
    `artifacts/results.json` and the finished figures.

What comes out
    `artifacts/site/` — a self-contained folder the portfolio copies:

    - `results.json`  copy of the aggregated results (the audit table).
    - `figures/`      the five published SVGs.
    - `models/seen.onnx|unseen.onnx` + `.meta.json`  the two demo models.
    - `demo.bin`      17 gestures x 28 channels x 2048 samples, int16
                      little-endian — the central 2.0 s of subject 1,
                      session 1, TRIAL 6, one clip per gesture.
    - `demo.json`     everything the page needs to read demo.bin and to
                      describe it: fs, shapes, the dequantisation scale,
                      channel groups, class names, provenance, and the
                      measured accuracy of each model on these clips.

How the demo clip is built
    Trial 6 is chosen because it is inside the TEST set of BOTH models —
    verified here, not assumed: subject 1 must be in fold3's test set and
    s01_d1's test trials must be {6,7}. The trial's 35 stored windows
    overlap by half (step 128 of 256); consecutive windows must agree
    bit-for-bit on the overlap or the stitch would be a lie, so that is
    asserted before the bytes are trusted. Stitching gives 4608 samples;
    the central [1280:3328] are kept. One global scale = max|x| / 32000
    maps the whole file into int16 (units are mV-ish float16; int16 keeps
    the file at ~1.9 MB instead of ~3.8 MB float32).

What would go wrong if skipped
    The web page would need the whole 7.7 GB memmap, or hand-copied
    numbers that can drift from the real results. This file makes the
    demo auditable: the models and the clips are the same ones the
    results table talks about.

CLI: `uv run python -m emgdec.site_assets`
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from emgdec import splits
from emgdec.baselines import _load_meta
from emgdec.config import GESTURE_NAMES, N_GESTURES
from emgdec.evaluate import RUNS_DIR
from emgdec.models import CHANNELS
from emgdec.train import _open_X

REPO = Path(__file__).resolve().parents[2]
SITE = REPO / "artifacts" / "site"
ONNX_DIR = REPO / "artifacts" / "onnx"
FIG_DIR = REPO / "artifacts" / "figures"

DEMO_SUBJECT, DEMO_SESSION, DEMO_TRIAL = 1, 1, 6
SEEN = ("within_subject", "s01_d1")
UNSEEN = ("cross_subject", "fold3")
STEP, WIN, SEG = 128, 256, 2048
N_WINDOWS = (4608 - WIN) // STEP + 1  # 35 windows per trial
FIGS = ["ablation_window_length.svg",
        "per_class_f1_cnn_both_cross_subject.svg",
        "confusion_cnn_both_cross_subject.svg",
        "confusion_lda_both_cross_subject.svg",
        "accuracy_by_split.svg"]


def _check_demo_is_held_out(meta: dict[str, np.ndarray]) -> None:
    """The demo clip must be in the TEST side of both exported models."""
    sp = {s.name: s for s in splits.make("cross_subject", meta)}[UNSEEN[1]]
    assert DEMO_SUBJECT in set(meta["subject"][sp.test].tolist()), \
        "subject 1 is not in fold3's test set"
    sp = {s.name: s for s in splits.make("within_subject", meta)}[SEEN[1]]
    trials = set(meta["trial"][sp.test].tolist())
    assert trials == {6, 7}, f"s01_d1 test trials are {trials}, not {{6,7}}"
    assert DEMO_TRIAL in trials


def _trial_windows(X: np.ndarray, meta: dict[str, np.ndarray],
                   gesture: int) -> np.ndarray:
    """The 35 ordered windows of (subject 1, session 1, gesture, trial 6)."""
    sel = np.flatnonzero(
        (meta["subject"] == DEMO_SUBJECT) &
        (meta["session"] == DEMO_SESSION) &
        (meta["trial"] == DEMO_TRIAL) & (meta["gesture"] == gesture))
    assert len(sel) == N_WINDOWS, \
        f"gesture {gesture}: {len(sel)} windows, expected {N_WINDOWS}"
    return np.asarray(X[sel])


def stitch(windows: np.ndarray) -> np.ndarray:
    """Rebuild the continuous trace from overlapping windows.

    Window k and k+1 share 128 samples; they must be bit-identical or the
    stitch is untrustworthy. Result: [C, 4608].
    """
    for k in range(len(windows) - 1):
        a = windows[k][:, STEP:]
        b = windows[k + 1][:, :STEP]
        assert np.array_equal(a, b), \
            f"overlap mismatch between windows {k} and {k+1}"
    return np.concatenate(
        [windows[0]] + [w[:, STEP:] for w in windows[1:]], axis=1)


def _stitch_trial(X: np.ndarray, meta: dict[str, np.ndarray],
                  gesture: int) -> np.ndarray:
    """Continuous 4608-sample trace for one gesture, cropped to 2048."""
    sig = stitch(_trial_windows(X, meta, gesture))
    assert sig.shape[1] == 4608, sig.shape
    return sig[:, 1280:1280 + SEG]


def _quantize(x: np.ndarray, scale: float) -> np.ndarray:
    return np.round(x.astype(np.float32) / scale).astype("<i2")


def _onnx_argmax(sess, windows: np.ndarray) -> np.ndarray:
    return sess.run(["logits"], {"emg": windows.astype(np.float32)})[0].argmax(1)


def _demo_metrics() -> dict:
    """Run each ONNX over the 255 demo windows twice — original float16
    and int16-dequantised — and report agreement + accuracy."""
    import onnxruntime as ort

    demo = np.fromfile(SITE / "demo.bin", dtype="<i2").reshape(17, 28, SEG)
    meta = json.loads((SITE / "demo.json").read_text())
    scale = meta["scale"]
    deq = demo.astype(np.float32) * scale
    starts = np.arange(0, SEG - WIN + 1, STEP)
    labels = np.repeat(np.arange(17), len(starts))
    out = {}
    for key in ("seen", "unseen"):
        sess = ort.InferenceSession(str(SITE / "models" / f"{key}.onnx"),
                                    providers=["CPUExecutionProvider"])
        wins_q = np.stack([deq[g, :, s:s + WIN]
                           for g in range(17) for s in starts])
        raw = _raw_windows()
        wins_f = np.stack([raw[g][:, s:s + WIN]
                           for g in range(17) for s in starts])
        a_q, a_f = _onnx_argmax(sess, wins_q), _onnx_argmax(sess, wins_f)
        out[key] = {
            "accuracy_on_demo_clips": float((a_q == labels).mean()),
            "quantisation_agreement": float((a_q == a_f).mean()),
        }
    return out


_raw_cache: np.ndarray | None = None


def _raw_windows() -> np.ndarray:
    """The original float16 demo segments [17, 28, 2048] (pre-quantise)."""
    global _raw_cache
    if _raw_cache is None:
        meta = _load_meta()
        X, _ = _open_X()
        _raw_cache = np.stack([_stitch_trial(X, meta, g)
                               for g in range(1, N_GESTURES + 1)])
    return _raw_cache


def build() -> dict:
    meta = _load_meta()
    _check_demo_is_held_out(meta)
    X, _ = _open_X()
    (SITE / "models").mkdir(parents=True, exist_ok=True)
    (SITE / "figures").mkdir(parents=True, exist_ok=True)

    clips = np.stack([_stitch_trial(X, meta, g)
                      for g in range(1, N_GESTURES + 1)])
    scale = float(np.abs(clips).max() / 32000)
    q = _quantize(clips, scale)
    (SITE / "demo.bin").write_bytes(q.tobytes(order="C"))
    assert (SITE / "demo.bin").stat().st_size == 17 * 28 * SEG * 2

    for key, (kind, name) in (("seen", SEEN), ("unseen", UNSEEN)):
        shutil.copy(ONNX_DIR / f"cnn_both_{kind}_{name}.onnx",
                    SITE / "models" / f"{key}.onnx")
        shutil.copy(ONNX_DIR / f"cnn_both_{kind}_{name}.meta.json",
                    SITE / "models" / f"{key}.meta.json")

    shutil.copy(REPO / "artifacts" / "results.json", SITE / "results.json")
    for f in FIGS:
        shutil.copy(FIG_DIR / f, SITE / "figures" / f)

    starts = np.arange(0, SEG - WIN + 1, STEP)
    run_acc = {}
    for key, (kind, name) in (("seen", SEEN), ("unseen", UNSEEN)):
        rec = json.loads((RUNS_DIR / "cnn" / "both" / kind
                          / f"{name}.json").read_text())
        cm = np.asarray(rec["confusion"])
        run_acc[key] = float(np.trace(cm) / cm.sum())

    demo_json = {
        "fs": 1024, "n_gestures": 17, "n_channels": 28, "n_samples": SEG,
        "window": WIN, "step": STEP, "scale": scale,
        "channels": {"forearm": list(CHANNELS["forearm"]),
                     "wrist": list(CHANNELS["wrist"])},
        "classes": GESTURE_NAMES,
        "source": {"dataset": "GrabMyo v1.1.0", "subject": DEMO_SUBJECT,
                   "session": DEMO_SESSION, "trial": DEMO_TRIAL,
                   "licence": "CC BY 4.0", "doi": "10.13026/89dm-f662"},
        "models": {k: {"split_test_accuracy": run_acc[k]}
                   for k in ("seen", "unseen")},
    }
    (SITE / "demo.json").write_text(json.dumps(demo_json, indent=1))
    metrics = _demo_metrics()
    for k in ("seen", "unseen"):
        demo_json["models"][k].update(metrics[k])
    # For the browser-vs-Python parity check: argmax of the first 15
    # windows of gesture 1 on the DEQUANTISED stream.
    import onnxruntime as ort
    deq = q[0].astype(np.float32) * scale
    wins = np.stack([deq[:, s:s + WIN] for s in starts])
    for k in ("seen", "unseen"):
        sess = ort.InferenceSession(str(SITE / "models" / f"{k}.onnx"),
                                    providers=["CPUExecutionProvider"])
        demo_json[f"first_15_argmax_{k}"] = _onnx_argmax(sess, wins).tolist()
    (SITE / "demo.json").write_text(json.dumps(demo_json, indent=1))

    for k in ("seen", "unseen"):
        m = demo_json["models"][k]
        print(f"{k}: demo-acc={m['accuracy_on_demo_clips']:.3f} "
              f"quant-agree={m['quantisation_agreement']:.3f} "
              f"split-test={m['split_test_accuracy']:.3f}")
    print(f"site assets -> {SITE} "
          f"(demo.bin {(SITE / 'demo.bin').stat().st_size} bytes)")
    return demo_json


def main() -> None:
    build()


if __name__ == "__main__":
    main()
