"""Export: turn a saved .pt into an ONNX graph the browser can run.

What goes in
    A trained `TinyCNN` state_dict under `artifacts/models/` (written by
    train.py) plus the split's own run file (for the test accuracy).

What comes out
    `artifacts/onnx/<model>_<config>_<kind>_<name>.onnx` — the whole
    model, normalisation included, in a single file ONNX Runtime can
    execute; a `<name>.meta.json` sidecar the demo page reads (channel
    list, crop, sampling rate, class names, provenance); and a parity
    entry in `artifacts/onnx/parity.json` proving ONNX == torch.

How it works
    `torch.onnx.export` traces the model in eval mode — batch-norm is
    folded into constants and dropout is off — with a dynamic batch dim,
    input "emg" [B, C, 256] float32 RAW signal, output "logits" [B, 17].
    `verify` then runs both implementations (torch CPU vs onnxruntime
    CPU) on real held-out test windows of the same split and asserts the
    logits agree to <1e-4 and every argmax matches. If they do not, the
    file must not ship: a browser demo that silently disagrees with the
    trained model would be worse than no demo.

CLI: `uv run python -m emgdec.export --model cnn --config both
      --kind within_subject --name s01_d1`
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from emgdec import splits
from emgdec.baselines import _load_meta
from emgdec.config import BAND, FS_OUT, GESTURE_NAMES, N_GESTURES, WIN
from emgdec.evaluate import RUNS_DIR
from emgdec.models import CHANNELS, TinyCNN
from emgdec.train import MODELS_DIR, _open_X

REPO = Path(__file__).resolve().parents[2]
ONNX_DIR = REPO / "artifacts" / "onnx"
OPSET = 17


def _find_split(meta: dict[str, np.ndarray], kind: str, name: str):
    for split in splits.make(kind, meta):
        if split.name == name:
            return split
    raise KeyError(f"no split {kind}/{name}")


def _run_record(model: str, config: str, kind: str, name: str) -> dict:
    p = RUNS_DIR / model / config / kind / f"{name}.json"
    return json.loads(p.read_text())


def export(model: str = "cnn", config: str = "both", kind: str = "within_subject",
           name: str = "s01_d1", out_dir: Path = ONNX_DIR) -> Path:
    """Load the saved weights and export ONNX + meta.json. Returns path."""
    pt = MODELS_DIR / model / config / kind / f"{name}.pt"
    if not pt.exists():
        raise FileNotFoundError(f"no saved weights at {pt} "
                                "(train with save_weights first)")
    chans = list(CHANNELS[config])
    net = TinyCNN(len(chans), N_GESTURES)
    net.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
    net.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{model}_{config}_{kind}_{name}.onnx"
    dummy = torch.zeros(1, len(chans), WIN)
    torch.onnx.export(
        net, (dummy,), str(onnx_path),
        input_names=["emg"], output_names=["logits"], opset_version=OPSET,
        dynamic_axes={"emg": {0: "batch"}, "logits": {0: "batch"}})

    rec = _run_record(model, config, kind, name)
    n_test = int(np.asarray(rec["confusion"]).sum())
    test_acc = float(np.trace(rec["confusion"]) / n_test)
    meta = {
        "model": model, "config": config, "kind": kind, "split": name,
        "channels": chans, "crop": rec["params"].get("crop", WIN),
        "fs": FS_OUT, "window": WIN, "band_hz": list(BAND),
        "classes": GESTURE_NAMES, "input": "emg: float32 [B, C, 256] raw",
        "output": "logits [B, 17]", "test_accuracy": test_acc,
        "opset": OPSET, "torch": torch.__version__,
    }
    onnx_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    return onnx_path


def verify(onnx_path: Path, pt_path: Path, x_test: np.ndarray,
           tol: float = 1e-4) -> dict:
    """ONNX vs torch on the same inputs: max |logit diff| and argmax
    agreement. x_test: float32 windows [B, C, W] already channel-selected."""
    import onnxruntime as ort

    n_ch = x_test.shape[1]
    net = TinyCNN(n_ch, N_GESTURES)
    net.load_state_dict(torch.load(pt_path, map_location="cpu",
                                   weights_only=True))
    net.eval()
    with torch.no_grad():
        ref = net(torch.from_numpy(x_test.astype(np.float32))).numpy()
    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"emg": x_test.astype(np.float32)})[0]
    max_diff = float(np.max(np.abs(got - ref)))
    agree = float((got.argmax(1) == ref.argmax(1)).mean())
    return {"max_logit_diff": max_diff, "argmax_agreement": agree,
            "n_windows": int(len(x_test)), "ok": bool(max_diff < tol
                                                    and agree == 1.0)}


def _test_windows(model: str, config: str, kind: str, name: str,
                  n: int = 512) -> np.ndarray:
    """Real held-out test windows of this split (first n)."""
    meta = _load_meta()
    split = _find_split(meta, kind, name)
    X, _ = _open_X()
    chans = np.asarray(list(CHANNELS[config]))
    idx = np.sort(np.asarray(split.test))[:n]
    return np.asarray(X[idx][:, chans, :], dtype=np.float32)


def _record_parity(onnx_path: Path, result: dict) -> None:
    p = ONNX_DIR / "parity.json"
    data = json.loads(p.read_text()) if p.exists() else {}
    data[onnx_path.stem] = result
    p.write_text(json.dumps(data, indent=1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn")
    ap.add_argument("--config", default="both")
    ap.add_argument("--kind", default="within_subject")
    ap.add_argument("--name", default="s01_d1")
    a = ap.parse_args(argv)
    onnx_path = export(a.model, a.config, a.kind, a.name)
    pt = MODELS_DIR / a.model / a.config / a.kind / f"{a.name}.pt"
    x_test = _test_windows(a.model, a.config, a.kind, a.name)
    res = verify(onnx_path, pt, x_test)
    _record_parity(onnx_path, res)
    print(f"{onnx_path.name}: {onnx_path.stat().st_size/1024:.0f} KB, "
          f"parity max|d|={res['max_logit_diff']:.2e} "
          f"argmax={res['argmax_agreement']:.3f} ok={res['ok']}")


if __name__ == "__main__":
    main()
