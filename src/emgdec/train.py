"""Training: fit the TinyCNN on one split, the same rules as the LDA sweep.

What goes in
    The float16 window memmap `data/memmap/X.f16` [N, 28, 256], the meta
    table, and a split kind + channel config + optional time crop.

What comes out
    One `RunRecord` per split under `artifacts/runs/cnn*/<config>/<kind>/`,
    written through `evaluate.save_run` — the only writer of results.
    Where the demo needs weights, a `state_dict` is also saved under
    `artifacts/models/` (gitignored).

How it works
    `splits.assert_no_leak` runs first, exactly as baselines.py does.
    The split's train/val/test windows are pulled from the memmap once,
    into RAM as float16 (indices sorted first so reads are sequential —
    the ~5.4 GB cross-subject train set stays sequential and cheap to
    page). Windows are converted to float32 per batch on the device, so
    peak RAM stays at the float16 copy, not double.

    Labels are 0-based (gesture - 1), as in baselines. Optimiser AdamW
    (lr 2e-3, weight decay 1e-2), cosine LR over max_epochs, batch 256,
    plain cross-entropy — no augmentation, no label smoothing: the base
    model is kept simple so ablations can show what each extra buys.
    Early stopping watches VALIDATION accuracy only; the best-val weights
    are restored before the test set is scored exactly once, at the end.
    The test windows are never touched during training.

    `crop` centre-crops the time axis (256 → 250 ms, 128 → 125 ms,
    64 → 62.5 ms) and is applied identically to train/val/test. Crop 256
    keeps the model name "cnn"; smaller crops are recorded as models
    "cnn_w128" / "cnn_w64".

Seeding and reproducibility
    `torch.manual_seed(SEED + fold_index)` and the same for numpy, with a
    torch Generator for shuffling. CPU runs are bitwise reproducible; MPS
    runs are reproducible up to float-order noise. Device and torch
    version are stored in params.

CLI: `uv run python -m emgdec.train
      --kind within_subject|cross_day|cross_subject|all
      --config forearm|wrist|both|all [--crop 256] [--device mps|cpu]
      [--max-epochs N] [--patience P] [--limit N] [--force]`
    --limit runs only the first N splits (smoke runs); splits whose run
    file already exists are skipped unless --force, so a killed sweep
    resumes where it stopped.
"""
from __future__ import annotations

import argparse
import copy
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from emgdec import metrics, splits
from emgdec.baselines import _load_meta
from emgdec.config import N_GESTURES, SEED
from emgdec.evaluate import RUNS_DIR, RunRecord, exp_runs, save_run
from emgdec.provenance import collect as prov_collect, sha256_file
from emgdec.models import CHANNELS, TinyCNN, n_params

REPO = Path(__file__).resolve().parents[2]
MEMMAP_DIR = REPO / "data" / "memmap"
X_F16 = MEMMAP_DIR / "X.f16"
MODELS_DIR = REPO / "artifacts" / "models"

# Per-kind training budget. Cross-subject trains on ~35x more windows, so
# it gets fewer epochs and less patience — an honest budget decision,
# recorded in DECISIONS.md.
BUDGET = {
    "within_subject": {"max_epochs": 40, "patience": 6},
    "cross_day": {"max_epochs": 40, "patience": 6},
    "cross_subject": {"max_epochs": 15, "patience": 3},
}

HYPER = {"optimizer": "adamw", "lr": 2e-3, "weight_decay": 1e-2,
         "batch": 256, "loss": "cross_entropy", "schedule": "cosine"}


def _centre_crop(x: np.ndarray, crop: int) -> np.ndarray:
    """Keep the middle `crop` samples of the time axis (last axis)."""
    w = x.shape[-1]
    if crop >= w:
        return x
    lo = (w - crop) // 2
    return x[..., lo:lo + crop]


def _load_windows(X: np.ndarray, idx: np.ndarray, chans: np.ndarray,
                  crop: int, chunk: int = 20000) -> np.ndarray:
    """Pull rows `idx`, channels `chans`, centre-cropped, as float16 in RAM.

    Chunked: at most `chunk` rows are materialised from the memmap at a
    time, channel-selection and crop applied per chunk into a single
    preallocated output. Peak extra memory = one chunk, not a second full
    copy — the old `X[idx][:, chans, :]` made two 5.4 GB copies for
    config both and thrashed the machine.
    """
    idx = np.sort(np.asarray(idx))
    lo, hi = _crop_bounds(X.shape[2], crop)
    out = np.empty((len(idx), len(chans), hi - lo), dtype=np.float16)
    for i in range(0, len(idx), chunk):
        sl = idx[i:i + chunk]
        out[i:i + chunk] = np.asarray(X[sl][:, chans, lo:hi])
    return out


def _crop_bounds(w: int, crop: int) -> tuple[int, int]:
    crop = min(crop, w)
    lo = (w - crop) // 2
    return lo, lo + crop


def _norm_stats(Xtr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std via float64 sum + sum-of-squares accumulation
    in chunks — numerically equal to numpy's mean/std to ~1e-7, but never
    materialises a float32 copy (the old `tr32 = Xtr.astype(np.float32)`
    was another 10.8 GB).
    """
    n, c, w = Xtr.shape
    s1 = np.zeros(c, np.float64)
    s2 = np.zeros(c, np.float64)
    cnt = np.float64(n) * w
    for i in range(0, n, 20000):
        x = np.asarray(Xtr[i:i + 20000], dtype=np.float64)
        s1 += x.sum(axis=(0, 2))
        s2 += (x * x).sum(axis=(0, 2))
    mean = s1 / cnt
    var = np.clip(s2 / cnt - mean * mean, 0, None)
    return mean, np.sqrt(var)


def _batch_iter(x: np.ndarray, y: np.ndarray, batch: int,
                gen: torch.Generator):
    order = torch.randperm(len(x), generator=gen).tolist()
    for lo in range(0, len(x), batch):
        sel = order[lo:lo + batch]
        yield x[sel], y[sel]


@torch.no_grad()
def _accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray,
              device: torch.device, batch: int = 1024) -> float:
    model.eval()
    correct = 0
    for lo in range(0, len(x), batch):
        xb = torch.from_numpy(x[lo:lo + batch].astype(np.float32)).to(device)
        correct += (model(xb).argmax(1).cpu().numpy()
                    == y[lo:lo + batch]).sum()
    return correct / len(x)


@torch.no_grad()
def _confusion(model: nn.Module, x: np.ndarray, y: np.ndarray,
               device: torch.device, batch: int = 1024) -> np.ndarray:
    model.eval()
    preds = np.empty(len(x), dtype=np.int64)
    for lo in range(0, len(x), batch):
        xb = torch.from_numpy(x[lo:lo + batch].astype(np.float32)).to(device)
        preds[lo:lo + batch] = model(xb).argmax(1).cpu().numpy()
    return metrics.confusion(y, preds, N_GESTURES)


def fit(model: nn.Module, Xtr: np.ndarray, ytr: np.ndarray,
        Xva: np.ndarray | None, yva: np.ndarray | None,
        *, device: torch.device, max_epochs: int, patience: int,
        seed: int, batch: int = HYPER["batch"], lr: float = HYPER["lr"],
        weight_decay: float = HYPER["weight_decay"]
        ) -> tuple[dict, float | None, int]:
    """The training loop: AdamW + cosine, early stop on val accuracy.

    Returns (best_state_dict, best_val_accuracy, epochs_run). With no
    validation set, runs all epochs and returns the final weights.
    """
    # Caller seeds globally before model init; we seed only the shuffle
    # generator here so both paths end up deterministic.
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max_epochs)
    loss_fn = nn.CrossEntropyLoss()
    best_state, best_val, epochs_run, stale = None, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb_np, yb_np in _batch_iter(Xtr, ytr, batch, gen):
            xb = torch.from_numpy(xb_np.astype(np.float32)).to(device)
            yb = torch.from_numpy(yb_np.astype(np.int64)).to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()
        epochs_run = epoch
        if Xva is None or len(Xva) == 0:
            continue
        va = _accuracy(model, Xva, yva, device)
        if best_val is None or va > best_val:
            best_val, stale = va, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:  # no val set: final weights are the answer
        best_state = copy.deepcopy(model.state_dict())
    return best_state, best_val, epochs_run


def _should_save_pt(kind: str, config: str) -> bool:
    """Which splits get a saved .pt: all cross_subject folds (15 small
    files) and every 'both' config (the demo and the exporter use those).
    Ablation models are not exported, so no .pt for them."""
    return kind == "cross_subject" or config == "both"


def train_split(X: np.ndarray, meta: dict[str, np.ndarray], split,
                config: str, *, crop: int = 256, seed_offset: int = 0,
                device: torch.device | str | None = None,
                max_epochs: int | None = None, patience: int | None = None,
                seed: int = SEED, fold_index: int = 0,
                batch: int = HYPER["batch"], lr: float = HYPER["lr"],
                weight_decay: float = HYPER["weight_decay"],
                save_weights: bool | None = None,
                x_dir: Path | None = None,
                max_rss_gb: float | None = 12.0,
                models_dir: Path = MODELS_DIR) -> RunRecord:
    """Train one split, return (and save) the RunRecord; test scored once."""
    splits.assert_no_leak(split, meta)
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    budget = BUDGET.get(split.kind, BUDGET["within_subject"])
    max_epochs = max_epochs or budget["max_epochs"]
    patience = patience or budget["patience"]

    chans = np.asarray(list(CHANNELS[config]))
    est_gb = len(split.train) * len(chans) * crop * 2 / 1e9
    print(f"  loading train: ~{est_gb:.1f} GB float16", flush=True)
    Xtr = _load_windows(X, split.train, chans, crop)
    Xva = _load_windows(X, split.val, chans, crop)
    Xte = _load_windows(X, split.test, chans, crop)
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print(f"  loaded: RSS {rss_gb:.1f} GB", flush=True)
    if max_rss_gb is not None and rss_gb > max_rss_gb:
        raise SystemExit(
            f"RSS {rss_gb:.1f} GB exceeds --max-rss-gb {max_rss_gb} "
            "after loading; aborting before training")
    ytr = meta["gesture"][np.sort(np.asarray(split.train))].astype(int) - 1
    yva = meta["gesture"][np.sort(np.asarray(split.val))].astype(int) - 1
    yte = meta["gesture"][np.sort(np.asarray(split.test))].astype(int) - 1

    # Seed BEFORE the model exists, so init order is identical whether a
    # split runs alone or after another (resume-order independence).
    torch.manual_seed(seed + 10000 * seed_offset + fold_index)
    np.random.seed(seed + 10000 * seed_offset + fold_index)
    model = TinyCNN(len(chans), N_GESTURES)
    # Per-channel stats over TRAIN windows only; clamped inside the model.
    _m, _s = _norm_stats(Xtr)
    mean = torch.from_numpy(_m.astype(np.float32))
    std = torch.from_numpy(_s.astype(np.float32))
    model.set_normalisation(mean, std)
    model.to(device)

    t0 = time.time()
    best_state, best_val, epochs_run = fit(
        model, Xtr, ytr, Xva, yva, device=device, max_epochs=max_epochs,
        patience=patience, seed=seed + 10000 * seed_offset + fold_index, batch=batch, lr=lr,
        weight_decay=weight_decay)
    model.load_state_dict(best_state)
    train_seconds = time.time() - t0

    cm = _confusion(model, Xte, yte, device)
    name = "cnn" if crop == 256 else f"cnn_w{crop}"
    split_name = split.name + (f"_seed{seed_offset}" if seed_offset else "")
    params = dict(HYPER, crop=crop, max_epochs=max_epochs, patience=patience,
                  device=str(device), n_params=n_params(model),
                  torch=torch.__version__)
    if save_weights is None:
        save_weights = _should_save_pt(split.kind, config) and crop == 256
    prov = prov_collect(
        seed=seed + 10000 * seed_offset + fold_index, opt=params, device=str(device), split=split,
        extra={"selected_epoch": epochs_run,
               "selection_rule": f"max val accuracy, patience {patience}",
               "preprocessing_mode": (json.loads(
                   (Path(x_dir) / "shape.json").read_text()
               ).get("preprocess", "offline") if x_dir else "offline"),
               "x_dir": str(x_dir) if x_dir else str(MEMMAP_DIR)})
    if save_weights:
        p = (Path(models_dir) / name / config / split.kind
             / f"{split_name}.pt")
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), p)
        prov["checkpoint_sha256"] = sha256_file(p)
    return RunRecord(
        model=name, config=config, kind=split.kind, name=split_name,
        confusion=cm.tolist(),
        n_train=len(split.train), n_val=len(split.val),
        n_test=len(split.test), train_seconds=train_seconds,
        seed=seed + 10000 * seed_offset + fold_index, params=params,
        val_accuracy=best_val, epochs_run=epochs_run,
        provenance=prov)


def _open_X(x_dir: Path = MEMMAP_DIR) -> tuple[np.memmap, dict]:
    shape = json.loads((Path(x_dir) / "shape.json").read_text())
    return np.memmap(Path(x_dir) / "X.f16", dtype=np.float16, mode="r",
                     shape=(shape["N"], shape["C"], shape["W"])), shape


def _run_path(model: str, config: str, kind: str, name: str) -> Path:
    return RUNS_DIR / model / config / kind / f"{name}.json"


def _prov_matches(rec_path: Path, split, seed: int | None = None) -> tuple[bool, str]:
    # A finished record counts for resume only if git commit, preprocessing
    # config and split fingerprint match what this code would produce now.
    try:
        d = json.loads(rec_path.read_text())
        prov = d.get("provenance") or {}
        if not prov:
            return False, "no provenance"
        cur = prov_collect(seed=seed or 0, opt={}, device="-", split=split)
        if prov.get("git_commit") != cur["git_commit"]:
            return False, "git commit differs"
        if prov.get("preprocessing") != cur["preprocessing"]:
            return False, "preprocessing config differs"
        if prov.get("split_fingerprint") != cur["split_fingerprint"]:
            return False, "split fingerprint differs"
        if prov.get("seed") != cur.get("seed"):
            return False, "training seed differs"
        if d.get("status") != "finished":
            return False, "status is not finished"
        return True, ""
    except Exception as e:
        return False, f"unreadable record: {e}"


def run(kind: str, config: str, *, crop: int = 256,
        device: str | None = None, max_epochs: int | None = None,
        patience: int | None = None, limit: int | None = None,
        force: bool = False, experiment: str = "dev",
        mode: str = "fresh", seed_offset: int = 0,
        protocol: str = "v1", x_dir: Path = MEMMAP_DIR,
        max_rss_gb: float | None = 12.0) -> int:
    if mode == "report":
        from emgdec.evaluate import main as eval_main
        eval_main(["--experiment", experiment])
        return 0
    meta = _load_meta()
    X, shape = _open_X(x_dir)
    runs_dir = exp_runs(experiment)
    model = "cnn" if crop == 256 else f"cnn_w{crop}"
    kinds = list(splits.ALL) if kind == "all" else [kind]
    configs = list(CHANNELS) if config == "all" else [config]
    n = 0
    for k in kinds:
        for cfg in configs:
            if protocol != "v1" and k != "cross_subject":
                continue
            existing = runs_dir / model / cfg / k
            if (mode == "fresh" and not force and existing.exists()
                    and any(existing.glob("*.json"))):
                raise SystemExit(
                    f"fresh mode: {existing} already has runs; "
                    "use --mode resume or --force")
            for i, split in enumerate(splits.make(k, meta, protocol)):
                if limit is not None and i >= limit:
                    break
                rec_name = split.name + \
                    (f"_seed{seed_offset}" if seed_offset else "")
                rec_path = runs_dir / model / cfg / k / f"{rec_name}.json"
                if rec_path.exists() and not force:
                    if mode == "fresh":
                        raise SystemExit(
                            f"fresh mode: {rec_path} exists; "
                            "use --mode resume or --force")
                    ok, why = _prov_matches(
                        rec_path, split,
                        seed=SEED + 10000 * seed_offset + i)
                    if not ok:
                        raise SystemExit(
                            f"resume mode: {split.name} mismatch: {why}")
                    continue
                rec = train_split(X, meta, split, cfg, crop=crop,
                                  device=device, max_epochs=max_epochs,
                                  patience=patience, fold_index=i,
                                  seed_offset=seed_offset, x_dir=x_dir,
                                  max_rss_gb=max_rss_gb,
                                  models_dir=MODELS_DIR / experiment)
                save_run(rec, runs_dir)
                n += 1
                acc = metrics.accuracy(np.asarray(rec.confusion))
                print(f"{model}/{cfg}/{k}/{split.name}: "
                      f"ep={rec.epochs_run} val={rec.val_accuracy or 0:.3f} "
                      f"test={acc:.3f} {rec.train_seconds:.0f}s", flush=True)
        peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
        print(f"{k}: peak RSS {peak_gb:.1f} GB", flush=True)
    return n
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="all",
                    choices=["within_subject", "cross_day",
                             "cross_subject", "all"])
    ap.add_argument("--config", default="all",
                    choices=["forearm", "wrist", "both", "all"])
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--experiment", default="dev")
    ap.add_argument("--mode", choices=["fresh", "resume", "report"],
                    default="fresh")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--protocol", choices=["v1", "v2"], default="v1")
    ap.add_argument("--x", default=None,
                    help="override window memmap dir (e.g. data/memmap_causal)")
    ap.add_argument("--max-rss-gb", type=float, default=12.0)
    a = ap.parse_args(argv)
    n = run(a.kind, a.config, crop=a.crop, device=a.device,
            max_epochs=a.max_epochs, patience=a.patience,
            limit=a.limit, force=a.force, experiment=a.experiment,
            mode=a.mode, seed_offset=a.seed_offset, protocol=a.protocol,
            x_dir=Path(a.x) if a.x else MEMMAP_DIR,
            max_rss_gb=a.max_rss_gb)
    print(f"done: {n} runs")


if __name__ == "__main__":
    main()
