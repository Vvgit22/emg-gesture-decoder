"""Generate the portfolio thumbnail from the packed demo.bin.

Renders ~8 real EMG traces (mixed forearm/wrist) of one gesture in a
plasma-ish palette on a dark tile, with a small 17-cell grid bottom-right
and one cell lit. Output: portfolio's public/thumbs/emg-decoder.webp
(2400x1500 canvas -> 1200x750 webp). Run:

    python scripts/thumbnail.py path/to/portfolio
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "artifacts" / "site" / "demo.bin"
SEG, NC = 2048, 28
GESTURE = 15  # "Hand Close" — busy, readable trace
CHS = [0, 3, 7, 11, 15, 18, 22, 26]  # forearm then wrist


def plasma(t: float) -> tuple[float, float, float]:
    stops = np.array([
        [0.05, 0.03, 0.53], [0.42, 0.13, 0.57], [0.80, 0.28, 0.50],
        [0.97, 0.58, 0.25], [0.94, 0.98, 0.62],
    ])
    xs = np.linspace(0, 1, len(stops))
    return tuple(np.interp(t, xs, stops[:, i]) for i in range(3))


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp")
    clips = np.fromfile(BIN, dtype="<i2").reshape(17, NC, SEG)
    g = clips[GESTURE].astype(np.float32)

    fig = plt.figure(figsize=(12, 7.5), dpi=200)
    fig.patch.set_facecolor("#0b1020")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0b1020")
    # faint radial glow behind centre
    yy, xx = np.mgrid[0:100, 0:100]
    glow = np.exp(-(((xx - 55) / 35) ** 2 + ((yy - 45) / 30) ** 2))
    ax.imshow(glow, extent=[0, SEG, 0, len(CHS)], aspect="auto",
              cmap="magma", alpha=0.10, zorder=0)

    for i, ch in enumerate(CHS):
        tr = g[ch]
        tr = tr / np.abs(tr).max()
        for t, w, a in ((0.0, 4.0, 0.18), (0.5, 1.6, 0.9)):
            ax.plot(np.arange(SEG), len(CHS) - 1 - i + tr * 0.32 + t * 0.02,
                    color=plasma(0.15 + 0.7 * i / (len(CHS) - 1)),
                    lw=w, alpha=a, solid_capstyle="round")
    ax.set_xlim(0, SEG); ax.set_ylim(-0.6, len(CHS) - 0.2)
    ax.axis("off")

    # 17-cell grid, bottom-right, true gesture lit
    gx, gy, s, pad = 0.78, 0.06, 0.030, 0.008
    for i in range(17):
        r, c = divmod(i, 6)
        rect = plt.Rectangle(
            (gx + c * (s + pad), gy + r * (s + pad)), s, s,
            transform=ax.transAxes,
            facecolor=(plasma(0.9) if i == GESTURE else "#1a2140"),
            edgecolor=plasma(0.9) if i == GESTURE else "#2a3358",
            linewidth=1.2, zorder=5)
        ax.add_patch(rect)

    png = out_dir / "emg-decoder-thumb.png"
    fig.savefig(png, dpi=200, facecolor="#0b1020")
    plt.close(fig)
    print(f"png -> {png}")
    try:
        from PIL import Image
        im = Image.open(png).resize((1200, 750), Image.LANCZOS)
        webp = out_dir / "emg-decoder.webp"
        im.save(webp, "WEBP", quality=82)
        print(f"webp -> {webp} ({webp.stat().st_size // 1024} KB)")
    except ImportError:
        print("PIL unavailable; convert the PNG separately")


if __name__ == "__main__":
    main()
