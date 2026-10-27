"""Report: keep the README results table honest.

What goes in: `artifacts/results.json` and the README's
`<!-- results:start -->` / `<!-- results:end -->` markers.
What comes out: the README with the block between the markers rewritten
from the JSON — a model x config table with mean ± std per split kind,
plus the demo link. No number in that block is typed by hand, so it can
never drift from the audit trail.

CLI: `uv run python -m emgdec.report`
"""
from __future__ import annotations

import json
from pathlib import Path

from emgdec.evaluate import exp_results
RESULTS = exp_results("protocol_v1")

REPO = Path(__file__).resolve().parents[2]
README = REPO / "README.md"
START, END = "<!-- results:start -->", "<!-- results:end -->"

KINDS = ["within_subject", "cross_day", "cross_subject"]
KIND_HEAD = {"within_subject": "Within-subject", "cross_day": "Cross-day",
             "cross_subject": "Cross-subject"}


def table(results: dict) -> str:
    cells = results["cells"]
    lines = [
        "| Model | Channels | " + " | ".join(KIND_HEAD[k] for k in KINDS)
        + " |",
        "|---|---|" + "---|" * len(KINDS),
    ]
    for model in results.get("model_order",
                             sorted(cells)):
        for config in ("forearm", "wrist", "both"):
            row = [model, config]
            if config not in cells.get(model, {}):
                continue
            for kind in KINDS:
                cell = cells[model][config].get(kind)
                row.append("—" if cell is None else
                           f"{cell['accuracy_mean'] * 100:.1f} ± "
                           f"{cell['accuracy_std'] * 100:.1f}")
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Accuracies in %, mean ± std across splits "
                 "(129 within-subject, 43 cross-day, 5 cross-subject). "
                 "Chance is 1/17 ≈ 5.9%.")
    return "\n".join(lines)


def rewrite(readme: Path = README, results: dict | None = None) -> str:
    if results is None:
        results = json.loads(RESULTS.read_text())
    text = readme.read_text()
    a, b = text.index(START), text.index(END)
    block = (START + "\n\n" + table(results) + "\n\n"
             "Live demo: https://vvgit22.github.io/emg/\n\n")
    readme.write_text(text[:a] + block + text[b:])
    return block


def main() -> None:
    block = rewrite()
    print(f"README results block rewritten:\n\n{block}")


if __name__ == "__main__":
    main()
