"""Plot: for one config, how many questions had a golden source among the retrieved.

Answers "out of 17, how many retrieved sources are present in golden?" by reusing the
same module-code matching as evaluation/score_retrieval.py (a question counts as a hit
if ANY golden code appears anywhere in its retrieved_source list). Simple Found vs
Not-found bar chart.

Usage:
  python plot_hybrid_source_hit.py [config] [predictions.json]
Defaults: config=hybrid, predictions.json next to this script.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_RE = re.compile(r"[A-Z0-9]{3}-\d{2}\.\d{3}")
COLORS = {"vector": "#4a9eff", "graph": "#2ecc71", "hybrid": "#a78bfa"}
MISS_C = "#e74c3c"


def _codes(value) -> set[str]:
    text = " ".join(value) if isinstance(value, (list, tuple)) else str(value or "")
    return {c.upper() for c in CODE_RE.findall(text)}


def has_hit(golden: set[str], retrieved: list[str]) -> bool:
    return any(_codes(name) & golden for name in retrieved)


def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
    here = Path(__file__).resolve().parent
    pred_file = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "predictions.json"
    data = json.loads(pred_file.read_text(encoding="utf-8"))

    rows = [r for r in data if r.get("config") == config]
    if not rows:
        raise SystemExit(f"No rows for config={config!r} in {pred_file}")

    n = len(rows)
    found = sum(
        1 for r in rows
        if has_hit(_codes(r.get("golden_source")), r.get("retrieved_source") or [])
    )
    missed = n - found
    print(f"config={config}: found={found}, not found={missed}, n={n} "
          f"({found / n:.1%})")

    plt.figure(figsize=(6, 5))
    bars = plt.bar(
        ["Found", "Not found"], [found, missed],
        color=[COLORS.get(config, "#4a9eff"), MISS_C], edgecolor="white", width=0.6,
    )
    for b, v in zip(bars, [found, missed]):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.15, str(v),
                 ha="center", va="bottom", fontsize=14, fontweight="bold")
    plt.ylim(0, n + 1)
    plt.ylabel(f"questions (out of {n})")
    plt.title(f"Retrieved source present in golden — {config} ({found}/{n})")
    plt.tight_layout()

    out_path = here / f"plot_source_hit_{config}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
