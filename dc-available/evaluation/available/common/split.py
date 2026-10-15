"""One-time split of the combined retrieval.json into per-leg files.

Reads   evaluation/results/synthetic/retrieval.json
Writes  evaluation/results/synthetic/{vector,graph,hybrid}.json

This only reshapes the existing file — no retrieval re-run, and golden_rank is
left exactly as-is (the scoring fix is a separate step).

Run from the project root (dc-available):
  poetry run python -m evaluation.available.common.split
"""
from __future__ import annotations

import json
from pathlib import Path

RES = Path("evaluation/results/synthetic")
LEGS = ["vector", "graph", "hybrid"]


def main() -> None:
    src = RES / "retrieval.json"
    if not src.exists():
        raise SystemExit(f"Missing {src}")
    records = json.loads(src.read_text(encoding="utf-8"))
    for leg in LEGS:
        rows = [r for r in records if r.get("retrieval_type") == leg]
        out = RES / f"{leg}.json"
        out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{leg}: {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
