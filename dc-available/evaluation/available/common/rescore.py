"""Recompute golden_rank in the per-leg files with the corrected name matcher.

Fixes the scoring bug where mid-name dots were treated as file extensions.
Reads each of vector/graph/hybrid.json, recomputes golden_rank from the stored
retrieved_docs using the same _norm as synthetic.py, overwrites the file
in place, and rewrites the combined retrieval_metrics.json. No retrieval re-run.

Run from the project root (dc-available):
  poetry run python -m evaluation.available.common.rescore
"""
from __future__ import annotations

import json
import re
from pathlib import Path

RES = Path("evaluation/results/synthetic")
LEGS = ["vector", "graph", "hybrid"]
K_LIST = [1, 3, 10]

_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,5}$")


def _norm(value) -> str:
    """Same matcher as retrieval.py: drop a trailing file extension, lower-case,
    collapse runs of non-alphanumerics to a single space."""
    s = _EXT_RE.sub("", str(value or "").strip()).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _rank_of(golden: str, docs) -> int | None:
    key = _norm(golden)
    if not key:
        return None
    for i, name in enumerate(docs, 1):
        if _norm(name) == key:
            return i
    return None


def main() -> None:
    metrics = {}
    for leg in LEGS:
        path = RES / f"{leg}.json"
        if not path.exists():
            raise SystemExit(f"Missing {path} — run split.py first.")
        rows = json.loads(path.read_text(encoding="utf-8"))
        for r in rows:
            r["golden_rank"] = _rank_of(r.get("golden_source", ""), r.get("retrieved_docs", []))
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

        n = len(rows) or 1
        m = {f"hit@{k}": round(sum(1 for r in rows
                                   if r["golden_rank"] and r["golden_rank"] <= k) / n, 4)
             for k in K_LIST}
        m["mrr"] = round(sum(1.0 / r["golden_rank"] for r in rows if r["golden_rank"]) / n, 4)
        m["n"] = len(rows)
        metrics[leg] = m
        print(f"{leg:7} n={m['n']}  hit@1={m['hit@1']:.3f}  hit@3={m['hit@3']:.3f}  "
              f"hit@10={m['hit@10']:.3f}  mrr={m['mrr']:.3f}")

    (RES / "retrieval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nRewrote {', '.join(f'{l}.json' for l in LEGS)} and retrieval_metrics.json")


if __name__ == "__main__":
    main()
