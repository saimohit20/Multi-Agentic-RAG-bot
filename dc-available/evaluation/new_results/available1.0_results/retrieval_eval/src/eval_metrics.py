"""
Retrieval metrics computed from a list of result records.

All metrics derive from a single field, `golden_rank`:
  * identification = golden source present in the retrieved top-k (rank not None)
  * hit@k          = golden_rank is not None and golden_rank <= k
  * MRR            = mean over ALL questions of 1/golden_rank (0 for misses)
"""
from __future__ import annotations

from typing import Any, Dict, List

import paths_config as cfg


def compute(records: List[Dict[str, Any]], hit_k: int = cfg.HIT_K) -> Dict[str, Any]:
    n = len(records)
    ranks = [r.get("golden_rank") for r in records]

    identified = sum(1 for x in ranks if x is not None)
    hit_count = sum(1 for x in ranks if x is not None and x <= hit_k)
    mrr = (sum(1.0 / x for x in ranks if x is not None) / n) if n else 0.0

    return {
        "n": n,
        "identified": identified,
        "not_identified": n - identified,
        "hit_k": hit_k,
        f"hit@{hit_k}_count": hit_count,
        f"hit@{hit_k}": (hit_count / n) if n else 0.0,
        "mrr": mrr,
    }


def format_summary(topic: str, m: Dict[str, Any]) -> str:
    k = m["hit_k"]
    return (
        f"{topic:<12} n={m['n']:<4} "
        f"identified={m['identified']}/{m['n']}  "
        f"hit@{k}={m[f'hit@{k}']:.3f} ({m[f'hit@{k}_count']}/{m['n']})  "
        f"MRR={m['mrr']:.3f}"
    )
