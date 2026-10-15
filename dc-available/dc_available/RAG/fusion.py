"""Stage D — rank fusion (pure, no I/O).

Weighted Reciprocal Rank Fusion (RRF) over two independent retrieval legs
(vector + Stage-C graph). Rank-based, so the legs' incomparable score scales
(cosine 0-1 vs. graph 0-8) never meet — only their positions do:

    fused(item) = Σ_leg  weight_leg / (K + rank_leg(item))     rank starts at 1

The same function fuses the CHUNK track (key = chunk_id) and the DOCUMENT track
(key = (topic, document)). It takes ordered id lists and returns a fused,
descending list — it never touches Qdrant, the docstore, or the reranker, so it
can be unit-tested on plain lists.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from dc_available.config import data_config


@dataclass
class FusionItem:
    """One fused key with its provenance (for debug / the reranker candidate set)."""

    key: Any                          # chunk_id (str) or (topic, document) tuple
    score: float                      # fused weighted-RRF score
    vector_rank: int | None = None    # 1-based rank in the vector leg (None if absent)
    graph_rank: int | None = None     # 1-based rank in the graph leg (None if absent)


def _fuse(legs: Sequence[tuple[Sequence[Any], float]], k: int) -> list[FusionItem]:
    """Core weighted-RRF. ``legs`` = [(ranked_keys, weight), ...]; leg 0 = vector,
    leg 1 = graph. Pure and side-effect free. A key repeated inside one leg counts
    once (best/first rank). A zero-weight leg still records ranks but adds no score
    (so an abstained graph leg contributes nothing yet stays visible in the trace).
    """
    fused: dict[Any, float] = defaultdict(float)
    ranks: dict[Any, list[int | None]] = {}
    n_legs = len(legs)
    for leg_idx, (ranked_keys, weight) in enumerate(legs):
        seen: set[Any] = set()
        for rank, key in enumerate(ranked_keys, start=1):
            if key is None or key in seen:
                continue
            seen.add(key)
            if key not in ranks:
                ranks[key] = [None] * n_legs
            ranks[key][leg_idx] = rank
            if weight > 0:
                fused[key] += weight / (k + rank)
    items = [
        FusionItem(
            key=key,
            score=fused[key],
            vector_rank=ranks[key][0] if n_legs > 0 else None,
            graph_rank=ranks[key][1] if n_legs > 1 else None,
        )
        for key in ranks
    ]
    items.sort(key=lambda it: -it.score)
    return items


def graph_leg_weight(list_weight: float, abstained: bool) -> float:
    """RRF weight for the graph leg, per config. 0.0 when the graph abstained, so
    fusion collapses cleanly to vector-only.
    """
    if abstained:
        return 0.0
    if data_config.FUSION_GRAPH_WEIGHT_MODE == "fixed":
        return float(data_config.FUSION_GRAPH_WEIGHT_FIXED)
    return float(list_weight)


def rrf_fuse(
    vector_ranked: Sequence[Any],
    graph_ranked: Sequence[Any],
    *,
    graph_weight: float,
    vector_weight: float | None = None,
    k: int | None = None,
) -> list[FusionItem]:
    """Fuse two ordered key lists (vector first, graph second) by weighted RRF.

    Works for both tracks: pass chunk_ids for the chunk track, or (topic, document)
    keys for the document track. Weights/K default to the Stage D config.
    """
    if vector_weight is None:
        vector_weight = data_config.FUSION_VECTOR_WEIGHT
    if k is None:
        k = data_config.FUSION_RRF_K
    return _fuse([(vector_ranked, vector_weight), (graph_ranked, graph_weight)], k)


def union_fuse(
    vector_ranked: Sequence[Any],
    graph_ranked: Sequence[Any],
) -> list[FusionItem]:
    """Merge two ranked key lists into a de-duplicated union — no weights, no RRF.

    Both legs' keys survive, so neither leg can bury the other (the failure mode of
    rank fusion when one leg is longer or weighted heavier). Keys are interleaved by
    rank (v1, g1, v2, g2, ...) so the pre-rerank order is balanced; a key present in
    both legs is emitted once, at its earliest position, with both ranks recorded for
    provenance. ``score`` stays 0.0 — the cross-encoder reranker assigns the real
    order downstream. Works for either track (chunk_ids or document keys).
    """
    vec_rank: dict[Any, int] = {}
    for rank, key in enumerate(vector_ranked, start=1):
        if key is not None and key not in vec_rank:
            vec_rank[key] = rank
    graph_rank: dict[Any, int] = {}
    for rank, key in enumerate(graph_ranked, start=1):
        if key is not None and key not in graph_rank:
            graph_rank[key] = rank

    vec_keys = list(vec_rank)
    graph_keys = list(graph_rank)
    ordered: list[Any] = []
    seen: set[Any] = set()
    for i in range(max(len(vec_keys), len(graph_keys))):
        for keys in (vec_keys, graph_keys):
            if i < len(keys) and keys[i] not in seen:
                seen.add(keys[i])
                ordered.append(keys[i])
    return [
        FusionItem(
            key=key,
            score=0.0,
            vector_rank=vec_rank.get(key),
            graph_rank=graph_rank.get(key),
        )
        for key in ordered
    ]


def rank_documents_from_chunks(chunk_ordered_doc_keys: Sequence[Any]) -> list[Any]:
    """Collapse a chunk-ordered list of document keys into a document ranking.

    The vector leg has no document ranking of its own — its documents are implied
    by the order of its chunks. Walking chunks top-down and taking each document's
    first appearance is MaxP-on-ranks: a document inherits the rank of its best
    chunk. Returns unique doc keys in that order.
    """
    seen: set[Any] = set()
    ordered: list[Any] = []
    for key in chunk_ordered_doc_keys:
        if key is None or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _demo() -> None:
    """Tiny sanity run: `python -m dc_available.RAG.fusion` (D2 acceptance)."""
    vector = ["A", "B", "C"]        # vector leg order
    graph = ["C", "D", "A"]         # graph leg order (weight 0.6)
    fused = rrf_fuse(vector, graph, graph_weight=0.6)
    print("\nfused (vector=[A,B,C], graph=[C,D,A], graph_weight=0.6, k=60):\n")
    print(f"  {'key':<4}{'score':>10}  {'vec_rank':>8}  {'graph_rank':>10}")
    for it in fused:
        print(f"  {str(it.key):<4}{it.score:>10.5f}  {str(it.vector_rank):>8}  {str(it.graph_rank):>10}")
    print("\nexpected: A on top (rank1 vector), C close 2nd (rank1 graph), then B, then D.\n")


if __name__ == "__main__":
    _demo()
