"""Stage D — cross-encoder reranker (Option 1).

Fusion (RAG/fusion.py) builds a candidate set by *rank agreement* between the
vector and graph legs — good for recall, blind to the actual text. This module
reorders those candidates by reading query+passage together with a cross-encoder
(``BAAI/bge-reranker-v2-m3`` — the sibling of the BGE-M3 embedding model, so it
rides the same FlagEmbedding stack with no new dependency). Fusion for recall,
the reranker for precision.

The model is a lazy singleton (a few-GB one-time download on first use), mirroring
``retrieval_helpers.get_embeddings``.
"""
from __future__ import annotations

from typing import Any

from dc_available.config import data_config

_reranker: Any = None


def get_reranker(model_name: str | None = None) -> Any:
    """Load (once) and return the FlagReranker singleton."""
    global _reranker
    if _reranker is None:
        import torch
        from FlagEmbedding import FlagReranker

        model_name = model_name or data_config.RERANKER_MODEL
        use_fp16 = torch.cuda.is_available()
        print(f"[Reranker] Loading {model_name} (fp16={use_fp16}).")
        _reranker = FlagReranker(model_name, use_fp16=use_fp16)
    return _reranker


def rerank(query: str, passages: list[str], top_k: int | None = None) -> list[tuple[int, float]]:
    """Score each passage against the query with the cross-encoder.

    Returns ``(original_index, score)`` sorted by score descending, truncated to
    ``top_k``. Scores are normalized to 0-1 (sigmoid) so they are comparable
    across queries. Returns ``[]`` for empty input. The caller keeps ownership of
    the passages, so it maps the returned indices back to its Documents.
    """
    if not passages:
        return []
    reranker = get_reranker()
    pairs = [[query, passage] for passage in passages]
    scores = reranker.compute_score(pairs, normalize=True)
    # compute_score returns a bare float for a single pair, a list otherwise.
    if isinstance(scores, (int, float)):
        scores = [scores]
    ranked = sorted(enumerate(scores), key=lambda item: -item[1])
    if top_k is not None:
        ranked = ranked[:top_k]
    return [(idx, float(score)) for idx, score in ranked]


def _demo() -> None:
    """Tiny sanity run: `python -m dc_available.RAG.reranker` (D3 acceptance).

    Downloads the reranker model on first run (a few GB). Expect the two
    sales-order passages to rank above the unrelated hydraulics one.
    """
    query = "How to create a sales order in SAP?"
    passages = [
        "To create a sales order in SAP S/4HANA, use transaction VA01 and enter the sold-to party and material.",
        "The hydraulic pump requires 200 bar pressure for normal operation.",
        "A sales order is a contractual agreement between a company and a customer for goods or services.",
    ]
    print(f'\nquery: {query}\n')
    for idx, score in rerank(query, passages):
        print(f"  {score:.4f}  {passages[idx]}")
    print()


if __name__ == "__main__":
    _demo()
