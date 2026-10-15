"""Probe one question: what does the VECTOR leg retrieve vs the GRAPH leg?

Manual inspection tool — no hybrid, no scoring. Prints each leg's final top-10
documents (after the same reranker production uses), so you can eyeball whether
graph surfaces a doc vector misses.

Run:
  poetry run python evaluation/probe.py "your question here"
  poetry run python evaluation/probe.py            # then type questions interactively
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]

from dc_available.config import data_config

data_config.update_data_config()
data_config.RETRIEVAL_MODE = "fusion"
data_config.GRAPH_RETRIEVAL_MODE = "ppr"
data_config.PPR_SEED_MODE = "merge"

from dc_available.RAG import qa_functions as qa
from dc_available.RAG.reranker import rerank
from dc_available.RAG.data import AvailableMetaData
from dc_available.graph.retrieval.ppr_retriever import ppr_retrieve
from dc_available.config.data_config import DEFAULT_GRAPH_ARTIFACTS_PATH

DOC = AvailableMetaData.PART_OF_DOCUMENT
ID = AvailableMetaData.ID


def _show(title: str, docs: list) -> None:
    print(f"\n{title}")
    seen: list[str] = []
    for d in docs:
        name = str(d.metadata.get(DOC) or "N/A")
        if name in seen:
            continue
        seen.append(name)
        print(f"  {len(seen):2d}. {name[:72]}")
        if len(seen) >= 10:
            break
    if not seen:
        print("  (nothing retrieved / graph abstained)")


def _rerank_docs(question: str, docs: list, k: int) -> list:
    if data_config.RERANKER_ENABLED and docs:
        order = rerank(question, [d.page_content for d in docs], top_k=k)
        return [docs[i] for i, _ in order]
    return docs[:k]


def probe(question: str) -> None:
    print(f"\n{'='*74}\nQ: {question}\n{'='*74}")
    kv, kg, ok = (data_config.FUSION_VECTOR_K, data_config.FUSION_GRAPH_K,
                  data_config.FUSION_OUTPUT_K)

    # --- vector leg (hybrid dense+sparse) ---
    vec = qa.vectorstore.similarity_search_with_score(question, k=kv)
    _show("VECTOR (dense+sparse):", _rerank_docs(question, [d for d, _ in vec], ok))

    # --- graph leg (entity-seeded PPR, no vector seeds) ---
    try:
        res = ppr_retrieve(question, DEFAULT_GRAPH_ARTIFACTS_PATH, vector_hits=[], debug=False)
        gids = [c for c, _ in res.chunks[:kg]] if (res and not res.abstained) else []
    except Exception as exc:
        print(f"  [graph error] {exc}")
        gids = []
    gdocs = [d for d in qa.retriever.docstore.mget(gids) if d is not None] if gids else []
    _show("GRAPH (PPR merge):", _rerank_docs(question, gdocs, ok))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        probe(" ".join(sys.argv[1:]))
    else:
        print("Type a question (blank line to quit):")
        while True:
            try:
                q = input("\n> ").strip()
            except EOFError:
                break
            if not q:
                break
            probe(q)
