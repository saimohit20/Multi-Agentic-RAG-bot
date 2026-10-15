"""RICH graph build — the graph leg's ingest-time artifacts for the RICH feature.

Mirrors the QA graph phases (pipeline/index.py) but stays fully isolated from QA:
it resolves RICH's own paths via ``source_artifact_paths("rich")`` (NO mutation of the
global data_config that the live QA app relies on) and writes entities into RICH's own
``QDRANT_RICH_ENTITY_COLLECTION`` so the recreate never wipes QA's entities.

Two phases, matching the QA pipeline:
  structural (no LLM): build_per_topic_graph + prepare_semantic_batches
  semantic   (LLM):    extract -> merge -> integrate -> build_entity_index

Usage:
    python -m dc_available.RAG.rich_graph_build --build-graph            # structural + semantic
    python -m dc_available.RAG.rich_graph_build --build-graph --structural-only
    python -m dc_available.RAG.rich_graph_build --build-graph --refresh
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dc_available.config import data_config


def rich_graph_root() -> Path:
    """RICH's graph artifacts root: ``Artifacts/rich/graph`` (sibling of ``ingestion``)."""
    paths = data_config.source_artifact_paths("rich")
    return paths["ingestion"].parent / "graph"


def build_rich_graph(
    refresh: bool = False,
    semantic: bool = True,
    semantic_model: str = "gpt-4o",
) -> dict[str, Any]:
    """Build RICH's graph artifacts (+ entity index) from RICH's ingestion index files.

    When ``semantic`` is False only the structural graph + semantic batches are produced
    (no LLM, no cost) — useful for a first validation pass.
    """
    from dc_available.graph.construct.graph_builder import build_per_topic_graph
    from dc_available.graph.construct.graph_enricher import prepare_semantic_batches

    paths = data_config.source_artifact_paths("rich")
    ingestion_root = paths["ingestion"]
    graph_root = rich_graph_root()

    print(f"[RICH-graph] ingestion : {ingestion_root}")
    print(f"[RICH-graph] graph root : {graph_root}")

    # --- Structural phase (no LLM) ---------------------------------------
    print("[RICH-graph] -- Structural phase --")
    graph_stats = build_per_topic_graph(ingestion_root, graph_root, refresh=refresh)
    print(
        "[RICH-graph] Graph ready with "
        f"{graph_stats['topics']} topic(s), "
        f"{graph_stats['trainings']} training node(s), "
        f"{graph_stats['documents']} document node(s), "
        f"{graph_stats['chunks']} chunk node(s)."
    )

    print("[RICH-graph] Prepare semantic batches")
    batch_stats = prepare_semantic_batches(
        ingestion_root,
        graph_root,
        batch_size=data_config.DEFAULT_SEMANTIC_BATCH_SIZE,
        overlap=data_config.DEFAULT_SEMANTIC_BATCH_OVERLAP,
        min_chars=data_config.DEFAULT_SEMANTIC_MIN_CHARS,
    )
    print(
        "[RICH-graph] Semantic batching ready with "
        f"{batch_stats['topics']} topic(s), {batch_stats['batches']} batch(es)."
    )

    if not semantic:
        print("[RICH-graph] structural-only: skipping LLM semantic phase.")
        return {"graph": graph_stats, "batches": batch_stats}

    # --- Semantic phase (LLM) --------------------------------------------
    from dc_available.graph.construct.semantic_extractor import extract_semantic_candidates
    from dc_available.graph.construct.semantic_merger import merge_semantic_candidates
    from dc_available.graph.construct.semantic_graph_integrator import integrate_semantic_graph
    from dc_available.graph.construct.entity_index import build_entity_index
    from dc_available.RAG.retrieval_helpers import get_embeddings

    print("[RICH-graph] -- Semantic phase (LLM) --")
    extraction_stats = extract_semantic_candidates(graph_root, model_name=semantic_model)
    print(
        "[RICH-graph] Extraction: "
        f"{extraction_stats['entities']} entity candidate(s), "
        f"{extraction_stats['relations']} relation candidate(s)."
    )

    merge_stats = merge_semantic_candidates(graph_root)
    print(
        "[RICH-graph] Merge: "
        f"{merge_stats['entities']} entity(ies), {merge_stats['relations']} relation(s)."
    )

    integration_stats = integrate_semantic_graph(graph_root)
    print(
        "[RICH-graph] Integration: "
        f"{integration_stats['semantic_entities']} entity node(s), "
        f"{integration_stats['mentions_edges']} mentions edge(s), "
        f"{integration_stats['semantic_edges']} semantic edge(s)."
    )

    entity_index_stats = build_entity_index(
        graph_root,
        embeddings=get_embeddings(),
        collection_name=data_config.QDRANT_RICH_ENTITY_COLLECTION,
    )
    print(
        f"[RICH-graph] Entity index '{data_config.QDRANT_RICH_ENTITY_COLLECTION}': "
        f"{entity_index_stats['points']} point(s) across {entity_index_stats['topics']} topic(s)."
    )

    return {
        "graph": graph_stats,
        "batches": batch_stats,
        "extraction": extraction_stats,
        "merge": merge_stats,
        "integration": integration_stats,
        "entity_index": entity_index_stats,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="RICH — build the RICH graph + entity index")
    ap.add_argument("--build-graph", action="store_true", help="build the RICH graph artifacts")
    ap.add_argument("--structural-only", action="store_true", help="skip the LLM semantic phase (no cost)")
    ap.add_argument("--refresh", action="store_true", help="clean rebuild (wipe the RICH graph root first)")
    ap.add_argument("--semantic-model", default="gpt-4o", help="LLM for entity extraction (default: gpt-4o)")
    args = ap.parse_args()

    if not args.build_graph:
        ap.print_help()
        return

    build_rich_graph(
        refresh=args.refresh,
        semantic=not args.structural_only,
        semantic_model=args.semantic_model,
    )
    print("[RICH-graph] build complete.")


if __name__ == "__main__":
    main()
