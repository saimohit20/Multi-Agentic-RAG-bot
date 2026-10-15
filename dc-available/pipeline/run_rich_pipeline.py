"""
pipeline/run_rich_pipeline.py — the automated RICH pipeline orchestrator.

The RICH counterpart of pipeline/run_pipeline.py. Runs RICH's stages in sequence,
gated on whether sync actually changed anything:

    sync -> (no changes? stop) -> preprocess -> vector build -> graph build -> trust

This is the single command for RICH:

    python -m pipeline.run_rich_pipeline

Like run_pipeline, it adds no new logic — it just calls the existing RICH stage
functions in the right order. RICH stays fully isolated from QA: its own Qdrant
collections, its own Artifacts/rich/ paths, its own trust files.
"""

from dc_available.config import data_config
from pipeline import preprocess, sync

RICH_SOURCE = "rich"


def run_rich_pipeline() -> bool:
    """Run RICH sync, preprocess, vector build, graph build, trust. Returns True if it did work."""
    data_config.update_data_config(RICH_SOURCE)

    print("=" * 52)
    print("RICH PIPELINE · sync -> preprocess -> vector -> graph -> trust")
    print("=" * 52)

    # ---- STAGE 1: sync — returns True if new/updated/deleted files ----------
    has_changes = sync.run(source=RICH_SOURCE)

    if not has_changes:
        print("\n[RICH] No changes in SharePoint - stopping after sync.")
        return False

    # ---- STAGE 2: preprocess — copy pre-made artifacts, skip the handbook ---
    print("\n[RICH] Changes detected - running preprocess...")
    preprocess.run_ingestion(copy_existing=True, build_handbook=False)

    # ---- STAGE 3: vector build — (re)build the RICH Qdrant collection -------
    print("\n[RICH] Building vector collection (Qdrant 'RICH')...")
    from dc_available.RAG.retrieval_helpers import get_rich_retriever

    get_rich_retriever(refresh=True)

    # ---- STAGE 4: graph build — RICH graph + entity index -------------------
    print("\n[RICH] Building graph + entity index...")
    from dc_available.RAG.rich_graph_build import build_rich_graph

    build_rich_graph(refresh=True, semantic=True)

    # ---- STAGE 5: trust — metadata -> content -> prior -> trustrank ---------
    print("\n[RICH] Building trust scores...")
    from dc_available.trust_score import (
        collect_rich_metadata,
        score_rich_content,
        score_rich_prior,
        trustrank_rich,
    )

    collect_rich_metadata.collect()
    score_rich_content.run()
    score_rich_prior.run()
    trustrank_rich.run()

    print("\n[RICH] Pipeline complete.")
    return True


def main() -> None:
    run_rich_pipeline()


if __name__ == "__main__":
    main()
