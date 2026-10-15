import argparse
import sys
from pathlib import Path

from dc_available.config import data_config


def _count_training_indexes(data_path: Path) -> int:

    # Count how many training-level index.jsonl files exist under the given path.
    return sum(1 for _ in data_path.rglob("**/*/index.jsonl"))


def run_indexing(
    refresh: bool = False,
    run_vectordb: bool = True,
    run_graph: bool = True,
    run_semantic: bool = True,
    semantic_model: str | None = None,
    semantic_max_batches_per_topic: int = 0,
    semantic_topics: list[str] | None = None,
    semantic_min_confidence: float = 0.0,
    semantic_min_occurrences: int = 1,
) -> None:

    # Check for existing ingestion outputs before proceeding with indexing.
    available_indexes = _count_training_indexes(data_config.DEFAULT_INGESTION_ARTIFACTS_PATH)
    if available_indexes == 0:
        print(
            "[Indexing] ERROR: No training index.jsonl files found under "
            f"{data_config.DEFAULT_INGESTION_ARTIFACTS_PATH}. Run ingestion first."
        )
        sys.exit(1)

    print(f"[Indexing] Found {available_indexes} training index file(s).")

    # ------------------------------------------------------------------
    # PHASE 1 — Vector DB: embed chunks into Qdrant + cache images.
    # ------------------------------------------------------------------
    if run_vectordb:
        # Import after update_data_config() has already run so the helper module
        # reads the correct source-specific paths at import time.
        from dc_available.RAG import retrieval_helpers

        print("\n[Indexing] ── Vector DB phase (Qdrant) ──")
        print("[Indexing] Load or create retriever")
        retriever = retrieval_helpers.get_retriever(
            embedding_model_name=data_config.DEFAULT_EMBEDDING_MODEL,
            docs_path=data_config.DEFAULT_INGESTION_ARTIFACTS_PATH,
            cache_path=data_config.DEFAULT_INDEX_ARTIFACTS_PATH,
            refresh=refresh,
        )
        vector_doc_count = retriever.vectorstore.client.count(
            collection_name=retriever.vectorstore.collection_name, exact=True
        ).count
        print(f"[Indexing] Retriever ready with {vector_doc_count} vectorstore entries.")

        print("[Indexing] Prepare image cache")
        retrieval_helpers.prepare_image_cache(
            retriever=retriever,
            image_cache_path=data_config.DEFAULT_IMAGE_ARTIFACTS_PATH,
        )

    # ------------------------------------------------------------------
    # PHASE 2 — Graph (structural): topic/subtopic/level/training/document/chunk
    # tree + semantic batch preparation (no LLM here).
    # ------------------------------------------------------------------
    if run_graph:
        from dc_available.graph.construct.graph_builder import build_per_topic_graph
        from dc_available.graph.construct.graph_enricher import prepare_semantic_batches

        print("\n[Indexing] ── Graph phase (structural) ──")
        graph_stats = build_per_topic_graph(
            ingestion_root=data_config.DEFAULT_INGESTION_ARTIFACTS_PATH,
            graph_root=data_config.DEFAULT_GRAPH_ARTIFACTS_PATH,
            refresh=refresh,
        )
        print(
            "[Indexing] Graph ready with "
            f"{graph_stats['topics']} topic(s), "
            f"{graph_stats['trainings']} training node(s), "
            f"{graph_stats['documents']} document node(s), "
            f"{graph_stats['chunks']} chunk node(s)."
        )

        print("[Indexing] Prepare semantic batches")
        semantic_batch_stats = prepare_semantic_batches(
            ingestion_root=data_config.DEFAULT_INGESTION_ARTIFACTS_PATH,
            graph_root=data_config.DEFAULT_GRAPH_ARTIFACTS_PATH,
            batch_size=data_config.DEFAULT_SEMANTIC_BATCH_SIZE,
            overlap=data_config.DEFAULT_SEMANTIC_BATCH_OVERLAP,
            min_chars=data_config.DEFAULT_SEMANTIC_MIN_CHARS,
        )
        print(
            "[Indexing] Semantic batching ready with "
            f"{semantic_batch_stats['topics']} topic(s), "
            f"{semantic_batch_stats['documents']} document(s), "
            f"{semantic_batch_stats['chunks']} chunk(s), "
            f"{semantic_batch_stats['batches']} batch(es)."
        )

    # ------------------------------------------------------------------
    # PHASE 3 — Semantic (LLM): extract entities/relations -> merge -> integrate
    # into the graph as Entity nodes + `mentions` edges. Needs the graph phase
    # to have run at least once (reads semantic_batches.jsonl).
    # ------------------------------------------------------------------
    if run_semantic:
        from dc_available.graph.construct.semantic_extractor import extract_semantic_candidates
        from dc_available.graph.construct.semantic_merger import merge_semantic_candidates
        from dc_available.graph.construct.semantic_graph_integrator import integrate_semantic_graph
        from dc_available.graph.construct.entity_index import build_entity_index
        from dc_available.RAG.retrieval_helpers import get_embeddings

        if not (data_config.DEFAULT_GRAPH_ARTIFACTS_PATH / "build_manifest.json").is_file():
            print(
                "[Indexing] WARNING: semantic phase requested but no graph found. "
                "Run the graph phase first (plain run, or --graph)."
            )

        print("\n[Indexing] ── Semantic phase (LLM) ──")
        print("[Indexing] Run semantic extraction (LLM)")
        extraction_stats = extract_semantic_candidates(
            graph_root=data_config.DEFAULT_GRAPH_ARTIFACTS_PATH,
            model_name=semantic_model or "gpt-4o",
            max_batches_per_topic=semantic_max_batches_per_topic,
            topic_keys=semantic_topics,
        )
        print(
            "[Indexing] Semantic extraction ready with "
            f"{extraction_stats['topics']} topic(s), "
            f"{extraction_stats['batches']} batch(es), "
            f"{extraction_stats['entities']} entity candidate(s), "
            f"{extraction_stats['relations']} relation candidate(s)."
        )

        print("[Indexing] Merge semantic candidates")
        merge_stats = merge_semantic_candidates(
            graph_root=data_config.DEFAULT_GRAPH_ARTIFACTS_PATH,
            topic_keys=semantic_topics,
            min_confidence=semantic_min_confidence,
            min_occurrences=semantic_min_occurrences,
        )
        print(
            "[Indexing] Merge ready with "
            f"{merge_stats['topics']} topic(s), "
            f"{merge_stats['entities']} merged entity(ies), "
            f"{merge_stats['relations']} merged relation(s)."
        )

        print("[Indexing] Integrate semantic graph")
        integration_stats = integrate_semantic_graph(
            graph_root=data_config.DEFAULT_GRAPH_ARTIFACTS_PATH,
            topic_keys=semantic_topics,
        )
        print(
            "[Indexing] Semantic graph integration ready with "
            f"{integration_stats['topics']} topic(s), "
            f"{integration_stats['semantic_entities']} semantic entity node(s), "
            f"{integration_stats['mentions_edges']} mentions edge(s), "
            f"{integration_stats['semantic_edges']} semantic relation edge(s)."
        )

        print("[Indexing] Build entity index (Qdrant)")
        entity_index_stats = build_entity_index(
            graph_root=data_config.DEFAULT_GRAPH_ARTIFACTS_PATH,
            embeddings=get_embeddings(),
        )
        print(
            "[Indexing] Entity index ready with "
            f"{entity_index_stats['points']} point(s) across "
            f"{entity_index_stats['topics']} topic(s)."
        )

    print("\n[Indexing] ══════════════════════════════════════")
    print(f"[Indexing] Ingestion artifacts : {data_config.DEFAULT_INGESTION_ARTIFACTS_PATH}")
    print(f"[Indexing] Cache path          : {data_config.DEFAULT_INDEX_ARTIFACTS_PATH}")
    print(f"[Indexing] Image cache         : {data_config.DEFAULT_IMAGE_ARTIFACTS_PATH}")
    print(f"[Indexing] Graph artifacts     : {data_config.DEFAULT_GRAPH_ARTIFACTS_PATH}")
    print(f"[Indexing] Refresh      : {refresh}")
    print("[Indexing] Stage completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AVAILABLE — Indexing stage: build Qdrant vectors, the graph, and the semantic layer."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Data source key from _data_config in data_config.py "
            "(e.g. 'dc_available', 'test'). "
            "Defaults to the value of AVAILABLE_APP_SOURCE env var or DEFAULT_DATA_SOURCE ('available')."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        default=False,
        help="Force a clean rebuild (recreate the Qdrant collection and rebuild the graph from scratch).",
    )

    # --- Phase selectors -------------------------------------------------
    # If NONE of these is given, all three phases run (the default full build).
    parser.add_argument(
        "--vectordb",
        action="store_true",
        default=False,
        help="Phase: embed chunks into Qdrant + cache images.",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        default=False,
        help="Phase: build the structural graph (topic/subtopic/.../chunk) + semantic batches. No LLM.",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        default=False,
        help="Phase: LLM entity extraction -> merge -> integrate (adds Entity nodes + mentions edges). Needs the graph phase first.",
    )

    # --- Semantic-phase modifiers ---------------------------------------
    parser.add_argument(
        "--semantic-model",
        type=str,
        default="gpt-4o",
        help="Model name for semantic extraction (default: gpt-4o).",
    )
    parser.add_argument(
        "--semantic-max-batches-per-topic",
        type=int,
        default=0,
        help="Limit extracted batches per topic for test runs (0 means all batches).",
    )
    parser.add_argument(
        "--semantic-topics",
        type=str,
        default="",
        help="Comma-separated topic keys to limit the semantic phase (e.g. hydraulics,digitaltransformation). Empty means all topics.",
    )
    parser.add_argument(
        "--semantic-min-confidence",
        type=float,
        default=0.0,
        help="Drop entities/relations with confidence below this threshold (0.0 = keep all).",
    )
    parser.add_argument(
        "--semantic-min-occurrences",
        type=int,
        default=1,
        help="Drop entities/relations that appear fewer times than this threshold.",
    )
    args = parser.parse_args()

    if args.source:
        data_config.update_data_config(args.source)
    else:
        data_config.update_data_config()

    # No phase flag -> run everything. Any phase flag -> run only those phases.
    any_phase = args.vectordb or args.graph or args.semantic
    run_vectordb = args.vectordb or not any_phase
    run_graph = args.graph or not any_phase
    run_semantic = args.semantic or not any_phase

    phases = ", ".join(
        name
        for name, on in (("vectordb", run_vectordb), ("graph", run_graph), ("semantic", run_semantic))
        if on
    )

    print("[Indexing] ══════════════════════════════════════")
    print(f"[Indexing] Ingestion artifacts : {data_config.DEFAULT_INGESTION_ARTIFACTS_PATH}")
    print(f"[Indexing] Cache path          : {data_config.DEFAULT_INDEX_ARTIFACTS_PATH}")
    print(f"[Indexing] Image cache         : {data_config.DEFAULT_IMAGE_ARTIFACTS_PATH}")
    print(f"[Indexing] Graph artifacts     : {data_config.DEFAULT_GRAPH_ARTIFACTS_PATH}")
    print(f"[Indexing] Phases       : {phases}")
    print(f"[Indexing] Refresh      : {args.refresh}")
    print("[Indexing] ══════════════════════════════════════\n")

    run_indexing(
        refresh=args.refresh,
        run_vectordb=run_vectordb,
        run_graph=run_graph,
        run_semantic=run_semantic,
        semantic_model=args.semantic_model,
        semantic_max_batches_per_topic=args.semantic_max_batches_per_topic,
        semantic_topics=[item.strip() for item in args.semantic_topics.split(",") if item.strip()] or None,
        semantic_min_confidence=args.semantic_min_confidence,
        semantic_min_occurrences=args.semantic_min_occurrences,
    )


if __name__ == "__main__":
    main()
