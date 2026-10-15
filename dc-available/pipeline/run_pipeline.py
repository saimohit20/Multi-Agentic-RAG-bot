"""
pipeline/run_pipeline.py — the automated pipeline orchestrator.

Runs the stages in sequence, gated on whether anything actually changed:

    sync -> (no changes? stop) -> preprocess -> index -> trust score

This is the single command the Azure CronJob runs:

    python -m pipeline.run_pipeline [--source available]

It does NOT add new logic — it just calls the existing stage functions in the
right order and short-circuits when sync reports no new/updated/deleted files.
Preprocess already skips unchanged trainings (source-signature check), so only
what changed is reprocessed; index then rebuilds vectors + graph + semantic.
"""

import argparse

from dc_available.config import data_config
from pipeline import index, preprocess, sync, trust_score


def run_pipeline(source: str | None = None) -> bool:
    """Run sync, preprocess, index, and trust scoring. Returns True if it did work."""
    if source:
        data_config.update_data_config(source)
    else:
        data_config.update_data_config()  # AVAILABLE_APP_SOURCE env var or default

    print("=" * 52)
    print("AUTO PIPELINE · sync -> preprocess -> index -> trust score")
    print("=" * 52)

    # ---- STAGE 1: sync — returns True if new/updated/deleted files ----------
    has_changes = sync.run()

    if not has_changes:
        print("\n[Auto] No changes in SharePoint - stopping after sync.")
        return False

    # ---- STAGE 2: preprocess - skips unchanged trainings by source signature
    print("\n[Auto] Changes detected - running preprocess...")
    preprocess.run_ingestion()

    # ---- STAGE 3: index - full rebuild (Qdrant + graph + semantic) ----------
    # NOTE: full rebuild on any change is simple + correct, but re-embeds all
    # chunks and re-runs semantic (gpt-4o). Incremental index is a later
    # optimization (only add/remove the changed chunks).
    print("\n[Auto] Rebuilding index (Qdrant + graph + semantic)...")
    index.run_indexing(refresh=True)

    # TrustRank requires the graph written by the indexing stage.
    trust_score.run()

    print("\n[Auto] Pipeline complete.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AVAILABLE - Automated pipeline: sync -> preprocess -> index -> trust score."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Data source key (defaults to AVAILABLE_APP_SOURCE env var or 'available').",
    )
    args = parser.parse_args()
    run_pipeline(source=args.source)


if __name__ == "__main__":
    main()
