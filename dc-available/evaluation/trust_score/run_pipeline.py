"""Consolidated Trust-Score evaluation pipeline — both experiments.

One entry point that orchestrates the existing step modules (no scoring is
re-implemented), so results match running each stage by hand.

Experiments
-----------
perturbation   Experiment 1 — pillar-sensitivity analysis:
                 select_sample → perturb → score → analyze → stats
rag_ab         Experiment 2 — RAG A/B poisoning test:
                 select_docs → make_twins → graph_twins → build_index →
                 run_ab → score_generation → score_ragas → analyze_ab

Run from the project root (dc-available/):
    poetry run python -m evaluation.trust_score.run_pipeline               # both
    poetry run python -m evaluation.trust_score.run_pipeline perturbation
    poetry run python -m evaluation.trust_score.run_pipeline rag_ab
    poetry run python -m evaluation.trust_score.run_pipeline rag_ab --limit 6   # dry run
    poetry run python -m evaluation.trust_score.run_pipeline --no-plot

Notes
-----
* rag_ab needs Qdrant (its own `available_trust_test` collection) + Azure OpenAI;
  perturbation needs Azure OpenAI (LLM content scoring/corruption).
* Every step is resumable (skips completed work); `--limit` caps the LLM-heavy
  rag_ab stages for a quick dry run.
* Step modules are imported lazily so `--help` doesn't boot anything.
"""
from __future__ import annotations

import argparse


# ── Experiment 1 — perturbation ─────────────────────────────────────────────
def run_perturbation(plot: bool = True) -> None:
    print("\n" + "=" * 64 + "\nEXPERIMENT 1 — PERTURBATION SENSITIVITY\n" + "=" * 64)
    from evaluation.trust_score.perturbation import (
        select_sample, perturb, score, analyze, stats,
    )
    select_sample.run()
    perturb.run()
    score.run()
    analyze.run(plot=plot)
    stats.run()


# ── Experiment 2 — rag_ab ───────────────────────────────────────────────────
def run_rag_ab(limit: int | None = None, plot: bool = True) -> None:
    print("\n" + "=" * 64 + "\nEXPERIMENT 2 — RAG A/B POISONING\n" + "=" * 64)
    from evaluation.trust_score.rag_ab import (
        select_docs, make_twins, graph_twins, build_index,
        run_ab, score_generation, score_ragas, analyze_ab,
    )
    select_docs.run()
    make_twins.run()
    graph_twins.run()
    build_index.run()
    run_ab.run(limit=limit)
    score_generation.run(limit=limit)
    score_ragas.run(limit=limit)
    analyze_ab.run(plot=plot)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Trust-Score evaluation — perturbation + rag_ab experiments."
    )
    p.add_argument("experiment", nargs="?", choices=["perturbation", "rag_ab", "all"],
                   default="all", help="which experiment to run (default: all).")
    p.add_argument("--limit", type=int, default=None,
                   help="cap the LLM-heavy rag_ab stages for a dry run.")
    p.add_argument("--no-plot", action="store_true",
                   help="skip the analysis figures.")
    args = p.parse_args()
    plot = not args.no_plot

    if args.experiment in ("perturbation", "all"):
        run_perturbation(plot=plot)
    if args.experiment in ("rag_ab", "all"):
        run_rag_ab(limit=args.limit, plot=plot)

    print("\n[run_pipeline] done.")


if __name__ == "__main__":
    main()
