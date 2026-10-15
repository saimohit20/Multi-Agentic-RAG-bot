"""Consolidated AVAILABLE 2.0 evaluation pipeline — retrieval + generation.

One entry point for the whole AVAILABLE evaluation. It orchestrates the existing
step modules (it does NOT re-implement any scoring), so results are identical to
running the steps by hand.

Stages
------
retrieval   grownfield + hydraulics + synthetic retrieval evals
              (writes evaluation/results/available_2.0/{grownfield,hydraulics}/
               and evaluation/results/synthetic/)
generation  RAGAS on grownfield+hydraulics (generate → score → combined plot),
            synthetic answer generation + RAGAS, and the answer_correctness backfill
              (writes evaluation/results/RAGAS/ and evaluation/results/synthetic/)

Run from the project root (dc-available/):
    poetry run python -m evaluation.available.run_pipeline                 # all
    poetry run python -m evaluation.available.run_pipeline --only retrieval
    poetry run python -m evaluation.available.run_pipeline --only generation
    poetry run python -m evaluation.available.run_pipeline --topics grownfield
    poetry run python -m evaluation.available.run_pipeline --qa-dataset evaluation/datasets/test/qa_dataset.xlsx

Notes
-----
* Needs the Poetry env, Qdrant reachable, and Azure OpenAI auth (same as the app).
* Step modules are imported lazily inside each stage so `--help` and partial runs
  don't boot the whole RAG stack.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


# ── retrieval stage ─────────────────────────────────────────────────────────
def run_retrieval(topics: list[str], skip_synthetic: bool = False) -> None:
    print("\n" + "=" * 64 + "\nRETRIEVAL EVALUATION\n" + "=" * 64)

    if "grownfield" in topics:
        print("\n[retrieval] grownfield …")
        from evaluation.available.retrieval import grownfield
        grownfield.run()

    if "hydraulics" in topics:
        print("\n[retrieval] hydraulics …")
        from evaluation.available.retrieval import hydraulics
        hydraulics.run()

    if "grownfield" in topics and "hydraulics" in topics:
        print("\n[retrieval] combined cross-topic plots …")
        from evaluation.available.retrieval import available_eval
        available_eval.plot_combined()

    if not skip_synthetic:
        print("\n[retrieval] synthetic …")
        from evaluation.available.retrieval import synthetic
        synthetic.main()


# ── generation stage ────────────────────────────────────────────────────────
def run_generation(
    topics: list[str],
    skip_synthetic: bool = False,
    skip_answer_correctness: bool = False,
    qa_dataset: Path | None = None,
    limit: int | None = None,
) -> None:
    print("\n" + "=" * 64 + "\nGENERATION EVALUATION\n" + "=" * 64)

    # RAGAS on grownfield + hydraulics: generate answers → score → combined plot
    print("\n[generation] RAGAS generate …")
    from evaluation.available.generation import generate as _gen
    _gen.main(topics=topics)

    print("\n[generation] RAGAS score …")
    from evaluation.available.generation import ragas_eval as _ragas
    _ragas.main(topics=topics)

    if len(topics) >= 2:
        print("\n[generation] RAGAS combined plot …")
        from evaluation.available.generation import ragas_combined
        ragas_combined.main()

    # synthetic answer generation + RAGAS
    if not skip_synthetic:
        print("\n[generation] synthetic generation …")
        from evaluation.available.generation import synthetic_generation
        synthetic_generation.main()

    # answer_correctness backfill across datasets
    if not skip_answer_correctness:
        print("\n[generation] answer_correctness …")
        from evaluation.available.generation import answer_correctness
        answer_correctness.main(limit=limit)

    # optional: generic QA-dataset RAGAS flow (retrieval + generation + scoring)
    if qa_dataset is not None:
        print(f"\n[generation] QA-dataset flow: {qa_dataset} …")
        from evaluation.available.common.dataset_loader import load_qa_dataset
        from evaluation.available.generation.qa_runner import run_qa_dataset
        from evaluation.available.generation.evaluate_qa import run_ragas_evaluation

        out_dir = Path("evaluation/results") / datetime.now().strftime("%Y-%m-%d_%H-%M")
        dataset = load_qa_dataset(qa_dataset)
        results = run_qa_dataset(dataset, out_dir / "qa_intermediate")
        run_ragas_evaluation(results, out_dir)
        print(f"[generation] QA-dataset results → {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="AVAILABLE 2.0 evaluation — retrieval + generation."
    )
    p.add_argument("--only", choices=["retrieval", "generation"], default=None,
                   help="run only one stage (default: both).")
    p.add_argument("--topics", nargs="+", default=["grownfield", "hydraulics"],
                   metavar="TOPIC", help="topics for the graph/RAGAS evals.")
    p.add_argument("--skip-synthetic", action="store_true",
                   help="skip the synthetic dataset in both stages.")
    p.add_argument("--skip-answer-correctness", action="store_true",
                   help="skip the answer_correctness backfill in the generation stage.")
    p.add_argument("--qa-dataset", type=Path, default=None,
                   help="also run the generic QA-dataset RAGAS flow on this xlsx.")
    p.add_argument("--limit", type=int, default=None,
                   help="cap items where supported (answer_correctness).")
    args = p.parse_args()

    do_retrieval = args.only in (None, "retrieval")
    do_generation = args.only in (None, "generation")

    if do_retrieval:
        run_retrieval(args.topics, skip_synthetic=args.skip_synthetic)
    if do_generation:
        run_generation(
            args.topics,
            skip_synthetic=args.skip_synthetic,
            skip_answer_correctness=args.skip_answer_correctness,
            qa_dataset=args.qa_dataset,
            limit=args.limit,
        )

    print("\n[run_pipeline] done.")


if __name__ == "__main__":
    main()
