"""
Run the FULL generation / RAGAS evaluation in one shot:
  1) Generate answers  -> grownfield_gen_results.json + hydraulics_gen_results.json
  2) RAGAS scoring     -> updates those files + ragas_summary.json
  3) Plots             -> ragas_metrics.png + ragas_radar.png

    .venv/Scripts/python.exe evaluation/generation_eval/src/run_gen_all.py

Prerequisite: retrieval eval must have been run first so the input JSONs exist.
"""
from __future__ import annotations

import generate_answers
import run_ragas
import make_gen_plots


def main(topics: list[str] | None = None) -> None:
    print("\n========== 1/3  GENERATE ANSWERS ==========")
    generate_answers.main(topics)

    print("\n========== 2/3  RAGAS SCORING ==========")
    run_ragas.main(topics)

    print("\n========== 3/3  PLOTS ==========")
    make_gen_plots.main()

    print("\n[Done] Generation evaluation complete.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--topic",
        choices=["grownfield", "hydraulics", "synthetic", "all"],
        default="all",
    )
    args = ap.parse_args()
    main(None if args.topic == "all" else [args.topic])
