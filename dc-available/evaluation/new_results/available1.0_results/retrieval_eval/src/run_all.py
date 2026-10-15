"""
Run the FULL retrieval evaluation in one shot:
  1) Grownfield  -> results/grownfield_results.json
  2) Hydraulics  -> results/hydraulics_results.json
  3) Plots       -> results/plots/{identification,retrieval_metrics}.png

    .venv/Scripts/python.exe evaluation/retrieval_eval/src/run_all.py

The FAISS index + embeddings load only once (retriever_api caches them), so
running both topics here is cheaper than launching the two scripts separately.
"""
from __future__ import annotations

import run_grownfield
import run_hydraulics
import run_synthetic
import make_plots


def main() -> None:
    print("\n========== 1/4  GROWNFIELD ==========")
    run_grownfield.main()

    print("\n========== 2/4  HYDRAULICS ==========")
    run_hydraulics.main()

    print("\n========== 3/4  SYNTHETIC ==========")
    run_synthetic.main()

    print("\n========== 4/4  PLOTS ==========")
    make_plots.main()

    print("\n[Done] Retrieval evaluation complete.")


if __name__ == "__main__":
    main()
