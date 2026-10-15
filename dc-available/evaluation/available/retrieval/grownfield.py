"""Available 2.0 — leg comparison on Grownfield: vector vs graph (PPR-merge) vs hybrid.

Document-level identification + retrieval metrics (RAGAS generation later).
All logic is in available_eval.py; this file only declares the SPEC + legs.

Usage:
  poetry run python -m evaluation.available.retrieval.grownfield [dataset.xlsx] [out_dir]
Default out_dir: evaluation/results/available_2.0/grownfield/
"""
from __future__ import annotations

import sys
from pathlib import Path

from evaluation.available.retrieval import common as gc
from evaluation.available.retrieval import available_eval as ae

SPEC = gc.DatasetSpec(
    name="grownfield",
    dataset_path=Path("evaluation/datasets/Grownfield/dataset_full.xlsx"),
    match_mode="code",
    col_qid="question_id",
    col_golden="golden_codes",
    col_module="modules",
    col_submodule="submodules",
    col_answer="golden_answer",
    col_qtype="question_type",
)

LEGS = {
    "vector": {"FUSION_LEGS": "vector"},
    "graph":  {"FUSION_LEGS": "graph", "GRAPH_RETRIEVAL_MODE": "ppr", "PPR_SEED_MODE": "merge"},
    "hybrid": {"FUSION_LEGS": "both",  "GRAPH_RETRIEVAL_MODE": "ppr", "PPR_SEED_MODE": "merge"},
}
LABELS = {"vector": "Vector", "graph": "Graph\n(PPR merge)", "hybrid": "Hybrid"}


def run() -> None:
    ae.cli_main(SPEC, LEGS, LABELS)


if __name__ == "__main__":
    run()
