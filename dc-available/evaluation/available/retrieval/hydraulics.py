"""Available 2.0 — leg comparison on hydraulics: vector vs graph (PPR-merge) vs hybrid.

Document-level identification + retrieval metrics (RAGAS generation later). Single
source file per question, so match_mode="filename" and no module/submodule.
All logic is in available_eval.py; this file only declares the SPEC + legs.

Usage:
  poetry run python -m evaluation.available.retrieval.hydraulics [dataset.xlsx] [out_dir]
Default out_dir: evaluation/results/available_2.0/hydraulics/
"""
from __future__ import annotations

import sys
from pathlib import Path

from evaluation.available.retrieval import common as gc
from evaluation.available.retrieval import available_eval as ae

SPEC = gc.DatasetSpec(
    name="hydraulics",
    dataset_path=Path("evaluation/datasets/hydraluics/01_Teilnehmerunterlage_RT04-004_DE_dataset.xlsx"),
    match_mode="filename",
    col_qid=None,             # no question_id -> row index
    col_golden="source",      # golden = the single source file
    col_module=None,          # single file -> no module/submodule
    col_answer="answer",
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
