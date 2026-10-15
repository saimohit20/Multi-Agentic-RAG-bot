"""
Central paths + constants for the generation / RAGAS evaluation.

Module names are intentionally unique (gen_paths_config, generate_answers, …)
to avoid shadowing installed packages.
"""
from __future__ import annotations

import sys
from pathlib import Path

# generation_eval/src/gen_paths_config.py
#   parents[0] = src
#   parents[1] = generation_eval
#   parents[2] = evaluation
#   parents[3] = dc-available  (repo root)
REPO_ROOT = Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Directories ------------------------------------------------------------
EVAL_DIR        = REPO_ROOT / "evaluation"
RETRIEVAL_DIR   = EVAL_DIR / "retrieval_eval" / "results"
RESULTS_DIR     = EVAL_DIR / "generation_eval" / "results"
PLOTS_DIR       = RESULTS_DIR / "plots"

# --- Input files (from retrieval eval) --------------------------------------
GROWNFIELD_RETRIEVAL = RETRIEVAL_DIR / "grownfield_results.json"
HYDRAULICS_RETRIEVAL = RETRIEVAL_DIR / "hydraulics_results.json"
SYNTHETIC_RETRIEVAL  = RETRIEVAL_DIR / "synthetic_results.json"

# --- Output files -----------------------------------------------------------
GROWNFIELD_GEN_RESULTS = RESULTS_DIR / "grownfield_gen_results.json"
HYDRAULICS_GEN_RESULTS = RESULTS_DIR / "hydraulics_gen_results.json"
SYNTHETIC_GEN_RESULTS  = RESULTS_DIR / "synthetic_gen_results.json"
RAGAS_SUMMARY          = RESULTS_DIR / "ragas_summary.json"

# Per-topic summary file name, e.g. results/synthetic_ragas_summary.json
def ragas_summary_path(topic_key: str) -> Path:
    return RESULTS_DIR / f"{topic_key}_ragas_summary.json"

# --- RAGAS metric names (must match ragas metric .name attributes) ----------
RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
