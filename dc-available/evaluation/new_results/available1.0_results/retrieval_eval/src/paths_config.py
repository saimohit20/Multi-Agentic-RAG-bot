"""
Central paths + constants for the retrieval evaluation.

Importing this module also puts the repo root on sys.path so that
`import dc_available...` works no matter where the script is launched from.

Module names in this package are intentionally unique (paths_config, data_prep,
retriever_api, ...) to avoid shadowing installed packages such as `datasets`
or `config` that dc_available / sentence-transformers import at runtime.
"""
from __future__ import annotations

import sys
from pathlib import Path

# evaluation/retrieval_eval/src/paths_config.py
#   parents[0] = src
#   parents[1] = retrieval_eval
#   parents[2] = evaluation
#   parents[3] = dc-available   (repo root)
REPO_ROOT = Path(__file__).resolve().parents[3]

# Make `dc_available` importable regardless of the current working directory.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Directories ------------------------------------------------------------
EVAL_DIR = REPO_ROOT / "evaluation"
DATASETS_DIR = EVAL_DIR / "datasets"
RESULTS_DIR = EVAL_DIR / "retrieval_eval" / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# --- Dataset files ----------------------------------------------------------
GROWNFIELD_XLSX = DATASETS_DIR / "Grownfield" / "dataset_full.xlsx"
HYDRAULICS_XLSX = DATASETS_DIR / "hydraluics" / "01_Teilnehmerunterlage_RT04-004_DE_dataset.xlsx"
SYNTHETIC_XLSX = DATASETS_DIR / "Synthetic" / "generated_dataset.xlsx"

# --- Result files -----------------------------------------------------------
GROWNFIELD_RESULTS = RESULTS_DIR / "grownfield_results.json"
HYDRAULICS_RESULTS = RESULTS_DIR / "hydraulics_results.json"
SYNTHETIC_RESULTS = RESULTS_DIR / "synthetic_results.json"

# --- Evaluation constants ---------------------------------------------------
TOP_K = 10   # how many chunks we retrieve + store per question
HIT_K = 5    # hit@k threshold used for the headline metric


def ensure_dirs() -> None:
    """Create the results/plots folders if they don't exist yet."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
