"""Shared paths + constants for the perturbation experiment.

Paths are anchored to the existing trust artifacts dir so this experiment lands
in its own ``perturbation/`` sub-folder and leaves the real pipeline untouched.
"""
from __future__ import annotations

from pathlib import Path

from dc_available.trust_score.components.score_content import _TRUST_DIR

# --- models ---------------------------------------------------------------- #
# Corrupt with a DIFFERENT model from the gpt-4o scorer, to avoid circularity.
CORRUPT_MODEL = "gpt-5.2"

# --- sampling -------------------------------------------------------------- #
SEED = 42
N_DOCS = 40

# --- perturbation taxonomy ------------------------------------------------- #
CONTENT_TYPES = ("procedural", "technical", "filler")      # LLM-corrupted summary
METADATA_TYPES = ("authority", "freshness")                # field override, no LLM
ALL_TYPES = CONTENT_TYPES + METADATA_TYPES

# Each perturbation -> the pillar it is meant to damage (the headline metric).
TARGET_PILLAR = {
    "procedural": "process_completeness",
    "technical": "technical_depth",
    "filler": "focus",
    "authority": "authority_score",
    "freshness": "freshness_score",
}
# procedural is also expected to pull down training_usefulness
SECONDARY_PILLAR = {"procedural": "training_usefulness"}

FRESHNESS_BACKDATE_YEARS = 4   # strong staleness

# --- files / folders (isolated) ------------------------------------------- #
PERT_DIR = Path(__file__).resolve().parent
SUMMARIES_DIR = PERT_DIR / "summaries"
FIGURES_DIR = PERT_DIR / "figures"

BASELINE_FILE = _TRUST_DIR / "metadata_trust.jsonl"   # source of the 40 + baseline pillars
SAMPLE_FILE = PERT_DIR / "sample.jsonl"
MANIFEST_FILE = PERT_DIR / "perturbations.jsonl"
RESULTS_FILE = PERT_DIR / "results.jsonl"
