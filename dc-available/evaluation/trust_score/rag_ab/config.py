"""Shared paths + constants for the RAG A/B poisoning test (Part 2).

Isolated from production: its own Qdrant collection, docstore cache, and trust
table all live under a dedicated ``rag_ab/`` folder.
"""
from __future__ import annotations

from pathlib import Path

from dc_available.trust_score.components.score_content import _TRUST_DIR

# --- models --------------------------------------------------------------- #
CORRUPT_MODEL = "gpt-5.2"          # same corruptor as Part 1 (writes the twins)

# --- test set ------------------------------------------------------------- #
SEED = 42
N_DOCS = 20                        # good docs; +1 twin each = 40-doc corpus
N_FRESHNESS = 10                   # twins made low-trust via freshness
N_AUTHORITY = 10                   # twins made low-trust via authority
FACTS_PER_TWIN = 10                # ~10 changed facts -> ~10 questions -> ~200 total

# --- isolated index ------------------------------------------------------- #
TEST_COLLECTION = "available_trust_test"     # separate Qdrant collection

# --- files / folders ------------------------------------------------------ #
AB_DIR = Path(__file__).resolve().parent
TWINS_DIR = AB_DIR / "twins"                 # twin document text
INDEX_CACHE = AB_DIR / "index_cache"         # test docstore pickle (index_ret.pkl)
FIGURES_DIR = AB_DIR / "figures"

BASELINE_FILE = _TRUST_DIR / "metadata_trust.jsonl"   # source of the 20 good docs
DOCS_FILE = AB_DIR / "docs.jsonl"            # the 20 selected good docs (+ lever)
DATASET_FILE = AB_DIR / "dataset.jsonl"      # ~200 questions (question, gold, good/twin ids, lever)
TEST_TRUST_FILE = AB_DIR / "test_trust.jsonl"  # trust table for the 40-doc corpus (good high / twin low)
AB_RESULTS_FILE = AB_DIR / "ab_results.jsonl"  # (legacy) per-question ON/OFF outcomes

# Per-arm result/debug files written by run_ab.py. Each is BOTH the debug view
# (docs + chunks with rerank|trust|final) and the RAGAS input (question, answer,
# contexts, ground_truth). One file per arm so each is easy to eyeball on its own.
AB_ARM_FILES = {
    "off":   AB_DIR / "ab_off.jsonl",
    "prior": AB_DIR / "ab_prior.jsonl",
    "full":  AB_DIR / "ab_full.jsonl",
}
AB_SCORES_FILE = AB_DIR / "ab_scores.jsonl"    # RAGAS answer_correctness per (question, arm)
AB_GEN_FILE = AB_DIR / "ab_gen.jsonl"          # generation graders: exact + own LLM judge per (question, arm)
