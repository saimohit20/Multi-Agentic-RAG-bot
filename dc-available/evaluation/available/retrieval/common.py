"""Shared helpers for the AVAILABLE retrieval evaluations.

Dataset-neutral: the `DatasetSpec` (column map + match mode), golden-key matching,
and the shared constants used by `available_eval.py`. Each dataset's script
(`grownfield.py` / `hydraulics.py`) declares a `DatasetSpec` + leg overrides and
calls the shared evaluation loop in `available_eval.py`.

`match_mode` decides how a retrieved document is judged "golden":
    "code"      Grownfield source codes  (e.g. O2C-02.010)
    "filename"  a single source file      (e.g. ..._DE.pdf)

The graph itself is never told which dataset a question came from — a question is
just a query; topic routing inside the RAG stack sends it to the right graph.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePath

CODE_RE = re.compile(r"[A-Z0-9]{3}-\d{2}\.\d{3}")
K_LIST = [1, 3, 10]
TOP_DOCS = 10


# --- dataset spec ------------------------------------------------------------
@dataclass
class DatasetSpec:
    """Everything dataset-specific the shared code needs. Declared by each
    dataset's scripts; nothing here is Grownfield- or hydraulics-aware."""
    name: str                                   # used in the results folder name
    dataset_path: Path
    match_mode: str = "code"                    # "code" | "filename"
    col_question: str = "question"
    col_qid: str | None = "question_id"         # None -> use the row index
    col_golden: str = "golden_codes"            # codes column OR source-filename column
    col_module: str | None = "modules"          # None -> not a grownfield-style dataset
    col_qtype: str = "question_type"
    col_golden_chunk: str | None = None         # None -> no chunk-level scoring
    col_golden_conf: str | None = None          # optional confidence column (carried through)
    col_answer: str | None = None               # golden/reference answer (for RAGAS later)
    col_submodule: str | None = None            # grownfield only (carried through)


# --- golden / matching (the only place match_mode branches) ------------------
def _codes(value) -> set[str]:
    """All source codes (e.g. O2C-02.010) found in a string/list."""
    text = " ".join(value) if isinstance(value, (list, tuple)) else str(value or "")
    return {c.upper() for c in CODE_RE.findall(text)}


def _stem(value) -> str:
    """Normalise a file reference to a comparable key: basename, no extension,
    lower-cased. '01_..._DE.pdf' and '01_..._DE' both -> '01_..._de'."""
    return PurePath(str(value or "").strip()).stem.lower()


def golden_of(value, match_mode: str) -> set[str]:
    """The set of golden DOCUMENT keys for one dataset row."""
    if match_mode == "filename":
        stem = _stem(value)
        return {stem} if stem else set()
    return _codes(value)


def _doc_keys(name: str, match_mode: str) -> set[str]:
    """The golden keys a single retrieved document name maps to."""
    return {_stem(name)} if match_mode == "filename" else _codes(name)
