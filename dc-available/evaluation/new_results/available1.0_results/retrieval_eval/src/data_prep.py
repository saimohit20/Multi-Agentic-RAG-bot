"""
Load the two evaluation datasets (Excel) and normalize them into a common
record shape used by the runners.

Grownfield  -> golden_source is a LIST of document codes (e.g. ["O2C-02.010"]).
Hydraulics  -> golden_source is a single source-document name (e.g. the PDF).

`golden_chunk` / `golden_conf` (hydraulics) are intentionally ignored.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import pandas as pd

import paths_config as cfg


def _clean(val: Any) -> str | None:
    """Return a stripped string, or None for NaN/empty."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    return s or None


def _split_codes(val: Any) -> List[str]:
    """Split a golden_codes cell into a list of codes.

    Handles multi-valued cells like "Q2E-02.010, Q2E-02.011" or newline-separated.
    """
    s = _clean(val)
    if not s:
        return []
    parts = re.split(r"[,\n;/]+", s)
    return [p.strip() for p in parts if p.strip()]


def load_grownfield() -> List[Dict[str, Any]]:
    df = pd.read_excel(cfg.GROWNFIELD_XLSX)
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        records.append(
            {
                "question_id": _clean(row.get("question_id")),
                "question": _clean(row.get("question")),
                "question_type": (_clean(row.get("question_type")) or "").lower() or None,
                "level": _clean(row.get("level")),
                "type": _clean(row.get("type")),
                "golden_source": _split_codes(row.get("golden_codes")),
                "golden_answer": _clean(row.get("golden_answer")),
                "module": _clean(row.get("modules")),
                "submodule": _clean(row.get("submodules")),
            }
        )
    return records


def load_hydraulics() -> List[Dict[str, Any]]:
    df = pd.read_excel(cfg.HYDRAULICS_XLSX)
    records: List[Dict[str, Any]] = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        records.append(
            {
                "question_id": f"hy_{i:03d}",  # dataset has no id column
                "question": _clean(row.get("question")),
                "question_type": (_clean(row.get("question_type")) or "").lower() or None,
                "level": _clean(row.get("level")),
                "type": _clean(row.get("type")),
                "golden_source": _clean(row.get("source")),  # single document name
                "golden_answer": _clean(row.get("answer")),
            }
        )
    return records


def load_synthetic() -> List[Dict[str, Any]]:
    """Synthetic dataset: single `source_document` as golden source (like Hydraulics).

    Columns: topic, level, type, source_document, question, question_type,
    answer, keywords. No id column -> generate syn_0001 … ids.
    """
    df = pd.read_excel(cfg.SYNTHETIC_XLSX)
    records: List[Dict[str, Any]] = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        records.append(
            {
                "question_id": f"syn_{i:04d}",  # dataset has no id column
                "question": _clean(row.get("question")),
                "question_type": (_clean(row.get("question_type")) or "").lower() or None,
                "level": _clean(row.get("level")),
                "type": _clean(row.get("type")),
                "golden_source": _clean(row.get("source_document")),  # single document name
                "golden_answer": _clean(row.get("answer")),
                "topic": _clean(row.get("topic")),
                "keywords": _clean(row.get("keywords")),
            }
        )
    return records
