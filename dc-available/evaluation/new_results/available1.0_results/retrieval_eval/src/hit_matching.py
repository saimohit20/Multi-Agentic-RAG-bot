"""
Per-topic golden matching (document level).

A "hit" = the golden source document appears among the retrieved chunks.
`golden_rank` = 1-based position of the FIRST retrieved chunk whose document is
the golden source; None if the golden document is absent from the top-k.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def golden_rank_grownfield(golden_codes: List[str], retrieved: List[Dict[str, Any]]) -> Optional[int]:
    """Grownfield: hit when a chunk's document code is one of the golden codes."""
    golden = {c for c in (golden_codes or [])}
    if not golden:
        return None
    for rank, item in enumerate(retrieved, start=1):
        if item.get("code") in golden:
            return rank
    return None


def golden_rank_hydraulics(golden_source: Optional[str], retrieved: List[Dict[str, Any]]) -> Optional[int]:
    """Hydraulics: hit when a chunk belongs to the golden source document.

    The dataset stores the source with extension (``*.pdf``); index chunks use
    the stem (part_of_document). Compare on the stem, tolerating either side
    being a substring of the other.
    """
    if not golden_source:
        return None
    stem = Path(str(golden_source)).stem
    if not stem:
        return None
    for rank, item in enumerate(retrieved, start=1):
        name = item.get("doc_name") or ""
        if name == stem or stem in name or name in stem:
            return rank
    return None


def golden_rank_synthetic(golden_source: Optional[str], retrieved: List[Dict[str, Any]]) -> Optional[int]:
    """Synthetic: EXACT document match, case-insensitive.

    Golden source is a filename with extension (e.g. ``Kapitel1.mp4``); index
    chunks expose the bare stem (e.g. ``Kapitel1``). Compare stems exactly after
    lowercasing, so ``Kapitel1`` does NOT match ``Kapitel10 … Kapitel19``.
    """
    if not golden_source:
        return None
    stem = Path(str(golden_source)).stem.strip().lower()
    if not stem:
        return None
    for rank, item in enumerate(retrieved, start=1):
        name = (item.get("doc_name") or "").strip().lower()
        if name == stem:
            return rank
    return None
