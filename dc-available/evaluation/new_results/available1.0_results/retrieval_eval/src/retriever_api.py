"""
Thin wrapper around the existing AVAILABLE retriever.

We load the already-built FAISS index (no rebuild) and run a plain
`similarity_search`, exactly like dc_available/check_retrieval.py, so we get the
raw child chunks with their metadata instead of the formatted curriculum output.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import paths_config as cfg  # noqa: F401  (side effect: puts repo root on sys.path)

_vectorstore = None

# Document codes look like  O2C-02.010 / I2O-02.010 / P2P-03.012 / STI-00.001
_CODE_RE = re.compile(r"[A-Z0-9]{2,4}-\d{2}\.\d{3}")

_IMAGE_SUFFIXES = ("__pptx_images", "__pdf_images", "__docx_images", "_images")


def get_vectorstore():
    """Load (once) and return the FAISS vectorstore from the existing cache."""
    global _vectorstore
    if _vectorstore is None:
        from dc_available.RAG.retrieval_helpers import get_retriever

        print("[Eval] Loading retriever (embeddings + existing FAISS index)...")
        retriever = get_retriever()
        if retriever is None:
            raise RuntimeError("Retriever could not be loaded (get_retriever returned None).")
        _vectorstore = retriever.vectorstore
    return _vectorstore


def chunk_doc_name(metadata: Dict[str, Any]) -> str:
    """Human/document-level name for a retrieved chunk.

    Prefers `part_of_document`; falls back to parsing the storage_path folder
    (dropping trailing *_images markers), mirroring check_retrieval._doc_name.
    """
    name = metadata.get("part_of_document")
    if name:
        return str(name)

    storage_path = metadata.get("storage_path") or ""
    p = Path(str(storage_path))
    parent = p.parent.name
    for suffix in _IMAGE_SUFFIXES:  # specific (__*_images) before generic (_images)
        if parent.endswith(suffix):
            parent = parent[: -len(suffix)]
            break
    return parent or p.stem or "Unknown"


def doc_code(doc_name: str | None) -> str | None:
    """Extract the Grownfield document code (e.g. O2C-02.010) from a doc name."""
    if not doc_name:
        return None
    m = _CODE_RE.search(doc_name)
    return m.group(0) if m else None


def search(query: str, k: int = cfg.TOP_K) -> List[Dict[str, Any]]:
    """Return the top-k retrieved chunks as lightweight dicts (rank order)."""
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=k)
    results: List[Dict[str, Any]] = []
    for d in docs:
        name = chunk_doc_name(d.metadata)
        results.append(
            {
                "doc_name": name,
                "code": doc_code(name),
                "text": d.page_content,
            }
        )
    return results
