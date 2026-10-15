"""Stage 1 — the Personalized-PageRank graph object.

Loads a topic's ``nodes.jsonl`` / ``edges.jsonl`` (the same artifacts the current
graph leg reads) and assembles ONE sparse, symmetric, weighted adjacency over all
node types, so Personalized PageRank can propagate across BOTH layers we already
built:

    * semantic   — Chunk --mentions--> Entity, and the ~70 Entity<->Entity
                   relation types (requires / precedes / prerequisite_for / ...)
    * structural — Chunk --part_of/contains--> Document --> Level --> Subtopic
                   (this is the hierarchy the current leg ignores), plus
                   Chunk <-> Chunk sequence links (related_to)

Each edge family gets its own weight (``EdgeWeights``) so we can tune how much the
structure vs. the semantics pulls the random walk. The object exposes the
column-stochastic transition matrix ``S`` used by the PPR power iteration in
Stage 3; seeding and read-out live there, not here — this module is pure graph
construction and is safe to unit-test on its own.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

# Edge families. Everything that is not one of the four structural/semantic-link
# types below is an LLM-extracted Entity<->Entity relation (requires, precedes,
# leads_to, ...), which we treat as one "semantic relation" family.
_MENTIONS = "mentions"
_RELATED = "related_to"
_STRUCTURAL = {"part_of", "contains"}


@dataclass(frozen=True)
class EdgeWeights:
    """Per-family edge weights for the PPR adjacency (tunable in Stage 5)."""

    mentions: float = 1.0        # Chunk <-> Entity (the core signal)
    semantic: float = 0.8        # Entity <-> Entity relations (multi-hop reasoning)
    related_to: float = 0.5      # Chunk <-> Chunk sequence links
    structural: float = 0.3      # part_of / contains hierarchy (pool inside a doc/module)

    def key(self) -> tuple:
        return (self.mentions, self.semantic, self.related_to, self.structural)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class PPRGraph:
    """One topic's graph as a sparse transition matrix, ready for PPR.

    Attributes:
        node_ids:   index -> node id (str).
        id_to_idx:  node id -> index.
        node_type:  index -> node type ("Chunk", "Entity", "Document", ...).
        chunk_idx:  1-D int array of the indices whose type == "Chunk".
        chunk_document: chunk node id -> owning document name (for read-out/debug).
        A:          symmetric weighted adjacency (CSR).
    """

    def __init__(self, topic_key: str, weights: EdgeWeights) -> None:
        self.topic_key = topic_key
        self.weights = weights
        self.node_ids: list[str] = []
        self.id_to_idx: dict[str, int] = {}
        self.node_type: list[str] = []
        self.chunk_document: dict[str, str] = {}
        self.A: sparse.csr_matrix | None = None
        self.chunk_idx: np.ndarray | None = None
        self._S: sparse.csr_matrix | None = None

    # -- construction --------------------------------------------------------

    @classmethod
    def build(
        cls, graph_root: Path, topic_key: str, weights: EdgeWeights | None = None
    ) -> "PPRGraph":
        weights = weights or EdgeWeights()
        self = cls(topic_key, weights)
        topic_dir = Path(graph_root) / topic_key
        nodes = _read_jsonl(topic_dir / "nodes.jsonl")
        edges = _read_jsonl(topic_dir / "edges.jsonl")

        # 1. Index every node; remember its type and (for chunks) its document.
        for node in nodes:
            nid = str(node["id"])
            if nid in self.id_to_idx:
                continue
            self.id_to_idx[nid] = len(self.node_ids)
            self.node_ids.append(nid)
            ntype = str(node.get("type", ""))
            self.node_type.append(ntype)
            if ntype == "Chunk":
                doc = (node.get("properties") or {}).get("document_id")
                if doc:
                    self.chunk_document[nid] = str(doc)

        n = len(self.node_ids)

        # 2. Build the symmetric weighted adjacency from the edges. Skip any edge
        #    whose endpoint is not a known node (dangling references stay out).
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for edge in edges:
            src = self.id_to_idx.get(str(edge.get("source")))
            dst = self.id_to_idx.get(str(edge.get("target")))
            if src is None or dst is None or src == dst:
                continue
            w = self._edge_weight(edge)
            if w <= 0.0:
                continue
            # undirected: add both directions
            rows.append(src); cols.append(dst); data.append(w)
            rows.append(dst); cols.append(src); data.append(w)

        # coo -> csr sums duplicate (i, j) entries, so parallel edges accumulate.
        self.A = sparse.coo_matrix(
            (data, (rows, cols)), shape=(n, n), dtype=np.float64
        ).tocsr()
        self.chunk_idx = np.array(
            [i for i, t in enumerate(self.node_type) if t == "Chunk"], dtype=np.int64
        )
        return self

    def _edge_weight(self, edge: dict[str, Any]) -> float:
        etype = str(edge.get("type", ""))
        w = self.weights
        if etype == _MENTIONS:
            return w.mentions
        if etype == _RELATED:
            return w.related_to
        if etype in _STRUCTURAL:
            return w.structural
        # Any other type is an LLM Entity<->Entity relation; scale by confidence.
        conf = (edge.get("properties") or {}).get("confidence", 1.0)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 1.0
        return w.semantic * conf

    # -- accessors -----------------------------------------------------------

    def transition_matrix(self) -> sparse.csr_matrix:
        """Column-stochastic transition ``S = A · D⁻¹`` (cached).

        Each column j is normalised by that node's total edge weight, so PPR
        ``π = α·S·π + (1−α)·e`` spreads a node's mass evenly over its weighted
        neighbours — a high-degree hub hands each neighbour only a small share
        (the structural analogue of IDF).
        """
        if self._S is None:
            assert self.A is not None
            deg = np.asarray(self.A.sum(axis=0)).ravel()
            inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
            self._S = (self.A @ sparse.diags(inv)).tocsr()
        return self._S

    def idx_of(self, node_id: str) -> int | None:
        return self.id_to_idx.get(node_id)

    def id_of(self, idx: int) -> str:
        return self.node_ids[idx]

    def num_nodes(self) -> int:
        return len(self.node_ids)

    def stats(self) -> dict[str, int]:
        from collections import Counter

        types = Counter(self.node_type)
        return {
            "nodes": len(self.node_ids),
            "edges_nnz": int(self.A.nnz) if self.A is not None else 0,
            "chunks": types.get("Chunk", 0),
            "entities": types.get("Entity", 0),
            "documents": types.get("Document", 0),
        }


# ---------------------------------------------------------------------------
# Module-level cache: one built graph per (topic, edge-weight configuration).
# ---------------------------------------------------------------------------
_ppr_cache: dict[tuple[str, tuple], PPRGraph] = {}


def get_ppr_graph(
    graph_root: Path, topic_key: str, weights: EdgeWeights | None = None
) -> PPRGraph:
    """Return the cached PPR graph for a topic, building it on first use."""
    weights = weights or EdgeWeights()
    cache_key = (topic_key, weights.key())
    if cache_key not in _ppr_cache:
        _ppr_cache[cache_key] = PPRGraph.build(graph_root, topic_key, weights)
    return _ppr_cache[cache_key]
