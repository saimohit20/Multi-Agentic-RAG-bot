"""Graph-based retrieval expansion (Approach 1).

Flow:
    vector search chunk IDs
        → Chunk nodes in graph
        → follow `mentions` edges → Entity nodes
        → follow reverse `mentions` edges → neighbouring Chunk nodes
        → return expanded chunk IDs not already in vector search results
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Any

import opik

from dc_available.config import data_config

os.environ.setdefault("OPIK_URL_OVERRIDE", "http://localhost:5173/api")
os.environ.setdefault("OPIK_PROJECT_NAME", "dc-available-rag")
os.environ.setdefault("OPIK_WORKSPACE", "default")

logger = logging.getLogger(__name__)
_MODULE_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{1,6}$")
_QUERY_STOP_TOKENS = {
    "the", "and", "for", "with", "from", "into", "onto", "over", "under",
    "what", "which", "when", "where", "who", "whom", "whose", "why", "how",
    "should", "could", "would", "can", "used", "using", "use", "create",
    "creating", "about", "this", "that", "these", "those", "have", "has", "had",
    "your", "their", "there", "here", "into", "than", "then",
}


def infer_topic_keys_from_storage_paths(
    storage_paths: list[str], graph_root: Path
) -> list[str]:
    """Map document storage paths to graph topic-folder names (the top-level trees).

    A path belongs to topic <T> when '/<T>/' appears in it (case-insensitive),
    where <T> is an existing graph topic folder. This keeps retrieval aligned
    with how the graph is keyed (one folder per top-level topic), so expansion
    looks in the right tree even when a path has subtopic branches
    (e.g. GROWNFIELD/O2C) or is an image chunk.
    """
    graph_root = Path(graph_root)
    if not graph_root.is_dir():
        return []
    dirs = [d.name for d in graph_root.iterdir() if d.is_dir()]
    keys: set[str] = set()
    for sp in storage_paths:
        low = str(sp).replace("\\", "/").lower()
        for name in dirs:
            if f"/{name.lower()}/" in low:
                keys.add(name)
    return sorted(keys)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


class _GraphIndex:
    """Lightweight in-memory graph index for a single topic.

    Stores only what we need for retrieval:
    - chunk_ids: set of all Chunk node ids
    - chunk_to_entities: chunk_id -> list of entity_ids (via mentions edges)
    - entity_to_chunks: entity_id -> list of chunk_ids (reverse mentions)
    - chunk_to_document: chunk_id -> document_id (Stage C MaxP prior; doc_of)
    """

    def __init__(self, topic_graph_dir: Path) -> None:
        self.topic_graph_dir = topic_graph_dir
        self.chunk_ids: set[str] = set()
        self.chunk_to_entities: dict[str, list[str]] = {}
        self.entity_to_chunks: dict[str, list[str]] = {}
        self.chunk_to_document: dict[str, str] = {}
        self.document_to_chunks: dict[str, list[str]] = {}
        self.document_to_module: dict[str, str] = {}
        self.module_to_documents: dict[str, list[str]] = {}
        self.chunk_neighbors: dict[str, list[str]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        nodes_path = self.topic_graph_dir / "nodes.jsonl"
        edges_path = self.topic_graph_dir / "edges.jsonl"

        # Collect chunk node ids (+ their owning Document, for Stage C doc_of)
        for node in _read_jsonl(nodes_path):
            if node.get("type") == "Chunk":
                chunk_id = str(node["id"])
                self.chunk_ids.add(chunk_id)
                document_id = node.get("properties", {}).get("document_id")
                if document_id is not None:
                    document_id = str(document_id)
                    self.chunk_to_document[chunk_id] = document_id
                    self.document_to_chunks.setdefault(document_id, []).append(chunk_id)
                    module = _module_key_from_document(document_id)
                    self.document_to_module[document_id] = module
                    docs = self.module_to_documents.setdefault(module, [])
                    if document_id not in docs:
                        docs.append(document_id)

        # Build mentions mappings and lightweight local chunk-neighbor links.
        for edge in _read_jsonl(edges_path):
            edge_type = str(edge.get("type"))
            source = str(edge.get("source"))
            target = str(edge.get("target"))

            if edge_type == "mentions":
                if source not in self.chunk_to_entities:
                    self.chunk_to_entities[source] = []
                self.chunk_to_entities[source].append(target)

                if target not in self.entity_to_chunks:
                    self.entity_to_chunks[target] = []
                self.entity_to_chunks[target].append(source)
                continue

            if edge_type == "related_to" and source in self.chunk_ids and target in self.chunk_ids:
                src_neighbors = self.chunk_neighbors.setdefault(source, [])
                if target not in src_neighbors:
                    src_neighbors.append(target)

        self._loaded = True
        logger.info(
            "GraphIndex loaded for %s: %d chunks, %d entities, %d documents, %d modules",
            self.topic_graph_dir.name,
            len(self.chunk_ids),
            len(self.entity_to_chunks),
            len(self.document_to_chunks),
            len(self.module_to_documents),
        )

    # -- Stage C accessors (query-seeded retrieve). Do not affect expand(). ---

    def chunks_mentioning(self, entity_id: str) -> list[str]:
        """Chunk ids that mention this entity (reverse `mentions`), in this topic."""
        self._ensure_loaded()
        return self.entity_to_chunks.get(entity_id, [])

    def entity_count(self, chunk_id: str) -> int:
        """Total entities mentioned by this chunk (length-normalisation divisor)."""
        self._ensure_loaded()
        return len(self.chunk_to_entities.get(chunk_id, []))

    def entities_of(self, chunk_id: str) -> list[str]:
        """Entity ids this chunk mentions (forward `mentions`), in this topic."""
        self._ensure_loaded()
        return self.chunk_to_entities.get(chunk_id, [])

    def doc_of(self, chunk_id: str) -> str | None:
        """Document id owning this chunk (MaxP prior). None if unknown."""
        self._ensure_loaded()
        return self.chunk_to_document.get(chunk_id)

    def chunks_of_doc(self, document_id: str) -> list[str]:
        self._ensure_loaded()
        return self.document_to_chunks.get(document_id, [])

    def module_of_doc(self, document_id: str) -> str:
        self._ensure_loaded()
        return self.document_to_module.get(document_id, "unknown")

    def module_of_chunk(self, chunk_id: str) -> str:
        self._ensure_loaded()
        document_id = self.chunk_to_document.get(chunk_id)
        if not document_id:
            return "unknown"
        return self.document_to_module.get(document_id, "unknown")

    def docs_of_module(self, module: str) -> list[str]:
        self._ensure_loaded()
        return self.module_to_documents.get(module, [])

    def neighbor_chunks(self, chunk_id: str) -> list[str]:
        self._ensure_loaded()
        return self.chunk_neighbors.get(chunk_id, [])

    def expand_with_traversal(
        self, seed_chunk_ids: list[str], max_expanded: int = 20
    ) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
        """Entity-hop expansion that also returns traversal data for graph viz.

        Collects entities from the seed chunks, then the chunks that mention them
        (excluding seeds), ranked by shared-entity count and capped at max_expanded.

        Returns:
            (expanded_ids, seed_to_entities, entity_to_expanded)
            - seed_to_entities:   {seed_chunk_id: [entity_id, ...]}
            - entity_to_expanded: {entity_id: [expanded_chunk_id, ...]}
        """
        self._ensure_loaded()
        seed_set = set(seed_chunk_ids)

        # Step 1: collect entities from seed chunks
        seed_to_entities: dict[str, list[str]] = {}
        reached_entities: set[str] = set()
        for chunk_id in seed_chunk_ids:
            entities = self.chunk_to_entities.get(chunk_id, [])
            if entities:
                seed_to_entities[chunk_id] = entities
            reached_entities.update(entities)

        if not reached_entities:
            return [], {}, {}

        # Step 2: find candidate chunks and track which entity linked them
        candidate_score: dict[str, int] = {}
        entity_to_candidates: dict[str, list[str]] = {}
        for entity_id in reached_entities:
            for chunk_id in self.entity_to_chunks.get(entity_id, []):
                if chunk_id in seed_set:
                    continue
                candidate_score[chunk_id] = candidate_score.get(chunk_id, 0) + 1
                entity_to_candidates.setdefault(entity_id, []).append(chunk_id)

        # Step 3: rank, cap, filter traversal to only returned chunks
        ranked = sorted(candidate_score.items(), key=lambda x: -x[1])
        expanded_ids = [cid for cid, _ in ranked[:max_expanded]]
        expanded_set = set(expanded_ids)

        entity_to_expanded: dict[str, list[str]] = {
            eid: [cid for cid in cids if cid in expanded_set]
            for eid, cids in entity_to_candidates.items()
            if any(cid in expanded_set for cid in cids)
        }

        return expanded_ids, seed_to_entities, entity_to_expanded


# ---------------------------------------------------------------------------
# Module-level cache: one index per topic key, loaded on first use
# ---------------------------------------------------------------------------
_index_cache: dict[str, _GraphIndex] = {}


def _get_index(graph_root: Path, topic_key: str) -> _GraphIndex:
    if topic_key not in _index_cache:
        topic_dir = graph_root / topic_key
        _index_cache[topic_key] = _GraphIndex(topic_dir)
    return _index_cache[topic_key]


def entities_for_chunks(
    graph_root: Path,
    chunk_ids: list[str],
    topic_keys: list[str] | None = None,
    max_per_chunk: int | None = None,
) -> dict[str, list[str]]:
    """Map each chunk id -> the entity ids it mentions, across topics.

    Reads the same `mentions` edges the retriever uses, so it resolves entities for
    ANY chunk (vector- or graph-sourced) — used by the QA retrieval-graph viz. When
    ``max_per_chunk`` is set, each chunk keeps only its first N entities.
    """
    if not chunk_ids:
        return {}
    if not topic_keys:
        topic_keys = [d.name for d in graph_root.iterdir() if d.is_dir()]
    wanted = set(chunk_ids)
    result: dict[str, list[str]] = {}
    for topic_key in topic_keys:
        index = _get_index(graph_root, topic_key)
        for cid in wanted:
            ents = index.entities_of(cid)
            if not ents:
                continue
            bucket = result.setdefault(cid, [])
            for e in ents:
                if e not in bucket:
                    bucket.append(e)
    if max_per_chunk is not None:
        result = {cid: ents[:max_per_chunk] for cid, ents in result.items()}
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@opik.track(name="graph_retrieval")
def expand_chunks_via_graph_with_traversal(
    vector_chunk_ids: list[str],
    graph_root: Path,
    topic_keys: list[str],
    max_expanded: int = 20,
) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    """Entity-hop graph expansion that also returns entity traversal data.

    Given vector-search chunk IDs, returns additional chunk IDs reached via shared
    entities (per topic, deduplicated, ranked by shared-entity count), plus:
        (expanded_ids, seed_to_entities, entity_to_expanded)
        - seed_to_entities:   {seed_chunk_id: [entity_id, ...]}
        - entity_to_expanded: {entity_id: [expanded_chunk_id, ...]}
    """
    if not vector_chunk_ids:
        return [], {}, {}

    if not topic_keys:
        topic_keys = [d.name for d in graph_root.iterdir() if d.is_dir()]

    seed_set = set(vector_chunk_ids)
    all_expanded: dict[str, int] = {}
    merged_seed_to_entities: dict[str, list[str]] = {}
    merged_entity_to_expanded: dict[str, list[str]] = {}

    for topic_key in topic_keys:
        index = _get_index(graph_root, topic_key)
        expanded, seed_to_ent, ent_to_exp = index.expand_with_traversal(
            vector_chunk_ids, max_expanded=max_expanded
        )
        for chunk_id in expanded:
            if chunk_id not in seed_set:
                all_expanded[chunk_id] = all_expanded.get(chunk_id, 0) + 1

        for chunk_id, entities in seed_to_ent.items():
            existing = merged_seed_to_entities.setdefault(chunk_id, [])
            for e in entities:
                if e not in existing:
                    existing.append(e)

        for entity_id, chunks in ent_to_exp.items():
            existing = merged_entity_to_expanded.setdefault(entity_id, [])
            for c in chunks:
                if c not in existing:
                    existing.append(c)

    ranked = sorted(all_expanded.items(), key=lambda x: -x[1])
    result = [cid for cid, _ in ranked[:max_expanded]]
    expanded_set = set(result)

    # Trim entity_to_expanded to only chunks that made the final cut
    final_entity_to_expanded = {
        eid: [cid for cid in cids if cid in expanded_set]
        for eid, cids in merged_entity_to_expanded.items()
        if any(cid in expanded_set for cid in cids)
    }

    all_entities = list(merged_seed_to_entities.values())
    unique_entities = {e for entities in all_entities for e in entities}
    opik.update_current_span(
        metadata={
            "chunks_before_expansion": len(vector_chunk_ids),
            "chunks_after_expansion": len(result),
            "entities_found": len(unique_entities),
            "seed_chunk_ids": vector_chunk_ids,
            "expanded_chunk_ids": result,
        }
    )
    return result, merged_seed_to_entities, final_entity_to_expanded


# ===========================================================================
# Stage C — query-seeded graph retrieval.
#
# Independent of the vector retriever (constraint #4): the only external call is
# entity_index.search_entities_scored, which queries the *entity* collection and
# reuses the shared embedding model — it never touches the chunk vectorstore.
# ===========================================================================

@dataclass
class GraphResult:
    """Return type of Stage C `retrieve()`.

    chunks: (chunk_id, score) sorted descending; rank = list position.
    documents: top-N recommended documents by MaxP score (coarse output). Each is
        a dict {topic, document, score, n_chunks, best_chunk}, ranked descending.
    list_weight: single RRF weight for the whole list (Stage D). 0.0 when abstained.
    """

    chunks: list[tuple[str, float]]
    list_weight: float
    abstained: bool
    documents: list[dict] = field(default_factory=list)   # top-N MaxP documents
    matched_entities: list = field(default_factory=list)  # logging only
    concentration: float = 0.0                            # logging only
    fallback_mode: bool = False                           # logging only
    trace: dict | None = None                             # populated only when debug=True
    pi_nodes: list = field(default_factory=list)          # top-π nodes for the retrieval-graph view
                                                          # (all node types; only when topk_pi_nodes>0)


def _signal_score(entity: dict[str, Any]) -> float:
    """Confidence score for an entity under GRAPH_CONFIDENCE_SIGNAL.

    "fused" = hybrid RRF score; "dense" = dense-only cosine; "max" = the larger of
    the two ("relevant by either route" — dense rescues concept queries, fused
    rescues exact codes). Drives the floor, the walk weight, the tier, and the
    list weight, so they all agree on one notion of confidence.
    """
    fused = entity.get("fused_score")
    dense = entity.get("dense_score")
    fused = float(fused) if fused is not None else 0.0
    dense = float(dense) if dense is not None else 0.0
    signal = data_config.GRAPH_CONFIDENCE_SIGNAL
    if signal == "dense":
        return dense
    if signal == "max":
        return max(fused, dense)
    return fused


def _module_key_from_document(document_id: str | None) -> str:
    """Best-effort module key from document id (e.g. O2C-02.010 -> O2C)."""
    if not document_id:
        return "unknown"
    head = str(document_id).strip().split()[0]
    if "-" not in head:
        return "unknown"
    module = head.split("-", 1)[0].upper()
    return module if _MODULE_CODE_RE.match(module) else "unknown"


def _query_terms(text: str) -> set[str]:
    terms = {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) >= 3 and token not in _QUERY_STOP_TOKENS
    }
    return terms


def _document_title_terms(document_id: str | None) -> set[str]:
    if not document_id:
        return set()
    document = str(document_id)
    title = document.split(" ", 1)[1] if " " in document else document
    return {
        token
        for token in re.findall(r"[a-z0-9]+", title.lower())
        if len(token) >= 3 and token not in _QUERY_STOP_TOKENS
    }


def _document_query_overlap(query_terms: set[str], document_id: str | None) -> float:
    """Fraction of query terms covered by document-title terms."""
    if not query_terms:
        return 0.0
    title_terms = _document_title_terms(document_id)
    if not title_terms:
        return 0.0
    overlap = len(query_terms.intersection(title_terms))
    return overlap / float(len(query_terms))


def _compute_module_debug(
    kept_entities: list[dict[str, Any]],
    graph_root: Path,
) -> dict[str, Any]:
    """Compute hierarchy-style module scores for debug/observability only.

    This does not influence retrieval yet; it only helps inspect whether the
    query has a clear module winner for a future hierarchical switch.
    """
    module_scores: dict[str, float] = defaultdict(float)
    module_docs: dict[str, set[str]] = defaultdict(set)
    module_hits: dict[str, int] = defaultdict(int)

    for entity in kept_entities:
        topic_key = entity.get("topic_key")
        weight = _signal_score(entity) * float(entity.get("idf") or 0.0)
        if weight <= 0.0:
            continue

        index = _get_index(graph_root, topic_key)
        for chunk_id in set(index.chunks_mentioning(str(entity.get("entity_id")))):
            document_id = index.doc_of(chunk_id)
            module = _module_key_from_document(document_id)
            module_scores[module] += weight
            module_hits[module] += 1
            if document_id:
                module_docs[module].add(str(document_id))

    ordered = sorted(module_scores.items(), key=lambda item: -item[1])
    rows = [
        {
            "module": module,
            "score": score,
            "docs": len(module_docs.get(module, set())),
            "hits": module_hits.get(module, 0),
        }
        for module, score in ordered
    ]

    top1 = rows[0]["score"] if rows else 0.0
    top2 = rows[1]["score"] if len(rows) > 1 else 0.0
    ratio = (top1 / top2) if top2 > 0 else None
    hierarchical_would_switch = (
        rows
        and top1 >= data_config.GRAPH_HIERARCHICAL_MODULE_SCORE_MIN
        and (ratio is None or ratio >= data_config.GRAPH_HIERARCHICAL_MODULE_RATIO_MIN)
    )
    selected_modules = [r["module"] for r in rows[: data_config.GRAPH_HIERARCHICAL_TOP_MODULES]]

    return {
        "rows": rows,
        "top1": top1,
        "top2": top2,
        "ratio": ratio,
        "selected_modules": selected_modules,
        "hierarchical_would_switch": bool(hierarchical_would_switch),
    }


def _group_chunk_scores_by_document(
    chunk_scores: dict[str, float],
    chunk_topic: dict[str, str],
    graph_root: Path,
) -> dict[tuple[str, str], float]:
    doc_scores: dict[tuple[str, str], float] = defaultdict(float)
    for chunk_id, score in chunk_scores.items():
        topic_key = chunk_topic[chunk_id]
        document_id = _get_index(graph_root, topic_key).doc_of(chunk_id)
        if not document_id:
            continue
        key = (topic_key, document_id)
        if score > doc_scores[key]:
            doc_scores[key] = score
    return doc_scores


def _expand_same_module_candidates(
    query: str,
    raw_scores: dict[str, float],
    kept_entities: list[dict[str, Any]],
    chunk_topic: dict[str, str],
    graph_root: Path,
    module_debug: dict[str, Any],
    trace: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Hierarchical same-module local expansion.

    For each selected module:
    - pick top seed docs from current raw chunk evidence
    - iterate all chunks in those seed docs
    - expand locally inside the same module via:
      - shared matched entities
      - explicit chunk->chunk related edges
    - produce a module-local candidate pool
    Then merge module-local candidates into one graph list.
    """
    selected_modules = [str(m).upper() for m in module_debug.get("selected_modules", []) if m]
    if not selected_modules:
        return {}, {}

    query_terms = _query_terms(query)
    doc_query_boost = max(0.0, float(getattr(data_config, "GRAPH_HIERARCHICAL_DOC_QUERY_BOOST", 0.0)))

    entity_weights = {
        str(entity.get("entity_id")): _signal_score(entity) * float(entity.get("idf") or 0.0)
        for entity in kept_entities
        if _signal_score(entity) * float(entity.get("idf") or 0.0) > 0.0
    }
    doc_scores = _group_chunk_scores_by_document(raw_scores, chunk_topic, graph_root)

    merged_scores: dict[str, float] = defaultdict(float)
    merged_topics: dict[str, str] = {}
    module_trace_rows: list[dict[str, Any]] = []
    doc_overlap_cache: dict[tuple[str, str], float] = {}

    def _doc_overlap(topic_key: str, document_id: str | None) -> float:
        if not document_id:
            return 0.0
        key = (topic_key, str(document_id))
        if key not in doc_overlap_cache:
            doc_overlap_cache[key] = _document_query_overlap(query_terms, str(document_id))
        return doc_overlap_cache[key]

    for module in selected_modules:
        seed_doc_rows = []
        for (topic_key, document_id), score in sorted(doc_scores.items(), key=lambda item: -item[1]):
            if _module_key_from_document(document_id) != module:
                continue
            lexical_overlap = _doc_overlap(topic_key, document_id)
            rank_score = score * (1.0 + doc_query_boost * lexical_overlap)
            seed_doc_rows.append(
                {
                    "topic": topic_key,
                    "document": document_id,
                    "score": score,
                    "lexical_overlap": lexical_overlap,
                    "rank_score": rank_score,
                }
            )
        seed_docs = sorted(seed_doc_rows, key=lambda row: -float(row["rank_score"]))[
            : data_config.GRAPH_HIERARCHICAL_TOP_DOCS
        ]
        if not seed_docs:
            module_trace_rows.append(
                {
                    "module": module,
                    "seed_docs": [],
                    "expanded_docs": [],
                    "top_chunks": [],
                }
            )
            continue

        module_candidate_scores: dict[str, float] = defaultdict(float)
        module_candidate_topics: dict[str, str] = {}
        expanded_docs: set[str] = set()

        for seed in seed_docs:
            topic_key = str(seed["topic"])
            document_id = str(seed["document"])
            index = _get_index(graph_root, topic_key)
            seed_doc_score = float(seed["score"])
            seed_overlap = float(seed.get("lexical_overlap", 0.0))
            seed_boost = 1.0 + doc_query_boost * seed_overlap

            for seed_chunk in index.chunks_of_doc(document_id):
                if index.module_of_chunk(seed_chunk) != module:
                    continue

                # Keep existing direct evidence for matched chunks in the seed doc.
                direct = raw_scores.get(seed_chunk, 0.0)
                if direct > 0.0:
                    module_candidate_scores[seed_chunk] += direct * seed_boost
                    module_candidate_topics[seed_chunk] = topic_key
                    expanded_docs.add(document_id)

                # Expand via matched entities only, scoped to the same module.
                for entity_id in index.entities_of(seed_chunk):
                    weight = entity_weights.get(entity_id, 0.0)
                    if weight <= 0.0:
                        continue
                    for candidate in index.chunks_mentioning(entity_id):
                        if index.module_of_chunk(candidate) != module:
                            continue
                        candidate_doc = index.doc_of(candidate)
                        if candidate_doc:
                            expanded_docs.add(candidate_doc)
                        candidate_overlap = _doc_overlap(topic_key, candidate_doc)
                        candidate_boost = 1.0 + doc_query_boost * candidate_overlap
                        module_candidate_scores[candidate] += weight * candidate_boost
                        module_candidate_topics[candidate] = topic_key

                # Expand via explicit chunk->chunk links inside the same module.
                link_bonus = seed_doc_score * 0.2 if seed_doc_score > 0 else 0.0
                if link_bonus > 0.0:
                    for candidate in index.neighbor_chunks(seed_chunk):
                        if index.module_of_chunk(candidate) != module:
                            continue
                        candidate_doc = index.doc_of(candidate)
                        if candidate_doc:
                            expanded_docs.add(candidate_doc)
                        candidate_overlap = _doc_overlap(topic_key, candidate_doc)
                        candidate_boost = 1.0 + doc_query_boost * candidate_overlap
                        module_candidate_scores[candidate] += link_bonus * candidate_boost
                        module_candidate_topics[candidate] = topic_key

        # Normalize within module so one large module does not dominate purely by size.
        if module_candidate_scores:
            max_score = max(module_candidate_scores.values())
            if max_score > 0:
                for chunk_id in list(module_candidate_scores.keys()):
                    normalized = module_candidate_scores[chunk_id] / max_score
                    merged_scores[chunk_id] += normalized
                    merged_topics[chunk_id] = module_candidate_topics[chunk_id]

        module_trace_rows.append(
            {
                "module": module,
                "seed_docs": seed_docs,
                "expanded_docs": sorted(expanded_docs)[: data_config.GRAPH_TRACE_TOPN],
                "top_chunks": [
                    {
                        "chunk_id": chunk_id,
                        "score": score,
                        "document": _get_index(graph_root, module_candidate_topics[chunk_id]).doc_of(chunk_id),
                    }
                    for chunk_id, score in sorted(module_candidate_scores.items(), key=lambda item: -item[1])
                    [: data_config.GRAPH_TRACE_TOPN]
                ],
            }
        )

    if trace is not None:
        trace["hierarchical_modules"] = module_trace_rows
        trace["selected_topics"] = sorted({row["topic"] for mod in module_trace_rows for row in mod["seed_docs"]})

    return dict(merged_scores), merged_topics


def _match_seed_entities(
    query: str,
    trace: dict[str, Any] | None = None,
    entity_collection: str | None = None,
) -> tuple[list[dict[str, Any]], float, float] | None:
    """Part A of retrieve(): match query -> seed entities, guard, gate, concentration.

    Returns (kept_entities, best_confidence, concentration), or None to abstain.
    Ordering/entity-weight uses the fused score (Step 5); the confidence signal
    (fused|dense) drives only the tier floors and list weight (C5). When `trace`
    is given, per-step debug data is written into it (STEP 1 + STEP 2).
    """
    if data_config.GRAPH_QUERY_PHRASE_MODE != "raw":
        logger.warning(
            "GRAPH_QUERY_PHRASE_MODE=%r not implemented; using raw query.",
            data_config.GRAPH_QUERY_PHRASE_MODE,
        )

    from dc_available.graph.construct.entity_index import search_entities_scored

    entities = search_entities_scored(query, k=data_config.GRAPH_ENTITY_SEARCH_K, collection_name=entity_collection)
    if trace is not None:
        trace["total_entities"] = len(entities)
    if not entities:
        logger.info("graph.retrieve abstain (no entity matches) query=%r", query)
        if trace is not None:
            trace["abstain_reason"] = "no entity matches"
        return None

    # C7: idf must exist for every entity. A silent 0.0 hides a Stage A bug.
    missing = [e for e in entities if e.get("idf") is None]
    if trace is not None:
        trace["idf_missing"] = len(missing)
    if missing:
        ratio = len(missing) / len(entities)
        logger.warning(
            "graph.retrieve: %d/%d matched entities missing idf (%.1f%%); treating as 0.0",
            len(missing), len(entities), 100 * ratio,
        )
        if ratio > data_config.GRAPH_IDF_MISSING_MAX_RATIO:
            raise ValueError(
                f"{len(missing)}/{len(entities)} matched entities lack 'idf' "
                f"({ratio:.1%} > {data_config.GRAPH_IDF_MISSING_MAX_RATIO:.1%}); "
                "Stage A (semantic_merger IDF) likely did not run for this corpus."
            )

    signal = data_config.GRAPH_CONFIDENCE_SIGNAL

    kept = [e for e in entities if _signal_score(e) >= data_config.GRAPH_MIN_FLOOR]
    kept = kept[: data_config.GRAPH_MAX_SEED_ENTITIES]  # entities already ordered by fused
    if trace is not None:
        trace["signal"] = signal
        trace["kept"] = len(kept)
    if not kept:
        best_fetched = max((_signal_score(e) for e in entities), default=0.0)
        logger.info(
            "graph.retrieve abstain (best confidence < %.2f) query=%r",
            data_config.GRAPH_MIN_FLOOR, query,
        )
        if trace is not None:
            trace["abstain_reason"] = (
                f"best {best_fetched:.3f} < min {data_config.GRAPH_MIN_FLOOR}"
            )
        return None

    best = max(_signal_score(e) for e in kept)

    # Topic concentration — LOG ONLY (D2). Not a gate while the flag is off.
    mass: dict[str, float] = defaultdict(float)
    for e in kept:
        mass[e.get("topic_key")] += _signal_score(e) * float(e.get("idf") or 0.0)
    total_mass = sum(mass.values())
    concentration = (max(mass.values()) / total_mass) if total_mass > 0 else 0.0
    logger.info(
        "graph.retrieve concentration=%.3f would_abstain=%s (gate_enabled=%s) query=%r",
        concentration,
        concentration < data_config.GRAPH_TOPIC_CONCENTRATION_MIN,
        data_config.GRAPH_TOPIC_GATE_ENABLED,
        query,
    )

    if trace is not None:
        top_n = data_config.GRAPH_TRACE_TOPN
        trace["best"] = best
        trace["concentration"] = concentration
        trace["would_abstain"] = concentration < data_config.GRAPH_TOPIC_CONCENTRATION_MIN
        trace["concentration_by_topic"] = sorted(
            ((t, (m / total_mass if total_mass > 0 else 0.0)) for t, m in mass.items()),
            key=lambda item: -item[1],
        )
        trace["seed_rows"] = [
            {
                "fused": e.get("fused_score"),
                "dense": e.get("dense_score"),
                "idf": e.get("idf"),
                "weight": _signal_score(e) * float(e.get("idf") or 0.0),
                "topic": e.get("topic_key"),
                "entity": f"{e.get('label')} [{e.get('entity_type')}]",
                "cf": e.get("chunk_frequency"),
            }
            for e in kept[:top_n]
        ]

    return kept, best, concentration


def retrieve(query: str, graph_root: Path, debug: bool = False) -> GraphResult:
    """Query-seeded graph retrieval (Stage C). Independent of the vector retriever.

    When debug=True, per-step intermediate values are collected into
    GraphResult.trace. Off by default -> no overhead
    and no effect on the serving path.
    """
    graph_root = Path(graph_root)
    trace: dict[str, Any] | None = {"query": query} if debug else None

    # Step 1: seed selection. "legacy" = the top-8-by-fused gate; "merge" reuses
    # the PPR v2 seed pipeline (threshold + boilerplate filter + concept merge +
    # topic isolation) so entity_1hop can be compared to ppr_merge on IDENTICAL
    # seeds (fair best-vs-best). See data_config.GRAPH_ENTITY_HOP_SEED_MODE.
    seed_mode = str(getattr(data_config, "GRAPH_ENTITY_HOP_SEED_MODE", "legacy")).lower()
    seed_weights: dict[str, float] | None = None
    if seed_mode == "merge":
        from dc_available.graph.retrieval.ppr_retriever import _select_seeds

        groups, kept, best, concentration = _select_seeds(query, debug=debug)
        if not kept:
            if trace is not None:
                trace["retrieval_path"] = "current"
            return GraphResult(chunks=[], list_weight=0.0, abstained=True, trace=trace)
        # poured per-entity weights (group weight split across members) — the same
        # numbers ppr_merge seeds with, so only the ENGINE differs.
        seed_weights = {}
        for g in groups:
            n = max(1, len(g.members))
            for m in g.members:
                eid = str(m.get("entity_id"))
                seed_weights[eid] = seed_weights.get(eid, 0.0) + g.weight / n
    else:
        matched = _match_seed_entities(query, trace=trace)
        if matched is None:
            if trace is not None:
                trace["retrieval_path"] = "current"
            return GraphResult(chunks=[], list_weight=0.0, abstained=True, trace=trace)
        kept, best, concentration = matched

    # Topic isolation — applied to BOTH seed modes (engine-level parity with PPR,
    # which isolates in _candidate_topics). Honors data_config.PPR_TOPIC_ISOLATION.
    from dc_available.graph.retrieval.ppr_retriever import _candidate_topics

    topic_set = {t.lower() for t in _candidate_topics(graph_root, kept, debug=debug)}
    kept = [e for e in kept if str(e.get("topic_key") or "").lower() in topic_set]
    if not kept:
        if trace is not None:
            trace["retrieval_path"] = "current"
        return GraphResult(chunks=[], list_weight=0.0, abstained=True, trace=trace)

    if trace is not None:
        module_debug = _compute_module_debug(kept, graph_root)
        trace["retrieval_path"] = "current"
        trace["module_rows"] = module_debug["rows"][: data_config.GRAPH_TRACE_TOPN]
        trace["module_top1"] = module_debug["top1"]
        trace["module_top2"] = module_debug["top2"]
        trace["module_ratio"] = module_debug["ratio"]
        trace["selected_modules"] = module_debug["selected_modules"]
        trace["hierarchical_would_switch"] = module_debug["hierarchical_would_switch"]
    else:
        module_debug = _compute_module_debug(kept, graph_root)

    # 1. Walk one hop: entity -> chunks (routed per topic). Weight uses the fused
    #    score (ordering signal, C5); confidence signal only drove the tier.
    raw: dict[str, float] = defaultdict(float)
    chunk_topic: dict[str, str] = {}
    for entity in kept:
        topic_key = entity.get("topic_key")
        if seed_weights is not None:
            weight = float(seed_weights.get(str(entity.get("entity_id")), 0.0))
        else:
            weight = _signal_score(entity) * float(entity.get("idf") or 0.0)
        if weight <= 0.0:
            continue
        index = _get_index(graph_root, topic_key)
        for chunk_id in set(index.chunks_mentioning(str(entity.get("entity_id")))):
            raw[chunk_id] += weight
            chunk_topic[chunk_id] = topic_key

    if not raw:
        logger.info("graph.retrieve: seeds matched but no chunks reached query=%r", query)
        if trace is not None:
            trace["total_chunks"] = 0
        return GraphResult(
            chunks=[], list_weight=0.0, abstained=False,
            matched_entities=kept, concentration=concentration, fallback_mode=False, trace=trace,
        )

    # 1b. Extra hops (k>1): walk chunk -> its entities -> more chunks, decayed.
    #     k=1 skips this block entirely, so `raw` stays exactly the single-hop
    #     result above. Each new chunk is scored once (dedup via seen_chunks), an
    #     entity is expanded once across hops (seen_entities), generic hub
    #     entities are skipped, and the walk stops as soon as a hop adds nothing
    #     new. We look up neighbours in each chunk's OWN topic index, so topic
    #     isolation carries over to every hop.
    hop_k = int(getattr(data_config, "GRAPH_ENTITY_HOP_K", 1) or 1)
    if hop_k > 1:
        decay = float(getattr(data_config, "GRAPH_ENTITY_HOP_DECAY", 0.5))
        hub_cap = int(getattr(data_config, "GRAPH_ENTITY_HOP_HUB_CAP", 0) or 0)
        frontier = list(raw.keys())                    # chunks reached at hop 1
        seen_chunks: set[str] = set(frontier)
        seen_entities: set[tuple[str, str]] = set()
        for _hop in range(2, hop_k + 1):
            gain: dict[str, float] = {}                # new chunk -> best decayed weight
            gain_topic: dict[str, str] = {}
            hop_entities: set[tuple[str, str]] = set()
            for chunk_id in frontier:
                topic_key = chunk_topic[chunk_id]
                src = raw[chunk_id] * decay
                if src <= 0.0:
                    continue
                index = _get_index(graph_root, topic_key)
                for entity_id in index.entities_of(chunk_id):
                    ekey = (topic_key, str(entity_id))
                    if ekey in seen_entities:
                        continue
                    hop_entities.add(ekey)
                    reached = set(index.chunks_mentioning(str(entity_id)))
                    if hub_cap and len(reached) > hub_cap:
                        continue                       # skip generic hub entities
                    for new_id in reached:
                        if new_id in seen_chunks:
                            continue                   # dedup: score each chunk once
                        if src > gain.get(new_id, 0.0):
                            gain[new_id] = src
                            gain_topic[new_id] = topic_key
            seen_entities |= hop_entities
            if not gain:
                break                                  # stop when no new chunks
            for new_id, weight in gain.items():
                raw[new_id] += weight
                chunk_topic[new_id] = gain_topic[new_id]
                seen_chunks.add(new_id)
            frontier = list(gain.keys())

    # Optional hierarchical retrieval: same-module seed-doc local expansion.
    hierarchy_enabled = bool(getattr(data_config, "GRAPH_HIERARCHICAL_ENABLED", False))
    use_hierarchical = hierarchy_enabled and bool(module_debug.get("hierarchical_would_switch"))
    if use_hierarchical:
        expanded_raw, expanded_topics = _expand_same_module_candidates(
            query=query,
            raw_scores=dict(raw),
            kept_entities=kept,
            chunk_topic=chunk_topic,
            graph_root=graph_root,
            module_debug=module_debug,
            trace=trace,
        )
        if expanded_raw:
            raw = defaultdict(float, expanded_raw)
            chunk_topic = expanded_topics
            if trace is not None:
                trace["retrieval_path"] = "hierarchical"
                trace["hierarchical_chunks_after_local_expansion"] = len(raw)
        else:
            if trace is not None:
                trace["retrieval_path"] = "current"
                trace["hierarchical_fallback_reason"] = "empty_after_local_expansion"
    else:
        if trace is not None:
            trace["retrieval_path"] = "current"
            trace["hierarchical_fallback_reason"] = (
                "disabled" if not hierarchy_enabled else "low_module_confidence"
            )

    # 2. Length-normalise by the TOTAL entity count of each chunk (not matched-only).
    scores: dict[str, float] = {}
    for chunk_id, summed in raw.items():
        total_entities = _get_index(graph_root, chunk_topic[chunk_id]).entity_count(chunk_id)
        scores[chunk_id] = summed / sqrt(total_entities) if total_entities > 0 else summed
    scores_before = dict(scores) if trace is not None else None

    # 3. MaxP document prior — max, never sum (size-robust); keyed (topic, document).
    #    doc_best_chunk / doc_members also make the document ranking a first-class
    #    output (top-N "recommended documents"), not just an internal prior.
    doc_key: dict[str, tuple[str, str | None]] = {}
    doc_best: dict[tuple[str, str | None], float] = defaultdict(float)
    doc_best_chunk: dict[tuple[str, str | None], str] = {}
    doc_members: dict[tuple[str, str | None], list[str]] = defaultdict(list)
    for chunk_id, score in scores.items():
        topic_key = chunk_topic[chunk_id]
        key = (topic_key, _get_index(graph_root, topic_key).doc_of(chunk_id))
        doc_key[chunk_id] = key
        doc_members[key].append(chunk_id)
        if score > doc_best[key]:
            doc_best[key] = score
            doc_best_chunk[key] = chunk_id
    top_doc = max(doc_best.values())
    beta = data_config.GRAPH_DOC_PRIOR_BETA
    if top_doc > 0 and beta:
        prior = {key: value / top_doc for key, value in doc_best.items()}
        scores = {cid: s * (1 + beta * prior[doc_key[cid]]) for cid, s in scores.items()}

    # Top-N recommended documents by MaxP (pre-prior score = document relevance).
    documents = [
        {
            "topic": topic_key,
            "document": document,
            "score": maxp,
            "n_chunks": len(doc_members[(topic_key, document)]),
            "best_chunk": doc_best_chunk.get((topic_key, document)),
        }
        for (topic_key, document), maxp in sorted(
            doc_best.items(), key=lambda item: -item[1]
        )[: data_config.GRAPH_DOC_TOPN]
    ]

    # 4. Cut: relative floor, then top-k (normal vs. fallback mode).
    ranked = sorted(scores.items(), key=lambda item: -item[1])
    floor = data_config.GRAPH_SCORE_FLOOR_RATIO * ranked[0][1]
    survivors = [(cid, s) for cid, s in ranked if s >= floor]

    # Document-level diversity cap: prevent one very broad document from taking
    # almost all final chunk slots.
    doc_cap = max(1, int(getattr(data_config, "GRAPH_MAX_CHUNKS_PER_DOC", 1)))
    capped: list[tuple[str, float]] = []
    doc_counts: dict[tuple[str, str | None], int] = defaultdict(int)
    skipped_by_cap = 0
    for chunk_id, score in survivors:
        topic_key = chunk_topic[chunk_id]
        document_id = _get_index(graph_root, topic_key).doc_of(chunk_id)
        key = (topic_key, document_id)
        if doc_counts[key] >= doc_cap:
            skipped_by_cap += 1
            continue
        doc_counts[key] += 1
        capped.append((chunk_id, score))

    fallback_mode = best < data_config.GRAPH_ENTITY_MATCH_FLOOR
    k = data_config.GRAPH_FALLBACK_K if fallback_mode else data_config.GRAPH_CANDIDATE_K
    ranked = capped[:k]

    # 5. One confidence weight for the whole list (Stage D RRF weight).
    list_weight = max(best, data_config.GRAPH_LIST_WEIGHT_MIN)

    if trace is not None:
        top_n = data_config.GRAPH_TRACE_TOPN

        def _chunk_rows(score_map: dict[str, float]) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for cid, score in sorted(score_map.items(), key=lambda item: -item[1])[:top_n]:
                index = _get_index(graph_root, chunk_topic[cid])
                rows.append(
                    {
                        "chunk_id": cid,
                        "raw": raw[cid],
                        "n_ent": index.entity_count(cid),
                        "score": score,
                        "doc": index.doc_of(cid),
                        "topic": chunk_topic[cid],
                    }
                )
            return rows

        trace["total_chunks"] = len(raw)
        trace["chunks_before_prior"] = _chunk_rows(scores_before)
        trace["chunks_after_prior"] = _chunk_rows(scores)
        trace["documents"] = documents
        trace["floor"] = floor
        trace["survive"] = len(survivors)
        trace["survive_after_doc_cap"] = len(capped)
        trace["graph_max_chunks_per_doc"] = doc_cap
        trace["skipped_by_doc_cap"] = skipped_by_cap
        trace["k"] = k
        trace["fallback_mode"] = fallback_mode
        trace["list_weight"] = list_weight
        trace["final"] = [
            {
                "rank": i + 1,
                "topic": chunk_topic[cid],
                "document": _get_index(graph_root, chunk_topic[cid]).doc_of(cid),
                "chunk_id": cid,
                "score": score,
            }
            for i, (cid, score) in enumerate(ranked[:top_n])
        ]

    logger.info(
        "graph.retrieve query=%r seeds=%d chunks=%d fallback=%s list_weight=%.3f conc=%.3f",
        query, len(kept), len(ranked), fallback_mode, list_weight, concentration,
    )
    return GraphResult(
        chunks=ranked,
        list_weight=list_weight,
        abstained=False,
        documents=documents,
        matched_entities=kept,
        concentration=concentration,
        fallback_mode=fallback_mode,
        trace=trace,
    )
