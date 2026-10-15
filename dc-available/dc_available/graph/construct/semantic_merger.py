from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import log
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_jsonl(file_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(file_path: Path, rows: list[dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _count_total_chunks(graph_root: Path) -> int:
    """Count distinct Chunk-type nodes across ALL topics (global corpus total).

    Used as the IDF numerator. Counted over every topic in ``graph_root`` even
    when only a subset is being merged, so the denominator stays global and IDF
    values remain comparable across topics.
    """
    chunk_ids: set[str] = set()
    for topic_dir in sorted(path for path in graph_root.iterdir() if path.is_dir()):
        nodes_path = topic_dir / "nodes.jsonl"
        if not nodes_path.is_file():
            continue
        for node in _read_jsonl(nodes_path):
            if node.get("type") == "Chunk":
                chunk_ids.add(str(node.get("id")))
    return len(chunk_ids)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _entity_id(normalized_label: str, entity_type: str) -> str:
    slug_label = re.sub(r"[^a-z0-9]+", "_", normalized_label).strip("_")
    slug_type = re.sub(r"[^a-z0-9]+", "_", entity_type.lower()).strip("_")
    return f"entity::{slug_type}::{slug_label}"


def _relation_key(source: str, relation_type: str, target: str) -> tuple[str, str, str]:
    return (_normalize_label(source), relation_type.lower().strip(), _normalize_label(target))


# ---------------------------------------------------------------------------
# Entity merge
# ---------------------------------------------------------------------------

def _merge_entities(extraction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group raw entities by (normalized_label, entity_type), then aggregate."""

    # key -> list of raw entity dicts annotated with batch_id
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in extraction_rows:
        if row.get("error"):
            continue
        batch_id = row.get("batch_id", "")
        for ent in row.get("entities", []):
            label = str(ent.get("label", "")).strip()
            if not label:
                continue
            entity_type = _normalize_label(str(ent.get("entity_type", "unknown")) or "unknown")
            normalized = _normalize_label(label)
            groups[(normalized, entity_type)].append({**ent, "_batch_id": batch_id})

    merged: list[dict[str, Any]] = []
    for (normalized_label, entity_type), occurrences in sorted(groups.items()):
        # canonical label: most frequent original label in this group
        label_counts = Counter(str(o.get("label", "")).strip() for o in occurrences)
        canonical_label = label_counts.most_common(1)[0][0]

        # aggregate evidence chunk ids (unique, ordered)
        evidence_set: dict[str, None] = {}
        for o in occurrences:
            for cid in o.get("evidence_chunk_ids", []):
                evidence_set[str(cid)] = None

        # aggregate source batch ids (unique, ordered)
        batch_set: dict[str, None] = {}
        for o in occurrences:
            bid = o.get("_batch_id", "")
            if bid:
                batch_set[bid] = None

        # average confidence
        confidences = [float(o.get("confidence", 0.5)) for o in occurrences]
        avg_confidence = round(sum(confidences) / len(confidences), 4)

        merged.append(
            {
                "entity_id": _entity_id(normalized_label, entity_type),
                "label": canonical_label,
                "normalized_label": normalized_label,
                "entity_type": entity_type,  # already normalized to lowercase
                "confidence": avg_confidence,
                "occurrence_count": len(occurrences),
                "chunk_frequency": len(evidence_set),  # distinct chunks -> IDF denom
                "evidence_chunk_ids": list(evidence_set),
                "source_batch_ids": list(batch_set),
            }
        )

    # sort by occurrence_count descending so most prominent entities appear first
    merged.sort(key=lambda x: (-x["occurrence_count"], x["normalized_label"]))
    return merged


# ---------------------------------------------------------------------------
# Relation merge
# ---------------------------------------------------------------------------

def _merge_relations(extraction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group raw relations by (source_normalized, relation_type, target_normalized), aggregate."""

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in extraction_rows:
        if row.get("error"):
            continue
        batch_id = row.get("batch_id", "")
        for rel in row.get("relations", []):
            source = str(rel.get("source", "")).strip()
            target = str(rel.get("target", "")).strip()
            relation_type = str(rel.get("relation_type", "related_to")).strip() or "related_to"
            if not source or not target:
                continue
            key = _relation_key(source, relation_type, target)
            groups[key].append({**rel, "_batch_id": batch_id})
    merged: list[dict[str, Any]] = []
    for (src_norm, rel_type, tgt_norm), occurrences in sorted(groups.items()):
        # canonical labels: most frequent
        src_counts = Counter(str(o.get("source", "")).strip() for o in occurrences)
        tgt_counts = Counter(str(o.get("target", "")).strip() for o in occurrences)
        canonical_source = src_counts.most_common(1)[0][0]
        canonical_target = tgt_counts.most_common(1)[0][0]

        evidence_set: dict[str, None] = {}
        for o in occurrences:
            for cid in o.get("evidence_chunk_ids", []):
                evidence_set[str(cid)] = None

        batch_set: dict[str, None] = {}
        for o in occurrences:
            bid = o.get("_batch_id", "")
            if bid:
                batch_set[bid] = None

        confidences = [float(o.get("confidence", 0.5)) for o in occurrences]
        avg_confidence = round(sum(confidences) / len(confidences), 4)

        merged.append(
            {
                "source": canonical_source,
                "source_normalized": src_norm,
                "target": canonical_target,
                "target_normalized": tgt_norm,
                "relation_type": rel_type,
                "confidence": avg_confidence,
                "occurrence_count": len(occurrences),
                "evidence_chunk_ids": list(evidence_set),
                "source_batch_ids": list(batch_set),
            }
        )

    merged.sort(key=lambda x: (-x["occurrence_count"], x["source_normalized"], x["relation_type"]))
    return merged


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def merge_semantic_candidates(
    graph_root: str | Path,
    topic_keys: list[str] | None = None,
    min_confidence: float = 0.0,
    min_occurrences: int = 1,
) -> dict[str, Any]:
    """
    Read semantic_extractions.jsonl per topic, merge entities and relations,
    write semantic_entities_merged.jsonl and semantic_relations_merged.jsonl.
    """
    graph_root = Path(graph_root)
    topic_dirs = sorted(path for path in graph_root.iterdir() if path.is_dir())

    allowed: set[str] | None = None
    if topic_keys:
        allowed = {t.strip().lower() for t in topic_keys if t.strip()}

    # Global chunk count (IDF numerator) — over ALL topics, not just `allowed`.
    total_chunks = _count_total_chunks(graph_root)

    totals: dict[str, Any] = {
        "topics": 0,
        "entities": 0,
        "relations": 0,
        "total_chunks": total_chunks,
    }

    for topic_dir in topic_dirs:
        if allowed is not None and topic_dir.name.lower() not in allowed:
            continue

        extraction_path = topic_dir / "semantic_extractions.jsonl"
        if not extraction_path.is_file():
            continue

        rows = _read_jsonl(extraction_path)
        if not rows:
            continue

        entities = _merge_entities(rows)
        relations = _merge_relations(rows)

        # apply threshold filters
        if min_confidence > 0.0:
            entities = [e for e in entities if e["confidence"] >= min_confidence]
            relations = [r for r in relations if r["confidence"] >= min_confidence]
        if min_occurrences > 1:
            entities = [e for e in entities if e["occurrence_count"] >= min_occurrences]
            relations = [r for r in relations if r["occurrence_count"] >= min_occurrences]

        # Stage A: document-frequency IDF. Rare entities (low chunk_frequency)
        # get a high weight; ubiquitous ones tend toward 0. No tunable knob.
        for entity in entities:
            chunk_frequency = entity.get("chunk_frequency", 0)
            entity["idf"] = (
                round(log(total_chunks / chunk_frequency), 6)
                if chunk_frequency > 0 and total_chunks > 0
                else 0.0
            )

        _write_jsonl(topic_dir / "semantic_entities_merged.jsonl", entities)
        _write_jsonl(topic_dir / "semantic_relations_merged.jsonl", relations)

        manifest: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "topic_key": topic_dir.name,
            "total_chunks": total_chunks,
            "entity_count": len(entities),
            "relation_count": len(relations),
            "min_confidence": min_confidence,
            "min_occurrences": min_occurrences,
            "source_batches": len(rows),
            "failed_source_batches": sum(1 for r in rows if r.get("error")),
        }
        with (topic_dir / "semantic_merge_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

        totals["topics"] += 1
        totals["entities"] += len(entities)
        totals["relations"] += len(relations)

    return totals
