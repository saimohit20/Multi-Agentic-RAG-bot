from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dc_available.RAG.data import GraphEdgeType, GraphNodeType


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


def _normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _load_manifest(file_path: Path) -> dict[str, Any]:
    if not file_path.is_file():
        return {}
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _update_manifest(file_path: Path, updates: dict[str, Any]) -> None:
    manifest = _load_manifest(file_path)
    manifest.update(updates)
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def _select_entity_id(
    normalized_label: str,
    label_to_entities: dict[str, list[dict[str, Any]]],
) -> str | None:
    candidates = label_to_entities.get(normalized_label, [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return str(candidates[0]["entity_id"])
    ordered = sorted(
        candidates,
        key=lambda item: (-int(item.get("occurrence_count", 0)), str(item.get("label", "")), str(item.get("entity_type", ""))),
    )
    return str(ordered[0]["entity_id"])


def _merge_ids(existing: list[str], new_values: list[str]) -> list[str]:
    merged: dict[str, None] = {str(v): None for v in existing if str(v).strip()}
    for value in new_values:
        text = str(value).strip()
        if text:
            merged[text] = None
    return list(merged.keys())


def _merge_mentions(
    entities: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    mention_map: dict[tuple[str, str], dict[str, Any]] = {}
    mentions_written = 0

    for entity in entities:
        entity_id = str(entity["entity_id"])
        entity_label = str(entity.get("label", ""))
        evidence_chunk_ids = [str(v) for v in entity.get("evidence_chunk_ids", []) if str(v).strip()]
        source_batch_ids = [str(v) for v in entity.get("source_batch_ids", []) if str(v).strip()]

        for chunk_id in evidence_chunk_ids:
            key = (chunk_id, entity_id)
            existing = mention_map.get(key)
            if existing is None:
                mention_map[key] = {
                    "source": chunk_id,
                    "target": entity_id,
                    "type": GraphEdgeType.MENTIONS.value,
                    "properties": {
                        "origin": "semantic_entity_evidence",
                        "entity_label": entity_label,
                        "entity_type": entity.get("entity_type"),
                        "evidence_chunk_ids": [chunk_id],
                        "source_batch_ids": list(source_batch_ids),
                        "confidence": entity.get("confidence"),
                    },
                }
                mentions_written += 1
                continue

            props = existing["properties"]
            props["evidence_chunk_ids"] = _merge_ids(props.get("evidence_chunk_ids", []), [chunk_id])
            props["source_batch_ids"] = _merge_ids(props.get("source_batch_ids", []), source_batch_ids)
            existing["properties"] = props

    return list(mention_map.values()), mentions_written


def _merge_relations(
    relations: list[dict[str, Any]],
    label_to_entities: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], int]:
    relation_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    skipped = 0

    for relation in relations:
        source_label = str(relation.get("source", "")).strip()
        target_label = str(relation.get("target", "")).strip()
        relation_type = str(relation.get("relation_type", "related_to")).strip() or "related_to"
        if not source_label or not target_label:
            continue

        source_id = _select_entity_id(_normalize_label(source_label), label_to_entities)
        target_id = _select_entity_id(_normalize_label(target_label), label_to_entities)
        if not source_id or not target_id:
            skipped += 1
            continue

        key = (source_id, relation_type, target_id)
        existing = relation_map.get(key)
        evidence_chunk_ids = [str(v) for v in relation.get("evidence_chunk_ids", []) if str(v).strip()]
        source_batch_ids = [str(v) for v in relation.get("source_batch_ids", []) if str(v).strip()]
        confidence = relation.get("confidence")

        if existing is None:
            relation_map[key] = {
                "source": source_id,
                "target": target_id,
                "type": relation_type,
                "properties": {
                    "origin": "semantic_relation",
                    "source_label": source_label,
                    "target_label": target_label,
                    "source_normalized": _normalize_label(source_label),
                    "target_normalized": _normalize_label(target_label),
                    "confidence": confidence,
                    "evidence_chunk_ids": list(evidence_chunk_ids),
                    "source_batch_ids": list(source_batch_ids),
                    "relation_type": relation_type,
                },
            }
            continue

        props = existing["properties"]
        props["evidence_chunk_ids"] = _merge_ids(props.get("evidence_chunk_ids", []), evidence_chunk_ids)
        props["source_batch_ids"] = _merge_ids(props.get("source_batch_ids", []), source_batch_ids)
        existing["properties"] = props

    return list(relation_map.values()), skipped


def integrate_semantic_graph(
    graph_root: str | Path,
    topic_keys: list[str] | None = None,
) -> dict[str, Any]:
    graph_root = Path(graph_root)
    topic_dirs = sorted(path for path in graph_root.iterdir() if path.is_dir())

    allowed: set[str] | None = None
    if topic_keys:
        allowed = {t.strip().lower() for t in topic_keys if t.strip()}

    totals: dict[str, Any] = {
        "topics": 0,
        "semantic_entities": 0,
        "semantic_relations": 0,
        "mentions_edges": 0,
        "semantic_edges": 0,
        "skipped_relations": 0,
    }

    for topic_dir in topic_dirs:
        if allowed is not None and topic_dir.name.lower() not in allowed:
            continue

        entities_path = topic_dir / "semantic_entities_merged.jsonl"
        relations_path = topic_dir / "semantic_relations_merged.jsonl"
        if not entities_path.is_file() or not relations_path.is_file():
            continue

        merged_entities = _read_jsonl(entities_path)
        merged_relations = _read_jsonl(relations_path)
        if not merged_entities and not merged_relations:
            continue

        nodes_path = topic_dir / "nodes.jsonl"
        edges_path = topic_dir / "edges.jsonl"

        existing_nodes = _read_jsonl(nodes_path) if nodes_path.is_file() else []
        existing_edges = _read_jsonl(edges_path) if edges_path.is_file() else []

        node_map: dict[str, dict[str, Any]] = {str(node["id"]): node for node in existing_nodes if node.get("id")}
        edge_set: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for edge in existing_edges:
            key = (
                str(edge.get("source", "")),
                str(edge.get("target", "")),
                str(edge.get("type", "")),
                json.dumps(edge.get("properties", {}), sort_keys=True),
            )
            edge_set[key] = edge

        label_to_entities: dict[str, list[dict[str, Any]]] = defaultdict(list)
        semantic_node_count = 0
        for entity in merged_entities:
            entity_id = str(entity.get("entity_id", "")).strip()
            label = str(entity.get("label", "")).strip()
            normalized_label = str(entity.get("normalized_label", _normalize_label(label))).strip()
            entity_type = str(entity.get("entity_type", "unknown")).strip().lower() or "unknown"
            if not entity_id or not label:
                continue

            node_map[entity_id] = {
                "id": entity_id,
                "type": GraphNodeType.ENTITY.value,
                "properties": {
                    "label": label,
                    "normalized_label": normalized_label,
                    "entity_type": entity_type,
                    "confidence": entity.get("confidence"),
                    "occurrence_count": entity.get("occurrence_count", 0),
                    "evidence_chunk_ids": entity.get("evidence_chunk_ids", []),
                    "source_batch_ids": entity.get("source_batch_ids", []),
                },
            }
            label_to_entities[normalized_label].append(entity)
            semantic_node_count += 1

        mention_edges, mention_count = _merge_mentions(merged_entities)
        semantic_relation_edges, skipped_relations = _merge_relations(merged_relations, label_to_entities)

        for edge in mention_edges + semantic_relation_edges:
            key = (
                str(edge.get("source", "")),
                str(edge.get("target", "")),
                str(edge.get("type", "")),
                json.dumps(edge.get("properties", {}), sort_keys=True),
            )
            edge_set[key] = edge

        updated_nodes = [node_map[node_id] for node_id in sorted(node_map.keys())]
        updated_edges = [edge_set[key] for key in sorted(edge_set.keys())]

        _write_jsonl(nodes_path, updated_nodes)
        _write_jsonl(edges_path, updated_edges)

        _update_manifest(
            topic_dir / "build_manifest.json",
            {
                "node_count": len(updated_nodes),
                "edge_count": len(updated_edges),
                "semantic_entity_count": len(merged_entities),
                "semantic_relation_count": len(merged_relations),
                "semantic_node_count": semantic_node_count,
                "semantic_mentions_edge_count": mention_count,
                "semantic_edge_count": len(semantic_relation_edges),
            },
        )

        integration_manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "topic_key": topic_dir.name,
            "semantic_entity_count": len(merged_entities),
            "semantic_relation_count": len(merged_relations),
            "semantic_node_count": semantic_node_count,
            "semantic_mentions_edge_count": mention_count,
            "semantic_edge_count": len(semantic_relation_edges),
            "skipped_relations": skipped_relations,
            "node_count": len(updated_nodes),
            "edge_count": len(updated_edges),
        }
        with (topic_dir / "semantic_integration_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(integration_manifest, handle, indent=2, sort_keys=True)

        totals["topics"] += 1
        totals["semantic_entities"] += len(merged_entities)
        totals["semantic_relations"] += len(merged_relations)
        totals["mentions_edges"] += mention_count
        totals["semantic_edges"] += len(semantic_relation_edges)
        totals["skipped_relations"] += skipped_relations

    return totals
