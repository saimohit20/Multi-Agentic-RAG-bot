from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dc_available.config.llm_config import DEFAULT_MODEL, get_langchain_llm


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
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    return normalized


def _clamp_confidence(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, score))


def _extract_text_response(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        fragments: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                fragments.append(str(item["text"]))
            elif isinstance(item, str):
                fragments.append(item)
        return "\n".join(fragments)

    return str(content)


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if fenced_match:
        cleaned = fenced_match.group(1)
    else:
        object_match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
        if object_match:
            cleaned = object_match.group(1)

    return json.loads(cleaned)


def _batch_prompt(batch: dict[str, Any]) -> str:
    chunks = batch.get("chunks", [])
    formatted_chunks: list[str] = []
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        chunk_id = str(chunk.get("chunk_id", ""))
        truncated = text[:1200]
        formatted_chunks.append(f"- chunk_id: {chunk_id}\n  text: {truncated}")

    chunk_block = "\n".join(formatted_chunks) if formatted_chunks else "- no chunks"
    return (
        "Extract entities and relations from this semantic batch.\n"
        "Return strict JSON only, with this shape:\n"
        "{\n"
        '  "entities": [{"label": "", "entity_type": "", "confidence": 0.0, "evidence_chunk_ids": []}],\n'
        '  "relations": [{"source": "", "target": "", "relation_type": "related_to", "confidence": 0.0, "evidence_chunk_ids": []}]\n'
        "}\n"
        "Rules:\n"
        "- Use only information that is explicitly supported by the provided chunks.\n"
        "- Keep entity labels short and canonical.\n"
        "- Keep confidence between 0 and 1.\n"
        "- If unsure, return empty arrays.\n\n"
        f"topic: {batch.get('topic', '')}\n"
        f"training_id: {batch.get('training_id', '')}\n"
        f"document_id: {batch.get('document_id', '')}\n"
        f"batch_id: {batch.get('batch_id', '')}\n"
        f"chunks:\n{chunk_block}"
    )


def _sanitize_entities(raw_entities: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_entities, list):
        return []

    entities: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        entity_type = str(item.get("entity_type", "unknown")).strip() or "unknown"
        normalized_label = _normalize_label(label)
        key = (normalized_label, entity_type.lower())
        if key in seen:
            continue
        seen.add(key)
        evidence = item.get("evidence_chunk_ids", [])
        if not isinstance(evidence, list):
            evidence = []
        entities.append(
            {
                "label": label,
                "normalized_label": normalized_label,
                "entity_type": entity_type,
                "confidence": _clamp_confidence(item.get("confidence", 0.5)),
                "evidence_chunk_ids": [str(v) for v in evidence if str(v).strip()],
            }
        )
    return entities


def _sanitize_relations(raw_relations: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_relations, list):
        return []

    relations: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw_relations:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        relation_type = str(item.get("relation_type", "related_to")).strip() or "related_to"
        if not source or not target:
            continue
        key = (_normalize_label(source), _normalize_label(target), relation_type.lower())
        if key in seen:
            continue
        seen.add(key)
        evidence = item.get("evidence_chunk_ids", [])
        if not isinstance(evidence, list):
            evidence = []
        relations.append(
            {
                "source": source,
                "target": target,
                "relation_type": relation_type,
                "confidence": _clamp_confidence(item.get("confidence", 0.5)),
                "evidence_chunk_ids": [str(v) for v in evidence if str(v).strip()],
            }
        )
    return relations


def extract_semantic_candidates(
    graph_root: str | Path,
    model_name: str = DEFAULT_MODEL,
    max_batches_per_topic: int = 0,
    topic_keys: list[str] | None = None,
) -> dict[str, Any]:
    graph_root = Path(graph_root)
    topic_dirs = sorted(path for path in graph_root.iterdir() if path.is_dir())
    if not topic_dirs:
        return {"topics": 0, "batches": 0, "entities": 0, "relations": 0}

    allowed_topics: set[str] | None = None
    if topic_keys:
        allowed_topics = {str(topic).strip().lower() for topic in topic_keys if str(topic).strip()}

    llm = get_langchain_llm(model_name=model_name)

    totals = {"topics": 0, "batches": 0, "entities": 0, "relations": 0}
    for topic_dir in topic_dirs:
        if allowed_topics is not None and topic_dir.name.lower() not in allowed_topics:
            continue

        batch_path = topic_dir / "semantic_batches.jsonl"
        if not batch_path.is_file():
            continue

        rows = _read_jsonl(batch_path)
        if not rows:
            continue

        if max_batches_per_topic > 0:
            rows = rows[:max_batches_per_topic]

        extraction_rows: list[dict[str, Any]] = []
        topic_entity_count = 0
        topic_relation_count = 0

        for batch in rows:
            prompt = _batch_prompt(batch)
            try:
                response = llm.invoke(prompt)
                raw_payload = _extract_json_payload(_extract_text_response(response))
                entities = _sanitize_entities(raw_payload.get("entities", []))
                relations = _sanitize_relations(raw_payload.get("relations", []))
                error = None
            except Exception as exc:  # pragma: no cover - external API call
                entities = []
                relations = []
                error = str(exc)

            topic_entity_count += len(entities)
            topic_relation_count += len(relations)

            extraction_rows.append(
                {
                    "batch_id": batch.get("batch_id"),
                    "topic": batch.get("topic"),
                    "training_id": batch.get("training_id"),
                    "document_id": batch.get("document_id"),
                    "chunk_count": batch.get("chunk_count"),
                    "entity_count": len(entities),
                    "relation_count": len(relations),
                    "entities": entities,
                    "relations": relations,
                    "error": error,
                }
            )

        _write_jsonl(topic_dir / "semantic_extractions.jsonl", extraction_rows)
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "topic_key": topic_dir.name,
            "model_name": model_name,
            "batch_count": len(rows),
            "entity_count": topic_entity_count,
            "relation_count": topic_relation_count,
            "failed_batches": sum(1 for row in extraction_rows if row.get("error")),
        }
        with (topic_dir / "semantic_extraction_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

        totals["topics"] += 1
        totals["batches"] += len(rows)
        totals["entities"] += topic_entity_count
        totals["relations"] += topic_relation_count

    return totals