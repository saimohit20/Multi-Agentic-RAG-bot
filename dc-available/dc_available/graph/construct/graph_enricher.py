from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dc_available.common.data_loading import load_docs_from_jsonl
from dc_available.common.helpers import topic_from_storage_path
from dc_available.RAG.data import AvailableMetaData  # data.py stays in RAG (shared with retrieval)


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-z0-9_\-]", "", value)
    return value or "unknown"


def _infer_topic(index_file: Path, ingestion_root: Path, storage_path: str) -> str:
    # Mirror graph_builder._infer_topic_level_training: the leading path folder is only
    # the real topic when the path is deep enough (<topic>/<subtopic...>/<level>/<training>/
    # index.jsonl, i.e. >= 4 parts). RICH's shallow layout (<level>/<interview>/index.jsonl)
    # has no topic folder, so fall back to the storage-path topic instead — this keeps the
    # structural and semantic phases on the SAME topic key (e.g. both 'rich', not 'expert').
    try:
        relative = index_file.relative_to(ingestion_root)
        if len(relative.parts) >= 4:
            return relative.parts[0]
    except ValueError:
        pass

    storage_topic = topic_from_storage_path(storage_path)
    if storage_topic:
        return storage_topic

    normalized = storage_path.replace("\\", "/")
    marker = "/DC_data/"
    if marker in normalized:
        tail = normalized.split(marker, 1)[1]
        parts = [part for part in tail.split("/") if part]
        if parts:
            return parts[0]
    return "Unknown"


@dataclass(frozen=True)
class SemanticChunk:
    chunk_id: str
    page_content: str
    document_id: str
    training_id: str
    topic: str
    page_id: int | None
    storage_path: str | None

    def to_batch_dict(self, index_in_batch: int) -> dict[str, Any]:
        return {
            "index": index_in_batch,
            "chunk_id": self.chunk_id,
            "page_id": self.page_id,
            "storage_path": self.storage_path,
            "text": self.page_content,
        }


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _is_valid_semantic_chunk(metadata: dict[str, Any], page_content: str, min_chars: int) -> bool:
    if not page_content or len(page_content.strip()) < min_chars:
        return False
    if metadata.get(AvailableMetaData.IS_META_SUMMARY.value):
        return False
    if AvailableMetaData.PART_OF_DOCUMENT.value not in metadata:
        return False
    file_type = str(metadata.get(AvailableMetaData.FILE_TYPE.value, ""))
    if file_type == "folder":
        return False
    return True


def _build_overlapping_ranges(total: int, batch_size: int, overlap: int) -> list[tuple[int, int]]:
    if total <= 0:
        return []
    if total <= batch_size:
        return [(0, total)]

    step = max(1, batch_size - overlap)
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(total, start + batch_size)
        ranges.append((start, end))
        if end == total:
            break
        start += step
    return ranges


def _write_jsonl(file_path: Path, rows: list[dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def prepare_semantic_batches(
    ingestion_root: str | Path,
    graph_root: str | Path,
    batch_size: int,
    overlap: int,
    min_chars: int,
) -> dict[str, Any]:
    ingestion_root = Path(ingestion_root)
    graph_root = Path(graph_root)

    index_files = sorted(ingestion_root.rglob("**/*/index.jsonl"))
    if not index_files:
        return {"topics": 0, "documents": 0, "chunks": 0, "batches": 0}

    topic_documents: dict[str, dict[tuple[str, str], list[SemanticChunk]]] = defaultdict(lambda: defaultdict(list))
    seen_texts: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    for index_file in index_files:
        docs = load_docs_from_jsonl(index_file)
        if not docs:
            continue
        # Resolve the topic ONCE per index file (from the first chunk), exactly like
        # graph_builder does. RICH chunks carry per-modality storage paths of differing
        # depths (…/pptx/…, …/pdf/…, …/InterviewN), so a per-chunk lookup would scatter one
        # interview across several topics; the first chunk (the interview summary) anchors
        # the whole file to a single topic key, matching the structural graph.
        file_storage_path = str(docs[0].metadata.get(AvailableMetaData.STORAGE_PATH.value, ""))
        topic = _infer_topic(index_file, ingestion_root, file_storage_path)
        for doc in docs:
            metadata = doc.metadata
            page_content = doc.page_content or ""
            if not _is_valid_semantic_chunk(metadata, page_content, min_chars):
                continue

            document_id = str(metadata[AvailableMetaData.PART_OF_DOCUMENT.value])
            chunk_id = str(metadata[AvailableMetaData.ID.value])
            training_id = str(metadata.get(AvailableMetaData.PART_OF_TRAINING.value, ""))
            storage_path = str(metadata.get(AvailableMetaData.STORAGE_PATH.value, ""))
            dedup_key = (topic, training_id, document_id)
            normalized_text = _normalize_text(page_content)
            if normalized_text in seen_texts[dedup_key]:
                continue
            seen_texts[dedup_key].add(normalized_text)

            raw_page_id = metadata.get(AvailableMetaData.PAGE_ID.value)
            page_id = None
            if raw_page_id is not None:
                try:
                    page_id = int(raw_page_id)
                except (TypeError, ValueError):
                    page_id = None

            topic_documents[_slugify(topic)][(document_id, training_id)].append(
                SemanticChunk(
                    chunk_id=chunk_id,
                    page_content=page_content.strip(),
                    document_id=document_id,
                    training_id=training_id,
                    topic=topic,
                    page_id=page_id,
                    storage_path=storage_path or None,
                )
            )

    totals = {"topics": 0, "documents": 0, "chunks": 0, "batches": 0}

    for topic_key in sorted(topic_documents.keys()):
        document_chunks = topic_documents[topic_key]
        batch_rows: list[dict[str, Any]] = []
        manifest_documents: list[dict[str, Any]] = []

        for (document_id, training_id) in sorted(document_chunks.keys()):
            chunks = document_chunks[(document_id, training_id)]
            chunks.sort(key=lambda item: (item.page_id if item.page_id is not None else 10**9, item.chunk_id))
            ranges = _build_overlapping_ranges(len(chunks), batch_size=batch_size, overlap=overlap)
            manifest_documents.append(
                {
                    "document_id": document_id,
                    "chunk_count": len(chunks),
                    "batch_count": len(ranges),
                    "training_id": training_id,
                }
            )
            totals["documents"] += 1
            totals["chunks"] += len(chunks)

            for batch_index, (start, end) in enumerate(ranges):
                selected = chunks[start:end]
                batch_rows.append(
                    {
                        "batch_id": f"{topic_key}::{training_id}::{document_id}::{batch_index}",
                        "topic": selected[0].topic,
                        "training_id": selected[0].training_id,
                        "document_id": document_id,
                        "chunk_count": len(selected),
                        "source_range": {"start": start, "end": end - 1},
                        "chunks": [chunk.to_batch_dict(index_in_batch=i) for i, chunk in enumerate(selected)],
                    }
                )

            totals["batches"] += len(ranges)

        if not batch_rows:
            continue

        topic_dir = graph_root / topic_key
        _write_jsonl(topic_dir / "semantic_batches.jsonl", batch_rows)
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "topic_key": topic_key,
            "document_count": len(manifest_documents),
            "chunk_count": sum(item["chunk_count"] for item in manifest_documents),
            "batch_count": len(batch_rows),
            "batch_size": batch_size,
            "overlap": overlap,
            "min_chars": min_chars,
            "documents": manifest_documents,
        }
        with (topic_dir / "semantic_batch_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        totals["topics"] += 1

    return totals