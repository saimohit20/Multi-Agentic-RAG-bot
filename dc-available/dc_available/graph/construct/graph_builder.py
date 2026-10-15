from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dc_available.common.data_loading import load_docs_from_jsonl
from dc_available.common.helpers import topic_from_storage_path
from dc_available.RAG.data import AvailableMetaData, GraphEdgeType, GraphNodeType  # data.py stays in RAG (shared with retrieval)


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-z0-9_\-]", "", value)
    return value or "unknown"


def _training_fallback_id(topic: str, level: str, training: str) -> str:
    return f"training::{_slugify(topic)}::{_slugify(level)}::{_slugify(training)}"


def _topic_from_storage_path(storage_path: str) -> str | None:
    # Bottom-up topic detection (ignores wrapper folders above the topic).
    return topic_from_storage_path(storage_path)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str):
        if value.strip().startswith("[") and value.strip().endswith("]"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed if v]
            except json.JSONDecodeError:
                pass
        return [value] if value else []
    return [str(value)]


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    edge_type: str
    properties: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.edge_type,
            "properties": {k: v for k, v in self.properties},
        }


class TopicGraphAccumulator:
    def __init__(self, topic_name: str):
        self.topic_name = topic_name
        self.topic_id = f"topic::{_slugify(topic_name)}"
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: set[GraphEdge] = set()
        self.lookup = {
            "chunks": {},
            "documents": {},
            "trainings": {},
        }
        self.processed_files: list[str] = []
        self.document_chunks: dict[str, set[str]] = defaultdict(set)
        self.training_documents: dict[str, set[str]] = defaultdict(set)

        self.add_node(
            node_id=self.topic_id,
            node_type=GraphNodeType.TOPIC.value,
            properties={"name": topic_name},
        )

    def add_node(self, node_id: str, node_type: str, properties: dict[str, Any]) -> None:
        existing = self.nodes.get(node_id)
        serialized_props = {
            k: v
            for k, v in properties.items()
            if v is not None and v != ""
        }
        if existing is None:
            self.nodes[node_id] = {
                "id": node_id,
                "type": node_type,
                "properties": serialized_props,
            }
            return
        existing["properties"].update(serialized_props)

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        if not source or not target:
            return
        props = properties or {}
        serialized = tuple(sorted((k, str(v)) for k, v in props.items() if v is not None))
        self.edges.add(
            GraphEdge(
                source=source,
                target=target,
                edge_type=edge_type,
                properties=serialized,
            )
        )

    def add_contains_pair(self, parent_id: str, child_id: str) -> None:
        self.add_edge(parent_id, child_id, GraphEdgeType.CONTAINS.value)
        self.add_edge(child_id, parent_id, GraphEdgeType.PART_OF.value)

    def add_training_level(
        self, level_name: str, training_id: str, subtopics: list[str] | None = None
    ) -> str:
        # Hierarchy: Topic -> [Subtopic chain ->] Level -> Training.
        # Subtopics are the branch folders inside a tree (e.g. GROWNFIELD/O2C).
        subtopics = subtopics or []
        parent_id = self.topic_id
        slug_chain = [_slugify(self.topic_name)]
        for sub_name in subtopics:
            slug_chain.append(_slugify(sub_name))
            subtopic_id = "subtopic::" + "::".join(slug_chain)
            self.add_node(
                node_id=subtopic_id,
                node_type=GraphNodeType.SUBTOPIC.value,
                properties={"name": sub_name, "topic": self.topic_name},
            )
            self.add_contains_pair(parent_id, subtopic_id)
            parent_id = subtopic_id
        # Include the subtopic chain in the level id so the same level name under
        # different branches (e.g. O2C/Advanced vs B2P/Advanced) stays distinct.
        level_id = "level::" + "::".join(slug_chain + [_slugify(level_name)])
        self.add_node(
            node_id=level_id,
            node_type=GraphNodeType.LEVEL.value,
            properties={"name": level_name, "topic": self.topic_name},
        )
        self.add_contains_pair(parent_id, level_id)
        self.add_contains_pair(level_id, training_id)
        return level_id

    def register_document_chunk_links(self) -> None:
        for training_id, document_ids in self.training_documents.items():
            for document_id in sorted(document_ids):
                self.add_contains_pair(training_id, document_id)

        for document_id, chunk_ids in self.document_chunks.items():
            for chunk_id in sorted(chunk_ids):
                self.add_contains_pair(document_id, chunk_id)

    def serialize(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        nodes = [self.nodes[node_id] for node_id in sorted(self.nodes.keys())]
        edges = [edge.to_dict() for edge in sorted(self.edges, key=lambda e: (e.source, e.edge_type, e.target, e.properties))]
        return nodes, edges


def _infer_topic_level_training(
    index_file: Path, ingestion_root: Path
) -> tuple[str | None, list[str], str | None, str | None]:
    try:
        relative = index_file.relative_to(ingestion_root)
        parts = relative.parts
        # Path: .../<topic>/<subtopic...>/<level>/<training>/index.jsonl
        # Top-down: the topic is the FIRST folder (one tree per top-level topic,
        # e.g. Hydraulics / DigitalTransformation / GROWNFIELD). Any folders
        # between the topic and the trailing <level>/<training> are subtopic
        # branches (e.g. GROWNFIELD/O2C -> subtopic 'O2C'). DT/Hydraulics have none.
        if len(parts) >= 4:
            topic = parts[0]
            level = parts[-3]
            training = parts[-2]
            subtopics = list(parts[1:-3])
            return topic, subtopics, level, training
    except ValueError:
        pass
    return None, [], None, None


def _get_or_create_topic_graph(topic_graphs: dict[str, TopicGraphAccumulator], topic_name: str) -> TopicGraphAccumulator:
    key = _slugify(topic_name)
    if key not in topic_graphs:
        topic_graphs[key] = TopicGraphAccumulator(topic_name=topic_name)
    return topic_graphs[key]


def _apply_explicit_links(graph: TopicGraphAccumulator, chunk_id: str, metadata: dict[str, Any]) -> None:
    parent_key = AvailableMetaData.LINK_TO_PARENT.value
    children_key = AvailableMetaData.LINK_TO_CHILDREN.value
    next_key = AvailableMetaData.LINK_TO_NEXT.value
    prev_key = AvailableMetaData.LINK_TO_PREVIOUS.value
    related_key = AvailableMetaData.LINK_TO_RELATED.value

    for parent_id in _as_list(metadata.get(parent_key)):
        graph.add_edge(chunk_id, parent_id, GraphEdgeType.PART_OF.value, {"origin": parent_key})
        graph.add_edge(parent_id, chunk_id, GraphEdgeType.CONTAINS.value, {"origin": parent_key})

    for child_id in _as_list(metadata.get(children_key)):
        graph.add_edge(chunk_id, child_id, GraphEdgeType.CONTAINS.value, {"origin": children_key})
        graph.add_edge(child_id, chunk_id, GraphEdgeType.PART_OF.value, {"origin": children_key})

    for next_id in _as_list(metadata.get(next_key)):
        graph.add_edge(chunk_id, next_id, GraphEdgeType.RELATED_TO.value, {"origin": next_key, "relation": "next"})

    for prev_id in _as_list(metadata.get(prev_key)):
        graph.add_edge(chunk_id, prev_id, GraphEdgeType.RELATED_TO.value, {"origin": prev_key, "relation": "previous"})

    for related_id in _as_list(metadata.get(related_key)):
        graph.add_edge(chunk_id, related_id, GraphEdgeType.RELATED_TO.value, {"origin": related_key, "relation": "related"})


def _write_jsonl(file_path: Path, rows: list[dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_per_topic_graph(
    ingestion_root: str | Path,
    graph_root: str | Path,
    refresh: bool = False,
) -> dict[str, Any]:
    ingestion_root = Path(ingestion_root)
    graph_root = Path(graph_root)

    if not ingestion_root.exists():
        raise FileNotFoundError(f"Ingestion root does not exist: {ingestion_root}")

    index_files = sorted(ingestion_root.rglob("**/*/index.jsonl"))
    if not index_files:
        return {
            "topics": 0,
            "trainings": 0,
            "documents": 0,
            "chunks": 0,
            "index_files": 0,
        }

    if refresh and graph_root.exists():
        shutil.rmtree(graph_root)
    graph_root.mkdir(parents=True, exist_ok=True)

    topic_graphs: dict[str, TopicGraphAccumulator] = {}

    for index_file in index_files:
        default_topic, default_subtopics, default_level, default_training_name = _infer_topic_level_training(index_file, ingestion_root)

        docs = load_docs_from_jsonl(index_file)
        if not docs:
            continue

        first_meta = docs[0].metadata
        topic_name = default_topic
        if not topic_name:
            topic_name = _topic_from_storage_path(str(first_meta.get(AvailableMetaData.STORAGE_PATH.value, "")))
        topic_name = topic_name or "Unknown"

        graph = _get_or_create_topic_graph(topic_graphs, topic_name)
        graph.processed_files.append(str(index_file))

        for doc in docs:
            metadata = doc.metadata
            chunk_id = str(metadata.get(AvailableMetaData.ID.value, "")).strip()
            if not chunk_id:
                continue

            level_name = str(metadata.get(AvailableMetaData.LEVEL.value, "") or default_level or "Unknown")
            training_id = str(
                metadata.get(AvailableMetaData.PART_OF_TRAINING.value)
                or _training_fallback_id(topic_name, level_name, default_training_name or "unknown")
            )

            document_id = str(
                metadata.get(AvailableMetaData.PART_OF_DOCUMENT.value)
                or metadata.get(AvailableMetaData.ID.value)
            )

            graph.add_node(
                node_id=training_id,
                node_type=GraphNodeType.TRAINING.value,
                properties={
                    "name": default_training_name,
                    "topic": topic_name,
                    "level": level_name,
                },
            )
            graph.add_training_level(
                level_name=level_name,
                training_id=training_id,
                subtopics=default_subtopics,
            )

            graph.add_node(
                node_id=document_id,
                node_type=GraphNodeType.DOCUMENT.value,
                properties={
                    "training_id": training_id,
                    "document_type": metadata.get(AvailableMetaData.DOCUMENT_TYPE.value),
                    "file_type": metadata.get(AvailableMetaData.FILE_TYPE.value),
                    "storage_path": metadata.get(AvailableMetaData.STORAGE_PATH.value),
                },
            )

            graph.add_node(
                node_id=chunk_id,
                node_type=GraphNodeType.CHUNK.value,
                properties={
                    "document_id": document_id,
                    "training_id": training_id,
                    "start_time": metadata.get(AvailableMetaData.START_TIME.value),
                    "end_time": metadata.get(AvailableMetaData.END_TIME.value),
                    "trust_score": metadata.get(AvailableMetaData.TRUST_SCORE.value),
                    "storage_path": metadata.get(AvailableMetaData.STORAGE_PATH.value),
                },
            )

            graph.document_chunks[document_id].add(chunk_id)
            graph.training_documents[training_id].add(document_id)

            graph.lookup["chunks"][chunk_id] = {
                "topic": topic_name,
                "training_id": training_id,
                "document_id": document_id,
                "node_id": chunk_id,
            }
            graph.lookup["documents"][document_id] = {
                "topic": topic_name,
                "training_id": training_id,
                "node_id": document_id,
            }
            graph.lookup["trainings"][training_id] = {
                "topic": topic_name,
                "level": level_name,
                "node_id": training_id,
            }

            _apply_explicit_links(graph, chunk_id, metadata)

    all_training_ids: set[str] = set()
    all_document_ids: set[str] = set()
    all_chunk_ids: set[str] = set()
    topic_registry: list[dict[str, Any]] = []

    for topic_key in sorted(topic_graphs.keys()):
        graph = topic_graphs[topic_key]
        graph.register_document_chunk_links()
        nodes, edges = graph.serialize()

        topic_dir = graph_root / topic_key
        _write_jsonl(topic_dir / "nodes.jsonl", nodes)
        _write_jsonl(topic_dir / "edges.jsonl", edges)

        lookup_path = topic_dir / "id_lookup.json"
        with lookup_path.open("w", encoding="utf-8") as handle:
            json.dump(graph.lookup, handle, indent=2, sort_keys=True)

        topic_manifest = {
            "topic": graph.topic_name,
            "topic_key": topic_key,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "training_count": len(graph.lookup["trainings"]),
            "document_count": len(graph.lookup["documents"]),
            "chunk_count": len(graph.lookup["chunks"]),
            "source_index_files": sorted(graph.processed_files),
        }
        with (topic_dir / "build_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(topic_manifest, handle, indent=2, sort_keys=True)

        all_training_ids.update(graph.lookup["trainings"].keys())
        all_document_ids.update(graph.lookup["documents"].keys())
        all_chunk_ids.update(graph.lookup["chunks"].keys())

        topic_registry.append(
            {
                "topic": graph.topic_name,
                "topic_key": topic_key,
                "path": str(topic_dir),
                "training_count": len(graph.lookup["trainings"]),
                "document_count": len(graph.lookup["documents"]),
                "chunk_count": len(graph.lookup["chunks"]),
            }
        )

    global_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "topics": len(topic_graphs),
        "trainings": len(all_training_ids),
        "documents": len(all_document_ids),
        "chunks": len(all_chunk_ids),
        "index_files": len(index_files),
        "topic_registry": topic_registry,
    }
    with (graph_root / "build_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(global_manifest, handle, indent=2, sort_keys=True)

    return {
        "topics": global_manifest["topics"],
        "trainings": global_manifest["trainings"],
        "documents": global_manifest["documents"],
        "chunks": global_manifest["chunks"],
        "index_files": global_manifest["index_files"],
    }