"""Stage B — the graph's query-time entry point.

Builds a *second* Qdrant collection (``QDRANT_ENTITY_COLLECTION``) holding one
point per graph entity, per topic. Each point embeds the short string
``"label — entity_type"`` (type disambiguates e.g. "Pressure" the Parameter from
"Pressure" the Concept) and carries the Stage-A ``idf`` / ``chunk_frequency`` in
its payload, so a query-time match already knows how rare the entity is.

The collection mirrors the chunk collection: hybrid dense (BGE-M3, 1024-d,
cosine) + sparse, so exact codes / part numbers match lexically while concepts
match semantically. It reuses the embedding singleton from
``retrieval_helpers.get_embeddings`` — no second model is loaded.

CLI (standalone build, no ``--semantic`` re-run needed):
    python -m dc_available.graph.construct.entity_index --build [--source SOURCE]
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from dc_available.config import data_config

# Deterministic point IDs: Qdrant requires int/UUID, but entity_id is a string
# like "entity::protocol::profinet". uuid5 maps (topic, entity_id) -> stable UUID
# so re-builds upsert the same point instead of duplicating it.
_ENTITY_NAMESPACE = uuid.NAMESPACE_URL


def _entity_point_id(topic_key: str, entity_id: str) -> str:
    return str(uuid.uuid5(_ENTITY_NAMESPACE, f"{topic_key}::{entity_id}"))


def _load_entities(graph_root: Path) -> list[tuple[str, dict[str, Any]]]:
    """Return (topic_key, entity_record) for every entity across all topics.

    One record per (entity, topic) — the same label in two topics yields two
    records (and later two points), preserving each topic's own IDF.
    """
    records: list[tuple[str, dict[str, Any]]] = []
    for topic_dir in sorted(path for path in graph_root.iterdir() if path.is_dir()):
        entities_path = topic_dir / "semantic_entities_merged.jsonl"
        if not entities_path.is_file():
            continue
        with entities_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append((topic_dir.name, json.loads(line)))
    return records


# Constructing a QdrantVectorStore runs langchain's collection validation, which
# embeds "dummy_text" via BGE-M3 (~4s on CPU) just to check the vector size. Doing
# that on every query cost ~9s per search. The entity collection is built once at
# ingest and never changes at query time, so we (a) skip that validation probe
# (validate_collection_config=False) and (b) cache the two vectorstore wrappers so
# they are constructed once per (client, embeddings) pair and reused thereafter.
_VS_CACHE: dict[tuple[int, int, str], QdrantVectorStore] = {}


def _get_vectorstore(
    client: QdrantClient, embeddings: Any, collection_name: str | None = None
) -> QdrantVectorStore:
    """Wrap the entity collection in a hybrid QdrantVectorStore (same config as chunks)."""
    collection_name = collection_name or data_config.QDRANT_ENTITY_COLLECTION
    # Collection name is part of the cache key so separate collections (e.g. QA vs RICH)
    # never alias onto the same cached wrapper.
    key = (id(client), id(embeddings), "hybrid", collection_name)
    vs = _VS_CACHE.get(key)
    if vs is None:
        vs = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
            sparse_embedding=embeddings.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name="sparse",
            validate_collection_config=False,
        )
        _VS_CACHE[key] = vs
    return vs


def _get_vectorstore_dense(
    client: QdrantClient, embeddings: Any, collection_name: str | None = None
) -> QdrantVectorStore:
    """Dense-only view of the entity collection (for the raw dense score, Stage C / C5)."""
    collection_name = collection_name or data_config.QDRANT_ENTITY_COLLECTION
    key = (id(client), id(embeddings), "dense", collection_name)
    vs = _VS_CACHE.get(key)
    if vs is None:
        vs = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
            retrieval_mode=RetrievalMode.DENSE,
            validate_collection_config=False,
        )
        _VS_CACHE[key] = vs
    return vs


def build_entity_index(
    graph_root: str | Path,
    embeddings: Any | None = None,
    client: QdrantClient | None = None,
    collection_name: str | None = None,
) -> dict[str, Any]:
    """(Re)build the entity collection from the merged entity JSONL files.

    Always recreates the collection so entities removed upstream cannot linger.
    Does NOT touch the chunk collection or the parent docstore.

    ``collection_name`` targets a specific entity collection (defaults to the QA
    ``QDRANT_ENTITY_COLLECTION``). A separate source (e.g. RICH) passes its own name so the
    recreate never wipes another source's entities.
    """
    graph_root = Path(graph_root)
    if embeddings is None:
        from dc_available.RAG.retrieval_helpers import get_embeddings

        embeddings = get_embeddings()
    if client is None:
        client = QdrantClient(url=data_config.QDRANT_URL)

    collection = collection_name or data_config.QDRANT_ENTITY_COLLECTION
    records = _load_entities(graph_root)
    if not records:
        print(f"[entity-index] No entities found under {graph_root}; nothing to build.")
        return {"points": 0, "topics": 0}

    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []
    topics: set[str] = set()
    for topic_key, entity in records:
        label = str(entity.get("label", "")).strip()
        entity_type = str(entity.get("entity_type", "")).strip()
        entity_id = str(entity.get("entity_id", "")).strip()
        if not label or not entity_id:
            continue
        texts.append(f"{label} — {entity_type}" if entity_type else label)
        metadatas.append(
            {
                "entity_id": entity_id,
                "label": label,
                "entity_type": entity_type,
                "topic_key": topic_key,
                "idf": entity.get("idf"),
                "chunk_frequency": entity.get("chunk_frequency"),
            }
        )
        ids.append(_entity_point_id(topic_key, entity_id))
        topics.add(topic_key)

    # Recreate the collection fresh (mirror the chunk collection: dense + sparse).
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=embeddings.embedding_dim,
            distance=Distance.COSINE,
            on_disk=False,
        ),
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=True))
        },
    )

    vectorstore = _get_vectorstore(client, embeddings, collection)
    vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
        batch_size=data_config.ENTITY_INDEX_BATCH_SIZE,
    )

    print(
        f"[entity-index] Upserted {len(ids)} point(s) across {len(topics)} topic(s) "
        f"into '{collection}'."
    )
    return {"points": len(ids), "topics": len(topics)}


def search_entities(
    query: str,
    k: int = 10,
    client: QdrantClient | None = None,
    embeddings: Any | None = None,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid-search the entity index. Reused by Stage C and the acceptance script.

    ``collection_name`` selects which entity collection to search (defaults to the QA
    ``QDRANT_ENTITY_COLLECTION``); a separate source such as RICH passes its own name.
    """
    if embeddings is None:
        from dc_available.RAG.retrieval_helpers import get_embeddings

        embeddings = get_embeddings()
    if client is None:
        client = QdrantClient(url=data_config.QDRANT_URL)

    vectorstore = _get_vectorstore(client, embeddings, collection_name)
    hits = vectorstore.similarity_search_with_score(query, k=k)

    results: list[dict[str, Any]] = []
    for document, score in hits:
        payload = document.metadata or {}
        results.append(
            {
                "score": float(score),
                "entity_id": payload.get("entity_id"),
                "label": payload.get("label"),
                "entity_type": payload.get("entity_type"),
                "topic_key": payload.get("topic_key"),
                "idf": payload.get("idf"),
                "chunk_frequency": payload.get("chunk_frequency"),
            }
        )
    return results


def search_entities_scored(
    query: str,
    k: int = 50,
    client: QdrantClient | None = None,
    embeddings: Any | None = None,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    """Match a query to entities, returning BOTH the hybrid fused score and the
    raw dense score for each hit.

    - ``fused_score``: hybrid (dense+sparse) RRF score — drives ordering.
    - ``dense_score``: dense-only cosine — the alternative confidence signal
      under evaluation (Stage C / C5). ``None`` if the entity fell outside the
      dense top-k.

    Hits are ordered by ``fused_score`` (descending). Results are joined on
    ``(entity_id, topic_key)``, which uniquely identifies a point (per-topic).

    ``collection_name`` selects which entity collection to search (defaults to the QA
    ``QDRANT_ENTITY_COLLECTION``); the RICH graph leg passes ``QDRANT_RICH_ENTITY_COLLECTION``.
    """
    if embeddings is None:
        from dc_available.RAG.retrieval_helpers import get_embeddings

        embeddings = get_embeddings()
    if client is None:
        client = QdrantClient(url=data_config.QDRANT_URL)

    from time import perf_counter as _pc
    _t = _pc()
    hybrid_hits = _get_vectorstore(client, embeddings, collection_name).similarity_search_with_score(query, k=k)
    _t_hybrid = _pc() - _t; _t = _pc()
    dense_hits = _get_vectorstore_dense(client, embeddings, collection_name).similarity_search_with_score(query, k=k)
    _t_dense = _pc() - _t
    print(f"[LAT] entity_search  hybrid(dense+sparse)={_t_hybrid:.2f}s  "
          f"dense_only={_t_dense:.2f}s", flush=True)

    dense_by_key: dict[tuple[Any, Any], float] = {}
    for document, score in dense_hits:
        payload = document.metadata or {}
        dense_by_key[(payload.get("entity_id"), payload.get("topic_key"))] = float(score)

    results: list[dict[str, Any]] = []
    for document, score in hybrid_hits:
        payload = document.metadata or {}
        key = (payload.get("entity_id"), payload.get("topic_key"))
        results.append(
            {
                "entity_id": payload.get("entity_id"),
                "label": payload.get("label"),
                "entity_type": payload.get("entity_type"),
                "topic_key": payload.get("topic_key"),
                "idf": payload.get("idf"),
                "chunk_frequency": payload.get("chunk_frequency"),
                "fused_score": float(score),
                "dense_score": dense_by_key.get(key),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Stage-B entity index.")
    parser.add_argument("--build", action="store_true", help="(Re)build the entity collection.")
    parser.add_argument("--source", default=None, help="Data source key (default: env/DEFAULT_DATA_SOURCE).")
    args = parser.parse_args()

    # Resolve source-specific paths before touching path-dependent modules.
    if args.source:
        data_config.update_data_config(args.source)
    else:
        data_config.update_data_config()

    if not args.build:
        parser.print_help()
        return

    stats = build_entity_index(graph_root=data_config.DEFAULT_GRAPH_ARTIFACTS_PATH)
    print(f"[entity-index] Done: {stats}")


if __name__ == "__main__":
    main()
