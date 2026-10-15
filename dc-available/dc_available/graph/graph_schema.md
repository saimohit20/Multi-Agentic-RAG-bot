# Graph Schema (Phase 1)

This document defines the initial graph schema for hybrid retrieval.
It is designed as a parallel index that reuses current retrieval metadata and IDs.

## Scope

- Keep vector (Qdrant) retrieval unchanged.
- Build a graph index in parallel from ingestion artifacts (`index.jsonl`, `summary.txt`).
- Use existing `id` and hierarchy metadata as bridge keys.

## Node Types

- `Topic`
  - `id`: `topic::<topic_name>`  (one tree per top-level topic, e.g. Hydraulics / DigitalTransformation / GROWNFIELD)
  - `name`
- `Subtopic`
  - `id`: `subtopic::<topic_name>::<subtopic_name>`  (branch inside a topic tree, e.g. GROWNFIELD/O2C; absent for topics without sub-areas)
  - `name`
- `Level`
  - `id`: `level::<topic_name>[::<subtopic_name>...]::<level_name>`
  - `name`
- `Training`
  - `id`: metadata `part_of_training` when available; otherwise deterministic fallback from path
  - `name`
- `Document`
  - `id`: metadata `part_of_document` for child chunks; or document-level metadata `id`
  - `document_type`, `file_type`, `storage_path`
- `Chunk`
  - `id`: metadata `id` (global bridge key to Qdrant/docstore)
  - `text`, `start_time`, `end_time`, `trust_score`
- `Entity`
  - `id`: `entity::<normalized_label>`
  - `label`, `normalized_label`, optional `entity_type`

## Edge Types

- Hierarchy
  - `contains`
  - `part_of`
- Semantic
  - `mentions`
  - `related_to`
- Evidence
  - `supports`
  - `contradicts`

## Deterministic ID Policy

- Reuse metadata IDs from ingestion as primary keys.
- Use metadata `id` as canonical chunk key.
- Use metadata `part_of_document` as canonical document key when present.
- Use metadata `part_of_training` as canonical training key when present.
- Derive topic/level/training fallback IDs from normalized `storage_path` when explicit IDs are missing.
- Entity keys are generated from a canonical normalized label (lowercase, trimmed, whitespace collapsed).

## Metadata Mapping

Current metadata source of truth: `dc_available/RAG/data.py` (`AvailableMetaData` and `GRAPH_METADATA_FIELD_MAP`).

- `id` -> `Chunk.id`
- `part_of_document` -> `Document.id`
- `part_of_training` -> `Training.id`
- `level` -> `Level.name`
- `storage_path` -> hierarchy fallback derivation (`Topic`, `Level`, `Training`)
- `document_type` -> modality on `Document`/`Chunk`
- `file_type` -> `Document.file_type`
- `trust_score` -> base trust signal on `Document`/`Chunk`
- `start_time`, `end_time` -> chunk temporal span
- `link_to_parent`, `link_to_children` -> optional hierarchy edges when present
- `link_to_next`, `link_to_previous`, `link_to_related` -> `related_to` relations

## Initial Build Rules (Phase 2 input)

- First pass is deterministic and non-LLM.
- Build hierarchy first (`Topic` -> `Subtopic` (optional) -> `Level` -> `Training` -> `Document` -> `Chunk`).
- Add only explicit link-based relations from metadata in deterministic pass.
- Semantic and evidence edges (`mentions`, `supports`, `contradicts`) are populated in later phases.

## Bridge to Hybrid Retrieval

- Shared key between the vector store (Qdrant) and graph is chunk metadata `id`.
- Document-level trust can be inherited by chunks when chunk trust is missing.
- Retrieval merge and rerank can combine:
  - vector similarity
  - graph relevance (path/hop/edge weight)
  - trust score cache