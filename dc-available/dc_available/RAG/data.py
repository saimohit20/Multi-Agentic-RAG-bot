from enum import Enum
from pathlib import Path
from typing import List, Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class AvailableDataloader(BaseLoader):
    def __init__(
        self,
        docs_path: str | Path,
    ):
        self._docs_path = Path(docs_path)

    def load(self) -> List[Document]:
        """Load data into Document objects."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        from dc_available.common.data_loading import load_docs_from_jsonl

        documents = []
        if self._docs_path.is_file():
            documents.extend(load_docs_from_jsonl(self._docs_path))
        else:
            for file in self._docs_path.rglob("**/*/index.jsonl"):
                documents.extend(load_docs_from_jsonl(file))
        for doc in documents:
            if "source" not in doc.metadata and "id" in doc.metadata:
                doc.metadata["source"] = doc.metadata["id"]
        return iter(documents)


class AvailableMetaData(str, Enum):
    """Enumerator of the different meta information from the AVAILABLE documents."""

    ID = "id"
    PAGE_ID = "page_id"
    DOCUMENT_TYPE = "document_type"
    FILE_TYPE = "file_type"
    TEXT = "text"
    LEVEL = "level"
    TRUST_SCORE = "trust_score"

    # Hierarchy information
    PART_OF_TRAINING = "part_of_training"
    PART_OF_DOCUMENT = "part_of_document"
    IS_META_SUMMARY = "is_meta_summary"

    # Information on storage paths
    STORAGE_PATH = "storage_path"
    IMAGE_PATH = "image_path"
    AUDIO_PATH = "audio_path"
    VIDEO_PATH = "video_path"

    # Links to related documents
    LINK_TO_PARENT = "link_to_parent"
    LINK_TO_CHILDREN = "link_to_children"
    LINK_TO_NEXT = "link_to_next"
    LINK_TO_PREVIOUS = "link_to_previous"
    LINK_TO_RELATED = "link_to_related"

    # Sequence information
    START_TIME = "start_time"
    END_TIME = "end_time"


class GraphNodeType(str, Enum):
    """Canonical node labels for the graph index."""

    TOPIC = "Topic"
    SUBTOPIC = "Subtopic"
    LEVEL = "Level"
    TRAINING = "Training"
    DOCUMENT = "Document"
    CHUNK = "Chunk"
    ENTITY = "Entity"


class GraphEdgeType(str, Enum):
    """Canonical edge labels for graph traversal and scoring."""

    CONTAINS = "contains"
    PART_OF = "part_of"
    MENTIONS = "mentions"
    RELATED_TO = "related_to"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"


# Mapping from the existing retrieval metadata into graph construction roles.
GRAPH_METADATA_FIELD_MAP: dict[str, dict[str, str]] = {
    AvailableMetaData.ID.value: {
        "graph_role": "chunk_id",
        "target": "Chunk.id (global bridge key to Qdrant/docstore)",
    },
    AvailableMetaData.PART_OF_DOCUMENT.value: {
        "graph_role": "document_id",
        "target": "Document.id",
    },
    AvailableMetaData.PART_OF_TRAINING.value: {
        "graph_role": "training_id",
        "target": "Training.id",
    },
    AvailableMetaData.LEVEL.value: {
        "graph_role": "level",
        "target": "Level.name",
    },
    AvailableMetaData.STORAGE_PATH.value: {
        "graph_role": "hierarchy_fallback",
        "target": "Topic/Level/Training derivation when explicit fields are missing",
    },
    AvailableMetaData.DOCUMENT_TYPE.value: {
        "graph_role": "document_modality",
        "target": "Document.modality and Chunk.modality",
    },
    AvailableMetaData.FILE_TYPE.value: {
        "graph_role": "file_type",
        "target": "Document.file_type",
    },
    AvailableMetaData.TRUST_SCORE.value: {
        "graph_role": "trust_signal",
        "target": "Document.base_trust and Chunk.base_trust",
    },
    AvailableMetaData.START_TIME.value: {
        "graph_role": "time_range_start",
        "target": "Chunk.start_time",
    },
    AvailableMetaData.END_TIME.value: {
        "graph_role": "time_range_end",
        "target": "Chunk.end_time",
    },
    AvailableMetaData.LINK_TO_PARENT.value: {
        "graph_role": "explicit_parent",
        "target": "part_of edges when source metadata already links parent",
    },
    AvailableMetaData.LINK_TO_CHILDREN.value: {
        "graph_role": "explicit_children",
        "target": "contains edges when source metadata already links children",
    },
    AvailableMetaData.LINK_TO_NEXT.value: {
        "graph_role": "sequence_next",
        "target": "related_to edge with relation=next",
    },
    AvailableMetaData.LINK_TO_PREVIOUS.value: {
        "graph_role": "sequence_previous",
        "target": "related_to edge with relation=previous",
    },
    AvailableMetaData.LINK_TO_RELATED.value: {
        "graph_role": "explicit_related",
        "target": "related_to edge",
    },
}
