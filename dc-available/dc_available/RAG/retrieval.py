import json
import os
import pickle
import re
import uuid
from pathlib import Path
from typing import Any, List, Optional, Union

from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.multi_vector import SearchType
from langchain_community.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from dc_available.RAG.data import AvailableMetaData
from dc_available.summarization.markdown_utils import (
    doc_to_markdown,
)


# ---------------------------------------------------------------------------
# Embedding-text construction.
#
# The ingestion stores each chunk as a fenced JSON blob ({"title":..,
# "description":..}) and the document's code/title live only in metadata, never
# in the embedded text. That hurts retrieval of thin "overview"/parent documents
# (e.g. the '.000' files): a query like "what billing types exist?" cannot match
# 'O2C-04.000 Overview Billing Types' because neither the code nor the title is
# in the vectorised text. The helpers below fix that at index time by
#   1. prepending a small identity header (module · code+title · level), and
#   2. embedding clean prose instead of the raw JSON blob.
# A separate per-document "card" (see _build_document_cards) gives each document
# one strong representative vector.
# ---------------------------------------------------------------------------
_CODE_RE = re.compile(r"([A-Za-z0-9]{2,4})-\d{2}\.\d{3}")
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _module_code(meta: dict) -> str:
    """Best-effort module code (e.g. 'O2C') from the document name."""
    name = str(meta.get(AvailableMetaData.PART_OF_DOCUMENT) or "").strip()
    match = _CODE_RE.match(name)
    return match.group(1).upper() if match else ""


def _identity_header(meta: dict) -> str:
    """Compact identity line prepended to a chunk's embedded text.

    Carries the owning document's module, code+title and level into the text that
    is actually embedded, so a natural-language query or an exact code can match
    the document even when the chunk body is generic.
    """
    name = str(meta.get(AvailableMetaData.PART_OF_DOCUMENT) or "").strip()
    level = str(meta.get(AvailableMetaData.LEVEL) or "").strip()
    module = _module_code(meta)
    bits = []
    if module:
        bits.append(module)
    if name:
        bits.append(name)
    if level and level.lower() != "unknown":
        bits.append(level)
    return "[" + " | ".join(bits) + "]" if bits else ""


def _flatten_text(value: Any, out: List[str]) -> None:
    """Collect all string leaves of a (possibly nested) JSON value, in order."""
    if isinstance(value, str):
        text = value.strip()
        if text:
            out.append(text)
    elif isinstance(value, list):
        for item in value:
            _flatten_text(item, out)
    elif isinstance(value, dict):
        for item in value.values():
            _flatten_text(item, out)


def _clean_summary_text(text: str) -> str:
    """Turn a stored ```json {title, description}``` summary into plain prose.

    Embedding the raw JSON wastes model capacity on syntax tokens. This strips the
    code fence, parses the JSON and flattens title+description into readable text.
    Non-JSON content (video transcripts, HTML) is returned unchanged apart from
    fence removal, so it is always safe to call.
    """
    if not text:
        return ""
    raw = text.strip()
    fenced = _FENCE_RE.search(raw)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        obj = json.loads(raw)
    except Exception:
        return raw
    parts: List[str] = []
    if isinstance(obj, dict):
        title = obj.get("title")
        if isinstance(title, str) and title.strip():
            parts.append(title.strip())
        for key, value in obj.items():
            if key == "title":
                continue
            _flatten_text(value, parts)
    else:
        _flatten_text(obj, parts)
    return "\n".join(parts).strip() or raw


def _embedding_text(meta: dict, page_content: str) -> str:
    """Final text to embed for a chunk: identity header + cleaned body."""
    header = _identity_header(meta)
    body = _clean_summary_text(page_content)
    return f"{header}\n{body}".strip() if header else body


def _build_document_cards(documents: List[Document]) -> List[Document]:
    """One synthetic 'document card' vector per source document.

    Overview/parent documents (e.g. the '.000' files) have thin, generic chunks
    that lose to richer sibling documents. A card is a single strong vector that
    carries the document's code, title and overall description, giving the whole
    document a fair chance to be retrieved. Records without a document name
    (e.g. training-level summaries) get no card.
    """
    from collections import OrderedDict

    groups: "OrderedDict[str, List[Document]]" = OrderedDict()
    for doc in documents:
        name = doc.metadata.get(AvailableMetaData.PART_OF_DOCUMENT)
        if not name:
            continue
        groups.setdefault(str(name), []).append(doc)

    cards: List[Document] = []
    for name, members in groups.items():
        # Prefer the file-level meta summary as the description source.
        source = next(
            (
                d
                for d in members
                if d.metadata.get(AvailableMetaData.IS_META_SUMMARY)
                or str(d.metadata.get(AvailableMetaData.PAGE_ID)) == "-1"
            ),
            members[0],
        )
        meta = dict(source.metadata)
        header = _identity_header(meta)
        description = _clean_summary_text(source.page_content)[:1500].strip()
        # Repeat the name as the first body line so title-like queries
        # ("how to create a service order") match the card directly.
        card_text = "\n".join(part for part in (header, name, description) if part)

        meta[AvailableMetaData.ID] = str(uuid.uuid4())
        meta[AvailableMetaData.DOCUMENT_TYPE] = "document_card"
        meta[AvailableMetaData.PAGE_ID] = -1
        meta[AvailableMetaData.IS_META_SUMMARY] = True
        cards.append(Document(page_content=card_text, metadata=meta))
    return cards


class AvailableRetriever(ParentDocumentRetriever):
    id_key: str = AvailableMetaData.ID
    vectorstore: Union[Qdrant, QdrantVectorStore]  # Support both old and new vectorstore
    format_outputs: bool = True

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        add_to_docstore: bool = True,
    ) -> None:
        """Adds documents to the docstore and vectorstore."""
        if self.parent_splitter is not None:
            documents = self.parent_splitter.split_documents(documents)
        if ids is None:
            doc_ids = [doc.metadata[AvailableMetaData.ID] for doc in documents]
        else:
            assert len(documents) == len(ids)
            doc_ids = ids

        docs_for_vectorstore = []
        docs_for_docstore = []
        for i, doc in enumerate(documents):
            _id = doc_ids[i]
            sub_docs = self.child_splitter.split_documents([doc])

            doc_type = doc.metadata[AvailableMetaData.DOCUMENT_TYPE]
            for _doc in sub_docs:
                _doc.metadata[self.id_key] = _id
                _doc.metadata[AvailableMetaData.DOCUMENT_TYPE] = doc_type
                # Embed an identity header + cleaned prose instead of the raw JSON
                # blob, so each chunk carries its document's code/title.
                _doc.page_content = _embedding_text(_doc.metadata, _doc.page_content)
            docs_for_vectorstore.extend(sub_docs)

            # Add all documents to the doc store for later retrieval
            docs_for_docstore.append((_id, doc))

        # One strong "document card" vector per source document so thin
        # overview/parent documents can surface as a whole.
        docs_for_vectorstore.extend(_build_document_cards(documents))

        self.vectorstore.add_documents(docs_for_vectorstore)
        if add_to_docstore:
            self.docstore.mset(docs_for_docstore)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant to a query.
        
        Uses hybrid retrieval (dense + sparse vectors) via Qdrant.
        Fusion of dense semantic similarity and sparse keyword matching is 
        performed automatically by Qdrant (Reciprocal Rank Fusion).
        
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        search_kwargs = self.search_kwargs.copy()
        search_kwargs.update(kwargs)
        if self.search_type == SearchType.mmr:
            sub_docs = self.vectorstore.max_marginal_relevance_search(
                query, **search_kwargs
            )
        else:
            # Note: QdrantVectorStore with RetrievalMode.HYBRID automatically
            # performs dense + sparse fusion inside Qdrant
            sub_docs = self.vectorstore.similarity_search(query, **search_kwargs)

        # We do this to maintain the order of the ids that are returned
        trainings = {}
        for d, doc in enumerate(sub_docs):
            training_id = doc.metadata[AvailableMetaData.PART_OF_TRAINING]
            if training_id not in trainings:
                trainings[training_id] = []
            trainings[training_id].append(doc)

        outputs = []
        for training_id, docs in trainings.items():
            summary_doc = self.docstore.mget([training_id])[0]
            if summary_doc is None:
                summary_path = docs[0].metadata[AvailableMetaData.STORAGE_PATH]
            else:
                summary_path = Path(
                    summary_doc.metadata[AvailableMetaData.STORAGE_PATH]
                )

            if self.format_outputs:
                overview_text = self.format_training_with_paragraphs(docs, summary_doc)
                outputs.append(
                    Document(
                        page_content=overview_text,
                        metadata={"source": summary_path.name, "documents": docs},
                    )
                )
            else:
                outputs.extend(docs)
        return outputs

    def format_training_with_paragraphs(
        self, training_documents: list[Document], summary_doc: Document = None
    ) -> str:
        assert len(training_documents) > 0, "You must provide at least one document"

        if summary_doc is None:
            training_id = training_documents[0].metadata[
                AvailableMetaData.PART_OF_TRAINING
            ]
            summary_doc = self.docstore.mget(training_id)[0]
        else:
            training_id = summary_doc.metadata[AvailableMetaData.ID]
        assert all(
            doc.metadata[AvailableMetaData.PART_OF_TRAINING] == training_id
            for doc in training_documents
        ), "All texts must be from the same training"
        summary_path = Path(summary_doc.metadata[AvailableMetaData.STORAGE_PATH])
        summary_description = doc_to_markdown(summary_doc, method="retrieval").strip()

        # TODO: improve this heuristic
        if "basic" in str(summary_path).lower():
            level = "Beginner"
        elif (
            "intermediate" in str(summary_path).lower()
            or "advanced" in str(summary_path).lower()
        ):
            level = "Advanced"
        elif "expert" in str(summary_path).lower():
            level = "Expert"
        else:
            level = "Unknown"

        name = f"{summary_path.parent.parent.stem}/{summary_path.parent.stem}/{summary_path.stem}"
        output = "Name of training: " + name + "\n"
        output += "Level: " + level + "\n"
        output += "Description: " + summary_description + "\n"
        output += "Relevant texts:\n"
        for doc in training_documents:
            if doc == summary_doc:
                continue

            source_path = Path(doc.metadata[AvailableMetaData.STORAGE_PATH])
            if source_path.as_posix().endswith(".pdf"):
                try:
                    source_path = Path(doc.metadata[AvailableMetaData.IMAGE_PATH])
                except:
                    pass
            try:
                relative_source_path = source_path.relative_to(summary_path.parent)
            except:
                print(
                    f"[Error] Cannot set {source_path} relative to {summary_path.parent}"
                )
                relative_source_path = source_path
            source = f"Source file: {relative_source_path}"
            try:
                page = int(doc.metadata[AvailableMetaData.PAGE_ID])
                if page >= 0:
                    source += f"; Page {page}"
            except:
                pass
            doc_content = doc_to_markdown(doc, method="retrieval").strip()
            output += f"- {doc_content} [{source}]\n"

        # As a reference for later usage.
        all_ids = [doc.metadata[self.id_key] for doc in training_documents]
        output += f"IDs for internal use: [{all_ids}] (do not display to user)"
        return output

    def save_local(self, folder_path: str, index_name: str = "index") -> None:
        """Save the retriever's docstore to disk.

        The child-chunk vectors live in the Qdrant collection (persisted by the
        Qdrant server), so only the parent docstore + splitter config are pickled.
        These two must always be rebuilt together — see get_retriever().
        """
        path = Path(folder_path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / f"{index_name}_ret.pkl", "wb") as f:
            pickle.dump((self.docstore, self.id_key, self.child_splitter), f)

    @classmethod
    def load_local(
        cls,
        folder_path: str,
        embeddings: Embeddings,
        index_name: str = "index",
        *,
        client: Any,
        collection_name: str,
        vectorstore: Any | None = None,
    ) -> "AvailableRetriever":
        """Rebuild the retriever from an existing Qdrant collection + docstore pickle.

        Args:
            folder_path: folder holding the ``<index_name>_ret.pkl`` docstore pickle.
            embeddings: embeddings used to encode queries.
            index_name: base name of the docstore pickle file.
            client: a connected ``QdrantClient``.
            collection_name: the Qdrant collection holding the child-chunk vectors.
            vectorstore: optional pre-built vectorstore (used for hybrid Qdrant).
        """
        path = Path(folder_path)
        if vectorstore is None:
            vectorstore = Qdrant(
                client=client,
                collection_name=collection_name,
                embeddings=embeddings,
            )
        with open(path / f"{index_name}_ret.pkl", "rb") as f:
            docstore, id_key, child_splitter = pickle.load(f)
        return cls(
            vectorstore=vectorstore,
            docstore=docstore,
            id_key=id_key,
            child_splitter=child_splitter,
        )
