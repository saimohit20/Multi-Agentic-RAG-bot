"""Step 4 — build the ISOLATED test index (40 docs: 20 good + 20 twins).

Its own Qdrant collection (`available_trust_test`) + docstore, so the production
`available` index is untouched. Each doc is indexed as a short summary text with
`part_of_document = <doc_id>` (g01 / g01_twin) so `trust_for` resolves it against
the test trust tables written by graph_twins.

Reuses the exact hybrid (dense+sparse BGE-M3) setup from `get_retriever`, but adds
our own Documents instead of loading from disk.

RUN (from the dc-available directory, Qdrant must be up):
    python -m evaluation.trust_score.rag_ab.build_index
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain.storage import InMemoryStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from dc_available.config.data_config import QDRANT_URL
from dc_available.RAG.data import AvailableMetaData
from dc_available.RAG.retrieval import AvailableRetriever
from dc_available.RAG.retrieval_helpers import get_embeddings
from dc_available.trust_score.components.score_content import _read_text
from evaluation.trust_score.rag_ab.config import (
    DOCS_FILE,
    INDEX_CACHE,
    TEST_COLLECTION,
    TWINS_DIR,
)

_MD = AvailableMetaData


def _make_documents() -> list[Document]:
    rows = [json.loads(line) for line in DOCS_FILE.open(encoding="utf-8")]
    out: list[Document] = []
    for d in rows:
        did = d["doc_id"]
        good_text = _read_text(Path(d["summary_path"]))
        twin_text = _read_text(TWINS_DIR / f"{did}_twin.txt")
        for doc_key, text in ((did, good_text), (f"{did}_twin", twin_text)):
            meta = {
                _MD.ID.value: doc_key,
                _MD.DOCUMENT_TYPE.value: "summary",
                _MD.PART_OF_DOCUMENT.value: doc_key,     # trust_for key
                _MD.STORAGE_PATH.value: f"rag_ab/{doc_key}.txt",
                _MD.FILE_TYPE.value: "txt",
            }
            out.append(Document(page_content=text, metadata=meta))
    return out


def run() -> None:
    documents = _make_documents()
    n = len(documents)
    print(f"building '{TEST_COLLECTION}' — {n} docs ({n // 2} good + {n // 2} twin)")

    embeddings = get_embeddings()
    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)
    client.create_collection(
        collection_name=TEST_COLLECTION,
        vectors_config=VectorParams(size=embeddings.embedding_dim,
                                    distance=Distance.COSINE, on_disk=False),
        sparse_vectors_config={"sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=True))},
    )
    vectorstore = QdrantVectorStore(
        client=client, collection_name=TEST_COLLECTION, embedding=embeddings,
        sparse_embedding=embeddings.sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID, sparse_vector_name="sparse",
    )
    retriever = AvailableRetriever(
        vectorstore=vectorstore, docstore=InMemoryStore(),
        child_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=200),
    )
    retriever.add_documents(documents)

    INDEX_CACHE.mkdir(parents=True, exist_ok=True)
    retriever.save_local(str(INDEX_CACHE))
    print(f"indexed {n} docs -> collection '{TEST_COLLECTION}', docstore -> {INDEX_CACHE}")


if __name__ == "__main__":
    run()
