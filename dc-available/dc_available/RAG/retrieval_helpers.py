from __future__ import annotations

import os
import shutil
from pathlib import Path

from langchain.storage import InMemoryStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from dc_available.RAG.hybrid_embeddings import HybridBGEM3Embeddings

from dc_available.common.helpers import join_paths
from dc_available import DATA_DIR

from dc_available.RAG.retrieval import (
    AvailableRetriever,
)
from dc_available.RAG.data import (
    AvailableDataloader,
    AvailableMetaData,
)
from dc_available.config import data_config
from dc_available.config.data_config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_DATA_PATH,
    DEFAULT_CACHE_PATH,
    DEFAULT_IMAGE_PATH,
    DEFAULT_IMAGE_ARTIFACTS_PATH,
    MAIN_FOLDER,
    QDRANT_URL,
    QDRANT_COLLECTION,
    QDRANT_RICH_COLLECTION,
)

retriever: AvailableRetriever = None
embeddings: HybridBGEM3Embeddings = None
# RICH feature: its own retriever singleton, kept separate from the QA `retriever`
# above so building/loading RICH never disturbs the QA collection.
rich_retriever: AvailableRetriever = None


def get_embeddings(
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> HybridBGEM3Embeddings:
    """Get or create hybrid embeddings using BGE-M3 model."""
    global embeddings
    if embeddings is None:
        from time import perf_counter as _pc
        print("[Retrieval] Load Hybrid BGE-M3 Embeddings.")
        _t = _pc()
        embeddings = HybridBGEM3Embeddings(model_name=embedding_model_name)
        print(f"[LAT] embedding model load={_pc()-_t:.2f}s", flush=True)
    return embeddings


def get_retriever(
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    docs_path: str | Path = DEFAULT_DATA_PATH,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    refresh : bool = False
) -> AvailableRetriever:
    global embeddings, retriever
    if embeddings is None:
        embeddings = get_embeddings(embedding_model_name)

    if retriever is None or refresh:
        client = QdrantClient(url=QDRANT_URL)
        # The vectors (Qdrant collection) and the parent docstore (pickle) are the
        # "two boxes" that must always agree. Only load if BOTH exist; otherwise
        # rebuild both together.
        pickle_ok = (Path(cache_path) / "index_ret.pkl").exists()
        collection_ok = client.collection_exists(QDRANT_COLLECTION)

        if refresh or not (pickle_ok and collection_ok):
            print("[Retrieval] Create Qdrant collection with hybrid vectors (dense + sparse).")
            embed_dim = embeddings.embedding_dim  # BGE-M3 = 1024 dims

            # Create collection with both dense and sparse vectors.
            if client.collection_exists(QDRANT_COLLECTION):
                client.delete_collection(QDRANT_COLLECTION)
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=embed_dim, 
                    distance=Distance.COSINE,
                    on_disk=False
                ),
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=True)
                    )
                }
            )
            
            # Use QdrantVectorStore with HYBRID retrieval mode
            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=QDRANT_COLLECTION,
                embedding=embeddings,
                sparse_embedding=embeddings.sparse_embeddings,
                retrieval_mode=RetrievalMode.HYBRID,
                sparse_vector_name="sparse",
            )
            
            documents = AvailableDataloader(docs_path).load()

            print(f"[Retrieval] Found {len(documents)} documents.")
            retriever = AvailableRetriever(
                vectorstore=vectorstore,
                docstore=InMemoryStore(),
                child_splitter=CharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                ),
            )
            retriever.add_documents(documents)
            print("[Retrieval] Save docstore (vectors persist in Qdrant).")
            retriever.save_local(cache_path)
        else:
            print("[Retrieval] Load from Qdrant + docstore.")
            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=QDRANT_COLLECTION,
                embedding=embeddings,
                sparse_embedding=embeddings.sparse_embeddings,
                retrieval_mode=RetrievalMode.HYBRID,
                sparse_vector_name="sparse",
            )
            retriever = AvailableRetriever.load_local(
                cache_path,
                embeddings,
                client=client,
                collection_name=QDRANT_COLLECTION,
                vectorstore=vectorstore
            )
    else:
        print("[Retrieval] Reuse from cache")
    return retriever


def get_rich_retriever(refresh: bool = False) -> AvailableRetriever:
    """Build or load the RICH retriever (plain RAG over the separate RICH collection).

    Mirrors get_retriever() but targets QDRANT_RICH_COLLECTION and the RICH artifact
    paths, and keeps its own module-level singleton so the QA retriever/collection is
    never touched. Paths are resolved WITHOUT switching the global data_config (QA and
    RICH share one process). The child-chunk vectors persist in Qdrant; the parent
    docstore is pickled under the RICH index path.
    """
    global embeddings, rich_retriever
    if embeddings is None:
        embeddings = get_embeddings()

    if rich_retriever is not None and not refresh:
        print("[RICH] Reuse from cache")
        return rich_retriever

    paths = data_config.source_artifact_paths("rich")
    docs_path = paths["ingestion"]
    cache_path = paths["index"]

    client = QdrantClient(url=QDRANT_URL)
    pickle_ok = (Path(cache_path) / "index_ret.pkl").exists()
    collection_ok = client.collection_exists(QDRANT_RICH_COLLECTION)

    if refresh or not (pickle_ok and collection_ok):
        print(f"[RICH] Create Qdrant collection '{QDRANT_RICH_COLLECTION}' (dense + sparse).")
        embed_dim = embeddings.embedding_dim

        if client.collection_exists(QDRANT_RICH_COLLECTION):
            client.delete_collection(QDRANT_RICH_COLLECTION)
        client.create_collection(
            collection_name=QDRANT_RICH_COLLECTION,
            vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE, on_disk=False),
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=True))
            },
        )
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_RICH_COLLECTION,
            embedding=embeddings,
            sparse_embedding=embeddings.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name="sparse",
        )

        documents = AvailableDataloader(docs_path).load()
        print(f"[RICH] Found {len(documents)} documents under {docs_path}.")
        rich_retriever = AvailableRetriever(
            vectorstore=vectorstore,
            docstore=InMemoryStore(),
            child_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=200),
        )
        rich_retriever.add_documents(documents)
        print("[RICH] Save docstore (vectors persist in Qdrant).")
        rich_retriever.save_local(cache_path)
    else:
        print("[RICH] Load from Qdrant + docstore.")
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_RICH_COLLECTION,
            embedding=embeddings,
            sparse_embedding=embeddings.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name="sparse",
        )
        rich_retriever = AvailableRetriever.load_local(
            cache_path,
            embeddings,
            client=client,
            collection_name=QDRANT_RICH_COLLECTION,
            vectorstore=vectorstore,
        )
    return rich_retriever


def prepare_path(document: Document, meta_data_field: str) -> Path:
    path = document.metadata[meta_data_field]

    # Adjust the new folder structure (with expert levels)
    if "DigitalTransformation_Training" in path:
        path = path.replace(
            "DigitalTransformation_Training", "DigitalTransformation/Basic/Training"
        )
    elif "Hydraulics_Training" in path:
        path = path.replace("Hydraulics_Training", "Hydraulics/Basic/Training")

    # Adjust the new folder structure (with main_data_test)
    if "aim205-actionable-llms/aim205_actionable_llms/use_cases" in path:
        path = path.replace(
            "aim205-actionable-llms/aim205_actionable_llms/use_cases",
            "aim205_actionable_llms/main_data_test",
        )

    path = Path(path)
    if path.exists():
        # Already a valid local path (e.g. the 'available' source stores files
        # under dc_available/data/... which is not under DATA_DIR/resources).
        adjusted_path = path
    else:
        # Legacy layout: rebase a path stored with a different root onto DATA_DIR.
        try:
            adjusted_path = join_paths(path, DATA_DIR, DATA_DIR.name)
        except ValueError:
            adjusted_path = path
    adjusted_path = adjusted_path.as_posix()
    if f"/{DATA_DIR.name}/data/" in adjusted_path and not (DATA_DIR / "data").exists():
        adjusted_path = adjusted_path.replace("/data/", "/")
    adjusted_path = Path(adjusted_path)
    return adjusted_path


def _long_path(p: str | Path) -> str:
    """On Windows, prefix with \\\\?\\ to bypass the 260-char MAX_PATH limit."""
    s = os.path.abspath(os.fspath(Path(p)))
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


def get_image_cache_name(img_path: Path) -> str:
    # Images live under two roots: raw/pptx images under MAIN_FOLDER
    # (e.g. local_sharepoint/...__pptx_images/) and video slide images under
    # Extracted_resources/. Anchor on whichever is present so both are handled.
    parts = img_path.parts
    for anchor in (MAIN_FOLDER, "Extracted_resources"):
        if anchor in parts:
            return "__".join(parts[parts.index(anchor) + 1 :])
    # Last resort: keep a unique-ish name instead of crashing.
    return "__".join(parts[-3:])


def prepare_image_cache(
    retriever: AvailableRetriever = None,
    image_cache_path: str | Path = DEFAULT_IMAGE_PATH,
):
    if retriever is None:
        retriever = get_retriever()
    client = retriever.vectorstore.client
    collection_name = retriever.vectorstore.collection_name

    print("[Setup] Copy images to cache directory")
    image_cache_path = Path(image_cache_path)
    image_cache_path.mkdir(parents=True, exist_ok=True)

    # Scroll every point in the collection (Qdrant has no in-memory docstore._dict).
    # The community Qdrant wrapper stores each chunk's metadata under the "metadata"
    # payload key.
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        for point in points:
            meta = (point.payload or {}).get("metadata", {})
            if AvailableMetaData.IMAGE_PATH in meta:
                img_path = Path(meta[AvailableMetaData.IMAGE_PATH])
                # The stored image path is already absolute/local for this source.
                # Only rebase (legacy "resources"-rooted layout) when the raw path
                # is missing; join_paths raises if the anchor folder isn't present.
                if not img_path.exists():
                    try:
                        img_path = join_paths(img_path, DATA_DIR, DATA_DIR.name)
                    except ValueError:
                        pass
                new_filename = get_image_cache_name(img_path)
                target_path = image_cache_path / new_filename

                if img_path.exists() and not target_path.exists():
                    try:
                        shutil.copy2(_long_path(img_path), _long_path(target_path))
                    except OSError as exc:
                        print(f"[Setup] Skipped image {new_filename}: {exc}")
        if offset is None:
            break
