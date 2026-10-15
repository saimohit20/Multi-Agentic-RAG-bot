"""Hybrid BGE-M3 embeddings for dense + sparse retrieval."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import torch
from FlagEmbedding import BGEM3FlagModel
from langchain_core.embeddings import Embeddings
from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector


class BGEM3SparseEmbeddings(SparseEmbeddings):
    """Sparse adapter for Qdrant hybrid retrieval."""

    def __init__(self, parent: "HybridBGEM3Embeddings"):
        self._parent = parent

    def embed_documents(self, texts: List[str]) -> List[SparseVector]:
        return self._parent.get_sparse_embeddings(texts)

    def embed_query(self, text: str) -> SparseVector:
        return self._parent.get_query_sparse_embedding(text)


class HybridBGEM3Embeddings(Embeddings):
    """Dense embeddings + sparse adapter backed by BGE-M3."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[HybridEmbeddings] Loading {model_name} on {self.device}")
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=(self.device == "cuda"),
            device=self.device,
        )

        # Cache embedding dimension
        self._embedding_dim: int | None = None
        self.sparse_embeddings = BGEM3SparseEmbeddings(self)

        # Per-query encode cache. One question is embedded by several search calls
        # in a single request (chunk hybrid + entity hybrid + entity dense). BGE-M3
        # is a heavy CPU model, so we memoize the last few queries' dense/sparse
        # vectors and reuse them across those calls. Identical query + model =>
        # identical vectors => identical results, just no repeated forward passes.
        self._q_dense_cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._q_sparse_cache: "OrderedDict[str, SparseVector]" = OrderedDict()
        self._q_cache_max = 16

    def _encode(self, texts: List[str], return_dense: bool, return_sparse: bool) -> dict:
        print(f"[ENCODE] real BGE-M3 pass  n={len(texts)} dense={return_dense} "
              f"sparse={return_sparse} first48={texts[0][:48]!r}", flush=True)
        return self.model.encode(
            texts,
            batch_size=32,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )

    @staticmethod
    def _dense_rows(payload: dict) -> List[List[float]]:
        dense = payload.get("dense_vecs")
        if dense is None:
            return []
        if hasattr(dense, "tolist"):
            dense = dense.tolist()
        return dense

    @staticmethod
    def _weights_to_sparse_vector(weights: Dict[str, float]) -> SparseVector:
        pairs = []
        for key, value in weights.items():
            try:
                pairs.append((int(key), float(value)))
            except (TypeError, ValueError):
                continue
        pairs.sort(key=lambda item: item[0])
        return SparseVector(
            indices=[idx for idx, _ in pairs],
            values=[val for _, val in pairs],
        )

    @property
    def embedding_dim(self) -> int:
        """Get dense embedding dimension (1024 for BGE-M3)."""
        if self._embedding_dim is None:
            rows = self._dense_rows(self._encode(["test"], return_dense=True, return_sparse=False))
            if not rows:
                raise RuntimeError("Failed to infer BGE-M3 embedding dimension.")
            self._embedding_dim = len(rows[0])
        return self._embedding_dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = self._encode(texts, return_dense=True, return_sparse=False)
        return self._dense_rows(payload)

    def _cache_query_embeddings(self, text: str) -> None:
        dense_rows, sparse_rows = self.get_dense_and_sparse([text])
        dense_vector = dense_rows[0] if dense_rows else []
        sparse_vector = sparse_rows[0] if sparse_rows else SparseVector(indices=[], values=[])

        self._q_dense_cache[text] = dense_vector
        self._q_sparse_cache[text] = sparse_vector
        if len(self._q_dense_cache) > self._q_cache_max:
            self._q_dense_cache.popitem(last=False)
        if len(self._q_sparse_cache) > self._q_cache_max:
            self._q_sparse_cache.popitem(last=False)

    def embed_query(self, text: str) -> List[float]:
        cached = self._q_dense_cache.get(text)
        if cached is not None:
            self._q_dense_cache.move_to_end(text)
            return cached
        self._cache_query_embeddings(text)
        return self._q_dense_cache[text]

    def get_sparse_embeddings(self, texts: List[str]) -> List[SparseVector]:
        payload = self._encode(texts, return_dense=False, return_sparse=True)
        lexical_weights = payload.get("lexical_weights") or []
        return [self._weights_to_sparse_vector(weights) for weights in lexical_weights]

    def get_query_sparse_embedding(self, text: str) -> SparseVector:
        cached = self._q_sparse_cache.get(text)
        if cached is not None:
            self._q_sparse_cache.move_to_end(text)
            return cached
        self._cache_query_embeddings(text)
        return self._q_sparse_cache[text]

    def get_dense_and_sparse(self, texts: List[str]) -> tuple[List[List[float]], List[SparseVector]]:
        payload = self._encode(texts, return_dense=True, return_sparse=True)
        dense_rows = self._dense_rows(payload)
        sparse_rows = [self._weights_to_sparse_vector(weights) for weights in (payload.get("lexical_weights") or [])]
        return dense_rows, sparse_rows
