# Vector

*Part of AVAILABLE 2.0 · Pipeline STEP 4 (`pipeline/index.py --vectordb`, `dc_available/RAG/`)*

---

## Aim

Given a natural-language question, find the passages that are **about the same thing** — quickly, and
across a large corpus. This is the "first eye" of retrieval: matching by **meaning** and by
**keywords**, carried over and upgraded from the 1.0 baseline.

---

## Embedding model (BGE-M3)

An **embedding** turns a piece of text into a list of numbers (a vector) so that texts with similar
meaning end up close together. The question and every chunk are embedded the same way, and
"relevant" becomes "nearby".

The model is **`BAAI/bge-m3`**. What makes it useful here is that it produces **two** representations
of each text at once:

- a **dense** vector (1024 numbers) — captures the **meaning** (semantic similarity), so a question
  matches a passage even when the words differ;
- a **sparse** vector — captures the **exact words** (keyword/lexical match), so specific codes and
  terms (e.g. an SAP transaction code) still match precisely.

It runs on the GPU when one is available, otherwise the CPU.

> AVAILABLE 2.0 changed the embedding model (and the store below). The 1.0 baseline used a single
> dense model with FAISS; 2.0 adds the sparse/keyword signal on top of the dense one.

---

## Qdrant (the vector store)

**Qdrant** is the vector database that stores all the chunk vectors and answers similarity searches.

- Chunks are kept in a **collection** (named `available`); dense vectors are compared by **cosine**
  similarity, and the sparse vectors have their own index.
- Qdrant runs as a service (by default at `http://localhost:6333`), so the app and a Docker
  container can point at the same store.
- It is used in **hybrid retrieval mode**: for each query Qdrant searches **both** the dense and the
  sparse vectors and **fuses** the two result lists automatically. So the vector eye already blends
  *semantic* and *keyword* matching in one search.

> This dense + sparse blend is **inside** the vector eye. It is not the same as the vector-plus-graph
> "two-eye" combination — that one is described in [hybrid](hybrid.md).

---

## The vector eye (how retrieval works)

Retrieval uses a **parent-document** pattern — search small pieces, but return the larger passage
around them:

- **At index time** — each document is split into small **child chunks**. The child chunks are
  embedded and stored in Qdrant, while the larger **parent** passages are kept in a separate store.
  Searching small chunks makes matching precise; returning the parent gives the LLM enough context.
- **A small index-time trick** — before embedding, each chunk gets a short identity header
  (module · code+title · level), and every document also gets one strong "card" vector. This helps
  thin overview documents still be found by a natural-language question.
- **At query time** — the question is embedded (dense + sparse), Qdrant returns the most similar
  child chunks, and their parent passages are returned as context, grouped per training.

```
question
   │  embed  (dense + sparse, BGE-M3)
   ▼
Qdrant hybrid search   (semantic + keyword, fused inside Qdrant)
   │  most similar child chunks
   ▼
return the parent passages  (grouped per training)   ──►  hybrid (two-eye)
```

> **Two-box rule:** the chunk vectors live in the Qdrant collection, while the parent passages are
> saved in a small file (a pickle). These two must always be built and loaded **together** — the app
> checks for both before serving.

---

## How this fits into AVAILABLE 2.0

The vector store is built in **STEP 4**, next to the graph:

```
ingest → sync → pre-process → index → serve
                                 │
                                 ├─ vector store  (Qdrant)   ← this doc
                                 └─ graph
```

At query time the vector eye runs alongside the [graph](graph.md) eye, and their results are combined
in the [hybrid](hybrid.md) step. Compared with 1.0 this leg was upgraded from FAISS to Qdrant, moved
to the BGE-M3 model, and gained the sparse keyword signal.

---

## How to run

```bash
python -m pipeline.index --vectordb    # build/refresh the Qdrant collection (add --refresh to rebuild)
```

Configuration:

| Variable | Purpose |
|---|---|
| `QDRANT_URL` | where the Qdrant service runs (default `http://localhost:6333`) |
| `QDRANT_COLLECTION` | the collection name (default `available`) |
| embedding model | `BAAI/bge-m3` (dense + sparse) |
```
