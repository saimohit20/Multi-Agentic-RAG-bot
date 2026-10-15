# DC AVAILABLE

DC AVAILABLE is a **RAG** (Retrieval-Augmented Generation) chatbot and learning assistant built on
the **Bosch Rexroth Training Academy** materials — PDFs, PowerPoints, videos, emails, and
Confluence/Docupedia pages. It answers a user's question, points to the exact training material the
answer comes from, and attaches a **trust score** to each source.

This document describes **what the project contains**. Each part is explained briefly here; detailed
per-topic documentation will be added to this `documentation/` folder later.

---

## 1. Introduction

DC AVAILABLE helps users learn from the Bosch Rexroth training corpus. Instead of searching through
folders of slides and videos, a user asks a question in natural language and receives a written
answer together with references to the source material. The content is multi-modal (text, slide
images, and video transcripts) and every answer can be traced back to the document it came from.

The application offers two ways to interact with the same knowledge base:

- **Assistant** — a guided tutor that first understands the topic and the user's level, then
  recommends suitable training material.
- **Q&A** — a single question in, a single sourced answer out.

The project has two generations:

- **AVAILABLE 1.0** — the original system built by the Bosch team.
- **AVAILABLE 2.0** — the improvements added in this thesis work, described below.

---

## 2. AVAILABLE 1.0 — the baseline (Bosch team)

The first version was a working RAG chatbot with **pure semantic search**:

- Training documents were converted to text, embedded, and stored in a **FAISS** vector index.
- A user question was embedded and matched against that index by semantic similarity.
- The most similar passages were passed to an LLM, which wrote an answer and cited its sources.
- Document trustworthiness could be set **manually** through an optional `trust.json` file per
  training folder (`rating: unchecked / trusted / outdated / wrong`).

This established the core idea and the user interface, and served as the foundation for the thesis.

---

## 3. Identified improvements in AVAILABLE 1.0

While functional, the baseline had several limitations:

- **modularity** — ingestion, indexing, and serving were run as one-shot scripts; individual
  stages could not be run, re-run, or automated independently.
- **retrieval approaches** — retrieval relied entirely on vector (semantic) similarity, with
  no alternative way to connect related content.
- **observability** — there was no way to trace, inspect, or monitor what happened inside the
  retrieval and answering pipeline.
- **evaluation** — answer and retrieval quality were not measured in any systematic way.
- **automated trust score** — trust existed only as manual `trust.json` ratings; nothing computed
  trustworthiness from the documents themselves.

---

## 4. AVAILABLE 2.0 — the thesis improvements

AVAILABLE 2.0 addresses each of those gaps:

- **Modularity** — a clear, staged **pipeline** (`ingest → sync → pre-process → index → serve`),
  where every stage can be run on its own or chained automatically.
- **Two-eye retrieval** — a second **graph** retrieval leg was added next to the vector leg, and the
  two are combined into one **hybrid** result.
- **New vector backend** — the vector store was migrated from **FAISS to Qdrant**, the embedding
  model was updated, and a cross-encoder **reranker** was added.
- **Observability** — **Opik** tracing was added across the whole pipeline, together with an
  LLM-based judge that scores each answer.
- **Automated trust score** — a document trust score is now computed automatically from document
  metadata and the knowledge graph.
- **Evaluation** — an evaluation harness measures retrieval and answer quality (**RAGAS** plus a
  deterministic source-hit metric).

---

## Architecture Flow

End to end, data and queries move through the following stages.

### 1. SharePoint integration
The training material lives in **SharePoint**. A Microsoft Graph client handles authentication,
listing, downloading, and uploading of files. It also resolves shortcuts — `.lnk` links to the real
target file and `.url` links to their Confluence/Docupedia page. Two SharePoint locations are used:
a *grownfield* location (raw source material) and the *AVAILABLE* location (cleaned and organized).

### 2. Ingestion process (ETL pipeline)
*(ETL = Extract, Transform, Load.)* Raw material is fetched from the grownfield SharePoint,
**classified** by Topic / Level / Training, and uploaded in an organized structure to the AVAILABLE
SharePoint. This is the entry point that turns scattered source files into a consistent library.

### 3. Sync
The organized SharePoint content is pulled down into a **local mirror**. A manifest file records
what has already been downloaded, so each run only transfers files that are new or changed
(a *delta* sync).

### 4. Pre-process
Each raw file is turned into text **summaries** and **chunks**. Format-specific extractors handle
PDF, PowerPoint/Office, video, email, and HTML. Videos are processed into a transcript plus slide
screenshots. The step also produces per-file and per-training summaries and builds a browsable
**handbook** of the corpus.

### 5. Vector
The text chunks are embedded (`BAAI/bge-m3`, dense + sparse) and stored in the **Qdrant** vector store.
Retrieval uses a parent-document pattern: small child chunks are searched for similarity, and the
larger parent passage is returned as context. This is the semantic-search leg carried over and
upgraded from 1.0.

### 6. Graph
Alongside the vectors, a **knowledge graph** is built from the same content. It has a structural
layer (`Topic → Subtopic → Level → Training → Document → Chunk`) and a semantic layer of **entities**
and **typed relations**, where each chunk is linked to the entities it mentions. Two graph retrieval
approaches exist:

- **Entity hop** *(experiment)* — the query is matched to seed entities, then takes a single hop to
  the chunks that mention them. Specific (rare) entities are weighted more than generic ones (IDF
  weighting).
- **PPR (Personalized PageRank)** — weight is placed on the query's seed entities and allowed to
  flow across the whole graph through its connections; the chunks where the most weight settles are
  retrieved.

### 7. Hybrid
The vector leg and the graph leg are combined into a single result — the **two-eye** approach:

- **Union** — candidate passages from both legs are merged into one pool.
- **Reranker** — a cross-encoder model (`bge-reranker-v2-m3`) re-scores the merged pool so the most
  relevant passages end up on top.

### 8. Trust score
An automated, query-independent trust score is computed for each document at ingestion time, in
three parts:

- **Prior (metadata)** — a base score from document metadata: freshness, authority
  (internal vs. external), file format, and an LLM rating of the document's content quality.
- **TrustRank (graph)** — the prior is propagated over a document-to-document graph, where two
  documents are connected when they share specific (high-IDF) entities. Trust flows from reliable
  sources to the documents that corroborate their claims (the same PageRank idea as PPR, applied to
  trust).
- **Trust score (final)** — the prior and the propagated TrustRank value are blended into one final
  score, which is used during re-ranking to gently favor more trustworthy sources.

### 9. Opik
**Opik** provides observability. Every retrieval and generation step is traced as a hierarchy of
spans (a step), traces (a full request), and threads (a session). An LLM-based judge additionally
scores each answer for qualities such as hallucination, answer relevance, and context precision.

### 10. Evaluation
An evaluation harness measures quality. An LLM generates a question–answer dataset from the ingested
material; each question is run through the real pipeline; and the results are scored with **RAGAS**
metrics plus a deterministic **source-hit** metric that checks whether the correct source document
was retrieved.

---

## Pipeline commands

**Main pipeline** (run from the `dc-available` directory):

```bash
python -m pipeline.ingest              # grownfield SharePoint → AVAILABLE SharePoint
python -m pipeline.sync                # AVAILABLE SharePoint → local mirror (delta)
python -m pipeline.preprocess          # raw files → summaries + chunks + handbook
python -m pipeline.index               # build Qdrant vectors + knowledge graph
python -m pipeline.serve --mode qa     # launch the app (qa | assistant)

python -m pipeline.run_pipeline        # unattended: sync → pre-process → index
```

**Index in phases** (optional — no flag runs all):

```bash
python -m pipeline.index --vectordb    # vector store only
python -m pipeline.index --graph       # structural graph + semantic batches
python -m pipeline.index --semantic    # LLM entity / relation extraction
```

**Evaluation**:

```bash
poetry run python -m evaluation.available.run_pipeline
```

---

## Tech stack

| Concern | Choice |
|---|---|
| Language / packaging | Python, Poetry |
| Vector store | Qdrant (was FAISS in 1.0) |
| Embeddings | `BAAI/bge-m3` |
| Reranker | `bge-reranker-v2-m3` cross-encoder |
| LLM | Azure OpenAI (`gpt-4o` family) |
| RAG framework | LangChain |
| Assistant agents | Microsoft AutoGen |
| Knowledge graph | custom graph + Personalized PageRank |
| User interface | Streamlit + MkDocs handbook |
| Observability | Opik + LLM-judge scores |
| Evaluation | RAGAS + source-hit metric |
| Data source | SharePoint / Microsoft Graph, Confluence/Docupedia |

---

## Project structure

A high-level skeleton (folders, not files):

```
dc-available/
├─ dc_available/            # core application
│  ├─ config/               # paths + Azure OpenAI configuration
│  ├─ summarization/        # raw files → summaries + chunks (incl. VideoIE)
│  ├─ common/               # data loading + artifact-path helpers
│  ├─ RAG/                  # vector retrieval + question answering
│  ├─ graph/                # knowledge graph + graph retrieval (entity hop, PPR)
│  ├─ trust_score/          # metadata prior → TrustRank → final trust
│  ├─ agents/               # AutoGen assistant (curriculum tutor)
│  ├─ frontend/             # Streamlit multipage UI
│  └─ data/                 # local mirror + generated artifacts
│
├─ pipeline/                # ingest → sync → pre-process → index → serve (+ auto)
├─ evaluation/              # dataset generation → predictions → RAGAS scoring
└─ documentation/           # project documentation (this folder)
```

> Detailed documentation for each step above will be added as separate files in this folder.
