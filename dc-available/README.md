<!---

	Copyright (c) 2024 Robert Bosch GmbH and its subsidiaries.

-->

# DC AVAILABLE
AVAILABLE is an innovative chatbot designed to assist users in understanding and navigating
the complex world of Bosch Rexroth Training materials. This cutting-edge RAG chatbot is based
on a wealth of real-world training materials, including text documents, PowerPoints, videos,
and more, sourced directly from the prestigious Bosch Rexroth Training Academy.


## Overview
The primary goal of AVAILABLE is to serve as a learning helper, providing users with comprehensive
and accurate answers to their queries while also offering references to the relevant training materials.
Whether users are seeking information on hydraulic systems, automation technologies, or any other
topic covered in the Bosch Rexroth Training Academy, AVAILABLE is equipped to provide valuable insights
and guidance.

One of the key features of AVAILABLE is its ability to understand the specific issues and challenges
faced by users. By leveraging advanced natural language processing capabilities, the chatbot can
effectively interpret user queries and provide tailored responses that address the unique needs
of each individual. This personalized approach ensures that users receive the most relevant and
helpful information, ultimately enhancing their learning experience.

In addition to providing direct answers to user queries, AVAILABLE also offers references to the
training materials from which its knowledge is derived. This not only helps users to understand the
reasoning behind the chatbot's responses but also encourages them to explore the training materials in
greater depth, further enriching their understanding of the subject matter.


## Project Structure

```
dc_available/        Core application package
├── RAG/             Retrieval → fusion → rerank → answer generation
├── graph/           Knowledge-graph build & graph retrieval (entity / PPR)
├── summarization/   Extract & summarize source docs (pdf, pptx, video, msg, xlsx …)
├── trust_score/     Document reliability scoring used in ranking
├── agents/          AutoGen multi-agent curriculum / assistant orchestration
├── frontend/        Streamlit UI (Assistant, QA, upload pages)
├── config/          LLM (Azure OpenAI) & data / embedding configuration
├── common/          Shared helpers (data loading, paths, video extraction)
└── data/            Local data & generated artifacts (gitignored, not pushed)
pipeline/            Offline build stages: sync → preprocess → index → trust_score → serve
evaluation/          RAGAS + retrieval / graph evaluation, datasets, results
scripts/             Utilities (e.g. graph visualization)
deployment/          Deployment artifacts
├── docker/          local Docker image build (Dockerfile + Taskfile)
└── kubernetes/      production Kubernetes / AKS manifests
docs/                Setup guides (Qdrant, Opik)
```

## Architecture

AVAILABLE has two flows: an **offline build** that turns raw training material into a searchable
index + knowledge graph, and a **per-question query** flow that answers the user.

```
OFFLINE BUILD  (pipeline/run_pipeline.py)
───────────────────────────────────────────────────────────────────
  SharePoint ──sync──▶  data/local_sharepoint/     (raw mirror)
                              │
                         preprocess                (summarize pdf / pptx / video / msg / xlsx → chunks)
                              │
                              ▼
                           index ──────▶  Qdrant          (dense + sparse BGE-M3 vectors)
                              │      └───▶  Knowledge graph (topic→level→training→doc→chunk + entities)
                              ▼
                        trust_score               (per-document reliability)


QUERY  (per question)
───────────────────────────────────────────────────────────────────
        User question
              │
      ┌───────┴────────────────────────┐
      ▼                                 ▼
 Vector retrieval                 Graph retrieval
   (Qdrant)                       (entity 1-hop / PPR)
      └───────────────┬───────────────┘
                      ▼
                 Union fusion
                      ▼
             Cross-encoder rerank        (BAAI/bge-reranker-v2-m3)
                      ▼
               Trust weighting
                      ▼
            Azure OpenAI answer  ─────▶  Streamlit UI
                      │
                      └── every stage traced by Opik
```

## Getting Started

### 1. Install dependencies

```
poetry install
```

### 2. Add your data

Place your training material under the local data folder used by the default `available` source:

```
dc_available/data/local_sharepoint/
└── <Topic>/                 e.g. DigitalTransformation
    └── <Level>/             at least one; use "Basic" if you have no levels
        └── <Training>/      one folder per training / document collection
            ├── document.pdf
            ├── presentation.pptx
            └── video.mp4
```

Rules:
- At least one **Level** folder must exist under each Topic (use `Basic` if you don't need levels).
- Any number of documents can go inside a Training folder; **subfolders are not searched**.
- Trainings are discovered anywhere under `local_sharepoint/`, so wrapper folders are optional.
- `dc_available/data/` is **gitignored** — your data and generated artifacts stay local and are never pushed.

> Document trust is now computed automatically by the pipeline (`pipeline/trust_score.py`) —
> no manual `trust.json` file is required.

### 3. Configure

All data/path and retrieval settings live in **`dc_available/config/data_config.py`**.
LLM (Azure OpenAI) settings live in `dc_available/config/llm_config.py`.

The active data source is the `available` entry of `_data_config` — its folder path is defined here:

```python
# dc_available/config/data_config.py
"available": {
    "main_folder": "local_sharepoint",                       # raw data folder name
    "artifact_folder": "resources",                          # Artifacts/ + Extracted_resources/ layout
    "base_dir": DATA_DIR.parent / "dc_available" / "data",   # ← the data root you fill in step 2
},
```

- To change **where data lives**, edit `base_dir` / `main_folder` above.
- To point at a **different Qdrant**, set `QDRANT_URL` (default `http://localhost:6333`).
- To switch data source, set `DEFAULT_DATA_SOURCE` or run with `AVAILABLE_APP_SOURCE=<key>`.

### 4. Start Qdrant

The vector store must be running before you build the index or serve the app.
See **[docs/qdrant.md](docs/qdrant.md)** for the one-command Docker setup.

### 5. [Optional] Start Opik

Opik traces and evaluates the RAG / agent pipeline. It's optional for local use.
See **[docs/opik.md](docs/opik.md)** for the setup.

## How to Run

### Without Docker

**1. Build the index** (run once, then whenever data changes):

```
poetry run python -m pipeline.run_pipeline --source available
```

This runs the full pipeline in order — `sync → preprocess → index → trust_score` — and stops early
if `sync` reports no changes. To run stages individually (e.g. when iterating locally):

```
poetry run python -m pipeline.preprocess --source available
poetry run python -m pipeline.index --source available
```

*(`trust_score` runs automatically as the last stage of `run_pipeline`.)*

**2. Start the frontend:**

```
poetry run python -m pipeline.serve --source available --mode assistant   # full assistant
poetry run python -m pipeline.serve --source available --mode qa          # Q&A-only page
```

The app opens at **http://localhost:8502**. (Equivalent direct call:
`poetry run streamlit run dc_available/frontend/Assistant.py`.)

### With Docker

The pipeline stages are containerized in **`docker-compose.pipeline.yml`** (shared image, one command
per service). Qdrant is **not** part of this compose file — start it separately (see
[docs/qdrant.md](docs/qdrant.md)) and make sure `QDRANT_URL` points at a reachable instance.

```
# Build the shared image once
docker compose -f docker-compose.pipeline.yml build

# Stage 1 — preprocess (raw data → ingestion artifacts)
docker compose -f docker-compose.pipeline.yml run --rm preprocess

# Stage 2 — index (chunks → Qdrant + knowledge graph)
docker compose -f docker-compose.pipeline.yml run --rm index

# Stage 3 — serve (Streamlit UI on :8502)
docker compose -f docker-compose.pipeline.yml up serve
```

For production Kubernetes / AKS manifests, see `deployment/kubernetes/`.

## About

### Contributors
*The DC AVAILABLE team from AIM-205*
* [Lukas Lange (CR/AIR3)](mailto:lukas.lange@de.bosch.com)
* [Paul Baireuther (CR/AIR2.2)](mailto:paul.baireuther@de.bosch.com)
* [Zhengyu Zhou (CR/RHI2-NA)](mailto:zhengyu.zhou2@us.bosch.com) a.k.a. Jessica

### Useful Links
* Documentation: https://pages.github.boschdevcloud.com/CR/aim205-actionable-llms/
* CRIS: https://rb-cris.bosch.com/cris-web/#/activity/details/115079/overview

### Additional Development
* [Mohamed Amine Guedria (DC/SSD6)](mailto:mohamedamine.guedria@bosch.com)
* [Sai Mohit Vanapala (DC/SSD3)](mailto:vas1ul@bosch.com)

#### Enhancements & Technical Contributions

* Enhanced user interface for more intuitive navigation and improved user experience
* Developed automated data upload system for streamlined content management
* Built custom UI for data reviewing and quality assessment
* Automated data integration processes within the RAG system
* Improved code to effectively process and analyze long video content
* Managed hosting and maintenance of the chatbot on Azure cloud infrastructure

### License

> Copyright (c) 2024 Robert Bosch GmbH and its subsidiaries.
