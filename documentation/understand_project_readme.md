# Understand Project — DC AVAILABLE (What Is Implemented So Far)

> Personal reference document. It explains, in depth, what has been built across the
> three parts of the codebase you care about:
>
> 1. **`dc_available/`** — the core RAG application (ingestion → indexing → retrieval → serving)
> 2. **`pipeline/`** — the orchestration layer that automates data sync + build + serve
> 3. **`evaluation/`** — dataset generation + RAGAS-based quality measurement
>
> Deployment (`deployment/` — `docker/` + `kubernetes/`, Dockerfiles) is intentionally **out of scope** here.

---

## 0. What the product is

**DC AVAILABLE** is a Retrieval-Augmented Generation (RAG) chatbot / learning assistant built on
**Bosch Rexroth Training Academy** materials (PDFs, PowerPoints, videos, emails, Confluence/Docupedia
pages). It answers user questions and points them at the exact source training material, with a
per-source **trust score** and multi-modal handling (text, slide images, video transcripts).

Two user-facing modes exist:
- **Assistant** — a multi-agent "curriculum" tutor (topic → knowledge → teaching-material assessment).
- **Q&A** — a single-shot question-answering page over the same index.

The system is **source-configurable**: one codebase can serve different data collections
(`dc_available`, `test`, `other`, and the SharePoint-backed `available`) selected via the
`AVAILABLE_APP_SOURCE` environment variable or `--source` CLI flag.

### The big picture (data flow)

```
                         ┌───────────────────────── pipeline/ (orchestration) ─────────────────────────┐
 Grownfield SharePoint   │                                                                              │
        │  ingest.py     │   sync.py            preprocess.py           index.py            serve.py     │
        ▼                │   (delta pull)       (summarize)             (build indexes)     (Streamlit)  │
 AVAILABLE SharePoint ───┼─► local mirror ──► ingestion artifacts ──► ┌ Qdrant vectors ┐ ──► RAG app    │
                         │   (raw files)       (index.jsonl,           │ graph (topic   │                │
                         │                      summary.txt)           │  tree+entities)│                │
                         │                                             └ image cache    ┘                │
                         └──────────────────────────────────────────────────────────────────────────────┘
                                                                                │
   dc_available/ implements every stage's actual logic; pipeline/ just calls    ▼
   them in order.                                              evaluation/ measures answer quality (RAGAS)
```

**Key idea:** `pipeline/` adds *no new intelligence*. It is a thin, well-logged orchestrator that
calls functions living in `dc_available/`. All the real work (summarization, embedding, graph
building, retrieval) is in `dc_available/`.

---

## 1. Repository layout (the parts in scope)

```
dc_available/            ← core application package
  config/                data_config.py (paths & sources), llm_config.py (Azure OpenAI)
  summarization/         raw files → text/summary Documents (per-format extractors)
  common/                data loading, artifact-path mirroring, helpers, VideoIE/,
                         sharepoint_connector.py (MS Graph client: auth/list/download/upload/.lnk/.url/classify)
  RAG/                   data.py (metadata schema), retrieval*, qa_functions.py
  graph/  construct/     graph_builder, graph_enricher, semantic_*, entity_index
           retrieval/     entity_hop_retriever, ppr_retriever, ppr_graph
  agents/                AutoGen curriculum agents + Opik stage tracing
  frontend/              Streamlit UI (Assistant, QA, Handbook, Upload, Review, About)
  create_handbook.py     builds an MkDocs "handbook" site from summaries
  curriculum_app.py      the Assistant orchestrator (3-stage agent flow)

pipeline/                orchestration layer (thin wrappers around dc_available)
  ingest.py              STEP 1: grownfield SharePoint → AVAILABLE SharePoint
  sync.py                STEP 2: AVAILABLE SharePoint → local mirror (delta pull)
  preprocess.py          STEP 3: mirror → ingestion artifacts (summaries + handbook)
  index.py               STEP 4: artifacts → Qdrant + graph + semantic layer
  serve.py               STEP 5: launch the Streamlit app (checks index is ready)
  run_pipeline.py        one-shot: sync → (if changed) preprocess → index

evaluation/              quality measurement (two independent pipelines)
  available/             AVAILABLE 2.0 RAG evaluation
    retrieval/           grownfield / hydraulics / synthetic retrieval evals
    generation/          RAGAS generate+score, synthetic gen, answer_correctness
    common/              dataset_loader, dataset_generation/, plots
    run_pipeline.py      consolidated runner (stages: retrieval, generation, all)
  trust_score/           trust-score evaluation
    perturbation/        Experiment 1 — pillar sensitivity
    rag_ab/              Experiment 2 — RAG A/B poisoning
    run_pipeline.py      consolidated runner (both experiments)
  datasets/              shared QA datasets (xlsx)
  results/               timestamped output folders
```

### Recent structural changes (from git history)

The last commits reshaped the system significantly:
- **Vector store migrated FAISS → Qdrant** (`qdrant-client ~1.9.2`, pinned for the LangChain
  community wrapper API).
- **Graph restructured to a per-topic "3-tree" topology** (one graph tree per top-level topic,
  e.g. Hydraulics / DigitalTransformation / GROWNFIELD).
- **Pipeline phase flags added** (`index.py --vectordb/--graph/--semantic`).
- **Knowledge graph + Opik observability** were added on top of the original RAG.

---

## 2. `dc_available/` — the core application

### 2.1 Configuration — `config/`

**[data_config.py](dc_available/config/data_config.py)** is the single source of truth for *where things live*.

- A module-level dict `_data_config` defines **data sources**. Each source has a `main_folder`
  and `handbook_folder`; the `available` source additionally sets `artifact_folder` and `base_dir`
  to enable the *new pipeline layout*.
- `update_data_config(app_name)` re-points every `DEFAULT_*` path global based on the selected
  source. It reads `AVAILABLE_APP_SOURCE` env var (falls back to the argument, default `available`).
- **Two layouts** are supported:
  - *Legacy* (`dc_available`/`test`/`other`): artifacts live under `<base>/artifacts/<main_folder>/…`
    and extracted media inside the data folder.
  - *New pipeline* (`available`): generated files **never** touch the raw data folder —
    ```
    <base>/local_sharepoint/                 raw mirror (read-only for the pipeline)
    <base>/Artifacts/resources/{ingestion,index,graph}
    <base>/Artifacts/handbook/
    <base>/Extracted_resources/resources/{images,video,pptx,pdf}
    ```
- **Qdrant config**: `QDRANT_URL` (default `http://localhost:6333`) and `QDRANT_COLLECTION`
  (default `available`), both overridable by env var (so Docker can point at a `qdrant` service).
- Embedding model default: `BAAI/bge-small-en-v1.5`.
- Semantic batching knobs: `DEFAULT_SEMANTIC_BATCH_SIZE=5`, `…_OVERLAP=1`, `…_MIN_CHARS=120`.

**[llm_config.py](dc_available/config/llm_config.py)** wires up **Azure OpenAI** with
`DefaultAzureCredential` (token-provider auth, no API keys in code). It exposes:
- `get_langchain_llm(...)` → `AzureChatOpenAI` (used for QA, semantic extraction, RAGAS, judge).
- `get_openai_client(...)` → raw Azure OpenAI client.
- `get_autogen_llm_config/…_wrapper(...)` → config for the AutoGen agents.
- Model names: `DEFAULT_MODEL="gpt-4o"` and `DEFAULT_SERVE_MODEL="gpt-4o"` — serving
  Q&A/agents use the `gpt-4o` deployment directly.

### 2.2 Ingestion / summarization — `summarization/`

This turns **raw files into LangChain `Document`s** (chunks + a per-file "meta summary" + a
per-training summary). Entry point: **[trainings.py](dc_available/summarization/trainings.py)**.

- `ACCEPTED_FORMATS = .mp4, .pdf, .pptx, .pptm, .msg, .xlsx, .docx, .html`.
- `summarize_single_file(...)` dispatches by suffix to format-specific extractors:
  - **PDF** → `generate_pdf_or_pptx_summary_for_retrieval` (page texts + summaries, images).
  - **PPTX/PPTM/XLSX/DOCX** → converted to PDF then slide summaries + slide images.
  - **MP4** → audio extraction → `Speech2Text`, slide screenshots + image understanding, merged
    into timestamped chunks (the whole `common/VideoIE/` subsystem).
  - **MSG** (email) → body + attachments (attachments are queued for processing too).
  - **HTML** (Confluence/Docupedia export) → treated as text.
- **Scratch-dir isolation:** raw media is copied into a temp dir; extraction runs there so derived
  artifacts (`__pdf_images`, `__pptx.pdf`, `__mp4_slides`, …) are produced in scratch; then
  `_promote_derived_to_extracted(...)` **moves only derived outputs** into `Extracted_resources/`
  and **rewrites metadata paths**. The raw file and intermediate audio are discarded. This is what
  keeps generated media out of the raw SharePoint mirror.
- **Trust score** is computed per file and per training via `summarization/trust_score.py`
  (`get_trust_score` / `get_meta_trust_score`) and stamped into metadata.
- `summarize_training_for_retrieval(folder)` processes every file in a training folder, then
  builds a **training-level meta summary** `Document` (with a fresh UUID id, `is_meta_summary=True`,
  `link_to_children` to file summaries). It **caches**: if `index.jsonl` + `summary.txt` already
  exist for that training it reloads them (unless `overwrite=True`).
- Outputs per training (written to the *ingestion artifacts* mirror):
  - `index.jsonl` — all chunks + summaries (the canonical retrieval input),
  - `summary.txt` — human-readable training summary,
  - per-file `*__summary.txt` and `*__summary_docs.jsonl`.

`common/artifact_paths.py` provides `mirror_*_to_artifacts(...)` which maps a raw file/training
path to its corresponding artifact path, so ingestion outputs mirror the source tree structure.

### 2.3 Handbook — `create_handbook.py`

After preprocessing, `create_handbook(all_trainings, handbook_path)` assembles the summaries into
an **MkDocs Material** static site (a browsable "handbook" of the training corpus). Triggered by
`preprocess.run_ingestion()` whenever at least one training was (re)built.

### 2.4 Retrieval metadata schema — `RAG/data.py`

**[data.py](dc_available/RAG/data.py)** is shared by retrieval *and* graph code. It defines:
- `AvailableDataloader` — loads `Document`s from `index.jsonl` files (recursively).
- **`AvailableMetaData`** (str Enum) — the canonical metadata field names: `id`, `page_id`,
  `document_type`, `file_type`, `level`, `trust_score`, `part_of_training`, `part_of_document`,
  `is_meta_summary`, `storage_path`, `image_path`, `audio_path`, `video_path`,
  `link_to_{parent,children,next,previous,related}`, `start_time`, `end_time`.
  → The chunk `id` (a UUID) is the **global bridge key** shared between Qdrant, the docstore,
  and the graph.
- **`GraphNodeType`** = Topic, Subtopic, Level, Training, Document, Chunk, Entity.
- **`GraphEdgeType`** = contains, part_of, mentions, related_to, supports, contradicts.
- `GRAPH_METADATA_FIELD_MAP` — documents how each retrieval field maps into a graph role.

### 2.5 Vector retrieval — `RAG/retrieval.py` + `RAG/retrieval_helpers.py`

**Parent-Document Retriever pattern** (LangChain):
- **[retrieval.py](dc_available/RAG/retrieval.py)** — `AvailableRetriever(ParentDocumentRetriever)`.
  - `add_documents` splits each parent doc into **child chunks** (`CharacterTextSplitter`,
    size 1000 / overlap 200); child vectors go to **Qdrant**, parents stay in an in-memory docstore.
  - `_get_relevant_documents` does similarity (or MMR) search, groups hits by `part_of_training`,
    and formats a per-training "overview" block (`format_training_with_paragraphs`) with training
    name/level/description + relevant excerpts + hidden chunk IDs for later source resolution.
  - `save_local` pickles only the **docstore + splitter** (`index_ret.pkl`); the vectors live in
    the Qdrant server. `load_local` rebuilds the retriever from *both* the pickle and the Qdrant
    collection — the "two boxes that must always agree".
- **[retrieval_helpers.py](dc_available/RAG/retrieval_helpers.py)** — module-level singletons.
  - `get_embeddings` → `HuggingFaceEmbeddings` (CUDA if available).
  - `get_retriever(refresh=…)` → connects `QdrantClient`, and **only reuses** an existing index if
    *both* the pickle and the Qdrant collection exist; otherwise it `recreate_collection` (cosine),
    loads all `index.jsonl`, adds documents, and saves the docstore.
  - `prepare_image_cache(...)` → scrolls every point in the Qdrant collection and copies referenced
    slide/page images into the image cache directory (Windows long-path safe).

### 2.6 QA pipeline — `RAG/qa_functions.py`

This is the **heart of serving** (imported at module load, which triggers `init_global_vars()` to
warm up embeddings, retriever, vectorstore, and LLM). Key pieces:

- **`search_question(question)`** — the retrieval stage:
  1. `_vector_retrieval` → Qdrant `similarity_search_with_score` (top `NUM_PAGES_SEARCHED=5`),
     scores kept as cosine similarity.
  2. **Graph expansion** (if `GRAPH_EXPANSION_ENABLED`): infer topic keys from result paths, then
     `expand_chunks_via_graph_with_traversal(...)` finds extra chunks via entity hops (weight
     `GRAPH_EXPANSION_SCORE_WEIGHT=0.5`, capped at `GRAPH_EXPANSION_MAX=10`).
  3. **Rerank** (`_rerank_chunks`) — merge vector + graph pools, sort by score, keep top `2×5=10`.
  4. Optional multi-modal expansion (`add_all_modalities`) pulls the audio/video/image siblings of
     a matched chunk (same `image_path`).
- **`ask_question(question, history, …)`** — the generation stage:
  - builds a `ConversationBufferWindowMemory` (last 5 turns),
  - runs `load_qa_with_sources_chain` (chain_type `stuff`) over the retrieved docs,
  - tracks token usage + estimated cost (`_estimate_cost` with a per-model price table),
  - resolves cited chunk IDs (UUID regex) back to friendly source titles + trust-score badge,
  - renders answer + source documents in Streamlit,
  - fires the **LLM judge** (`_run_llm_judge`, uses `gpt-4o-mini`) that scores hallucination /
    answer relevance / context precision and attaches them to the Opik trace as feedback scores.
- **Opik tracing** is woven throughout via `@opik.track` (`rag_pipeline` → `search_question` →
  `vector_retrieval` / `graph_retrieval` / `reranker`).
- **`show_qa()`** — the Streamlit Q&A page: example prompts, question box, calls `ask_question`,
  and (nice touch) renders an **interactive PyVis graph** of the traversal
  (blue = vector chunks, orange = entities, green = graph-expanded chunks) via
  `_build_traversal_graph_html`.

### 2.7 The knowledge graph — `graph/`

A **parallel index** built alongside the vectors from the same `index.jsonl` files. It has two layers.

**Layer A — structural graph (deterministic, no LLM)** — `graph/construct/graph_builder.py`
([graph_schema.md](dc_available/graph/graph_schema.md) is the spec).
- One tree per **top-level topic** → a `TopicGraphAccumulator` writes each topic to its own folder:
  `graph/<topic_key>/nodes.jsonl`, `edges.jsonl`, `id_lookup.json`, `build_manifest.json`.
- Hierarchy: **Topic → (Subtopic…) → Level → Training → Document → Chunk**, connected by
  `contains` / `part_of` edge pairs. Path structure `topic/[subtopic…]/level/training/index.jsonl`
  drives the inference (`_infer_topic_level_training`). Subtopics are the branch folders inside a
  tree (e.g. `GROWNFIELD/O2C`); DT/Hydraulics have none.
- Explicit `link_to_*` metadata becomes `related_to`/`contains`/`part_of` edges.

**Layer B — semantic graph (LLM-driven)** — four stages, all keyed per topic:
1. **`graph_enricher.prepare_semantic_batches(...)`** — groups valid chunks per document into
   overlapping batches (size/overlap/min-chars from config), dedups repeated text, writes
   `semantic_batches.jsonl`. *No LLM here.*
2. **`semantic_extractor.extract_semantic_candidates(...)`** — for each batch, prompts the LLM
   (`gpt-4o` by default) to return strict JSON of `entities` + `relations` with confidence and
   evidence chunk IDs; sanitizes/dedups them → `semantic_extractions.jsonl`.
3. **`semantic_merger.merge_semantic_candidates(...)`** — groups entities by
   `(normalized_label, entity_type)` and relations by `(src, type, tgt)`, aggregates occurrence
   counts / evidence / avg confidence, applies `min_confidence` / `min_occurrences` thresholds →
   `semantic_entities_merged.jsonl`, `semantic_relations_merged.jsonl`.
4. **`semantic_graph_integrator.integrate_semantic_graph(...)`** — writes `Entity` nodes back into
   each topic's `nodes.jsonl`, adds `mentions` edges (chunk → entity, from evidence) and semantic
   relation edges (entity → entity), updates manifests.

**Graph retrieval** — `graph/retrieval/entity_hop_retriever.py`
- Loads a lightweight in-memory `_GraphIndex` per topic (chunk↔entity mention maps), cached.
- `expand(...)`: seed chunk IDs → entities they mention → other chunks mentioning those entities,
  ranked by number of shared entities, excluding seeds.
- `expand_with_traversal(...)`: same, but also returns `seed→entities` and `entity→expanded` maps
  for the UI visualization.
- `infer_topic_keys_from_storage_paths(...)`: maps result paths to the right topic folder(s) so
  expansion searches the correct tree.

`scripts/visualize_graph.py` renders the built graph for inspection.

### 2.8 The Assistant (agents) — `agents/` + `curriculum_app.py`

A **multi-agent curriculum tutor** built on **AutoGen**, orchestrated by
[curriculum_app.py](dc_available/curriculum_app.py) → `show_assistant()`. Three sequential stages,
each a group chat that ends when the agent emits `SUMMARY:`:
1. **topic_assessor** — figures out what the user wants to learn.
2. **knowledge_assessor** — gauges their current level (fed the topic summary).
3. **teaching_material_assessor** — RAG-enabled; calls `get_information_on_topic` (which reuses the
   same vector+graph retrieval as Q&A) to recommend materials (fed both prior summaries).

- **[curriculum_agents.py](dc_available/agents/curriculum_agents.py)** — agent factory functions +
  a `LangchainRetrieveUserProxyAgent`.
- **[curriculum_chat.py](dc_available/agents/curriculum_chat.py)** — `SLGroupChat` (a `GroupChat`
  that mirrors visible messages into Streamlit), a custom speaker-selection policy, and
  `controlled_chat(...)` that drives turns and detects the `SUMMARY`/`TERMINATE` sentinels.
- **[curriculum_rag.py](dc_available/agents/curriculum_rag.py)** — the `get_information_on_topic`
  tool (language-aware) that runs retrieval and renders results + the same graph-traversal view.
- **[agent_tracing.py](dc_available/agents/agent_tracing.py)** — solves the "Streamlit re-runs the
  whole script each message" problem: keeps **one Opik trace alive per stage** across re-runs by
  storing `TraceData` in `session_state` and re-activating it as the current trace so retrieval
  spans nest correctly.

### 2.9 Frontend — `frontend/`

Streamlit multipage app (entry: [Assistant.py](dc_available/frontend/Assistant.py)):
- **Assistant** (`curriculum_app.show_assistant`) — the 3-stage agent tutor.
- **2_QA** — the single-shot Q&A page (`qa_functions.show_qa`).
- **3_Handbook** — serves/links the generated MkDocs handbook.
- **4_Upload_Documents** — custom UI to upload new training files into `resources/DC_data`,
  with Teams webhook notifications (`utils/teams_sender.py`) and a temp-upload staging area.
- **5_Review_Documents** — password-gated expert **review portal**: preview PDFs/images, approve or
  reject uploaded documents (moves files), logs to `expert_review.log`.
- **6_About** — team/credits page.
- `streamlit_helpers.py` — layout, session prep, per-document rendering, trust-score badges.

### 2.10 Observability — Opik

Fully documented in **[opik-observability.md](opik-observability.md)**. Summary:
- **Span → Trace → Thread** hierarchy: a step (`vector_retrieval`), a full request
  (`rag_pipeline`), a session (all questions).
- `@opik.track` decorators auto-nest spans; `update_current_span/trace` attach cost/usage/output.
- An **LLM judge** attaches per-answer feedback scores (hallucination, answer relevance, context
  precision).
- The self-hosted Opik server is a *separate* project (cloned next to this repo); only the `opik`
  SDK is a dependency here. Config via `OPIK_URL_OVERRIDE`, `OPIK_PROJECT_NAME=dc-available-rag`,
  `OPIK_WORKSPACE=default` (set with `os.environ.setdefault`).

---

## 3. `pipeline/` — orchestration

A numbered set of stages, each runnable as `python -m pipeline.<stage>`. It depends on
`pipeline/.env` (via `dc_available/common/sharepoint_connector.py`) for the SharePoint/Graph
+ Docupedia credentials.

### 3.1 `dc_available/common/sharepoint_connector.py` — the shared MS Graph client
One place for everything the stages need: app-only OAuth token, site/drive resolution, recursive
`list_files`, `download`/`upload` (chunked upload sessions for large `.mp4`), plus:
- **`.url` shortcuts** → fetched as **Confluence/Docupedia** page HTML via REST + PAT
  (`url_to_html`, `_page_id_from_link` handles `/pages/<id>`, `pageId=`, tiny `/x/<code>`, and
  `/display/<space>/<title>` link forms).
- **`.lnk` shortcuts** → resolved to the real file by name via a cached library index (`pylnk3`).
- **classify** helpers (`topic_of`, `level_of`, `training_of`) that derive Topic/Level/Training
  from a file name (3-char topic code, keyword-based Basic/Advanced/Expert level).

### 3.2 STEP 1 — `ingest.py` (grownfield → AVAILABLE SharePoint)
Brings raw material from the "grownfield" SharePoint into the AVAILABLE SharePoint, in 3 phases:
- **fetch** grownfield → `data/grownfield/staging/` (`.pptx`/`.mp4` direct, `.url`→HTML, `.lnk`→file).
- **classify** staging → `data/grownfield/organized/<Topic>/<Level>/training/` (a **move**).
- **upload** organized → AVAILABLE SharePoint; it also **seeds** `sync.py`'s manifest with the
  uploaded eTags so the next sync won't re-download files this stage just uploaded.

### 3.3 STEP 2 — `sync.py` (AVAILABLE SharePoint → local mirror)
Delta-pulls the whole `DC_DATA` into `dc_available/data/local_sharepoint/`. A `.manifest.json`
(relative path → eTag) decides what changed:
- new file → download; missing-in-cloud → delete locally; existing → skip (even if eTag changed —
  only metadata is updated). Returns **`has_changes`** (used by `run_pipeline.py`).

### 3.4 STEP 3 — `preprocess.py` (mirror → ingestion artifacts)
Wraps `summarize_training_for_retrieval` + `create_handbook`:
- `_discover_trainings` finds every folder that directly contains a supported file (topic/level are
  its grandparent/parent, so wrapper folders are ignored).
- **Skips already-processed trainings** (checks cached `index.jsonl` + `summary.txt`); a
  `_training_signature` (name+size+mtime hash, stored as `.source_sig`) is available to detect
  source changes.
- Rebuilds the handbook only if something was (re)built. Optional `--cleanup-downloads` removes the
  raw mirror after success (keeps the manifest). `--overwrite`, `--verbose`, `--training <path>`,
  `--source <key>` flags.

### 3.5 STEP 4 — `index.py` (artifacts → indexes)
Three phases, individually selectable (no flag = run all three):
- **`--vectordb`** — build/refresh the **Qdrant** collection via `get_retriever` + copy images
  (`prepare_image_cache`).
- **`--graph`** — build the structural per-topic graph (`build_per_topic_graph`) + prepare semantic
  batches (`prepare_semantic_batches`). No LLM.
- **`--semantic`** — LLM extraction → merge → integrate (adds Entity nodes + `mentions` edges).
  Modifiers: `--semantic-model`, `--semantic-topics`, `--semantic-max-batches-per-topic`,
  `--semantic-min-confidence`, `--semantic-min-occurrences`.
- `--refresh` forces a clean rebuild (recreate collection, rebuild graph from scratch).

### 3.6 STEP 5 — `serve.py` (launch the app)
Validates the index is ready (**both** the docstore pickle *and* the Qdrant collection must exist),
then launches Streamlit (`--mode assistant|qa`), passing the source through the env var so the app
uses the same data source that was validated.

### 3.7 `run_pipeline.py` (the CronJob one-liner)
`sync → (stop if no changes) → preprocess → index --refresh`. This is the single command an Azure
CronJob runs for unattended updates. Full re-index on any change (simple + correct; incremental
indexing is noted as a future optimization).

**Typical local run order:**
```bash
python -m pipeline.sync            # or: ingest then sync
python -m pipeline.preprocess
python -m pipeline.index           # all phases; or --vectordb / --graph / --semantic
python -m pipeline.serve --mode qa
# unattended:  python -m pipeline.run_pipeline
```

---

## 4. `evaluation/` — quality measurement

A two-phase harness: **generate a QA dataset**, then **run it through the real pipeline and score it**.

### 4.1 Dataset generation — `available/common/dataset_generation/`
[generate_dataset.py](evaluation/available/common/dataset_generation/generate_dataset.py) reads the ingestion
artifacts (`__summary_docs.jsonl`), and for each source file asks the LLM
([prompts.py](evaluation/available/common/dataset_generation/prompts.py) `QA_GENERATION_PROMPT`) to produce **exactly
5 questions** — one each of factual / conceptual / procedural / application / summary. Rules force
"learner-style" questions (never "according to the slide…"). Output is an **Excel** file with
columns: `question, ground_truth, keywords, answer_type, subtopic, topic, level, source_document`.
Flags: `--input-root`, `--output`, `--max-files`, `--dry-run`.

### 4.2 Dataset loading — `available/common/dataset_loader.py`
`load_qa_dataset(path)` reads an Excel file, requires `question` + `ground_truth` columns, and uses
an optional `module` column as the **golden source** (matched as a substring against retrieved
source paths). Returns `[{question, ground_truth, golden_source}]`.

### 4.3 Phase 1 — predictions — `available/generation/qa_runner.py`
- `run_single_question(q)` replicates the production retrieval + generation **without Streamlit**:
  `search_question(q)` → build contexts + source paths → `load_qa_with_sources_chain` → answer.
- `_source_match(golden, retrieved)` computes a **deterministic retrieval hit metric**:
  `source_hit` (bool), `source_hit_rank` (1-based rank of first match), `source_matches` (count).
- `run_qa_dataset(...)` runs every question, **saves `qa_intermediate.json` after each one**
  (crash-safe + resumable), and prints a source-match rate.

### 4.4 Phase 2 — scoring — `available/generation/evaluate_qa.py`
Runs **RAGAS** over the collected results:
- Reference-free metrics: `faithfulness`, `answer_relevancy`, `context_precision`.
- Reference metric (only if ground truth present): `answer_correctness`.
- Uses `gpt-4o` as the RAGAS judge LLM and the project's embeddings. Writes `qa_scores.xlsx`,
  `qa_scores.json`, a combined `qa_results_combined.xlsx` (Phase 1 + Phase 2 side by side), and a
  `summary.json` of aggregate scores.

### 4.5 Entry point — `available/run_pipeline.py`
```bash
# whole AVAILABLE eval: retrieval + generation
poetry run python -m evaluation.available.run_pipeline

# one stage only
poetry run python -m evaluation.available.run_pipeline --only retrieval

# the generic QA-dataset RAGAS flow (predict + score) on one xlsx
poetry run python -m evaluation.available.run_pipeline --qa-dataset evaluation/datasets/test/qa_dataset.xlsx
```
Results are written under `evaluation/results/` (per-dataset folders; the QA-dataset flow
uses a timestamped subfolder). The trust-score experiments run separately via
`python -m evaluation.trust_score.run_pipeline`.

---

## 5. Cross-cutting concepts to remember

- **Chunk `id` is the universal key** linking Qdrant vectors ↔ in-memory docstore ↔ graph nodes ↔
  source resolution in answers. If you change how IDs are generated, everything must be rebuilt.
- **Two-box index invariant:** the Qdrant collection (child vectors) and the `index_ret.pkl`
  docstore (parents) must always be built/loaded **together**. `serve.py` checks for both.
- **Source-configurable paths:** always call `update_data_config(source)` *before* importing the
  path-dependent modules (that's why `index.py` imports helpers *after* configuring).
- **Artifacts never pollute raw data** in the `available` layout (scratch-dir promotion +
  `Artifacts/` / `Extracted_resources/` split).
- **Everything expensive is cached/skippable:** preprocess skips unchanged trainings; retriever
  reuses an existing Qdrant collection; graph refresh is opt-in.

---

## 6. Tech stack (in-scope parts)

| Concern | Choice |
|---|---|
| Language / packaging | Python 3.10–3.11, Poetry |
| Vector store | **Qdrant** (`qdrant-client ~1.9.2`, LangChain community wrapper) |
| Embeddings | `HuggingFaceEmbeddings` — `BAAI/bge-small-en-v1.5` |
| LLM | Azure OpenAI (`gpt-4o`, `gpt-4o-mini`) via `AzureChatOpenAI` + AAD token |
| RAG framework | LangChain (`ParentDocumentRetriever`, `qa_with_sources` chain) |
| Agents | Microsoft **AutoGen** (pyautogen 0.5.x) |
| Graph | custom JSONL per-topic graph + PyVis visualization |
| UI | Streamlit (multipage) + MkDocs Material handbook |
| Observability | **Opik** (self-hosted) + LLM-judge feedback scores |
| Evaluation | **RAGAS** + custom deterministic source-hit metric |
| Source of data | SharePoint / MS Graph, Confluence/Docupedia REST |

---

## 7. Quick mental checklist — "what have I built?"

- [x] Multi-format **ingestion** (PDF/PPTX/Office/**video**/email/HTML) → summaries + chunks, with
      trust scoring and scratch-dir artifact isolation.
- [x] **Qdrant**-backed parent-document vector retrieval (migrated off FAISS).
- [x] A **knowledge graph** (structural per-topic tree + LLM semantic entities/relations) used for
      **retrieval expansion** and an interactive traversal view.
- [x] A **Q&A** page and a **multi-agent Assistant** (topic → knowledge → teaching-material) sharing
      the same retrieval core.
- [x] End-to-end **Opik observability** (spans/traces/threads + LLM-judge quality scores).
- [x] A **pipeline** that ingests from SharePoint, syncs a local mirror, preprocesses, builds all
      indexes (with phase flags), and serves — plus an `auto` mode for CronJobs.
- [x] An **evaluation** harness: LLM-generated QA datasets → real-pipeline predictions →
      RAGAS scoring + deterministic source-hit metric, with resumable intermediate results.
- [x] Content-management UI: **upload** + expert **review** portals with Teams notifications.
```
