# Pre-process

*Part of AVAILABLE 2.0 · Pipeline STEP 3 (`pipeline/preprocess.py`, `dc_available/summarization/`)*

---

## Aim

The synced files are still in their original formats — PDF, PowerPoint, video, email, HTML. A
retrieval system cannot search those directly; it needs **clean text** with a consistent structure.

The aim of this step is to turn every training file into **text summaries and chunks**, each carrying
a little metadata, so the next step can build the search indexes from a single, uniform input.

---

## Solution

A **per-training summarization pass**: every file in a training folder is summarized according to its
format, the results are gathered into one index per training, and a browsable handbook is produced.

```
local mirror   (topic/level/training/files)
      │  discover trainings
      ▼
  for each file → summarize by format
      pdf / pptx / office → page or slide text + summary + images
      mp4 (video)         → transcript + slides + descriptions   (VideoIE)
      msg (email)         → body text + attachments
      html                → text
      │   (raw files handled in a scratch folder; only derived media is kept)
      ▼
  per file     →  __summary.txt   +   __summary_docs.jsonl
      │  aggregate
      ▼
  per training →  index.jsonl   +   summary.txt
      │  when anything was (re)built
      ▼
  handbook   (browsable MkDocs website)   ──►  STEP 4: index
```

---

## How we solved it

### 1. Discover trainings
A *training* is any folder that directly contains a supported file. Its parent folder is treated as
the **Level** and its grandparent as the **Topic**, so the `Topic / Level / training` structure
created during ingestion is picked up automatically, and any wrapper folders above it are ignored.

### 2. Summarize each file by format
Each file is passed to a format-specific extractor. The accepted formats and what they produce:

| Format | What is produced |
|---|---|
| **PDF** | per-page text + per-page summary + page images |
| **PPTX / PPTM / XLSX / DOCX** | converted to PDF, then per-slide text + summary + slide images |
| **MP4** (video) | transcript + slide screenshots + image descriptions (see below) |
| **MSG** (email) | email body text + a summary; attachments are queued and processed too |
| **HTML** (Docupedia/Confluence page) | treated as text |

Every file also gets a short **meta-summary** — a summary of the whole file — which becomes its
headline entry.

### 3. Video handling (VideoIE)
Videos are handled by a dedicated subsystem. In short: the audio is transcribed to text, the slides
shown in the video are captured as screenshots and de-duplicated, each distinct slide is described by
an image model, and the transcript and slides are merged into timestamped chunks — so a video becomes
searchable text just like a document.

### 4. Keeping the raw data clean
Raw media is never processed in place. Each file is copied into a temporary scratch folder, all
extraction happens there, and only the **derived** outputs (images, converted PDFs, slide
screenshots) are moved into a separate `Extracted_resources` area. The raw file (and, for video, the
extracted audio) is discarded. This keeps generated files from ever mixing into the mirrored source
data.

### 5. Aggregate per training
For each training, all the file chunks and summaries are collected into a single **`index.jsonl`**
(the canonical input for the next step) and a training-level **`summary.txt`**. Every chunk carries
lightweight **metadata** — an overview of where it belongs: which document and training it came from,
what type of content it is, links to related pieces, and a trust score. This is what lets later
stages understand how the pieces fit together. *(The trust score stamped here is a simple initial
value; the full automated trust scoring has its own doc.)*

### 6. Caching and incremental rebuild
Summarizing is expensive because it calls an LLM, so results are cached. A training that already has
its `index.jsonl` and `summary.txt` is skipped, unless a rebuild is forced. Re-running pre-process
therefore only processes what is new or changed.

### 7. Handbook
Whenever at least one training was (re)built, the summaries are assembled into a **handbook** — a
static MkDocs website that lets a person browse through the whole corpus.

---

## Outputs

- **Ingestion artifacts** (text): `index.jsonl` + `summary.txt` per training, plus per-file
  `__summary.txt` and `__summary_docs.jsonl`.
- **Extracted resources** (media): page/slide images and video slides.
- **Handbook**: a browsable static site generated from the summaries.

---

## How this fits into AVAILABLE 2.0

This is **STEP 3 of the modular pipeline**:

```
ingest → sync → pre-process → index → serve
```

Pre-process turns the synced raw files into `index.jsonl`, which is the **single input** the next
step uses to build **both** the vector store and the knowledge graph. Cleaner, well-chunked text
here directly improves retrieval quality later.

Summarization existed in the 1.0 baseline, but as one monolithic script. AVAILABLE 2.0 turns it into
a **cached, incremental pipeline stage** that keeps raw and generated data cleanly separated.

---

## Configuration

Pre-process uses the app's data-source selection and the Azure OpenAI configuration (the LLM that
writes the summaries):

| Setting | Purpose |
|---|---|
| `AVAILABLE_APP_SOURCE` (or `--source`) | which data collection to process |
| Azure OpenAI credentials | used by the summarizer to generate the text summaries |

---

## How to run

```bash
python -m pipeline.preprocess                    # all trainings (cached ones are skipped)
python -m pipeline.preprocess --overwrite        # rebuild every training
python -m pipeline.preprocess --training <path>  # process a single training folder
python -m pipeline.preprocess --verbose          # show the underlying extraction / LLM logs
python -m pipeline.preprocess --cleanup-downloads # after success, delete mirrored raw files
```
