# SharePoint Integration

*Part of AVAILABLE 2.0 · Pipeline STEP 1 (`pipeline/ingest.py`, `dc_available/common/sharepoint_connector.py`)*

---

## Aim

The training material does not start in a clean state. It lives on a **"grownfield" SharePoint**,
where it is:

- **unstructured** — files are not organized in a way the RAG pipeline can use, and
- **mixed with shortcut files** that point elsewhere instead of holding real content:
  - **`.lnk`** (Windows *Verknüpfung*) — a shortcut to the real file somewhere else in the library,
  - **`.url`** — a shortcut to a Confluence/Docupedia web page.

A shortcut is useless to a retrieval system: there is nothing to read inside it. The aim of this
step is therefore to **collect the real content**, **resolve every shortcut to actual data**, and
**store it in a clean, structured library** (organized by *Topic / Level / Training*) that the rest
of the pipeline can rely on.

---

## Solution

A **Microsoft Graph client** connects to SharePoint, and a **three-phase ETL** (Extract, Transform,
Load) moves the data from the messy grownfield source into a clean **AVAILABLE SharePoint**:

```
grownfield SharePoint  (raw, unstructured, shortcuts)
        │
        │  FETCH        .pptx / .mp4  → downloaded directly
        │               .url          → resolved to page HTML
        │               .lnk          → resolved to the real file
        ▼
   staging/            (all real content, still flat)
        │
        │  CLASSIFY     read Topic / Level from the file name   (move)
        ▼
   organized/<Topic>/<Level>/training/
        │
        │  UPLOAD       push to the cloud + seed the sync manifest
        ▼
AVAILABLE SharePoint  (clean, structured)  ──►  STEP 2: sync → local mirror
```

Two SharePoint locations are involved:

- **SOURCE** = the *grownfield* site (raw input).
- **DEST** = the *AVAILABLE* site (clean output that the next pipeline step mirrors).

---

## How we solved it

### 1. Connecting to SharePoint

The client authenticates to **Microsoft Graph** with an **app-only** login (a registered app's
client ID and secret — no user sign-in). With that token it can resolve a site, open its document
drive, list files recursively, and download or upload content. The same client is used for both the
SOURCE and the DEST site.

### 2. Fetch — collect real content and resolve shortcuts

Every file under the grownfield source folder is walked once. What happens depends on its type:

- **`.pptx` / `.mp4`** — a real document; downloaded as-is.
- **`.url`** — a Docupedia/Confluence shortcut. The link inside is read, the target **page is fetched
  through the Docupedia REST API** (using a personal access token — no browser), and its body is
  saved as an `.html` file. The web page effectively becomes a local document.
- **`.lnk`** — a shortcut to another file in the same library. The shortcut is parsed to find the
  **target file name**, which is looked up in a **one-time index of every file name in the library**,
  and the *real* file it points to is downloaded instead of the shortcut.
- Other types (e.g. `.srt`) are ignored.

Everything collected lands in a flat **`staging/`** folder.

### 3. Classify — impose a Topic / Level / Training structure

Each staged file is sorted into a folder structure derived **from its file name**:

- **Topic** — the 3-character code before the first `-` (e.g. `O2C-02.010 …` → **O2C**).
- **Level** — inferred from keywords in the name (**Basic / Advanced / Expert**).
- **Training** — a single fixed `training` folder per Topic/Level.

Files are **moved** (not copied) from `staging/` into `organized/<Topic>/<Level>/training/`. For
example:

```
organized/
└─ O2C/
   └─ Advanced/
      └─ training/
         ├─ O2C-02.010 Standard Sales Order.pptx
         └─ O2C-03.003 Free of Charge Under Delivery (FOC) - video.mp4
```

### 4. Upload — publish the clean library

The whole `organized/` tree is uploaded to the **AVAILABLE SharePoint** (large files such as videos
are sent in chunks). During upload, each file's version tag (**eTag**) is recorded and used to
**seed the sync manifest**, so the next pipeline step does not re-download the files that were just
uploaded.

---

## How this fits into AVAILABLE 2.0

This is **STEP 1 of the modular pipeline** and the entry point for all data:

```
ingest → sync → pre-process → index → serve
```

The clean `Topic / Level / Training` library it produces is exactly what the later stages depend on:

- **sync** (STEP 2) mirrors the AVAILABLE SharePoint into a local folder,
- **pre-process** (STEP 3) turns those files into summaries and chunks, and
- the **knowledge graph** is later built on the same Topic → Level → Training hierarchy.

SharePoint integration is a **new capability in 2.0** — the 1.0 baseline had no SharePoint ingestion,
no shortcut resolution, and no automatic structuring of the corpus.

---

## Configuration

The client reads its settings from **`pipeline/.env`**:

| Variable | Purpose |
|---|---|
| `GRAPH_TENANT_ID`, `GRAPH_CLIENT_ID`, `GRAPH_CLIENT_SECRET` | Microsoft Graph app-only login |
| `SHAREPOINT_HOST` | SharePoint host name (used with the site paths below) |
| `SOURCE_SITE_ID` *or* `SOURCE_SITE_PATH` | the grownfield site |
| `SOURCE_FOLDER_PATH` | folder to read raw material from |
| `DEST_SITE_ID` *or* `DEST_SITE_PATH` | the AVAILABLE site |
| `DEST_FOLDER_PATH` | folder to upload the organized library to |
| `AVAILABLE_FOLDER_PATH` | the folder that `sync` mirrors (used to seed its manifest) |
| `DOCUPEDIA_BASE_URL`, `DOCUPEDIA_PAT` | Docupedia/Confluence REST access for `.url` shortcuts |

---

## How to run

```bash
# full step: fetch → classify → upload
python -m pipeline.ingest

# run a single phase only
python -m pipeline.ingest --phase fetch
python -m pipeline.ingest --phase classify
python -m pipeline.ingest --phase upload
```
