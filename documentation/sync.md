# Sync

*Part of AVAILABLE 2.0 · Pipeline STEP 2 (`pipeline/sync.py`)*

---

## Aim

After [SharePoint integration](sharepoint-integration.md), the clean training library lives in the
**AVAILABLE SharePoint** — in the cloud. But the next step (pre-process) reads files from the **local
disk**. So the cloud library has to be brought down locally.

Doing that naively (re-downloading everything on every run) would be slow and wasteful, especially
with large video files. The aim of this step is to keep a **local mirror** of the AVAILABLE
SharePoint that is updated **incrementally** — transferring only what actually changed since the last
run.

---

## Solution

A **delta pull**: the step compares the current cloud contents against a small record of what it
pulled last time (a **manifest**), and acts only on the differences.

```
AVAILABLE SharePoint  (DC_DATA)
        │  list every file  (+ its version tag / eTag)
        ▼
   compare against  .manifest.json
        ├─ new in the cloud       → download into the mirror
        ├─ missing from the cloud → delete from the mirror
        └─ already known          → skip (no download)
        ▼
local mirror   dc_available/data/local_sharepoint/   ──►  STEP 3: pre-process
```

The **manifest** is a small JSON file (`relative path → eTag`) that acts as the memory between runs.
An **eTag** is SharePoint's version tag for a file.

---

## How we solved it

### 1. List the cloud

Using the shared Microsoft Graph client, the step walks the AVAILABLE folder (`DC_DATA`) recursively
and collects every file together with its **eTag**.

### 2. Compare against the manifest

For each file, the decision is made purely from the manifest — not from what happens to be on disk:

- **New** (path not in the manifest) → **download** it into the mirror and record its eTag.
- **Already known** (path in the manifest) → **skip**, do not re-download. If its eTag changed in the
  cloud, the manifest is updated (metadata only) but the file is **not** downloaded again.
- **Deleted** (path in the manifest but no longer in the cloud) → **remove** it from the mirror and
  drop it from the manifest.

Because the decision is driven by the manifest and not by file presence, the local mirror can safely
be cleaned up after pre-processing without causing everything to be re-downloaded next time.

### 3. Report what changed

At the end, the step prints a short report (new / updated / deleted / unchanged) and returns a
simple **`has_changes`** flag (true when anything was added or deleted). This flag is what lets the
unattended `auto` mode decide whether the later stages need to run again.

### 4. Link back to ingest (no double download)

When [ingest](sharepoint-integration.md) uploads a file to the AVAILABLE SharePoint, it also **seeds
the manifest** with that file's eTag (and copies it straight into the mirror). As a result, sync
recognizes those files as already known and does **not** download back the files that were just
uploaded.

---

## How this fits into AVAILABLE 2.0

This is **STEP 2 of the modular pipeline**:

```
ingest → sync → pre-process → index → serve
```

It sits between ingest (which fills the cloud) and pre-process (which reads local files). By pulling
only the differences and exposing a `has_changes` flag, it makes **incremental updates** and the
**unattended `auto` run** possible — the pipeline only reprocesses when something actually changed.

Like the rest of the pipeline, sync is a **new 2.0 capability**: the 1.0 baseline had no cloud mirror
and no incremental update mechanism.

---

## Configuration

Settings come from **`pipeline/.env`** (the same file used by SharePoint integration):

| Variable | Purpose |
|---|---|
| `AVAILABLE_SITE_ID` | the AVAILABLE SharePoint site to mirror |
| `AVAILABLE_FOLDER_PATH` | the folder to mirror (e.g. `…/DC_DATA`) |
| `GRAPH_TENANT_ID`, `GRAPH_CLIENT_ID`, `GRAPH_CLIENT_SECRET` | Microsoft Graph login (shared) |

The local mirror is written to `dc_available/data/local_sharepoint/`, with its manifest at
`dc_available/data/local_sharepoint/.manifest.json`.

---

## How to run

```bash
python -m pipeline.sync
```
