# RICH Interview Workflow

RICH is an interview-oriented extension with a separate frontend page and data
pipeline. Do not assume the standard training build populates it.

## Isolation

| Resource | RICH value |
| --- | --- |
| Source key | `rich` |
| Raw mirror | `dc_available/data/local_sharepoint/rich/` |
| Artifacts | `dc_available/data/Artifacts/rich/` |
| Vector collection | `RICH` by default. |
| Entity collection | `available_rich_entities` by default. |
| Remote folder | `RICH_FOLDER_PATH` in the configured AVAILABLE SharePoint site. |

Set the remote folder explicitly and verify that corpus discovery does not mix
interviews into the training corpus. Keep collections and output paths separate.

## Build Workflow

With approved integration access, run from `dc-available`:

```powershell
$env:AVAILABLE_APP_SOURCE = "rich"
poetry run python -m pipeline.run_rich_pipeline
```

The runner synchronizes RICH, copies existing preprocessing artifacts without
building the handbook, builds its vector and semantic graph assets, and runs its
own trust stages. It expects the interview material and prepared artifacts
required by that workflow; it is not the standard arbitrary-PDF walkthrough.
Like the main sync workflow, it stops without downstream work when sync reports
no additions or deletions. Known-file remote updates have the same limitation
described in [SharePoint integration](sharepoint.md).

After running RICH commands, reset the source for standard training commands:

```powershell
$env:AVAILABLE_APP_SOURCE = "available"
```

The RICH page depends on deployment availability, its own artifacts, and the
configured models. Validate it independently using an interview whose content
and access restrictions you know. Interview data and traces may contain personal
information; confirm processing and telemetry permissions before use.