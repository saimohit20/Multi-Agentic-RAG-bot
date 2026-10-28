# SharePoint Integration

SharePoint supports two distinct workflows: publishing an organized training
library and synchronizing that library to the local mirror. Neither is required
when preparing a small corpus entirely from local files.

## Credentials and Locations

The connector loads integration settings from `pipeline/.env`. Use a private local
file or the approved deployment secret mechanism; never commit populated values.

| Setting | Purpose |
| --- | --- |
| `GRAPH_TENANT_ID`, `GRAPH_CLIENT_ID`, `GRAPH_CLIENT_SECRET` | Microsoft Graph app-only authentication. |
| `SHAREPOINT_HOST` | SharePoint hostname for site resolution. |
| `SOURCE_SITE_ID` or `SOURCE_SITE_PATH` | Unorganized source library location. |
| `SOURCE_FOLDER_PATH` | Input folder for ingestion and GROWNFIELD metadata collection. |
| `DEST_SITE_ID` or `DEST_SITE_PATH` | Destination for the organized library. |
| `DEST_FOLDER_PATH` | Upload destination within that library. |
| `AVAILABLE_SITE_ID`, `AVAILABLE_FOLDER_PATH` | Site and drive-relative folder mirrored by the standard source. |
| `RICH_FOLDER_PATH` | Optional separate interview folder for the RICH source. |
| `DOCUPEDIA_BASE_URL`, `DOCUPEDIA_PAT` | Access for resolving supported web-page shortcuts. |

Ask the service owner for least-privilege site access. SharePoint Graph
credentials are separate from the Azure model authentication used for answers.
Use a drive-relative folder, not a full browser URL or sharing link.

## Publish an Organized Library

From `dc-available`:

```powershell
poetry run python -m pipeline.ingest --phase fetch
poetry run python -m pipeline.ingest --phase classify
poetry run python -m pipeline.ingest --phase upload
```

Fetch resolves supported shortcuts to content, classify arranges the staging
files by topic/level/training, and upload publishes the resulting library.
Classification is based on naming conventions, so inspect its output before
uploading. The upload step writes remotely; the commands are documented here
but are not needed to preview this documentation site.

## Synchronize the Library

```powershell
poetry run python -m pipeline.sync --source available
```

The local mirror keeps a manifest of known remote paths and version tags. Sync
downloads new paths and removes local files whose remote paths disappeared.

!!! warning "Known-file updates are not content refreshes"
    Existing manifest paths are skipped even when their remote version tag
    changes or their local file is absent. The manifest may record a newer tag
    without downloading newer content. Only additions and deletions make sync
    report work to the automatic pipeline. Do not interpret "no changes" as
    proof that local content matches the remote library.

If content is stale or cleaned raw files must be recovered, stop automated jobs,
back up the mirror and manifest, and plan a controlled resynchronization with
the operator. Do not delete the manifest casually: that can redownload the whole
library and trigger expensive rebuilding. Preserve an authoritative source copy.

For RICH, use `--source rich` and verify its own folder and manifest. If
`RICH_FOLDER_PATH` is absent, the sync code derives a sibling `RICH` location;
configure it explicitly to avoid an unintended folder selection.