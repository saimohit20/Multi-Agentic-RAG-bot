# Getting Started

This guide runs AVAILABLE on your workstation with local training files and a
local Qdrant instance. SharePoint synchronization is optional for this workflow.
Commands below use PowerShell and run from the `dc-available` project directory.

## Prerequisites

| Requirement | Why it is needed |
| --- | --- |
| Python 3.10 or 3.11 and Poetry | Install the project and run its commands. |
| Docker Desktop | Run Qdrant locally. |
| Azure CLI and an authorized Azure identity | Authenticate to the configured model deployments. |
| Access to training material | Build a searchable corpus. No sample corpus is bundled with this guide. |
| Network access to Azure and Hugging Face | Call the models and download embedding weights. |

Office and video extraction have additional system dependencies. The pipeline
image installs LibreOffice and FFmpeg; a native installation needs equivalent
tools for the formats being processed. Start with a small PDF training to reduce
setup complexity. Allow disk space and memory for PyTorch, model weights, media,
and derived artifacts; first-time downloads can be substantial.

!!! note "Local application, external models"
    Running locally does not make the application offline. Summarization,
    semantic extraction, and answer generation use configured Azure services.
    Confirm that your corpus is approved for those services before processing it.

## 1. Install the Project

From the workspace root:

```powershell
Set-Location dc-available
poetry install --with dev
poetry run python --version
```

The supported Python range is `>=3.10,<3.12`. If Poetry selects a different
interpreter, select an installed Python 3.11 interpreter with `poetry env use`
before installing. Dependency installation may need your organization's proxy
and package-index settings.

## 2. Configure Access

Use values supplied by the Azure resource owner, not another deployment's
endpoint. Sign in with an identity authorized to call the model:

```powershell
az login
$env:AZURE_ENDPOINT = "https://<your-resource>.openai.azure.com/"
$env:API_VERSION = "2024-02-15-preview"
$env:AVAILABLE_APP_SOURCE = "available"
$env:QDRANT_URL = "http://localhost:6333"
$env:QDRANT_COLLECTION = "available2.0"
$env:QDRANT_ENTITY_COLLECTION = "available_entities"
$env:STREAMLIT_SERVER_PORT = "8502"
```

The default serving and semantic-extraction model is `gpt-4o`; the corresponding
deployment must be available to your identity. Azure roles and deployment names
must be confirmed with the resource owner. See the
[configuration reference](../reference/configuration.md) for authentication,
source selection, and retrieval defaults. Environment variables above apply to
this terminal session and its child processes.

## 3. Start Qdrant

For a new local instance:

```powershell
docker run -d --name qdrant -p 127.0.0.1:6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Open <http://localhost:6333/dashboard> to confirm it is reachable. If a container
named `qdrant` already exists, use `docker start qdrant` instead. The named volume
persists collections across container restarts. This command uses the upstream
default image tag for local experimentation; pin a reviewed version for shared
environments. Do not expose an unauthenticated instance to a public network.

## 4. Add a Small Training

Place files under this hierarchy, relative to `dc-available`:

```text
dc_available/data/local_sharepoint/
  ExampleTopic/
    Basic/
      ExampleTraining/
        introduction.pdf
```

A training is a folder that directly contains supported files. Its parent
supplies the level and its grandparent supplies the topic. Use this explicit
hierarchy even if your corpus has only one topic. Generated artifacts are stored
outside the raw mirror; see [pipeline operations](../operations/pipeline.md).

## 5. Build the Knowledge Base

Run the stages directly for local files:

```powershell
poetry run python -m pipeline.preprocess --source available
poetry run python -m pipeline.index --source available
```

Preprocessing and semantic indexing can make paid model calls. Wait for each
stage to finish successfully before starting the next one.

Trust scoring is not required for this local-file smoke test. The current trust
builder collects GROWNFIELD SharePoint metadata and is not a generic scorer for
arbitrary local PDFs. Use the trust workflow only with its required integration
access. Without a matching trust table, ranking uses a common fallback value
rather than verified document-specific scores.

!!! warning "Do not use the sync orchestrator for this first build"
    `pipeline.run_pipeline` begins with SharePoint synchronization and stops if
    no additions or deletions are reported. Updates to already-known remote files
    do not download new content or trigger this rebuild. It is not a forced
    rebuild of manually supplied local files.

## 6. Open the Application

Choose one experience:

```powershell
poetry run python -m pipeline.serve --source available --mode assistant
```

Or start Q&A instead:

```powershell
poetry run python -m pipeline.serve --source available --mode qa
```

With `STREAMLIT_SERVER_PORT=8502` set above, open <http://localhost:8502>.
The launcher checks for the parent-document store and the configured Qdrant
collection before starting Streamlit. It does not certify that every graph,
model, or Azure permission is ready.

## Check Your First Answer

Ask a question whose answer you can locate in the PDF you supplied. Confirm that
the cited document is the expected one, inspect the supporting passage, and
compare it with the original. Continue with the [user guide](../user-guide/index.md)
or [troubleshooting](../reference/troubleshooting.md) if startup or retrieval fails.