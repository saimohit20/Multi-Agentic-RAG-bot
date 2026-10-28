# Configuration Reference

Set environment variables before starting a command. They are read by the
application process; changing them in a different terminal does not update an
already running app. Examples in these guides use PowerShell `$env:NAME` syntax.

## Data Source and Storage

| Environment variable | Current default | Purpose |
| --- | --- | --- |
| `AVAILABLE_APP_SOURCE` | `available` | Select the configured source. An explicitly set environment value takes precedence over the source argument in data configuration. |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP endpoint. |
| `QDRANT_COLLECTION` | `available2.0` | Main dense and sparse chunk collection. |
| `QDRANT_ENTITY_COLLECTION` | `available_entities` | Entity collection for graph query seeding. |
| `QDRANT_RICH_COLLECTION` | `RICH` | Separate collection for the RICH interview feature. |
| `QDRANT_RICH_ENTITY_COLLECTION` | `available_rich_entities` | Separate RICH graph entity collection. |

The `available` source uses these project-relative paths:

| Content | Path under `dc-available` |
| --- | --- |
| Raw mirror | `dc_available/data/local_sharepoint/` |
| Preprocessing output | `dc_available/data/Artifacts/resources/ingestion/` |
| Parent-document store | `dc_available/data/Artifacts/resources/index/index_ret.pkl` |
| Graph artifacts | `dc_available/data/Artifacts/resources/graph/` |
| Extracted images and media | `dc_available/data/Extracted_resources/resources/` |
| Generated handbook | `dc_available/data/Artifacts/handbook/` |

Source roots and artifact layouts are configured in
`dc_available/config/data_config.py`; they are not generic environment variables.
Keep the selected source, mounted data directory, Qdrant collection, and graph
artifacts consistent. Never load an untrusted pickle/document-store backup.

## Azure Models and Authentication

| Environment variable | Purpose |
| --- | --- |
| `AZURE_ENDPOINT` | Classic Azure OpenAI-compatible resource endpoint. Set your approved endpoint explicitly. |
| `API_VERSION` | Classic endpoint API version; current default is `2024-02-15-preview`. |
| `AZURE_V1_ENDPOINT` | Foundry OpenAI v1 endpoint used by the configured v1 model path. |

The application uses Microsoft Entra bearer tokens through Azure's default
credential chain, not a required Azure OpenAI API key. For native local use,
Azure CLI sign-in is one supported credential source. Workload Identity is the
deployment path for AKS. A host's Azure CLI login is not automatically available
inside a Docker container.

Model choices are source settings in `dc_available/config/llm_config.py`.
`DEFAULT_SERVE_MODEL` and the default general model are currently `gpt-4o`.
`gpt-5.2` has a separate Foundry v1 path; do not assume every agent and model
client uses that path. Changing an endpoint alone does not provision a model
deployment or grant permission to invoke it.

## Retrieval Defaults

These are **source configuration constants**, not environment variables.
The values below describe the checked working tree; verify them again when
changing branches or preparing a release.

| Setting | Current value | Meaning |
| --- | --- | --- |
| `DEFAULT_EMBEDDING_MODEL` | `BAAI/bge-m3` | Dense and sparse embeddings. |
| `FUSION_LEGS` | `both` | Vector and graph retrieval contribute candidates. |
| `FUSION_STRATEGY` | `union` | Merge and deduplicate candidates; `rrf` is an alternative. |
| `FUSION_VECTOR_K` / `FUSION_GRAPH_K` | `10` / `10` | Candidate count from each leg. |
| `FUSION_OUTPUT_K` | `10` | Context passages retained for generation. |
| `FUSION_DOC_TOPN` | `5` | Recommended source-document limit. |
| `GRAPH_RETRIEVAL_MODE` | `ppr` | Personalized PageRank; `entity_1hop` is the alternative engine. |
| `PPR_SEED_MODE` | `merge` | Filter and combine matched entity seeds. |
| `PPR_TOPIC_ISOLATION` | `True` | Restrict graph walks to dominant matched topics. |
| `RERANKER_ENABLED` | `False` | Cross-encoder reranking is available but currently disabled. |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Model used when reranking is enabled. |
| `TRUST_ENABLED` / `TRUST_BETA` | `True` / `0.5` | Enable and scale the document-trust adjustment. |

Changing embeddings requires compatible collections and a coordinated rebuild.
Changing retrieval controls should be evaluated on a fixed dataset before being
used in a shared environment. A diagram may show optional stages that are not
enabled in the current configuration.

## UI, Integrations, and Secrets

`STREAMLIT_SERVER_PORT` is a Streamlit setting, not a pipeline CLI option. The
getting-started guide explicitly sets it to `8502` to avoid relying on a local
Streamlit configuration.

See [SharePoint integration](../operations/sharepoint.md) and
[observability](../operations/observability.md) for additional integration settings.
Do not assume every entry point loads the same `.env` file. Explicit shell
environment variables are the least ambiguous setup for the native commands.
Do not commit passwords, tokens, private keys, or populated secret manifests.