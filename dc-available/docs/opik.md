<!---

	Copyright (c) 2024 Robert Bosch GmbH and its subsidiaries.

-->

# Opik Setup (Optional)

[Opik](https://github.com/comet-ml/opik) (by Comet ML) is used to **trace** and **evaluate** the
RAG / agent pipeline. It is optional — the app runs without it. It has two parts:

1. A **self-hosted Opik server** that runs locally via Docker Compose (UI at `http://localhost:5173`).
2. The **`opik` Python SDK**, already declared in `pyproject.toml`, which the project code uses to send
   traces/evaluations to that server (see `dc_available/agents/agent_tracing.py`).

> The Opik server source is **not** part of this repository. It is the upstream `comet-ml/opik`
> project and must be cloned separately (next to this repo).

## 1. Run the Opik server (Docker)

### First-time setup

```
# 1. Clone the Opik server source next to this repo (alongside dc-available/)
cd ..
git clone https://github.com/comet-ml/opik.git
cd opik/deployment/docker-compose

# 2. Edit docker-compose.yaml (~line 235)
#    change:  privileged: true   ->   privileged: false

# 3. Edit docker-compose.override.yaml — add at the bottom:
#    frontend:
#      image: opik-frontend-custom
#      volumes: !override []

# 4. Build the custom frontend image
docker build -t opik-frontend-custom -f Dockerfile.frontend .

# 5. Start the stack (--pull never uses the locally built image)
docker compose --profile opik up --detach --pull never
```

Then open the UI at **http://localhost:5173**.

> Steps 2–4 work around a corporate-network constraint: `privileged: false` and a locally built
> frontend image avoid pulling the upstream frontend image.

### Start / stop on later runs (skip steps 1–4)

```
cd opik/deployment/docker-compose

# Start
docker compose --profile opik up --detach --pull never

# Stop
docker compose --profile opik down
```

## 2. Configuration

No extra configuration is required for the default local setup. The code already sets these via
`os.environ.setdefault(...)` (in `curriculum_app.py`, `RAG/qa_functions.py`, and
`graph/retrieval/entity_hop_retriever.py`):

```
OPIK_URL_OVERRIDE=http://localhost:5173/api
OPIK_PROJECT_NAME=dc-available-rag
OPIK_WORKSPACE=default
```

Only override these (e.g. in your `.env`) if the Opik server runs elsewhere or you want a different
project/workspace name.
