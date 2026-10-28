# Opik Setup

The recorded setup uses a local, self-hosted Opik stack from the upstream
`comet-ml/opik` repository. The server runs separately from AVAILABLE; AVAILABLE
uses the Opik SDK already declared in its project dependencies to send traces.

## Setup at a Glance

| Item | Recorded local value |
| --- | --- |
| Server source | <https://github.com/comet-ml/opik> |
| Checkout location | `opik/`, beside `dc-available/` |
| Working directory | `opik/deployment/docker-compose/` |
| Compose profile | `opik` |
| Custom frontend image | `opik-frontend-custom` |
| Dashboard | <http://localhost:5173> |
| SDK API address | `http://localhost:5173/api` |
| AVAILABLE project | `dc-available-rag` |
| Workspace | `default` |

!!! note "Environment-specific setup record"
    The original notes do not identify the upstream commit/release or provide
    the custom frontend Dockerfile. The recipe below records the setup used; it
    is not a guaranteed installation path for the latest upstream release.
    Obtain the original checkout and custom build files, or establish a reviewed
    equivalent, before claiming the environment is reproducible.

## 1. Prepare the Server Checkout

Use Docker Desktop with Docker Compose. From `dc-available`, the original layout
was prepared with:

```powershell
Set-Location ..
git clone https://github.com/comet-ml/opik.git
Set-Location .\opik\deployment\docker-compose
```

Cloning downloads an external repository; it is only needed when obtaining a
checkout, not each time the stack starts. For an existing installation, use its
existing checkout and record its commit rather than replacing it with a fresh
default-branch clone. These documentation changes do not perform the download,
install the stack, or modify that external checkout.

## 2. Apply the Recorded Corporate-Network Workarounds

The original setup avoided pulling the upstream frontend image and changed a
privileged-container setting:

1. In `docker-compose.yaml`, locate the service that originally used
   `privileged: true` and set that field to `false`. The original notes did not
   record the service name; confirm it against the original checkout rather
   than changing unrelated services.
2. In `docker-compose.override.yaml`, merge the following under the existing
   top-level `services` mapping. Preserve any other overrides:

```yaml
services:
  frontend:
    image: opik-frontend-custom
    volumes: !override []
```

The override selects the local custom frontend image and replaces inherited
frontend volume mounts. `!override` requires a Docker Compose version that
supports the tag; check `docker compose version` and validate the merged config.
The privilege change and frontend override are local workarounds, not general
requirements for every Opik installation.

## 3. Validate and Build the Custom Frontend

From `opik/deployment/docker-compose`, with the original custom
`Dockerfile.frontend` and its build inputs available:

```powershell
docker compose version
docker compose --profile opik config --quiet
docker build -t opik-frontend-custom -f Dockerfile.frontend .
```

If the Dockerfile is absent, recover it from the original setup owner rather
than assuming it is shipped by every upstream version. The build may still need
network access for dependencies. The local image only avoids pulling the
corresponding prebuilt frontend image; it does not make all dependencies offline.

## 4. Start and Verify the Stack

From the same Compose directory:

```powershell
docker compose --profile opik up --detach --pull never
docker compose --profile opik ps
docker compose --profile opik logs --tail 80
```

`--pull never` requires the needed images to be present locally. If an image is
missing, obtain it through your approved network/image process; this flag does
not bypass a missing dependency. Confirm the services are healthy and open
<http://localhost:5173>.

Review the resolved Compose port bindings and authentication before using
restricted data. A local dashboard URL alone does not prove that every service
is bound only to loopback.

## 5. Connect AVAILABLE

In the separate terminal that launches AVAILABLE:

```powershell
$env:OPIK_URL_OVERRIDE = "http://localhost:5173/api"
$env:OPIK_UI_URL = "http://localhost:5173"
$env:OPIK_PROJECT_NAME = "dc-available-rag"
$env:OPIK_WORKSPACE = "default"
```

Restart the app, submit a test question, and confirm its trace appears in the
intended project. Check the Q&A Logs link as well as the dashboard. Use an
approved reachable service URL instead of `localhost` when AVAILABLE runs in a
container or another machine.

The optional LLM judge is off by default. See
[observability](observability.md#optional-llm-judge) for enabling it and
interpreting feedback scores. Starting Opik does not itself enable judge calls.

## Subsequent Starts and Shutdown

Return to the same checkout and Compose directory for each operation:

```powershell
docker compose --profile opik up --detach --pull never
docker compose --profile opik down
```

`down` removes the stack's containers and networks; named volumes are normally
retained unless explicitly removed. Verify the actual checkout's volume and
bind-mount configuration before relying on persistence. Do not add `--volumes`
as a routine shutdown step.

## Storage and Handover

The original AVAILABLE notes do not specify Opik's data-volume names or a tested
backup procedure. Inspect the resolved Compose configuration to identify the
database/object-storage volumes and bind mounts for that version. Record the
checkout commit, local overrides, image digests, storage locations, and tested
restore procedure before upgrading or moving the stack.

Traces can contain training passages, user questions, and generated answers.
Apply approved access control and retention rules to both running storage and
backups. For trace interpretation, costs, privacy, and the dashboard example,
continue with [Observability with Opik](observability.md).