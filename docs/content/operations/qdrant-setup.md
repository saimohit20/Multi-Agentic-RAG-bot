# Qdrant Setup

The recorded local setup runs Qdrant as a standalone Docker container, separate
from AVAILABLE and its pipeline Compose services. Indexing writes the chunk and
entity collections; serving reads them. Start Qdrant before either workflow.

## Setup at a Glance

| Item | Recorded local value |
| --- | --- |
| Image | `qdrant/qdrant` |
| Container name | `qdrant` |
| Published HTTP port | `6333` |
| Persistent Docker volume | `qdrant_storage` |
| Storage inside the container | `/qdrant/storage` |
| Dashboard | <http://localhost:6333/dashboard> |
| Main chunk collection | `available2.0` |
| Entity collection | `available_entities` |

The original setup did not pin an image version. Record the running image tag
and digest before recreating or upgrading an existing instance. Do not assume a
new default image is compatible with every older snapshot or client.

## First-Time Setup

Docker Desktop must be running. The original command published port `6333` on
all host interfaces. The following local-only variant retains the same image,
container name, and storage while binding the port to loopback:

```powershell
docker run -d --name qdrant -p 127.0.0.1:6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Use this only for a new instance. If the container already exists, start it
instead of creating another one. Port `6333` is the HTTP API/dashboard port;
this command does not publish the separate gRPC port.

The named volume holds the collections independently of the container's writable
layer. A restart does not rebuild or erase them. A named volume is persistence,
not an off-machine backup.

## Connect AVAILABLE

In the terminal used to start the pipeline or app:

```powershell
$env:QDRANT_URL = "http://localhost:6333"
$env:QDRANT_COLLECTION = "available2.0"
$env:QDRANT_ENTITY_COLLECTION = "available_entities"
```

The [index stage](pipeline.md#indexing-controls) creates the collections; do not
create empty stand-ins manually. The parent-document store and graph remain
filesystem artifacts and must match the collections. See
[configuration](../reference/configuration.md) for their paths.

For containerized AVAILABLE, `localhost` means the app container, not the host.
Choose a reviewed Docker network or reachable host endpoint; the loopback-only
binding above may not be reachable through `host.docker.internal` on every
Docker Desktop configuration. Do not broaden network exposure without checking
access control.

## Start, Stop, and Inspect

Run these from any PowerShell directory:

```powershell
docker start qdrant
docker ps --filter name=qdrant
docker logs --tail 80 qdrant
docker volume inspect qdrant_storage
```

Verify the API and collection list:

```powershell
Invoke-RestMethod -Uri "http://localhost:6333/collections"
```

Open the dashboard to inspect the expected collections and point counts. On a
fresh instance, an empty collection list is expected until indexing finishes.
Use counts from the current corpus build, not an old deployment record.

When finished:

```powershell
docker stop qdrant
```

Stopping Qdrant makes retrieval unavailable but keeps the volume. Do not remove
the volume or refresh collections as a routine restart procedure.

## Backup and Recovery

Before a collection refresh or upgrade, capture tested Qdrant snapshots and a
matching copy of the parent-document store, graph, trust outputs, and extracted
resources. Record the image version, collection names, embedding model, and build
identity. Restore to an isolated instance and test a known question before
switching the application. See [AKS backup guidance](../deployment/azure-aks.md#backup-restore-and-rollback)
for the same cross-store consistency requirements.

## Azure Deployment Is Separate

In the recorded AKS topology, Qdrant runs as a StatefulSet with an Azure Disk,
not this workstation's Docker volume. The app uses `http://qdrant:6333` inside
the cluster. See [Azure AKS deployment](../deployment/azure-aks.md); local Docker
commands do not start or repair the cluster instance.