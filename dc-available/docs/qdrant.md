<!---

	Copyright (c) 2024 Robert Bosch GmbH and its subsidiaries.

-->

# Qdrant Setup

AVAILABLE uses [Qdrant](https://qdrant.tech/) as its vector store. The `index` stage writes
embeddings into it and the app reads from it at query time, so **Qdrant must be running before you
build the index or serve the app**.

## Run Qdrant locally (Docker)

```
docker run -d --name qdrant \
  -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

- `-p 6333:6333` exposes the REST/gRPC API.
- `-v qdrant_storage:/qdrant/storage` persists the collections across restarts.

Verify it's up by opening the dashboard at **http://localhost:6333/dashboard**.

Stop / start again later:

```
docker stop qdrant
docker start qdrant
```

## Configuration

The app reads these from the environment (defaults in `dc_available/config/data_config.py`):

| Variable                   | Default                 | Purpose                                  |
| -------------------------- | ----------------------- | ---------------------------------------- |
| `QDRANT_URL`               | `http://localhost:6333` | Qdrant endpoint                          |
| `QDRANT_COLLECTION`        | `available2.0`          | Main chunk collection (dense + sparse)   |
| `QDRANT_ENTITY_COLLECTION` | `available_entities`    | One point per graph entity (graph leg)   |

Both collections are created automatically by the `index` stage — you don't need to pre-create them.
Only set these variables if Qdrant runs elsewhere (e.g. a remote host or a Kubernetes service).
