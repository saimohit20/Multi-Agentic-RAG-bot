# Local Docker

The shared pipeline image contains the application and the system tools used for
extraction. Building or running it is optional; the native local workflow is
documented in [Getting Started](../getting-started/index.md).

## Build Locally

From `dc-available`:

```powershell
docker build -f Dockerfile.pipeline -t available-pipeline:local .
```

This command builds a local image only. It does not publish to a registry or
update any deployment. Docker still needs access to package repositories and
image registries to fetch build dependencies.

## Corporate Proxy Troubleshooting

If the APT step reports that `host.docker.internal` cannot be resolved and then
cannot find LibreOffice, investigate proxy/DNS access before changing package
names. In the previously verified Windows setup, the corporate proxy listened
on port `3128` and Docker Desktop used `http://127.0.0.1:3128` as its upstream
HTTP/HTTPS proxy. The corresponding build used:

```powershell
docker build --add-host=host.docker.internal:host-gateway --build-arg HTTP_PROXY=http://host.docker.internal:3128 --build-arg HTTPS_PROXY=http://host.docker.internal:3128 -f Dockerfile.pipeline -t available-pipeline:local .
```

These addresses are environment-specific. The build container's `127.0.0.1`
is not the Windows host. Do not disable TLS validation or embed proxy credentials
in a Dockerfile to bypass connectivity failures.

## Run Against Prepared Artifacts

First prepare the data using the native workflow. Keep local Qdrant running and
provide an approved container-compatible Azure credential mechanism. A host
`az login` session is not automatically inherited by the image.

For Docker Desktop, after providing a private `container.local.env` containing
your approved runtime configuration:

```powershell
$data = (Resolve-Path .\dc_available\data).Path
docker run --rm --name available-local -p 127.0.0.1:8502:8501 --env-file .\container.local.env -e AVAILABLE_APP_SOURCE=available -e QDRANT_URL=http://host.docker.internal:6333 -e QDRANT_COLLECTION=available2.0 -e STREAMLIT_SERVER_ADDRESS=0.0.0.0 -e STREAMLIT_SERVER_PORT=8501 --mount "type=bind,source=$data,target=/app/dc_available/data" -v available_hf_cache:/root/.cache/huggingface available-pipeline:local
```

Use this only when Qdrant is reachable from the container at that address; a
loopback-only host binding may require a reviewed Docker network arrangement.
Keep the environment file out of Git and do not place secrets in command history.
Open <http://localhost:8502> once the launcher reports successful startup.

The bind mount supplies the parent-document store, graph, extracted media, and
handbook. Qdrant data is separate. The model-cache volume avoids repeating large
downloads. Container authentication and network reachability must be validated
before treating this as a complete first-run path.

## Existing Compose File

`docker-compose.pipeline.yml` defines preprocess, index, and serve services, but
its mounts and comments still reference an older `resources`/FAISS layout while
the active `available` source uses `dc_available/data`. It also does not provision
Qdrant. Do not use the unmodified Compose file as the canonical setup recipe.
This documentation work does not change it or any deployment manifests.