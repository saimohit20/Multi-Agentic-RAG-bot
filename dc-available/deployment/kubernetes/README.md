# AVAILABLE — Azure (AKS) Deployment

This folder deploys the AVAILABLE RAG app to **Azure Kubernetes Service (AKS)**:
an **automated pipeline** (SharePoint → preprocess → index) that keeps the data
fresh, plus a **public Streamlit app** that answers questions.

> New to Kubernetes? Each YAML file has heavy comments explaining *what* and *why*.
> This README is the map + the runbook.

---

## Architecture

```
                         ┌───────────────────────── AKS cluster (namespace: available-2) ─────────────────────────┐
                         │                                                                                          │
  SharePoint  ──Graph──► │   CronJob: available-pipeline   (python -m pipeline.run_pipeline, hourly)                │
  (source docs)          │        sync → (changes?) → preprocess (only changed) → index                            │
                         │          │ writes                              │ writes vectors                          │
                         │          ▼                                     ▼                                          │
                         │   ┌──────────────┐  (Azure Files, RWX)   ┌──────────┐  (Azure Disk)                      │
                         │   │ available-   │◄────── reads ─────────│  Qdrant  │  StatefulSet                        │
                         │   │ data volume  │                       │  :6333   │                                     │
                         │   │ (mirror +    │                       └──────────┘                                     │
                         │   │  artifacts + │                             ▲                                          │
                         │   │  extracted)  │                             │ searches                                 │
                         │   └──────┬───────┘                             │                                          │
                         │          │ reads                     ┌─────────┴─────────┐                                │
                         │          └──────────────────────────►│  Deployment: serve │  (Streamlit :8501)            │
                         │                                       └─────────┬─────────┘                               │
                         └─────────────────────────────────────────────────┼─────────────────────────────────────┘
                                                                            │
                                                             Ingress (nginx) → PUBLIC URL  ◄── users
```

- **Qdrant** = vector search engine (needs a running server + its own disk).
- **available-data** = one shared Azure Files disk (RWX) holding `local_sharepoint/`,
  `Artifacts/`, `Extracted_resources/`.
- **serve** reads the artifacts + talks to Qdrant; **Ingress** exposes it publicly.

---

## Files

| File | Kind | Purpose |
|---|---|---|
| `namespace.yaml` | Namespace | groups everything under `available-2` |
| `mirror-pvc.yaml` | PVC | `available-data` — the shared data disk |
| `qdrant.yaml` | StatefulSet + Service | the vector DB at `qdrant:6333` |
| `sharepoint-secret.example.yaml` | Secret (template) | Graph creds for `sync` |
| `azure-openai-config.yaml` | ConfigMap | OpenAI endpoint + api version (no key) |
| `workload-identity-sa.yaml` | ServiceAccount | the pod's Azure identity for OpenAI |
| `sync-cronjob.yaml` | CronJob | runs **sync only** (test that stage) |
| `auto-pipeline-cronjob.yaml` | CronJob | runs **sync→preprocess→index** (production) |
| `serve.yaml` | Deployment + Service + Ingress | the public app |

Also in the repo: `Dockerfile.pipeline` (adds LibreOffice/ffmpeg), `pipeline/run_pipeline.py`
(the orchestrator), `pipeline/sync.py` (change report + `has_changes`).

---

## Authentication (two different styles)

| Service | How it authenticates | In Azure |
|---|---|---|
| **SharePoint** (sync) | client secret | a Kubernetes **Secret** (`available-sharepoint`) |
| **Azure OpenAI** (preprocess/index/serve) | Azure AD **identity** — no key | **Workload Identity** (`available-workload-identity` SA) |

---

## Prerequisites (create once, in Azure)

1. **Resource Group**, **ACR** (container registry), **AKS** cluster; attach ACR to AKS.
2. **nginx ingress controller** (Helm) — provides the public IP.
3. **Azure OpenAI** resource. Then a **User-Assigned Managed Identity**:
   - grant it role **“Cognitive Services OpenAI User”** on the OpenAI resource,
   - enable **OIDC + Workload Identity** on AKS,
   - add a **federated credential** linking it to ServiceAccount
     `available-2/available-workload-identity`.
4. A **SharePoint / Microsoft Graph app registration** (tenant + client id + client secret)
   with permission to read the AVAILABLE site.
5. A **DNS record** pointing your hostname at the ingress public IP.
6. Storage classes `azurefile-csi` + `managed-csi` are **built into AKS** (nothing to do).

---

## Placeholders to fill

| Placeholder | File | Value |
|---|---|---|
| `<your-acr-name>` | `auto-pipeline-cronjob.yaml`, `serve.yaml` | your ACR name |
| `AZURE_ENDPOINT` | `azure-openai-config.yaml` | your Azure OpenAI URL |
| `<managed-identity-client-id>` | `workload-identity-sa.yaml` | the identity's client id |
| `<your-app-hostname>` | `serve.yaml` | your public DNS name |
| Graph creds | (created via `kubectl create secret`) | tenant / client / secret / site / folder |

---

## Build & deploy (runbook)

```bash
# 0. Build the app image and push to ACR (once, and after code changes)
docker build -f Dockerfile.pipeline -t <acr>.azurecr.io/dc-available-pipeline:latest .
az acr login -n <acr>
docker push <acr>.azurecr.io/dc-available-pipeline:latest

# 1. Cluster credentials
az aks get-credentials -g <rg> -n <aks>

# 2. Apply the base objects
kubectl apply -f deployment/kubernetes/namespace.yaml
kubectl apply -f deployment/kubernetes/mirror-pvc.yaml
kubectl apply -f deployment/kubernetes/qdrant.yaml
kubectl apply -f deployment/kubernetes/azure-openai-config.yaml
kubectl apply -f deployment/kubernetes/workload-identity-sa.yaml

# 3. Create the SharePoint secret (real values — NOT the example file)
kubectl create secret generic available-sharepoint -n available-2 \
  --from-literal=GRAPH_TENANT_ID=... \
  --from-literal=GRAPH_CLIENT_ID=... \
  --from-literal=GRAPH_CLIENT_SECRET=... \
  --from-literal=AVAILABLE_SITE_ID=... \
  --from-literal=AVAILABLE_FOLDER_PATH=DC_DATA

# 4. Run the pipeline once to populate data + Qdrant
#    (either wait for the CronJob, or trigger it now:)
kubectl apply -f deployment/kubernetes/auto-pipeline-cronjob.yaml
kubectl create job -n available-2 --from=cronjob/available-pipeline pipeline-run-1

# 5. Once the pipeline job succeeds, start the app
kubectl apply -f deployment/kubernetes/serve.yaml
```

---

## Verify

```bash
kubectl get pods -n available-2          # qdrant + serve Running; pipeline job Completed
kubectl logs -n available-2 job/pipeline-run-1     # watch sync report + preprocess/index
kubectl get ingress -n available-2       # shows the public host/address
```
Then open the app URL and ask a question — you should get an answer with sources.

---

## How the automation works

- The CronJob runs `python -m pipeline.run_pipeline` on a schedule.
- **sync** compares SharePoint eTags vs a manifest → prints a change report and
  returns `has_changes`.
- **No changes → it stops after sync.**
- **Changes → preprocess** rebuilds only the changed trainings (source-signature
  skip), then **index** rebuilds Qdrant + graph + semantic.
- `serve` reads the refreshed `available-data` volume + Qdrant automatically.

---

## Notes / gotchas

- **Deploy `serve` after the pipeline has run once** — it refuses to start until the
  docstore + Qdrant collection exist (it CrashLoop-restarts until they do).
- **serve = 1 replica** — Streamlit keeps session state in memory; add sticky
  sessions before scaling up.
- **Index on change = full rebuild** (re-embeds all + re-runs gpt-4o semantic).
  Incremental index is a future optimization.
- **HTTPS, Opik, Key Vault, backups, autoscaling** = Phase 5 hardening (not here yet).
- Secrets never go in git — use `kubectl create secret` (or Azure Key Vault later).
