# AVAILABLE 2.0 — Azure AKS Deployment Guide

A plain-language record of how AVAILABLE 2.0 was deployed to Azure Kubernetes
Service (AKS), what each phase did, where we are now, and what's left.

_Last updated: 2026-09-21._

---

## 1. Big picture

AVAILABLE 2.0 is a Streamlit RAG app (LangChain + AutoGen). Retrieval = hybrid
Qdrant (dense + sparse BGE-M3) + a knowledge graph (PPR). One Docker image
(`Dockerfile.pipeline`) runs **both** the app (`python -m pipeline.serve`) and the
data pipeline (`python -m pipeline.run_pipeline`).

**Two storage "boxes" in the cluster:**
| Box | Kind | Holds |
|---|---|---|
| **Qdrant** | StatefulSet + Azure Disk | vectors: collections `available2.0` (chunks) + `available_entities` |
| **`available-data`** | PVC (Azure Files, RWX) | graph folder, docstore `index_ret.pkl`, images |

**Front door:** NGINX Ingress → the `serve` Service → the app pod.

### Azure facts
| Item | Value |
|---|---|
| Resource group | `AVAILABLE2.0` |
| AKS cluster | `available` (region germanywestcentral) |
| ACR (registry) | `acravailable.azurecr.io`, image `dc-available-pipeline:latest` |
| Namespace | `available-2` |
| Managed identity (app) | `available-identity` |
| Ingress external IP | `131.189.39.94` |
| Node resource group | `MC_AVAILABLE2.0_available_germanywestcentral` |

### Critical constraint — no local `kubectl`
The Bosch corporate proxy (`localhost:3128`) does TLS inspection, which breaks
`kubectl`'s certificate check against the AKS API. **Workaround used for every
kubectl step:** run it server-side via ARM:
```
az aks command invoke -g AVAILABLE2.0 -n available `
  --command "kubectl apply -f FILE.yaml" --file FILE.yaml
```
`az` (ARM) and Azure data-plane HTTPS work through the proxy; raw `kubectl` and
SMB (port 445) do not.

---

## 2. Phase 0–1 — Prerequisites & image (DONE before this guide)

- Created AKS, ACR (attached later — see blocker), NGINX ingress, managed
  identity + OIDC/Workload Identity, federated credential.
- Built and pushed the app image to ACR (`dc-available-pipeline:latest`).

---

## 3. Phase 2 — Foundation manifests (DONE)

Applied, in order, into namespace `available-2` (all via `az aks command invoke`):

| Step | Object | What it is | Azure cost |
|---|---|---|---|
| 1 | `namespace.yaml` | the `available-2` namespace (a folder) | €0 |
| 2 | `workload-identity-sa.yaml` | ServiceAccount the pods run as (keyless Azure auth) | €0 |
| 3 | federated credential | trust link: this SA ↔ `available-identity` | €0 |
| 4 | `azure-openai-config.yaml` | ConfigMap with the OpenAI endpoint + API version | €0 |
| 5 | `sharepoint-secret.local.yaml` | Secret with Microsoft Graph creds (for sync) | €0 |
| 6 | `qdrant.yaml` | Qdrant Service + StatefulSet + **20 Gi disk** | ~€2–3/mo |
| 7 | `mirror-pvc.yaml` | `available-data` PVC (Azure Files, RWX) | usage-based (~€0 empty) |

**Notes learned here:**
- The namespace was renamed mid-deploy `dc-available` → `available-2`. Because the
  federated credential subject hard-codes the namespace
  (`system:serviceaccount:available-2:available-workload-identity`), every manifest
  **and** the credential had to match. Lesson: pick the namespace once.
- SharePoint creds: sync reads `AVAILABLE_SITE_ID` + `AVAILABLE_FOLDER_PATH` (a
  Graph site id + a drive-relative folder). The site id was resolved from the
  SharePoint URL via a Graph call; the folder is everything after "Shared Documents/".

---

## 4. Phase 3 — Reuse local artifacts, no recompute (DONE)

Goal: load the app's data by **copying the artifacts already built locally**,
instead of re-running the pipeline in the cluster.

**A. Export Qdrant snapshots (local).** Snapshot each collection and download it:
`POST http://localhost:6333/collections/<name>/snapshots` → `.snapshot` file.
Result: `available2.0.snapshot` (107 MB), `available_entities.snapshot` (71 MB).

**B. Upload the data folder → PVC.** `dc_available/data/` (~595 MB: `Artifacts/`,
`Extracted_resources/`, handbook) was pushed straight into the Azure Files share
behind the PVC using `az storage file upload` (HTTPS, proxy-friendly). The two
snapshots were parked at `_snapshots/` on the same share.
- Gotcha: some image paths exceed Windows' 260-char limit → `az` (Python) can't
  open them. Fixed by `robocopy`-ing the tree to a short root (`C:\au`) first, then
  uploading from there. (`azcopy` would be nicer but its installer was blocked by
  the proxy; SMB/robocopy-direct needs port 445, also blocked.)

**C. Restore snapshots into cluster Qdrant.** A one-off Job
(`qdrant-restore-job.yaml`) mounts the PVC and POSTs each snapshot to
`http://qdrant:6333/collections/<name>/snapshots/upload`. Verified point counts:
`available2.0` = **9,825**, `available_entities` = **7,350**.

**D. Wire the collection name.** Local chunks live in `available2.0`, but the app
defaults to `available`. Decision: keep `available2.0` and set env
`QDRANT_COLLECTION=available2.0` in `serve.yaml` + `auto-pipeline-cronjob.yaml`
(sync doesn't use Qdrant). `available_entities` already matched the default.

---

## 5. Phase 4 — Deploy the app (IN PROGRESS)

Edited `serve.yaml`: set `image: acravailable.azurecr.io/dc-available-pipeline:latest`
and removed the ingress `host:` line so it matches any host (browse via the raw IP).
Applied it → Deployment + Service + Ingress created.

- **Image validated locally:** running the image with local Qdrant + the data
  folder mounted showed `[Serve] Vector index ready: collection 'available2.0'`,
  Streamlit started, BGE-M3 embedding model loaded. The image is good.
- **serve startup guard** ([pipeline/serve.py]) requires the docstore
  `index_ret.pkl` **and** the Qdrant collection to exist — both present.

### ⛔ Current blocker
The cluster can't pull the image: **`401 Unauthorized` from ACR** — the cluster's
kubelet identity lacks the `AcrPull` role, and `az aks update --attach-acr` failed
because the current user isn't an Owner. **An admin must grant this.**

---

## 6. Two permission gates (need an admin/Owner)

Both are "assign a role" tasks, done in the Portal via **Access control (IAM) →
Add role assignment → Managed identity**:

| Where (resource) | Role to grant | Grant to (identity) | Unblocks |
|---|---|---|---|
| ACR **`acravailable`** | **AcrPull** (or "Container Registry Repository Reader") | kubelet MI **`available-agentpool`** | pod can pull the image → **starts** |
| **Azure OpenAI** resource | **Cognitive Services OpenAI User** | MI **`available-identity`** | app can **answer** with gpt-4o |

CLI equivalent for the first (if the admin prefers):
`az aks update -g AVAILABLE2.0 -n available --attach-acr acravailable`.

---

## 6b. Identities & authentication (keyless) — explained simply

Think of each protected resource (the registry, the AI model) as a **locked door**.
Two ways in:
- **API key** = a shared password. Works, but anyone with the string gets in (less
  secure). The Azure AI Foundry resource has key auth **disabled on purpose** — do
  **not** enable it.
- **Managed identity (Azure AD)** = an **ID badge**. A resource carries a badge and
  shows it at the door; nothing to leak or rotate. **This is what we use.**

A **role assignment** is the **guest list**: it says "this badge is allowed through
this door." A badge with no guest-list entry gets rejected — which is exactly the two
blockers above.

### There are TWO different identities — don't mix them up
| Identity | Type | It's the badge for… | Gets which role, where |
|---|---|---|---|
| **`available-agentpool`** | AKS kubelet managed identity | the **cluster** pulling the app image | `AcrPull` on ACR `acravailable` |
| **`available-identity`** | User-assigned managed identity | the **app pod** calling the AI model | `Cognitive Services OpenAI User` on the Azure AI Foundry resource |

`available-identity` is a standalone **Managed Identity** resource (Portal → search
**Managed Identities**, in RG `AVAILABLE2.0`, client-id `1136bd00-…`). It is **not**
part of App Service — this app runs on **AKS**. The pod "wears" this identity via
**Workload Identity** (the ServiceAccount `available-workload-identity` + the
federated credential set up in Phase 2).

### How the app reaches the model (the flow)
1. Models are deployed in **Azure AI Foundry** (backed by an Azure AI Services /
   Cognitive Services resource).
2. The app calls the model at the **OpenAI-compatible endpoint**
   (`…cognitiveservices.azure.com/`, in the `available-openai` ConfigMap) using
   `DefaultAzureCredential` — i.e. it shows the **`available-identity` badge**
   ([dc_available/config/llm_config.py]).
3. The **role** `Cognitive Services OpenAI User` puts that badge on the model's guest
   list → the call is authorized → gpt-4o answers. **No API key involved.**

> If a Foundry model were **non-OpenAI** (Llama/Mistral via the inference API), the
> role would be `Cognitive Services User` / `Azure AI Developer` instead. We use
> **gpt-4o**, so `Cognitive Services OpenAI User` is correct.

---

## 7. What's left

**Finish Phase 4 (after ACR role):**
1. Force the pod to retry: `kubectl delete pod -n available-2 -l app=serve`.
2. Watch it: `kubectl get pods -n available-2 -l app=serve` → `1/1 Running`.
3. Logs: `kubectl logs -n available-2 deploy/serve` → "Vector index ready".
4. `kubectl get ingress -n available-2` → ADDRESS `131.189.39.94`.
5. Open **http://131.189.39.94**. UI + retrieval work; answering needs the OpenAI role.

**Phase 5 (future hardening):**
- Deploy the pipeline **cronjobs** (`auto-pipeline-cronjob.yaml`) for auto-refresh
  from SharePoint. Deferred for now because it would rebuild `available2.0`
  (overwriting the reused data) and needs the OpenAI role.
- Pre-seed the **BGE-M3 model** into the PVC (`.hf_cache`) so the pod doesn't
  download ~2 GB on first start.
- HTTPS/TLS (cert-manager), Opik tracing, secrets → Azure Key Vault, backups,
  autoscaling.

---

## 8. Handy commands (all proxy-safe)

```powershell
# Any kubectl action:
az aks command invoke -g AVAILABLE2.0 -n available --command "kubectl get pods -n available-2"

# Pod logs:
az aks command invoke -g AVAILABLE2.0 -n available --command "kubectl logs -n available-2 deploy/serve --tail=40"

# Check Qdrant collections (throwaway curl pod):
az aks command invoke -g AVAILABLE2.0 -n available --command "kubectl run qcheck -n available-2 --image=curlimages/curl:8.10.1 --restart=Never -i --rm -- curl -sS http://qdrant:6333/collections"

# Save money while waiting (preserves everything):
az aks stop  -g AVAILABLE2.0 -n available
az aks start -g AVAILABLE2.0 -n available
```

**Security note:** `sharepoint-secret.local.yaml` and any storage keys are secrets —
never commit them. The `*.example.yaml` template is safe to share.
```
## Deployment command

az aks command invoke -g <rg> -n <aks-cluster> \
  --command "kubectl create job --from=cronjob/available-pipeline dc-run-1 -n available-2"

