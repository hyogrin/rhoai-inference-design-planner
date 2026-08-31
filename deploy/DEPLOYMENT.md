# Deployment Guide

This guide covers the complete process for deploying the **Inference Design Planner**
to an OpenShift cluster. The system consists of three components — a FastAPI/LangGraph
backend, a Next.js frontend, and a PostgreSQL database — deployed as a single Helm release.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Architecture Overview](#architecture-overview)
- [Environment Configuration](#environment-configuration)
- [Step 1: Build Container Images](#step-1-build-container-images)
- [Step 2: Push to Registry](#step-2-push-to-registry)
- [Step 3: Deploy with Helm](#step-3-deploy-with-helm)
- [Post-Deployment Verification](#post-deployment-verification)
- [MLflow Integration](#mlflow-integration)
- [Updating an Existing Deployment](#updating-an-existing-deployment)
- [Troubleshooting](#troubleshooting)
- [Cleanup](#cleanup)

---

## Prerequisites

### Cluster Requirements

| Component | Version | How to Verify |
|---|---|---|
| OpenShift | 4.17+ | `oc version` |
| RHOAI Operator | 3.4+ | `oc get csv -n redhat-ods-operator` |
| Helm | 3+ | `helm version` |
| LLM Model Serving | Running | `oc get inferenceservice -A` |

### Workstation Tools

| Tool | Version | Purpose |
|---|---|---|
| `oc` | 4.17+ | OpenShift CLI |
| `helm` | 3+ | Kubernetes package manager |
| `docker` / `podman` | Latest | Container image builds |
| `node` | 20+ | Frontend builds |
| `uv` | 0.4+ | Python package manager |

### Required Model Endpoint

The planner requires one LLM endpoint (OpenAI-compatible chat/completions API):

```bash
# Direct vLLM route (recommended — avoids MaaS non-streaming bug)
oc get inferenceservice <model-name> -n <namespace> \
  -o jsonpath='https://{.status.url}/v1'

# Or create a direct Route to the workload service
oc get svc -n <namespace> | grep workload
oc expose svc/<model>-kserve-workload-svc -n <namespace> --name=<model>-direct
```

> **Important:** The MaaS API gateway has a known bug where non-streaming responses
> return empty bodies (Kuadrant WASM plugin drops the response). Use a **direct Route
> to the vLLM pod** for reliable non-streaming LLM calls.

---

## Architecture Overview

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────┐
│   Frontend       │──────│   Backend         │──────│  PostgreSQL  │
│   (Next.js)      │ REST │   (FastAPI +      │ SQL  │  (15-alpine) │
│   Port 3000      │ SSE  │    LangGraph)     │      │  Port 5432   │
└─────────────────┘      │   Port 8000       │      └─────────────┘
                          └────────┬─────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
              ┌─────▼────┐  ┌─────▼────┐  ┌─────▼────┐
              │ HuggingFace│  │   vLLM    │  │  SearXNG  │
              │   Hub API  │  │  (LLM)   │  │  (Search) │
              └───────────┘  └──────────┘  └──────────┘
```

### What Gets Deployed

| Component | Replicas | Description |
|---|---|---|
| `backend` | 1 | LangGraph orchestrator + FastAPI REST/SSE (port 8000) |
| `frontend` | 1 | Next.js wizard UI (port 3000) |
| `postgresql` | 1 | Sessions, checkpointing, recommendations (port 5432) |
| `db-migrate` (Job) | 1 | Alembic schema migration (runs post-install/upgrade) |

### Container Images

| Image | Source | Description |
|---|---|---|
| `inference-planner-backend` | `Dockerfile.backend` | Python 3.11 + FastAPI + LangGraph + connectors |
| `inference-planner-frontend` | `frontend-next/Dockerfile` | Next.js 15 standalone build |

---

## Environment Configuration

### Local Development

```bash
cp sample.env .env
# Edit .env with your values
```

### Key Variables

```bash
# --- LLM (OpenAI-compatible) ---
OPENAI_API_KEY=not-needed                    # or actual key if using OpenAI/MaaS
OPENAI_BASE_URL=https://your-vllm-route/v1  # Direct vLLM route recommended
LLM_MODEL_NAME=your-model-name

# --- Hugging Face (optional, for gated models) ---
HF_TOKEN=hf_xxxxx

# --- PostgreSQL ---
DATABASE_URL=postgresql+asyncpg://planner:planner@localhost:5432/inference_planner
DATABASE_URL_SYNC=postgresql://planner:planner@localhost:5432/inference_planner

# --- Web Search ---
SEARXNG_URL=https://your-searxng-instance

# --- MLflow Observability ---
MLFLOW_TRACKING_URI=https://your-mlflow-endpoint/mlflow
MLFLOW_TRACKING_TOKEN=sha256~xxxxx
MLFLOW_WORKSPACE=inference-planner

# --- Frontend ---
NEXT_PUBLIC_API_URL=http://localhost:7001
NEXT_PUBLIC_AGENT_URL=http://localhost:7001/agent
```

> **Two LLM access modes:**
>
> | Mode | `OPENAI_BASE_URL` | Notes |
> |---|---|---|
> | Direct vLLM route | `https://<model>-direct.<domain>/v1` | Recommended — avoids MaaS bug |
> | MaaS API gateway | `https://maas.<domain>/<ns>/<model>/v1` | Use streaming mode only |

---

## Step 1: Build Container Images

> **Cross-architecture note:** If building on Apple Silicon (ARM64) for an x86_64 cluster,
> always use `--platform linux/amd64`. Without this flag, pods will fail with `exec format error`.

### Backend

```bash
docker build --platform linux/amd64 \
  -f Dockerfile.backend \
  -t quay.io/<your-org>/inference-planner-backend:latest .
```

### Frontend

The frontend bakes `NEXT_PUBLIC_*` variables at build time. You **must** pass them as build args:

```bash
DOMAIN=apps.your-cluster.example.com
NAMESPACE=inference-planner

docker build --platform linux/amd64 \
  --build-arg NEXT_PUBLIC_API_URL=https://inference-planner-api.${DOMAIN} \
  --build-arg NEXT_PUBLIC_AGENT_URL=https://inference-planner-api.${DOMAIN}/agent \
  -t quay.io/<your-org>/inference-planner-frontend:latest \
  frontend-next/
```

> **This is critical:** Next.js `NEXT_PUBLIC_*` variables are embedded during `npm run build`.
> Runtime environment variables do NOT override them. If you change the namespace or domain,
> you must **rebuild the frontend image**.

---

## Step 2: Push to Registry

```bash
# Login to Quay.io
docker login quay.io

# Push both images
docker push quay.io/<your-org>/inference-planner-backend:latest
docker push quay.io/<your-org>/inference-planner-frontend:latest
```

> **Private registries:** If your Quay.io repositories are private, create an image pull
> secret on the cluster (the Helm chart references `quay-pull-secret` by default):
>
> ```bash
> oc create secret generic quay-pull-secret \
>   --from-file=.dockerconfigjson=$HOME/.docker/config.json \
>   --type=kubernetes.io/dockerconfigjson \
>   -n <namespace>
>
> oc secrets link default quay-pull-secret --for=pull -n <namespace>
> ```

---

## Step 3: Deploy with Helm

### 3.1 Login to Cluster

```bash
oc login https://api.your-cluster.example.com:6443 \
  -u kubeadmin -p <password> --insecure-skip-tls-verify=true
```

### 3.2 Create Namespace

```bash
oc new-project inference-planner
```

### 3.3 Configure values.yaml

Edit `deploy/helm/inference-design-planner/values.yaml`:

```yaml
namespace: inference-planner
domain: apps.your-cluster.example.com

backend:
  replicas: 1
  image: quay.io/<your-org>/inference-planner-backend:latest
  port: 8000
  resources:
    requests: { cpu: 250m, memory: 512Mi }
    limits:   { cpu: "1",  memory: 1Gi }

frontend:
  replicas: 1
  image: quay.io/<your-org>/inference-planner-frontend:latest
  port: 3000
  resources:
    requests: { cpu: 100m, memory: 256Mi }
    limits:   { cpu: 500m, memory: 512Mi }

postgresql:
  enabled: true
  image: postgres:15-alpine
  storage: 5Gi
  user: planner
  password: <change-me>
  database: inference_planner

config:
  APP_ENV: production
  APP_LOG_LEVEL: INFO
  APP_CORS_ORIGINS: "https://inference-planner.apps.your-cluster.example.com"
  LLM_MODEL_NAME: your-model-name
  VERIFY_SSL: "false"
  MLFLOW_TRACKING_URI: "https://your-mlflow-endpoint/mlflow"
  MLFLOW_TRACKING_INSECURE_TLS: "true"
  MLFLOW_WORKSPACE: inference-planner
  MLFLOW_EXPERIMENT_NAME: inference-design-planner
  PROMETHEUS_ENABLED: "true"
  SEARXNG_URL: "https://your-searxng-instance"

secret:
  OPENAI_API_KEY: not-needed
  OPENAI_BASE_URL: "https://your-vllm-direct-route/v1"
  DATABASE_URL: "postgresql+asyncpg://planner:<password>@postgresql:5432/inference_planner"
  DATABASE_URL_SYNC: "postgresql://planner:<password>@postgresql:5432/inference_planner"
  MLFLOW_TRACKING_TOKEN: "sha256~xxxxx"
  # HF_TOKEN: "hf_xxxxx"  # Uncomment for gated model access
```

### 3.4 Install

```bash
helm install inference-planner deploy/helm/inference-design-planner \
  -n inference-planner
```

### 3.5 Verify Pods

```bash
oc get pods -n inference-planner
```

Expected output:

```
NAME                          READY   STATUS      RESTARTS   AGE
backend-xxx                   1/1     Running     0          2m
frontend-xxx                  1/1     Running     0          2m
postgresql-xxx                1/1     Running     0          2m
db-migrate-1-xxx              0/1     Completed   0          90s
```

---

## Post-Deployment Verification

### Access URLs

| Service | URL |
|---|---|
| Frontend | `https://inference-planner.<domain>` |
| Backend API | `https://inference-planner-api.<domain>` |
| Health Check | `https://inference-planner-api.<domain>/api/v1/health` |

### Verify Endpoints

```bash
DOMAIN=$(oc get ingresses.config/cluster -o jsonpath='{.spec.domain}')

# Backend health
curl -sk https://inference-planner-api.${DOMAIN}/api/v1/health
# {"status":"healthy"}

# Frontend (should return HTML)
curl -sk https://inference-planner.${DOMAIN}/ | head -5
```

### Check Backend Logs

```bash
oc logs deployment/backend -n inference-planner --tail=30
```

Look for:
- `application_startup` — application ready
- `langgraph_checkpointer_initialized` — database connected
- No `PermissionError` or `Illegal header value` errors

---

## MLflow Integration

The backend integrates with MLflow managed by OpenShift AI for experiment tracking.

### Configuration

```yaml
# In values.yaml → config section
MLFLOW_TRACKING_URI: "https://rh-ai.apps.your-cluster.example.com/mlflow"
MLFLOW_TRACKING_INSECURE_TLS: "true"
MLFLOW_WORKSPACE: inference-planner

# In values.yaml → secret section
MLFLOW_TRACKING_TOKEN: "sha256~xxxxx"  # OpenShift service account token
```

### RBAC Setup (if using kubernetes-namespaced auth)

```bash
oc create rolebinding backend-mlflow-integration \
  --clusterrole=mlflow-operator-mlflow-integration \
  --serviceaccount=inference-planner:default \
  -n inference-planner
```

### Verify MLflow Connectivity

```bash
oc logs deployment/backend -n inference-planner | grep -i mlflow

# Success: "MLflow tracing enabled"
# Error:   "MLflow server unreachable" → check token/URI/RoleBinding
```

---

## Updating an Existing Deployment

### Code Change → Rebuild → Redeploy

```bash
# 1. Rebuild backend
docker build --platform linux/amd64 \
  -f Dockerfile.backend \
  -t quay.io/<your-org>/inference-planner-backend:latest .
docker push quay.io/<your-org>/inference-planner-backend:latest

# 2. Restart pod (pulls new image due to imagePullPolicy: Always)
oc rollout restart deployment/backend -n inference-planner

# 3. For frontend changes (must re-bake NEXT_PUBLIC_* URLs)
docker build --platform linux/amd64 \
  --build-arg NEXT_PUBLIC_API_URL=https://inference-planner-api.${DOMAIN} \
  --build-arg NEXT_PUBLIC_AGENT_URL=https://inference-planner-api.${DOMAIN}/agent \
  -t quay.io/<your-org>/inference-planner-frontend:latest \
  frontend-next/
docker push quay.io/<your-org>/inference-planner-frontend:latest
oc rollout restart deployment/frontend -n inference-planner
```

### Configuration Change Only

```bash
# Edit values.yaml, then:
helm upgrade inference-planner deploy/helm/inference-design-planner \
  -n inference-planner

# Or patch individual resources:
oc patch configmap planner-config -n inference-planner \
  --type merge -p '{"data":{"APP_LOG_LEVEL":"DEBUG"}}'
oc rollout restart deployment/backend -n inference-planner
```

### Database Migration

Alembic migrations run automatically via a Helm hook Job on `helm install` / `helm upgrade`.
To run manually:

```bash
oc exec deployment/backend -n inference-planner -- alembic upgrade head
```

---

## Troubleshooting

### Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| `ImagePullBackOff` | Private registry, no pull secret | Create `quay-pull-secret` and link to `default` SA |
| `exec format error` | ARM image on x86 cluster | Rebuild with `--platform linux/amd64` |
| `Illegal header value b'Bearer '` | Empty `HF_TOKEN` in Secret | Remove `HF_TOKEN` from Secret or set a valid token |
| `Permission denied: '/.cache'` | HuggingFace cache dir not writable | Ensure `HF_HOME=/tmp/hf_cache` in ConfigMap |
| `result_snapshot does not exist` | DB migration not applied | Run `oc exec deployment/backend -- alembic upgrade head` |
| Frontend shows `localhost` URLs | `NEXT_PUBLIC_*` not set at build time | Rebuild frontend with correct `--build-arg` values |
| LLM empty body / JSON error | MaaS non-streaming bug | Use direct vLLM Route instead of MaaS gateway |
| `initdb: directory not empty` | PostgreSQL PVC `lost+found` | Set `PGDATA=/var/lib/postgresql/data/pgdata` (already in chart) |
| MLflow `UNAUTHENTICATED` | Missing token or RoleBinding | Check `MLFLOW_TRACKING_TOKEN` and create RoleBinding |
| `db-migrate` Job failed | DB URL mismatch | Verify `DATABASE_URL_SYNC` in Secret matches PostgreSQL |

### Checking Logs

```bash
# Backend
oc logs deployment/backend -n inference-planner -f

# Frontend
oc logs deployment/frontend -n inference-planner -f

# PostgreSQL
oc logs deployment/postgresql -n inference-planner -f

# Migration Job
oc logs job/db-migrate-<revision> -n inference-planner
```

### Restarting Components

```bash
# Single component
oc rollout restart deployment/backend -n inference-planner

# All components
oc rollout restart deployment -n inference-planner
```

---

## Helm Chart Reference

The Helm chart lives at `deploy/helm/inference-design-planner/`.

### Templates

| Template | Resources Created |
|---|---|
| `backend-deployment.yaml` | Deployment, Service, Route (API) |
| `frontend-deployment.yaml` | Deployment, Service, Route (UI) |
| `postgresql.yaml` | PVC, Deployment, Service |
| `configmap.yaml` | ConfigMap (`planner-config`) |
| `secret.yaml` | Secret (`planner-secrets`) |
| `db-migrate-job.yaml` | Job (Alembic migration, Helm hook) |

### Values Reference

| Key | Default | Description |
|---|---|---|
| `namespace` | `inference-planner` | Target OpenShift namespace |
| `domain` | — | Cluster apps domain (e.g. `apps.cluster.example.com`) |
| `backend.image` | — | Backend container image |
| `backend.port` | `8000` | Backend service port |
| `frontend.image` | — | Frontend container image |
| `frontend.port` | `3000` | Frontend service port |
| `postgresql.enabled` | `true` | Deploy PostgreSQL |
| `postgresql.storage` | `5Gi` | PVC size |
| `postgresql.user` | `planner` | DB username |
| `postgresql.password` | `planner` | DB password |
| `postgresql.database` | `inference_planner` | DB name |
| `config.*` | — | Non-sensitive env vars → ConfigMap |
| `secret.*` | — | Sensitive env vars → Secret |

---

## Cleanup

```bash
# Remove Helm release + all resources
helm uninstall inference-planner -n inference-planner

# Delete namespace entirely
oc delete project inference-planner

# Clean up PVC (if namespace is kept)
oc delete pvc postgresql-data -n inference-planner
```
