# Inference Design Planner

Evidence-backed inference design planning for Red Hat OpenShift AI. This application helps you determine optimal GPU topology, serving configuration, and cost estimates for deploying LLM inference workloads on OpenShift AI with vLLM.

## Features

- **Model Analysis**: Fetch and normalize Hugging Face model architecture metadata
- **Evidence Discovery**: Gather vLLM recipes, Red Hat evaluations, and community evidence
- **Validation Gate**: Assess readiness with structured checks and remediation guidance
- **Workload Profiling**: Structured input for traffic, latency SLOs, and operational requirements
- **Deterministic Sizing**: Memory, capacity, and cost calculations with full provenance
- **Recommendation Dashboard**: Deployment topology, parallelism, and benchmark plans

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Next.js 15 UI  │────▶│  FastAPI Backend  │────▶│   PostgreSQL    │
│  (Wizard + Chat)│◀────│  (AG-UI / REST)   │◀────│  (State + Ckpt) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │       │
                    ┌─────────┘       └─────────┐
                    ▼                             ▼
           ┌──────────────┐            ┌──────────────────┐
           │  LangGraph   │            │  MCP Web Search  │
           │  Orchestrator│            │  (FastMCP / SSE) │
           └──────────────┘            └──────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- PostgreSQL 15+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Install dependencies
make dev

# Copy environment config
cp sample.env .env

# Run database migrations
make db-migrate

# Start backend
make run-backend

# Start frontend (in another terminal)
make run-frontend
```

### Docker Compose

```bash
make docker-build
make docker-up
```

### Testing

```bash
make test          # Run all tests
make test-cov      # Run with coverage
make smoke-test    # Run smoke tests only
make lint          # Run linters
make type-check    # Run type checking
```

## Project Structure

```
├── agents/                  # LangGraph orchestrator
│   └── inference_planner/
├── backend/                 # FastAPI application
├── connectors/              # External source adapters
├── domain/                  # Pydantic domain models
├── estimators/              # Deterministic sizing engines
├── mcp_servers/             # MCP tool servers
├── frontend-next/           # Next.js 15 frontend
├── templates/               # Config generation templates
├── deploy/                  # OpenShift/Helm deployment
├── scripts/                 # Development utilities
└── tests/                   # Test suite
```

## Cluster Deployment

The system deploys to OpenShift as a **single Helm release** — backend, frontend,
and PostgreSQL all run in one namespace.

### Deployment Options

| Option | When to Use | Command |
|---|---|---|
| **Local Build + Quay.io** | You have a container registry | See below |
| **Docker Compose** | Local development/testing | `make docker-up` |

### Quick Deploy (TL;DR)

```bash
# 1. Build images (use --platform linux/amd64 for x86 clusters)
docker build --platform linux/amd64 -f Dockerfile.backend \
  -t quay.io/<your-org>/inference-planner-backend:latest .

docker build --platform linux/amd64 \
  --build-arg NEXT_PUBLIC_API_URL=https://inference-planner-api.<domain> \
  --build-arg NEXT_PUBLIC_AGENT_URL=https://inference-planner-api.<domain>/agent \
  -t quay.io/<your-org>/inference-planner-frontend:latest frontend-next/

# 2. Push to registry
docker push quay.io/<your-org>/inference-planner-backend:latest
docker push quay.io/<your-org>/inference-planner-frontend:latest

# 3. Deploy with Helm
oc login https://api.your-cluster.example.com:6443
oc new-project inference-planner
helm install inference-planner deploy/helm/inference-design-planner -n inference-planner
```

### What Gets Deployed

| Component | Replicas | Description |
|---|---|---|
| `backend` | 1 | LangGraph orchestrator + FastAPI REST/SSE |
| `frontend` | 1 | Next.js wizard UI |
| `postgresql` | 1 | Sessions, checkpointing, recommendations |
| `db-migrate` (Job) | 1 | Alembic schema migration |

### Access URLs

| Service | URL |
|---|---|
| Frontend | `https://inference-planner.<cluster-domain>` |
| Backend API | `https://inference-planner-api.<cluster-domain>` |

**Detailed deployment instructions: [deploy/DEPLOYMENT.md](deploy/DEPLOYMENT.md)**

## System Ports

| Service | Port | Protocol | Description |
|---|---|---|---|
| Next.js Frontend | 3000 (dev: 7000) | HTTP | Wizard UI (AG-UI runtime) |
| FastAPI Backend | 8000 (dev: 7001) | HTTP + SSE | API server + LangGraph agent |
| PostgreSQL | 5432 | TCP | Sessions, checkpointing, recommendations |

## License

Apache-2.0
