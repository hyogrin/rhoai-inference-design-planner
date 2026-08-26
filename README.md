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

## License

Apache-2.0
