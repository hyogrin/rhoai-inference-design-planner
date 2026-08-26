# AGENTS.md — Inference Design Planner

## Project Overview

Evidence-backed inference design planning for Red Hat OpenShift AI with vLLM.
This repository implements a wizard-driven application that gathers model metadata,
community evidence, and hardware constraints, then produces a structured deployment
recommendation with full provenance.

## Architecture Principles

1. **Domain JSON First**: The LLM produces only schema-constrained JSON.
   The backend validates, normalizes, enriches, and verifies before updating state.
   The frontend renders only predefined React components from a versioned view model.

2. **Deterministic Calculations**: Memory, capacity, cost, and feasibility are computed
   by pure functions in `estimators/`. The LLM explains tradeoffs but does not calculate.

3. **Evidence Provenance**: Every claim in a recommendation must trace back to an
   `EvidenceItem` with source URL, retrieval time, and verification level.

## Technology Stack

- Backend: Python 3.11+, FastAPI, Pydantic v2, LangGraph, PostgreSQL
- Frontend: Next.js 15, React 19, TypeScript, Tailwind CSS
- Protocol: AG-UI over SSE for streaming, REST for CRUD
- Deployment: Docker, OpenShift, Helm

## Key Directories

- `domain/` — Pydantic domain models (source of truth for all schemas)
- `estimators/` — Deterministic sizing engines (no LLM dependency)
- `connectors/` — External source adapters (HuggingFace, vLLM recipes, etc.)
- `agents/` — LangGraph orchestrator
- `backend/` — FastAPI application and database
- `frontend-next/` — Next.js wizard UI
- `tests/` — Test suite with model fixtures

## Development Commands

```bash
make dev           # Install all dependencies
make test          # Run tests
make lint          # Run linters
make generate-types # Generate TS types from Pydantic
make run-backend   # Start backend
make run-frontend  # Start frontend
```

## Conventions

- All domain types defined in Python first, TypeScript generated
- Use `extra="forbid"` on all Pydantic models
- Use structured logging (structlog)
- All estimators are pure functions with versioned inputs/outputs
- Tests do not require LLM access
- English for code, comments, and documentation
- Korean UI localization via i18n messages
