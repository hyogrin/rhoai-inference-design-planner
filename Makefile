.PHONY: help install dev lint format test test-cov type-check generate-types run-backend run-frontend docker-build docker-up docker-down clean db-migrate smoke-test

PYTHON := python3
UV := uv
NPM := npm

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	$(UV) sync
	cd frontend-next && $(NPM) install

dev: ## Install with dev dependencies
	$(UV) sync --all-extras
	cd frontend-next && $(NPM) install

lint: ## Run linters
	$(UV) run ruff check .
	cd frontend-next && $(NPM) run lint

format: ## Format code
	$(UV) run ruff format .
	cd frontend-next && $(NPM) run format

test: ## Run backend tests
	$(UV) run pytest tests/ -v

test-cov: ## Run tests with coverage
	$(UV) run pytest tests/ -v --cov=domain --cov=estimators --cov=backend --cov-report=term-missing

type-check: ## Run type checking
	$(UV) run mypy domain/ backend/ estimators/

generate-types: ## Generate TypeScript types from Pydantic schemas
	$(UV) run python scripts/generate_types.py
	cd frontend-next && $(NPM) run format

run-backend: ## Run backend dev server
	$(UV) run uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000

run-frontend: ## Run frontend dev server
	cd frontend-next && $(NPM) run dev

docker-build: ## Build Docker images
	docker compose build

docker-up: ## Start all services
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

db-migrate: ## Run database migrations
	$(UV) run alembic upgrade head

smoke-test: ## Run local smoke test
	@echo "Running smoke tests..."
	$(UV) run pytest tests/ -v -m "smoke" --tb=short

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .ruff_cache .mypy_cache htmlcov .coverage
