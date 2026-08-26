#!/usr/bin/env bash
set -euo pipefail

echo "=== Inference Design Planner — Setup ==="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.11+."
    exit 1
fi

# Check uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Install Python deps
echo "Installing Python dependencies..."
uv sync

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "WARNING: Node.js not found. Frontend will not build."
else
    echo "Installing frontend dependencies..."
    cd frontend-next && npm install && cd ..
fi

# Setup PostgreSQL
if command -v docker &> /dev/null; then
    echo "Starting PostgreSQL via Docker..."
    docker compose up -d postgres
    echo "Waiting for PostgreSQL to be ready..."
    sleep 3
    echo "Running migrations..."
    uv run alembic upgrade head 2>/dev/null || echo "Migrations skipped (tables may already exist)"
elif command -v psql &> /dev/null; then
    echo "PostgreSQL CLI available. Please ensure the database exists:"
    echo "  createdb -U planner inference_planner"
    echo "Then run: make db-migrate"
else
    echo "WARNING: Neither Docker nor psql found. Database setup skipped."
    echo "Please provision PostgreSQL manually and set DATABASE_URL in .env"
fi

# Copy env if not exists
if [ ! -f .env ]; then
    cp sample.env .env
    echo "Created .env from sample.env — please update with your values."
fi

echo ""
echo "=== Setup Complete ==="
echo "  Start backend:  make run-backend"
echo "  Start frontend: make run-frontend"
echo "  Run tests:      make test"
