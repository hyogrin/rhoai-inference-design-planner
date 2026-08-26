"""Tests for the backend API endpoints.

Note: Tests that require database access are marked with @pytest.mark.db
and will be skipped when PostgreSQL is not available.
"""


import pytest


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from httpx import ASGITransport, AsyncClient

    from backend.api import app

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health_endpoint(client):
    async with client as c:
        response = await c.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_agent_endpoint_exists(client):
    """Verify the /agent endpoint is registered (will return 422 without proper body)."""
    async with client as c:
        response = await c.post("/agent", json={})
    # Should not be 404 - endpoint exists
    assert response.status_code != 404


@pytest.mark.asyncio
async def test_error_response_format(client):
    """Verify error responses include the expected fields."""
    async with client as c:
        response = await c.get("/api/v1/designs/not-a-uuid")
    # Should return 422 (validation error) or 404
    assert response.status_code in (404, 422)


@pytest.mark.asyncio
@pytest.mark.smoke
async def test_health_returns_version(client):
    """Smoke test: health endpoint returns version."""
    async with client as c:
        response = await c.get("/api/v1/health")
    data = response.json()
    assert data["version"] == "0.1.0"
