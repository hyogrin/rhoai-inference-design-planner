from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.requests import Request
from starlette.responses import Response

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
    registry=registry,
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    registry=registry,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

DESIGN_SESSIONS_CREATED = Counter(
    "design_sessions_created_total",
    "Total design sessions created",
    registry=registry,
)

AGENT_INVOCATIONS = Counter(
    "agent_invocations_total",
    "Total agent invocations",
    ["status"],
    registry=registry,
)

LLM_REQUEST_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency in seconds",
    ["model"],
    registry=registry,
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)


async def metrics_endpoint(_request: Request) -> Response:
    data = generate_latest(registry)
    return Response(content=data, media_type="text/plain; charset=utf-8")
