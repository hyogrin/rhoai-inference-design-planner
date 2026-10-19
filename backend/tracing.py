"""MLflow tracing initialisation and JSONL trace export utilities."""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TRACES_DIR = Path(__file__).resolve().parent.parent / "data" / "traces"
DESIGN_SUGGESTIONS_JSONL = TRACES_DIR / "design_suggestions.jsonl"

_jsonl_lock = threading.Lock()
_tracing_enabled = False


def is_tracing_enabled() -> bool:
    return _tracing_enabled


def _infer_mlflow_from_openai_url(base_url: str) -> tuple[str, str]:
    """Derive MLflow tracking URI and workspace from LLM_BASE_URL.

    Convention for RHOAI MaaS gateway:
        LLM_BASE_URL = https://maas.apps.<cluster>/<workspace>/<model>/v1
        MLflow URI       = https://rh-ai.apps.<cluster>/mlflow
        Workspace        = <workspace>  (first path segment)

    Returns (mlflow_uri, workspace) or ("", "") if not a cluster URL.
    """
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    host = parsed.hostname or ""

    # Only infer for cluster-style URLs: *.apps.<cluster_domain>
    parts = host.split(".", 1)
    if len(parts) < 2 or ".apps." not in host:
        return "", ""

    apps_domain = parts[1]  # apps.<cluster_domain>
    mlflow_uri = f"https://rh-ai.{apps_domain}/mlflow"

    # Extract workspace from first path segment: /<workspace>/<model>/v1
    path_parts = [p for p in parsed.path.strip("/").split("/") if p]
    workspace = path_parts[0] if path_parts else ""

    return mlflow_uri, workspace


def init_tracing() -> None:
    """Configure MLflow tracking.

    Resolution order for MLflow URI:
      1. Explicit ``MLFLOW_TRACKING_URI`` env / config
      2. Auto-inferred from ``LLM_BASE_URL`` (RHOAI cluster convention)
      3. Disabled

    Auth resolution:
      1. Explicit ``MLFLOW_TRACKING_TOKEN`` → bearer-token auth (cluster SA)
      2. Explicit ``MLFLOW_TRACKING_AUTH`` → respect it
      3. Fallback → ``kubernetes-namespaced`` (local ``oc login``)
    """
    global _tracing_enabled  # noqa: PLW0603

    from backend.config import get_settings

    settings = get_settings()

    uri = settings.mlflow_tracking_uri
    workspace = settings.mlflow_workspace
    insecure_tls = settings.mlflow_tracking_insecure_tls

    # Auto-infer from LLM_BASE_URL when MLflow URI is not explicit
    if not uri and settings.llm_base_url:
        inferred_uri, inferred_ws = _infer_mlflow_from_openai_url(settings.llm_base_url)
        if inferred_uri:
            uri = inferred_uri
            workspace = workspace or inferred_ws
            insecure_tls = True
            logger.info(
                "MLflow auto-detected from LLM_BASE_URL: uri=%s, workspace=%s",
                uri, workspace,
            )

    if not uri:
        logger.info("MLFLOW_TRACKING_URI not set and cannot infer — tracing disabled")
        return

    try:
        import mlflow
    except ImportError:
        logger.info("mlflow not installed — tracing disabled")
        return

    import os

    if insecure_tls:
        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        os.environ["PYTHONHTTPSVERIFY"] = "0"

    # Auth: token (cluster) > explicit auth mode > auto kubernetes-namespaced (local)
    if settings.mlflow_tracking_token:
        os.environ["MLFLOW_TRACKING_TOKEN"] = settings.mlflow_tracking_token
    elif settings.mlflow_tracking_auth:
        os.environ["MLFLOW_TRACKING_AUTH"] = settings.mlflow_tracking_auth
    elif not os.environ.get("MLFLOW_TRACKING_AUTH"):
        os.environ["MLFLOW_TRACKING_AUTH"] = "kubernetes-namespaced"
        logger.info("No MLFLOW_TRACKING_TOKEN — using kubernetes-namespaced auth (oc login)")

    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

    try:
        mlflow.set_tracking_uri(uri)

        experiment_name = settings.mlflow_experiment_name
        if workspace:
            experiment_path = f"/{workspace}/{experiment_name}"
        else:
            experiment_path = experiment_name

        mlflow.set_experiment(experiment_path)
    except Exception as exc:
        logger.warning("MLflow server unreachable (%s) — tracing disabled", exc)
        return

    import contextlib

    with contextlib.suppress(Exception):
        mlflow.langchain.autolog(log_models=False)

    _tracing_enabled = True
    logger.info("MLflow tracing enabled (uri=%s, workspace=%s)", uri, workspace)


def append_trace_record(record: dict[str, Any]) -> None:
    """Thread-safe append of a single JSON record to the design-suggestions JSONL file."""
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with _jsonl_lock, open(DESIGN_SUGGESTIONS_JSONL, "a", encoding="utf-8") as f:
        f.write(line)
    logger.debug("Trace record appended to %s", DESIGN_SUGGESTIONS_JSONL)


def build_trace_record(
    *,
    session_id: str,
    model_used: str,
    input_context: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    output: str,
    latency_ms: float,
    trace_id: str = "",
) -> dict[str, Any]:
    """Construct a normalised trace record dict."""
    return {
        "trace_id": trace_id,
        "session_id": session_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "model_used": model_used,
        "input": input_context,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "output": output,
        "latency_ms": round(latency_ms, 2),
    }
