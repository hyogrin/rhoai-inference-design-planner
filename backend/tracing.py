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


def init_tracing() -> None:
    """Configure MLflow tracking. Skips silently if connection fails."""
    global _tracing_enabled  # noqa: PLW0603

    from backend.config import get_settings

    settings = get_settings()
    uri = settings.mlflow_tracking_uri
    if not uri:
        logger.info("MLFLOW_TRACKING_URI not set — MLflow tracing disabled")
        return

    try:
        import mlflow
    except ImportError:
        logger.info("mlflow not installed — tracing disabled")
        return

    import os

    if settings.mlflow_tracking_insecure_tls:
        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        os.environ["PYTHONHTTPSVERIFY"] = "0"

    if settings.mlflow_tracking_token:
        os.environ["MLFLOW_TRACKING_TOKEN"] = settings.mlflow_tracking_token

    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

    try:
        mlflow.set_tracking_uri(uri)

        # RHOAI MLflow requires workspace context in the experiment path
        workspace = settings.mlflow_workspace
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
    logger.info("MLflow tracing enabled (uri=%s)", uri)


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
