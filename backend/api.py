import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.dependencies import DesignRepoDep
from backend.metrics import metrics_endpoint
from backend.repositories.design_session import OptimisticLockError
from backend.schemas import (
    CreateDesignRequest,
    DesignListResponse,
    DesignSessionResponse,
    ErrorResponse,
    UpdateHardwareRequest,
    UpdateWorkloadRequest,
)

logger = structlog.get_logger(__name__)


def _configure_logging(log_level: str) -> None:
    import logging

    numeric_level = logging.getLevelName(log_level.upper())
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.app_log_level)
    await logger.ainfo("application_startup", env=settings.app_env)

    # Initialize PostgreSQL checkpointer for LangGraph
    pg_url = settings.database_url_sync
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        checkpointer_ctx = AsyncPostgresSaver.from_conn_string(pg_url)
        checkpointer = await checkpointer_ctx.__aenter__()
        await checkpointer.setup()
        app.state.checkpointer = checkpointer
        app.state.checkpointer_ctx = checkpointer_ctx
        logger.info("langgraph_checkpointer_initialized", pg_url=pg_url[:40])
    except Exception as e:
        logger.warning("checkpointer_init_failed_using_memory", error=str(e))
        from langgraph.checkpoint.memory import MemorySaver

        app.state.checkpointer = MemorySaver()
        app.state.checkpointer_ctx = None

    yield

    if getattr(app.state, "checkpointer_ctx", None):
        await app.state.checkpointer_ctx.__aexit__(None, None, None)
    await logger.ainfo("application_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Inference Design Planner",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            origin.strip()
            for origin in settings.app_cors_origins.split(",")
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_exception_handlers(app)
    _register_routes(app)

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        correlation_id = request.headers.get(
            "x-correlation-id", str(uuid.uuid4())
        )
        error = ErrorResponse(
            code=f"HTTP_{exc.status_code}",
            message=str(exc.detail),
            retryable=exc.status_code >= 500,
            correlation_id=correlation_id,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error.model_dump(exclude_none=True),
        )

    @app.exception_handler(OptimisticLockError)
    async def optimistic_lock_handler(
        request: Request, exc: OptimisticLockError
    ) -> JSONResponse:
        correlation_id = request.headers.get(
            "x-correlation-id", str(uuid.uuid4())
        )
        error = ErrorResponse(
            code="CONFLICT",
            message=str(exc),
            retryable=True,
            correlation_id=correlation_id,
        )
        return JSONResponse(status_code=409, content=error.model_dump(exclude_none=True))

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        correlation_id = request.headers.get(
            "x-correlation-id", str(uuid.uuid4())
        )
        logger.error(
            "unhandled_exception",
            error=str(exc),
            correlation_id=correlation_id,
            path=request.url.path,
        )
        error = ErrorResponse(
            code="INTERNAL_ERROR",
            message="An unexpected error occurred",
            retryable=True,
            correlation_id=correlation_id,
        )
        return JSONResponse(status_code=500, content=error.model_dump(exclude_none=True))


def _register_routes(app: FastAPI) -> None:
    @app.get("/api/v1/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/metrics")
    async def metrics(request: Request):
        return await metrics_endpoint(request)

    @app.post(
        "/api/v1/designs",
        response_model=DesignSessionResponse,
        status_code=201,
    )
    async def create_design(
        request: CreateDesignRequest,
        repo: DesignRepoDep,
    ) -> DesignSessionResponse:
        session_data = {
            "model_repo_id": request.model_repo_id,
            "model_revision": request.model_revision,
            "title": request.title or f"Design: {request.model_repo_id}",
            "status": "intake",
            "current_step": 1,
        }
        design = await repo.create(session_data)
        await logger.ainfo(
            "design_session_created",
            session_id=str(design.id),
            model_repo_id=design.model_repo_id,
        )
        return _to_response(design)

    @app.get(
        "/api/v1/designs/{design_id}",
        response_model=DesignSessionResponse,
    )
    async def get_design(
        design_id: uuid.UUID,
        repo: DesignRepoDep,
    ) -> DesignSessionResponse:
        design = await repo.get(design_id)
        if design is None:
            raise HTTPException(status_code=404, detail="Design session not found")
        return _to_response(design)

    @app.delete("/api/v1/designs/{design_id}", status_code=204)
    async def delete_design(
        design_id: uuid.UUID,
        repo: DesignRepoDep,
    ) -> None:
        deleted = await repo.delete(design_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Design session not found")

    @app.put(
        "/api/v1/designs/{design_id}/hardware",
        response_model=DesignSessionResponse,
    )
    async def update_hardware(
        design_id: uuid.UUID,
        request: UpdateHardwareRequest,
        repo: DesignRepoDep,
    ) -> DesignSessionResponse:
        design = await repo.get(design_id)
        if design is None:
            raise HTTPException(status_code=404, detail="Design session not found")

        state = design.state_snapshot or {}
        state["hardware"] = request.model_dump(exclude_none=True)

        updated = await repo.update(
            design_id,
            {"state_snapshot": state, "status": "hardware_configured"},
            expected_version=design.version,
        )
        return _to_response(updated)

    @app.put(
        "/api/v1/designs/{design_id}/workload",
        response_model=DesignSessionResponse,
    )
    async def update_workload(
        design_id: uuid.UUID,
        request: UpdateWorkloadRequest,
        repo: DesignRepoDep,
    ) -> DesignSessionResponse:
        design = await repo.get(design_id)
        if design is None:
            raise HTTPException(status_code=404, detail="Design session not found")

        state = design.state_snapshot or {}
        state["workload"] = request.model_dump(exclude_none=True)

        updated = await repo.update(
            design_id,
            {"state_snapshot": state, "status": "workload_configured"},
            expected_version=design.version,
        )
        return _to_response(updated)

    @app.get("/api/v1/designs", response_model=DesignListResponse)
    async def list_designs(
        repo: DesignRepoDep,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> DesignListResponse:
        items, total = await repo.list_all(limit=limit, offset=offset)
        return DesignListResponse(
            items=[_to_response(item) for item in items],
            total=total,
        )

    @app.post("/agent")
    async def agent_endpoint(req: AgUiRunInput, request: Request):
        """AG-UI protocol endpoint — streams inference design planner as SSE events."""
        messages = req.messages or []
        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    query = " ".join(
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                elif isinstance(content, str):
                    query = content
                break

        run_id = req.run_id or str(uuid.uuid4())
        thread_id = req.thread_id or str(uuid.uuid4())
        forwarded = req.forwarded_props
        state_payload = req.state

        return StreamingResponse(
            _stream_agui(
                run_id=run_id,
                thread_id=thread_id,
                query=query.strip(),
                forwarded_props=forwarded,
                state_payload=state_payload,
                checkpointer=getattr(request.app.state, "checkpointer", None),
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


def _to_response(design) -> DesignSessionResponse:
    return DesignSessionResponse(
        session_id=design.id,
        title=design.title,
        status=design.status,
        model_repo_id=design.model_repo_id,
        model_revision=design.model_revision,
        current_step=design.current_step,
        created_at=design.created_at,
        updated_at=design.updated_at,
        version=design.version,
    )


# ---------------------------------------------------------------------------
# AG-UI Protocol Types & Helpers
# ---------------------------------------------------------------------------


class AgUiRunInput(BaseModel):
    """Matches the RunAgentInput schema sent by @ag-ui/client HttpAgent."""

    model_config = {"populate_by_name": True}

    run_id: str = Field(default="", alias="runId")
    thread_id: str = Field(default="", alias="threadId")
    messages: list = []
    state: dict | None = None
    tools: list = []
    context: list = []
    forwarded_props: dict = Field(default_factory=dict, alias="forwardedProps")


def _agui_event(payload: dict) -> str:
    """Format a dict as an AG-UI SSE data line."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


_HEARTBEAT_INTERVAL = 15  # seconds between keepalive comments


async def _stream_agui(
    *,
    run_id: str,
    thread_id: str,
    query: str,
    forwarded_props: dict,
    state_payload: dict | None,
    checkpointer,
) -> AsyncGenerator[str, None]:
    """Run the inference planner graph and yield AG-UI protocol SSE events."""
    from agents.inference_planner.graph import compile_graph

    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

    msg_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    compiled_graph = compile_graph(checkpointer=checkpointer)

    # Determine if this is a resume (pending interrupt for this thread)
    resume_data: dict | None = None
    try:
        graph_state = await compiled_graph.aget_state(config)
        pending_tasks = getattr(graph_state, "tasks", [])
        has_interrupt = any(
            hasattr(t, "interrupts") and t.interrupts for t in pending_tasks
        )
        if has_interrupt and (state_payload or query):
            resume_data = state_payload if state_payload else {"user_input": query}
    except Exception:
        pass

    yield _agui_event({"type": "RUN_STARTED", "runId": run_id, "threadId": thread_id})

    if not query and not resume_data:
        yield _agui_event({"type": "TEXT_MESSAGE_START", "messageId": msg_id})
        yield _agui_event({
            "type": "TEXT_MESSAGE_CONTENT",
            "messageId": msg_id,
            "delta": "Please provide a model repository ID to begin the inference design analysis.",
        })
        yield _agui_event({"type": "TEXT_MESSAGE_END", "messageId": msg_id})
        yield _agui_event({"type": "RUN_FINISHED", "runId": run_id, "threadId": thread_id})
        return

    # Build graph input
    if resume_data:
        from langgraph.types import Command

        graph_input = Command(resume=resume_data)
    else:
        model_repo_id = forwarded_props.get("model_repo_id", query)
        model_revision = forwarded_props.get("model_revision", "main")
        graph_input = {
            "session_id": thread_id[:12],
            "model_repo_id": model_repo_id,
            "model_revision": model_revision,
            "current_phase": "intake",
            "current_step": 1,
            "phase_history": [],
        }

    event_queue: asyncio.Queue = asyncio.Queue()
    accumulated_steps: list[dict] = []
    text_started = False

    async def _run_graph():
        try:
            async for mode, chunk in compiled_graph.astream(
                graph_input,
                config=config,
                stream_mode=["updates", "custom"],
            ):
                await event_queue.put((mode, chunk))
        except Exception as exc:
            logger.error("graph_stream_error", error=str(exc), exc_info=True)
            await event_queue.put(("error", {"__error__": str(exc)}))
        finally:
            await event_queue.put(None)

    graph_task = asyncio.create_task(_run_graph())

    try:
        while True:
            try:
                item = await asyncio.wait_for(event_queue.get(), timeout=_HEARTBEAT_INTERVAL)
            except TimeoutError:
                yield ": heartbeat\n\n"
                continue

            if item is None:
                break

            mode, chunk = item

            if mode == "error" or (isinstance(chunk, dict) and "__error__" in chunk):
                error_msg = chunk.get("__error__", "Unknown error") if isinstance(chunk, dict) else str(chunk)
                yield _agui_event({"type": "RUN_ERROR", "message": error_msg})
                return

            if mode == "custom":
                yield _agui_event({"type": "CUSTOM", "name": "progress", "value": chunk})
                continue

            # mode == "updates"
            if not isinstance(chunk, dict) or not chunk:
                continue

            node_name = next(iter(chunk))
            node_output = chunk[node_name]
            if not isinstance(node_output, dict):
                continue

            # Compute quality score from node output
            quality_score = _compute_quality_score(node_name, node_output)
            source_urls = _extract_source_urls(node_output)

            step_entry = {
                "id": str(uuid.uuid4())[:8],
                "node": node_name,
                "phase": node_output.get("current_phase", ""),
                "timestamp": int(time.time() * 1000),
                "quality_score": quality_score,
                "source_urls": source_urls,
                "error": node_output.get("error"),
            }
            accumulated_steps.append(step_entry)
            yield _agui_event({"type": "CUSTOM", "name": "step", "value": step_entry})

            # Emit view_model when finalize_view_model completes
            if "view_model" in node_output and node_output["view_model"]:
                yield _agui_event({
                    "type": "CUSTOM",
                    "name": "view_model",
                    "value": node_output["view_model"],
                })

            # Periodic state snapshot
            snapshot_data = {
                "steps": accumulated_steps,
                "current_phase": node_output.get("current_phase", ""),
                "model_repo_id": node_output.get("model_repo_id", ""),
                "evidence_count": (
                    len(node_output.get("evidence_items", []))
                    if "evidence_items" in node_output
                    else None
                ),
            }
            snapshot_data = {k: v for k, v in snapshot_data.items() if v is not None}
            yield _agui_event({"type": "STATE_SNAPSHOT", "snapshot": snapshot_data})

        # Check for HITL interrupts after the graph stream ends
        graph_state = await compiled_graph.aget_state(config)
        pending_tasks = getattr(graph_state, "tasks", [])
        interrupt_data = None
        for task in pending_tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupt_data = task.interrupts[0].value
                break

        if interrupt_data:
            if text_started:
                yield _agui_event({"type": "TEXT_MESSAGE_END", "messageId": msg_id})
                text_started = False

            yield _agui_event({
                "type": "CUSTOM",
                "name": "workload_interrupt",
                "value": interrupt_data,
            })
            yield _agui_event({
                "type": "STATE_SNAPSHOT",
                "snapshot": {
                    "steps": accumulated_steps,
                    "interrupt": interrupt_data,
                },
            })
            yield _agui_event({"type": "RUN_FINISHED", "runId": run_id, "threadId": thread_id})
        else:
            if text_started:
                yield _agui_event({"type": "TEXT_MESSAGE_END", "messageId": msg_id})

            # Emit final state from graph
            state_values = getattr(graph_state, "values", {}) if graph_state else {}
            final_snapshot = {
                "steps": accumulated_steps,
                "current_phase": state_values.get("current_phase", "complete"),
                "recommendation": state_values.get("recommendation"),
                "view_model": state_values.get("view_model"),
            }
            final_snapshot = {k: v for k, v in final_snapshot.items() if v is not None}
            yield _agui_event({"type": "STATE_SNAPSHOT", "snapshot": final_snapshot})
            yield _agui_event({"type": "RUN_FINISHED", "runId": run_id, "threadId": thread_id})

    except Exception as exc:
        logger.exception("agui_stream_error")
        yield _agui_event({"type": "RUN_ERROR", "message": str(exc)})
        return
    finally:
        if not graph_task.done():
            graph_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await graph_task


def _compute_quality_score(node_name: str, output: dict) -> int:
    """Compute a 1-5 quality score for a discovery node's output."""
    if output.get("error"):
        return 1

    if node_name == "fetch_huggingface_metadata":
        arch = output.get("model_architecture") or {}
        identity = output.get("model_identity") or {}
        score = 1
        if identity.get("repo_id") or identity.get("source_url"):
            score += 1
        if arch.get("architecture_names"):
            score += 1
        if arch.get("parameter_count_total"):
            score += 1
        if arch.get("max_position_embeddings") or arch.get("num_attention_heads"):
            score += 1
        return min(score, 5)

    if node_name == "discover_vllm_recipe":
        evidence = output.get("evidence_items", [])
        if not evidence:
            return 1
        score = 2
        for item in evidence:
            if isinstance(item, dict):
                if item.get("hardware_signature"):
                    score += 1
                if item.get("vllm_version"):
                    score += 1
        return min(score, 5)

    if node_name == "discover_redhat_evaluations":
        evidence = output.get("evidence_items", [])
        if not evidence:
            return 1
        return min(2 + len(evidence), 5)

    if node_name == "discover_community_evidence":
        evidence = output.get("evidence_items", [])
        if not evidence:
            return 1
        return min(1 + len(evidence), 5)

    if node_name == "check_rhoai_compatibility":
        evidence = output.get("evidence_items", [])
        if not evidence:
            return 1
        has_validated = any(
            isinstance(e, dict) and "validated" in (e.get("title", "")).lower()
            for e in evidence
        )
        return 5 if has_validated else min(2 + len(evidence), 5)

    if node_name == "fetch_pricing":
        evidence = output.get("evidence_items", [])
        if not evidence:
            return 1
        return min(2 + len(evidence), 5)

    return 3


def _extract_source_urls(output: dict) -> list[str]:
    """Extract unique source URLs from node output."""
    urls: list[str] = []
    seen: set[str] = set()

    # From evidence_items
    for item in output.get("evidence_items", []):
        if isinstance(item, dict):
            url = item.get("source_url", "")
            if url and url not in seen:
                urls.append(url)
                seen.add(url)

    # From model_identity
    identity = output.get("model_identity")
    if isinstance(identity, dict):
        url = identity.get("source_url", "")
        if url and url not in seen:
            urls.append(url)
            seen.add(url)

    return urls[:5]


app = create_app()
