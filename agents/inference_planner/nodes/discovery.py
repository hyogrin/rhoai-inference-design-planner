"""Discovery nodes - fetch evidence from external sources.

Each node calls one connector, serializes the results into the graph state,
and appends to the shared evidence_items list.  Nodes are designed to run
in parallel within LangGraph's fan-out pattern.

NOTE: evidence_items and phase_history use operator.add reducers in state,
so each node returns only NEW items to append (not the full list).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from agents.inference_planner.state import PlannerState
from connectors.huggingface import HuggingFaceConnector, ModelNotFoundError
from connectors.pricing import PricingConnector
from connectors.redhat_model_cards import RedHatModelCardConnector
from connectors.rhoai_compatibility import RhoaiCompatibilityConnector
from connectors.vllm_recipes import VllmRecipeConnector

logger = logging.getLogger(__name__)


async def fetch_huggingface_metadata(state: PlannerState) -> dict[str, Any]:
    """Fetch model metadata from Hugging Face and parse architecture."""
    repo_id = state.get("model_repo_id", "")
    revision = state.get("model_revision") or "main"
    token = state.get("hf_token") or os.getenv("HF_TOKEN") or None

    if not repo_id or "/" not in repo_id or len(repo_id) > 200:
        return {
            "error": f"Invalid model repo_id: '{repo_id[:80]}'" if repo_id else "No model repo_id provided",
            "phase_history": ["hf_metadata_skipped"],
        }

    connector = HuggingFaceConnector(token=token)

    try:
        identity = await connector.fetch_model_identity(repo_id, revision)
        architecture = await connector.fetch_model_architecture(repo_id, revision)
    except ModelNotFoundError as exc:
        logger.warning("Model not found: %s", exc)
        return {
            "error": str(exc),
            "phase_history": ["hf_metadata_failed"],
        }
    except Exception as exc:
        logger.error("Unexpected HF connector error for %s: %s", repo_id, exc, exc_info=True)
        return {
            "error": f"HuggingFace fetch failed: {exc}",
            "phase_history": ["hf_metadata_failed"],
        }

    from datetime import UTC, datetime

    from domain.evidence import EvidenceItem

    arch_dict = architecture.model_dump(mode="json")
    arch_summary_parts = []
    if arch_dict.get("architecture_type"):
        arch_summary_parts.append(f"type={arch_dict['architecture_type']}")
    if arch_dict.get("num_hidden_layers"):
        arch_summary_parts.append(f"layers={arch_dict['num_hidden_layers']}")
    if arch_dict.get("kv_lora_rank"):
        arch_summary_parts.append(f"MLA(kv_lora_rank={arch_dict['kv_lora_rank']})")
    if arch_dict.get("sliding_attention_layers"):
        arch_summary_parts.append(
            f"hybrid({arch_dict['sliding_attention_layers']}sliding+"
            f"{arch_dict.get('full_attention_layers', '?')}full)"
        )
    if arch_dict.get("num_experts_total"):
        arch_summary_parts.append(f"MoE({arch_dict['num_experts_total']}experts)")

    evidence_item = EvidenceItem(
        category="model_metadata",
        claim_type="architecture",
        title=f"HuggingFace config.json for {repo_id}",
        summary=f"Official model architecture: {', '.join(arch_summary_parts) or 'standard transformer'}. "
                f"Parameters: {arch_dict.get('parameter_count_total')}, "
                f"head_dim: {arch_dict.get('head_dim')}, "
                f"num_kv_heads: {arch_dict.get('num_kv_heads')}.",
        source_url=f"https://huggingface.co/{repo_id}/blob/{revision}/config.json",
        source_domain="huggingface.co",
        publisher="Hugging Face",
        retrieved_at=datetime.now(UTC),
        model_revision=revision,
        source_tier="primary",
        verification_level="verified",
    )

    return {
        "model_identity": identity.model_dump(mode="json"),
        "model_architecture": arch_dict,
        "evidence_items": [evidence_item.model_dump(mode="json")],
        "phase_history": ["hf_metadata_fetched"],
    }


async def discover_vllm_recipe(state: PlannerState) -> dict[str, Any]:
    """Discover vLLM recipe for the model."""
    repo_id = state.get("model_repo_id", "")
    if not repo_id:
        return {"phase_history": ["vllm_recipe_skipped"]}

    connector = VllmRecipeConnector()

    try:
        evidence_items = await connector.find_recipe(repo_id)
    except Exception as exc:
        logger.warning("vLLM recipe discovery failed for %s: %s", repo_id, exc)
        return {"phase_history": ["vllm_recipe_failed"]}

    new_evidence = [item.model_dump(mode="json") for item in evidence_items]
    logger.info("Found %d vLLM recipe evidence items for %s", len(new_evidence), repo_id)

    return {
        "evidence_items": new_evidence,
        "phase_history": ["vllm_recipe_discovered"],
    }


async def discover_redhat_evaluations(state: PlannerState) -> dict[str, Any]:
    """Discover Red Hat AI evaluations."""
    repo_id = state.get("model_repo_id", "")
    if not repo_id:
        return {"phase_history": ["redhat_evaluations_skipped"]}

    token = state.get("hf_token") or os.getenv("HF_TOKEN") or None
    connector = RedHatModelCardConnector(token=token)

    try:
        evidence_items = await connector.find_redhat_evidence(repo_id)
    except Exception as exc:
        logger.warning("Red Hat evaluation discovery failed for %s: %s", repo_id, exc)
        return {"phase_history": ["redhat_evaluations_failed"]}

    new_evidence = [item.model_dump(mode="json") for item in evidence_items]
    logger.info("Found %d Red Hat evaluation evidence items for %s", len(new_evidence), repo_id)

    return {
        "evidence_items": new_evidence,
        "phase_history": ["redhat_evaluations_discovered"],
    }


async def discover_community_evidence(state: PlannerState) -> dict[str, Any]:
    """Contents Research: read pages already found by Web Search nodes.

    Runs AFTER validate_discovery so that evidence_items from HuggingFace,
    vLLM recipes, Red Hat evaluations, and RHOAI compatibility are available.

    Pipeline:
      1. Collect source_urls from existing evidence_items
      2. Playwright reads top pages (with thumbnail screenshots streamed to frontend)
      3. LLM analyzes page content → enriches evidence summaries

    All stages traced as MLflow spans with the same session_id.
    """
    import contextlib
    import time
    from typing import Any as _Any

    repo_id = state.get("model_repo_id", "")
    if not repo_id:
        return {"phase_history": ["community_evidence_skipped"]}

    session_id = state.get("session_id", "")
    evidence_items = state.get("evidence_items", [])

    if not evidence_items:
        logger.info("No evidence items to enrich for %s", repo_id)
        return {"phase_history": ["community_evidence_skipped"]}

    # Stream writer for real-time page preview thumbnails
    writer = None
    try:
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
    except Exception:
        pass

    def _on_screenshot(url: str, title: str, b64_jpeg: str) -> None:
        if writer is None:
            return
        try:
            writer({
                "type": "web_preview",
                "url": url,
                "title": title,
                "thumbnail": b64_jpeg,
            })
        except Exception:
            pass

    # MLflow tracing
    _should_trace = False
    try:
        import mlflow

        from backend.tracing import is_tracing_enabled
        _should_trace = is_tracing_enabled()
    except ImportError:
        pass

    parent_cm: _Any = contextlib.nullcontext()
    if _should_trace:
        try:
            parent_cm = mlflow.start_span(
                name="contents_research", span_type="CHAIN",
            )
        except Exception:
            parent_cm = contextlib.nullcontext()

    # Collect unique source URLs from existing evidence (prioritize high-tier sources)
    seen_urls: set[str] = set()
    seen_categories: set[str] = set()
    urls_to_read: list[dict[str, str]] = []

    # Pick the best URL from each category (1 per category)
    target_categories = [
        "model_metadata",           # Model Architecture (HuggingFace)
        "recipe",                   # vLLM Recipe
        "redhat_evaluation",        # Red Hat Evaluations
        "platform_compatibility",   # RHOAI Compatibility
    ]

    tier_order = {"primary": 0, "official_secondary": 1, "secondary": 2, "community": 3}
    sorted_items = sorted(
        evidence_items,
        key=lambda x: tier_order.get(x.get("source_tier", "community"), 3),
    )

    for item in sorted_items:
        cat = item.get("category", "")
        if cat in seen_categories:
            continue
        if cat not in target_categories:
            continue
        url = item.get("source_url", "")
        if not url or url in seen_urls:
            continue
        if url.endswith(".json") or "/raw/" in url:
            continue
        if "/blob/" in url and "github.com" in url:
            continue
        seen_urls.add(url)
        seen_categories.add(cat)
        urls_to_read.append({
            "url": url,
            "title": item.get("title", ""),
            "category": cat,
            "evidence_id": str(item.get("evidence_id", "")),
        })
        if len(seen_categories) >= len(target_categories):
            break

    if not urls_to_read:
        logger.info("No readable URLs in evidence for %s", repo_id)
        return {"phase_history": ["community_evidence_no_urls"]}

    t0 = time.monotonic()
    enriched_items: list[dict[str, _Any]] = []

    with parent_cm as parent_span:
        if parent_span is not None:
            try:
                parent_span.set_inputs({
                    "repo_id": repo_id,
                    "urls_to_read": [u["url"] for u in urls_to_read],
                    "session_id": session_id,
                })
                if session_id:
                    mlflow.update_current_trace(session_id=session_id)
            except Exception:
                pass

        # ── Step 1: Playwright Page Reading ──
        read_span_cm: _Any = contextlib.nullcontext()
        if _should_trace:
            try:
                read_span_cm = mlflow.start_span(
                    name="playwright_read_pages", span_type="TOOL",
                )
            except Exception:
                read_span_cm = contextlib.nullcontext()

        t1 = time.monotonic()
        page_contents: list[dict[str, str]] = []

        with read_span_cm as read_span:
            from connectors.web_reader import read_page

            for entry in urls_to_read:
                url = entry["url"]
                try:
                    content = await read_page(
                        url, max_chars=2000, on_screenshot=_on_screenshot,
                    )
                    if content and not content.startswith("[Failed") and not content.startswith("[Skipped"):
                        page_contents.append({**entry, "content": content})
                except Exception as exc:
                    logger.debug("Page read failed for %s: %s", url, exc)

            if read_span is not None:
                try:
                    read_span.set_outputs({
                        "pages_read": len(page_contents),
                        "latency_ms": round((time.monotonic() - t1) * 1000),
                    })
                except Exception:
                    pass

        # ── Step 2: LLM Content Analysis ──
        llm_span_cm: _Any = contextlib.nullcontext()
        if _should_trace:
            try:
                llm_span_cm = mlflow.start_span(
                    name="llm_analyze_pages", span_type="LLM",
                )
            except Exception:
                llm_span_cm = contextlib.nullcontext()

        t2 = time.monotonic()
        llm_calls = 0
        total_input_tokens = 0
        total_output_tokens = 0

        with llm_span_cm as llm_span:
            from connectors.community_search import CommunitySearchConnector

            connector = CommunitySearchConnector()

            for pc in page_contents:
                content = pc["content"]
                if len(content) < 50:
                    continue

                summary, usage = await connector._llm_analyze(
                    repo_id=repo_id,
                    evidence_type=pc.get("category", "compatibility"),
                    url=pc["url"],
                    content=content,
                )
                llm_calls += 1
                total_input_tokens += usage.get("input_tokens", 0)
                total_output_tokens += usage.get("output_tokens", 0)

                if summary and summary != "NOT_RELEVANT":
                    from datetime import UTC, datetime
                    from uuid import uuid4

                    from domain.evidence import EvidenceItem

                    item = EvidenceItem(
                        evidence_id=uuid4(),
                        category="community",
                        claim_type="compatibility",
                        title=f"[Enriched] {pc.get('title', pc['url'])}",
                        summary=summary[:500],
                        source_url=pc["url"],
                        source_domain=pc.get("url", "").split("/")[2] if "/" in pc.get("url", "") else None,
                        retrieved_at=datetime.now(UTC),
                        source_tier="official_secondary",
                        verification_level="reported",
                    )
                    enriched_items.append(item.model_dump(mode="json"))

            total_tokens = total_input_tokens + total_output_tokens
            if llm_span is not None:
                try:
                    llm_span.set_outputs({
                        "llm_calls": llm_calls,
                        "enriched_items": len(enriched_items),
                        "latency_ms": round((time.monotonic() - t2) * 1000),
                        "token_usage": {
                            "input_tokens": total_input_tokens,
                            "output_tokens": total_output_tokens,
                            "total_tokens": total_tokens,
                        },
                    })
                    llm_span.set_attribute("mlflow.chat.tokenUsage", {
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                        "total_tokens": total_tokens,
                    })
                except Exception:
                    pass

        total_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "Contents research for %s: read %d pages, enriched %d items, %.0fms",
            repo_id, len(page_contents), len(enriched_items), total_ms,
        )

        if parent_span is not None:
            try:
                parent_span.set_outputs({
                    "pages_read": len(page_contents),
                    "enriched_count": len(enriched_items),
                    "total_ms": round(total_ms),
                })
            except Exception:
                pass

    return {
        "evidence_items": enriched_items,
        "phase_history": ["community_evidence_discovered"],
    }


async def check_rhoai_compatibility(state: PlannerState) -> dict[str, Any]:
    """Check platform compatibility with the user's RHOAI version and validated models matrix."""
    workload = state.get("workload_profile") or {}
    rhoai_version = workload.get("rhoai_version")

    vllm_min = None
    for item in state.get("evidence_items", []):
        if isinstance(item, dict) and item.get("category") == "recipe":
            vllm_min = item.get("vllm_version")
            if vllm_min:
                break

    connector = RhoaiCompatibilityConnector()

    try:
        evidence_items = await connector.check_compatibility(
            rhoai_version=rhoai_version,
            vllm_version_target=vllm_min,
        )
    except Exception as exc:
        logger.warning("RHOAI compatibility check failed: %s", exc)
        return {"phase_history": ["rhoai_compatibility_failed"]}

    # Also check the validated models matrix
    repo_id = state.get("model_repo_id", "")
    try:
        validated_evidence = await connector.check_validated_models_matrix(repo_id)
        evidence_items.extend(validated_evidence)
    except Exception as exc:
        logger.warning("Validated models matrix check failed: %s", exc)

    new_evidence = [item.model_dump(mode="json") for item in evidence_items]
    logger.info("Generated %d RHOAI compatibility evidence items", len(new_evidence))

    return {
        "evidence_items": new_evidence,
        "phase_history": ["rhoai_compatibility_checked"],
    }


async def fetch_pricing(state: PlannerState) -> dict[str, Any]:
    """Fetch GPU pricing data based on architecture and hardware config."""
    architecture = state.get("model_architecture")
    if not architecture:
        return {"phase_history": ["pricing_skipped"]}

    hardware = state.get("hardware_inventory") or {}
    environment_type = hardware.get("environment_type", "on_prem")

    gpu_type = hardware.get("gpu_type")
    gpu_count = hardware.get("gpu_count") or 1

    if not gpu_type:
        logger.info("No GPU type specified by user — skipping pricing discovery")
        return {"phase_history": ["pricing_skipped_no_gpu"]}

    connector = PricingConnector()

    try:
        evidence_items = await connector.get_pricing_evidence(
            gpu_type, gpu_count, environment_type=environment_type
        )
    except Exception as exc:
        logger.warning("Pricing fetch failed: %s", exc)
        return {"phase_history": ["pricing_failed"]}

    new_evidence = [item.model_dump(mode="json") for item in evidence_items]
    logger.info(
        "Found %d pricing evidence items for %s x%d (%s)",
        len(new_evidence), gpu_type, gpu_count, environment_type,
    )

    return {
        "evidence_items": new_evidence,
        "phase_history": ["pricing_fetched"],
    }
