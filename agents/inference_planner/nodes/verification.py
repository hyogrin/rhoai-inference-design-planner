"""Verification & finalize — validate consistency and emit final view model."""

from __future__ import annotations

import logging
from typing import Any

from agents.inference_planner.state import PlannerState

logger = logging.getLogger(__name__)


async def verify_recommendation(state: PlannerState) -> dict:
    """Verify recommendation consistency (memory fits, versions compatible, etc.)."""
    rec = state.get("recommendation") or {}
    mem = rec.get("memory_estimate") or {}
    evidence_summary = rec.get("evidence_summary") or {}
    deployment = rec.get("deployment_config") or {}

    warnings: list[str] = []

    # Check hardware compatibility block
    if mem.get("hw_blocked"):
        reason = mem.get("hw_blocked_reason", "Hardware incompatibility detected")
        warnings.append(reason)

    # Check memory fit (only meaningful if not hw_blocked)
    elif mem.get("fits") is False:
        total_req = mem.get("total_required_min_gb") or mem.get("total_required_gb", "?")
        total_avail = mem.get("total_available_gb", "?")
        warnings.append(
            f"Model requires {total_req} GB but only "
            f"{total_avail} GB available"
        )

    # Check vLLM version compatibility
    recipe_info = evidence_summary.get("vllm_recipe") or {}
    min_vllm = recipe_info.get("min_vllm_version")
    target_vllm = deployment.get("vllm_version")
    if min_vllm and target_vllm:
        if target_vllm < min_vllm:
            warnings.append(
                f"Target vLLM {target_vllm} < recipe minimum {min_vllm}"
            )

    # Check if model is validated for RHOAI
    compat = evidence_summary.get("rhoai_compatibility") or {}
    if not compat.get("is_validated"):
        warnings.append("Model not found in Red Hat validated models matrix")

    rec["verification"] = {
        "warnings": warnings,
        "status": "pass" if not warnings else "warnings",
    }

    logger.info("Verification: %s (%d warnings)", rec["verification"]["status"], len(warnings))

    return {
        "recommendation": rec,
        "phase_history": ["recommendation_verified"],
    }


async def finalize_view_model(state: PlannerState) -> dict:
    """Package the recommendation into the final view model for the frontend."""
    rec = state.get("recommendation") or {}

    view_model = {
        "sections": [
            {
                "id": "model_summary",
                "title": "Model Overview",
                "data": rec.get("model_summary", {}),
            },
            {
                "id": "evidence_summary",
                "title": "Collected Evidence",
                "data": rec.get("evidence_summary", {}),
            },
            {
                "id": "deployment_config",
                "title": "Deployment Configuration",
                "data": rec.get("deployment_config", {}),
            },
            {
                "id": "memory_estimate",
                "title": "Memory Analysis",
                "data": rec.get("memory_estimate", {}),
            },
            {
                "id": "cost_estimate",
                "title": "Cost Estimate",
                "data": rec.get("cost_estimate", {}),
            },
            {
                "id": "performance_forecast",
                "title": "Performance Forecast",
                "data": rec.get("performance_forecast", {}),
            },
            {
                "id": "design_suggestion",
                "title": "Inference Design Suggestion",
                "data": rec.get("design_suggestion", {}),
            },
        ],
        "verification": rec.get("verification", {}),
    }

    logger.info("Finalized view model with %d sections", len(view_model["sections"]))

    return {
        "view_model": view_model,
        "recommendation": rec,
        "current_phase": "completed",
        "current_step": 5,
        "phase_history": ["view_model_finalized"],
    }
