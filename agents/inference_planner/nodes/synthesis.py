"""Synthesis node — aggregate collected evidence into a structured summary.

No LLM needed here. Pure data aggregation from state.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.inference_planner.state import PlannerState

logger = logging.getLogger(__name__)


def _summarize_evidence(evidence_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Group and summarize evidence items by category."""
    by_category: dict[str, list[dict[str, Any]]] = {}
    for item in evidence_items:
        cat = item.get("category", "other")
        by_category.setdefault(cat, []).append(item)

    summary: dict[str, Any] = {}

    # vLLM Recipe
    recipe_items = by_category.get("recipe", [])
    if recipe_items:
        vllm_version = None
        hardware_verified = []
        launch_args = []
        features = []
        for item in recipe_items:
            if item.get("vllm_version") and not vllm_version:
                vllm_version = item["vllm_version"]
            if item.get("hardware_signature"):
                hardware_verified.append(item["hardware_signature"])
            title = item.get("title", "")
            if "argument" in title.lower():
                launch_args.append(item.get("summary", ""))
            elif "feature" in title.lower() or "strategy" in title.lower():
                features.append(item.get("title", ""))
        summary["vllm_recipe"] = {
            "count": len(recipe_items),
            "min_vllm_version": vllm_version,
            "verified_hardware": list(set(hardware_verified))[:5],
            "launch_args": launch_args[:3],
            "features": features[:5],
        }

    # Red Hat Evaluations
    eval_items = by_category.get("redhat_evaluation", [])
    if eval_items:
        accuracy_items = [i for i in eval_items if i.get("claim_type") == "accuracy"]
        perf_items = [i for i in eval_items if i.get("claim_type") == "serving_performance"]
        summary["evaluations"] = {
            "count": len(eval_items),
            "accuracy_benchmarks": len(accuracy_items),
            "performance_benchmarks": len(perf_items),
            "sources": list({i.get("source_url", "") for i in eval_items if i.get("source_url")})[:3],
        }

    # RHOAI Compatibility
    compat_items = by_category.get("rhoai_compatibility", [])
    if compat_items:
        validated = any("validated" in (i.get("summary") or "").lower() for i in compat_items)
        supported_gpus = []
        for item in compat_items:
            s = item.get("summary", "")
            if "gpu" in s.lower() or "accelerator" in s.lower():
                supported_gpus.append(s[:100])
        summary["rhoai_compatibility"] = {
            "count": len(compat_items),
            "is_validated": validated,
            "supported_gpus": supported_gpus[:3],
            "sources": list({i.get("source_url", "") for i in compat_items if i.get("source_url")})[:2],
        }

    # Pricing
    pricing_items = by_category.get("pricing", [])
    if pricing_items:
        summary["pricing"] = {
            "count": len(pricing_items),
            "sources": list({i.get("source_url", "") for i in pricing_items if i.get("source_url")})[:3],
        }

    return summary


async def synthesize_recommendation(state: PlannerState) -> dict:
    """Synthesize collected evidence into a structured recommendation."""
    evidence_items = state.get("evidence_items", [])
    model_identity = state.get("model_identity") or {}
    model_architecture = state.get("model_architecture") or {}
    workload = state.get("workload_profile") or {}
    existing_rec = state.get("recommendation") or {}

    # Extract or infer parameters
    import re
    params_raw = model_architecture.get("parameters")
    if not params_raw:
        repo_id = state.get("model_repo_id", "")
        name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        m = re.search(r"(\d+\.?\d*)B", name)
        if m:
            params_raw = float(m.group(1)) * 1e9

    # Use sizing node's calculated value if available
    if not params_raw:
        rec = existing_rec or {}
        mem_est = rec.get("memory_estimate") or {}
        if mem_est.get("parameters_billions"):
            params_raw = mem_est["parameters_billions"] * 1e9

    arch_type = "Unknown"
    if model_architecture.get("architectures"):
        arch_type = model_architecture["architectures"][0]
    elif model_architecture.get("model_type"):
        arch_type = model_architecture["model_type"]

    # Build model summary
    model_summary = {
        "repo_id": state.get("model_repo_id", ""),
        "model_name": model_identity.get("model_name", ""),
        "architecture_type": arch_type,
        "parameters": params_raw,
        "parameters_display": _format_params(params_raw),
        "context_length": model_architecture.get("max_position_embeddings") or model_architecture.get("max_sequence_length"),
        "num_layers": model_architecture.get("num_hidden_layers"),
        "hidden_size": model_architecture.get("hidden_size"),
        "num_attention_heads": model_architecture.get("num_attention_heads"),
        "num_kv_heads": model_architecture.get("num_key_value_heads"),
        "vocab_size": model_architecture.get("vocab_size"),
        "license": model_identity.get("license"),
    }

    # Build evidence summary
    evidence_summary = _summarize_evidence(evidence_items)
    evidence_summary["total_items"] = len(evidence_items)

    # Deployment config (from workload + sizing)
    deployment_config = {
        "gpu_type": workload.get("gpu_type"),
        "gpu_count": workload.get("gpu_count", 1),
        "rhoai_version": workload.get("rhoai_version"),
        "vllm_version": workload.get("vllm_version"),
        "use_case_presets": workload.get("use_case_presets", []),
        "target_end_users": workload.get("target_end_users"),
        "max_concurrent_requests": workload.get("max_concurrent_requests"),
        "ttft_target_ms": workload.get("ttft_ms"),
        "tpot_target_ms": workload.get("tpot_ms"),
    }

    # Merge into recommendation
    existing_rec["model_summary"] = model_summary
    existing_rec["evidence_summary"] = evidence_summary
    existing_rec["deployment_config"] = deployment_config

    logger.info("Synthesized recommendation: %d evidence items summarized", len(evidence_items))

    return {
        "recommendation": existing_rec,
        "current_phase": "recommending",
        "phase_history": ["recommendation_synthesized"],
    }


def _format_params(params: int | float | None) -> str:
    if not params:
        return "Unknown"
    if params >= 1e9:
        return f"{params / 1e9:.1f}B"
    if params >= 1e6:
        return f"{params / 1e6:.0f}M"
    return str(params)
