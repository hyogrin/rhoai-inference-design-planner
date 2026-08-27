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
from connectors.community_search import CommunitySearchConnector
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
    token = state.get("hf_token") or os.getenv("HF_TOKEN")

    if not repo_id:
        return {
            "error": "No model repo_id provided",
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

    from datetime import datetime, timezone
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
        retrieved_at=datetime.now(timezone.utc),
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

    token = state.get("hf_token") or os.getenv("HF_TOKEN")
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
    """Discover community evidence via web search."""
    repo_id = state.get("model_repo_id", "")
    if not repo_id:
        return {"phase_history": ["community_evidence_skipped"]}

    connector = CommunitySearchConnector()

    try:
        evidence_items = await connector.search_model_evidence(
            repo_id=repo_id,
            evidence_types=["compatibility", "performance", "deployment"],
            max_results_per_type=3,
        )
    except Exception as exc:
        logger.warning("Community evidence search failed for %s: %s", repo_id, exc)
        return {"phase_history": ["community_evidence_failed"]}

    new_evidence = [item.model_dump(mode="json") for item in evidence_items]
    logger.info("Found %d community evidence items for %s", len(new_evidence), repo_id)

    return {
        "evidence_items": new_evidence,
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
