"""Validation node - assess readiness for sizing based on discovery results."""

from agents.inference_planner.state import PlannerState


async def validate_discovery(state: PlannerState) -> dict:
    """Validate gathered evidence and determine readiness for the sizing phase."""
    issues: list[str] = []
    warnings: list[str] = []

    model_architecture = state.get("model_architecture")
    if not model_architecture:
        issues.append("model_architecture not available — HuggingFace fetch may have failed")
    else:
        if not model_architecture.get("parameter_count_total"):
            warnings.append("parameter count unknown — sizing estimates will be approximate")
        if not model_architecture.get("architecture_type") or model_architecture.get("architecture_type") == "unknown":
            warnings.append("architecture type unknown — using generic estimation")

    evidence_items = state.get("evidence_items", [])
    evidence_count = len(evidence_items)
    if evidence_count == 0:
        warnings.append("no external evidence collected — recommendation will rely on heuristics")

    has_vllm_recipe = any(
        e.get("source_type") == "vllm_recipe" for e in evidence_items
    )
    has_community = any(
        e.get("source_type") == "community_report" for e in evidence_items
    )

    if issues:
        status = "blocked"
    elif warnings:
        status = "ready_with_limitations"
    else:
        status = "ready_for_sizing"

    validation_report = {
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "evidence_summary": {
            "total_items": evidence_count,
            "has_vllm_recipe": has_vllm_recipe,
            "has_community_evidence": has_community,
            "has_architecture": model_architecture is not None,
        },
    }

    return {
        "validation_report": validation_report,
        "current_phase": "validating",
        "current_step": 3,
        "phase_history": ["validation_complete"],
    }


def route_readiness(state: PlannerState) -> str:
    """Route based on validation result."""
    validation_report = state.get("validation_report")
    if validation_report:
        status = validation_report.get("status", "blocked")
        if status in ("ready_for_sizing", "ready_with_limitations"):
            return "collect_workload"
        return "blocked"
    return "collect_workload"
