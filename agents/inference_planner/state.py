"""Planner state — TypedDict-based state for LangGraph."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langgraph.graph import MessagesState


class PlannerState(MessagesState):
    """Full state for the inference design planner graph.

    Fields using Annotated[..., operator.add] allow concurrent updates
    from parallel nodes (fan-out pattern). Each node returns only the
    NEW items to append, not the full list.
    """

    # Session
    session_id: str
    current_phase: str
    current_step: int
    language: str

    # Model intake
    model_repo_id: str
    model_revision: str
    hf_token: str | None

    # Discovered data (stored as dicts for serialization)
    model_identity: dict[str, Any] | None
    model_architecture: dict[str, Any] | None
    hardware_inventory: dict[str, Any] | None

    # Evidence — uses add reducer for concurrent parallel updates
    evidence_items: Annotated[list[dict[str, Any]], operator.add]

    # Validation
    validation_report: dict[str, Any] | None

    # Model analysis (LLM-interpreted, user-confirmed)
    model_analysis: dict[str, Any] | None

    # Workload (from HITL)
    workload_profile: dict[str, Any] | None

    # Results
    recommendation: dict[str, Any] | None
    view_model: dict[str, Any] | None

    # Control
    error: str | None
    phase_history: Annotated[list[str], operator.add]
