"""Intake node - normalizes initial input."""

from agents.inference_planner.state import PlannerState


async def normalize_intake(state: PlannerState) -> dict:
    """Normalize and validate the initial model/hardware intake."""
    return {
        "current_phase": "discovering",
        "current_step": 2,
        "phase_history": ["intake_complete"],
    }
