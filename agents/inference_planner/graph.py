"""LangGraph definition for the Inference Design Planner."""

from langgraph.graph import END, START, StateGraph

from agents.inference_planner.nodes.discovery import (
    check_rhoai_compatibility,
    discover_redhat_evaluations,
    discover_vllm_recipe,
    fetch_huggingface_metadata,
    fetch_pricing,
)
from agents.inference_planner.nodes.hitl import collect_workload_interrupt
from agents.inference_planner.nodes.intake import normalize_intake
from agents.inference_planner.nodes.model_analysis import interpret_model_config
from agents.inference_planner.nodes.design_suggestion import generate_design_suggestion
from agents.inference_planner.nodes.sizing import (
    calculate_cost,
    calculate_memory_capacity,
    calculate_performance_forecast,
)
from agents.inference_planner.nodes.synthesis import synthesize_recommendation
from agents.inference_planner.nodes.validation import route_readiness, validate_discovery
from agents.inference_planner.nodes.verification import (
    finalize_view_model,
    verify_recommendation,
)
from agents.inference_planner.state import PlannerState


def _fan_out_discovery(state: PlannerState) -> list[str]:
    """Fan-out to parallel discovery nodes after HF metadata fetch."""
    return [
        "discover_vllm_recipe",
        "discover_redhat_evaluations",
        "check_rhoai_compatibility",
        "fetch_pricing",
    ]


def build_graph() -> StateGraph:
    """Construct the Inference Design Planner graph (uncompiled)."""
    graph = StateGraph(PlannerState)

    # --- Nodes ---
    graph.add_node("normalize_intake", normalize_intake)
    graph.add_node("fetch_huggingface_metadata", fetch_huggingface_metadata)
    graph.add_node("discover_vllm_recipe", discover_vllm_recipe)
    graph.add_node("discover_redhat_evaluations", discover_redhat_evaluations)
    graph.add_node("check_rhoai_compatibility", check_rhoai_compatibility)
    graph.add_node("fetch_pricing", fetch_pricing)
    graph.add_node("validate_discovery", validate_discovery)
    graph.add_node("interpret_model_config", interpret_model_config)
    graph.add_node("collect_workload_interrupt", collect_workload_interrupt)
    graph.add_node("calculate_memory_capacity", calculate_memory_capacity)
    graph.add_node("calculate_performance_forecast", calculate_performance_forecast)
    graph.add_node("calculate_cost", calculate_cost)
    graph.add_node("generate_design_suggestion", generate_design_suggestion)
    graph.add_node("synthesize_recommendation", synthesize_recommendation)
    graph.add_node("verify_recommendation", verify_recommendation)
    graph.add_node("finalize_view_model", finalize_view_model)

    # --- Edges ---
    # Intake
    graph.add_edge(START, "normalize_intake")
    graph.add_edge("normalize_intake", "fetch_huggingface_metadata")

    # Fan-out: HF metadata -> parallel discovery nodes
    graph.add_conditional_edges(
        "fetch_huggingface_metadata",
        _fan_out_discovery,
        [
            "discover_vllm_recipe",
            "discover_redhat_evaluations",
            "check_rhoai_compatibility",
            "fetch_pricing",
        ],
    )

    # Fan-in: all discovery nodes -> validate_discovery
    graph.add_edge("discover_vllm_recipe", "validate_discovery")
    graph.add_edge("discover_redhat_evaluations", "validate_discovery")
    graph.add_edge("check_rhoai_compatibility", "validate_discovery")
    graph.add_edge("fetch_pricing", "validate_discovery")

    # Conditional: readiness check
    graph.add_conditional_edges(
        "validate_discovery",
        route_readiness,
        {"collect_workload": "interpret_model_config", "blocked": END},
    )

    # Model analysis (LLM) then workload collection
    graph.add_edge("interpret_model_config", "collect_workload_interrupt")

    # Sizing pipeline
    graph.add_edge("collect_workload_interrupt", "calculate_memory_capacity")
    graph.add_edge("calculate_memory_capacity", "calculate_performance_forecast")
    graph.add_edge("calculate_performance_forecast", "calculate_cost")

    # Design suggestion (LLM) then synthesis
    graph.add_edge("calculate_cost", "generate_design_suggestion")
    graph.add_edge("generate_design_suggestion", "synthesize_recommendation")
    graph.add_edge("synthesize_recommendation", "verify_recommendation")
    graph.add_edge("verify_recommendation", "finalize_view_model")
    graph.add_edge("finalize_view_model", END)

    return graph


def compile_graph(checkpointer=None):
    """Compile the planner graph with an optional checkpointer."""
    return build_graph().compile(checkpointer=checkpointer)


# Pre-built compiled graph (no checkpointing, for tests)
planner_graph = compile_graph()
