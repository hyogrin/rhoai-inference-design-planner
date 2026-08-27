"""Human-in-the-loop node for collecting workload configuration."""

from langgraph.types import interrupt

from agents.inference_planner.state import PlannerState


async def collect_workload_interrupt(state: PlannerState) -> dict:
    """Interrupt execution to collect workload requirements from the user."""
    model_id = state.get("model_repo_id", "")
    architecture = state.get("model_architecture") or {}
    evidence_count = len(state.get("evidence_items", []))

    interrupt_payload = {
        "type": "workload_configuration",
        "model_repo_id": model_id,
        "model_analysis": state.get("model_analysis") or {},
        "architecture_summary": {
            "type": architecture.get("architecture_type", "unknown"),
            "family": architecture.get("family"),
            "parameters": architecture.get("parameter_count_total"),
            "max_context": architecture.get("max_position_embeddings"),
            "num_hidden_layers": architecture.get("num_hidden_layers"),
            "hidden_size": architecture.get("hidden_size"),
            "num_attention_heads": architecture.get("num_attention_heads"),
            "num_key_value_heads": architecture.get("num_kv_heads"),
            "head_dim": architecture.get("head_dim"),
            "sliding_window": architecture.get("sliding_window"),
            "sliding_attention_layers": architecture.get("sliding_attention_layers"),
            "full_attention_layers": architecture.get("full_attention_layers"),
            "global_head_dim": architecture.get("global_head_dim"),
            "num_global_kv_heads": architecture.get("num_global_kv_heads"),
            "kv_lora_rank": architecture.get("kv_lora_rank"),
            "qk_rope_head_dim": architecture.get("qk_rope_head_dim"),
        },
        "evidence_collected": evidence_count,
        "required_fields": [
            {
                "field": "gpu_type",
                "label": "GPU Type",
                "type": "select",
                "options": ["H100-80GB", "A100-80GB", "A100-40GB", "L40S-48GB", "L4-24GB"],
            },
            {
                "field": "gpu_count",
                "label": "Number of GPUs",
                "type": "number",
                "min": 1,
                "max": 16,
                "default": 1,
            },
            {
                "field": "target_latency_ms",
                "label": "Target TTFT (ms)",
                "type": "number",
                "min": 50,
                "default": 500,
            },
            {
                "field": "max_concurrent_requests",
                "label": "Max Concurrent Requests",
                "type": "number",
                "min": 1,
                "default": 32,
            },
            {
                "field": "max_sequence_length",
                "label": "Max Sequence Length",
                "type": "number",
                "min": 512,
                "default": 4096,
            },
            {
                "field": "rhoai_version",
                "label": "RHOAI Version",
                "type": "select",
                "options": ["2.16", "2.17", "2.18", "3.0"],
                "default": "2.18",
            },
        ],
        "message": (
            f"Model {model_id} analysis complete. "
            f"{evidence_count} evidence items collected. "
            "Please configure your deployment requirements."
        ),
    }

    workload_response = interrupt(interrupt_payload)
    workload_data = workload_response if isinstance(workload_response, dict) else {}

    return {
        "workload_profile": workload_data,
        "current_phase": "sizing",
        "current_step": 4,
        "phase_history": ["workload_collected"],
    }
