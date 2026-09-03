"""Design suggestion node — LLM-generated architectural recommendation.

Gathers all accumulated state (user inputs, evidence, sizing results)
and asks the LLM for a structured natural language design suggestion.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from agents.inference_planner.prompts import (
    DESIGN_SUGGESTION_SYSTEM,
    DESIGN_SUGGESTION_USER,
    LANGUAGE_INSTRUCTION,
    LANGUAGE_NAMES,
)
from agents.inference_planner.state import PlannerState

logger = logging.getLogger(__name__)

_mlflow_available = False
try:
    import mlflow

    _mlflow_available = True
except ImportError:
    pass


def _precision_label(bits: int) -> str:
    return {4: "INT4", 8: "FP8", 16: "FP16", 32: "FP32"}.get(bits, f"FP{bits}")


def _build_architecture_detail(arch: dict) -> str:
    """Build a human-readable architecture detail string."""
    parts = []

    num_experts = arch.get("num_experts_total")
    active_params = arch.get("parameter_count_active")
    if num_experts:
        active_b = ""
        if active_params and isinstance(active_params, (int, float)):
            ab = active_params / 1e9 if active_params > 1e6 else active_params
            active_b = f", {ab:.1f}B active per token"
        parts.append(f"MoE with {num_experts} experts{active_b}")

    sliding_layers = arch.get("sliding_attention_layers")
    full_layers = arch.get("full_attention_layers")
    if sliding_layers is not None and full_layers is not None:
        parts.append(f"{sliding_layers} linear-attention + {full_layers} full-attention layers (hybrid)")

    if not parts:
        parts.append("Dense transformer")

    return "; ".join(parts)


def _extract_rhoai_validation(evidence_items: list[dict]) -> str:
    """Extract RHOAI validation status from evidence items."""
    for item in evidence_items:
        summary = (item.get("summary") or "").lower()
        if "validated" in summary and ("rhoai" in summary or "rhaiis" in summary or "openshift ai" in summary):
            raw = item.get("summary") or ""
            vllm_ver = ""
            for token in raw.split():
                if token.startswith("0.") or token.startswith("v0."):
                    vllm_ver = token.rstrip(",;.)")
                    break
            if vllm_ver:
                return f"Yes (vLLM {vllm_ver} per model card)"
            return "Yes (per model card)"
    return "Not found in evidence"


def _build_context(state: PlannerState) -> dict[str, str]:
    """Extract all relevant context from state into template variables."""
    arch = state.get("model_architecture") or {}
    rec = state.get("recommendation") or {}
    mem = rec.get("memory_estimate") or {}
    perf = rec.get("performance_forecast") or {}
    cost = rec.get("cost_estimate") or {}
    workload = state.get("workload_profile") or {}
    hardware = state.get("hardware_inventory") or {}
    model_analysis = state.get("model_analysis") or {}

    gpu_type = workload.get("gpu_type") or hardware.get("gpu_type") or "Unknown"
    gpu_count = workload.get("gpu_count") or hardware.get("gpu_count") or 1

    use_cases = workload.get("use_case_presets", [])
    use_case_str = ", ".join(use_cases) if use_cases else "General inference"

    evidence_items = state.get("evidence_items", [])
    evidence_lines = []
    for item in evidence_items[:15]:
        cat = item.get("category", "")
        title = item.get("title", "")
        summary = (item.get("summary") or "")[:200]
        evidence_lines.append(f"- [{cat}] {title}: {summary}")
    evidence_summary = "\n".join(evidence_lines) if evidence_lines else "No evidence collected"

    arch_type = "Unknown"
    if arch.get("architecture_type") and arch["architecture_type"] != "unknown":
        arch_type = arch["architecture_type"]
        if arch.get("architecture_names"):
            arch_type = f"{arch_type} ({arch['architecture_names'][0]})"
    elif arch.get("architecture_names"):
        arch_type = arch["architecture_names"][0]

    platform = workload.get("platform") or hardware.get("environment_type") or "on-premise"

    # Architecture detail
    architecture_detail = _build_architecture_detail(arch)

    # Quantization / weight source
    quantization_method = mem.get("quantization_method") or model_analysis.get("weight_precision") or "not specified"
    weight_source = mem.get("weight_source", "param_count")

    # KV layout and concurrency
    arch_used = mem.get("arch_used") or {}
    kv_layout = model_analysis.get("kv_layout") or "standard"
    if mem.get("is_hybrid_attention"):
        kv_layout = "hybrid_sliding"
    elif mem.get("is_mla"):
        kv_layout = "MLA (compressed KV)"
    seq_len = arch_used.get("seq_len", 4096)
    concurrency_low = mem.get("concurrency_low", "?")
    concurrency_high = mem.get("concurrency_high", "?")
    effective_concurrent = arch_used.get("max_concurrent", "?")

    # RHOAI validation from evidence
    rhoai_validated = _extract_rhoai_validation(evidence_items)

    # MoE forecast warning
    is_moe = bool(arch.get("num_experts_total"))
    active_params = arch.get("parameter_count_active")
    active_params_b = ""
    if active_params and isinstance(active_params, (int, float)):
        active_params_b = f"{active_params / 1e9 if active_params > 1e6 else active_params:.1f}"

    moe_warning = ""
    if is_moe:
        moe_warning = (
            f"WARNING: This is a MoE model. The roofline values below use total model weight "
            f"for bandwidth calculation. Actual throughput depends on active parameters "
            f"({active_params_b or '?'}B), expert batching, and routing distribution. "
            f"Do NOT use these numbers as production capacity estimates.\n"
        )

    # TTFT input length assumption (must match sizing.py)
    ttft_input_tokens = perf.get("ttft_assumed_input_tokens", 512)

    return {
        "model_repo_id": state.get("model_repo_id", "Unknown"),
        "architecture_type": arch_type,
        "architecture_detail": architecture_detail,
        "parameters_display": mem.get("parameters_billions", "?"),
        "context_length": arch.get("max_position_embeddings") or arch.get("max_sequence_length") or "Unknown",
        "precision_bits": mem.get("precision_bits", 16),
        "precision_label": _precision_label(mem.get("precision_bits", 16)),
        "quantization_method": quantization_method,
        "weight_source": weight_source,
        "platform": platform,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "total_vram_gb": mem.get("total_available_gb", "?"),
        "model_weights_gb": mem.get("model_weights_gb", "?"),
        "kv_cache_gb": mem.get("kv_cache_gb", "?"),
        "kv_layout": kv_layout,
        "seq_len": seq_len,
        "concurrency_low": concurrency_low,
        "concurrency_high": concurrency_high,
        "effective_concurrent": effective_concurrent,
        "overhead_gb": mem.get("overhead_min_total_gb") or mem.get("overhead_gb", "~5-12"),
        "total_required_gb": mem.get("total_required_min_gb") or mem.get("total_required_gb", "?"),
        "total_available_gb": mem.get("total_available_gb", "?"),
        "utilization_pct": mem.get("utilization_pct", "?"),
        "fits": "Yes" if mem.get("fits") else "No",
        "rhoai_validated": rhoai_validated,
        "use_cases": use_case_str,
        "target_users": workload.get("target_end_users", "?"),
        "max_concurrent": workload.get("max_concurrent_requests", "?"),
        "ttft_target_ms": workload.get("ttft_ms", "?"),
        "tpot_target_ms": workload.get("tpot_ms", "?"),
        "decode_tps": perf.get("theoretical_decode_tps", "?"),
        "estimated_tpot_ms": perf.get("estimated_tpot_ms", "?"),
        "estimated_ttft_ms": perf.get("estimated_ttft_ms", "?"),
        "ridge_batch": perf.get("ridge_batch_size", "?"),
        "max_batch_at_target": perf.get("max_batch_at_target_tpot", "?"),
        "ttft_input_tokens": ttft_input_tokens,
        "moe_warning": moe_warning,
        "cost_summary": cost.get("summary", "No cost data"),
        "evidence_summary": evidence_summary,
    }


async def generate_design_suggestion(state: PlannerState) -> dict[str, Any]:
    """Call LLM to generate an inference design suggestion from accumulated state."""
    from backend.config import get_settings
    from backend.tracing import append_trace_record, build_trace_record

    settings = get_settings()
    api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = settings.openai_base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = settings.llm_model_name or os.environ.get("LLM_MODEL_NAME", "gpt-4o")
    verify_ssl = settings.verify_ssl

    ctx = _build_context(state)
    user_prompt = DESIGN_SUGGESTION_USER.format(**{k: str(v) for k, v in ctx.items()})

    language = state.get("language", "en")
    language_name = LANGUAGE_NAMES.get(language)
    lang_instruction = ""
    if language_name:
        lang_instruction = LANGUAGE_INSTRUCTION.format(language_name=language_name)
        user_prompt += lang_instruction

    session_id = state.get("session_id", "")

    if not api_key:
        logger.warning("No LLM API key configured; skipping design suggestion")
        fallback = _build_fallback_suggestion(ctx)
        existing_rec = state.get("recommendation") or {}
        existing_rec["design_suggestion"] = fallback
        return {
            "recommendation": existing_rec,
            "phase_history": ["design_suggestion_fallback"],
        }

    span = None
    if _mlflow_available:
        from backend.tracing import is_tracing_enabled

        if is_tracing_enabled():
            try:
                span = mlflow.start_span(name="design_suggestion")
                span.set_inputs({
                    "model_repo_id": ctx.get("model_repo_id"),
                    "gpu_type": ctx.get("gpu_type"),
                    "precision_label": ctx.get("precision_label"),
                    "model_used": model_name,
                    "temperature": 0.3,
                    "max_tokens": 1500,
                })
                span.set_attributes({
                    "session_id": session_id,
                    "platform": ctx.get("platform", ""),
                    "utilization_pct": str(ctx.get("utilization_pct", "")),
                })
            except Exception:
                logger.debug("MLflow span creation skipped", exc_info=True)

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0), verify=verify_ssl
        ) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": DESIGN_SUGGESTION_SYSTEM + lang_instruction},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 1500,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            if not resp.content:
                raise ValueError(f"LLM returned empty response (HTTP {resp.status_code})")
            data = resp.json()
            suggestion_text = data["choices"][0]["message"]["content"].strip()

        latency_ms = (time.monotonic() - t0) * 1000
        logger.info("Design suggestion generated (%d chars, %.0fms)", len(suggestion_text), latency_ms)

        if span:
            try:
                span.set_outputs({"suggestion_length": len(suggestion_text), "source": "llm"})
                span.end()
            except Exception:
                pass

        trace_id = ""
        if _mlflow_available and is_tracing_enabled():
            try:
                active = mlflow.get_current_active_span()
                if active:
                    trace_id = active.request_id
            except Exception:
                pass

        try:
            record = build_trace_record(
                session_id=session_id,
                model_used=model_name,
                input_context={k: str(v) for k, v in ctx.items()},
                system_prompt=DESIGN_SUGGESTION_SYSTEM,
                user_prompt=user_prompt,
                output=suggestion_text,
                latency_ms=latency_ms,
                trace_id=trace_id,
            )
            append_trace_record(record)
        except Exception:
            logger.warning("Failed to write JSONL trace record", exc_info=True)

        existing_rec = state.get("recommendation") or {}
        existing_rec["design_suggestion"] = {
            "content": suggestion_text,
            "model_used": model_name,
            "source": "llm",
        }

        return {
            "recommendation": existing_rec,
            "phase_history": ["design_suggestion_generated"],
        }

    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        logger.warning(
            "Design suggestion LLM call failed: [%s] %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )

        if span:
            try:
                span.set_status("ERROR")
                span.end()
            except Exception:
                pass

        fallback = _build_fallback_suggestion(ctx)

        try:
            record = build_trace_record(
                session_id=session_id,
                model_used=f"fallback (error: {type(exc).__name__})",
                input_context={k: str(v) for k, v in ctx.items()},
                system_prompt=DESIGN_SUGGESTION_SYSTEM,
                user_prompt=user_prompt,
                output=fallback["content"],
                latency_ms=latency_ms,
            )
            append_trace_record(record)
        except Exception:
            pass

        existing_rec = state.get("recommendation") or {}
        existing_rec["design_suggestion"] = fallback
        return {
            "recommendation": existing_rec,
            "phase_history": ["design_suggestion_fallback"],
        }


def _build_fallback_suggestion(ctx: dict[str, str]) -> dict[str, Any]:
    """Build a deterministic fallback suggestion when LLM is unavailable."""
    raw_gpu = str(ctx.get("gpu_count", "1"))
    gpu_count = int(raw_gpu) if raw_gpu.isdigit() else 1
    raw_util = str(ctx.get("utilization_pct", "50")).replace(".", "")
    utilization_pct = float(ctx.get("utilization_pct", 50)) if raw_util.isdigit() else 50

    if gpu_count == 1:
        parallelism = "Single-GPU deployment — no parallelism needed."
    elif gpu_count <= 4:
        parallelism = f"Tensor parallelism (TP={gpu_count}) across {gpu_count} GPUs."
    else:
        half = gpu_count // 2
        parallelism = (
            f"Tensor parallelism (TP={gpu_count}) recommended. "
            f"Consider TP={half} with 2 replicas for redundancy."
        )

    fits = ctx.get("fits", "Yes")
    memory_note = (
        "Memory headroom is adequate."
        if fits == "Yes"
        else "WARNING: Model does not fit in available VRAM. Consider quantization or additional GPUs."
    )

    # Low utilization guidance
    low_util_note = ""
    if fits == "Yes" and utilization_pct < 50:
        gpu_type = ctx.get("gpu_type", "")
        mig_gpus = {"H100-80GB", "H200-141GB", "B200-192GB", "B300-288GB", "GB200-192GB", "GB300-288GB", "A100-80GB", "A100-40GB"}
        if gpu_type in mig_gpus:
            mig_hint = (
                f"- GPU utilization is only {utilization_pct:.0f}% — consider using "
                f"**NVIDIA MIG** to partition the GPU into smaller instances, "
                f"co-locating multiple models or serving replicas\n"
            )
        else:
            mig_hint = (
                f"- GPU utilization is only {utilization_pct:.0f}% — the selected "
                f"GPU is over-provisioned for this model\n"
            )
        vram = ctx.get("total_available_gb", "?")
        low_util_note = (
            f"{mig_hint}"
            f"- Alternatively, deploy a larger or higher-precision variant to "
            f"better utilize the available {vram} GB VRAM "
            f"(e.g., FP16 variant or a bigger model in the same family)\n"
        )

    precision_bits = int(ctx.get("precision_bits", 16)) if str(ctx.get("precision_bits", "16")).isdigit() else 16
    if precision_bits >= 16:
        alt = "- Consider FP8 quantization for ~50% memory savings with minimal quality loss"
    elif precision_bits == 8:
        alt = "- Consider INT4 quantization (AWQ/GPTQ) for further ~50% memory savings if quality permits"
    else:
        alt = "- Consider increasing GPU count or using a higher-memory GPU"

    considerations = (
        f"- {memory_note}\n"
        f"- Use cases: {ctx.get('use_cases', 'general')}\n"
        f"- Target {ctx.get('max_concurrent', '?')} concurrent requests with "
        f"TPOT target {ctx.get('tpot_target_ms', '?')}ms"
    )
    if low_util_note:
        considerations += f"\n{low_util_note.rstrip()}"

    return {
        "content": (
            f"### Architecture Direction\n"
            f"{parallelism} Deploy on RHOAI with vLLM as the serving engine. "
            f"{memory_note}\n\n"
            f"### Key Considerations\n"
            f"{considerations}\n\n"
            f"### Risk Factors\n"
            f"- Actual performance is typically 50-70% of theoretical roofline\n"
            f"- KV cache pressure increases with concurrent requests\n\n"
            f"### Alternative Approaches\n"
            f"{alt}"
        ),
        "model_used": "fallback",
        "source": "deterministic",
    }
