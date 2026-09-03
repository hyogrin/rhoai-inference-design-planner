"""Sizing nodes — deterministic memory, cost, and throughput calculations.

These are pure functions that compute estimates from collected evidence
and the user's workload profile. No LLM dependency.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from agents.inference_planner.state import PlannerState

logger = logging.getLogger(__name__)

GPU_SPECS: dict[str, dict[str, Any]] = {
    "B300-288GB": {"memory_gb": 288, "bandwidth_tbps": 8.0, "flops_tflops": 2250, "arch": "blackwell"},
    "GB300-288GB": {"memory_gb": 288, "bandwidth_tbps": 8.0, "flops_tflops": 2250, "arch": "blackwell"},
    "B200-192GB": {"memory_gb": 192, "bandwidth_tbps": 8.0, "flops_tflops": 2250, "arch": "blackwell"},
    "GB200-192GB": {"memory_gb": 192, "bandwidth_tbps": 8.0, "flops_tflops": 2250, "arch": "blackwell"},
    "H200-141GB": {"memory_gb": 141, "bandwidth_tbps": 4.8, "flops_tflops": 989, "arch": "hopper"},
    "H100-80GB": {"memory_gb": 80, "bandwidth_tbps": 3.35, "flops_tflops": 989, "arch": "hopper"},
    "MI300X-192GB": {"memory_gb": 192, "bandwidth_tbps": 5.3, "flops_tflops": 1307, "arch": "cdna3"},
    "A100-80GB": {"memory_gb": 80, "bandwidth_tbps": 2.0, "flops_tflops": 312, "arch": "ampere"},
    "A100-40GB": {"memory_gb": 40, "bandwidth_tbps": 1.55, "flops_tflops": 312, "arch": "ampere"},
    "RTX-PRO-6000-96GB": {"memory_gb": 96, "bandwidth_tbps": 1.597, "flops_tflops": 500, "arch": "blackwell"},
    "L40S-48GB": {"memory_gb": 48, "bandwidth_tbps": 0.864, "flops_tflops": 362, "arch": "ada"},
    "A10G-24GB": {"memory_gb": 24, "bandwidth_tbps": 0.6, "flops_tflops": 125, "arch": "ampere"},
    "L4-24GB": {"memory_gb": 24, "bandwidth_tbps": 0.3, "flops_tflops": 121, "arch": "ada"},
    "T4-16GB": {"memory_gb": 16, "bandwidth_tbps": 0.3, "flops_tflops": 65, "arch": "turing"},
}

# Quantization format → required GPU architectures
QUANT_GPU_REQUIREMENTS: dict[str, set[str]] = {
    "nvfp4": {"blackwell"},
}

GPU_HOURLY_COST: dict[str, float] = {
    "B300-288GB": 17.80,
    "GB300-288GB": 17.80,
    "B200-192GB": 14.24,
    "GB200-192GB": 14.24,
    "H200-141GB": 8.5,
    "H100-80GB": 5.5,
    "MI300X-192GB": 5.0,
    "A100-80GB": 3.5,
    "A100-40GB": 2.5,
    "RTX-PRO-6000-96GB": 3.0,
    "L40S-48GB": 2.0,
    "A10G-24GB": 1.2,
    "L4-24GB": 0.8,
    "T4-16GB": 0.5,
}


def _get_param_count(state: PlannerState) -> float | None:
    """Extract parameter count (in billions) from architecture or model name."""
    import re

    arch = state.get("model_architecture") or {}

    # Primary: parameter_count_total from HuggingFace safetensors metadata
    params = arch.get("parameter_count_total")
    if params and isinstance(params, (int, float)):
        if params > 1e6:
            return params / 1e9
        return params

    # Fallback 1: extract from model name
    # Match XB preceded by separator (-, _, x) to avoid version numbers
    # e.g., "Qwen3.5-35B-A3B" -> 35B, "Mixtral-8x22B" -> 22B
    repo_id = state.get("model_repo_id", "")
    name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    matches = re.findall(r"(?:^|[-_x])(\d+\.?\d*)B", name)
    if matches:
        # Use the largest value (most likely the param count, not version)
        val = max(float(m) for m in matches)
        if val > 0:
            return val

    # Fallback 2: stored from LLM estimation (set during async path)
    llm_estimate = arch.get("estimated_parameters_b")
    if llm_estimate and isinstance(llm_estimate, (int, float)):
        return float(llm_estimate)

    return None


async def _estimate_params_with_llm(repo_id: str) -> float | None:
    """Use LLM to estimate parameter count when regex and HF metadata fail."""
    import os
    import httpx

    from backend.config import get_settings
    settings = get_settings()

    api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = settings.openai_base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = settings.llm_model_name or os.environ.get("LLM_MODEL_NAME", "gpt-4o")
    verify_ssl = settings.verify_ssl

    if not api_key:
        return None

    prompt = (
        f"What is the approximate parameter count (in billions) for the model '{repo_id}'? "
        f"Reply with ONLY a number (e.g., 7, 13, 70, 405). "
        f"If you are unsure, reply with your best estimate based on the model name."
    )

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0), verify=verify_ssl) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 20,
                    "temperature": 0,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()

            import re
            match = re.search(r"([\d.]+)", content)
            if match:
                val = float(match.group(1))
                if 0.1 <= val <= 2000:
                    logger.info("LLM estimated %s params as %.1fB", repo_id, val)
                    return val
    except Exception as exc:
        logger.warning("LLM param estimation failed for %s: %s", repo_id, exc)

    return None


def _precision_label(bits: int) -> str:
    """Map precision bits to a human-readable label."""
    return {4: "INT4", 8: "FP8", 16: "FP16", 32: "FP32"}.get(bits, f"FP{bits}")


def _estimate_model_memory_gb(params_b: float, precision_bits: int = 16) -> float:
    """Estimate model weight memory in GB."""
    bytes_per_param = precision_bits / 8
    return params_b * 1e9 * bytes_per_param / (1024**3)


async def calculate_memory_capacity(state: PlannerState) -> dict:
    """Calculate memory requirements and capacity analysis."""
    workload = state.get("workload_profile") or {}
    hardware = state.get("hardware_inventory") or {}
    gpu_type = workload.get("gpu_type") or hardware.get("gpu_type") or ""
    gpu_count = workload.get("gpu_count") or hardware.get("gpu_count") or 1

    params_b = _get_param_count(state)

    # LLM fallback if regex and metadata both failed
    if params_b is None:
        repo_id = state.get("model_repo_id", "")
        params_b = await _estimate_params_with_llm(repo_id)
    gpu_spec = GPU_SPECS.get(gpu_type)
    if not gpu_spec:
        for key, spec in GPU_SPECS.items():
            if key.startswith(gpu_type + "-") or key.startswith(gpu_type):
                gpu_spec = spec
                gpu_type = key
                break
    if not gpu_spec:
        logger.warning("Unknown GPU type '%s' — cannot estimate memory", gpu_type)
        return {
            "recommendation": {
                "memory_estimate": {
                    "error": f"Unknown GPU type: {gpu_type}",
                    "gpu_type": gpu_type,
                    "gpu_count": gpu_count,
                }
            },
            "phase_history": ["memory_calculation_failed"],
        }

    per_gpu_mem = gpu_spec["memory_gb"]
    total_gpu_mem = per_gpu_mem * gpu_count

    # Hardware compatibility check: quantization format vs GPU architecture
    arch = state.get("model_architecture") or {}
    repo_id = state.get("model_repo_id", "")
    repo_lower = repo_id.lower()
    gpu_arch = gpu_spec.get("arch", "unknown")

    hw_compat_warnings: list[str] = []
    hw_blocked = False
    blocked_reason = None

    for quant_format, required_archs in QUANT_GPU_REQUIREMENTS.items():
        if quant_format in repo_lower or quant_format in (arch.get("weight_precision") or "").lower():
            if gpu_arch not in required_archs:
                hw_blocked = True
                blocked_reason = (
                    f"{quant_format.upper()} quantization requires "
                    f"{'/'.join(a.title() for a in required_archs)}-class GPUs. "
                    f"{gpu_type} is {gpu_arch.title()} architecture and does not "
                    f"provide native {quant_format.upper()} execution support."
                )
                hw_compat_warnings.append(blocked_reason)
                break

    # Use model_analysis from LLM (user-confirmed) if available
    model_analysis = state.get("model_analysis") or {}
    precision_bits = model_analysis.get("effective_bits", 16)
    kv_layout = model_analysis.get("kv_layout", "standard_gqa")
    kv_bytes_per_element_from_analysis = model_analysis.get("kv_cache_bytes_per_element")

    # Model weight memory: checkpoint size preferred, else param_count × effective_bits
    checkpoint_size_bytes = arch.get("checkpoint_size_bytes")
    if checkpoint_size_bytes:
        model_mem = checkpoint_size_bytes / (1024**3)
    elif params_b:
        model_mem = _estimate_model_memory_gb(params_b, precision_bits)
    else:
        model_mem = None

    # KV cache estimate
    warnings: list[str] = []
    max_concurrent = workload.get("max_concurrent_requests", 32)
    arch = state.get("model_architecture") or {}
    num_layers = arch.get("num_hidden_layers")
    hidden_size = arch.get("hidden_size")
    num_kv_heads = arch.get("num_kv_heads") or arch.get("num_key_value_heads")
    num_attention_heads = arch.get("num_attention_heads")
    explicit_head_dim = arch.get("head_dim")

    if not all([num_layers, hidden_size, num_kv_heads, num_attention_heads]):
        warnings.append(
            "Architecture metadata incomplete — KV cache estimate uses model defaults and may be inaccurate"
        )
        num_layers = num_layers or 32
        hidden_size = hidden_size or 4096
        num_kv_heads = num_kv_heads or 8
        num_attention_heads = num_attention_heads or 32

    # Prefer explicit head_dim from model config (Gemma, Phi, etc.)
    if explicit_head_dim and isinstance(explicit_head_dim, int):
        head_dim = explicit_head_dim
    else:
        head_dim = hidden_size // max(num_attention_heads, 1)

    seq_len = workload.get("max_sequence_length") or 4096

    # KV cache bytes: use model_analysis value (user-confirmed) or default
    kv_bytes_per_element = kv_bytes_per_element_from_analysis or (1 if precision_bits <= 8 else 2)

    # KV layout from model_analysis determines calculation path
    kv_lora_rank = arch.get("kv_lora_rank")
    qk_rope_head_dim = arch.get("qk_rope_head_dim")
    is_mla = kv_layout == "mla" and kv_lora_rank is not None and qk_rope_head_dim is not None

    # Hybrid attention: sliding + full attention layers (e.g., Gemma 4)
    # Hybrid attention raw params
    sliding_layers = arch.get("sliding_attention_layers")
    full_layers = arch.get("full_attention_layers")
    sliding_window = arch.get("sliding_window")
    global_head_dim = arch.get("global_head_dim")
    num_global_kv_heads = arch.get("num_global_kv_heads")

    is_hybrid_attention = (
        kv_layout == "hybrid_sliding"
        and sliding_layers is not None
        and full_layers is not None
        and sliding_window is not None
    )

    if is_mla:
        # MLA stores a compressed KV latent per token per layer:
        # kv_lora_rank + qk_rope_head_dim elements × dtype_bytes
        mla_elements_per_token_per_layer = kv_lora_rank + qk_rope_head_dim
        kv_per_request_bytes = (
            num_layers * mla_elements_per_token_per_layer
            * kv_bytes_per_element * seq_len
        )
        # MLA KV cache is NOT sharded by TP — each GPU holds full copy
        kv_replicated_across_tp = True
    elif is_hybrid_attention:
        # Sliding attention layers: cache limited to sliding_window tokens
        sliding_cache_tokens = min(seq_len, sliding_window)
        sliding_kv_per_request = (
            2 * sliding_layers * num_kv_heads * head_dim
            * kv_bytes_per_element * sliding_cache_tokens
        )
        # Full attention layers: cache for the full sequence
        full_head_dim = global_head_dim if global_head_dim else head_dim
        full_kv_heads = num_global_kv_heads if num_global_kv_heads else num_kv_heads
        full_kv_per_request = (
            2 * full_layers * full_kv_heads * full_head_dim
            * kv_bytes_per_element * seq_len
        )
        kv_per_request_bytes = sliding_kv_per_request + full_kv_per_request
        kv_replicated_across_tp = False
    else:
        # Standard homogeneous attention
        kv_per_token_bytes = 2 * num_layers * num_kv_heads * head_dim * kv_bytes_per_element
        kv_per_request_bytes = kv_per_token_bytes * seq_len
        kv_replicated_across_tp = False

    # vLLM-style calculation: KV cache is bounded by available GPU memory
    # after model weights and runtime overhead are loaded.
    # Runtime overhead is PER-GPU (CUDA graphs, activations, comm buffers)
    overhead_min_per_gpu_gb = 2.5
    overhead_max_per_gpu_gb = 6.0
    gpu_usable_fraction = 0.90  # vLLM default gpu_memory_utilization

    # Per-GPU memory budget
    per_gpu_mem = total_gpu_mem / max(gpu_count, 1)
    per_gpu_budget = per_gpu_mem * gpu_usable_fraction
    model_mem_per_gpu = (model_mem or 0) / max(gpu_count, 1)

    # KV cache per request per GPU depends on whether it's TP-sharded or replicated
    if kv_replicated_across_tp:
        # MLA: each GPU holds full KV cache (not sharded)
        kv_per_request_per_gpu = kv_per_request_bytes
    else:
        # Standard/Hybrid: KV cache is sharded across TP GPUs
        kv_per_request_per_gpu = kv_per_request_bytes / max(gpu_count, 1)

    # Memory-feasible concurrency range (accounts for overhead uncertainty) — PER GPU
    available_optimistic = per_gpu_budget - model_mem_per_gpu - overhead_min_per_gpu_gb
    available_conservative = per_gpu_budget - model_mem_per_gpu - overhead_max_per_gpu_gb

    if available_optimistic > 0:
        concurrency_high = max(1, int(available_optimistic * (1024**3) / kv_per_request_per_gpu))
    else:
        concurrency_high = 0

    if available_conservative > 0:
        concurrency_low = max(1, int(available_conservative * (1024**3) / kv_per_request_per_gpu))
    else:
        concurrency_low = 0

    # Use optimistic estimate for display KV cache (with min overhead)
    effective_concurrent = min(max_concurrent, concurrency_high)
    # Aggregate KV cache across all GPUs
    if kv_replicated_across_tp:
        kv_cache_gb = (effective_concurrent * kv_per_request_bytes * gpu_count) / (1024**3)
    else:
        kv_cache_gb = (effective_concurrent * kv_per_request_bytes) / (1024**3)

    # Total aggregate range
    overhead_min_total = overhead_min_per_gpu_gb * gpu_count
    overhead_max_total = overhead_max_per_gpu_gb * gpu_count
    total_required_min = (model_mem or 0) + kv_cache_gb + overhead_min_total
    total_required_max = (model_mem or 0) + kv_cache_gb + overhead_max_total
    utilization = (total_required_min / total_gpu_mem * 100) if total_gpu_mem > 0 else 0

    # "Fits" means the model weights + max overhead fit AND there's room for at least 1 request
    # AND hardware is compatible with quantization format
    per_gpu_fits = model_mem_per_gpu + overhead_max_per_gpu_gb <= per_gpu_mem * 0.95
    fits = per_gpu_fits and concurrency_low >= 1 and not hw_blocked

    if concurrency_high < max_concurrent:
        warnings.append(
            f"Memory-feasible concurrency: approximately {concurrency_low}–{concurrency_high} "
            f"full-length active sequences (requested {max_concurrent}) at seq_len={seq_len} "
            f"on {per_gpu_mem:.0f} GiB/GPU × {gpu_count} with 90% utilization budget. "
            f"Consider increasing TP, reducing --max-model-len, or lowering concurrency."
        )

    memory_estimate: dict[str, Any] = {
        "model_weights_gb": round(model_mem, 1) if model_mem else None,
        "kv_cache_gb": round(kv_cache_gb, 1),
        "overhead_min_per_gpu_gb": overhead_min_per_gpu_gb,
        "overhead_max_per_gpu_gb": overhead_max_per_gpu_gb,
        "overhead_min_total_gb": round(overhead_min_total, 1),
        "overhead_max_total_gb": round(overhead_max_total, 1),
        "total_required_min_gb": round(total_required_min, 1),
        "total_required_max_gb": round(total_required_max, 1),
        "total_available_gb": total_gpu_mem,
        "per_gpu_mem_gb": round(per_gpu_mem, 1),
        "utilization_pct": round(utilization, 1),
        "fits": fits,
        "hw_blocked": hw_blocked,
        "hw_blocked_reason": blocked_reason,
        "precision_bits": precision_bits,
        "quantization_method": model_analysis.get("weight_precision") or None,
        "weight_source": "checkpoint" if checkpoint_size_bytes else "param_count",
        "kv_cache_dtype_bytes": kv_bytes_per_element,
        "parameters_billions": round(params_b, 1) if params_b else None,
        "gpu_type": gpu_type,
        "gpu_arch": gpu_arch,
        "gpu_count": gpu_count,
        "concurrency_low": concurrency_low,
        "concurrency_high": concurrency_high,
        "kv_per_request_gb": round(kv_per_request_bytes / (1024**3), 2),
        "is_hybrid_attention": is_hybrid_attention,
        "is_mla": is_mla,
        "kv_replicated_across_tp": kv_replicated_across_tp,
        "arch_used": {
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "seq_len": seq_len,
            "max_concurrent": max_concurrent,
            "sliding_window": sliding_window if is_hybrid_attention else None,
            "sliding_layers": sliding_layers,
            "full_layers": full_layers,
            "global_head_dim": global_head_dim if is_hybrid_attention else None,
            "num_global_kv_heads": num_global_kv_heads if is_hybrid_attention else None,
            "kv_lora_rank": kv_lora_rank if is_mla else None,
            "qk_rope_head_dim": qk_rope_head_dim if is_mla else None,
        },
    }
    all_warnings = hw_compat_warnings + warnings
    if all_warnings:
        memory_estimate["warnings"] = all_warnings

    logger.info("Memory estimate: %.1f–%.1f GiB required / %d GiB available (%.1f%%), "
                "memory-feasible concurrency: %d–%d (requested %d)",
                total_required_min, total_required_max, total_gpu_mem, utilization,
                concurrency_low, concurrency_high, max_concurrent)

    return {
        "recommendation": {"memory_estimate": memory_estimate},
        "phase_history": ["memory_calculated"],
    }


async def calculate_performance_forecast(state: PlannerState) -> dict:
    """Estimate theoretical throughput using a simplified Roofline model.

    Key concepts:
    - Decode (autoregressive) is MEMORY-BANDWIDTH bound at low batch sizes:
      Each output token requires reading all model weights once from HBM.
      Single-request throughput = HBM_bandwidth / model_weight_bytes

    - At higher batch sizes, multiple requests share the weight read,
      so throughput scales linearly with batch — UNTIL hitting the COMPUTE ceiling.
      Compute ceiling = FLOPS / (2 * params_per_token)

    - Prefill is COMPUTE bound:
      TTFT ≈ (2 * params * input_seq_len) / FLOPS

    The crossover point (ridge point) is:
      batch_ridge = compute_ceiling / memory_ceiling
    """
    workload = state.get("workload_profile") or {}
    hardware = state.get("hardware_inventory") or {}
    gpu_type = workload.get("gpu_type") or hardware.get("gpu_type") or ""
    gpu_count = workload.get("gpu_count") or hardware.get("gpu_count") or 1
    params_b = _get_param_count(state)
    if params_b is None:
        params_b = await _estimate_params_with_llm(state.get("model_repo_id", ""))

    if params_b is None:
        logger.warning("Cannot determine parameter count — skipping performance forecast")
        existing_rec = state.get("recommendation") or {}
        existing_rec["performance_forecast"] = {"error": "Parameter count unavailable"}
        return {
            "recommendation": existing_rec,
            "phase_history": ["performance_forecast_skipped"],
        }

    gpu_spec = GPU_SPECS.get(gpu_type)
    if not gpu_spec:
        logger.warning("Unknown GPU type '%s' — cannot forecast performance", gpu_type)
        existing_rec = state.get("recommendation") or {}
        existing_rec["performance_forecast"] = {"error": f"Unknown GPU type: {gpu_type}"}
        return {
            "recommendation": existing_rec,
            "phase_history": ["performance_forecast_failed"],
        }

    bandwidth_tbps = gpu_spec["bandwidth_tbps"]
    flops_tflops = gpu_spec["flops_tflops"]

    total_bandwidth_bytes = bandwidth_tbps * 1e12 * gpu_count
    total_flops = flops_tflops * 1e12 * gpu_count

    # Determine precision
    precision_bits = 16
    rec = state.get("recommendation") or {}
    mem_est = rec.get("memory_estimate") or {}
    if mem_est.get("precision_bits"):
        precision_bits = mem_est["precision_bits"]

    bytes_per_param = precision_bits / 8
    model_size_bytes = params_b * 1e9 * bytes_per_param

    # For MoE models, compute ceiling uses active params (not total)
    arch = state.get("model_architecture") or {}
    active_params = arch.get("parameter_count_active")
    if active_params and isinstance(active_params, (int, float)):
        active_params_b = active_params / 1e9 if active_params > 1e6 else active_params
    else:
        active_params_b = params_b

    is_moe = bool(arch.get("num_experts_total"))

    # --- Decode Roofline ---
    # Memory-bandwidth ceiling: max tokens/s for batch=1
    # Dense model: reading all weights once per token
    mem_ceiling_tps = total_bandwidth_bytes / model_size_bytes if model_size_bytes > 0 else 1

    # MoE: per-token weight read is approximately active params, not total
    moe_active_decode_tps = None
    if is_moe and active_params_b != params_b:
        active_model_bytes = active_params_b * 1e9 * bytes_per_param
        if active_model_bytes > 0:
            moe_active_decode_tps = total_bandwidth_bytes / active_model_bytes

    # Compute ceiling: max tokens/s regardless of batch
    # Each token requires ~2*active_params FLOPs (matmul forward pass)
    flops_per_token = 2 * active_params_b * 1e9
    compute_ceiling_tps = total_flops / flops_per_token if flops_per_token > 0 else 1e6

    # Ridge point: batch size where memory ceiling * batch = compute ceiling
    ridge_batch = compute_ceiling_tps / mem_ceiling_tps if mem_ceiling_tps > 0 else 1

    # Single-request metrics
    tpot_single_ms = (1000.0 / mem_ceiling_tps) if mem_ceiling_tps > 0 else 999
    ttft_input_tokens = 512
    prefill_flops = 2 * active_params_b * 1e9 * ttft_input_tokens
    ttft_ms = (prefill_flops / total_flops * 1000) if total_flops > 0 else 999

    # Target latency analysis
    target_tpot = workload.get("tpot_ms", 30)
    # At batch B, per-request TPOT ≈ tpot_single * max(1, B/ridge_batch) for B > ridge
    # Max batch where TPOT <= target:
    if tpot_single_ms <= target_tpot:
        # We can batch more; TPOT stays flat until ridge, then grows linearly
        max_batch_at_target = math.floor(ridge_batch * (target_tpot / tpot_single_ms))
    else:
        max_batch_at_target = 1
    max_batch_at_target = max(max_batch_at_target, 1)

    # --- Chart data: throughput vs batch size ---
    chart_data = []
    for batch in [1, 2, 4, 8, 16, 32, 64, 128]:
        # Roofline: throughput = min(batch * mem_ceiling, compute_ceiling)
        throughput = min(batch * mem_ceiling_tps, compute_ceiling_tps)
        # Per-request latency at this batch
        per_request_tpot = (1000.0 * batch / throughput) if throughput > 0 else 999
        chart_data.append({
            "batch_size": batch,
            "throughput_tokens_per_sec": round(throughput, 1),
            "latency_per_token_ms": round(per_request_tpot, 2),
        })

    forecast = {
        "theoretical_decode_tps": round(mem_ceiling_tps, 1),
        "compute_ceiling_tps": round(compute_ceiling_tps, 1),
        "ridge_batch_size": round(ridge_batch, 1),
        "estimated_tpot_ms": round(tpot_single_ms, 2),
        "estimated_ttft_ms": round(ttft_ms, 1),
        "ttft_assumed_input_tokens": ttft_input_tokens,
        "max_batch_at_target_tpot": max_batch_at_target,
        "is_moe": is_moe,
        "chart_data": chart_data,
        "explanation": {
            "model": f"{params_b:.1f}B{'(' + f'{active_params_b:.1f}B active)' if active_params_b != params_b else ''} params @ {_precision_label(precision_bits)}",
            "hardware": f"{gpu_count}× {gpu_type}",
            "bandwidth": f"{total_bandwidth_bytes / 1e12:.1f} TB/s total",
            "compute": f"{total_flops / 1e12:.0f} TFLOPS total",
            "memory_bound": f"Batch 1->{math.floor(ridge_batch)}: throughput scales linearly",
            "compute_bound": f"Batch >{math.floor(ridge_batch)}: saturates at {compute_ceiling_tps:.0f} tok/s",
        },
    }

    if is_moe:
        forecast["moe_warning"] = (
            "Dense-model approximation; actual MoE throughput depends on "
            "active parameters, routing distribution, and expert batching"
        )
        if moe_active_decode_tps is not None:
            forecast["moe_active_decode_tps"] = round(moe_active_decode_tps, 1)

    existing_rec = state.get("recommendation") or {}
    existing_rec["performance_forecast"] = forecast

    logger.info(
        "Performance forecast: mem_ceil=%.0f tok/s, compute_ceil=%.0f tok/s, "
        "ridge@batch=%d, TPOT=%.1fms, TTFT=%.1fms",
        mem_ceiling_tps, compute_ceiling_tps, math.floor(ridge_batch),
        tpot_single_ms, ttft_ms,
    )

    return {
        "recommendation": existing_rec,
        "phase_history": ["performance_forecasted"],
    }


async def calculate_cost(state: PlannerState) -> dict:
    """Calculate cost estimates based on platform, GPU type, and count.

    - Cloud (AWS/Azure/GCP): uses vendor instance pricing (on-demand, spot, reserved)
    - On-premise: calculates TCO (hardware + power + 3-year depreciation)
    """
    workload = state.get("workload_profile") or {}
    gpu_type = workload.get("gpu_type") or ""
    gpu_count = workload.get("gpu_count", 1)

    # Determine platform from evidence or workload
    evidence_items = state.get("evidence_items", [])
    platform = _detect_platform(evidence_items, workload)

    from connectors.pricing import PricingConnector
    pricing = PricingConnector()

    if platform == "on-premise":
        cost_estimate = _calculate_onprem_tco(pricing, gpu_type, gpu_count)
    else:
        cost_estimate = _calculate_cloud_cost(pricing, platform, gpu_type, gpu_count)

    existing_rec = state.get("recommendation") or {}
    existing_rec["cost_estimate"] = cost_estimate

    logger.info("Cost estimate (%s): %s", platform, cost_estimate.get("summary", ""))

    return {
        "recommendation": existing_rec,
        "phase_history": ["cost_calculated"],
    }


def _detect_platform(evidence_items: list[dict], workload: dict) -> str:
    """Detect platform from workload config (set by frontend)."""
    # Frontend passes platform directly
    platform = workload.get("platform")
    if platform and platform != "null":
        return platform

    # Infer from GPU type naming conventions
    gpu = workload.get("gpu_type", "")
    if "A10G" in gpu:
        return "aws"
    return "on-premise"


def _calculate_cloud_cost(pricing, platform: str, gpu_type: str, gpu_count: int) -> dict:
    """Calculate cloud cost from vendor pricing data."""
    # Map our GPU IDs to pricing connector GPU names
    gpu_name_map = {
        "B300-288GB": "B300", "GB300-288GB": "GB300",
        "B200-192GB": "B200", "GB200-192GB": "GB200",
        "H200-141GB": "H200", "H100-80GB": "H100",
        "MI300X-192GB": "MI300X", "A100-80GB": "A100-80GB", "A100-40GB": "A100",
        "RTX-PRO-6000-96GB": "RTX-PRO-6000",
        "L40S-48GB": "L40S", "A10G-24GB": "A10G", "L4-24GB": "L4", "T4-16GB": "T4",
    }
    search_gpu = gpu_name_map.get(gpu_type, gpu_type.split("-")[0])

    # GB200/GB300 NVL72 systems use B200/B300 GPUs internally;
    # search for both names so we find all matching cloud instances
    search_gpus = [search_gpu]
    _gb_fallback = {"GB200": "B200", "GB300": "B300"}
    if search_gpu in _gb_fallback:
        search_gpus.append(_gb_fallback[search_gpu])

    # Get all instances with this GPU type (no minimum filter)
    instances = []
    for sg in search_gpus:
        instances.extend(pricing.get_cloud_instances_for_gpu(sg, min_gpu_count=1))
    provider_keys = _PLATFORM_PROVIDER_MAP.get(platform, [])
    platform_instances = [i for i in instances if i["provider"].lower().replace(" ", "") in provider_keys]

    if not platform_instances:
        platform_instances = instances

    if not platform_instances:
        hourly_rate = GPU_HOURLY_COST.get(gpu_type)
        if hourly_rate is None:
            logger.warning("No pricing data for GPU '%s' on platform '%s'", gpu_type, platform)
            return {
                "type": "cloud",
                "platform": platform,
                "gpu_type": gpu_type,
                "gpu_count": gpu_count,
                "error": f"No pricing data available for {gpu_type} on {platform}",
                "summary": f"No pricing data available for {gpu_type} on {platform}",
            }
        hourly = hourly_rate * gpu_count
        return {
            "type": "cloud",
            "platform": platform,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "on_demand_hourly_usd": round(hourly, 2),
            "monthly_on_demand_usd": round(hourly * 730, 0),
            "source_url": None,
            "summary": f"${hourly:.2f}/hr (reference rate, no exact instance match)",
        }

    # Select instance: prefer exact match, then smallest that fits TP requirement
    # For TP to work, all GPUs must be in one machine, so pick the smallest
    # instance with enough GPUs. If none fits, use largest and multiply.
    import math as _math
    fits = [i for i in platform_instances if i["gpu_count"] >= gpu_count]
    if fits:
        best = min(fits, key=lambda x: x["gpu_count"])
    else:
        best = max(platform_instances, key=lambda x: x["gpu_count"])

    # Cloud instances are indivisible: ceil up to whole instances
    import math as _math
    instance_gpu_count = best["gpu_count"]
    num_instances = _math.ceil(gpu_count / instance_gpu_count)

    on_demand_hr = best["on_demand_hourly"] * num_instances
    spot_hr = best.get("spot_hourly", 0) * num_instances
    reserved_hr = best.get("reserved_1yr_hourly", 0) * num_instances

    provider_name = best["provider"]
    instance_name = best["instance_name"]

    source_urls = _PRICING_URLS.get(platform, "")

    note = ""
    actual_gpus = instance_gpu_count * num_instances
    if actual_gpus > gpu_count:
        note = f" (instance provides {actual_gpus} GPUs; {actual_gpus - gpu_count} unused)"

    return {
        "type": "cloud",
        "platform": platform,
        "provider": provider_name,
        "instance": instance_name,
        "instance_gpu_count": instance_gpu_count,
        "num_instances": num_instances,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "actual_gpu_count": actual_gpus,
        "region": best.get("region", ""),
        "on_demand_hourly_usd": round(on_demand_hr, 2),
        "spot_hourly_usd": round(spot_hr, 2),
        "reserved_1yr_hourly_usd": round(reserved_hr, 2),
        "monthly_on_demand_usd": round(on_demand_hr * 730, 0),
        "monthly_spot_usd": round(spot_hr * 730, 0),
        "monthly_reserved_usd": round(reserved_hr * 730, 0),
        "source_url": source_urls,
        "summary": (
            f"{provider_name} {num_instances}× {instance_name} ({instance_gpu_count}×{best['gpu']} each): "
            f"On-demand ${on_demand_hr:.2f}/hr, Spot ${spot_hr:.2f}/hr, "
            f"Reserved ${reserved_hr:.2f}/hr{note}"
        ),
    }


def _resolve_scaling_key(gpu_type: str) -> str:
    """Resolve a GPU type string to its scaling factor dictionary key."""
    _key_map = {
        "B300-288GB": "B300", "GB300-288GB": "GB300",
        "B200-192GB": "B200", "GB200-192GB": "GB200",
        "H200-141GB": "H200", "H100-80GB": "H100",
        "MI300X-192GB": "MI300X", "A100-80GB": "A100-80GB", "A100-40GB": "A100-40GB",
        "L40S-48GB": "L40S", "RTX-PRO-6000-96GB": "RTX-PRO-6000",
        "A10G-24GB": "A10G", "L4-24GB": "L4", "T4-16GB": "T4",
    }
    if gpu_type in _key_map:
        return _key_map[gpu_type]
    upper = gpu_type.upper()
    for canonical, key in _key_map.items():
        if upper == canonical.upper() or upper == key.upper():
            return key
    return gpu_type.split("-")[0]


def _calculate_onprem_tco(pricing, gpu_type: str, gpu_count: int) -> dict:
    """Calculate on-premises TCO using JSON reference data.

    Includes: hardware depreciation, power & cooling, colocation,
    staffing, and Red Hat AI subscription.
    """
    import json
    from pathlib import Path

    search_gpu = _resolve_scaling_key(gpu_type)

    tco_path = Path(__file__).parents[3] / "connectors" / "data" / "onprem_tco.json"
    try:
        with tco_path.open() as f:
            tco_data = json.load(f)
    except Exception:
        tco_data = {}

    ref_configs = tco_data.get("reference_configs", {})
    scaling = tco_data.get("cost_scaling_factors", {})

    # Find reference config or scale from H100 baseline
    ref = None
    for _key, cfg in ref_configs.items():
        if search_gpu.upper() in cfg.get("gpu_type", "").upper():
            ref = cfg
            break
    if not ref:
        ref = next(iter(ref_configs.values()), None)

    if ref:
        ref_gpu_count = ref["gpu_count"]
        scale = gpu_count / ref_gpu_count

        hw_ratio = scaling.get("hardware_cost_ratio", {}).get(search_gpu, 1.0)
        hw_monthly = ref["hardware"]["monthly_cost_usd"] * scale * hw_ratio

        kw_per_gpu = scaling.get("power_kw_per_gpu", {}).get(search_gpu, ref["power_and_cooling"]["kw_per_gpu"])
        pue = ref["power_and_cooling"]["pue_factor"]
        hours = ref["power_and_cooling"]["hours_per_month"]
        elec_rate = ref["power_and_cooling"]["electricity_rate_per_kwh_usd"]
        power_monthly = gpu_count * kw_per_gpu * pue * hours * elec_rate

        colo_monthly = ref["colocation"]["per_gpu_monthly_usd"] * gpu_count
        staff_monthly = ref["staffing"]["monthly_cost_usd"]

        rh_per_gpu_annual = ref["redhat_ai_subscription"]["list_price_per_gpu_annual_usd"]
        rh_monthly = (rh_per_gpu_annual * gpu_count) / 12

        monthly_total = hw_monthly + power_monthly + colo_monthly + staff_monthly + rh_monthly
    else:
        specs = pricing.get_gpu_specs(search_gpu)
        unit_price = (specs or {}).get("typical_street_price_usd", 25000)
        hw_monthly = (unit_price * gpu_count) / 36
        power_monthly = gpu_count * 1.0 * 1.35 * 720 * 0.07
        colo_monthly = gpu_count * 150
        staff_monthly = 6000
        rh_monthly = (2500 * gpu_count) / 12
        monthly_total = hw_monthly + power_monthly + colo_monthly + staff_monthly + rh_monthly

    return {
        "type": "on-premise",
        "platform": "on-premise",
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "monthly_hardware_usd": round(hw_monthly, 0),
        "monthly_power_usd": round(power_monthly, 0),
        "monthly_colocation_usd": round(colo_monthly, 0),
        "monthly_staffing_usd": round(staff_monthly, 0),
        "monthly_rh_subscription_usd": round(rh_monthly, 0),
        "monthly_total_usd": round(monthly_total, 0),
        "source_url": "https://lenovopress.lenovo.com/lp2368-on-premise-vs-cloud-generative-ai-total-cost-of-ownership-2026-edition",
        "summary": (
            f"{gpu_count}× {gpu_type}: "
            f"HW ${hw_monthly:,.0f} + Power ${power_monthly:,.0f} + "
            f"Colo ${colo_monthly:,.0f} + Staff ${staff_monthly:,.0f} + "
            f"RH AI ${rh_monthly:,.0f} = ${monthly_total:,.0f}/mo"
        ),
    }


_PLATFORM_PROVIDER_MAP = {
    "aws": ["aws"],
    "azure": ["microsoftazure"],
    "gcp": ["googlecloud"],
}

_PRICING_URLS = {
    "aws": "https://aws.amazon.com/ec2/pricing/on-demand/",
    "azure": "https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/",
    "gcp": "https://cloud.google.com/compute/gpus-pricing",
}
