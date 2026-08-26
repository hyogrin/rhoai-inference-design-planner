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
    "B200-192GB": {"memory_gb": 192, "bandwidth_tbps": 8.0, "flops_tflops": 2250},
    "H200-141GB": {"memory_gb": 141, "bandwidth_tbps": 4.8, "flops_tflops": 989},
    "H100-80GB": {"memory_gb": 80, "bandwidth_tbps": 3.35, "flops_tflops": 989},
    "MI300X-192GB": {"memory_gb": 192, "bandwidth_tbps": 5.3, "flops_tflops": 1307},
    "A100-80GB": {"memory_gb": 80, "bandwidth_tbps": 2.0, "flops_tflops": 312},
    "A100-40GB": {"memory_gb": 40, "bandwidth_tbps": 1.55, "flops_tflops": 312},
    "L40S-48GB": {"memory_gb": 48, "bandwidth_tbps": 0.864, "flops_tflops": 366},
    "A10G-24GB": {"memory_gb": 24, "bandwidth_tbps": 0.6, "flops_tflops": 70},
    "L4-24GB": {"memory_gb": 24, "bandwidth_tbps": 0.3, "flops_tflops": 121},
    "T4-16GB": {"memory_gb": 16, "bandwidth_tbps": 0.3, "flops_tflops": 65},
}

GPU_HOURLY_COST: dict[str, float] = {
    "B200-192GB": 12.0,
    "H200-141GB": 8.5,
    "H100-80GB": 5.5,
    "MI300X-192GB": 5.0,
    "A100-80GB": 3.5,
    "A100-40GB": 2.5,
    "L40S-48GB": 2.0,
    "A10G-24GB": 1.2,
    "L4-24GB": 0.8,
    "T4-16GB": 0.5,
}


def _get_param_count(state: PlannerState) -> float | None:
    """Extract parameter count (in billions) from architecture or model name."""
    import re

    arch = state.get("model_architecture") or {}
    params = arch.get("parameters")
    if params and isinstance(params, (int, float)):
        if params > 1e6:
            return params / 1e9
        return params

    # Fallback 1: extract from model name (e.g., "Qwen3.5-35B-A3B" -> 35B)
    repo_id = state.get("model_repo_id", "")
    name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    match = re.search(r"(\d+\.?\d*)B", name)
    if match:
        val = float(match.group(1))
        if val > 0:
            return val

    # Fallback 2: stored from LLM estimation (set during async path)
    llm_estimate = (state.get("model_architecture") or {}).get("estimated_parameters_b")
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


def _estimate_model_memory_gb(params_b: float, precision_bits: int = 16) -> float:
    """Estimate model weight memory in GB."""
    bytes_per_param = precision_bits / 8
    return params_b * 1e9 * bytes_per_param / (1024**3)


async def calculate_memory_capacity(state: PlannerState) -> dict:
    """Calculate memory requirements and capacity analysis."""
    workload = state.get("workload_profile") or {}
    gpu_type = workload.get("gpu_type") or ""
    gpu_count = workload.get("gpu_count", 1)

    params_b = _get_param_count(state)

    # LLM fallback if regex and metadata both failed
    if params_b is None:
        repo_id = state.get("model_repo_id", "")
        params_b = await _estimate_params_with_llm(repo_id)
    gpu_spec = GPU_SPECS.get(gpu_type, {})
    per_gpu_mem = gpu_spec.get("memory_gb", 80)
    total_gpu_mem = per_gpu_mem * gpu_count

    # Determine precision from evidence
    precision_bits = 16
    evidence_items = state.get("evidence_items", [])
    for ev in evidence_items:
        summary = (ev.get("summary") or "").lower()
        if "fp8" in summary or "int8" in summary:
            precision_bits = 8
            break
        if "int4" in summary or "w4a16" in summary:
            precision_bits = 4
            break

    model_mem = _estimate_model_memory_gb(params_b, precision_bits) if params_b else None

    # KV cache estimate
    max_concurrent = workload.get("max_concurrent_requests", 32)
    arch = state.get("model_architecture") or {}
    num_layers = arch.get("num_hidden_layers", 32)
    hidden_size = arch.get("hidden_size", 4096)
    num_kv_heads = arch.get("num_key_value_heads", 8)
    head_dim = hidden_size // max(arch.get("num_attention_heads", 32), 1)
    seq_len = 4096

    kv_per_token_bytes = 2 * num_layers * num_kv_heads * head_dim * 2
    kv_cache_gb = (max_concurrent * seq_len * kv_per_token_bytes) / (1024**3)

    overhead_gb = 2.0
    total_required = (model_mem or 0) + kv_cache_gb + overhead_gb
    utilization = (total_required / total_gpu_mem * 100) if total_gpu_mem > 0 else 0
    fits = total_required <= total_gpu_mem * 0.95

    memory_estimate = {
        "model_weights_gb": round(model_mem, 1) if model_mem else None,
        "kv_cache_gb": round(kv_cache_gb, 1),
        "overhead_gb": overhead_gb,
        "total_required_gb": round(total_required, 1),
        "total_available_gb": total_gpu_mem,
        "utilization_pct": round(utilization, 1),
        "fits": fits,
        "precision_bits": precision_bits,
        "parameters_billions": round(params_b, 1) if params_b else None,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
    }

    logger.info("Memory estimate: %.1f GB required / %d GB available (%.1f%%)",
                total_required, total_gpu_mem, utilization)

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
    gpu_type = workload.get("gpu_type") or ""
    gpu_count = workload.get("gpu_count", 1)
    params_b = _get_param_count(state)
    if params_b is None:
        params_b = await _estimate_params_with_llm(state.get("model_repo_id", ""))
    params_b = params_b or 7.0

    gpu_spec = GPU_SPECS.get(gpu_type, {})
    bandwidth_tbps = gpu_spec.get("bandwidth_tbps", 2.0)
    flops_tflops = gpu_spec.get("flops_tflops", 300)

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

    # --- Decode Roofline ---
    # Memory-bandwidth ceiling: max tokens/s for batch=1
    # Reading all weights once per token → 1 token per weight-read
    mem_ceiling_tps = total_bandwidth_bytes / model_size_bytes if model_size_bytes > 0 else 1

    # Compute ceiling: max tokens/s regardless of batch
    # Each token requires ~2*params FLOPs (matmul forward pass)
    flops_per_token = 2 * params_b * 1e9
    compute_ceiling_tps = total_flops / flops_per_token if flops_per_token > 0 else 1e6

    # Ridge point: batch size where memory ceiling * batch = compute ceiling
    ridge_batch = compute_ceiling_tps / mem_ceiling_tps if mem_ceiling_tps > 0 else 1

    # Single-request metrics
    tpot_single_ms = (1000.0 / mem_ceiling_tps) if mem_ceiling_tps > 0 else 999
    ttft_input_tokens = 512
    prefill_flops = 2 * params_b * 1e9 * ttft_input_tokens
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
        "max_batch_at_target_tpot": max_batch_at_target,
        "chart_data": chart_data,
        "explanation": {
            "model": f"{params_b:.1f}B params @ FP{precision_bits}",
            "hardware": f"{gpu_count}× {gpu_type}",
            "bandwidth": f"{total_bandwidth_bytes / 1e12:.1f} TB/s total",
            "compute": f"{total_flops / 1e12:.0f} TFLOPS total",
            "memory_bound": f"Batch 1→{math.floor(ridge_batch)}: throughput scales linearly",
            "compute_bound": f"Batch >{math.floor(ridge_batch)}: saturates at {compute_ceiling_tps:.0f} tok/s",
        },
    }

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
        "B200-192GB": "B200", "H200-141GB": "H200", "H100-80GB": "H100",
        "MI300X-192GB": "MI300X", "A100-80GB": "A100-80GB", "A100-40GB": "A100",
        "L40S-48GB": "L40S", "A10G-24GB": "A10G", "L4-24GB": "L4", "T4-16GB": "T4",
    }
    search_gpu = gpu_name_map.get(gpu_type, gpu_type.split("-")[0])

    instances = pricing.get_cloud_instances_for_gpu(search_gpu, min_gpu_count=gpu_count)
    platform_instances = [i for i in instances if i["provider"].lower().replace(" ", "") in _PLATFORM_PROVIDER_MAP.get(platform, [])]

    if not platform_instances:
        platform_instances = instances

    if not platform_instances:
        # Fallback to generic estimate
        hourly = GPU_HOURLY_COST.get(gpu_type, 3.0) * gpu_count
        return {
            "type": "cloud",
            "platform": platform,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "on_demand_hourly_usd": round(hourly, 2),
            "monthly_on_demand_usd": round(hourly * 730, 0),
            "source_url": None,
            "summary": f"${hourly:.2f}/hr (estimated, no exact instance match)",
        }

    # Use the best matching instance (closest GPU count)
    best = min(platform_instances, key=lambda x: abs(x["gpu_count"] - gpu_count))
    scale = gpu_count / best["gpu_count"]

    on_demand_hr = best["on_demand_hourly"] * scale
    spot_hr = best.get("spot_hourly", 0) * scale
    reserved_hr = best.get("reserved_1yr_hourly", 0) * scale

    provider_name = best["provider"]
    instance_name = best["instance_name"]

    source_urls = _PRICING_URLS.get(platform, "")

    return {
        "type": "cloud",
        "platform": platform,
        "provider": provider_name,
        "instance": instance_name,
        "instance_gpu_count": best["gpu_count"],
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "region": best.get("region", ""),
        "on_demand_hourly_usd": round(on_demand_hr, 2),
        "spot_hourly_usd": round(spot_hr, 2),
        "reserved_1yr_hourly_usd": round(reserved_hr, 2),
        "monthly_on_demand_usd": round(on_demand_hr * 730, 0),
        "monthly_spot_usd": round(spot_hr * 730, 0),
        "monthly_reserved_usd": round(reserved_hr * 730, 0),
        "source_url": source_urls,
        "summary": (
            f"{provider_name} {instance_name} ({best['gpu_count']}×{best['gpu']}): "
            f"On-demand ${on_demand_hr:.2f}/hr, Spot ${spot_hr:.2f}/hr, "
            f"Reserved ${reserved_hr:.2f}/hr"
        ),
    }


def _calculate_onprem_tco(pricing, gpu_type: str, gpu_count: int) -> dict:
    """Calculate on-premises TCO (3-year depreciation + power)."""
    gpu_name_map = {
        "B200-192GB": "B200", "H200-141GB": "H200", "H100-80GB": "H100",
        "MI300X-192GB": "MI300X", "A100-80GB": "A100-80GB", "A100-40GB": "A100",
        "L40S-48GB": "L40S", "A10G-24GB": "A10G", "L4-24GB": "L4", "T4-16GB": "T4",
    }
    search_gpu = gpu_name_map.get(gpu_type, gpu_type.split("-")[0])
    specs = pricing.get_gpu_specs(search_gpu)

    if not specs:
        # Fallback
        unit_price = 25000
        tdp = 400
    else:
        unit_price = specs.get("typical_street_price_usd", specs.get("list_price_usd", 25000))
        tdp = specs.get("tdp_watts", 400)

    # TCO calculation
    hardware_cost = unit_price * gpu_count
    depreciation_years = 3
    monthly_depreciation = hardware_cost / (depreciation_years * 12)

    # Power cost (GPU + ~30% overhead for cooling/networking)
    power_kw = (tdp * gpu_count * 1.3) / 1000
    electricity_per_kwh = 0.10  # US average
    monthly_power = power_kw * 24 * 30 * electricity_per_kwh

    monthly_total = monthly_depreciation + monthly_power

    return {
        "type": "on-premise",
        "platform": "on-premise",
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "gpu_unit_price_usd": unit_price,
        "hardware_total_usd": hardware_cost,
        "depreciation_years": depreciation_years,
        "monthly_depreciation_usd": round(monthly_depreciation, 0),
        "power_kw": round(power_kw, 2),
        "monthly_power_usd": round(monthly_power, 0),
        "monthly_total_usd": round(monthly_total, 0),
        "source_url": "https://www.nvidia.com/en-us/data-center/",
        "summary": (
            f"{gpu_count}× {gpu_type} @ ${unit_price:,}/ea: "
            f"${monthly_depreciation:.0f}/mo depreciation + ${monthly_power:.0f}/mo power"
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
