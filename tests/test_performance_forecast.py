"""Tests for performance forecast calculation correctness.

Validates the roofline model against manually computed expected values
for known GPU/model configurations.
"""

import math

import pytest

from agents.inference_planner.nodes.sizing import (
    GPU_SPECS,
    calculate_performance_forecast,
)


# ---------------------------------------------------------------------------
# Golden test cases — manually verified expected values
# ---------------------------------------------------------------------------

GOLDEN_CASES = [
    {
        "name": "Llama-3.1-70B FP16 on 2×H100-80GB",
        "state": {
            "model_repo_id": "meta-llama/Llama-3.1-70B-Instruct",
            "model_architecture": {
                "parameter_count_total": 70_000_000_000,
                "num_hidden_layers": 80,
                "hidden_size": 8192,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
            },
            "workload_profile": {
                "gpu_type": "H100-80GB",
                "gpu_count": 2,
                "max_concurrent_requests": 32,
                "max_sequence_length": 4096,
                "tpot_ms": 30,
            },
            "hardware_inventory": {},
            "recommendation": {
                "memory_estimate": {"precision_bits": 16},
            },
            "evidence_items": [],
        },
        "expected": {
            "params_b": 70.0,
            "precision_bits": 16,
            "model_size_bytes": 70e9 * 2,  # FP16 = 2 bytes/param
            "total_bandwidth_bytes": 3.35e12 * 2,  # 2× H100
            "total_flops": 989e12 * 2,
            # mem_ceiling = bandwidth / model_size
            "theoretical_decode_tps": (3.35e12 * 2) / (70e9 * 2),
            # compute_ceiling = flops / (2 * params)
            "compute_ceiling_tps": (989e12 * 2) / (2 * 70e9),
        },
    },
    {
        "name": "Gemma-4-31B FP8 on 1×H100-80GB",
        "state": {
            "model_repo_id": "google/gemma-4-31B-it-fp8",
            "model_architecture": {
                "parameter_count_total": 31_000_000_000,
                "num_hidden_layers": 48,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
            },
            "workload_profile": {
                "gpu_type": "H100-80GB",
                "gpu_count": 1,
                "max_concurrent_requests": 32,
                "max_sequence_length": 4096,
                "tpot_ms": 30,
            },
            "hardware_inventory": {},
            "recommendation": {
                "memory_estimate": {"precision_bits": 8},
            },
            "evidence_items": [],
        },
        "expected": {
            "params_b": 31.0,
            "precision_bits": 8,
            "model_size_bytes": 31e9 * 1,  # FP8 = 1 byte/param
            "total_bandwidth_bytes": 3.35e12,
            "total_flops": 989e12,
            "theoretical_decode_tps": 3.35e12 / (31e9 * 1),
            "compute_ceiling_tps": 989e12 / (2 * 31e9),
        },
    },
    {
        "name": "Llama-3.1-8B FP16 on 1×A100-80GB",
        "state": {
            "model_repo_id": "meta-llama/Llama-3.1-8B-Instruct",
            "model_architecture": {
                "parameter_count_total": 8_000_000_000,
                "num_hidden_layers": 32,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
            },
            "workload_profile": {
                "gpu_type": "A100-80GB",
                "gpu_count": 1,
                "max_concurrent_requests": 16,
                "max_sequence_length": 4096,
                "tpot_ms": 50,
            },
            "hardware_inventory": {},
            "recommendation": {
                "memory_estimate": {"precision_bits": 16},
            },
            "evidence_items": [],
        },
        "expected": {
            "params_b": 8.0,
            "precision_bits": 16,
            "model_size_bytes": 8e9 * 2,
            "total_bandwidth_bytes": 2.0e12,
            "total_flops": 312e12,
            "theoretical_decode_tps": 2.0e12 / (8e9 * 2),
            "compute_ceiling_tps": 312e12 / (2 * 8e9),
        },
    },
    {
        "name": "Qwen-72B INT4 on 4×H200-141GB",
        "state": {
            "model_repo_id": "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
            "model_architecture": {
                "parameter_count_total": 72_000_000_000,
                "num_hidden_layers": 80,
                "hidden_size": 8192,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
            },
            "workload_profile": {
                "gpu_type": "H200-141GB",
                "gpu_count": 4,
                "max_concurrent_requests": 64,
                "max_sequence_length": 8192,
                "tpot_ms": 20,
            },
            "hardware_inventory": {},
            "recommendation": {
                "memory_estimate": {"precision_bits": 4},
            },
            "evidence_items": [],
        },
        "expected": {
            "params_b": 72.0,
            "precision_bits": 4,
            "model_size_bytes": 72e9 * 0.5,  # INT4 = 0.5 bytes/param
            "total_bandwidth_bytes": 4.8e12 * 4,
            "total_flops": 989e12 * 4,
            "theoretical_decode_tps": (4.8e12 * 4) / (72e9 * 0.5),
            "compute_ceiling_tps": (989e12 * 4) / (2 * 72e9),
        },
    },
    {
        "name": "Mixtral-8x22B MoE FP16 on 4×H100-80GB (active params)",
        "state": {
            "model_repo_id": "mistralai/Mixtral-8x22B-Instruct-v0.1",
            "model_architecture": {
                "parameter_count_total": 141_000_000_000,
                "parameter_count_active": 39_000_000_000,
                "num_hidden_layers": 56,
                "hidden_size": 6144,
                "num_attention_heads": 48,
                "num_key_value_heads": 8,
            },
            "workload_profile": {
                "gpu_type": "H100-80GB",
                "gpu_count": 4,
                "max_concurrent_requests": 32,
                "max_sequence_length": 4096,
                "tpot_ms": 30,
            },
            "hardware_inventory": {},
            "recommendation": {
                "memory_estimate": {"precision_bits": 16},
            },
            "evidence_items": [],
        },
        "expected": {
            "params_b": 141.0,
            "active_params_b": 39.0,
            "precision_bits": 16,
            "model_size_bytes": 141e9 * 2,
            "total_bandwidth_bytes": 3.35e12 * 4,
            "total_flops": 989e12 * 4,
            # mem_ceiling uses total params (all weights in HBM)
            "theoretical_decode_tps": (3.35e12 * 4) / (141e9 * 2),
            # compute_ceiling uses active params
            "compute_ceiling_tps": (989e12 * 4) / (2 * 39e9),
        },
    },
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[c["name"] for c in GOLDEN_CASES])
@pytest.mark.asyncio
async def test_performance_forecast_values(case):
    """Verify roofline model outputs match expected golden values."""
    result = await calculate_performance_forecast(case["state"])
    forecast = result["recommendation"]["performance_forecast"]

    expected = case["expected"]

    # Decode throughput (memory-bandwidth ceiling)
    expected_decode_tps = expected["theoretical_decode_tps"]
    assert forecast["theoretical_decode_tps"] == pytest.approx(
        expected_decode_tps, rel=0.01
    ), f"Decode TPS mismatch: got {forecast['theoretical_decode_tps']}, expected {expected_decode_tps:.1f}"

    # Compute ceiling
    expected_compute_ceil = expected["compute_ceiling_tps"]
    assert forecast["compute_ceiling_tps"] == pytest.approx(
        expected_compute_ceil, rel=0.01
    ), f"Compute ceiling mismatch: got {forecast['compute_ceiling_tps']}, expected {expected_compute_ceil:.1f}"

    # Ridge batch (crossover point)
    expected_ridge = expected_compute_ceil / expected_decode_tps
    assert forecast["ridge_batch_size"] == pytest.approx(
        expected_ridge, rel=0.01
    ), f"Ridge batch mismatch: got {forecast['ridge_batch_size']}, expected {expected_ridge:.1f}"

    # TPOT (single request decode latency)
    expected_tpot = 1000.0 / expected_decode_tps
    assert forecast["estimated_tpot_ms"] == pytest.approx(
        expected_tpot, rel=0.01
    ), f"TPOT mismatch: got {forecast['estimated_tpot_ms']}, expected {expected_tpot:.2f}"

    # TTFT (prefill latency at 512 tokens) - uses active params for MoE
    total_flops = expected["total_flops"]
    active_params_b = expected.get("active_params_b", expected["params_b"])
    expected_ttft = (2 * active_params_b * 1e9 * 512) / total_flops * 1000
    assert forecast["estimated_ttft_ms"] == pytest.approx(
        expected_ttft, rel=0.01
    ), f"TTFT mismatch: got {forecast['estimated_ttft_ms']}, expected {expected_ttft:.1f}"


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[c["name"] for c in GOLDEN_CASES])
@pytest.mark.asyncio
async def test_performance_forecast_chart_roofline(case):
    """Verify chart data follows roofline model: throughput = min(batch * mem_ceil, compute_ceil)."""
    result = await calculate_performance_forecast(case["state"])
    forecast = result["recommendation"]["performance_forecast"]
    chart_data = forecast["chart_data"]

    expected = case["expected"]
    mem_ceil = expected["theoretical_decode_tps"]
    compute_ceil = expected["compute_ceiling_tps"]

    for point in chart_data:
        batch = point["batch_size"]
        expected_throughput = min(batch * mem_ceil, compute_ceil)
        assert point["throughput_tokens_per_sec"] == pytest.approx(
            expected_throughput, rel=0.01
        ), f"Batch {batch}: got {point['throughput_tokens_per_sec']}, expected {expected_throughput:.1f}"


@pytest.mark.asyncio
async def test_performance_forecast_max_batch_at_target():
    """Verify max_batch_at_target_tpot is correctly computed."""
    # Use Llama-70B on 2×H100 with target TPOT = 30ms
    case = GOLDEN_CASES[0]
    result = await calculate_performance_forecast(case["state"])
    forecast = result["recommendation"]["performance_forecast"]

    expected = case["expected"]
    mem_ceil = expected["theoretical_decode_tps"]
    compute_ceil = expected["compute_ceiling_tps"]
    ridge = compute_ceil / mem_ceil
    tpot_single = 1000.0 / mem_ceil
    target_tpot = case["state"]["workload_profile"]["tpot_ms"]

    if tpot_single <= target_tpot:
        expected_max_batch = math.floor(ridge * (target_tpot / tpot_single))
    else:
        expected_max_batch = 1
    expected_max_batch = max(expected_max_batch, 1)

    assert forecast["max_batch_at_target_tpot"] == expected_max_batch


@pytest.mark.asyncio
async def test_performance_forecast_missing_params():
    """When parameter count cannot be determined, forecast returns error."""
    state = {
        "model_repo_id": "unknown/model-with-no-size-hint",
        "model_architecture": {},
        "workload_profile": {"gpu_type": "H100-80GB", "gpu_count": 1},
        "hardware_inventory": {},
        "recommendation": {},
        "evidence_items": [],
    }
    result = await calculate_performance_forecast(state)
    forecast = result["recommendation"]["performance_forecast"]
    assert "error" in forecast


@pytest.mark.asyncio
async def test_performance_forecast_unknown_gpu():
    """When GPU type is not in specs, forecast returns error."""
    state = {
        "model_repo_id": "meta-llama/Llama-3.1-70B-Instruct",
        "model_architecture": {"parameter_count_total": 70_000_000_000},
        "workload_profile": {"gpu_type": "UNKNOWN-GPU-999", "gpu_count": 1},
        "hardware_inventory": {},
        "recommendation": {},
        "evidence_items": [],
    }
    result = await calculate_performance_forecast(state)
    forecast = result["recommendation"]["performance_forecast"]
    assert "error" in forecast


@pytest.mark.parametrize("gpu_type", list(GPU_SPECS.keys()))
@pytest.mark.asyncio
async def test_performance_forecast_all_gpus(gpu_type):
    """Ensure performance forecast runs without error for all supported GPUs."""
    state = {
        "model_repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "model_architecture": {"parameter_count_total": 8_000_000_000},
        "workload_profile": {"gpu_type": gpu_type, "gpu_count": 1, "tpot_ms": 50},
        "hardware_inventory": {},
        "recommendation": {"memory_estimate": {"precision_bits": 16}},
        "evidence_items": [],
    }
    result = await calculate_performance_forecast(state)
    forecast = result["recommendation"]["performance_forecast"]
    assert "error" not in forecast
    assert forecast["theoretical_decode_tps"] > 0
    assert forecast["compute_ceiling_tps"] > 0
    assert forecast["ridge_batch_size"] > 0
    assert forecast["estimated_tpot_ms"] > 0
    assert forecast["estimated_ttft_ms"] > 0
    assert len(forecast["chart_data"]) == 8
