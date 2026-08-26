"""Test configuration and shared fixtures."""

import pytest


@pytest.fixture
def sample_model_identity_data():
    """Sample ModelIdentity data for testing."""
    return {
        "repo_id": "meta-llama/Llama-3.1-70B-Instruct",
        "revision": "main",
        "resolved_commit_sha": "a82769e3b40e78c5e3ae37d7a9a07c58af260eea",
        "gated": True,
        "private": False,
        "license": "llama3.1",
        "pipeline_tag": "text-generation",
        "tasks": ["text-generation", "conversational"],
        "source_url": "https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct",
        "fetched_at": "2026-08-26T10:00:00Z",
    }


@pytest.fixture
def sample_hardware_pool_data():
    """Sample HardwarePool data for testing."""
    return {
        "pool_id": "h100-pool-1",
        "accelerator_vendor": "NVIDIA",
        "accelerator_model": "H100",
        "accelerator_count_per_node": 8,
        "node_count": 2,
        "hbm_gb_per_accelerator": 80.0,
        "mig_enabled": False,
        "intra_node_interconnect": "nvswitch",
        "inter_node_network": "infiniband",
        "inter_node_bandwidth_gbps": 400.0,
        "rdma_enabled": True,
        "cpu_model": "Intel Xeon Platinum 8480+",
        "cpu_count": 2,
        "ram_gb_per_node": 2048.0,
        "local_nvme_gb_per_node": 7680.0,
        "cloud_instance_type": None,
        "availability_zone": None,
        "quantity_available": 2,
        "hourly_price_override": None,
    }


@pytest.fixture
def sample_workload_profile_data():
    """Sample WorkloadProfile data for testing."""
    return {
        "use_case_type": "rag",
        "modality": "text",
        "streaming_enabled": True,
        "traffic_pattern": "bursty",
        "sustained_rps": 10.0,
        "peak_rps": 50.0,
        "burst_duration_seconds": 30,
        "target_concurrency": 20,
        "availability_target": 0.999,
        "minimum_replicas": 2,
        "maximum_replicas": 8,
        "latency_slo": {
            "ttft_p95_ms": 500.0,
            "tpot_p95_ms": 50.0,
            "e2e_p95_ms": None,
        },
        "isl_distribution": {"p50": 512, "p80": 1024, "p95": 2048, "max": 4096},
        "osl_distribution": {"p50": 256, "p80": 512, "p95": 1024, "max": 2048},
        "common_prefix_ratio": 0.3,
        "common_prefix_tokens_p50": 200,
        "session_turns_p50": 3,
        "tool_calls_per_request_p50": None,
        "images_per_request_p95": None,
        "quantization": {
            "weight_formats_allowed": ["BF16", "FP8"],
            "kv_cache_formats_allowed": ["BF16", "FP8"],
        },
        "quality_loss_tolerance": 0.02,
        "scale_to_zero_allowed": False,
        "data_residency_constraints": [],
    }
