"""Usable HBM and concurrent-token capacity estimator. Placeholder for Phase 4."""



def calculate_capacity(
    gross_hbm_gb: float,
    gpu_count: int,
    weight_memory_gb: float,
    gpu_memory_utilization: float = 0.90,
    runtime_reserve_gb: float = 0.5,
    graph_reserve_gb: float = 0.0,
    safety_margin_gb: float = 0.5,
    kv_bytes_per_token: float = 0.0,
) -> dict:
    """Calculate usable HBM and max resident tokens."""
    raise NotImplementedError("Phase 4 implementation")
