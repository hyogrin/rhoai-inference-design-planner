"""Weight memory estimator. Placeholder for Phase 4."""



def calculate_weight_memory_bytes(
    parameter_count_by_dtype: dict[str, int] | None = None,
    parameter_count_total: int | None = None,
    weight_precision: str | None = None,
) -> dict:
    """Calculate total weight memory in bytes.

    Preferred: sum(parameter_count_by_dtype[dtype] * bytes_per_parameter[dtype])
    Fallback: parameter_count_total * effective_bytes_per_parameter
    """
    raise NotImplementedError("Phase 4 implementation")
