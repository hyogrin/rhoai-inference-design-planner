"""KV-cache / state memory estimator. Placeholder for Phase 4."""



def calculate_kv_bytes_per_token(
    attention_layer_count: int,
    num_kv_heads: int,
    head_dim: int,
    kv_dtype_bytes: int = 2,
) -> dict:
    """Calculate KV bytes per token for standard MHA/GQA/MQA attention.

    Formula: 2 * attention_layer_count * num_kv_heads * head_dim * kv_dtype_bytes
    """
    raise NotImplementedError("Phase 4 implementation")
