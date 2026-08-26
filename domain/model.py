"""Model identity and architecture domain models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelIdentity(BaseModel):
    """Resolved identity of a Hugging Face model."""

    model_config = ConfigDict(extra="forbid")

    repo_id: str
    revision: str
    resolved_commit_sha: str | None = None
    gated: bool
    private: bool
    license: str | None = None
    pipeline_tag: str | None = None
    tasks: list[str] = Field(default_factory=list)
    source_url: str
    fetched_at: datetime


class ModelArchitecture(BaseModel):
    """Parsed architecture details extracted from model config files."""

    model_config = ConfigDict(extra="forbid")

    architecture_names: list[str] = Field(default_factory=list)
    family: str | None = None
    architecture_type: Literal["dense", "moe", "hybrid", "multimodal", "unknown"] = "unknown"

    parameter_count_total: int | None = None
    parameter_count_active: int | None = None
    parameter_count_by_dtype: dict[str, int] | None = None
    weight_format: str | None = None
    weight_precision: str | None = None
    quantization_method: str | None = None

    num_hidden_layers: int | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_attention_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    max_position_embeddings: int | None = None
    sliding_window: int | None = None

    attention_layer_count: int | None = None
    linear_attention_layer_count: int | None = None
    state_space_layer_count: int | None = None

    num_experts_total: int | None = None
    num_experts_active: int | None = None
    expert_intermediate_size: int | None = None

    vision_encoder_parameters: int | None = None

    kv_layout: Literal["mha_gqa", "mla", "hybrid", "unknown"] = "unknown"
    kv_bytes_per_token_formula: str | None = None

    parser_confidence: float = Field(ge=0.0, le=1.0)
    missing_fields: list[str] = Field(default_factory=list)
    raw_config_paths: list[str] = Field(default_factory=list)
