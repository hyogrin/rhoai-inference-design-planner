"""Workload profile domain models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DistributionPercentiles(BaseModel):
    """Token-length distribution percentiles."""

    model_config = ConfigDict(extra="forbid")

    p50: int = Field(ge=0)
    p80: int = Field(ge=0)
    p95: int = Field(ge=0)
    max: int = Field(ge=0)


class LatencySLO(BaseModel):
    """Latency service-level objectives."""

    model_config = ConfigDict(extra="forbid")

    ttft_p95_ms: float | None = None
    tpot_p95_ms: float | None = None
    e2e_p95_ms: float | None = None


class QuantizationPreference(BaseModel):
    """Acceptable quantization formats for weights and KV cache."""

    model_config = ConfigDict(extra="forbid")

    weight_formats_allowed: list[str] = Field(default_factory=list)
    kv_cache_formats_allowed: list[str] = Field(default_factory=list)


class WorkloadProfile(BaseModel):
    """Describes the expected inference workload characteristics."""

    model_config = ConfigDict(extra="forbid")

    use_case_type: Literal["chatbot", "rag", "agentic", "batch", "mixed"]
    modality: Literal["text", "vision", "audio", "multimodal"] = "text"
    streaming_enabled: bool = True
    traffic_pattern: Literal["steady", "bursty", "scheduled", "unknown"] = "unknown"

    sustained_rps: float | None = None
    peak_rps: float | None = None
    burst_duration_seconds: int | None = None
    target_concurrency: int | None = None

    availability_target: float | None = None
    minimum_replicas: int = Field(default=1, ge=0)
    maximum_replicas: int | None = None

    latency_slo: LatencySLO = Field(default_factory=LatencySLO)

    isl_distribution: DistributionPercentiles | None = None
    osl_distribution: DistributionPercentiles | None = None
    common_prefix_ratio: float | None = None
    common_prefix_tokens_p50: int | None = None
    session_turns_p50: int | None = None
    tool_calls_per_request_p50: int | None = None
    images_per_request_p95: int | None = None

    quantization: QuantizationPreference = Field(default_factory=QuantizationPreference)
    quality_loss_tolerance: float | None = None
    scale_to_zero_allowed: bool = False
    data_residency_constraints: list[str] = Field(default_factory=list)
