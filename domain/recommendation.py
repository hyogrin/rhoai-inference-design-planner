"""Inference design recommendation domain models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from domain.model import ModelIdentity

# ---------------------------------------------------------------------------
# Nested sub-structures
# ---------------------------------------------------------------------------


class DeploymentTopology(BaseModel):
    """How the model is distributed across GPUs / nodes."""

    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "single_gpu",
        "single_node_tp",
        "single_node_tp_ep",
        "multi_node",
        "kserve_vllm",
        "llmd",
    ]
    description: str
    node_count: int = Field(ge=1)
    gpus_per_node: int = Field(ge=1)
    total_gpus: int = Field(ge=1)


class ParallelismConfig(BaseModel):
    """Parallelism knobs for vLLM / serving runtime."""

    model_config = ConfigDict(extra="forbid")

    tensor_parallel_size: int = Field(default=1, ge=1)
    data_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    enable_expert_parallel: bool = False


class ReplicaPlan(BaseModel):
    """Horizontal scaling plan for inference replicas."""

    model_config = ConfigDict(extra="forbid")

    min_replicas: int = Field(ge=0)
    max_replicas: int = Field(ge=1)
    target_replicas: int = Field(ge=0)
    scaling_metric: str
    scale_to_zero: bool = False


class VllmConfiguration(BaseModel):
    """vLLM engine arguments as a flat key-value map."""

    model_config = ConfigDict(extra="forbid")

    args: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class PerGpuBreakdown(BaseModel):
    """Memory breakdown for a single GPU."""

    model_config = ConfigDict(extra="forbid")

    gpu_index: int = Field(ge=0)
    weight_memory_gb: float
    kv_cache_budget_gb: float
    runtime_reserve_gb: float
    total_used_gb: float
    total_available_gb: float


class MemoryCapacityEstimate(BaseModel):
    """Estimated GPU memory budget and capacity."""

    model_config = ConfigDict(extra="forbid")

    weight_memory_gb: float
    kv_cache_budget_gb: float
    max_resident_tokens: int | None = None
    max_concurrent_sequences: int | None = None
    gpu_memory_utilization: float = Field(ge=0.0, le=1.0)
    runtime_reserve_gb: float
    safety_margin_gb: float
    per_gpu_breakdown: list[PerGpuBreakdown] = Field(default_factory=list)


class RangeValue(BaseModel):
    """A numeric value expressed as a range (low / mid / high)."""

    model_config = ConfigDict(extra="forbid")

    low: float
    mid: float
    high: float


class PerformanceForecast(BaseModel):
    """Projected serving performance under stated workload."""

    model_config = ConfigDict(extra="forbid")

    estimated_throughput_rps: RangeValue | None = None
    estimated_ttft_ms: RangeValue | None = None
    estimated_tpot_ms: RangeValue | None = None
    estimated_e2e_latency_ms: RangeValue | None = None
    estimated_max_concurrency: RangeValue | None = None
    basis: str | None = None


class CostEstimate(BaseModel):
    """Monthly cost projection for the recommended deployment."""

    model_config = ConfigDict(extra="forbid")

    gpu_monthly_cost: float | None = None
    compute_monthly_cost: float | None = None
    network_monthly_cost: float | None = None
    storage_monthly_cost: float | None = None
    total_monthly_cost: float | None = None
    currency: str = "USD"
    cost_per_million_tokens: float | None = None
    assumptions: list[str] = Field(default_factory=list)


class AutoscalingRecommendation(BaseModel):
    """Recommended autoscaling configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    metric_name: str
    target_value: float
    scale_up_delay_seconds: int | None = None
    scale_down_delay_seconds: int | None = None
    min_replicas: int = Field(ge=0)
    max_replicas: int = Field(ge=1)


class RoutingAndCacheRecommendation(BaseModel):
    """Recommendations for request routing and prefix caching."""

    model_config = ConfigDict(extra="forbid")

    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = False
    routing_strategy: str | None = None
    notes: list[str] = Field(default_factory=list)


class Assumption(BaseModel):
    """An assumption underlying the recommendation."""

    model_config = ConfigDict(extra="forbid")

    assumption_id: str
    description: str
    impact_if_wrong: str | None = None
    evidence_ids: list[UUID] = Field(default_factory=list)


class Risk(BaseModel):
    """A risk associated with the recommendation."""

    model_config = ConfigDict(extra="forbid")

    risk_id: str
    severity: Literal["low", "medium", "high", "critical"]
    description: str
    mitigation: str | None = None
    evidence_ids: list[UUID] = Field(default_factory=list)


class Alternative(BaseModel):
    """An alternative design option that was considered."""

    model_config = ConfigDict(extra="forbid")

    alternative_id: str
    title: str
    description: str
    trade_offs: str
    reason_not_selected: str


class BenchmarkPlan(BaseModel):
    """Suggested benchmark plan to validate the recommendation."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    command_template: str | None = None
    scenarios: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level recommendation
# ---------------------------------------------------------------------------


class InferenceDesignRecommendation(BaseModel):
    """Complete inference design recommendation produced by the planner."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "0.1.0"
    design_id: UUID = Field(default_factory=uuid4)

    model_identity: ModelIdentity
    selected_hardware_pool_ids: list[str] = Field(default_factory=list)
    validation_summary: str

    deployment_topology: DeploymentTopology
    parallelism: ParallelismConfig
    replica_plan: ReplicaPlan
    vllm_configuration: VllmConfiguration
    memory_capacity_estimate: MemoryCapacityEstimate

    performance_forecast: PerformanceForecast | None = None
    cost_estimate: CostEstimate | None = None
    autoscaling_recommendation: AutoscalingRecommendation | None = None
    routing_and_cache_recommendation: RoutingAndCacheRecommendation | None = None

    assumptions: list[Assumption] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    alternatives: list[Alternative] = Field(default_factory=list)
    benchmark_plan: BenchmarkPlan | None = None

    evidence_index: list[UUID] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    generated_at: datetime
    estimator_version: str
