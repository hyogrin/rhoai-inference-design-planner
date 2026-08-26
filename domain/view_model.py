"""Design result view model with discriminated union panels."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

# ---------------------------------------------------------------------------
# Panel base
# ---------------------------------------------------------------------------


class _PanelBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Individual panel types
# ---------------------------------------------------------------------------


class ModelSummaryPanel(_PanelBase):
    panel_type: Literal["model_summary"] = "model_summary"
    title: str = "Model Summary"
    data: dict[str, Any] = Field(default_factory=dict)


class HardwareSummaryPanel(_PanelBase):
    panel_type: Literal["hardware_summary"] = "hardware_summary"
    title: str = "Hardware Summary"
    data: dict[str, Any] = Field(default_factory=dict)


class EvidenceCoveragePanel(_PanelBase):
    panel_type: Literal["evidence_coverage"] = "evidence_coverage"
    title: str = "Evidence Coverage"
    data: dict[str, Any] = Field(default_factory=dict)


class ValidationGatePanel(_PanelBase):
    panel_type: Literal["validation_gate"] = "validation_gate"
    title: str = "Validation Gate"
    data: dict[str, Any] = Field(default_factory=dict)


class WorkloadSummaryPanel(_PanelBase):
    panel_type: Literal["workload_summary"] = "workload_summary"
    title: str = "Workload Summary"
    data: dict[str, Any] = Field(default_factory=dict)


class DeploymentTopologyPanel(_PanelBase):
    panel_type: Literal["deployment_topology"] = "deployment_topology"
    title: str = "Deployment Topology"
    data: dict[str, Any] = Field(default_factory=dict)


class ParallelismPlanPanel(_PanelBase):
    panel_type: Literal["parallelism_plan"] = "parallelism_plan"
    title: str = "Parallelism Plan"
    data: dict[str, Any] = Field(default_factory=dict)


class VllmConfigurationPanel(_PanelBase):
    panel_type: Literal["vllm_configuration"] = "vllm_configuration"
    title: str = "vLLM Configuration"
    data: dict[str, Any] = Field(default_factory=dict)


class MemoryCapacityPanel(_PanelBase):
    panel_type: Literal["memory_capacity"] = "memory_capacity"
    title: str = "Memory Capacity"
    data: dict[str, Any] = Field(default_factory=dict)


class PerformanceForecastPanel(_PanelBase):
    panel_type: Literal["performance_forecast"] = "performance_forecast"
    title: str = "Performance Forecast"
    data: dict[str, Any] = Field(default_factory=dict)


class CostEstimatePanel(_PanelBase):
    panel_type: Literal["cost_estimate"] = "cost_estimate"
    title: str = "Cost Estimate"
    data: dict[str, Any] = Field(default_factory=dict)


class AssumptionsAndRisksPanel(_PanelBase):
    panel_type: Literal["assumptions_and_risks"] = "assumptions_and_risks"
    title: str = "Assumptions & Risks"
    data: dict[str, Any] = Field(default_factory=dict)


class AlternativesPanel(_PanelBase):
    panel_type: Literal["alternatives"] = "alternatives"
    title: str = "Alternatives"
    data: dict[str, Any] = Field(default_factory=dict)


class BenchmarkPlanPanel(_PanelBase):
    panel_type: Literal["benchmark_plan"] = "benchmark_plan"
    title: str = "Benchmark Plan"
    data: dict[str, Any] = Field(default_factory=dict)


class SourceListPanel(_PanelBase):
    panel_type: Literal["source_list"] = "source_list"
    title: str = "Sources"
    data: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------


def _get_panel_type(value: Any) -> str:
    if isinstance(value, dict):
        return value.get("panel_type", "")
    return getattr(value, "panel_type", "")


Panel = Annotated[
    Annotated[ModelSummaryPanel, Tag("model_summary")]
    | Annotated[HardwareSummaryPanel, Tag("hardware_summary")]
    | Annotated[EvidenceCoveragePanel, Tag("evidence_coverage")]
    | Annotated[ValidationGatePanel, Tag("validation_gate")]
    | Annotated[WorkloadSummaryPanel, Tag("workload_summary")]
    | Annotated[DeploymentTopologyPanel, Tag("deployment_topology")]
    | Annotated[ParallelismPlanPanel, Tag("parallelism_plan")]
    | Annotated[VllmConfigurationPanel, Tag("vllm_configuration")]
    | Annotated[MemoryCapacityPanel, Tag("memory_capacity")]
    | Annotated[PerformanceForecastPanel, Tag("performance_forecast")]
    | Annotated[CostEstimatePanel, Tag("cost_estimate")]
    | Annotated[AssumptionsAndRisksPanel, Tag("assumptions_and_risks")]
    | Annotated[AlternativesPanel, Tag("alternatives")]
    | Annotated[BenchmarkPlanPanel, Tag("benchmark_plan")]
    | Annotated[SourceListPanel, Tag("source_list")],
    Discriminator(_get_panel_type),
]


# ---------------------------------------------------------------------------
# Top-level view model
# ---------------------------------------------------------------------------


class DesignResultViewModel(BaseModel):
    """Presentation-layer view model for the design result UI."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "0.1.0"
    design_id: UUID
    panels: list[Panel] = Field(default_factory=list)
    generated_at: datetime
    estimator_version: str
