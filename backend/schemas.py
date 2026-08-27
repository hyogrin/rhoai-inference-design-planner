from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class CreateDesignRequest(BaseModel):
    model_repo_id: str = Field(..., min_length=1, max_length=255)
    model_revision: str | None = None
    title: str | None = None


class UpdateHardwareRequest(BaseModel):
    gpu_type: str | None = None
    gpu_count: int | None = Field(None, ge=1)
    gpu_memory_gb: float | None = Field(None, gt=0)
    cpu_cores: int | None = Field(None, ge=1)
    ram_gb: float | None = Field(None, gt=0)
    interconnect: str | None = None
    constraints: dict[str, Any] | None = None


class UpdateWorkloadRequest(BaseModel):
    use_case: str | None = None
    max_concurrent_requests: int | None = Field(None, ge=1)
    target_latency_ms: float | None = Field(None, gt=0)
    max_batch_size: int | None = Field(None, ge=1)
    sequence_length: int | None = Field(None, ge=1)
    expected_throughput_tps: float | None = Field(None, gt=0)
    constraints: dict[str, Any] | None = None


class SaveRecommendationRequest(BaseModel):
    recommendation: dict[str, Any]
    view_model: dict[str, Any]


class DesignSessionResponse(BaseModel):
    session_id: UUID
    title: str | None
    status: str
    model_repo_id: str | None
    model_revision: str | None
    current_step: int
    created_at: datetime
    updated_at: datetime
    version: int

    model_config = {"from_attributes": True}


class DesignSessionDetailResponse(BaseModel):
    session_id: UUID
    title: str | None
    status: str
    model_repo_id: str | None
    model_revision: str | None
    current_step: int
    state_snapshot: dict[str, Any] | None = None
    result_snapshot: dict[str, Any] | None = None
    completed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    version: int

    model_config = {"from_attributes": True}


class DesignSessionListItem(BaseModel):
    session_id: UUID
    title: str | None
    status: str
    model_repo_id: str | None
    completed_at: datetime | None = None
    created_at: datetime
    gpu_config: str | None = None
    memory_utilization: str | None = None
    fits: bool | None = None

    model_config = {"from_attributes": True}


class DesignListResponse(BaseModel):
    items: list[DesignSessionListItem]
    total: int


class ErrorResponse(BaseModel):
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] | None = None
    correlation_id: str | None = None
