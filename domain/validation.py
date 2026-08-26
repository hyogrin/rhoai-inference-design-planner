"""Validation report domain models."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ValidationCheck(BaseModel):
    """A single validation check result."""

    model_config = ConfigDict(extra="forbid")

    check_id: str
    status: Literal["passed", "warning", "failed", "skipped"]
    message: str
    evidence_ids: list[UUID] = Field(default_factory=list)
    remediation: str | None = None


class ValidationReport(BaseModel):
    """Aggregate validation gate report for model + hardware readiness."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready_for_sizing", "ready_with_limitations", "blocked"]
    checks: list[ValidationCheck] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    evidence_coverage: dict[str, bool] = Field(default_factory=dict)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    allowed_outputs: list[str] = Field(default_factory=list)
