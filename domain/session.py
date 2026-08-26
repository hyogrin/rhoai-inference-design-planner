"""Design session domain model."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class DesignSession(BaseModel):
    """Tracks the state of a single inference design planning session."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID = Field(default_factory=uuid4)
    title: str | None = None
    status: Literal[
        "intake",
        "discovering",
        "validating",
        "workload_input",
        "sizing",
        "recommending",
        "completed",
        "failed",
    ] = "intake"
    model_repo_id: str | None = None
    model_revision: str | None = None
    current_step: int = Field(default=1, ge=1, le=5)
    created_at: datetime
    updated_at: datetime
    version: int = Field(default=1, ge=1, description="Optimistic concurrency version")
