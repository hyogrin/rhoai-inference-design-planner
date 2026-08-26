"""Domain model for evidence items.

Every claim in a recommendation must trace back to an EvidenceItem
with source URL, retrieval time, and verification level.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel, extra="forbid"):
    """A single piece of provenance-tracked evidence."""

    evidence_id: UUID = Field(default_factory=uuid4)
    category: Literal[
        "model_metadata",
        "recipe",
        "redhat_evaluation",
        "community",
        "pricing",
        "platform_compatibility",
    ]
    claim_type: Literal[
        "architecture",
        "compatibility",
        "tested_hardware",
        "accuracy",
        "serving_performance",
        "model_strength",
        "limitation",
        "price",
    ]
    title: str
    summary: str
    source_url: str
    source_domain: str | None = None
    publisher: str | None = None
    published_at: datetime | None = None
    retrieved_at: datetime
    model_revision: str | None = None
    vllm_version: str | None = None
    hardware_signature: str | None = None
    workload_signature: str | None = None
    raw_excerpt_hash: str | None = None
    source_tier: Literal["primary", "official_secondary", "community"]
    verification_level: Literal["verified", "reported", "inferred", "unknown"]
    freshness_status: str | None = None
    parser_warnings: list[str] = []
