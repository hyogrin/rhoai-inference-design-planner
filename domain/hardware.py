"""Hardware inventory and pool domain models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HardwarePool(BaseModel):
    """A homogeneous pool of accelerator nodes."""

    model_config = ConfigDict(extra="forbid")

    pool_id: str
    accelerator_vendor: str
    accelerator_model: str
    accelerator_count_per_node: int = Field(ge=1)
    node_count: int = Field(ge=1)
    hbm_gb_per_accelerator: float = Field(gt=0)
    mig_enabled: bool = False

    intra_node_interconnect: Literal["none", "pcie", "nvlink", "nvswitch", "xgmi", "other"] = "none"
    inter_node_network: Literal["ethernet", "roce", "infiniband", "efa", "other"] = "ethernet"
    inter_node_bandwidth_gbps: float | None = None
    rdma_enabled: bool = False

    cpu_model: str | None = None
    cpu_count: int | None = None
    ram_gb_per_node: float | None = None
    local_nvme_gb_per_node: float | None = None

    cloud_instance_type: str | None = None
    availability_zone: str | None = None
    quantity_available: int | None = None
    hourly_price_override: float | None = None


class HardwareInventory(BaseModel):
    """Complete hardware environment description."""

    model_config = ConfigDict(extra="forbid")

    environment_type: Literal["on_prem", "aws", "azure", "gcp", "other_cloud", "hybrid"]
    region: str | None = None
    rhoai_version: str | None = None
    vllm_version_target: str | None = None
    currency: str = "USD"
    pools: list[HardwarePool] = Field(default_factory=list)
