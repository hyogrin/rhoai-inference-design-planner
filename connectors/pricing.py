"""Cloud and on-premises GPU pricing connector.

Provides cost data for GPU instances across major cloud providers
and on-premises GPU hardware for TCO calculations.
"""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from domain.evidence import EvidenceItem

logger = logging.getLogger(__name__)

_CLOUD_GPU_PRICING: dict[str, dict[str, Any]] = {
    "aws": {
        "provider": "AWS",
        "instances": {
            "p5.48xlarge": {
                "gpu": "H100",
                "gpu_count": 8,
                "gpu_memory_gb": 80,
                "vcpus": 192,
                "ram_gb": 2048,
                "on_demand_hourly": 98.32,
                "spot_hourly": 58.99,
                "reserved_1yr_hourly": 63.91,
                "region": "us-east-1",
            },
            "p5e.48xlarge": {
                "gpu": "H200",
                "gpu_count": 8,
                "gpu_memory_gb": 141,
                "vcpus": 192,
                "ram_gb": 2048,
                "on_demand_hourly": 120.00,
                "spot_hourly": 72.00,
                "reserved_1yr_hourly": 78.00,
                "region": "us-east-1",
            },
            "p4d.24xlarge": {
                "gpu": "A100-80GB",
                "gpu_count": 8,
                "gpu_memory_gb": 80,
                "vcpus": 96,
                "ram_gb": 1152,
                "on_demand_hourly": 32.77,
                "spot_hourly": 13.86,
                "reserved_1yr_hourly": 20.70,
                "region": "us-east-1",
            },
            "p4de.24xlarge": {
                "gpu": "A100-80GB",
                "gpu_count": 8,
                "gpu_memory_gb": 80,
                "vcpus": 96,
                "ram_gb": 1152,
                "on_demand_hourly": 40.96,
                "spot_hourly": 16.38,
                "reserved_1yr_hourly": 26.62,
                "region": "us-east-1",
            },
            "g5.xlarge": {
                "gpu": "A10G",
                "gpu_count": 1,
                "gpu_memory_gb": 24,
                "vcpus": 4,
                "ram_gb": 16,
                "on_demand_hourly": 1.01,
                "spot_hourly": 0.30,
                "reserved_1yr_hourly": 0.64,
                "region": "us-east-1",
            },
            "g5.12xlarge": {
                "gpu": "A10G",
                "gpu_count": 4,
                "gpu_memory_gb": 24,
                "vcpus": 48,
                "ram_gb": 192,
                "on_demand_hourly": 5.67,
                "spot_hourly": 1.70,
                "reserved_1yr_hourly": 3.58,
                "region": "us-east-1",
            },
            "g5.48xlarge": {
                "gpu": "A10G",
                "gpu_count": 8,
                "gpu_memory_gb": 24,
                "vcpus": 192,
                "ram_gb": 768,
                "on_demand_hourly": 16.29,
                "spot_hourly": 5.37,
                "reserved_1yr_hourly": 10.28,
                "region": "us-east-1",
            },
            "g6.xlarge": {
                "gpu": "L4",
                "gpu_count": 1,
                "gpu_memory_gb": 24,
                "vcpus": 4,
                "ram_gb": 16,
                "on_demand_hourly": 0.80,
                "spot_hourly": 0.24,
                "reserved_1yr_hourly": 0.51,
                "region": "us-east-1",
            },
            "g6.48xlarge": {
                "gpu": "L4",
                "gpu_count": 8,
                "gpu_memory_gb": 24,
                "vcpus": 192,
                "ram_gb": 768,
                "on_demand_hourly": 13.35,
                "spot_hourly": 4.41,
                "reserved_1yr_hourly": 8.43,
                "region": "us-east-1",
            },
            "g4dn.xlarge": {
                "gpu": "T4",
                "gpu_count": 1,
                "gpu_memory_gb": 16,
                "vcpus": 4,
                "ram_gb": 16,
                "on_demand_hourly": 0.53,
                "spot_hourly": 0.16,
                "reserved_1yr_hourly": 0.33,
                "region": "us-east-1",
            },
            "g4dn.12xlarge": {
                "gpu": "T4",
                "gpu_count": 4,
                "gpu_memory_gb": 16,
                "vcpus": 48,
                "ram_gb": 192,
                "on_demand_hourly": 3.91,
                "spot_hourly": 1.17,
                "reserved_1yr_hourly": 2.47,
                "region": "us-east-1",
            },
        },
        "source_url": "https://aws.amazon.com/ec2/pricing/on-demand/",
        "last_updated": "2026-07-01",
    },
    "gcp": {
        "provider": "Google Cloud",
        "instances": {
            "a3-highgpu-8g": {
                "gpu": "H100",
                "gpu_count": 8,
                "gpu_memory_gb": 80,
                "vcpus": 208,
                "ram_gb": 1872,
                "on_demand_hourly": 101.22,
                "spot_hourly": 30.37,
                "reserved_1yr_hourly": 63.77,
                "region": "us-central1",
            },
            "a3-megagpu-8g": {
                "gpu": "H200",
                "gpu_count": 8,
                "gpu_memory_gb": 141,
                "vcpus": 208,
                "ram_gb": 1872,
                "on_demand_hourly": 122.45,
                "spot_hourly": 36.74,
                "reserved_1yr_hourly": 79.59,
                "region": "us-central1",
            },
            "a2-highgpu-1g": {
                "gpu": "A100-80GB",
                "gpu_count": 1,
                "gpu_memory_gb": 80,
                "vcpus": 12,
                "ram_gb": 170,
                "on_demand_hourly": 5.12,
                "spot_hourly": 1.54,
                "reserved_1yr_hourly": 3.23,
                "region": "us-central1",
            },
            "a2-ultragpu-8g": {
                "gpu": "A100-80GB",
                "gpu_count": 8,
                "gpu_memory_gb": 80,
                "vcpus": 96,
                "ram_gb": 1360,
                "on_demand_hourly": 40.97,
                "spot_hourly": 12.29,
                "reserved_1yr_hourly": 25.81,
                "region": "us-central1",
            },
            "g2-standard-4": {
                "gpu": "L4",
                "gpu_count": 1,
                "gpu_memory_gb": 24,
                "vcpus": 4,
                "ram_gb": 16,
                "on_demand_hourly": 0.84,
                "spot_hourly": 0.25,
                "reserved_1yr_hourly": 0.53,
                "region": "us-central1",
            },
            "g2-standard-96": {
                "gpu": "L4",
                "gpu_count": 8,
                "gpu_memory_gb": 24,
                "vcpus": 96,
                "ram_gb": 384,
                "on_demand_hourly": 13.07,
                "spot_hourly": 3.92,
                "reserved_1yr_hourly": 8.23,
                "region": "us-central1",
            },
            "n1-standard-4-t4": {
                "gpu": "T4",
                "gpu_count": 1,
                "gpu_memory_gb": 16,
                "vcpus": 4,
                "ram_gb": 15,
                "on_demand_hourly": 0.45,
                "spot_hourly": 0.14,
                "reserved_1yr_hourly": 0.28,
                "region": "us-central1",
            },
        },
        "source_url": "https://cloud.google.com/compute/gpus-pricing",
        "last_updated": "2026-07-01",
    },
    "azure": {
        "provider": "Microsoft Azure",
        "instances": {
            "Standard_ND96isr_H100_v5": {
                "gpu": "H100",
                "gpu_count": 8,
                "gpu_memory_gb": 80,
                "vcpus": 96,
                "ram_gb": 1900,
                "on_demand_hourly": 98.32,
                "spot_hourly": 39.33,
                "reserved_1yr_hourly": 60.29,
                "region": "eastus",
            },
            "Standard_ND96isr_H200_v5": {
                "gpu": "H200",
                "gpu_count": 8,
                "gpu_memory_gb": 141,
                "vcpus": 96,
                "ram_gb": 1900,
                "on_demand_hourly": 118.50,
                "spot_hourly": 47.40,
                "reserved_1yr_hourly": 72.71,
                "region": "eastus",
            },
            "Standard_ND96amsr_A100_v4": {
                "gpu": "A100-80GB",
                "gpu_count": 8,
                "gpu_memory_gb": 80,
                "vcpus": 96,
                "ram_gb": 1900,
                "on_demand_hourly": 32.77,
                "spot_hourly": 9.83,
                "reserved_1yr_hourly": 20.70,
                "region": "eastus",
            },
            "Standard_NC24ads_A100_v4": {
                "gpu": "A100-80GB",
                "gpu_count": 1,
                "gpu_memory_gb": 80,
                "vcpus": 24,
                "ram_gb": 220,
                "on_demand_hourly": 3.67,
                "spot_hourly": 1.10,
                "reserved_1yr_hourly": 2.32,
                "region": "eastus",
            },
            "Standard_NC4as_T4_v3": {
                "gpu": "T4",
                "gpu_count": 1,
                "gpu_memory_gb": 16,
                "vcpus": 4,
                "ram_gb": 28,
                "on_demand_hourly": 0.53,
                "spot_hourly": 0.16,
                "reserved_1yr_hourly": 0.33,
                "region": "eastus",
            },
            "Standard_NC64as_T4_v3": {
                "gpu": "T4",
                "gpu_count": 4,
                "gpu_memory_gb": 16,
                "vcpus": 64,
                "ram_gb": 440,
                "on_demand_hourly": 4.35,
                "spot_hourly": 1.31,
                "reserved_1yr_hourly": 2.75,
                "region": "eastus",
            },
        },
        "source_url": "https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/",
        "last_updated": "2026-07-01",
    },
}

_ON_PREM_GPU_PRICING: dict[str, dict[str, Any]] = {
    "B200": {
        "gpu_name": "NVIDIA B200",
        "memory_gb": 192,
        "bf16_tflops": 2250,
        "fp8_tflops": 4500,
        "memory_bandwidth_tbps": 8.0,
        "tdp_watts": 1000,
        "list_price_usd": 40000,
        "typical_street_price_usd": 37000,
    },
    "H200": {
        "gpu_name": "NVIDIA H200 SXM",
        "memory_gb": 141,
        "bf16_tflops": 989,
        "fp8_tflops": 1979,
        "memory_bandwidth_tbps": 4.8,
        "tdp_watts": 700,
        "list_price_usd": 35000,
        "typical_street_price_usd": 30000,
    },
    "H100-SXM": {
        "gpu_name": "NVIDIA H100 SXM",
        "memory_gb": 80,
        "bf16_tflops": 989,
        "fp8_tflops": 1979,
        "memory_bandwidth_tbps": 3.35,
        "tdp_watts": 700,
        "list_price_usd": 30000,
        "typical_street_price_usd": 25000,
    },
    "H100-PCIe": {
        "gpu_name": "NVIDIA H100 PCIe",
        "memory_gb": 80,
        "bf16_tflops": 756,
        "fp8_tflops": 1513,
        "memory_bandwidth_tbps": 2.0,
        "tdp_watts": 350,
        "list_price_usd": 25000,
        "typical_street_price_usd": 22000,
    },
    "MI300X": {
        "gpu_name": "AMD Instinct MI300X",
        "memory_gb": 192,
        "bf16_tflops": 1307,
        "fp8_tflops": 2614,
        "memory_bandwidth_tbps": 5.3,
        "tdp_watts": 750,
        "list_price_usd": 20000,
        "typical_street_price_usd": 15000,
    },
    "A100-SXM-80GB": {
        "gpu_name": "NVIDIA A100 SXM 80GB",
        "memory_gb": 80,
        "bf16_tflops": 312,
        "fp8_tflops": 0,
        "memory_bandwidth_tbps": 2.0,
        "tdp_watts": 400,
        "list_price_usd": 15000,
        "typical_street_price_usd": 10000,
    },
    "A100-PCIe-80GB": {
        "gpu_name": "NVIDIA A100 PCIe 80GB",
        "memory_gb": 80,
        "bf16_tflops": 312,
        "fp8_tflops": 0,
        "memory_bandwidth_tbps": 2.0,
        "tdp_watts": 300,
        "list_price_usd": 11000,
        "typical_street_price_usd": 8000,
    },
    "L4": {
        "gpu_name": "NVIDIA L4",
        "memory_gb": 24,
        "bf16_tflops": 121,
        "fp8_tflops": 242,
        "memory_bandwidth_tbps": 0.3,
        "tdp_watts": 72,
        "list_price_usd": 2500,
        "typical_street_price_usd": 2200,
    },
    "L40S": {
        "gpu_name": "NVIDIA L40S",
        "memory_gb": 48,
        "bf16_tflops": 362,
        "fp8_tflops": 733,
        "memory_bandwidth_tbps": 0.864,
        "tdp_watts": 350,
        "list_price_usd": 7500,
        "typical_street_price_usd": 6500,
    },
    "A10G": {
        "gpu_name": "NVIDIA A10G",
        "memory_gb": 24,
        "bf16_tflops": 125,
        "fp8_tflops": 0,
        "memory_bandwidth_tbps": 0.6,
        "tdp_watts": 150,
        "list_price_usd": 3500,
        "typical_street_price_usd": 2500,
    },
    "T4": {
        "gpu_name": "NVIDIA T4",
        "memory_gb": 16,
        "bf16_tflops": 65,
        "fp8_tflops": 0,
        "memory_bandwidth_tbps": 0.32,
        "tdp_watts": 70,
        "list_price_usd": 2000,
        "typical_street_price_usd": 1500,
    },
}


class PricingConnector:
    """Provides GPU pricing data for cost estimation."""

    def __init__(self):
        self._cloud_data = _CLOUD_GPU_PRICING
        self._on_prem_data = _ON_PREM_GPU_PRICING

    async def get_pricing_evidence(
        self,
        gpu_type: str,
        gpu_count: int = 1,
        providers: list[str] | None = None,
    ) -> list[EvidenceItem]:
        """Get pricing evidence for a specific GPU configuration.

        Args:
            gpu_type: GPU model name (e.g., "H100", "A100-80GB")
            gpu_count: Number of GPUs needed
            providers: Specific cloud providers to check (default: all)

        Returns:
            List of EvidenceItem with pricing claims
        """
        evidence: list[EvidenceItem] = []
        target_providers = providers or list(self._cloud_data.keys())

        for provider_key in target_providers:
            provider_data = self._cloud_data.get(provider_key)
            if not provider_data:
                continue

            matching_instances = self._find_matching_instances(
                provider_data["instances"], gpu_type, gpu_count
            )
            for instance_name, instance_data in matching_instances.items():
                evidence.append(self._instance_to_evidence(
                    provider_data["provider"],
                    instance_name,
                    instance_data,
                    provider_data["source_url"],
                ))

        on_prem = self._get_on_prem_pricing(gpu_type)
        if on_prem:
            evidence.append(on_prem)

        return evidence

    def get_gpu_specs(self, gpu_type: str) -> dict[str, Any] | None:
        """Get hardware specifications for a GPU type."""
        normalized = gpu_type.upper().replace(" ", "-")
        for key, specs in self._on_prem_data.items():
            if normalized in key.upper() or key.upper() in normalized:
                return specs
        for _key, specs in self._on_prem_data.items():
            if gpu_type.upper() in specs["gpu_name"].upper():
                return specs
        return None

    def get_available_gpus(self) -> list[str]:
        """List all known GPU types."""
        return list(self._on_prem_data.keys())

    def get_cloud_instances_for_gpu(
        self, gpu_type: str, min_gpu_count: int = 1
    ) -> list[dict[str, Any]]:
        """Get all cloud instances that offer the specified GPU type."""
        results: list[dict[str, Any]] = []
        for _provider_key, provider_data in self._cloud_data.items():
            for instance_name, instance in provider_data["instances"].items():
                if (
                    gpu_type.upper() in instance["gpu"].upper()
                    and instance["gpu_count"] >= min_gpu_count
                ):
                    results.append({
                        "provider": provider_data["provider"],
                        "instance_name": instance_name,
                        "region": instance["region"],
                        **instance,
                    })
        return results

    def _find_matching_instances(
        self, instances: dict[str, Any], gpu_type: str, gpu_count: int
    ) -> dict[str, Any]:
        """Find instances matching the GPU type with sufficient count."""
        matches: dict[str, Any] = {}
        for name, data in instances.items():
            if (
                gpu_type.upper() in data["gpu"].upper()
                and data["gpu_count"] >= gpu_count
            ):
                matches[name] = data
        return matches

    def _instance_to_evidence(
        self,
        provider: str,
        instance_name: str,
        data: dict[str, Any],
        source_url: str,
    ) -> EvidenceItem:
        """Convert an instance entry to an EvidenceItem."""
        monthly_on_demand = data["on_demand_hourly"] * 730
        monthly_spot = data.get("spot_hourly", 0) * 730
        monthly_reserved = data.get("reserved_1yr_hourly", 0) * 730

        summary_parts = [
            f"{provider} {instance_name}: {data['gpu_count']}x {data['gpu']} ({data['gpu_memory_gb']}GB each)",
            f"On-demand: ${data['on_demand_hourly']:.2f}/hr (${monthly_on_demand:.0f}/mo)",
        ]
        if monthly_spot:
            summary_parts.append(f"Spot: ${data['spot_hourly']:.2f}/hr (${monthly_spot:.0f}/mo)")
        if monthly_reserved:
            summary_parts.append(
                f"1yr Reserved: ${data['reserved_1yr_hourly']:.2f}/hr (${monthly_reserved:.0f}/mo)"
            )

        return EvidenceItem(
            evidence_id=uuid4(),
            category="pricing",
            claim_type="price",
            title=f"{provider} {instance_name} pricing",
            summary="; ".join(summary_parts),
            source_url=source_url,
            source_domain=provider.lower().replace(" ", ""),
            publisher=provider,
            retrieved_at=datetime.now(UTC),
            hardware_signature=f"{data['gpu_count']}x{data['gpu']}",
            source_tier="primary",
            verification_level="reported",
            freshness_status="current",
        )

    def _get_on_prem_pricing(self, gpu_type: str) -> EvidenceItem | None:
        """Get on-premises pricing evidence for a GPU type."""
        specs = self.get_gpu_specs(gpu_type)
        if not specs:
            return None

        return EvidenceItem(
            evidence_id=uuid4(),
            category="pricing",
            claim_type="price",
            title=f"{specs['gpu_name']} on-premises pricing",
            summary=(
                f"{specs['gpu_name']}: {specs['memory_gb']}GB VRAM, "
                f"BF16 {specs['bf16_tflops']} TFLOPS, "
                f"MBW {specs['memory_bandwidth_tbps']} TB/s, "
                f"TDP {specs['tdp_watts']}W; "
                f"List ~${specs['list_price_usd']:,}, "
                f"Street ~${specs['typical_street_price_usd']:,}"
            ),
            source_url="https://www.nvidia.com/en-us/data-center/",
            source_domain="nvidia.com",
            publisher="NVIDIA",
            retrieved_at=datetime.now(UTC),
            hardware_signature=gpu_type,
            source_tier="primary",
            verification_level="reported",
            freshness_status="current",
        )
