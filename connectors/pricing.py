"""Cloud and on-premises GPU pricing connector.

Provides cost data for GPU instances across major cloud providers
and on-premises GPU hardware for TCO calculations.

Data is loaded from JSON files in connectors/data/ for easy updates
without code changes.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from domain.evidence import EvidenceItem

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"

# Canonical GPU name mapping: full sizing key → scaling factor key
_GPU_SCALING_KEY: dict[str, str] = {
    "B300-288GB": "B300",
    "GB300-288GB": "GB300",
    "B200-192GB": "B200",
    "GB200-192GB": "GB200",
    "H200-141GB": "H200",
    "H100-80GB": "H100",
    "MI300X-192GB": "MI300X",
    "A100-80GB": "A100-80GB",
    "A100-40GB": "A100-40GB",
    "L40S-48GB": "L40S",
    "RTX-PRO-6000-96GB": "RTX-PRO-6000",
    "A10G-24GB": "A10G",
    "L4-24GB": "L4",
    "T4-16GB": "T4",
}


def _resolve_scaling_key(gpu_type: str) -> str:
    """Resolve a GPU type string to its scaling factor dictionary key."""
    if gpu_type in _GPU_SCALING_KEY:
        return _GPU_SCALING_KEY[gpu_type]
    # Try exact match in scaling dicts (handles both short and full names)
    upper = gpu_type.upper()
    for canonical, key in _GPU_SCALING_KEY.items():
        if upper == canonical.upper() or upper == key.upper():
            return key
    # Fallback: first segment before dash
    return gpu_type.split("-")[0]


def _load_json(filename: str) -> dict[str, Any]:
    path = _DATA_DIR / filename
    with path.open() as f:
        return json.load(f)


_cloud_pricing_cache: dict[str, Any] | None = None
_gpu_specs_cache: dict[str, Any] | None = None
_onprem_tco_cache: dict[str, Any] | None = None


def _get_cloud_pricing() -> dict[str, Any]:
    global _cloud_pricing_cache
    if _cloud_pricing_cache is None:
        _cloud_pricing_cache = _load_json("cloud_gpu_pricing.json")
    return _cloud_pricing_cache


def _get_gpu_specs() -> dict[str, Any]:
    global _gpu_specs_cache
    if _gpu_specs_cache is None:
        _gpu_specs_cache = _load_json("gpu_specs.json")
    return _gpu_specs_cache


def _get_onprem_tco() -> dict[str, Any]:
    global _onprem_tco_cache
    if _onprem_tco_cache is None:
        _onprem_tco_cache = _load_json("onprem_tco.json")
    return _onprem_tco_cache


class PricingConnector:
    """Provides GPU pricing data for cost estimation."""

    def __init__(self):
        self._cloud_data = _get_cloud_pricing()
        self._gpu_specs = _get_gpu_specs()
        self._onprem_tco = _get_onprem_tco()

    async def get_pricing_evidence(
        self,
        gpu_type: str,
        gpu_count: int = 1,
        environment_type: str = "on_prem",
        providers: list[str] | None = None,
    ) -> list[EvidenceItem]:
        """Get pricing evidence based on environment type.

        For cloud environments, returns matching cloud instance pricing.
        For on-prem, returns TCO breakdown evidence.
        """
        if environment_type == "on_prem":
            return self._get_onprem_tco_evidence(gpu_type, gpu_count)

        return self._get_cloud_pricing_evidence(
            gpu_type, gpu_count, providers=[environment_type] if not providers else providers
        )

    def _get_cloud_pricing_evidence(
        self,
        gpu_type: str,
        gpu_count: int = 1,
        providers: list[str] | None = None,
    ) -> list[EvidenceItem]:
        """Get cloud pricing evidence for a specific GPU configuration."""
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

        on_prem = self._get_on_prem_gpu_pricing(gpu_type)
        if on_prem:
            evidence.append(on_prem)

        return evidence

    def _get_onprem_tco_evidence(
        self,
        gpu_type: str,
        gpu_count: int,
    ) -> list[EvidenceItem]:
        """Get on-premises TCO evidence with full cost breakdown."""
        evidence: list[EvidenceItem] = []
        ref_configs = self._onprem_tco.get("reference_configs", {})
        scaling = self._onprem_tco.get("cost_scaling_factors", {})
        sources = self._onprem_tco.get("tco_sources", {})

        ref_config = self._find_closest_reference(ref_configs, gpu_type, gpu_count)
        if not ref_config:
            gpu_specs = self.get_gpu_specs(gpu_type)
            if gpu_specs:
                return [self._build_scaled_tco_evidence(gpu_type, gpu_count, scaling, sources)]
            return evidence

        hw = ref_config["hardware"]
        power = ref_config["power_and_cooling"]
        colo = ref_config["colocation"]
        staff = ref_config["staffing"]
        rh_sub = ref_config["redhat_ai_subscription"]
        scale = gpu_count / ref_config["gpu_count"]

        # Apply hardware cost ratio if GPU type differs from reference
        scaling_key = _resolve_scaling_key(gpu_type)
        ref_scaling_key = _resolve_scaling_key(ref_config["gpu_type"])
        hw_ratio = scaling.get("hardware_cost_ratio", {}).get(scaling_key, 1.0)
        ref_hw_ratio = scaling.get("hardware_cost_ratio", {}).get(ref_scaling_key, 1.0)
        type_scale = hw_ratio / ref_hw_ratio if ref_hw_ratio else 1.0

        hw_monthly = hw["monthly_cost_usd"] * scale * type_scale
        power_monthly = self._calc_power_cost(gpu_type, gpu_count, scaling, power)
        colo_monthly = colo["per_gpu_monthly_usd"] * gpu_count
        staff_monthly = staff["monthly_cost_usd"]
        rh_monthly = (rh_sub["list_price_per_gpu_annual_usd"] * gpu_count) / 12
        total_monthly = hw_monthly + power_monthly + colo_monthly + staff_monthly + rh_monthly

        # Get actual kW for display
        actual_kw = scaling.get("power_kw_per_gpu", {}).get(
            scaling_key, power["kw_per_gpu"]
        )

        tco_summary = (
            f"On-Premises TCO for {gpu_count}x {gpu_type} (monthly): "
            f"Hardware depreciation (3yr): ${hw_monthly:,.0f}; "
            f"Power & cooling: ${power_monthly:,.0f}; "
            f"Colocation: ${colo_monthly:,.0f}; "
            f"Staffing (1 FTE shared): ${staff_monthly:,.0f}; "
            f"Red Hat AI Inference ({gpu_count} GPUs): ${rh_monthly:,.0f}; "
            f"Total: ${total_monthly:,.0f}/month"
        )

        evidence.append(EvidenceItem(
            evidence_id=uuid4(),
            category="pricing",
            claim_type="tco",
            title=f"On-Premises TCO: {gpu_count}x {gpu_type}",
            summary=tco_summary,
            source_url=sources.get("lenovo_tco_2026", {}).get("url", ""),
            source_domain="lenovopress.lenovo.com",
            publisher="Lenovo / AMCompute / Red Hat",
            retrieved_at=datetime.now(UTC),
            hardware_signature=f"{gpu_count}x{gpu_type}",
            source_tier="secondary",
            verification_level="estimated",
            freshness_status="current",
        ))

        evidence.append(EvidenceItem(
            evidence_id=uuid4(),
            category="pricing",
            claim_type="tco_breakdown",
            title=f"Hardware Depreciation: {gpu_count}x {gpu_type}",
            summary=(
                f"Server hardware ~${hw['total_cost_usd'] * scale * type_scale:,.0f} "
                f"over {hw['depreciation_years']}yr = ${hw_monthly:,.0f}/month. "
                f"Warranty/support typically included by OEM."
            ),
            source_url=hw["source_url"],
            source_domain="lenovopress.lenovo.com",
            publisher="Lenovo",
            retrieved_at=datetime.now(UTC),
            hardware_signature=f"{gpu_count}x{gpu_type}",
            source_tier="primary",
            verification_level="estimated",
            freshness_status="current",
        ))

        evidence.append(EvidenceItem(
            evidence_id=uuid4(),
            category="pricing",
            claim_type="tco_breakdown",
            title=f"Power & Cooling: {gpu_count}x {gpu_type}",
            summary=(
                f"{gpu_count} GPUs × {actual_kw}kW × "
                f"PUE {power['pue_factor']} × {power['hours_per_month']}hrs × "
                f"${power['electricity_rate_per_kwh_usd']}/kWh = ${power_monthly:,.0f}/month"
            ),
            source_url=power["source_url"],
            source_domain="amcompute.com",
            publisher="AMCompute",
            retrieved_at=datetime.now(UTC),
            hardware_signature=f"{gpu_count}x{gpu_type}",
            source_tier="secondary",
            verification_level="estimated",
            freshness_status="current",
        ))

        evidence.append(EvidenceItem(
            evidence_id=uuid4(),
            category="pricing",
            claim_type="tco_breakdown",
            title=f"Red Hat AI Subscription: {gpu_count}x {gpu_type}",
            summary=(
                f"Red Hat AI Inference (Premium): ${rh_sub['list_price_per_gpu_annual_usd']:,}/GPU/yr × "
                f"{gpu_count} GPUs = ${rh_sub['list_price_per_gpu_annual_usd'] * gpu_count:,}/yr "
                f"(${rh_monthly:,.0f}/month). "
                f"Alternative: RHAIE per-node at ${rh_sub['alternatives']['rhaie_per_node_annual_usd']:,}/yr."
            ),
            source_url=rh_sub["source_url"],
            source_domain="redhat.com",
            publisher="Red Hat",
            retrieved_at=datetime.now(UTC),
            hardware_signature=f"{gpu_count}x{gpu_type}",
            source_tier="primary",
            verification_level="reported",
            freshness_status="current",
        ))

        perf = ref_config.get("performance")
        if perf:
            token_scale = gpu_count / ref_config["gpu_count"]
            est_tokens_b = perf["conservative_monthly_tokens_billion"] * token_scale
            evidence.append(EvidenceItem(
                evidence_id=uuid4(),
                category="pricing",
                claim_type="capacity",
                title=f"Estimated Capacity: {gpu_count}x {gpu_type}",
                summary=(
                    f"Est. ~{est_tokens_b:.1f}B tokens/month "
                    f"({perf['input_output_ratio']} input/output ratio). "
                    f"Cost per 1M tokens: ~${(total_monthly / (est_tokens_b * 1000)) * 1000:.4f}"
                ),
                source_url=perf["source_url"],
                source_domain="mlcommons.org",
                publisher="MLCommons",
                retrieved_at=datetime.now(UTC),
                hardware_signature=f"{gpu_count}x{gpu_type}",
                source_tier="primary",
                verification_level="estimated",
                freshness_status="current",
            ))

        gpu_pricing = self._get_on_prem_gpu_pricing(gpu_type)
        if gpu_pricing:
            evidence.append(gpu_pricing)

        return evidence

    def _calc_power_cost(
        self,
        gpu_type: str,
        gpu_count: int,
        scaling: dict[str, Any],
        power_ref: dict[str, Any],
    ) -> float:
        """Calculate power & cooling cost, using per-GPU power from scaling factors if available."""
        scaling_key = _resolve_scaling_key(gpu_type)
        kw = scaling.get("power_kw_per_gpu", {}).get(
            scaling_key, power_ref["kw_per_gpu"]
        )
        return (
            gpu_count * kw * power_ref["pue_factor"]
            * power_ref["hours_per_month"] * power_ref["electricity_rate_per_kwh_usd"]
        )

    def _find_closest_reference(
        self,
        ref_configs: dict[str, Any],
        gpu_type: str,
        gpu_count: int,
    ) -> dict[str, Any] | None:
        """Find the closest matching reference configuration."""
        normalized = gpu_type.upper().replace(" ", "-")
        for _key, config in ref_configs.items():
            if normalized in config.get("gpu_type", "").upper():
                return config
        for _key, config in ref_configs.items():
            return config
        return None

    def _build_scaled_tco_evidence(
        self,
        gpu_type: str,
        gpu_count: int,
        scaling: dict[str, Any],
        sources: dict[str, Any],
    ) -> EvidenceItem:
        """Build estimated TCO for GPU types without a reference config."""
        scaling_key = _resolve_scaling_key(gpu_type)
        hw_ratio = scaling.get("hardware_cost_ratio", {}).get(scaling_key, 1.0)
        ref_hw_total = 397801
        hw_monthly = (ref_hw_total * hw_ratio * gpu_count / 8) / 36

        kw = scaling.get("power_kw_per_gpu", {}).get(scaling_key, 1.0)
        power_monthly = gpu_count * kw * 1.35 * 720 * 0.07
        colo_monthly = gpu_count * 150
        staff_monthly = 6000
        rh_monthly = (2500 * gpu_count) / 12
        total = hw_monthly + power_monthly + colo_monthly + staff_monthly + rh_monthly

        return EvidenceItem(
            evidence_id=uuid4(),
            category="pricing",
            claim_type="tco",
            title=f"On-Premises TCO (estimated): {gpu_count}x {gpu_type}",
            summary=(
                f"Estimated TCO for {gpu_count}x {gpu_type}: "
                f"Hardware: ${hw_monthly:,.0f}; Power: ${power_monthly:,.0f}; "
                f"Colo: ${colo_monthly:,.0f}; Staff: ${staff_monthly:,.0f}; "
                f"RH AI Inference: ${rh_monthly:,.0f}; "
                f"Total: ${total:,.0f}/month (scaled from H100 reference)"
            ),
            source_url=sources.get("lenovo_tco_2026", {}).get("url", ""),
            source_domain="lenovopress.lenovo.com",
            publisher="Lenovo / AMCompute / Red Hat",
            retrieved_at=datetime.now(UTC),
            hardware_signature=f"{gpu_count}x{gpu_type}",
            source_tier="secondary",
            verification_level="estimated",
            freshness_status="current",
        )

    def get_gpu_specs(self, gpu_type: str) -> dict[str, Any] | None:
        """Get hardware specifications for a GPU type."""
        normalized = gpu_type.upper().replace(" ", "-")
        for key, specs in self._gpu_specs.items():
            if normalized in key.upper() or key.upper() in normalized:
                return specs
        for _key, specs in self._gpu_specs.items():
            if gpu_type.upper() in specs["gpu_name"].upper():
                return specs
        return None

    def get_available_gpus(self) -> list[str]:
        """List all known GPU types."""
        return list(self._gpu_specs.keys())

    def get_instance_by_name(
        self, provider: str, instance_name: str
    ) -> dict[str, Any] | None:
        """Look up a specific cloud instance by provider key and instance SKU."""
        provider_data = self._cloud_data.get(provider)
        if not provider_data:
            return None
        inst = provider_data.get("instances", {}).get(instance_name)
        if inst:
            return {
                "provider": provider_data["provider"],
                "instance_name": instance_name,
                "region": inst.get("region", ""),
                **inst,
            }
        return None

    def get_cloud_instances_for_gpu(
        self, gpu_type: str, min_gpu_count: int = 1
    ) -> list[dict[str, Any]]:
        """Get all cloud instances that offer the specified GPU type."""
        gpu_upper = gpu_type.upper()
        base_gpu = gpu_type.split("-")[0].upper()
        results: list[dict[str, Any]] = []
        for _provider_key, provider_data in self._cloud_data.items():
            for instance_name, instance in provider_data["instances"].items():
                inst_gpu = instance["gpu"].upper()
                if (
                    (gpu_upper in inst_gpu or base_gpu in inst_gpu or inst_gpu in gpu_upper)
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
        gpu_upper = gpu_type.upper()
        base_gpu = gpu_type.split("-")[0].upper()
        matches: dict[str, Any] = {}
        for name, data in instances.items():
            data_gpu = data["gpu"].upper()
            if (
                (gpu_upper in data_gpu or base_gpu in data_gpu or data_gpu in gpu_upper)
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

    def _get_on_prem_gpu_pricing(self, gpu_type: str) -> EvidenceItem | None:
        """Get on-premises GPU hardware pricing evidence."""
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
