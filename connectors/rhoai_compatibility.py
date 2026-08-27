"""RHOAI compatibility connector.

Fetches the current Red Hat supported-configuration matrix to validate
that recommended features are available in the user's RHOAI version.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx

from domain.evidence import EvidenceItem

logger = logging.getLogger(__name__)

_RHOAI_DOCS_BASE = os.getenv(
    "RHOAI_DOCS_BASE_URL",
    "https://access.redhat.com/articles/rhoai-supported-configs",
)

_VALIDATED_MODELS_URL = (
    "https://docs.redhat.com/en/documentation/red_hat_ai/3/"
    "html-single/validated_models/index#model-support-matrix_validated-models"
)

_FEATURE_LABELS: dict[str, str] = {
    "tensor_parallel": "Tensor Parallelism",
    "pipeline_parallel": "Pipeline Parallelism",
    "expert_parallel": "Expert Parallelism",
    "prefix_caching": "Prefix Caching",
    "chunked_prefill": "Chunked Prefill",
    "speculative_decoding": "Speculative Decoding",
    "fp8_quantization": "FP8 Quantization",
    "lora_serving": "LoRA Multi-Adapter Serving",
    "llmd": "LLM-D (Distributed Inference Gateway)",
    "prefix_aware_routing": "Prefix-Aware Routing",
}

_STATUS_LABELS: dict[str, str] = {
    "ga": "Generally Available",
    "tp": "Technology Preview",
    "dev": "Developer Preview",
    "unsupported": "Not Supported",
}

_KNOWN_RHOAI_VERSIONS: dict[str, dict[str, Any]] = {
    "2.16": {
        "vllm_version": "0.6.3",
        "kserve_version": "0.13",
        "features": {
            "tensor_parallel": "ga",
            "pipeline_parallel": "tp",
            "expert_parallel": "unsupported",
            "prefix_caching": "tp",
            "chunked_prefill": "tp",
            "speculative_decoding": "tp",
            "fp8_quantization": "ga",
            "lora_serving": "tp",
        },
    },
    "2.17": {
        "vllm_version": "0.7.3",
        "kserve_version": "0.14",
        "features": {
            "tensor_parallel": "ga",
            "pipeline_parallel": "tp",
            "expert_parallel": "tp",
            "prefix_caching": "ga",
            "chunked_prefill": "ga",
            "speculative_decoding": "tp",
            "fp8_quantization": "ga",
            "lora_serving": "ga",
        },
    },
    "2.18": {
        "vllm_version": "0.8.4",
        "kserve_version": "0.14",
        "features": {
            "tensor_parallel": "ga",
            "pipeline_parallel": "ga",
            "expert_parallel": "ga",
            "prefix_caching": "ga",
            "chunked_prefill": "ga",
            "speculative_decoding": "tp",
            "fp8_quantization": "ga",
            "lora_serving": "ga",
            "llmd": "tp",
        },
    },
    "3.0": {
        "vllm_version": "0.9.1",
        "kserve_version": "0.15",
        "llmd_version": "0.1.0",
        "features": {
            "tensor_parallel": "ga",
            "pipeline_parallel": "ga",
            "expert_parallel": "ga",
            "prefix_caching": "ga",
            "chunked_prefill": "ga",
            "speculative_decoding": "ga",
            "fp8_quantization": "ga",
            "lora_serving": "ga",
            "llmd": "tp",
            "prefix_aware_routing": "tp",
        },
    },
    "3.4": {
        "vllm_version": "0.18.2",
        "kserve_version": "0.15",
        "llmd_version": "0.3.0",
        "features": {
            "tensor_parallel": "ga",
            "pipeline_parallel": "ga",
            "expert_parallel": "ga",
            "prefix_caching": "ga",
            "chunked_prefill": "ga",
            "speculative_decoding": "ga",
            "fp8_quantization": "ga",
            "lora_serving": "ga",
            "llmd": "ga",
            "prefix_aware_routing": "ga",
        },
    },
    "3.5": {
        "vllm_version": "0.24.0",
        "kserve_version": "0.16",
        "llmd_version": "0.4.0",
        "features": {
            "tensor_parallel": "ga",
            "pipeline_parallel": "ga",
            "expert_parallel": "ga",
            "prefix_caching": "ga",
            "chunked_prefill": "ga",
            "speculative_decoding": "ga",
            "fp8_quantization": "ga",
            "lora_serving": "ga",
            "llmd": "ga",
            "prefix_aware_routing": "ga",
        },
    },
}

_LIVE_FETCH_TIMEOUT = 15


class RhoaiCompatibilityConnector:
    """Checks RHOAI compatibility and feature support."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] | None = None
        self._cache_time: datetime | None = None
        self._cache_ttl_hours: int = 24

    async def check_compatibility(
        self,
        rhoai_version: str | None,
        vllm_version_target: str | None = None,
    ) -> list[EvidenceItem]:
        """Check RHOAI compatibility for the given version.

        Returns evidence items about platform compatibility, feature status,
        and optional vLLM version compatibility.
        """
        if rhoai_version is None:
            latest = max(_KNOWN_RHOAI_VERSIONS.keys(), key=_version_key)
            logger.info("No RHOAI version specified, defaulting to latest known: %s", latest)
            rhoai_version = latest

        version_data = await self._resolve_version_data(rhoai_version)
        if version_data is None:
            logger.warning("Unknown RHOAI version: %s", rhoai_version)
            return [
                EvidenceItem(
                    evidence_id=uuid4(),
                    category="platform_compatibility",
                    claim_type="compatibility",
                    title=f"Unknown RHOAI version {rhoai_version}",
                    summary=(
                        f"RHOAI version {rhoai_version} is not in the known compatibility "
                        f"matrix. Known versions: {', '.join(sorted(_KNOWN_RHOAI_VERSIONS, key=_version_key))}."
                    ),
                    source_url=_RHOAI_DOCS_BASE,
                    source_domain="access.redhat.com",
                    publisher="Red Hat",
                    retrieved_at=datetime.now(UTC),
                    source_tier="primary",
                    verification_level="verified",
                    parser_warnings=[f"RHOAI {rhoai_version} not found in compatibility matrix"],
                ),
            ]

        evidence = self._build_compatibility_evidence(rhoai_version, version_data)

        if vllm_version_target:
            compat = self.check_vllm_version_compatibility(rhoai_version, vllm_version_target)
            evidence.append(
                EvidenceItem(
                    evidence_id=uuid4(),
                    category="platform_compatibility",
                    claim_type="compatibility",
                    title=f"vLLM version compatibility — RHOAI {rhoai_version}",
                    summary=compat["summary"],
                    source_url=_RHOAI_DOCS_BASE,
                    source_domain="access.redhat.com",
                    publisher="Red Hat",
                    retrieved_at=datetime.now(UTC),
                    vllm_version=compat["included_vllm"],
                    source_tier="primary",
                    verification_level="verified",
                    parser_warnings=compat.get("warnings", []),
                ),
            )

        return evidence

    def get_vllm_version_for_rhoai(self, rhoai_version: str) -> str | None:
        """Get the included vLLM version for an RHOAI release."""
        data = _KNOWN_RHOAI_VERSIONS.get(rhoai_version)
        if data is None:
            return None
        return data.get("vllm_version")

    def get_feature_status(self, rhoai_version: str, feature: str) -> str:
        """Get feature lifecycle status.

        Returns one of ``"ga"``, ``"tp"``, ``"dev"``, or ``"unsupported"``.
        """
        data = _KNOWN_RHOAI_VERSIONS.get(rhoai_version)
        if data is None:
            return "unsupported"
        features: dict[str, str] = data.get("features", {})
        return features.get(feature, "unsupported")

    def check_vllm_version_compatibility(
        self, rhoai_version: str, required_vllm_version: str
    ) -> dict[str, Any]:
        """Check if the required vLLM version is available in the RHOAI release.

        Returns a dict with ``compatible`` (bool), ``included_vllm``,
        ``summary``, and optional ``warnings``.
        """
        included = self.get_vllm_version_for_rhoai(rhoai_version)
        if included is None:
            return {
                "compatible": False,
                "included_vllm": None,
                "summary": (
                    f"Cannot determine vLLM version for RHOAI {rhoai_version}."
                ),
                "warnings": [f"RHOAI {rhoai_version} not in known versions"],
            }

        included_parts = _parse_version(included)
        required_parts = _parse_version(required_vllm_version)

        if included_parts >= required_parts:
            return {
                "compatible": True,
                "included_vllm": included,
                "summary": (
                    f"RHOAI {rhoai_version} includes vLLM {included}, which satisfies "
                    f"the required version {required_vllm_version}."
                ),
            }

        return {
            "compatible": False,
            "included_vllm": included,
            "summary": (
                f"RHOAI {rhoai_version} includes vLLM {included}, but the model or "
                f"feature requires vLLM ≥{required_vllm_version}. Consider upgrading "
                f"RHOAI or adjusting your requirements."
            ),
            "warnings": [
                f"vLLM {included} < {required_vllm_version}",
            ],
        }

    async def check_validated_models_matrix(
        self,
        model_repo_id: str,
    ) -> list[EvidenceItem]:
        """Check if a model (or related variant) exists in the Red Hat validated models matrix.

        Fetches the validated models page and searches for the model name or family.
        Returns evidence items describing which validated variants are available and
        what GPU configurations they support.
        """
        if not model_repo_id:
            return []

        model_name = model_repo_id.split("/")[-1].lower()
        # Extract model family keywords for fuzzy matching
        family_keywords = []
        for part in model_name.replace("-", " ").replace("_", " ").split():
            if len(part) > 2 and not part.replace(".", "").isdigit():
                family_keywords.append(part)

        matches = await self._search_validated_matrix(model_name, family_keywords)

        now = datetime.now(UTC)
        evidence: list[EvidenceItem] = []

        matrix_ver = matches[0].get("matrix_version", "unknown") if matches else self._load_local_matrix().get("matrix_version", "unknown")

        if matches:
            gpu_info = []
            for m in matches[:5]:
                gpu_str = m.get("supported_gpus", "")
                status = m.get("status", "")
                name = m.get("model", "")
                vram = m.get("min_vram", m.get("min_vram_gb", "?"))
                if isinstance(vram, (int, float)):
                    vram = f"{vram} GB"
                gpu_info.append(
                    f"  • {name} [{status}] — {gpu_str} (min vRAM: {vram})"
                )

            evidence.append(
                EvidenceItem(
                    evidence_id=uuid4(),
                    category="platform_compatibility",
                    claim_type="compatibility",
                    title=f"Red Hat AI Validated Models (matrix v{matrix_ver}) — {model_repo_id}",
                    summary=(
                        f"Found {len(matches)} validated/enabled variant(s) in "
                        f"Red Hat AI model support matrix (v{matrix_ver}):\n"
                        + "\n".join(gpu_info)
                    ),
                    source_url=_VALIDATED_MODELS_URL,
                    source_domain="docs.redhat.com",
                    publisher="Red Hat",
                    retrieved_at=now,
                    source_tier="primary",
                    verification_level="verified",
                ),
            )
        else:
            evidence.append(
                EvidenceItem(
                    evidence_id=uuid4(),
                    category="platform_compatibility",
                    claim_type="limitation",
                    title=f"Model not in Red Hat AI validated matrix (v{matrix_ver}) — {model_repo_id}",
                    summary=(
                        f"Model '{model_repo_id}' was not found in the Red Hat AI "
                        f"validated models support matrix (v{matrix_ver}). This doesn't mean it won't "
                        f"work — it means it hasn't been officially tested and validated "
                        f"by Red Hat. Consider using a RedHatAI/ variant if available."
                    ),
                    source_url=_VALIDATED_MODELS_URL,
                    source_domain="docs.redhat.com",
                    publisher="Red Hat",
                    retrieved_at=now,
                    source_tier="primary",
                    verification_level="verified",
                    parser_warnings=["Model not found in validated matrix"],
                ),
            )

        return evidence

    async def _search_validated_matrix(
        self, model_name: str, family_keywords: list[str]
    ) -> list[dict[str, str]]:
        """Fetch and search the validated models matrix page."""
        verify_ssl = os.getenv("VERIFY_SSL", "true").lower() != "false"

        try:
            async with httpx.AsyncClient(
                timeout=20, follow_redirects=True, verify=verify_ssl
            ) as client:
                resp = await client.get(
                    _VALIDATED_MODELS_URL,
                    headers={"Accept": "text/html"},
                )
                resp.raise_for_status()
                html_text = resp.text
        except (httpx.HTTPError, ValueError) as exc:
            logger.debug("Failed to fetch validated models matrix: %s", exc)
            return self._search_local_matrix(model_name, family_keywords)

        matches = self._parse_matrix_matches(html_text, model_name, family_keywords)
        if not matches:
            logger.debug("Live matrix parse returned 0 matches; falling back to local matrix")
            return self._search_local_matrix(model_name, family_keywords)
        return matches

    def _parse_matrix_matches(
        self, html_text: str, model_name: str, family_keywords: list[str]
    ) -> list[dict[str, str]]:
        """Parse HTML table rows and find matching models."""
        import re

        matches: list[dict[str, str]] = []
        # Look for table rows with model links — pattern: [Model](url) | ... | Status | ...
        # The markdown-like format from Red Hat docs
        row_pattern = re.compile(
            r'\[([^\]]+)\]\(([^)]+)\)\s*\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)',
            re.IGNORECASE,
        )

        for match in row_pattern.finditer(html_text):
            row_model = match.group(1).strip()
            row_model_lower = row_model.lower()

            # Check if this row matches our model
            is_match = False
            if model_name in row_model_lower:
                is_match = True
            else:
                # Check family keywords (at least 2 must match)
                keyword_hits = sum(1 for kw in family_keywords if kw in row_model_lower)
                if keyword_hits >= 2:
                    is_match = True

            if is_match:
                matches.append({
                    "model": row_model,
                    "modelcar": match.group(3).strip(),
                    "status": match.group(4).strip(),
                    "min_vllm": match.group(5).strip(),
                    "min_rhaii": match.group(6).strip(),
                    "min_rhoai": match.group(7).strip(),
                    "min_vram": match.group(8).strip(),
                    "supported_gpus": match.group(9).strip(),
                })

        return matches

    def _search_local_matrix(
        self, model_name: str, family_keywords: list[str]
    ) -> list[dict[str, str]]:
        """Fallback: search against the local JSON model matrix."""
        matrix = self._load_local_matrix()
        models = matrix.get("models", [])
        matrix_ver = matrix.get("matrix_version", "unknown")

        matches = []
        for entry in models:
            entry_model = entry.get("model", "")
            entry_lower = entry_model.lower()

            if model_name in entry_lower:
                matches.append({**entry, "min_vram": f"{entry.get('min_vram_gb', '?')} GB", "matrix_version": matrix_ver})
            else:
                keyword_hits = sum(1 for kw in family_keywords if kw in entry_lower)
                if keyword_hits >= 2:
                    matches.append({**entry, "min_vram": f"{entry.get('min_vram_gb', '?')} GB", "matrix_version": matrix_ver})

        return matches

    @staticmethod
    def _load_local_matrix() -> dict[str, Any]:
        """Load the local model matrix JSON file."""
        import json
        from pathlib import Path

        matrix_path = Path(__file__).parent / "data" / "rhai_3.4_model_matrix.json"
        try:
            with matrix_path.open() as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to load local model matrix: %s", exc)
            return {"matrix_version": "unknown", "models": []}

    async def _resolve_version_data(self, rhoai_version: str) -> dict[str, Any] | None:
        """Return version data from cache, live fetch, or local fallback."""
        if self._cache and self._cache_time:
            age_hours = (datetime.now(UTC) - self._cache_time).total_seconds() / 3600
            if age_hours < self._cache_ttl_hours and rhoai_version in self._cache:
                return self._cache[rhoai_version]

        live = await self._fetch_live_compatibility()
        if live and rhoai_version in live:
            self._cache = live
            self._cache_time = datetime.now(UTC)
            return live[rhoai_version]

        return _KNOWN_RHOAI_VERSIONS.get(rhoai_version)

    async def _fetch_live_compatibility(self) -> dict[str, Any] | None:
        """Attempt to fetch live compatibility data from Red Hat.

        The public supported-configs article does not expose a structured
        JSON API, so this is a best-effort fetch.  If the page is
        unreachable or unparseable we return ``None`` and the caller
        falls back to the local table.
        """
        try:
            async with httpx.AsyncClient(
                timeout=_LIVE_FETCH_TIMEOUT,
                follow_redirects=True,
            ) as client:
                resp = await client.get(
                    _RHOAI_DOCS_BASE,
                    headers={"Accept": "application/json, text/html"},
                )
                resp.raise_for_status()

                content_type = resp.headers.get("content-type", "")
                if "json" in content_type:
                    data = resp.json()
                    if isinstance(data, dict) and any(
                        k in data for k in ("versions", "rhoai_versions")
                    ):
                        return data.get("versions") or data.get("rhoai_versions")

                logger.debug(
                    "Live compatibility page returned non-JSON (%s); using local fallback",
                    content_type,
                )
                return None
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            logger.debug("Live compatibility fetch failed: %s", exc)
            return None

    def _build_compatibility_evidence(
        self, rhoai_version: str, version_data: dict[str, Any]
    ) -> list[EvidenceItem]:
        """Build EvidenceItem list from compatibility data."""
        now = datetime.now(UTC)
        evidence: list[EvidenceItem] = []

        vllm_ver = version_data.get("vllm_version", "unknown")
        kserve_ver = version_data.get("kserve_version", "unknown")
        extra_parts = [f"vLLM {vllm_ver}", f"KServe {kserve_ver}"]
        if "llmd_version" in version_data:
            extra_parts.append(f"LLM-D {version_data['llmd_version']}")

        evidence.append(
            EvidenceItem(
                evidence_id=uuid4(),
                category="platform_compatibility",
                claim_type="compatibility",
                title=f"RHOAI {rhoai_version} platform versions",
                summary=(
                    f"RHOAI {rhoai_version} ships with {', '.join(extra_parts)}."
                ),
                source_url=_RHOAI_DOCS_BASE,
                source_domain="access.redhat.com",
                publisher="Red Hat",
                retrieved_at=now,
                vllm_version=vllm_ver,
                source_tier="primary",
                verification_level="verified",
            ),
        )

        features: dict[str, str] = version_data.get("features", {})
        ga_features: list[str] = []
        tp_features: list[str] = []
        unsupported_features: list[str] = []

        for feat, status in features.items():
            label = _FEATURE_LABELS.get(feat, feat)
            if status == "ga":
                ga_features.append(label)
            elif status in ("tp", "dev"):
                tp_features.append(f"{label} ({_STATUS_LABELS.get(status, status)})")
            else:
                unsupported_features.append(label)

        if ga_features:
            evidence.append(
                EvidenceItem(
                    evidence_id=uuid4(),
                    category="platform_compatibility",
                    claim_type="compatibility",
                    title=f"RHOAI {rhoai_version} — GA features",
                    summary=(
                        f"Generally available features in RHOAI {rhoai_version}: "
                        f"{', '.join(sorted(ga_features))}."
                    ),
                    source_url=_RHOAI_DOCS_BASE,
                    source_domain="access.redhat.com",
                    publisher="Red Hat",
                    retrieved_at=now,
                    vllm_version=vllm_ver,
                    source_tier="primary",
                    verification_level="verified",
                ),
            )

        if tp_features:
            evidence.append(
                EvidenceItem(
                    evidence_id=uuid4(),
                    category="platform_compatibility",
                    claim_type="limitation",
                    title=f"RHOAI {rhoai_version} — preview features",
                    summary=(
                        f"Preview features (not fully supported) in RHOAI {rhoai_version}: "
                        f"{', '.join(sorted(tp_features))}. "
                        f"Technology Preview features may change without notice and are "
                        f"not recommended for production workloads."
                    ),
                    source_url=_RHOAI_DOCS_BASE,
                    source_domain="access.redhat.com",
                    publisher="Red Hat",
                    retrieved_at=now,
                    vllm_version=vllm_ver,
                    source_tier="primary",
                    verification_level="verified",
                ),
            )

        if unsupported_features:
            evidence.append(
                EvidenceItem(
                    evidence_id=uuid4(),
                    category="platform_compatibility",
                    claim_type="limitation",
                    title=f"RHOAI {rhoai_version} — unsupported features",
                    summary=(
                        f"Features not available in RHOAI {rhoai_version}: "
                        f"{', '.join(sorted(unsupported_features))}."
                    ),
                    source_url=_RHOAI_DOCS_BASE,
                    source_domain="access.redhat.com",
                    publisher="Red Hat",
                    retrieved_at=now,
                    vllm_version=vllm_ver,
                    source_tier="primary",
                    verification_level="verified",
                ),
            )

        return evidence


def _version_key(v: str) -> tuple[int, ...]:
    """Convert a version string like ``"2.17"`` to a sortable tuple."""
    parts: list[int] = []
    for segment in v.split("."):
        try:
            parts.append(int(segment))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse a dotted version string into a comparable tuple."""
    return _version_key(v)
