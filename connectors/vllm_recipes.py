"""vLLM structured recipe connector.

Fetches and normalizes vLLM deployment recipes from the official recipe source.
Primary source: recipes.vllm.ai structured JSON API (taxonomy + per-model JSON)
Fallback: graceful degradation when the site is unreachable.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import yaml

from domain.evidence import EvidenceItem

logger = logging.getLogger(__name__)

RECIPES_BASE_URL = "https://recipes.vllm.ai"
TAXONOMY_JSON_URL = f"{RECIPES_BASE_URL}/taxonomy.json"
MODELS_JSON_URL = f"{RECIPES_BASE_URL}/models.json"
STRATEGIES_JSON_URL = f"{RECIPES_BASE_URL}/strategies.json"

_CLIENT_TIMEOUT = httpx.Timeout(30.0)
_DEFAULT_CACHE_TTL = timedelta(hours=24)


def _excerpt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class VllmRecipeConnector:
    """Fetches vLLM deployment recipes from recipes.vllm.ai."""

    def __init__(self, cache_ttl: timedelta = _DEFAULT_CACHE_TTL) -> None:
        self._models_cache: dict[str, Any] | None = None
        self._strategies_cache: dict[str, Any] | None = None
        self._cache_time: datetime | None = None
        self._cache_ttl = cache_ttl

    def _cache_valid(self) -> bool:
        if self._models_cache is None or self._cache_time is None:
            return False
        return datetime.now(UTC) - self._cache_time < self._cache_ttl

    async def find_recipe(self, repo_id: str) -> list[EvidenceItem]:
        """Find vLLM recipes matching the given model *repo_id*.

        Returns a list of EvidenceItem objects with recipe evidence.
        Returns an empty list when no recipe is found (not an error).
        """
        try:
            models_data = await self._fetch_models_index()
        except Exception:
            logger.warning("Could not fetch vLLM recipe index — skipping", exc_info=True)
            return []

        matches = self._match_model(models_data, repo_id)
        if not matches:
            logger.debug("No vLLM recipe match for %s", repo_id)
            return []

        evidence: list[EvidenceItem] = []
        for match in matches[:3]:
            recipe_url = match.get("json") or match.get("json_url") or match.get("url")
            if not recipe_url:
                continue
            detail = await self._fetch_recipe_detail(recipe_url)
            if detail:
                evidence.extend(self._parse_recipe_to_evidence(detail, repo_id))

        return evidence

    # ------------------------------------------------------------------
    # Network helpers
    # ------------------------------------------------------------------

    async def _fetch_models_index(self) -> list[dict[str, Any]] | dict[str, Any]:
        """Fetch the models index and cache it.

        Primary: models.json (flat list with hf_id, title, json fields)
        Fallback: taxonomy.json (categorized structure)
        """
        if self._cache_valid():
            assert self._models_cache is not None
            return self._models_cache

        async with httpx.AsyncClient(timeout=_CLIENT_TIMEOUT) as client:
            # Prefer models.json (flat list, easy to search)
            for url in (MODELS_JSON_URL, TAXONOMY_JSON_URL):
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    data = resp.json()
                    self._models_cache = data
                    self._cache_time = datetime.now(UTC)
                    entry_count = len(data) if isinstance(data, list) else "dict"
                    logger.info("Fetched vLLM recipe index from %s (%s entries)", url, entry_count)
                    return data
                except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                    logger.debug("Index URL %s unavailable: %s", url, exc)
                    continue

            raise ConnectionError("All vLLM recipe index URLs are unreachable")

    async def _fetch_strategies(self) -> dict[str, Any]:
        """Fetch strategies.json (best-effort, not required)."""
        if self._strategies_cache is not None:
            return self._strategies_cache
        async with httpx.AsyncClient(timeout=_CLIENT_TIMEOUT) as client:
            try:
                resp = await client.get(STRATEGIES_JSON_URL)
                resp.raise_for_status()
                self._strategies_cache = resp.json()
                return self._strategies_cache
            except (httpx.HTTPStatusError, httpx.RequestError):
                logger.debug("strategies.json unavailable")
                return {}

    async def _fetch_recipe_detail(self, recipe_url: str) -> dict[str, Any] | None:
        """Fetch a specific recipe's JSON or YAML detail page."""
        if not recipe_url.startswith("http"):
            recipe_url = f"{RECIPES_BASE_URL}/{recipe_url.lstrip('/')}"

        async with httpx.AsyncClient(timeout=_CLIENT_TIMEOUT) as client:
            try:
                resp = await client.get(recipe_url)
                resp.raise_for_status()
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                logger.warning("Failed to fetch recipe detail %s: %s", recipe_url, exc)
                return None

        content_type = resp.headers.get("content-type", "")
        body = resp.text

        if "json" in content_type or recipe_url.endswith(".json"):
            return resp.json()

        if recipe_url.endswith((".yaml", ".yml")) or "yaml" in content_type:
            try:
                return yaml.safe_load(body)
            except yaml.YAMLError as exc:
                logger.warning("YAML parse error for %s: %s", recipe_url, exc)
                return None

        try:
            return resp.json()
        except Exception:
            pass

        try:
            return yaml.safe_load(body)
        except Exception:
            logger.warning("Could not parse recipe detail as JSON or YAML: %s", recipe_url)
            return None

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _match_model(self, models_data: dict[str, Any] | list, repo_id: str) -> list[dict[str, Any]]:
        """Match a *repo_id* against the models index.

        Strategy:
        1. Exact match on ``hf_id`` (the primary key in recipes.vllm.ai)
        2. Base-model match (strip quantization suffixes like -FP8)
        3. Substring match on hf_id or title
        4. Model-family + parameter-size fuzzy match (e.g., Qwen3.5-35B)
        """
        repo_lower = repo_id.lower()
        matches: list[dict[str, Any]] = []
        entries = self._extract_entries(models_data)

        # Step 1: Exact match on hf_id
        for entry in entries:
            entry_id = self._get_entry_id(entry).lower()
            if entry_id == repo_lower:
                matches.append(entry)

        if matches:
            return matches

        # Step 2: Strip quant suffix and try exact match
        base_name = self._strip_quant_suffix(repo_id).lower()
        if base_name != repo_lower:
            for entry in entries:
                entry_id = self._get_entry_id(entry).lower()
                if entry_id == base_name:
                    matches.append(entry)

        if matches:
            return matches

        # Step 3: Substring match on hf_id or title
        for entry in entries:
            entry_id = self._get_entry_id(entry).lower()
            entry_title = (entry.get("title") or "").lower()
            if repo_lower in entry_id or entry_id in repo_lower:
                matches.append(entry)
            elif repo_lower in entry_title or entry_title in repo_lower:
                matches.append(entry)

        if matches:
            return matches

        # Step 4: Model family + parameter size fuzzy match
        family, param_size = self._extract_family_and_params(repo_id)
        if family:
            exact_params: list[dict[str, Any]] = []
            close_params: list[dict[str, Any]] = []
            family_only: list[dict[str, Any]] = []

            for entry in entries:
                entry_id = self._get_entry_id(entry)
                entry_title = entry.get("title") or entry_id
                entry_family, entry_params = self._extract_family_and_params(entry_id)
                if not entry_family:
                    entry_family, entry_params = self._extract_family_and_params(entry_title)
                if not entry_family:
                    continue

                if family in entry_family or entry_family in family:
                    if param_size and entry_params:
                        if param_size == entry_params:
                            exact_params.append(entry)
                        elif self._params_close(param_size, entry_params):
                            close_params.append(entry)
                    else:
                        family_only.append(entry)

            matches = exact_params or close_params or family_only

        return matches

    @staticmethod
    def _params_close(a: str, b: str) -> bool:
        """Check if two param sizes are within ~30% of each other (same class)."""
        import re
        ma = re.match(r"([\d.]+)b", a)
        mb = re.match(r"([\d.]+)b", b)
        if not (ma and mb):
            return False
        va, vb = float(ma.group(1)), float(mb.group(1))
        if va == 0 or vb == 0:
            return False
        ratio = max(va, vb) / min(va, vb)
        return ratio <= 1.3

    @staticmethod
    def _get_entry_id(entry: dict[str, Any]) -> str:
        """Extract the canonical model identifier from an index entry."""
        if entry.get("hf_id"):
            return entry["hf_id"]
        if entry.get("model_id"):
            return entry["model_id"]
        model_block = entry.get("model")
        if isinstance(model_block, dict) and model_block.get("model_id"):
            return model_block["model_id"]
        return ""

    @staticmethod
    def _extract_entries(models_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize the index into a flat list of recipe entries.

        Handles both ``{"models": [...]}`` and ``[...]`` shapes, as well
        as the taxonomy format ``{"categories": [...]}`` with nested items.
        """
        if isinstance(models_data, list):
            return models_data

        if "models" in models_data and isinstance(models_data["models"], list):
            return models_data["models"]

        if "categories" in models_data:
            entries: list[dict[str, Any]] = []
            for cat in models_data["categories"]:
                items = cat.get("items") or cat.get("models") or []
                entries.extend(items)
            return entries

        if "recipes" in models_data and isinstance(models_data["recipes"], list):
            return models_data["recipes"]

        return list(models_data.values()) if models_data else []

    @staticmethod
    def _extract_family_and_params(model_id: str) -> tuple[str, str]:
        """Extract model family name and parameter size from model ID.

        Examples:
            "RedHatAI/Qwen3.5-35B-A3B-FP8-dynamic" -> ("qwen3.5", "35b")
            "meta-llama/Llama-3.1-8B-Instruct" -> ("llama", "8b")
            "mistralai/Mistral-Small-24B" -> ("mistral", "24b")
        """
        import re

        # Get the model name part (after org/)
        name = model_id.split("/")[-1].lower() if "/" in model_id else model_id.lower()

        # Extract parameter size (e.g., 35b, 8b, 70b, 24b)
        param_match = re.search(r'(\d+\.?\d*)b(?:\b|[-_])', name)
        param_size = param_match.group(1).rstrip('.') + "b" if param_match else ""

        # Extract model family: take the first significant word(s) before numbers/suffixes
        # Strip org prefix, quant suffixes, etc.
        clean = re.sub(r'[-_](?:fp8|fp4|nvfp4|gptq|awq|int[48]|w[48]a\d+|quantized|dynamic|instruct|chat|hf|v\d+).*$', '', name, flags=re.IGNORECASE)
        # Remove param size from the name to get family
        clean = re.sub(r'[-_]?\d+\.?\d*b[-_]?.*$', '', clean, flags=re.IGNORECASE)
        # Remove trailing dashes/underscores
        family = clean.strip('-_')

        return (family, param_size)

    @staticmethod
    def _strip_quant_suffix(repo_id: str) -> str:
        """Remove common quantization suffixes to find base model.

        ``"meta-llama/Llama-3-70B-Instruct-FP8"`` → ``"meta-llama/Llama-3-70B-Instruct"``
        """
        import re

        return re.sub(
            r"[-_](?:FP8|FP4|NVFP4|GPTQ|AWQ|INT8|INT4|W4A16|W8A8|quantized\.\w+)$",
            "",
            repo_id,
            flags=re.IGNORECASE,
        )

    # ------------------------------------------------------------------
    # Evidence construction
    # ------------------------------------------------------------------

    def _parse_recipe_to_evidence(self, recipe_data: dict[str, Any], repo_id: str) -> list[EvidenceItem]:
        """Parse raw recipe data into EvidenceItem objects.

        Each distinct claim becomes a separate EvidenceItem so downstream
        consumers can reason about individual aspects.
        """
        now = datetime.now(UTC)
        evidence: list[EvidenceItem] = []

        model_block = recipe_data.get("model", recipe_data)
        meta_block = recipe_data.get("meta", {})
        model_id = model_block.get("model_id", repo_id)
        source_url = f"{RECIPES_BASE_URL}/{model_id}"

        common = {
            "category": "recipe",
            "source_url": source_url,
            "source_domain": "recipes.vllm.ai",
            "publisher": "vLLM Project",
            "retrieved_at": now,
            "source_tier": "primary",
            "verification_level": "verified",
        }

        min_vllm = model_block.get("min_vllm_version")
        if min_vllm:
            evidence.append(
                EvidenceItem(
                    claim_type="compatibility",
                    title=f"Minimum vLLM version for {model_id}",
                    summary=f"Requires vLLM >= {min_vllm}.",
                    vllm_version=min_vllm,
                    raw_excerpt_hash=_excerpt_hash(f"min_vllm_version:{min_vllm}"),
                    **common,
                )
            )

        hardware = meta_block.get("hardware") or recipe_data.get("hardware") or {}
        if hardware:
            verified_hw = [k for k, v in hardware.items() if v == "verified"]
            if verified_hw:
                hw_sig = ", ".join(verified_hw)
                evidence.append(
                    EvidenceItem(
                        claim_type="tested_hardware",
                        title=f"Verified hardware for {model_id}",
                        summary=f"Tested and verified on: {hw_sig}.",
                        hardware_signature=hw_sig,
                        vllm_version=min_vllm,
                        raw_excerpt_hash=_excerpt_hash(f"hardware:{hw_sig}"),
                        **common,
                    )
                )

        base_args = model_block.get("base_args") or []
        if base_args:
            args_str = " ".join(base_args) if isinstance(base_args, list) else str(base_args)
            evidence.append(
                EvidenceItem(
                    claim_type="compatibility",
                    title=f"Required vLLM arguments for {model_id}",
                    summary=f"Base launch arguments: {args_str}",
                    vllm_version=min_vllm,
                    raw_excerpt_hash=_excerpt_hash(f"base_args:{args_str}"),
                    **common,
                )
            )

        base_env = model_block.get("base_env") or {}
        if base_env:
            env_str = " ".join(f"{k}={v}" for k, v in base_env.items())
            evidence.append(
                EvidenceItem(
                    claim_type="compatibility",
                    title=f"Required environment variables for {model_id}",
                    summary=f"Environment: {env_str}",
                    vllm_version=min_vllm,
                    raw_excerpt_hash=_excerpt_hash(f"base_env:{env_str}"),
                    **common,
                )
            )

        compatible_strategies = recipe_data.get("compatible_strategies") or []
        if compatible_strategies:
            strats = ", ".join(compatible_strategies)
            evidence.append(
                EvidenceItem(
                    claim_type="compatibility",
                    title=f"Compatible deployment strategies for {model_id}",
                    summary=f"Supported strategies: {strats}.",
                    vllm_version=min_vllm,
                    raw_excerpt_hash=_excerpt_hash(f"strategies:{strats}"),
                    **common,
                )
            )

        features = recipe_data.get("features") or {}
        for feature_key, feature_val in features.items():
            desc = feature_val.get("description", feature_key) if isinstance(feature_val, dict) else str(feature_val)
            args = feature_val.get("args", []) if isinstance(feature_val, dict) else []
            summary_parts = [desc]
            if args:
                summary_parts.append(f"Args: {' '.join(args)}")
            evidence.append(
                EvidenceItem(
                    claim_type="compatibility",
                    title=f"Feature '{feature_key}' for {model_id}",
                    summary=" | ".join(summary_parts),
                    vllm_version=min_vllm,
                    raw_excerpt_hash=_excerpt_hash(f"feature:{feature_key}:{desc}"),
                    **common,
                )
            )

        variants = recipe_data.get("variants") or {}
        for variant_name, variant_val in variants.items():
            if not isinstance(variant_val, dict):
                continue
            precision = variant_val.get("precision", "unknown")
            vram = variant_val.get("vram_minimum_gb")
            variant_model = variant_val.get("model_id", model_id)
            parts = [f"Precision: {precision}"]
            if vram is not None:
                parts.append(f"Minimum VRAM: {vram} GB")
            if variant_val.get("description"):
                parts.append(variant_val["description"])
            evidence.append(
                EvidenceItem(
                    claim_type="compatibility",
                    title=f"Variant '{variant_name}' for {variant_model}",
                    summary=" | ".join(parts),
                    vllm_version=min_vllm,
                    raw_excerpt_hash=_excerpt_hash(f"variant:{variant_name}:{precision}:{vram}"),
                    **common,
                )
            )

        architecture = model_block.get("architecture")
        param_count = model_block.get("parameter_count")
        ctx_len = model_block.get("context_length")
        if architecture or param_count:
            parts = []
            if architecture:
                parts.append(f"Architecture: {architecture}")
            if param_count:
                parts.append(f"Parameters: {param_count}")
            if ctx_len is not None:
                parts.append(f"Context length: {ctx_len}")
            evidence.append(
                EvidenceItem(
                    claim_type="architecture",
                    title=f"Model architecture for {model_id}",
                    summary=" | ".join(parts),
                    vllm_version=min_vllm,
                    raw_excerpt_hash=_excerpt_hash(f"arch:{architecture}:{param_count}"),
                    **common,
                )
            )

        hw_overrides = recipe_data.get("hardware_overrides") or {}
        for gen_name, override in hw_overrides.items():
            if not isinstance(override, dict):
                continue
            extra_args = override.get("extra_args", [])
            extra_env = override.get("extra_env", {})
            parts = []
            if extra_args:
                parts.append(f"Extra args: {' '.join(extra_args)}")
            if extra_env:
                parts.append(f"Extra env: {' '.join(f'{k}={v}' for k, v in extra_env.items())}")
            if parts:
                evidence.append(
                    EvidenceItem(
                        claim_type="compatibility",
                        title=f"Hardware override ({gen_name}) for {model_id}",
                        summary=" | ".join(parts),
                        hardware_signature=gen_name,
                        vllm_version=min_vllm,
                        raw_excerpt_hash=_excerpt_hash(f"hw_override:{gen_name}:{parts}"),
                        **common,
                    )
                )

        if not evidence:
            title = meta_block.get("title") or model_id
            description = meta_block.get("description", "")
            evidence.append(
                EvidenceItem(
                    claim_type="compatibility",
                    title=f"vLLM recipe available for {title}",
                    summary=description or f"A vLLM deployment recipe exists for {model_id}.",
                    vllm_version=min_vllm,
                    raw_excerpt_hash=_excerpt_hash(f"generic:{model_id}"),
                    **common,
                )
            )

        return evidence
