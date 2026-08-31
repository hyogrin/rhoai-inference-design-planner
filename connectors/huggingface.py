"""Hugging Face model metadata connector.

Fetches and normalizes model architecture metadata from Hugging Face Hub.
Does NOT execute remote code or download full model weights.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from functools import partial
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from domain.model import ModelArchitecture, ModelIdentity

logger = logging.getLogger(__name__)

_FAMILY_PATTERNS: dict[str, list[str]] = {
    "llama": ["LlamaForCausalLM", "MistralForCausalLM"],
    "mixtral": ["MixtralForCausalLM"],
    "deepseek": ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"],
    "qwen": [
        "Qwen2ForCausalLM",
        "Qwen2MoeForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
    ],
    "gemma": ["Gemma2ForCausalLM", "GemmaForCausalLM"],
    "phi": ["PhiForCausalLM", "Phi3ForCausalLM", "PhiMoEForCausalLM"],
    "jamba": ["JambaForCausalLM"],
    "llava": ["LlavaForConditionalGeneration", "LlavaNextForConditionalGeneration"],
    "granite": ["GraniteForCausalLM", "GraniteMoeForCausalLM"],
}

_MOE_ARCHITECTURES: set[str] = {
    "MixtralForCausalLM",
    "Qwen2MoeForCausalLM",
    "Qwen3MoeForCausalLM",
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "PhiMoEForCausalLM",
    "GraniteMoeForCausalLM",
    "ArcticForCausalLM",
    "DbrxForCausalLM",
}

_MULTIMODAL_ARCHITECTURES: set[str] = {
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "InternVLChatModel",
    "PaliGemmaForConditionalGeneration",
    "MllamaForConditionalGeneration",
}

_HYBRID_ARCHITECTURES: set[str] = {
    "JambaForCausalLM",
    "ZambaForCausalLM",
}

_MLA_ARCHITECTURES: set[str] = {
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
}

_NESTED_CONFIG_KEYS = ("text_config", "language_config", "llm_config", "text_decoder")

_CRITICAL_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "intermediate_size",
    "max_position_embeddings",
)


class HuggingFaceConnectorError(Exception):
    """Base exception for HuggingFace connector errors."""


class ModelNotFoundError(HuggingFaceConnectorError):
    """Raised when the requested model does not exist."""


class HuggingFaceConnector:
    """Fetches and parses model metadata from Hugging Face Hub."""

    def __init__(self, token: str | None = None):
        self._api = HfApi(token=token or None)
        self._token = token or None

    async def fetch_model_identity(
        self, repo_id: str, revision: str = "main"
    ) -> ModelIdentity:
        """Fetch basic model identity and metadata."""
        loop = asyncio.get_running_loop()
        gated = False

        try:
            info = await loop.run_in_executor(
                None, partial(self._api.model_info, repo_id, revision=revision)
            )
        except GatedRepoError:
            gated = True
            info = await loop.run_in_executor(
                None, partial(self._api.model_info, repo_id, revision=revision)
            )
        except RepositoryNotFoundError as exc:
            raise ModelNotFoundError(
                f"Model repository '{repo_id}' not found"
            ) from exc
        except RevisionNotFoundError as exc:
            raise ModelNotFoundError(
                f"Revision '{revision}' not found for '{repo_id}'"
            ) from exc

        pipeline_tag = getattr(info, "pipeline_tag", None)
        tasks: list[str] = []
        if pipeline_tag:
            tasks.append(pipeline_tag)

        tags = getattr(info, "tags", []) or []
        for tag in tags:
            if tag.startswith("text-generation") and tag not in tasks:
                tasks.append(tag)

        license_value = None
        card_data = getattr(info, "card_data", None)
        if card_data and hasattr(card_data, "license"):
            license_value = card_data.license
        if not license_value:
            for tag in tags:
                if tag.startswith("license:"):
                    license_value = tag.removeprefix("license:")
                    break

        return ModelIdentity(
            repo_id=repo_id,
            revision=revision,
            resolved_commit_sha=getattr(info, "sha", None),
            gated=gated or bool(getattr(info, "gated", False)),
            private=bool(getattr(info, "private", False)),
            license=license_value,
            pipeline_tag=pipeline_tag,
            tasks=tasks,
            source_url=f"https://huggingface.co/{repo_id}/tree/{revision}",
            fetched_at=datetime.now(UTC),
        )

    async def fetch_model_architecture(
        self, repo_id: str, revision: str = "main"
    ) -> ModelArchitecture:
        """Fetch and parse model architecture from config files."""
        loop = asyncio.get_running_loop()

        config = await self._download_json(repo_id, "config.json", revision)
        if config is None:
            logger.warning("No config.json found for %s@%s", repo_id, revision)
            return ModelArchitecture(
                parser_confidence=0.0,
                missing_fields=list(_CRITICAL_FIELDS),
            )

        raw_config_paths = ["config.json"]

        gen_config = await self._download_json(
            repo_id, "generation_config.json", revision
        )
        if gen_config:
            raw_config_paths.append("generation_config.json")

        arch = self._parse_config(config, repo_id)

        total_params, params_by_dtype = await loop.run_in_executor(
            None, partial(self._get_parameter_counts, repo_id, revision)
        )
        if total_params is not None:
            arch = arch.model_copy(update={"parameter_count_total": total_params})
        if params_by_dtype is not None:
            arch = arch.model_copy(update={"parameter_count_by_dtype": params_by_dtype})

        if arch.parameter_count_total is None:
            info_params = await self._get_params_from_model_info(repo_id, revision)
            if info_params is not None:
                arch = arch.model_copy(update={"parameter_count_total": info_params})

        # Get checkpoint size from safetensors metadata or model_info
        checkpoint_size = await self._get_checkpoint_size(repo_id, revision, loop)
        if checkpoint_size:
            arch = arch.model_copy(update={"checkpoint_size_bytes": checkpoint_size})

        if gen_config and arch.max_position_embeddings is None and "max_length" in gen_config:
            arch = arch.model_copy(
                update={"max_position_embeddings": gen_config["max_length"]}
            )

        quant_method, weight_precision = self._detect_quantization(config, repo_id)
        arch = arch.model_copy(
            update={
                "quantization_method": quant_method or arch.quantization_method,
                "weight_precision": weight_precision or arch.weight_precision,
                "raw_config_paths": raw_config_paths,
            }
        )

        missing = self._compute_missing_fields(arch)
        confidence = self._compute_confidence(arch, missing)
        arch = arch.model_copy(
            update={"parser_confidence": confidence, "missing_fields": missing}
        )

        return arch

    def _parse_config(self, config: dict[str, Any], repo_id: str) -> ModelArchitecture:
        """Parse a config.json into ModelArchitecture."""
        nested = self._get_nested_config(config)
        effective = nested if nested else config

        arch_names = config.get("architectures", []) or []
        if not arch_names and "model_type" in config:
            arch_names = [config["model_type"]]

        architecture_type = self._detect_architecture_type(arch_names)
        family = self._detect_family(arch_names, repo_id)
        kv_layout = self._detect_kv_layout(arch_names, effective)

        num_hidden_layers = effective.get("num_hidden_layers")
        hidden_size = effective.get("hidden_size")
        intermediate_size = effective.get("intermediate_size")
        num_attention_heads = effective.get("num_attention_heads")
        num_kv_heads = (
            effective.get("num_key_value_heads")
            or effective.get("num_kv_heads")
        )
        head_dim = effective.get("head_dim")
        if head_dim is None and hidden_size and num_attention_heads:
            head_dim = hidden_size // num_attention_heads

        max_position_embeddings = (
            effective.get("max_position_embeddings")
            or effective.get("max_sequence_length")
        )
        sliding_window = effective.get("sliding_window")

        # MLA (Multi-head Latent Attention) — DeepSeek-V3, GLM-5.2, etc.
        kv_lora_rank = effective.get("kv_lora_rank")
        qk_rope_head_dim = effective.get("qk_rope_head_dim")

        num_experts_total = (
            effective.get("num_local_experts")
            or effective.get("num_experts")
            or effective.get("n_routed_experts")
        )
        num_experts_active = (
            effective.get("num_experts_per_tok")
            or effective.get("num_experts_per_token")
            or effective.get("top_k")
        )
        expert_intermediate_size = effective.get("expert_intermediate_size")
        if expert_intermediate_size is None and num_experts_total:
            moe_intermediate = effective.get("moe_intermediate_size")
            if moe_intermediate:
                expert_intermediate_size = moe_intermediate

        attention_layer_count = None
        linear_attention_layer_count = None
        state_space_layer_count = None
        sliding_attention_layers = None
        full_attention_layers = None
        global_head_dim = None
        num_global_kv_heads = None

        # Detect hybrid attention from layer_types (Gemma 4 style)
        layer_types = effective.get("layer_types", [])
        if layer_types and num_hidden_layers:
            from collections import Counter
            type_counts = Counter(str(t).lower() for t in layer_types)
            sliding_count = sum(
                v for k, v in type_counts.items()
                if "sliding" in k
            )
            full_count = sum(
                v for k, v in type_counts.items()
                if "full" in k or "global" in k
            )
            if sliding_count > 0 and full_count > 0:
                sliding_attention_layers = sliding_count
                full_attention_layers = full_count
                global_head_dim = effective.get("global_head_dim")
                num_global_kv_heads = effective.get("num_global_key_value_heads")

        if architecture_type == "hybrid":
            attn_pattern = effective.get("attn_layer_period") or effective.get(
                "attention_layer_period"
            )
            if attn_pattern and num_hidden_layers:
                attention_layer_count = num_hidden_layers // attn_pattern
                state_space_layer_count = num_hidden_layers - attention_layer_count
            else:
                block_types = effective.get("layers_block_type", [])
                if block_types:
                    attention_layer_count = sum(
                        1 for t in block_types if "attention" in t.lower()
                    )
                    state_space_layer_count = sum(
                        1 for t in block_types if "mamba" in t.lower() or "ssm" in t.lower()
                    )

        vision_encoder_parameters = None
        if architecture_type == "multimodal":
            vision_config = config.get("vision_config", {})
            if isinstance(vision_config, dict) and vision_config:
                vision_encoder_parameters = self._estimate_vision_params(vision_config)

        weight_format = self._detect_weight_format(config)
        kv_formula = self._compute_kv_formula(
            kv_layout, num_kv_heads, head_dim, num_hidden_layers
        )

        return ModelArchitecture(
            architecture_names=arch_names,
            family=family,
            architecture_type=architecture_type,
            weight_format=weight_format,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            sliding_window=sliding_window,
            sliding_attention_layers=sliding_attention_layers,
            full_attention_layers=full_attention_layers,
            global_head_dim=global_head_dim,
            num_global_kv_heads=num_global_kv_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            attention_layer_count=attention_layer_count,
            linear_attention_layer_count=linear_attention_layer_count,
            state_space_layer_count=state_space_layer_count,
            num_experts_total=num_experts_total,
            num_experts_active=num_experts_active,
            expert_intermediate_size=expert_intermediate_size,
            vision_encoder_parameters=vision_encoder_parameters,
            kv_layout=kv_layout,
            kv_bytes_per_token_formula=kv_formula,
            parser_confidence=0.0,
            missing_fields=[],
            raw_config_paths=[],
        )

    def _detect_architecture_type(self, arch_names: list[str]) -> str:
        """Detect architecture type from architecture class names."""
        for name in arch_names:
            if name in _MULTIMODAL_ARCHITECTURES:
                return "multimodal"
            if name in _HYBRID_ARCHITECTURES:
                return "hybrid"
            if name in _MOE_ARCHITECTURES:
                return "moe"
        # Pattern-based fallback for unlisted architectures
        import re
        for name in arch_names:
            lower = name.lower()
            if re.search(r"(moe|expert|exp(?=for))", lower):
                return "moe"
            if "conditional" in lower and any(k in lower for k in ("vl", "vision", "llava", "image")):
                return "multimodal"
        if arch_names:
            return "dense"
        return "unknown"

    def _detect_family(self, arch_names: list[str], repo_id: str) -> str | None:
        """Detect model family from architecture names or repo_id."""
        for family, patterns in _FAMILY_PATTERNS.items():
            for name in arch_names:
                if name in patterns:
                    return family

        repo_lower = repo_id.lower()
        for family in _FAMILY_PATTERNS:
            if family in repo_lower:
                return family
        return None

    def _detect_kv_layout(self, arch_names: list[str], config: dict[str, Any]) -> str:
        """Detect KV cache layout."""
        for name in arch_names:
            if name in _MLA_ARCHITECTURES:
                return "mla"

        for name in arch_names:
            if name in _HYBRID_ARCHITECTURES:
                return "hybrid"

        num_attention_heads = config.get("num_attention_heads")
        num_kv_heads = config.get("num_key_value_heads") or config.get("num_kv_heads")

        if num_attention_heads and num_kv_heads:
            return "mha_gqa"

        if num_attention_heads:
            return "mha_gqa"

        return "unknown"

    def _get_nested_config(self, config: dict[str, Any]) -> dict[str, Any] | None:
        """Extract the primary text/language config from nested multimodal configs."""
        for key in _NESTED_CONFIG_KEYS:
            nested = config.get(key)
            if isinstance(nested, dict) and nested:
                logger.debug("Using nested config key: %s", key)
                return nested
        return None

    def _get_parameter_counts(
        self, repo_id: str, revision: str
    ) -> tuple[int | None, dict[str, int] | None]:
        """Get parameter counts from safetensors metadata (preferred source)."""
        try:
            info = self._api.model_info(repo_id, revision=revision, files_metadata=True)
        except (RepositoryNotFoundError, GatedRepoError, HfHubHTTPError):
            return None, None

        safetensors = getattr(info, "safetensors", None)
        if safetensors is None:
            return None, None

        params_info = None
        if hasattr(safetensors, "parameters"):
            params_info = safetensors.parameters
        elif isinstance(safetensors, dict):
            params_info = safetensors.get("parameters")

        if not params_info:
            if hasattr(safetensors, "parameter_count"):
                params_info = safetensors.parameter_count
            elif isinstance(safetensors, dict):
                params_info = safetensors.get("parameter_count")

        if not params_info:
            return None, None

        if isinstance(params_info, dict):
            params_by_dtype = {k: int(v) for k, v in params_info.items()}
            total = sum(params_by_dtype.values())
            return total, params_by_dtype

        if isinstance(params_info, int):
            return params_info, None

        return None, None

    async def _get_params_from_model_info(
        self, repo_id: str, revision: str
    ) -> int | None:
        """Fallback: get parameter count from model card or metadata tags."""
        loop = asyncio.get_running_loop()
        try:
            info = await loop.run_in_executor(
                None, partial(self._api.model_info, repo_id, revision=revision)
            )
        except (RepositoryNotFoundError, GatedRepoError, HfHubHTTPError):
            return None

        tags = getattr(info, "tags", []) or []
        for tag in tags:
            if tag.startswith("params:"):
                try:
                    return self._parse_param_tag(tag.removeprefix("params:"))
                except ValueError:
                    continue

        card_data = getattr(info, "card_data", None)
        if card_data:
            params = getattr(card_data, "parameters", None)
            if params and isinstance(params, (int, float)):
                return int(params)

        return None

    def _parse_param_tag(self, value: str) -> int:
        """Parse a parameter count tag like '7B', '70B', '1.5B'."""
        value = value.strip().upper()
        multipliers = {"B": 1_000_000_000, "M": 1_000_000, "K": 1_000, "T": 1_000_000_000_000}
        for suffix, mult in multipliers.items():
            if value.endswith(suffix):
                return int(float(value.removesuffix(suffix)) * mult)
        return int(float(value))

    def _detect_quantization(
        self, config: dict[str, Any], repo_id: str
    ) -> tuple[str | None, str | None]:
        """Detect quantization method and weight precision."""
        quant_config = config.get("quantization_config", {})
        if isinstance(quant_config, dict) and quant_config:
            method = quant_config.get("quant_method")
            bits = quant_config.get("bits")
            # Detect NVFP4 / mixed precision from config
            if method == "fp8" and "nvfp4" in repo_id.lower():
                return "nvfp4_fp8_mixed", "nvfp4"
            if method == "compressed-tensors":
                # llm-compressor style: check config_groups for actual bits
                config_groups = quant_config.get("config_groups", {})
                precisions = set()
                for group_info in config_groups.values():
                    if isinstance(group_info, dict):
                        w_scheme = group_info.get("weights", {})
                        if isinstance(w_scheme, dict):
                            wbits = w_scheme.get("num_bits")
                            wtype = w_scheme.get("type", "")
                            if wbits:
                                precisions.add(f"{wtype}{wbits}" if wtype else f"int{wbits}")
                if precisions:
                    return method, "+".join(sorted(precisions))
            precision = f"int{bits}" if bits else None
            return method, precision

        repo_lower = repo_id.lower()
        quant_patterns = {
            "nvfp4": ("nvfp4", "nvfp4"),
            "awq": ("awq", "int4"),
            "gptq": ("gptq", "int4"),
            "gguf": ("gguf", None),
            "int8": (None, "int8"),
            "int4": (None, "int4"),
            "fp8": (None, "fp8"),
            "bnb": ("bitsandbytes", None),
        }
        for pattern, (method, precision) in quant_patterns.items():
            if pattern in repo_lower:
                return method, precision

        return None, None

    async def _get_checkpoint_size(
        self, repo_id: str, revision: str, loop: Any
    ) -> int | None:
        """Get total checkpoint file size from model_info siblings."""
        try:
            from huggingface_hub import model_info as get_model_info
            info = await loop.run_in_executor(
                None,
                partial(
                    get_model_info, repo_id, revision=revision,
                    token=self._token, files_metadata=True,
                ),
            )
            total = 0
            siblings = getattr(info, "siblings", []) or []
            for sibling in siblings:
                fname = getattr(sibling, "rfilename", "") or ""
                if fname.endswith((".safetensors", ".bin")):
                    size = getattr(sibling, "size", None)
                    if size and isinstance(size, int):
                        total += size
            return total if total > 0 else None
        except Exception:
            return None

    def _detect_weight_format(self, config: dict[str, Any]) -> str | None:
        """Detect weight storage format from config."""
        torch_dtype = config.get("torch_dtype")
        if torch_dtype:
            return str(torch_dtype)
        return None

    def _estimate_vision_params(self, vision_config: dict[str, Any]) -> int | None:
        """Rough estimate of vision encoder parameters from its config."""
        hidden = vision_config.get("hidden_size")
        layers = vision_config.get("num_hidden_layers")
        intermediate = vision_config.get("intermediate_size")
        if not (hidden and layers and intermediate):
            return None
        params_per_layer = (
            4 * hidden * hidden  # attention QKV + output
            + 2 * hidden * intermediate  # MLP up + down
        )
        return layers * params_per_layer

    def _compute_kv_formula(
        self,
        kv_layout: str,
        num_kv_heads: int | None,
        head_dim: int | None,
        num_layers: int | None,
    ) -> str | None:
        """Compute a symbolic formula for KV cache bytes per token."""
        if kv_layout == "mla":
            return "compressed_kv (model-specific)"
        if not (num_kv_heads and head_dim and num_layers):
            return None
        kv_size = 2 * num_kv_heads * head_dim * num_layers
        return f"2 * {num_kv_heads} * {head_dim} * {num_layers} * dtype_bytes = {kv_size} * dtype_bytes"

    def _compute_missing_fields(self, arch: ModelArchitecture) -> list[str]:
        """List critical fields that could not be determined."""
        missing: list[str] = []
        for field_name in _CRITICAL_FIELDS:
            if getattr(arch, field_name, None) is None:
                missing.append(field_name)
        if arch.parameter_count_total is None:
            missing.append("parameter_count_total")
        if arch.num_kv_heads is None:
            missing.append("num_kv_heads")
        return missing

    def _compute_confidence(
        self, arch: ModelArchitecture, missing: list[str]
    ) -> float:
        """Compute parser confidence [0, 1] based on filled critical fields."""
        all_fields = list(_CRITICAL_FIELDS) + ["parameter_count_total", "num_kv_heads"]
        total = len(all_fields)
        found = total - len(missing)
        base_confidence = found / total if total else 0.0

        if arch.architecture_type != "unknown":
            base_confidence = min(1.0, base_confidence + 0.05)
        if arch.family:
            base_confidence = min(1.0, base_confidence + 0.05)

        return round(base_confidence, 3)

    async def _download_json(
        self, repo_id: str, filename: str, revision: str
    ) -> dict[str, Any] | None:
        """Download and parse a JSON file from the repo."""
        loop = asyncio.get_running_loop()
        try:
            path = await loop.run_in_executor(
                None,
                partial(
                    hf_hub_download,
                    repo_id=repo_id,
                    filename=filename,
                    revision=revision,
                    token=self._token,
                ),
            )
        except EntryNotFoundError:
            logger.debug("%s not found in %s@%s", filename, repo_id, revision)
            return None
        except GatedRepoError:
            logger.warning("Gated repo: cannot download %s from %s", filename, repo_id)
            return None
        except RepositoryNotFoundError:
            return None
        except (RevisionNotFoundError, HfHubHTTPError) as exc:
            logger.warning("Failed to download %s from %s: %s", filename, repo_id, exc)
            return None

        try:
            with open(path, encoding="utf-8") as f:
                return json.loads(f.read())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to parse %s from %s: %s", filename, repo_id, exc)
            return None
