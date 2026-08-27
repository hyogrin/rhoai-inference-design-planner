"""Model analysis node — LLM interprets raw HF config into structured parameters.

Produces a standardized model analysis (precision, KV layout, weight memory)
that the user confirms/edits before deterministic sizing runs.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

import httpx
from pydantic import BaseModel

from agents.inference_planner.state import PlannerState

logger = logging.getLogger(__name__)

MODEL_ANALYSIS_SYSTEM = """\
You are a model inference expert. Given a HuggingFace model's architecture metadata, \
determine the key parameters needed for GPU memory estimation.

Output ONLY valid JSON matching the schema below. No commentary outside the JSON."""

MODEL_ANALYSIS_USER = """\
Analyze this model for inference memory estimation:

Model: {model_repo_id}
Architecture metadata:
{architecture_json}

Determine:
1. **weight_precision**: The actual weight storage format (e.g. "BF16", "FP8", "NVFP4 mixed (experts FP4, rest BF16)", "INT4 (GPTQ)")
2. **effective_bits**: The dominant bit-width for the majority of model weights (4, 8, or 16)
3. **kv_layout**: How KV cache is structured — one of: "standard_gqa", "mla", "hybrid_sliding"
   - "standard_gqa": regular GQA/MHA (2 × layers × kv_heads × head_dim per token)
   - "mla": Multi-head Latent Attention with kv_lora_rank (DeepSeek-V3, GLM-5.2)
   - "hybrid_sliding": mix of sliding-window and full-attention layers (Gemma 4)
4. **kv_cache_bytes_per_element**: bytes per KV element at inference (1 for FP8, 2 for BF16)
   - Quantized models (≤8 bit weights) typically use FP8 KV cache (1 byte)
   - Full-precision models use BF16 KV cache (2 bytes)
5. **explanation**: 1-2 sentence reasoning for your choices

Output JSON:
{{"weight_precision": "...", "effective_bits": N, "kv_layout": "...", "kv_cache_bytes_per_element": N, "explanation": "..."}}"""


class ModelAnalysisResult(BaseModel):
    weight_precision: str
    effective_bits: int
    kv_layout: Literal["standard_gqa", "mla", "hybrid_sliding"]
    kv_cache_bytes_per_element: int
    explanation: str


def _fallback_analysis(arch: dict[str, Any]) -> dict[str, Any]:
    """Deterministic fallback when LLM is unavailable."""
    weight_precision = arch.get("weight_precision") or ""
    quant_method = arch.get("quantization_method") or ""

    # Determine effective bits
    wp_lower = weight_precision.lower()
    qm_lower = quant_method.lower()
    if any(t in wp_lower for t in ["float4", "fp4", "nvfp4", "nf4"]) or "4" in wp_lower:
        effective_bits = 4
    elif any(t in wp_lower for t in ["float8", "fp8", "int8"]) or "8" in wp_lower:
        effective_bits = 8
    elif qm_lower in ("gptq", "awq"):
        effective_bits = 4
    elif qm_lower and qm_lower != "none":
        effective_bits = 8
    else:
        effective_bits = 16

    # KV layout
    if arch.get("kv_lora_rank") and arch.get("qk_rope_head_dim"):
        kv_layout = "mla"
    elif arch.get("sliding_attention_layers") and arch.get("full_attention_layers"):
        kv_layout = "hybrid_sliding"
    else:
        kv_layout = "standard_gqa"

    # KV bytes
    kv_cache_bytes = 1 if effective_bits <= 8 else 2

    precision_label = weight_precision or (f"{'BF' if effective_bits == 16 else 'FP'}{effective_bits}")

    return {
        "weight_precision": precision_label,
        "effective_bits": effective_bits,
        "kv_layout": kv_layout,
        "kv_cache_bytes_per_element": kv_cache_bytes,
        "explanation": f"Fallback: structured fields (quant={quant_method or 'none'}).",
    }


async def interpret_model_config(state: PlannerState) -> dict[str, Any]:
    """LLM interprets raw HF architecture config into structured analysis."""
    from backend.config import get_settings

    settings = get_settings()
    api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = settings.openai_base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = settings.llm_model_name or os.environ.get("LLM_MODEL_NAME", "gpt-4o")
    verify_ssl = settings.verify_ssl

    arch = state.get("model_architecture") or {}
    model_repo_id = state.get("model_repo_id", "")

    if not api_key:
        logger.warning("No LLM API key — using fallback model analysis")
        return {
            "model_analysis": _fallback_analysis(arch),
            "phase_history": ["model_analysis_fallback"],
        }

    # Build a concise architecture summary for the LLM (exclude nulls)
    arch_for_prompt = {k: v for k, v in arch.items() if v is not None}
    arch_json = json.dumps(arch_for_prompt, indent=2, default=str)

    user_prompt = MODEL_ANALYSIS_USER.format(
        model_repo_id=model_repo_id,
        architecture_json=arch_json,
    )

    try:
        async with httpx.AsyncClient(verify=verify_ssl, timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": MODEL_ANALYSIS_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)

            # Validate with Pydantic
            result = ModelAnalysisResult(**parsed)
            logger.info("LLM model analysis: %s", result.model_dump())

            return {
                "model_analysis": result.model_dump(),
                "phase_history": ["model_analysis_completed"],
            }
    except Exception as exc:
        logger.warning("LLM model analysis failed (%s) — using fallback", exc)
        return {
            "model_analysis": _fallback_analysis(arch),
            "phase_history": ["model_analysis_fallback"],
        }
