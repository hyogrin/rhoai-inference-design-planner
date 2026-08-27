"""Tests for Phase 2 connectors.

These tests use local fixtures and mocked HTTP calls to verify connector logic
without requiring network access or live API tokens.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from connectors.community_search import CommunitySearchConnector
from connectors.huggingface import HuggingFaceConnector
from connectors.pricing import PricingConnector
from connectors.redhat_model_cards import RedHatModelCardConnector
from connectors.rhoai_compatibility import RhoaiCompatibilityConnector
from connectors.vllm_recipes import VllmRecipeConnector

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MODELS_DIR = FIXTURES_DIR / "models"
EVIDENCE_DIR = FIXTURES_DIR / "evidence"


class TestHuggingFaceConnectorParsing:
    """Test HuggingFaceConnector config parsing (no network needed)."""

    def setup_method(self):
        self.connector = HuggingFaceConnector(token=None)

    def test_parse_llama3_1_70b(self):
        config = json.loads((MODELS_DIR / "llama3_1_70b_config.json").read_text())
        arch = self.connector._parse_config(config, "meta-llama/Llama-3.1-70B-Instruct")

        assert arch.architecture_type == "dense"
        assert arch.family == "llama"
        assert arch.architecture_names == ["LlamaForCausalLM"]
        assert arch.num_hidden_layers == 80
        assert arch.hidden_size == 8192
        assert arch.intermediate_size == 28672
        assert arch.num_attention_heads == 64
        assert arch.num_kv_heads == 8
        assert arch.head_dim == 128
        assert arch.max_position_embeddings == 131072
        assert arch.kv_layout == "mha_gqa"
        assert arch.kv_bytes_per_token_formula is not None

    def test_parse_mixtral_8x7b(self):
        config = json.loads((MODELS_DIR / "mixtral_8x7b_config.json").read_text())
        arch = self.connector._parse_config(config, "mistralai/Mixtral-8x7B-Instruct-v0.1")

        assert arch.architecture_type == "moe"
        assert arch.family == "mixtral"
        assert arch.num_experts_total == 8
        assert arch.num_experts_active == 2
        assert arch.num_hidden_layers == 32
        assert arch.hidden_size == 4096
        assert arch.num_kv_heads == 8
        assert arch.sliding_window == 4096
        assert arch.kv_layout == "mha_gqa"

    def test_parse_deepseek_v3(self):
        config = json.loads((MODELS_DIR / "deepseek_v3_config.json").read_text())
        arch = self.connector._parse_config(config, "deepseek-ai/DeepSeek-V3")

        assert arch.architecture_type == "moe"
        assert arch.family == "deepseek"
        assert arch.num_experts_total == 256
        assert arch.num_experts_active == 8
        assert arch.expert_intermediate_size == 2048
        assert arch.num_hidden_layers == 61
        assert arch.hidden_size == 7168
        assert arch.kv_layout == "mla"

    def test_parse_qwen2_vl(self):
        config = json.loads((MODELS_DIR / "qwen2_vl_config.json").read_text())
        arch = self.connector._parse_config(config, "Qwen/Qwen2-VL-72B-Instruct")

        assert arch.architecture_type == "multimodal"
        assert arch.hidden_size == 8192
        assert arch.num_hidden_layers == 80
        assert arch.num_attention_heads == 64
        assert arch.num_kv_heads == 8
        assert arch.vision_encoder_parameters is not None
        assert arch.vision_encoder_parameters > 0

    def test_unknown_architecture_defaults_to_dense(self):
        """When architectures is empty and model_type is unknown, defaults to dense."""
        config = {"model_type": "some_new_type", "architectures": []}
        arch = self.connector._parse_config(config, "some-org/some-model")
        assert arch.architecture_type == "dense"
        assert arch.family is None

    def test_truly_empty_architecture(self):
        """When no architectures and no model_type, architecture_type is unknown."""
        config = {}
        arch = self.connector._parse_config(config, "some-org/some-model")
        assert arch.architecture_type == "unknown"

    def test_parser_confidence_calculation(self):
        config = json.loads((MODELS_DIR / "llama3_1_70b_config.json").read_text())
        arch = self.connector._parse_config(config, "meta-llama/Llama-3.1-70B-Instruct")
        missing = self.connector._compute_missing_fields(arch)
        confidence = self.connector._compute_confidence(arch, missing)
        assert confidence >= 0.8

    def test_quantization_detection_from_repo_id(self):
        config = {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}
        method, precision = self.connector._detect_quantization(config, "org/model-AWQ")
        assert method == "awq"
        assert precision == "int4"

    def test_quantization_detection_from_config(self):
        config = {
            "architectures": ["LlamaForCausalLM"],
            "quantization_config": {"quant_method": "gptq", "bits": 4},
        }
        method, precision = self.connector._detect_quantization(config, "org/model")
        assert method == "gptq"
        assert precision == "int4"


class TestPricingConnector:
    """Test PricingConnector pricing lookups."""

    def setup_method(self):
        self.connector = PricingConnector()

    @pytest.mark.asyncio
    async def test_get_pricing_h100_onprem(self):
        evidence = await self.connector.get_pricing_evidence("H100", gpu_count=8, environment_type="on_prem")
        assert len(evidence) > 0
        for item in evidence:
            assert item.category == "pricing"
            assert "H100" in item.hardware_signature or "H100" in item.summary
        tco_items = [e for e in evidence if e.claim_type == "tco"]
        assert len(tco_items) >= 1

    @pytest.mark.asyncio
    async def test_get_pricing_h100_cloud(self):
        evidence = await self.connector.get_pricing_evidence("H100", gpu_count=8, environment_type="aws")
        assert len(evidence) > 0
        for item in evidence:
            assert item.category == "pricing"
            assert item.claim_type == "price"

    @pytest.mark.asyncio
    async def test_get_pricing_a100(self):
        evidence = await self.connector.get_pricing_evidence("A100-80GB", gpu_count=1, environment_type="aws")
        assert len(evidence) > 0

    @pytest.mark.asyncio
    async def test_get_pricing_specific_provider(self):
        evidence = await self.connector.get_pricing_evidence("H100", environment_type="aws", providers=["aws"])
        aws_evidence = [e for e in evidence if "AWS" in e.summary]
        assert len(aws_evidence) > 0

    @pytest.mark.asyncio
    async def test_get_pricing_unknown_gpu(self):
        evidence = await self.connector.get_pricing_evidence("NONEXISTENT_GPU", environment_type="aws")
        assert len(evidence) == 0

    def test_get_gpu_specs(self):
        specs = self.connector.get_gpu_specs("H100")
        assert specs is not None
        assert specs["memory_gb"] == 80
        assert specs["bf16_tflops"] > 0

    def test_get_available_gpus(self):
        gpus = self.connector.get_available_gpus()
        assert "H100-SXM" in gpus
        assert "A100-SXM-80GB" in gpus
        assert "L4" in gpus

    def test_cloud_instances_for_gpu(self):
        instances = self.connector.get_cloud_instances_for_gpu("H100", min_gpu_count=8)
        assert len(instances) >= 2  # at least AWS and GCP
        for inst in instances:
            assert inst["gpu_count"] >= 8


class TestCommunitySearchConnector:
    """Test CommunitySearchConnector logic without network calls."""

    def setup_method(self):
        self.connector = CommunitySearchConnector(mcp_url="http://localhost:9999")

    @pytest.mark.asyncio
    async def test_search_with_mocked_results(self):
        mock_results = [
            {"title": "vLLM Llama 3 deploy", "url": "https://blog.vllm.ai/llama3",
             "content": "Performance results"},
            {"title": "Issue #1234", "url": "https://github.com/vllm-project/vllm/issues/1234",
             "content": "Bug report"},
        ]

        with patch.object(self.connector, "_web_search", new_callable=AsyncMock, return_value=mock_results):
            evidence = await self.connector.search_model_evidence(
                "meta-llama/Llama-3.1-70B-Instruct",
                evidence_types=["compatibility"],
                max_results_per_type=5,
            )

        assert len(evidence) == 2
        for item in evidence:
            assert item.category == "community"
            assert item.verification_level == "reported"

    def test_determine_source_tier_official(self):
        assert self.connector._determine_source_tier("https://docs.vllm.ai/en/latest/") == "primary"
        assert self.connector._determine_source_tier("https://access.redhat.com/articles/123") == "primary"

    def test_determine_source_tier_secondary(self):
        assert self.connector._determine_source_tier("https://huggingface.co/meta-llama") == "official_secondary"
        assert self.connector._determine_source_tier("https://arxiv.org/abs/2301.00000") == "official_secondary"

    def test_determine_source_tier_community(self):
        assert self.connector._determine_source_tier("https://medium.com/article") == "community"
        assert self.connector._determine_source_tier("https://random-blog.com/post") == "community"

    def test_determine_claim_type_from_content(self):
        assert self.connector._determine_claim_type("performance", "throughput 5000 tokens/s") == "serving_performance"
        assert self.connector._determine_claim_type("strengths", "MMLU score 85.2") == "accuracy"
        assert self.connector._determine_claim_type("compatibility", "not supported on older GPUs") == "limitation"

    @pytest.mark.asyncio
    async def test_empty_types_returns_empty(self):
        evidence = await self.connector.search_model_evidence(
            "org/model",
            evidence_types=["nonexistent_type"],
        )
        assert evidence == []


class TestVllmRecipeConnector:
    """Test VllmRecipeConnector parsing logic."""

    def setup_method(self):
        self.connector = VllmRecipeConnector()

    def test_parse_recipe_to_evidence(self):
        recipe_data = json.loads((EVIDENCE_DIR / "vllm_recipe_llama3_1.json").read_text())
        evidence = self.connector._parse_recipe_to_evidence(recipe_data, "meta-llama/Llama-3.1-70B-Instruct")

        assert len(evidence) > 0
        categories = {e.category for e in evidence}
        assert "recipe" in categories

        claim_types = {e.claim_type for e in evidence}
        assert "tested_hardware" in claim_types or "compatibility" in claim_types

        for item in evidence:
            assert item.source_tier == "primary"
            assert item.verification_level == "verified"
            assert "recipes.vllm.ai" in item.source_domain

    def test_strip_quant_suffix(self):
        result = self.connector._strip_quant_suffix("meta-llama/Llama-3-70B-Instruct-FP8")
        assert result == "meta-llama/Llama-3-70B-Instruct"
        assert self.connector._strip_quant_suffix("org/model-AWQ") == "org/model"
        assert self.connector._strip_quant_suffix("org/model-GPTQ") == "org/model"
        assert self.connector._strip_quant_suffix("org/model") == "org/model"

    def test_match_model_exact(self):
        models_data = {
            "models": [
                {"model_id": "meta-llama/Llama-3.1-70B-Instruct", "url": "/recipes/llama"},
                {"model_id": "mistralai/Mixtral-8x7B-v0.1", "url": "/recipes/mixtral"},
            ]
        }
        matches = self.connector._match_model(models_data, "meta-llama/Llama-3.1-70B-Instruct")
        assert len(matches) == 1
        assert matches[0]["model_id"] == "meta-llama/Llama-3.1-70B-Instruct"

    def test_match_model_base_fallback(self):
        models_data = {
            "models": [
                {"model_id": "meta-llama/Llama-3.1-70B-Instruct", "url": "/recipes/llama"},
            ]
        }
        matches = self.connector._match_model(models_data, "meta-llama/Llama-3.1-70B-Instruct-FP8")
        assert len(matches) == 1

    def test_no_match_returns_empty(self):
        models_data = {"models": [{"model_id": "org/other-model", "url": "/x"}]}
        matches = self.connector._match_model(models_data, "meta-llama/Llama-3.1-70B-Instruct")
        assert len(matches) == 0


class TestRedHatModelCardConnector:
    """Test Red Hat model card Markdown parsing."""

    def setup_method(self):
        self.connector = RedHatModelCardConnector(token=None)

    def test_parse_model_card(self):
        card_text = (EVIDENCE_DIR / "redhat_card_nemotron.md").read_text()
        evidence = self.connector._parse_model_card(
            card_text, "RedHatAI/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
        )

        assert len(evidence) > 0
        categories = {e.category for e in evidence}
        assert "redhat_evaluation" in categories

        for item in evidence:
            assert item.source_tier == "official_secondary"
            assert item.verification_level == "verified"

    def test_extract_vllm_version(self):
        card = "This model was evaluated with vLLM 0.8.4 on H100 GPUs."
        version = self.connector._extract_vllm_version(card)
        assert version == "0.8.4"

    def test_extract_hardware(self):
        card = "Evaluated on 2x NVIDIA H100 80GB with tensor_parallel_size=2"
        hw = self.connector._extract_hardware(card)
        assert hw is not None
        assert "2" in hw

    def test_parse_evaluation_table(self):
        table = """| Benchmark | BF16 Score | FP8 Score | Recovery |
|-----------|-----------|-----------|----------|
| MMLU      | 72.1      | 71.8      | 99.6%    |
| ARC-C     | 62.5      | 62.1      | 99.4%    |"""
        rows = self.connector._parse_evaluation_table(table)
        assert len(rows) == 2
        assert rows[0]["Benchmark"] == "MMLU"
        assert rows[0]["BF16 Score"] == "72.1"

    def test_classify_evidence_accuracy(self):
        rows = [{"Benchmark": "MMLU", "Score": "72.1"}]
        result = self.connector._classify_evidence_type("Evaluation Results", rows)
        assert result == "accuracy"

    def test_classify_evidence_serving_performance(self):
        rows = [{"Metric": "TTFT", "p50": "100ms", "p99": "250ms"}]
        result = self.connector._classify_evidence_type("Serving Performance", rows)
        assert result == "serving_performance"


class TestRhoaiCompatibilityConnector:
    """Test RHOAI compatibility checks."""

    def setup_method(self):
        self.connector = RhoaiCompatibilityConnector()

    @pytest.mark.asyncio
    async def test_check_compatibility_known_version(self):
        evidence = await self.connector.check_compatibility("2.18")
        assert len(evidence) > 0
        for item in evidence:
            assert item.category == "platform_compatibility"
            assert item.source_tier == "primary"

    @pytest.mark.asyncio
    async def test_check_compatibility_unknown_version(self):
        evidence = await self.connector.check_compatibility("99.99")
        assert len(evidence) == 1
        assert "Unknown" in evidence[0].title or "not in the known" in evidence[0].summary

    def test_get_vllm_version(self):
        assert self.connector.get_vllm_version_for_rhoai("2.18") == "0.8.4"
        assert self.connector.get_vllm_version_for_rhoai("3.0") == "0.9.1"
        assert self.connector.get_vllm_version_for_rhoai("99.99") is None

    def test_get_feature_status(self):
        assert self.connector.get_feature_status("3.0", "tensor_parallel") == "ga"
        assert self.connector.get_feature_status("2.16", "expert_parallel") == "unsupported"
        assert self.connector.get_feature_status("3.0", "llmd") == "tp"

    def test_vllm_version_compatibility_satisfied(self):
        result = self.connector.check_vllm_version_compatibility("3.0", "0.8.0")
        assert result["compatible"] is True

    def test_vllm_version_compatibility_not_satisfied(self):
        result = self.connector.check_vllm_version_compatibility("2.16", "0.8.0")
        assert result["compatible"] is False
        assert "0.6.3" in result["summary"]

    @pytest.mark.asyncio
    async def test_check_compatibility_with_vllm_target(self):
        evidence = await self.connector.check_compatibility("2.18", vllm_version_target="0.8.0")
        vllm_items = [e for e in evidence if "vLLM version" in e.title]
        assert len(vllm_items) == 1
        assert "satisfies" in vllm_items[0].summary
