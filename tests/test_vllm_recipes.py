"""Unit tests for vLLM recipe connector.

Tests both the matching logic and live recipe fetching.
"""

import asyncio
import pytest

from connectors.vllm_recipes import VllmRecipeConnector


@pytest.fixture
def connector():
    return VllmRecipeConnector()


class TestExtractFamilyAndParams:
    """Test model family and parameter extraction."""

    def test_redhat_qwen_fp8(self, connector):
        family, params = connector._extract_family_and_params("RedHatAI/Qwen3.5-35B-A3B-FP8-dynamic")
        assert "qwen" in family
        assert params == "35b"

    def test_meta_llama(self, connector):
        family, params = connector._extract_family_and_params("meta-llama/Llama-3.1-8B-Instruct")
        assert "llama" in family
        assert params == "8b"

    def test_qwen_base(self, connector):
        family, params = connector._extract_family_and_params("Qwen/Qwen3-30B-A3B")
        assert "qwen" in family
        assert params == "30b"

    def test_mistral(self, connector):
        family, params = connector._extract_family_and_params("mistralai/Mistral-Small-24B-Instruct")
        assert "mistral" in family
        assert params == "24b"

    def test_no_params(self, connector):
        family, params = connector._extract_family_and_params("google/gemma-2-it")
        assert "gemma" in family

    def test_qwq(self, connector):
        family, params = connector._extract_family_and_params("Qwen/QwQ-32B")
        assert params == "32b"


class TestMatchModel:
    """Test model matching against a sample index."""

    @pytest.fixture
    def sample_index(self):
        return [
            {"hf_id": "Qwen/Qwen3-30B-A3B", "title": "Qwen3-30B-A3B", "json": "/Qwen/Qwen3-30B-A3B.json"},
            {"hf_id": "Qwen/Qwen3-32B", "title": "Qwen3-32B", "json": "/Qwen/Qwen3-32B.json"},
            {"hf_id": "meta-llama/Llama-3.1-8B-Instruct", "title": "Llama-3.1-8B-Instruct", "json": "/meta-llama/Llama-3.1-8B-Instruct.json"},
            {"hf_id": "meta-llama/Llama-3.1-70B-Instruct", "title": "Llama-3.1-70B-Instruct", "json": "/meta-llama/Llama-3.1-70B-Instruct.json"},
            {"hf_id": "mistralai/Mistral-Small-24B-Instruct", "title": "Mistral-Small-24B", "json": "/mistralai/Mistral-Small-24B.json"},
        ]

    def test_exact_match(self, connector, sample_index):
        matches = connector._match_model(sample_index, "meta-llama/Llama-3.1-8B-Instruct")
        assert len(matches) >= 1
        assert any(m.get("hf_id") == "meta-llama/Llama-3.1-8B-Instruct" for m in matches)

    def test_redhat_qwen_fuzzy_match(self, connector, sample_index):
        """RedHatAI/Qwen3.5-35B-A3B-FP8-dynamic should match Qwen3-30B-A3B by family."""
        matches = connector._match_model(sample_index, "RedHatAI/Qwen3.5-35B-A3B-FP8-dynamic")
        # Should find at least the Qwen family entries
        assert len(matches) >= 1, f"Expected at least one Qwen match, got {matches}"
        matched_ids = [m.get("hf_id") for m in matches]
        print(f"Matched: {matched_ids}")

    def test_quant_suffix_strip(self, connector, sample_index):
        """FP8 suffix should be stripped for base model matching."""
        matches = connector._match_model(sample_index, "meta-llama/Llama-3.1-8B-Instruct-FP8")
        assert len(matches) >= 1
        assert any("Llama-3.1-8B" in (m.get("hf_id") or "") for m in matches)

    def test_substring_match(self, connector, sample_index):
        """Substring of hf_id should match."""
        matches = connector._match_model(sample_index, "Qwen/Qwen3-32B")
        assert len(matches) >= 1


class TestFindRecipeLive:
    """Live integration tests (require network)."""

    @pytest.mark.asyncio
    async def test_llama_8b_finds_recipe(self, connector):
        """Llama-3.1-8B-Instruct should have a recipe on recipes.vllm.ai."""
        evidence = await connector.find_recipe("meta-llama/Llama-3.1-8B-Instruct")
        assert len(evidence) > 0, "Expected recipe evidence for Llama-3.1-8B"
        print(f"Found {len(evidence)} evidence items")
        for e in evidence[:3]:
            print(f"  - {e.title}: {e.summary[:80]}")

    @pytest.mark.asyncio
    async def test_redhat_qwen_finds_recipe(self, connector):
        """RedHatAI/Qwen3.5-35B-A3B-FP8-dynamic should find a Qwen recipe."""
        evidence = await connector.find_recipe("RedHatAI/Qwen3.5-35B-A3B-FP8-dynamic")
        print(f"Found {len(evidence)} evidence items for Qwen3.5-35B")
        for e in evidence[:5]:
            print(f"  - {e.title}: {e.summary[:80]}")
        # Even if not exact match, family match should work
        assert len(evidence) > 0, "Expected at least one recipe match via family search"

    @pytest.mark.asyncio
    async def test_nonexistent_model(self, connector):
        """A totally unknown model should return empty list (not error)."""
        evidence = await connector.find_recipe("fake-org/totally-nonexistent-model-xyz")
        assert evidence == []
