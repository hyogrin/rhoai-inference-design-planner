"""Tests for domain Pydantic models."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest


class TestModelIdentity:
    def test_valid_model_identity(self, sample_model_identity_data):
        from domain.model import ModelIdentity

        model = ModelIdentity(**sample_model_identity_data)
        assert model.repo_id == "meta-llama/Llama-3.1-70B-Instruct"
        assert model.gated is True
        assert model.tasks == ["text-generation", "conversational"]

    def test_model_identity_rejects_extra_fields(self, sample_model_identity_data):
        from domain.model import ModelIdentity

        sample_model_identity_data["unknown_field"] = "bad"
        with pytest.raises(ValueError):
            ModelIdentity(**sample_model_identity_data)


class TestModelArchitecture:
    def test_dense_model(self):
        from domain.model import ModelArchitecture

        arch = ModelArchitecture(
            architecture_names=["LlamaForCausalLM"],
            family="llama",
            architecture_type="dense",
            parameter_count_total=70_000_000_000,
            num_hidden_layers=80,
            hidden_size=8192,
            num_attention_heads=64,
            num_kv_heads=8,
            head_dim=128,
            max_position_embeddings=131072,
            kv_layout="mha_gqa",
            parser_confidence=1.0,
            missing_fields=[],
            raw_config_paths=["config.json"],
        )
        assert arch.architecture_type == "dense"
        assert arch.num_kv_heads == 8

    def test_moe_model(self):
        from domain.model import ModelArchitecture

        arch = ModelArchitecture(
            architecture_names=["MixtralForCausalLM"],
            family="mixtral",
            architecture_type="moe",
            parameter_count_total=46_700_000_000,
            parameter_count_active=12_900_000_000,
            num_hidden_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            num_experts_total=8,
            num_experts_active=2,
            kv_layout="mha_gqa",
            parser_confidence=1.0,
            missing_fields=[],
            raw_config_paths=["config.json"],
        )
        assert arch.architecture_type == "moe"
        assert arch.num_experts_total == 8
        assert arch.num_experts_active == 2


class TestHardwareInventory:
    def test_valid_inventory(self, sample_hardware_pool_data):
        from domain.hardware import HardwareInventory, HardwarePool

        pool = HardwarePool(**sample_hardware_pool_data)
        inventory = HardwareInventory(
            environment_type="on_prem",
            pools=[pool],
        )
        assert inventory.environment_type == "on_prem"
        assert len(inventory.pools) == 1
        assert inventory.pools[0].hbm_gb_per_accelerator == 80.0
        assert inventory.currency == "USD"


class TestEvidenceItem:
    def test_valid_evidence(self):
        from domain.evidence import EvidenceItem

        evidence = EvidenceItem(
            evidence_id=uuid4(),
            category="recipe",
            claim_type="tested_hardware",
            title="vLLM recipe for Llama-3.1-70B",
            summary="Tested on 4xH100 with TP=4",
            source_url="https://recipes.vllm.ai/models/meta-llama/Llama-3.1-70B-Instruct",
            source_domain="recipes.vllm.ai",
            retrieved_at=datetime.now(UTC),
            source_tier="primary",
            verification_level="verified",
            parser_warnings=[],
        )
        assert evidence.category == "recipe"
        assert evidence.source_tier == "primary"


class TestValidationReport:
    def test_ready_for_sizing(self):
        from domain.validation import ValidationCheck, ValidationReport

        report = ValidationReport(
            status="ready_for_sizing",
            checks=[
                ValidationCheck(
                    check_id="model_revision_resolved",
                    status="passed",
                    message="Model revision resolved to commit SHA",
                    evidence_ids=[uuid4()],
                    remediation=None,
                )
            ],
            blockers=[],
            warnings=[],
            missing_inputs=[],
            evidence_coverage={"model_metadata": True, "recipe": True},
            overall_confidence=0.85,
            allowed_outputs=["topology", "memory", "cost"],
        )
        assert report.status == "ready_for_sizing"
        assert report.overall_confidence == 0.85

    def test_blocked_status(self):
        from domain.validation import ValidationReport

        report = ValidationReport(
            status="blocked",
            checks=[],
            blockers=["Model repository not accessible"],
            warnings=[],
            missing_inputs=["hf_token"],
            evidence_coverage={},
            overall_confidence=0.0,
            allowed_outputs=[],
        )
        assert report.status == "blocked"
        assert len(report.blockers) == 1


class TestWorkloadProfile:
    def test_valid_workload(self, sample_workload_profile_data):
        from domain.workload import WorkloadProfile

        profile = WorkloadProfile(**sample_workload_profile_data)
        assert profile.use_case_type == "rag"
        assert profile.isl_distribution is not None
        assert profile.isl_distribution.p50 == 512
        assert profile.isl_distribution.p95 == 2048
        assert profile.latency_slo.ttft_p95_ms == 500.0


class TestDesignSession:
    def test_create_session(self):
        from domain.session import DesignSession

        session = DesignSession(
            session_id=uuid4(),
            status="intake",
            current_step=1,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            version=1,
        )
        assert session.status == "intake"
        assert session.current_step == 1
