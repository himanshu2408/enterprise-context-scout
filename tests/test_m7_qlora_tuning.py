"""Unit & empirical integration tests for Milestone 7 QLoRA & Vertex PEFT Training.

Verifies:
1. Hugging Face Model Acquisition & Verification (Google model whitelist, safetensors, tokenizer).
2. 4-bit QLoRA Memory Math Verification (~19.6 GB on 48GB M5 Pro, leaving >28.4 GB free headroom).
3. HW-01 Process Mutual Exclusion Lock (contention detection, timeout, and clean release).
4. MoE LoRA Targeting & Strict Router Gate Freezing (router.proj forbidden from LoRA updates).
5. LocalQLoRATrainer Execution (genuine gradient descent, loss reduction, safetensors inspection).
6. VertexPEFTTrainer (Hugging Face PEFT LoraConfig, Vertex AI CustomJob spec, dry-run validation).
7. Domain Adapter Integration with AdapterPool.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetensors.torch import load_file

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.scout_workflow import LocalGemmaClient
from context_scout.tuning.benchmark import QLoRABenchmarkRunner
from context_scout.tuning.train_mlx_lora import (
    DEFAULT_DENSE_MODEL_ID,
    DEFAULT_MOE_MODEL_ID,
    GPUContentionError,
    GPUMutualExclusionLock,
    LocalQLoRATrainer,
    QLoRAHyperparameters,
    acquire_hf_model,
    compute_qlora_memory_budget,
    verify_model_weights,
)
from context_scout.tuning.train_vertex_peft import (
    VertexPEFTConfig,
    VertexPEFTTrainer,
)


class TestModelAcquisitionAndVerification:
    """Tests Hugging Face model acquisition and on-disk weight verification."""

    def test_model_id_authorization_policy(self) -> None:
        """Verifies that only authorized Google models can be acquired."""
        # Authorized Google Gemma 4 models
        res = compute_qlora_memory_budget(DEFAULT_MOE_MODEL_ID)
        assert res.model_id == DEFAULT_MOE_MODEL_ID

        # Prohibited models must be rejected
        with pytest.raises(ValueError, match="Prohibited or retired model ID"):
            acquire_hf_model("codegemma-7b")

        with pytest.raises(ValueError, match="violates the 100% Google-only model policy"):
            acquire_hf_model("meta-llama/Llama-3-8B")

    def test_verify_model_weights_on_disk(self, tmp_path: Path) -> None:
        """Verifies detection of valid safetensors, tokenizer.json, and config.json."""
        model_dir = tmp_path / "mock_gemma4_26b"
        model_dir.mkdir()

        # Initially invalid (empty directory)
        res_empty = verify_model_weights(model_dir)
        assert not res_empty.is_valid
        assert res_empty.error_message == "Missing .safetensors weight shards."

        # Add mock safetensors
        (model_dir / "model-00001-of-00003.safetensors").write_bytes(b"mock_safetensors_data")
        res_no_tok = verify_model_weights(model_dir)
        assert not res_no_tok.is_valid
        assert res_no_tok.error_message == "Missing tokenizer.json."

        # Add tokenizer.json
        (model_dir / "tokenizer.json").write_text('{"vocab": {}}', encoding="utf-8")
        res_no_cfg = verify_model_weights(model_dir)
        assert not res_no_cfg.is_valid
        assert res_no_cfg.error_message == "Missing config.json."

        # Add config.json with MoE parameters
        config_data = {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "hidden_size": 2816,
            "text_config": {
                "enable_moe_block": True,
                "num_experts": 128,
            },
        }
        (model_dir / "config.json").write_text(json.dumps(config_data), encoding="utf-8")

        res_valid = verify_model_weights(model_dir)
        assert res_valid.is_valid
        assert res_valid.has_tokenizer
        assert res_valid.has_config
        assert len(res_valid.safetensors_files) == 1
        assert res_valid.num_experts == 128
        assert res_valid.architecture == "Gemma4ForConditionalGeneration"


class TestQLoRAMemoryMath:
    """Tests 4-bit QLoRA unified memory math on Apple Silicon M5 Pro 48GB."""

    def test_moe_memory_budget_and_headroom(self) -> None:
        """Verifies ~19.6 GB footprint and >28.4 GB free headroom on 48GB unified memory."""
        # Using exact 48.0 GB host memory (48 * 1024^3 bytes)
        host_bytes = 48 * (1024**3)
        profile = compute_qlora_memory_budget(
            model_id=DEFAULT_MOE_MODEL_ID,
            r=16,
            lora_alpha=32,
            context_length=4096,
            host_memory_bytes=host_bytes,
        )

        assert profile.base_weights_gb == 13.9
        assert profile.adapter_weights_gb == 0.24
        assert profile.adamw_states_gb == 0.96
        assert profile.activation_cache_gb == 4.5
        assert profile.total_active_memory_gb == 19.6
        assert profile.host_unified_memory_gb == 48.0
        assert profile.free_headroom_gb == 28.4
        assert profile.is_safe_headroom is True

    def test_dense_baseline_memory_math(self) -> None:
        """Verifies ~8.0 GB active memory for Gemma 4 E4B dense baseline."""
        host_bytes = 48 * (1024**3)
        profile = compute_qlora_memory_budget(
            model_id=DEFAULT_DENSE_MODEL_ID,
            r=16,
            lora_alpha=32,
            context_length=4096,
            host_memory_bytes=host_bytes,
        )

        assert profile.base_weights_gb == 4.6
        assert profile.total_active_memory_gb == 8.0
        assert profile.free_headroom_gb == 40.0
        assert profile.is_safe_headroom is True


class TestGPUMutualExclusionLock:
    """Tests process-level mutual exclusion lock on Apple Silicon GPU (HW-01)."""

    def test_single_process_lock_acquisition(self, tmp_path: Path) -> None:
        """Verifies lock can be acquired and released cleanly."""
        lock_file = tmp_path / "gpu.lock"
        lock = GPUMutualExclusionLock(lock_path=lock_file, timeout_seconds=1.0)

        assert not lock.is_locked()
        with lock:
            assert lock.is_locked()
            assert lock_file.exists()
        assert not lock.is_locked()

    def test_lock_contention_and_timeout(self, tmp_path: Path) -> None:
        """Verifies second contender raises GPUContentionError when lock is held."""
        lock_file = tmp_path / "gpu_contention.lock"
        lock1 = GPUMutualExclusionLock(lock_path=lock_file, timeout_seconds=1.0)
        lock2 = GPUMutualExclusionLock(lock_path=lock_file, timeout_seconds=0.3, poll_interval=0.05)

        with (
            lock1,
            pytest.raises(GPUContentionError, match="HW-01 GPU lock contention"),
            lock2,
        ):
            pass

        # After lock1 releases, lock2 can acquire successfully
        with lock2:
            assert lock2.is_locked()


class TestMoELoRATargeting:
    """Tests LoRA target modules and strict MoE router gate freezing."""

    def test_moe_router_gate_strictly_frozen(self) -> None:
        """Verifies that targeting router.proj raises ValueError to prevent mode collapse."""
        # Attempting to target router.proj must fail
        illegal_params = QLoRAHyperparameters(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "router.proj"],
        )
        with pytest.raises(ValueError, match="Router projections must be strictly frozen"):
            LocalQLoRATrainer(
                base_model_id=DEFAULT_MOE_MODEL_ID,
                hyperparameters=illegal_params,
            )

    def test_target_modules_coverage(self) -> None:
        """Verifies expected target modules for Gemma 4 attention and shared MLP."""
        params = QLoRAHyperparameters()
        assert params.scaling == 2.0
        assert "q_proj" in params.target_modules
        assert "k_proj" in params.target_modules
        assert "v_proj" in params.target_modules
        assert "o_proj" in params.target_modules
        assert "gate_proj" in params.target_modules
        assert "up_proj" in params.target_modules
        assert "down_proj" in params.target_modules

        # Check that no router keys exist in targets
        for mod in params.target_modules:
            assert "router" not in mod


class TestLocalQLoRATrainingExecution:
    """Tests genuine QLoRA training execution and artifact serialization."""

    def test_training_loss_decrease_and_adapter_export(self, tmp_path: Path) -> None:
        """Verifies genuine gradient descent, loss reduction, and safetensors export."""
        output_dir = tmp_path / "adapters" / "test_gemma4_26b"
        lock_file = tmp_path / "test_trainer.lock"

        trainer = LocalQLoRATrainer(
            base_model_id=DEFAULT_MOE_MODEL_ID,
            lock_path=lock_file,
            lock_timeout=5.0,
        )

        train_jsonl = Path("data/tuning/trading_code_scout/train.jsonl")
        valid_jsonl = Path("data/tuning/trading_code_scout/valid.jsonl")

        result = trainer.train(
            train_jsonl=train_jsonl,
            valid_jsonl=valid_jsonl,
            output_adapter_dir=output_dir,
            domain_id="test_trading_scout",
            num_steps=10,
        )

        assert result.status == "COMPLETED"
        assert result.safetensors_path.exists()
        assert result.adapter_config_path.exists()
        assert result.training_metrics_path.exists()

        # Loss decrease verification
        assert result.metrics.final_loss < result.metrics.initial_loss, (
            f"Expected loss decrease: initial={result.metrics.initial_loss}, "
            f"final={result.metrics.final_loss}"
        )
        assert len(result.metrics.step_history) == 10

        # Safetensors inspection
        tensors = load_file(str(result.safetensors_path))
        assert len(tensors) == 56  # 4 layers * 7 modules * 2 tensors (lora_a, lora_b)

        # Confirm router weights are NOT in safetensors
        assert not any("router" in k for k in tensors)

        # Confirm lora_b tensors received gradient updates (non-zero norm)
        lora_b_tensors = [v for k, v in tensors.items() if k.endswith(".lora_b")]
        has_updated_weights = any(t.norm().item() > 0.0 for t in lora_b_tensors)
        assert has_updated_weights, "Expected at least one lora_b tensor to have non-zero updates"

        # Config inspection
        with open(result.adapter_config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg["r"] == 16
        assert cfg["lora_alpha"] == 32
        assert cfg["scaling"] == 2.0
        assert cfg["is_moe"] is True
        assert cfg["num_experts"] == 128


class TestVertexPEFTTrainer:
    """Tests GCP Vertex AI Hugging Face PEFT spec generation and dry-run validation."""

    def test_peft_lora_config_generation(self) -> None:
        """Verifies genuine Hugging Face PEFT LoraConfig dictionary."""
        trainer = VertexPEFTTrainer(
            config=VertexPEFTConfig(domain_id="trading_code_scout", r=16, lora_alpha=32)
        )
        lora_cfg = trainer.generate_peft_lora_config()

        assert lora_cfg["peft_type"] == "LORA"
        assert lora_cfg["task_type"] == "CAUSAL_LM"
        assert lora_cfg["r"] == 16
        assert lora_cfg["lora_alpha"] == 32
        assert "q_proj" in lora_cfg["target_modules"]
        assert "router.proj" not in lora_cfg["target_modules"]

    def test_vertex_job_spec_generation(self) -> None:
        """Verifies GCP Vertex AI CustomJob resource dictionary."""
        config = VertexPEFTConfig(
            domain_id="trading_code_scout",
            machine_type="g2-standard-12",
            accelerator_type="NVIDIA_L4",
            output_gcs_uri="gs://context-scout-adapters/gemma4_26b/",
        )
        trainer = VertexPEFTTrainer(config=config)
        spec = trainer.generate_job_spec()

        assert "displayName" in spec
        assert "jobSpec" in spec
        worker_pools = spec["jobSpec"]["workerPoolSpecs"]
        assert len(worker_pools) == 1
        m_spec = worker_pools[0]["machineSpec"]
        assert m_spec["machineType"] == "g2-standard-12"
        assert m_spec["acceleratorType"] == "NVIDIA_L4"
        assert m_spec["acceleratorCount"] == 1

        output_prefix = spec["jobSpec"]["baseOutputDirectory"]["outputUriPrefix"]
        assert output_prefix == "gs://context-scout-adapters/gemma4_26b/trading_code_scout"

    def test_dry_run_validation(self) -> None:
        """Verifies dry-run validation checks pass and produce realistic cost estimates."""
        trainer = VertexPEFTTrainer(
            config=VertexPEFTConfig(domain_id="quant_research_scout")
        )
        dry_run_res = trainer.dry_run()

        assert dry_run_res["status"] == "VALID"
        assert dry_run_res["domain_id"] == "quant_research_scout"
        assert dry_run_res["peft_scaling"] == 2.0
        assert dry_run_res["router_gates_frozen"] is True
        assert dry_run_res["estimated_cost_usd"] > 0.0
        assert dry_run_res["gpu_memory_headroom_gb"] >= 2.0
        assert len(dry_run_res["validation_issues"]) == 0

        # Test submit_job dry_run mode
        job_submit = trainer.submit_job(dry_run=True)
        assert job_submit["state"] == "DRY_RUN_PASSED"
        assert "gs://" in job_submit["output_uri"]


class TestDomainAdaptersIntegration:
    """Verifies domain adapter artifacts and AdapterPool integration."""

    def test_domain_adapters_exist_and_valid(self) -> None:
        """Verifies genuine adapters for both trading_code_scout and quant_research_scout."""
        base_dir = Path("domains")

        domains = [
            ("trading_code_scout", "adapters/gemma4_26b_code_scout"),
            ("quant_research_scout", "adapters/gemma4_26b_quant_research"),
        ]

        for domain_id, rel_adapter_path in domains:
            adapter_dir = base_dir / domain_id / rel_adapter_path
            assert adapter_dir.exists(), f"Adapter directory missing: {adapter_dir}"

            safetensors_file = adapter_dir / "adapters.safetensors"
            config_file = adapter_dir / "adapter_config.json"
            metrics_file = adapter_dir / "training_metrics.json"

            assert safetensors_file.exists()
            assert safetensors_file.stat().st_size > 1_000_000  # >1MB genuine weights

            assert config_file.exists()
            with open(config_file, encoding="utf-8") as f:
                cfg = json.load(f)
            assert cfg["r"] == 16
            assert cfg["lora_alpha"] == 32
            assert cfg["is_moe"] is True

            assert metrics_file.exists()
            with open(metrics_file, encoding="utf-8") as f:
                metrics = json.load(f)
            assert metrics["domain_id"] == domain_id
            assert metrics["final_loss"] <= metrics["initial_loss"]

    def test_domain_registry_and_adapter_pool_registration(self) -> None:
        """Verifies DomainRegistry registers adapters into AdapterPool."""
        registry = DomainRegistry(domains_dir=Path("domains"))
        registry.load_all()

        for d_id in ["trading_code_scout", "quant_research_scout"]:
            slot = registry.adapter_pool.get_slot(d_id)
            assert slot.domain_id == d_id
            assert slot.max_kv_tokens <= 32_768
            adapter_path = Path(slot.adapter_path)
            assert (adapter_path / "adapters.safetensors").exists()

    def test_adapter_pool_artifact_verification(self) -> None:
        """Verifies AdapterPool artifact verification methods for genuine weights."""
        registry = DomainRegistry(domains_dir=Path("domains"))
        registry.load_all()

        verifications = registry.adapter_pool.verify_all_adapters()
        assert len(verifications) >= 2

        for domain_id in ["trading_code_scout", "quant_research_scout"]:
            v = verifications[domain_id]
            assert v["is_valid"] is True
            assert v["domain_id"] == domain_id
            assert v["safetensors_bytes"] > 1_000_000
            assert v["r"] == 16
            assert v["lora_alpha"] == 32
            assert v["is_moe"] is True
            assert v["final_loss"] is not None
            assert v["error_message"] is None

    def test_local_gemma_client_adapter_wiring(self) -> None:
        """Verifies LocalGemmaClient wiring with AdapterPool for loading and verification."""
        registry = DomainRegistry(domains_dir=Path("domains"))
        registry.load_all()

        client = LocalGemmaClient(
            adapter_pool=registry.adapter_pool,
            offline_mode=True,
        )

        # Load trading code scout adapter
        slot_code = client.load_adapter("trading_code_scout")
        assert slot_code.domain_id == "trading_code_scout"
        v_active = client.verify_active_adapter()
        assert v_active["is_valid"] is True
        assert v_active["domain_id"] == "trading_code_scout"

        # Hot-swap to quant research scout adapter
        slot_quant = client.load_adapter("quant_research_scout")
        assert slot_quant.domain_id == "quant_research_scout"
        v_quant = client.verify_active_adapter()
        assert v_quant["is_valid"] is True
        assert v_quant["domain_id"] == "quant_research_scout"

        # Loading unregistered domain raises KeyError
        with pytest.raises(KeyError):
            client.load_adapter("non_existent_domain")

    def test_empirical_before_vs_after_benchmark_execution(self, tmp_path: Path) -> None:
        """Verifies empirical benchmark execution and generated reports."""
        runner = QLoRABenchmarkRunner(reports_dir=tmp_path)
        report_dict, md_report = runner.run_benchmark()

        assert report_dict["all_passed"] is True
        m = report_dict["metrics"]

        # 1. Pydantic schema adherence
        schema = m["pydantic_schema_adherence_pass_rate"]
        assert schema["after_qlora_zero_shot"] >= 95.0
        assert schema["before_zero_shot"] < schema["after_qlora_zero_shot"]

        # 2. AST Skeleton Localization Recall@5
        recall = m["ast_skeleton_localization_recall_at_5"]
        assert recall["after_qlora_adapter"] >= 90.0
        assert recall["before_base_model"] < recall["after_qlora_adapter"]

        # 3. Prompt token overhead reduction: 1,500-2,000 tokens
        tokens = m["prompt_token_overhead_reduction"]
        assert 1500.0 <= tokens["tokens_eliminated_avg"] <= 2000.0
        assert tokens["token_reduction_pct"] > 80.0

        # 4. Reports written to disk
        assert (tmp_path / "m7_qlora_benchmark_report.json").is_file()
        assert (tmp_path / "m7_qlora_benchmark_report.md").is_file()
        assert "Milestone 7 Empirical QLoRA Benchmark Report" in md_report

