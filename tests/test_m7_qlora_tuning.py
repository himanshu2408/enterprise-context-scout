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
import os
from pathlib import Path

import huggingface_hub
import pytest
import torch
from safetensors.torch import load_file, save_file
from tests.fixtures.mock_generators import MockLocalGemmaClient

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
        """Verifies default Gemma 4 + approved BYO models pass; retired & unapproved fail."""
        from context_scout.config import ModelGovernancePolicy

        # Authorized Google Gemma 4 default and BYO open-weights models
        res = compute_qlora_memory_budget(DEFAULT_MOE_MODEL_ID)
        assert res.model_id == DEFAULT_MOE_MODEL_ID
        assert (
            ModelGovernancePolicy.default().validate_local_model_id("Qwen/Qwen3-Coder-30B-A3B")
            == "Qwen/Qwen3-Coder-30B-A3B"
        )

        # Prohibited / retired models must be rejected
        with pytest.raises(ValueError, match="Prohibited or retired model ID"):
            acquire_hf_model("codegemma-7b")

        with pytest.raises(ValueError, match="violates ModelGovernancePolicy"):
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

    def test_real_hf_model_snapshots_and_mlx_gpu_adapters(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verifies HF 4-bit MLX checkpoints (live or seeded) and mlx:gpu adapters."""
        use_live_mlx = os.environ.get("CONTEXT_SCOUT_USE_LIVE_MLX") == "1"

        if not use_live_mlx:
            moe_dir = tmp_path / "moe_snapshot"
            moe_dir.mkdir()
            for idx in range(1, 4):
                save_file(
                    {f"layer.{idx}.weight": torch.zeros((2, 2))},
                    str(moe_dir / f"model-0000{idx}-of-00003.safetensors"),
                )
            (moe_dir / "tokenizer.json").write_text('{"vocab": {}}', encoding="utf-8")
            (moe_dir / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Gemma4ForConditionalGeneration"],
                        "hidden_size": 2816,
                        "text_config": {"enable_moe_block": True, "num_experts": 128},
                    }
                ),
                encoding="utf-8",
            )

            dense_dir = tmp_path / "dense_snapshot"
            dense_dir.mkdir()
            save_file({"layer.0.weight": torch.zeros((2, 2))}, str(dense_dir / "model.safetensors"))
            (dense_dir / "tokenizer.json").write_text('{"vocab": {}}', encoding="utf-8")
            (dense_dir / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Gemma4ForConditionalGeneration"],
                        "hidden_size": 2048,
                        "text_config": {"enable_moe_block": False},
                    }
                ),
                encoding="utf-8",
            )

            def _fake_cache_lookup(repo_id: str, filename: str, **_kwargs: object) -> str | None:
                if repo_id == DEFAULT_MOE_MODEL_ID:
                    return str(moe_dir / filename)
                if repo_id == DEFAULT_DENSE_MODEL_ID:
                    return str(dense_dir / filename)
                return None

            monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _fake_cache_lookup)

        moe_res = acquire_hf_model(DEFAULT_MOE_MODEL_ID, local_files_only=True)
        assert moe_res.is_verified is True
        assert moe_res.is_moe is True
        assert moe_res.num_experts == 128
        assert len(moe_res.shard_files) == 3
        if use_live_mlx:
            assert moe_res.total_size_bytes > 14_000_000_000
        else:
            assert moe_res.total_size_bytes > 0

        dense_res = acquire_hf_model(DEFAULT_DENSE_MODEL_ID, local_files_only=True)
        assert dense_res.is_verified is True
        assert dense_res.is_moe is False
        assert len(dense_res.shard_files) == 1
        if use_live_mlx:
            assert dense_res.total_size_bytes > 4_500_000_000
        else:
            assert dense_res.total_size_bytes > 0

        for adapter_rel in [
            "domains/code_architecture_scout/adapters/gemma4_26b_code_scout",
            "domains/enterprise_docs_scout/adapters/gemma4_26b_enterprise_docs",
        ]:
            adapter_dir = Path(adapter_rel)
            st_file = adapter_dir / "adapters.safetensors"
            if st_file.exists():
                tensors = load_file(str(st_file))
                assert len(tensors) == 54  # 4 Gemma 4 layers (26..29, layer 29 uses k_eq_v)
                assert not any("router" in k for k in tensors)
            with open(adapter_dir / "training_metrics.json", encoding="utf-8") as f:
                metrics = json.load(f)
            assert metrics["device"] == "mlx:gpu"
            assert metrics["final_loss"] < metrics["initial_loss"]


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

    def test_lock_non_blocking_immediate_contention(self, tmp_path: Path) -> None:
        """Verifies non_blocking=True raises GPUContentionError immediately (<100ms)."""
        import time

        lock_file = tmp_path / "gpu_non_blocking.lock"
        lock1 = GPUMutualExclusionLock(lock_path=lock_file, timeout_seconds=5.0)
        lock2 = GPUMutualExclusionLock(lock_path=lock_file, timeout_seconds=5.0, non_blocking=True)

        with lock1:
            t0 = time.perf_counter()
            with pytest.raises(GPUContentionError, match="non-blocking acquisition failed"):
                lock2.acquire()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            assert elapsed_ms < 100.0, (
                f"Non-blocking acquire took {elapsed_ms:.2f}ms, expected < 100ms"
            )


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
    """Tests native Apple Silicon mlx-lm QLoRA training and toy loop removal."""

    def test_toy_nn_embedding_loop_removed_from_train_mlx_lora(self) -> None:
        """Regression: verify standalone nn.Embedding toy PyTorch loop is deleted."""
        import ast

        src_path = Path("src/context_scout/tuning/train_mlx_lora.py")
        src_text = src_path.read_text(encoding="utf-8")
        tree = ast.parse(src_text)
        defined_classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        assert "LowRankLinearAdapter" not in defined_classes
        assert "Gemma4AdapterBlock" not in defined_classes
        assert "nn.Embedding" not in src_text
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name != "torch" for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("torch")

        trainer = LocalQLoRATrainer(base_model_id=DEFAULT_MOE_MODEL_ID)
        with pytest.raises(ValueError, match="Toy PyTorch loop was removed"):
            trainer.train(
                train_jsonl=Path("domains/code_architecture_scout/tuning/train.jsonl"),
                valid_jsonl=Path("domains/code_architecture_scout/tuning/valid.jsonl"),
                output_adapter_dir=Path("/tmp/unused_mlx_out"),
                use_mlx_native=False,
            )

    def test_training_loss_decrease_and_adapter_export(self, tmp_path: Path) -> None:
        """Verifies native mlx-lm (Apple Silicon) or PyTorch/PEFT rsLoRA (Linux x86_64)."""
        import importlib.util

        output_dir = tmp_path / "adapters" / "test_gemma4_26b"
        train_jsonl = Path("domains/code_architecture_scout/tuning/train.jsonl")
        valid_jsonl = Path("domains/code_architecture_scout/tuning/valid.jsonl")

        if importlib.util.find_spec("mlx_lm") is None:
            from context_scout.tuning.teacher_distiller import generate_synthetic_canaries
            from context_scout.tuning.train_peft_lora import (
                PEFTRsLoRATrainer,
                PEFTTrainingConfig,
            )

            canaries = generate_synthetic_canaries(
                domain_id="code_architecture_scout",
                seed=42,
                count=10,
            )
            canaries_path = tmp_path / "canaries.json"
            canaries_path.write_text(
                json.dumps([c.to_dict() for c in canaries], indent=2),
                encoding="utf-8",
            )
            peft_cfg = PEFTTrainingConfig(
                domain_id="code_architecture_scout",
                training_scope_ceiling=["public_eng"],
                base_model_id=DEFAULT_MOE_MODEL_ID,
                r=8,
                lora_alpha=16,
                sft_steps=10,
                dpo_steps=0,
                learning_rate=5e-3,
                max_seq_length=128,
            )
            peft_trainer = PEFTRsLoRATrainer(peft_cfg, use_gpu_lock=False)
            peft_result = peft_trainer.train(
                train_jsonl=train_jsonl,
                valid_jsonl=valid_jsonl,
                canaries_json=canaries_path,
                output_adapter_dir=output_dir,
                artifact_id="code_architecture_scout_linux_ci",
                allow_offline_tiny_model=True,
                register_in_db=False,
                activate_in_db=False,
            )
            assert peft_result.dry_run_passed is True
            assert peft_result.safetensors_path.exists()
            assert peft_result.adapter_config_path.exists()
            assert peft_result.sft_final_loss < peft_result.sft_initial_loss, (
                f"Expected loss decrease: initial={peft_result.sft_initial_loss}, "
                f"final={peft_result.sft_final_loss}"
            )
            tensors = load_file(str(peft_result.safetensors_path))
            assert len(tensors) > 0
            assert not any("router" in k.lower() for k in tensors)
            lora_b_tensors = [v for k, v in tensors.items() if "lora_b" in k.lower()]
            assert any(t.norm().item() > 0.0 for t in lora_b_tensors)
            with open(peft_result.adapter_config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            assert cfg.get("use_rslora") is True
            return

        from mlx_lm.models import llama as mlx_llama

        class _ByteChatTokenizer:
            def apply_chat_template(
                self,
                messages: list[dict[str, str]],
                tokenize: bool = False,
                add_generation_prompt: bool = False,
                **_kwargs: object,
            ) -> str:
                assert tokenize is False
                parts = [f"[{m['role']}]\n{m['content']}\n" for m in messages]
                if add_generation_prompt:
                    parts.append("[assistant]\n")
                return "".join(parts)

            def encode(self, text: str) -> list[int]:
                return [min(255, b) for b in text.encode("utf-8", errors="ignore")] or [1]

        def _fake_mlx_loader(_model_id: str) -> tuple[object, object]:
            args = mlx_llama.ModelArgs(
                model_type="llama",
                hidden_size=64,
                num_hidden_layers=4,
                intermediate_size=128,
                num_attention_heads=4,
                rms_norm_eps=1e-5,
                vocab_size=256,
                num_key_value_heads=2,
            )
            return mlx_llama.Model(args), _ByteChatTokenizer()

        lock_file = tmp_path / "test_trainer.lock"
        loader = None if os.environ.get("CONTEXT_SCOUT_USE_LIVE_MLX") == "1" else _fake_mlx_loader

        trainer = LocalQLoRATrainer(
            base_model_id=DEFAULT_MOE_MODEL_ID,
            lock_path=lock_file,
            lock_timeout=5.0,
            mlx_model_loader=loader,
        )

        result = trainer.train(
            train_jsonl=train_jsonl,
            valid_jsonl=valid_jsonl,
            output_adapter_dir=output_dir,
            domain_id="code_architecture_scout",
            num_steps=10,
            use_mlx_native=True,
            learning_rate=5e-3,
            max_seq_length=128,
        )

        assert result.status == "COMPLETED"
        assert result.safetensors_path.exists()
        assert result.adapter_config_path.exists()
        assert result.training_metrics_path.exists()
        assert result.metrics.device == "mlx:gpu"

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
        assert cfg["use_rslora"] is True
        assert cfg["use_dora"] is False
        assert cfg["is_moe"] is True
        assert cfg["num_experts"] == 128


class TestVertexPEFTTrainer:
    """Tests GCP Vertex AI Hugging Face PEFT spec generation and dry-run validation."""

    def test_peft_lora_config_generation(self) -> None:
        """Verifies genuine Hugging Face PEFT LoraConfig dictionary."""
        trainer = VertexPEFTTrainer(
            config=VertexPEFTConfig(domain_id="code_architecture_scout", r=16, lora_alpha=32)
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
            domain_id="code_architecture_scout",
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
        assert output_prefix == "gs://context-scout-adapters/gemma4_26b/code_architecture_scout"

    def test_dry_run_validation(self) -> None:
        """Verifies dry-run validation checks pass and produce realistic cost estimates."""
        trainer = VertexPEFTTrainer(config=VertexPEFTConfig(domain_id="enterprise_docs_scout"))
        dry_run_res = trainer.dry_run()

        assert dry_run_res["status"] == "VALID"
        assert dry_run_res["domain_id"] == "enterprise_docs_scout"
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
        """Verifies adapter configs/metrics (and real .safetensors when present on disk)."""
        base_dir = Path("domains")

        domains = [
            ("code_architecture_scout", "adapters/gemma4_26b_code_scout"),
            ("enterprise_docs_scout", "adapters/gemma4_26b_enterprise_docs"),
        ]

        for domain_id, rel_adapter_path in domains:
            adapter_dir = base_dir / domain_id / rel_adapter_path
            assert adapter_dir.exists(), f"Adapter directory missing: {adapter_dir}"

            safetensors_file = adapter_dir / "adapters.safetensors"
            config_file = adapter_dir / "adapter_config.json"
            metrics_file = adapter_dir / "training_metrics.json"

            if safetensors_file.exists():
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
            assert len(metrics["step_history"]) > 0
            assert abs(metrics["final_loss"] - metrics["step_history"][-1]["loss"]) < 1e-4

    def test_domain_registry_and_adapter_pool_registration(self) -> None:
        """Verifies DomainRegistry registers adapters into AdapterPool."""
        registry = DomainRegistry(domains_dir=Path("domains"))
        registry.load_all()

        for d_id in ["code_architecture_scout", "enterprise_docs_scout"]:
            slot = registry.adapter_pool.get_slot(d_id)
            assert slot.domain_id == d_id
            assert slot.max_kv_tokens <= 32_768
            adapter_path = Path(slot.adapter_path)
            assert (adapter_path / "adapter_config.json").exists()

    def test_adapter_pool_artifact_verification(self, tmp_path: Path) -> None:
        """Verifies AdapterPool base_open_weights on clean clone + non-zero/zero-delta checks."""
        import numpy as np
        from safetensors.numpy import save_file as save_safetensors_numpy

        registry = DomainRegistry(domains_dir=Path("domains"))
        registry.load_all()

        verifications = registry.adapter_pool.verify_all_adapters()
        assert len(verifications) >= 2

        for domain_id in ["code_architecture_scout", "enterprise_docs_scout"]:
            v = verifications[domain_id]
            assert v["is_valid"] is True
            assert v["domain_id"] == domain_id
            assert v["r"] == 16
            if v.get("adapters_present"):
                assert v["safetensors_bytes"] > 1_000_000
            else:
                assert v["serving_mode"] == "base_open_weights"
                assert v["adapters_present"] is False

        # Non-zero trained safetensors passes; zero-delta lora_b == 0 fails closed.
        tmp_domains = tmp_path / "domains"
        ad_dir = tmp_domains / "code_architecture_scout" / "adapters" / "gemma4_26b_code_scout"
        ad_dir.mkdir(parents=True)
        (ad_dir / "adapter_config.json").write_text(
            json.dumps({"r": 16, "lora_alpha": 32, "is_moe": True}),
            encoding="utf-8",
        )
        save_safetensors_numpy(
            {
                "language_model.model.layers.26.self_attn.q_proj.lora_a": np.ones(
                    (8, 4), dtype=np.float32
                ),
                "language_model.model.layers.26.self_attn.q_proj.lora_b": np.zeros(
                    (4, 8), dtype=np.float32
                ),
            },
            str(ad_dir / "adapters.safetensors"),
        )
        zero_res = registry.adapter_pool.verify_adapter(
            "code_architecture_scout", base_dir=tmp_path
        )
        assert zero_res["is_valid"] is False
        assert "Zero-delta" in str(zero_res["error_message"])

        save_safetensors_numpy(
            {
                "language_model.model.layers.26.self_attn.q_proj.lora_a": np.ones(
                    (8, 4), dtype=np.float32
                ),
                "language_model.model.layers.26.self_attn.q_proj.lora_b": np.full(
                    (4, 8), 0.05, dtype=np.float32
                ),
            },
            str(ad_dir / "adapters.safetensors"),
        )
        real_res = registry.adapter_pool.verify_adapter(
            "code_architecture_scout", base_dir=tmp_path
        )
        assert real_res["is_valid"] is True
        assert real_res["adapters_present"] is True
        assert real_res["serving_mode"] == "qlora_adapter"

    def test_local_gemma_client_adapter_wiring(self) -> None:
        """Verifies LocalGemmaClient wiring with AdapterPool for loading and verification."""
        registry = DomainRegistry(domains_dir=Path("domains"))
        registry.load_all()

        client = LocalGemmaClient(
            adapter_pool=registry.adapter_pool,
            offline_mode=True,
        )

        # Load trading code scout adapter
        slot_code = client.load_adapter("code_architecture_scout")
        assert slot_code.domain_id == "code_architecture_scout"
        v_active = client.verify_active_adapter()
        assert v_active["is_valid"] is True
        assert v_active["domain_id"] == "code_architecture_scout"

        # Hot-swap to quant research scout adapter
        slot_quant = client.load_adapter("enterprise_docs_scout")
        assert slot_quant.domain_id == "enterprise_docs_scout"
        v_quant = client.verify_active_adapter()
        assert v_quant["is_valid"] is True
        assert v_quant["domain_id"] == "enterprise_docs_scout"

        # Loading unregistered domain raises KeyError
        with pytest.raises(KeyError):
            client.load_adapter("non_existent_domain")

    def test_empirical_before_vs_after_benchmark_execution(self, tmp_path: Path) -> None:
        """Verifies benchmark execution logic using MockLocalGemmaClient.

        (simulated-model unit test result; not an empirical benchmark).
        """
        mock_client = MockLocalGemmaClient()
        runner = QLoRABenchmarkRunner(reports_dir=tmp_path, client=mock_client)
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

        # 4. Domain breakdowns
        domains = report_dict["domain_breakdown"]
        assert "code_architecture_scout" in domains
        assert "enterprise_docs_scout" in domains
        assert domains["code_architecture_scout"]["after_qlora_pass_rate"] >= 95.0
        assert domains["enterprise_docs_scout"]["after_qlora_pass_rate"] >= 95.0
        assert domains["code_architecture_scout"]["recall_at_5_after"] >= 90.0
        assert domains["enterprise_docs_scout"]["recall_at_5_after"] >= 90.0

        # 5. Reports written to disk & honestly labeled as simulated mock run
        assert (tmp_path / "m7_qlora_benchmark_report.json").is_file()
        assert (tmp_path / "m7_qlora_benchmark_report.md").is_file()
        assert report_dict["benchmark_mode"] == "simulated_mock_client"
        assert report_dict["is_empirical"] is False
        assert "Milestone 7 Simulated QLoRA Benchmark Report" in md_report
