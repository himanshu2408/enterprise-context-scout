"""Unit, mutation, and negative-control tests for model governance and evaluation integrity."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
from evals.datasets.token_ab_dataset import (
    count_bpe_tokens,
    get_token_ab_benchmark_set,
)
from evals.harnesses.statistical_evaluator import compute_token_ab_significance

from context_scout.config import ModelGovernancePolicy, ScoutSettings, validate_model_id
from context_scout.db.repositories import InMemoryRLSRepository
from context_scout.evals.harness import (
    EvaluationHarness,
    EvaluationRunResult,
    MetricResult,
)
from context_scout.evals.runner import generate_markdown_report
from context_scout.schemas.contracts import (
    DocCitationSpan,
    ImplementationContextPack,
    PreFlightValidationReport,
)
from context_scout.tuning.teacher_distiller import TeacherDistiller

REPO_ROOT = Path(__file__).resolve().parents[1]


class _DummyTokenizer:
    """Test double implementing Hugging Face AutoTokenizer.apply_chat_template."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        assert tokenize is False
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


class _DummyRepoWithoutDocLookup:
    """Repository stub lacking doc-chunk lookup methods for fail-closed verification."""


def _sample_pack_with_citation(chunk_id: str, quote: str) -> ImplementationContextPack:
    return ImplementationContextPack(
        trace_id="trace-m0-02-neg-ctrl",
        domain_id="enterprise_docs_scout",
        user_task="Verify citation faithfulness fail-closed behavior",
        architectural_execution_plan=["Inspect cited chunk verbatim"],
        code_slices=[],
        supporting_doc_citations=[
            DocCitationSpan(
                chunk_id=chunk_id,
                doc_id="doc-404",
                doc_title="Missing Document",
                source_uri="urn:docs:missing",
                section_heading="1. Overview",
                verbatim_quote=quote,
            )
        ],
        preflight_report=PreFlightValidationReport(
            passed=True,
            normalized_rrf_score=0.92,
            citations_verified_100_pct=True,
        ),
        local_tokens_consumed=1000,
        pack_token_count=120,
    )


def test_model_governance_policy_defaults_and_byo_families() -> None:
    """Default Gemma stack + all approved BYO open-weights families are accepted."""
    policy = ModelGovernancePolicy.from_registry_file(REPO_ROOT / "config" / "model_registry.yaml")
    assert policy.allow_external_cloud_egress is False
    assert policy.require_safetensors_only is True
    assert policy.trust_remote_code is False

    local_and_byo_models = [
        "google/gemma-4-26B-A4B-it",
        "mlx-community/gemma-4-26b-a4b-it-4bit",
        "google/gemma-4-E4B-it",
        "mlx-community/gemma-4-e4b-it-4bit",
        "google/gemma-4-12B-it",
        "google/gemma-4-31B-it",
        "google/embeddinggemma-2",
        "google/shieldgemma-2-2b-it",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-Coder-30B-A3B",
        "mlx-community/Qwen3-Coder-30B-A3B-4bit",
        "mistralai/Devstral-Small-2505",
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "meta-llama/Llama-4-Scout-17B-16E",
        "microsoft/Phi-4",
        "nvidia/Nemotron-H-8B-Base-8K",
        "ibm-granite/granite-3.3-8b-instruct",
        "Qwen/Qwen3-Embedding-8B",
        "BAAI/bge-m3",
        "BAAI/bge-reranker-v2-m3",
        "Qwen/Qwen3-Reranker-8B",
        "meta-llama/Llama-Prompt-Guard-2-86M",
        "Qwen/Qwen3Guard-8B",
        "ibm-granite/granite-guardian-3.2-5b",
    ]
    for model_id in local_and_byo_models:
        assert policy.validate_local_model_id(model_id) == model_id
        assert validate_model_id(model_id) == model_id

    for retired in (
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "codegemma",
        "codegemma-7b",
        "text-embedding-004",
        "google/gemma-2-9b-it",
        "google/gemma-3-12b-it",
        "gemini-ultra",
    ):
        with pytest.raises(ValueError, match="Prohibited or retired model ID"):
            policy.validate_model_id(retired)

    for unapproved in ("gpt-4o", "claude-3-5-sonnet", "meta-llama/Llama-3-8b", ""):
        with pytest.raises(ValueError):
            policy.validate_model_id(unapproved)


def test_model_governance_cloud_egress_default_and_mutation_proof() -> None:
    """Cloud egress fails closed by default and opens only when policy flag is mutated True."""
    default_policy = ModelGovernancePolicy.default()
    assert default_policy.allow_external_cloud_egress is False

    for handoff_id in ("gemini-3.8-flash", "gemini-4-argon", "claude-sonnet-5.5", "gpt-5.6-sol"):
        with pytest.raises(ValueError, match="allow_external_cloud_egress=False"):
            default_policy.validate_cloud_egress(handoff_id)

        with pytest.raises(ValueError, match="not permitted for local self-hosted execution"):
            default_policy.validate_local_model_id(handoff_id)

    # Mutation proof: enabling allow_external_cloud_egress unlocks registered handoff models only
    enabled_policy = ModelGovernancePolicy(allow_external_cloud_egress=True)
    assert enabled_policy.validate_cloud_egress("gemini-3.8-flash") == "gemini-3.8-flash"
    assert enabled_policy.validate_cloud_egress("claude-sonnet-5.5") == "claude-sonnet-5.5"
    with pytest.raises(ValueError, match="Prohibited or retired model ID"):
        enabled_policy.validate_cloud_egress("gemini-2.5-pro")

    settings = ScoutSettings()
    assert settings.allow_external_cloud_egress is False


def test_model_governance_safetensors_trust_remote_code_and_spdx_mutations() -> None:
    """Supply-chain policy blocks unsafe weight formats, remote code, and non-approved SPDX."""
    policy = ModelGovernancePolicy.default()

    # Valid safetensors + clean config passes
    assert policy.validate_weights_safety(
        ["model-00001-of-00002.safetensors", "tokenizer.json"],
        config_payload={"architectures": ["Qwen3ForCausalLM"]},
    ) == ["model-00001-of-00002.safetensors"]

    # Rejects unsafe pickle/binary weights even when a safetensors shard is present
    for bad_file in ("pytorch_model.bin", "weights.pt", "adapter.pkl", "model.ckpt"):
        with pytest.raises(ValueError, match=r"requires \.safetensors weights only"):
            policy.validate_weights_safety(["model.safetensors", bad_file])

    # Rejects empty safetensors list
    with pytest.raises(ValueError, match=r"at least one \.safetensors weight shard"):
        policy.validate_weights_safety(["config.json", "tokenizer.json"])

    # Rejects trust_remote_code=True or auto_map in config.json
    with pytest.raises(ValueError, match="trust_remote_code=True"):
        policy.validate_weights_safety(["model.safetensors"], trust_remote_code=True)
    with pytest.raises(ValueError, match="auto_map"):
        policy.validate_weights_safety(
            ["model.safetensors"],
            config_payload={"auto_map": {"AutoModelForCausalLM": "custom.CustomModel"}},
        )

    # Mutation proof: disabling checks in a mutated policy allows what default policy blocks
    relaxed = ModelGovernancePolicy(require_safetensors_only=False, trust_remote_code=True)
    assert relaxed.validate_weights_safety(["pytorch_model.bin"], trust_remote_code=True) == []

    # SPDX allowlist validation
    for ok_spdx in ("Apache-2.0", "MIT", "BSD-3-Clause", "gemma", "qwen"):
        assert policy.validate_spdx_license(ok_spdx) == ok_spdx
    for bad_spdx in ("GPL-3.0-only", "AGPL-3.0", "Proprietary", ""):
        with pytest.raises(ValueError):
            policy.validate_spdx_license(bad_spdx)


def test_model_governance_chat_template_delegates_to_tokenizer() -> None:
    """format_chat_prompt uses tokenizer.apply_chat_template without hardcoded Gemma tags."""
    policy = ModelGovernancePolicy.default()
    rendered = policy.format_chat_prompt(
        [{"role": "user", "content": "Locate `ModelGovernancePolicy`"}],
        tokenizer=_DummyTokenizer(),
        add_generation_prompt=True,
    )
    assert "<|im_start|>user\nLocate `ModelGovernancePolicy`<|im_end|>" in rendered
    assert rendered.endswith("<|im_start|>assistant\n")
    assert "<start_of_turn>" not in rendered


def test_data_tuning_and_mock_generator_deleted_and_3way_splits_disjoint() -> None:
    """Verify data/tuning & mock_gemma_generator are deleted and domains splits are 3-way."""
    assert not (REPO_ROOT / "data" / "tuning").exists()
    assert not (REPO_ROOT / "src" / "context_scout" / "tuning" / "mock_gemma_generator.py").exists()

    for dom_dir in sorted((REPO_ROOT / "domains").iterdir()):
        tuning_dir = dom_dir / "tuning"
        if not tuning_dir.is_dir():
            continue
        hashes: dict[str, str] = {}
        for split in ("train.jsonl", "valid.jsonl", "eval.jsonl"):
            fp = tuning_dir / split
            assert fp.is_file() and fp.stat().st_size > 0, f"Missing {fp}"
            hashes[split] = hashlib.sha256(fp.read_bytes()).hexdigest()
        assert len(set(hashes.values())) == 3, f"{dom_dir.name} has duplicate split SHA-256"

        assert (
            TeacherDistiller.verify_zero_leakage(
                tuning_dir / "train.jsonl",
                tuning_dir / "valid.jsonl",
                eval_path=tuning_dir / "eval.jsonl",
                domain_id=dom_dir.name,
            )
            is True
        )

    proc = subprocess.run(
        [sys.executable, ".agents/skills/byo-lora-dryrun/scripts/check_splits_and_ceiling.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_eval_harness_missing_chunk_fails_closed_zero_percent() -> None:
    """Negative control: missing chunk_id scores 0.0% and fails faithfulness gate."""
    harness = EvaluationHarness(settings=ScoutSettings(), force_in_memory=True)
    missing_pack = _sample_pack_with_citation(
        chunk_id="nonexistent-chunk-id-000",
        quote="Some quote that must NEVER self-verify against itself.",
    )
    res = harness.evaluate_verbatim_citation_faithfulness([missing_pack])
    assert res.measured_value == 0.0
    assert res.passed is False
    assert res.details["total_doc_citations"] == 1
    assert res.details["valid_doc_citations"] == 0
    assert len(res.details["failures"]) == 1
    assert res.details["failures"][0]["chunk_id"] == "nonexistent-chunk-id-000"

    # Negative control 2: repo lacking doc-chunk lookup methods also fails closed (0%)
    harness_no_lookup = EvaluationHarness(
        settings=ScoutSettings(),
        repo=_DummyRepoWithoutDocLookup(),
    )
    res_no_lookup = harness_no_lookup.evaluate_verbatim_citation_faithfulness([missing_pack])
    assert res_no_lookup.measured_value == 0.0
    assert res_no_lookup.passed is False


def test_eval_runner_distinguishes_in_memory_from_live_pg17() -> None:
    """Eval runner markdown never claims PG17 FORCE RLS when using InMemoryRLSRepository."""
    harness = EvaluationHarness(settings=ScoutSettings(), force_in_memory=True)
    assert isinstance(harness.repo, InMemoryRLSRepository)

    metrics = {
        "zero_touch_sufficiency_rate": MetricResult(
            metric_id="zero_touch_sufficiency_rate",
            display_name="Zero-Touch Sufficiency Rate",
            measured_value=100.0,
            target_threshold=85.0,
            comparator=">=",
            passed=True,
        ),
        "file_and_symbol_recall_at_5": MetricResult(
            metric_id="file_and_symbol_recall_at_5",
            display_name="File & Symbol Recall@5",
            measured_value=100.0,
            target_threshold=90.0,
            comparator=">=",
            passed=True,
        ),
        "verbatim_citation_faithfulness": MetricResult(
            metric_id="verbatim_citation_faithfulness",
            display_name="Verbatim Citation Faithfulness",
            measured_value=100.0,
            target_threshold=100.0,
            comparator="==",
            passed=True,
        ),
        "frontier_token_savings_ratio": MetricResult(
            metric_id="frontier_token_savings_ratio",
            display_name="Frontier Token Savings Ratio",
            measured_value=90.0,
            target_threshold=80.0,
            comparator=">=",
            passed=True,
        ),
        "rbac_zero_leakage_rate": MetricResult(
            metric_id="rbac_zero_leakage_rate",
            display_name="RBAC Zero-Leakage Rate",
            measured_value=100.0,
            target_threshold=100.0,
            comparator="==",
            passed=True,
        ),
        "prompt_injection_block_rate": MetricResult(
            metric_id="prompt_injection_block_rate",
            display_name="Prompt-Injection Block Rate",
            measured_value=100.0,
            target_threshold=95.0,
            comparator=">=",
            passed=True,
        ),
    }
    result = EvaluationRunResult(
        timestamp="2026-11-18T00:00:00Z",
        all_passed=True,
        metrics=metrics,
        summary="ALL GATES PASSED",
        total_duration_ms=5,
        telemetry_spans_count=0,
    )

    md = generate_markdown_report(result, harness)
    assert "InMemoryRLSRepository" in md
    assert "live PostgreSQL 17 `FORCE ROW LEVEL SECURITY` not active" in md

    # Mutation proof: when _connection_manager is active with a non-in-memory repo, report PG17
    harness._connection_manager = object()  # type: ignore[assignment]
    harness.repo = object()
    md_pg = generate_markdown_report(result, harness)
    assert "Tested against live PostgreSQL 17 `FORCE ROW LEVEL SECURITY`." in md_pg


def test_token_ab_dataset_uses_real_bpe_tokenization() -> None:
    """Token A/B dataset measures token savings via real BPE tokenization."""
    short_tokens = count_bpe_tokens("def f(): return 1")
    long_tokens = count_bpe_tokens(
        "def compute_reconciliation(records: list[dict[str, float]]) -> float:\n"
        "    return sum(r['amount'] for r in records if r['amount'] > 0.0)\n"
    )
    assert short_tokens > 0
    assert long_tokens > short_tokens

    tasks = get_token_ab_benchmark_set()
    assert len(tasks) == 120
    ab_res = compute_token_ab_significance(
        [t.arm_a_exploration_tokens for t in tasks],
        [t.arm_b_scouted_tokens for t in tasks],
    )
    assert ab_res.median_savings_pct >= 80.0
    assert ab_res.p_value < 0.01
    assert ab_res.is_significant is True
    assert ab_res.target_reduction_met is True


def test_runner_markdown_report_uses_dynamic_configured_models() -> None:
    """Verifies generate_markdown_report reflects configured primary/frontier models dynamically."""
    custom_settings = ScoutSettings(
        primary_model_id="Qwen/Qwen3-Coder-30B-A3B",
        frontier_model_id="claude-sonnet-5.5",
        allow_external_cloud_egress=True,
    )
    harness = EvaluationHarness(settings=custom_settings, force_in_memory=True)
    metrics = {
        "zero_touch_sufficiency_rate": MetricResult(
            metric_id="zero_touch_sufficiency_rate",
            display_name="Zero-Touch Sufficiency Rate",
            measured_value=100.0,
            target_threshold=85.0,
            comparator=">=",
            passed=True,
        ),
        "file_and_symbol_recall_at_5": MetricResult(
            metric_id="file_and_symbol_recall_at_5",
            display_name="File & Symbol Recall@5",
            measured_value=100.0,
            target_threshold=90.0,
            comparator=">=",
            passed=True,
        ),
        "verbatim_citation_faithfulness": MetricResult(
            metric_id="verbatim_citation_faithfulness",
            display_name="Verbatim Citation Faithfulness",
            measured_value=100.0,
            target_threshold=100.0,
            comparator="==",
            passed=True,
        ),
        "frontier_token_savings_ratio": MetricResult(
            metric_id="frontier_token_savings_ratio",
            display_name="Frontier Token Savings Ratio",
            measured_value=90.0,
            target_threshold=80.0,
            comparator=">=",
            passed=True,
        ),
        "rbac_zero_leakage_rate": MetricResult(
            metric_id="rbac_zero_leakage_rate",
            display_name="RBAC Zero-Leakage Rate",
            measured_value=100.0,
            target_threshold=100.0,
            comparator="==",
            passed=True,
        ),
        "prompt_injection_block_rate": MetricResult(
            metric_id="prompt_injection_block_rate",
            display_name="Prompt-Injection Block Rate",
            measured_value=100.0,
            target_threshold=95.0,
            comparator=">=",
            passed=True,
        ),
    }
    result = EvaluationRunResult(
        timestamp="2026-11-19T00:00:00Z",
        all_passed=True,
        metrics=metrics,
        summary="ALL GATES PASSED",
        total_duration_ms=5,
        telemetry_spans_count=0,
    )

    md = generate_markdown_report(result, harness)
    assert "Qwen/Qwen3-Coder-30B-A3B" in md
    assert "$0.00 / 1M input" in md
    assert "claude-sonnet-5.5" in md
    assert "$3.00 / 1M input" in md
    assert "gemini-3.8-flash" not in md
    assert "gemini-4-argon" not in md


def test_run_compact_gates_executes_pytest_with_nonzero_passed() -> None:
    """run_compact_gates.sh must not shadow heredoc stdin and must run >0 tests."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = (
        repo_root / ".agents" / "skills" / "gate-verify" / "scripts" / "run_compact_gates.sh"
    )
    script_text = script_path.read_text(encoding="utf-8")
    assert "run_step" in script_text
    assert 'python3 - "$' not in script_text.split("run_step()", 1)[1].split('if [[ "$SCOPE"', 1)[1]
    assert 'totals["passed"] == 0' in script_text
    assert 'counts["passed"] == 0' in script_text

    proc = subprocess.run(
        [
            "bash",
            str(script_path),
            "--scope",
            "py",
            "--pytest-target",
            "tests/test_domain_neutrality_guard.py",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "passed, 0 failed, 0 skipped, 0 errors" in proc.stdout
