"""Regression and unit tests for the Unseen-Repository A/B Benchmark and M6 fixes.

Covers:
1. INJ-11 passive Google API key secret-leakage redaction (35-40 char suffix after AIza)
   and 22/22 (100.0%) EvaluationHarness prompt-injection & secret-redaction gate.
2. GeminiImplementationExecutor._build_gemini_prompt verbatim slice hydration inclusion.
3. Frozen QLoRA adapter SHA-256 integrity verification (verify_frozen_adapter_integrity).
4. Unseen-repo AST generalizations:
   - ASTExtractor class-scoped import filtering on multi-import modules
   - ASTDependencyExpander header slicing for large (>30-line) imported classes under SEC-03
   - PreFlightValidator relative import resolution and prose-word false-positive prevention
5. UnseenRepoBenchmarkSuite (N=30 tasks across encode/httpx and pallets/click) and
   statistical aggregation (median, IQR, Recall@5, Markdown report formatting).
"""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

import pytest

from context_scout.agents.gemini_executor import GeminiImplementationExecutor
from context_scout.agents.preflight_gate import PreFlightValidator
from context_scout.ast_engine.dependency_expander import ASTDependencyExpander
from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer
from context_scout.ast_engine.extractor import ASTExtractor
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    InMemoryRLSRepository,
)
from context_scout.evals.golden_dataset import get_default_golden_suite
from context_scout.evals.harness import EvaluationHarness
from context_scout.evals.unseen_ab_benchmark import (
    ArmExecutionResult,
    TaskABComparison,
    UnseenRepoABReport,
    check_file_recall,
    check_symbol_recall,
    compute_distribution_stats,
    compute_frontier_cost_usd,
    format_unseen_ab_markdown_report,
)
from context_scout.evals.unseen_repo_dataset import (
    FROZEN_CODE_ADAPTER_SHA256,
    FROZEN_QUANT_ADAPTER_SHA256,
    build_unseen_repo_suite,
    verify_frozen_adapter_integrity,
)
from context_scout.ingestion.git_merkle_indexer import GitMerkleIndexer
from context_scout.schemas.contracts import (
    ImplementationContextPack,
    ModificationIntent,
    PreFlightValidationReport,
    PrincipalContext,
    RoleType,
    VerbatimCodeSlice,
)
from context_scout.security.guardrails import SecretRedactor, ShieldGemmaGuardrail


class TestINJ11SecretRedactionAndGateRegression:
    """Regression tests for INJ-11 passive Google API key redaction and 22/22 gate pass."""

    def test_inj_11_google_api_key_redacted_and_blocked(self) -> None:
        suite = get_default_golden_suite()
        inj_11 = next(p for p in suite.prompt_injection_payloads if p.payload_id == "INJ-11")

        redacted_text, labels = SecretRedactor.redact(inj_11.raw_content)
        assert "GOOGLE_API_KEY" in labels
        assert "AIzaSyD" not in redacted_text

        guardrail = ShieldGemmaGuardrail()
        decision = guardrail.evaluate(inj_11.raw_content)
        assert not decision.passed
        assert "GOOGLE_API_KEY" in decision.redacted_tokens

    def test_evaluation_harness_prompt_injection_gate_is_22_of_22(self) -> None:
        suite = get_default_golden_suite()
        harness = EvaluationHarness()
        res = harness.evaluate_prompt_injection_block_rate(suite.prompt_injection_payloads)

        assert res.passed is True
        assert res.measured_value == 100.0
        assert res.details["total_payloads"] == 22
        assert res.details["blocked_count"] == 22
        assert res.details["prompt_injection_subcheck"] == {"total": 18, "blocked": 18}
        assert res.details["secret_redaction_subcheck"] == {"total": 4, "redacted": 4}


class TestGeminiExecutorPromptHydrationRegression:
    """Regression test ensuring _build_gemini_prompt includes hydrated code_slices."""

    def test_build_gemini_prompt_includes_all_verbatim_code_slices(self) -> None:
        executor = GeminiImplementationExecutor()
        slice_1 = VerbatimCodeSlice(
            repo_name="click",
            file_path="src/click/utils.py",
            start_line=410,
            end_line=415,
            qualified_symbol="click.utils._make_default_short_help",
            selection_reason="Target function",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="d036881798e34289a49b18a5c550c7cf687e5a7a",
            git_blob_sha="a" * 40,
            verbatim_code_with_lines=(
                " 410 | def _make_default_short_help(help: str, max_length: int = 45) -> str:\n"
                " 411 |     return help"
            ),
            auto_expanded_by_ast=False,
        )
        pack = ImplementationContextPack(
            trace_id="trace_test_slice_prompt",
            domain_id="trading_code_scout",
            user_task="Fix short help abbreviation",
            code_slices=[slice_1],
            supporting_doc_citations=[],
            architectural_execution_plan=["Update _make_default_short_help"],
            preflight_report=PreFlightValidationReport(
                passed=True,
                unresolved_symbols=[],
                missing_reference_identifiers=[],
                novel_target_identifiers=[],
                normalized_rrf_score=0.91,
                citations_verified_100_pct=True,
                auto_attached_ast_slices_count=0,
            ),
            local_tokens_consumed=250,
            pack_token_count=120,
        )
        prompt = executor._build_gemini_prompt(pack)
        assert "src/click/utils.py" in prompt
        assert "click.utils._make_default_short_help" in prompt
        assert "def _make_default_short_help" in prompt


class TestUnseenRepoASTAndPreFlightGeneralization:
    """Regression tests for the 3 unseen-repo AST/PreFlight bugs discovered on httpx and click."""

    def test_ast_extractor_filters_unused_module_imports_per_class(self) -> None:
        extractor = ASTExtractor()
        source = (
            "from .exceptions import Abort, Exit, MissingParameter, UsageError\n"
            "from .utils import echo\n\n"
            "class Option:\n"
            "    def raise_missing(self) -> None:\n"
            "        raise MissingParameter()\n\n"
            "class Sentinel:\n"
            "    def __copy__(self) -> object:\n"
            "        return self\n"
        )
        _symbols, edges = extractor.parse_source(source, "python", file_path="src/click/core.py")
        option_imports = {
            e.target_qualified_name
            for e in edges
            if e.source_qualified_name == "click.core.Option" and e.edge_kind == "IMPORTS"
        }
        sentinel_imports = {
            e.target_qualified_name
            for e in edges
            if e.source_qualified_name == "click.core.Sentinel" and e.edge_kind == "IMPORTS"
        }
        assert ".exceptions.MissingParameter" in option_imports
        assert ".exceptions.Abort" not in option_imports
        assert len(sentinel_imports) == 0

    def test_dependency_expander_caps_large_imported_class_slice_within_sec03_budget(
        self,
    ) -> None:
        repo = InMemoryRLSRepository()
        principal = PrincipalContext(
            user_id="test_eng",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        large_lines = ["class HugeResponse:"] + [f"    attr_{i}: int = {i}" for i in range(1, 200)]
        f_id = uuid.uuid4()
        repo.insert_code_file(
            principal,
            CodeFileRecord(
                file_id=f_id,
                domain_id="trading_code_scout",
                repo_name="httpx",
                file_path="httpx/_models.py",
                git_commit_sha="a" * 40,
                git_blob_sha="b" * 40,
                language="python",
                allowed_scopes=["public_eng"],
                raw_content="\n".join(large_lines),
                line_count=len(large_lines),
            ),
        )
        focus_id = uuid.uuid4()
        target_id = uuid.uuid4()
        focus_sym = ASTSymbolRecord(
            symbol_id=focus_id,
            file_id=f_id,
            domain_id="trading_code_scout",
            qualified_name="httpx._decoders.ZStandardDecoder",
            symbol_kind="class",
            start_line=1,
            end_line=10,
            signature_skeleton="class ZStandardDecoder:",
            contextual_summary="Decoder",
            allowed_scopes=["public_eng"],
            embedding=[0.1] * 768,
        )
        huge_sym = ASTSymbolRecord(
            symbol_id=target_id,
            file_id=f_id,
            domain_id="trading_code_scout",
            qualified_name="httpx._models.HugeResponse",
            symbol_kind="class",
            start_line=1,
            end_line=200,
            signature_skeleton="class HugeResponse:",
            contextual_summary="Huge response class",
            allowed_scopes=["public_eng"],
            embedding=[0.1] * 768,
        )
        repo.insert_ast_symbol(principal, focus_sym)
        repo.insert_ast_symbol(principal, huge_sym)
        repo.insert_ast_edge(
            principal,
            ASTEdgeRecord(
                edge_id=uuid.uuid4(),
                domain_id="trading_code_scout",
                source_symbol_id=focus_id,
                target_qualified_name=".._models.HugeResponse",
                target_symbol_id=None,
                edge_kind="IMPORTS",
                allowed_scopes=["public_eng"],
            ),
        )

        expander = ASTDependencyExpander()
        res = expander.expand_1hop(
            focus_symbol=focus_sym,
            repo=repo,
            principal=principal,
            domain_id="trading_code_scout",
        )
        assert len(res.imported_type_slices) == 1
        assert res.imported_type_slices[0].qualified_symbol == "httpx._models.HugeResponse"
        assert (res.imported_type_slices[0].end_line - res.imported_type_slices[0].start_line) <= 25
        assert not res.budget_exhausted

    def test_preflight_validator_ignores_unquoted_prose_words_matching_short_method_names(
        self,
    ) -> None:
        repo = InMemoryRLSRepository()
        principal = PrincipalContext(
            user_id="test_eng",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        f_id = uuid.uuid4()
        repo.insert_code_file(
            principal,
            CodeFileRecord(
                file_id=f_id,
                domain_id="trading_code_scout",
                repo_name="click",
                file_path="src/click/utils.py",
                git_commit_sha="a" * 40,
                git_blob_sha="b" * 40,
                language="python",
                allowed_scopes=["public_eng"],
                raw_content="def close():\n    pass\ndef _make_default_short_help():\n    pass\n",
                line_count=4,
            ),
        )
        sym_close = ASTSymbolRecord(
            symbol_id=uuid.uuid4(),
            file_id=f_id,
            domain_id="trading_code_scout",
            qualified_name="click.utils._LazyFile.close",
            symbol_kind="method",
            start_line=1,
            end_line=2,
            signature_skeleton="def close():",
            contextual_summary="close",
            allowed_scopes=["public_eng"],
            embedding=[0.1] * 768,
        )
        sym_target = ASTSymbolRecord(
            symbol_id=uuid.uuid4(),
            file_id=f_id,
            domain_id="trading_code_scout",
            qualified_name="click.utils._make_default_short_help",
            symbol_kind="function",
            start_line=3,
            end_line=4,
            signature_skeleton="def _make_default_short_help():",
            contextual_summary="short help",
            allowed_scopes=["public_eng"],
            embedding=[0.1] * 768,
        )
        repo.insert_ast_symbol(principal, sym_close)
        repo.insert_ast_symbol(principal, sym_target)

        validator = PreFlightValidator(
            repo=repo,
            slicer=DeterministicSlicer(),
            principal=principal,
            domain_id="trading_code_scout",
        )
        slices = [
            VerbatimCodeSlice(
                repo_name="click",
                file_path="src/click/utils.py",
                start_line=3,
                end_line=4,
                qualified_symbol="click.utils._make_default_short_help",
                selection_reason="Target",
                modification_intent=ModificationIntent.MODIFY_TARGET,
                git_commit_sha="a" * 40,
                git_blob_sha="b" * 40,
                verbatim_code_with_lines="   3 | def _make_default_short_help():\n   4 |     pass",
                auto_expanded_by_ast=False,
            )
        ]
        # Prompt uses unquoted English word "close" in prose ("close the issue")
        missing, _novel = validator.partition_prompt_identifiers(
            user_prompt="Fix `_make_default_short_help` to close the issue.",
            visible_symbols=[sym_close, sym_target],
            visible_files=repo.list_visible_code_files(principal, "trading_code_scout"),
            code_slices=slices,
        )
        assert "close" not in missing
        assert len(missing) == 0

    def test_git_merkle_indexer_strips_nul_bytes_from_binary_fixtures(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "mini_httpx"
        repo_dir.mkdir()
        subprocess.run(["git", "-C", str(repo_dir), "init"], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), "config", "user.email", "test@example.com"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "config", "user.name", "Test"],
            check=True,
            capture_output=True,
        )
        (repo_dir / "fixture.zst").write_bytes(b"hello\x00world\x00binary")
        (repo_dir / "mod.py").write_text("def decode() -> None:\n    pass\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), "commit", "-m", "init"],
            check=True,
            capture_output=True,
        )
        repo = InMemoryRLSRepository()
        principal = PrincipalContext(
            user_id="test_eng",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        indexer = GitMerkleIndexer()
        indexer.sync_in_memory(
            repo_dir=repo_dir,
            repo=repo,
            principal=principal,
            domain_id="trading_code_scout",
            repo_name="mini_httpx",
            allowed_scopes=["public_eng"],
        )
        files = repo.list_visible_code_files(principal, "trading_code_scout")
        assert len(files) == 2
        for f in files:
            assert "\x00" not in f.raw_content


class TestUnseenRepoBenchmarkSuiteAndStats:
    """Tests for the 30-task dataset, frozen adapter SHA-256 check, and report formatter."""

    def test_unseen_repo_suite_has_30_verified_tasks_and_no_workstation_paths(self) -> None:
        suite = build_unseen_repo_suite()
        assert len(suite.tasks) == 30
        assert len(suite.httpx_tasks) == 15
        assert len(suite.click_tasks) == 15
        assert len(suite.pytest_verification_tasks) == 10

        for task in suite.tasks:
            assert len(task.merge_commit_sha) == 40
            assert len(task.parent_commit_sha) == 40
            assert task.merge_commit_sha != task.parent_commit_sha
            assert "/Users/" not in task.problem_statement
            assert "/home/" not in task.problem_statement

    def test_verify_frozen_adapter_integrity_passes_and_detects_tampering(
        self, tmp_path: Path
    ) -> None:
        if os.getenv("GITHUB_ACTIONS") != "true":
            res = verify_frozen_adapter_integrity(Path("."))
            assert res["verified"] is True
            assert res["code_adapter_sha256"] == FROZEN_CODE_ADAPTER_SHA256
            assert res["quant_adapter_sha256"] == FROZEN_QUANT_ADAPTER_SHA256

        # Tamper check on temporary copy
        code_dir = tmp_path / "domains/trading_code_scout/adapters/gemma4_26b_code_scout"
        quant_dir = tmp_path / "domains/quant_research_scout/adapters/gemma4_26b_quant_research"
        code_dir.mkdir(parents=True)
        quant_dir.mkdir(parents=True)
        (code_dir / "adapters.safetensors").write_bytes(b"tampered")
        (quant_dir / "adapters.safetensors").write_bytes(b"tampered")

        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            verify_frozen_adapter_integrity(tmp_path)

    def test_distribution_stats_and_markdown_formatter(self) -> None:
        stats = compute_distribution_stats([100.0, 200.0, 300.0, 400.0, 500.0])
        assert stats.median == 300.0
        assert stats.iqr_25 == 200.0
        assert stats.iqr_75 == 400.0
        assert stats.iqr_span == 200.0
        assert stats.mean == 300.0

        assert check_file_recall(["httpx/_urls.py", "httpx/_models.py"], ("httpx/_urls.py",))
        assert check_symbol_recall(["click.core.Command.main"], ("click.core.Command",))
        assert compute_frontier_cost_usd(1_000_000, 1_000_000) == 4.5

        arm_a = ArmExecutionResult(
            arm="A_direct_exploration",
            task_id="HTTPX-01",
            repo_slug="encode/httpx",
            execution_model_id="mlx-community/gemma-4-26b-a4b-it-4bit",
            token_accounting_source="mlx_tokenizer_exact_bpe",
            turns_used=5,
            input_tokens=12000,
            output_tokens=300,
            total_tokens=12300,
            estimated_frontier_cost_usd=0.010125,
            wall_clock_latency_ms=4200.0,
            localized_files=["httpx/_decoders.py"],
            localized_symbols=["httpx._decoders.ZStandardDecoder"],
            file_recall_at_5=True,
            symbol_recall_at_5=True,
            preflight_gate_passed=True,
        )
        arm_b = ArmExecutionResult(
            arm="B_context_scout",
            task_id="HTTPX-01",
            repo_slug="encode/httpx",
            execution_model_id="mlx-community/gemma-4-26b-a4b-it-4bit",
            token_accounting_source="mlx_tokenizer_exact_bpe",
            turns_used=1,
            input_tokens=1100,
            output_tokens=96,
            total_tokens=1196,
            estimated_frontier_cost_usd=0.001185,
            wall_clock_latency_ms=1900.0,
            scout_indexing_and_preflight_ms=320.0,
            frozen_qlora_json_valid=True,
            frozen_qlora_latency_ms=1400.0,
            localized_files=["httpx/_decoders.py"],
            localized_symbols=["httpx._decoders.ZStandardDecoder"],
            file_recall_at_5=True,
            symbol_recall_at_5=True,
            preflight_gate_passed=True,
        )
        comp = TaskABComparison(
            task_id="HTTPX-01",
            repo_slug="encode/httpx",
            pr_number=3412,
            pr_url="https://github.com/encode/httpx/pull/3412",
            title="Handle empty zstd responses",
            ground_truth_files=["httpx/_decoders.py"],
            ground_truth_symbols=["httpx._decoders.ZStandardDecoder"],
            arm_a=arm_a,
            arm_b=arm_b,
            input_token_savings_pct=90.83,
            total_token_savings_pct=90.28,
            outcome_classification="BOTH_HIT",
        )
        report = UnseenRepoABReport(
            benchmark_id="test_ab",
            generated_at_utc="2026-11-04T23:00:00Z",
            sample_size_n=1,
            repos_evaluated=["encode/httpx", "pallets/click"],
            frozen_code_adapter_sha256=FROZEN_CODE_ADAPTER_SHA256,
            frozen_qlora_json_valid_count=1,
            frozen_qlora_json_valid_pct=100.0,
            execution_model_id="mlx-community/gemma-4-26b-a4b-it-4bit",
            token_accounting_source="mlx_tokenizer_exact_bpe",
            postgres_rls_verified=True,
            arm_a_input_tokens=stats,
            arm_b_input_tokens=stats,
            arm_a_total_tokens=stats,
            arm_b_total_tokens=stats,
            input_token_savings_pct=stats,
            total_token_savings_pct=stats,
            arm_a_latency_ms=stats,
            arm_b_latency_ms=stats,
            arm_a_file_recall_at_5_count=1,
            arm_a_file_recall_at_5_pct=100.0,
            arm_b_file_recall_at_5_count=1,
            arm_b_file_recall_at_5_pct=100.0,
            arm_a_symbol_recall_at_5_count=1,
            arm_a_symbol_recall_at_5_pct=100.0,
            arm_b_symbol_recall_at_5_count=1,
            arm_b_symbol_recall_at_5_pct=100.0,
            arm_a_pytest_pass_count=1,
            arm_b_pytest_pass_count=1,
            pytest_subset_size=1,
            arm_a_total_cost_usd=0.010125,
            arm_b_total_cost_usd=0.001185,
            honest_losses_and_failures=[],
            task_comparisons=[comp],
        )
        md = format_unseen_ab_markdown_report(report)
        assert "# Unseen-Repository Two-Arm A/B Benchmark Report" in md
        assert FROZEN_CODE_ADAPTER_SHA256 in md
        assert "HTTPX-01" in md
