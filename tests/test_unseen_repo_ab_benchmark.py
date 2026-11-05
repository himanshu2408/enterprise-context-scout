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
from fastapi import HTTPException

from context_scout.agents.gemini_executor import (
    GeminiImplementationExecutor,
    apply_search_replace_or_hunk_blocks,
)
from context_scout.agents.preflight_gate import BoundGraphRepository, PreFlightValidator
from context_scout.api.dependencies import (
    create_signed_principal_token,
    get_principal_context,
    verify_signed_principal_token,
)
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
    RepoExplorationTools,
    TaskABComparison,
    UnseenRepoABBenchmarkRunner,
    UnseenRepoABReport,
    _extract_unified_diff,
    check_file_recall,
    check_symbol_recall,
    compute_distribution_stats,
    compute_frontier_cost_usd,
    format_unseen_ab_markdown_report,
)
from context_scout.evals.unseen_repo_dataset import (
    FROZEN_CODE_ADAPTER_SHA256,
    FROZEN_QUANT_ADAPTER_SHA256,
    UnseenRepoBugTask,
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


class TestWorktreePatchVerificationAndEnterpriseHardening:
    """Regression tests for worktree git apply on parent_commit_sha, HMAC auth, and SQL pushdown."""

    def test_verify_task_in_worktree_pytest_applies_diff_to_parent_commit_not_merge_commit(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "mini_repo"
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
        # Parent commit has a broken function and a test that fails on parent_commit_sha
        (repo_dir / "calc.py").write_text(
            "def add(a: int, b: int) -> int:\n    return a - b\n",
            encoding="utf-8",
        )
        (repo_dir / "test_calc.py").write_text(
            "from calc import add\n\ndef test_add() -> None:\n    assert add(2, 3) == 5\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), "commit", "-m", "buggy parent"],
            check=True,
            capture_output=True,
        )
        parent_sha = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        # Merge commit fixes calc.py
        (repo_dir / "calc.py").write_text(
            "def add(a: int, b: int) -> int:\n    return a + b\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), "commit", "-m", "fixed merge"],
            check=True,
            capture_output=True,
        )
        merge_sha = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        task = UnseenRepoBugTask(
            task_id="MINI-01",
            repo_slug="encode/httpx",
            pr_number=1,
            pr_url="https://github.com/encode/httpx/pull/1",
            merge_commit_sha=merge_sha,
            parent_commit_sha=parent_sha,
            title="Fix add",
            problem_statement="Fix `add` in `calc.py`",
            ground_truth_files=("calc.py",),
            ground_truth_symbols=("calc.add",),
            test_files=("test_calc.py",),
            verify_with_pytest=True,
        )
        runner = UnseenRepoABBenchmarkRunner(
            repos_base_dir=tmp_path,
            repo_root=Path("."),
            prefer_gemini_api=False,
            verify_postgres_rls=False,
            run_live_frozen_qlora=False,
        )

        # 1. Empty/non-diff response MUST fail even when file_recall and symbol_recall are True
        arm_no_diff = ArmExecutionResult(
            arm="B_context_scout",
            task_id="MINI-01",
            repo_slug="encode/httpx",
            execution_model_id="test",
            token_accounting_source="mlx_tokenizer_exact_bpe",
            turns_used=1,
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            estimated_frontier_cost_usd=0.0,
            wall_clock_latency_ms=10.0,
            localized_files=["calc.py"],
            localized_symbols=["calc.add"],
            file_recall_at_5=True,
            symbol_recall_at_5=True,
            preflight_gate_passed=True,
            generated_patch_diff="Localized calc.py and calc.add without a unified diff",
        )
        passed_empty, msg_empty = runner.verify_task_in_worktree_pytest(task, repo_dir, arm_no_diff)
        assert passed_empty is False
        assert "No valid unified diff" in msg_empty

        # 2. Valid unified diff (even with '   1 | ' slice prefixes) applied on parent_sha passes
        valid_diff = (
            "```diff\n"
            "--- a/calc.py\n"
            "+++ b/calc.py\n"
            "@@ -1,2 +1,2 @@\n"
            "    1 | def add(a: int, b: int) -> int:\n"
            "-   2 |     return a - b\n"
            "+   2 |     return a + b\n"
            "```"
        )
        assert "return a + b" in _extract_unified_diff(valid_diff)
        arm_with_diff = arm_no_diff.model_copy(update={"generated_patch_diff": valid_diff})
        passed_valid, msg_valid = runner.verify_task_in_worktree_pytest(
            task, repo_dir, arm_with_diff
        )
        assert passed_valid is True, f"Expected pytest pass on patched parent, got: {msg_valid}"

    def test_hmac_signed_principal_token_and_require_signed_auth_enforcement(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token = create_signed_principal_token(
            user_id="alice_sec",
            role=RoleType.QUANT_RESEARCHER,
            allowed_scopes=["public_quant", "quant_restricted"],
            secret_key="unit-test-hmac-secret",
            ttl_seconds=300,
        )
        principal = verify_signed_principal_token(token, secret_key="unit-test-hmac-secret")
        assert principal.user_id == "alice_sec"
        assert principal.role == RoleType.QUANT_RESEARCHER
        assert "quant_restricted" in principal.allowed_scopes

        # Tampered signature must raise HTTP 401
        with pytest.raises(HTTPException) as exc_info:
            verify_signed_principal_token(token + "00", secret_key="unit-test-hmac-secret")
        assert exc_info.value.status_code == 401

        # Expired token must raise HTTP 401
        expired_tok = create_signed_principal_token(
            user_id="alice_sec",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
            secret_key="unit-test-hmac-secret",
            ttl_seconds=-10,
        )
        with pytest.raises(HTTPException) as exc_exp:
            verify_signed_principal_token(expired_tok, secret_key="unit-test-hmac-secret")
        assert exc_exp.value.status_code == 401

        # When SCOUT_REQUIRE_SIGNED_AUTH=true, unsigned X-Principal-* headers are rejected with 401
        monkeypatch.setenv("SCOUT_REQUIRE_SIGNED_AUTH", "true")
        monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", "unit-test-hmac-secret")
        with pytest.raises(HTTPException) as exc_unsigned:
            get_principal_context(
                x_principal_user_id="spoofed_admin",
                x_principal_role="compliance_officer",
                x_principal_scopes="quant_restricted",
                authorization=None,
            )
        assert exc_unsigned.value.status_code == 401

        # Valid Bearer token succeeds under SCOUT_REQUIRE_SIGNED_AUTH=true
        authed = get_principal_context(authorization=f"Bearer {token}")
        assert authed.user_id == "alice_sec"
        assert authed.role == RoleType.QUANT_RESEARCHER

    def test_bounded_ast_symbol_search_and_commit_isolated_merkle_indexing(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "iso_repo"
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
        (repo_dir / "engine.py").write_text("def compute_alpha() -> int:\n    return 1\n")
        subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), "commit", "-m", "c1"], check=True, capture_output=True
        )
        sha1 = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        (repo_dir / "engine.py").write_text("def compute_beta() -> int:\n    return 2\n")
        subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), "commit", "-m", "c2"], check=True, capture_output=True
        )
        sha2 = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        mem_repo = InMemoryRLSRepository()
        principal = PrincipalContext(
            user_id="test_eng",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        indexer = GitMerkleIndexer()
        indexer.sync_in_memory(
            repo_dir=repo_dir,
            repo=mem_repo,
            principal=principal,
            domain_id="trading_code_scout",
            repo_name="iso_repo",
            ref=sha1,
            isolate_by_commit=True,
        )
        indexer.sync_in_memory(
            repo_dir=repo_dir,
            repo=mem_repo,
            principal=principal,
            domain_id="trading_code_scout",
            repo_name="iso_repo",
            ref=sha2,
            isolate_by_commit=True,
        )

        # Both commit snapshots coexist without overwriting each other
        all_files = mem_repo.list_visible_code_files(principal, "trading_code_scout")
        repo_names = {f.repo_name for f in all_files}
        assert f"iso_repo@{sha1[:12]}" in repo_names
        assert f"iso_repo@{sha2[:12]}" in repo_names

        bound = BoundGraphRepository(mem_repo)
        alpha_hits = bound.search_visible_ast_symbols(
            principal, "trading_code_scout", "compute_alpha", limit=5
        )
        assert len(alpha_hits) == 1
        assert "compute_alpha" in alpha_hits[0].qualified_name

        fetched_files = bound.get_code_files_by_ids(
            principal, "trading_code_scout", [alpha_hits[0].file_id]
        )
        assert len(fetched_files) == 1
        assert fetched_files[0].repo_name == f"iso_repo@{sha1[:12]}"

    def test_context_only_diff_and_sibling_path_traversal_rejected(self, tmp_path: Path) -> None:
        # Context-only truncated diff (e.g., CLICK-09 docstring hunk without +/- lines) is rejected
        context_only_diff = (
            "```diff\n"
            "--- a/src/click/_utils.py\n"
            "+++ b/src/click/_utils.py\n"
            "@@ -12,7 +12,7 @@\n"
            "         `PEP 661 - Sentinel Values <https://peps.python.org/pep-0661/>`_.\n"
            '     """\n'
            "```"
        )
        assert _extract_unified_diff(context_only_diff) == ""

        # Sibling directory prefix traversal (e.g., /tmp/repo vs /tmp/repo_secrets) is blocked
        repo_dir = tmp_path / "repo"
        sibling_dir = tmp_path / "repo_secrets"
        repo_dir.mkdir()
        sibling_dir.mkdir()
        secret_file = sibling_dir / "key.py"
        secret_file.write_text("SECRET = 'top_secret'\n", encoding="utf-8")

        tools = RepoExplorationTools(repo_dir=repo_dir, source_subdir=".")
        out = tools.read_file_lines("../repo_secrets/key.py", 1, 10)
        assert out.startswith("File not found:")
        assert "top_secret" not in out

    def test_search_replace_and_hunk_fallback_patch_application(self, tmp_path: Path) -> None:
        wt_dir = tmp_path / "wt_sr"
        wt_dir.mkdir()
        target_f = wt_dir / "mod.py"
        target_f.write_text(
            "def compute(x: int) -> int:\n    if x is None:\n        return -1\n    return x * 2\n",
            encoding="utf-8",
        )
        sr_patch = (
            "### File: mod.py\n"
            "<<<<<<< SEARCH\n"
            "   2 |     if x is None:\n"
            "   3 |         return -1\n"
            "=======\n"
            "    if x is None:\n"
            "        return 0\n"
            ">>>>>>> REPLACE\n"
        )
        ok, msg = apply_search_replace_or_hunk_blocks(sr_patch, wt_dir)
        assert ok is True
        assert "SEARCH/REPLACE" in msg
        assert "return 0" in target_f.read_text(encoding="utf-8")
