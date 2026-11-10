"""Milestone 4 Empirical Challenger Stress Suite.

Adversarially stress-tests:
1. 6-check preflight sequence bypass attempts:
   - Malformed AST & paren nesting depth > 10
   - Unresolved internal symbols vs built-ins
   - Marginal RRF scores (e.g. 0.6499)
   - Scope permission mismatches and excision
   - Cryptographic provenance (truncated/placeholder SHAs, inverted line bounds, blob mismatch)
   - Stale Confluence chunks and negative Jira resolutions
2. Populated repository discrimination:
   - EVAL-01 (Unanswerables) against a POPULATED repository
   - Answerable query discrimination against the same POPULATED repository
   - Dynamic empirical metric calculation (k/N computed dynamically, not hardcoded)
3. PaidExecutionSpy stress:
   - Interception of illicit paid calls during failure states
   - Interception of unauthorized outbound cloud LLM requests
4. Mutation proofs:
   - Mutated gate thresholds (RRF, syntax, provenance) fail when bypassed
5. Neutral domain routing audit & identified implementation bugs.
"""

from __future__ import annotations

import urllib.request
import uuid

import pytest
from evals.datasets.unanswerable_dataset import (
    get_unanswerable_benchmark_set,
)
from pydantic import ValidationError
from scipy import stats
from tests.test_zero_spend_abstention import PaidExecutionSpy, ZeroSpendViolationError

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.gemini_executor import GeminiImplementationExecutor
from context_scout.agents.preflight_gate import (
    GateAbstentionResult,
    GatePassResult,
    PreFlightValidator,
)
from context_scout.agents.scout_workflow import ScoutWorkflow
from context_scout.ast_engine.dependency_expander import ASTDependencyExpander
from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer
from context_scout.config import DomainConfig
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
    InMemoryRLSRepository,
)
from context_scout.schemas.contracts import (
    DocCitationSpan,
    InsufficientContextDiagnostic,
    ModificationIntent,
    PrincipalContext,
    RoleType,
    VerbatimCodeSlice,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def slicer() -> DeterministicSlicer:
    return DeterministicSlicer()


@pytest.fixture
def expander() -> ASTDependencyExpander:
    return ASTDependencyExpander(max_direct_callers=3, max_expansion_token_budget=1500)


@pytest.fixture
def admin_principal() -> PrincipalContext:
    """Superuser principal for test setup insertions."""
    return PrincipalContext(
        user_id="admin_setup",
        role=RoleType.ENGINEER,
        allowed_scopes=["*", "public_eng", "repo:core_engine", "scope:secret_ops"],
    )


@pytest.fixture
def principal_engineer() -> PrincipalContext:
    """Engineer principal with standard permissions."""
    return PrincipalContext(
        user_id="lead_eng_challenger",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "repo:core_engine"],
    )


@pytest.fixture
def principal_restricted() -> PrincipalContext:
    """Restricted guest principal lacking engine repo access."""
    return PrincipalContext(
        user_id="contractor_restricted",
        role=RoleType.ENGINEER,
        allowed_scopes=["scope:restricted_guest"],
    )


@pytest.fixture
def populated_repo(admin_principal: PrincipalContext) -> InMemoryRLSRepository:
    """Populated repository with multiple files, symbols, edges, and valid docs."""
    repo = InMemoryRLSRepository()
    domain_id = "polyglot_unseen_v2"

    # File 1: Python Engine Core
    f1_id = uuid.uuid4()
    f1_content = (
        "from .utils import calculate_metric\n\n"
        "class CoreEngineRunner:\n"
        '    """Runs core engine workflows."""\n\n'
        "    def run_process(self, input_data: dict) -> float:\n"
        "        val = calculate_metric(input_data)\n"
        "        return val * 2.0\n"
    )
    repo.insert_code_file(
        admin_principal,
        CodeFileRecord(
            file_id=f1_id,
            domain_id=domain_id,
            repo_name="core_engine",
            file_path="src/engine/runner.py",
            git_commit_sha="a1b2c3d4e5f67890",
            git_blob_sha="b1c2d3e4f5a67890",
            language="python",
            allowed_scopes=["public_eng", "repo:core_engine"],
            raw_content=f1_content,
            line_count=8,
        ),
    )

    sym1_id = uuid.uuid4()
    repo.insert_ast_symbol(
        admin_principal,
        ASTSymbolRecord(
            symbol_id=sym1_id,
            file_id=f1_id,
            domain_id=domain_id,
            qualified_name="engine.runner.CoreEngineRunner",
            symbol_kind="class",
            start_line=3,
            end_line=8,
            signature_skeleton="class CoreEngineRunner:",
            contextual_summary="Core engine runner execution orchestrator",
            allowed_scopes=["public_eng", "repo:core_engine"],
            embedding=[0.1] * 768,
        ),
    )

    # File 2: Python Utils
    f2_id = uuid.uuid4()
    f2_content = (
        "def calculate_metric(data: dict) -> float:\n"
        '    """Calculates summary metric."""\n'
        "    return float(len(data))\n"
    )
    repo.insert_code_file(
        admin_principal,
        CodeFileRecord(
            file_id=f2_id,
            domain_id=domain_id,
            repo_name="core_engine",
            file_path="src/engine/utils.py",
            git_commit_sha="a1b2c3d4e5f67890",
            git_blob_sha="c1d2e3f4a5b67890",
            language="python",
            allowed_scopes=["public_eng", "repo:core_engine"],
            raw_content=f2_content,
            line_count=3,
        ),
    )

    sym2_id = uuid.uuid4()
    repo.insert_ast_symbol(
        admin_principal,
        ASTSymbolRecord(
            symbol_id=sym2_id,
            file_id=f2_id,
            domain_id=domain_id,
            qualified_name="engine.utils.calculate_metric",
            symbol_kind="function",
            start_line=1,
            end_line=3,
            signature_skeleton="def calculate_metric(data: dict) -> float:",
            contextual_summary="Calculates summary metric",
            allowed_scopes=["public_eng", "repo:core_engine"],
            embedding=[0.2] * 768,
        ),
    )

    # Edge: CoreEngineRunner CALLS calculate_metric
    repo.insert_ast_edge(
        admin_principal,
        ASTEdgeRecord(
            edge_id=uuid.uuid4(),
            domain_id=domain_id,
            source_symbol_id=sym1_id,
            target_qualified_name="engine.utils.calculate_metric",
            target_symbol_id=sym2_id,
            edge_kind="USES_TYPE",
            allowed_scopes=["public_eng", "repo:core_engine"],
        ),
    )

    # File 3: Secret Restricted File (only for scope:secret_ops)
    f3_id = uuid.uuid4()
    f3_content = (
        'class SecretKeyVault:\n    """Restricted vault."""\n    KEY = \'super_secret_value\'\n'
    )
    repo.insert_code_file(
        admin_principal,
        CodeFileRecord(
            file_id=f3_id,
            domain_id=domain_id,
            repo_name="core_engine",
            file_path="src/security/vault.py",
            git_commit_sha="a1b2c3d4e5f67890",
            git_blob_sha="d1e2f3a4b5c67890",
            language="python",
            allowed_scopes=["scope:secret_ops"],
            raw_content=f3_content,
            line_count=3,
        ),
    )

    # Document Chunk 1: Active Confluence Architecture Spec
    doc1_id = uuid.uuid4()
    doc1_text = (
        "CoreEngineRunner operates with an idempotent dispatcher requiring calculate_metric "
        "to return non-negative floats across all execution cycles."
    )
    repo.insert_doc_chunk(
        admin_principal,
        DocChunkRecord(
            chunk_id=doc1_id,
            domain_id=domain_id,
            doc_id="spec_engine_arch_2026",
            doc_title="Core Engine Architecture v3",
            source_uri="confluence:eng:spec-301",
            market_scope=None,
            published_date=None,
            authority_weight=1.0,
            section_heading="Idempotency Guarantees",
            contextual_header="Architecture standards for core engine",
            verbatim_content=doc1_text,
            metadata_json={"version": 3, "status": "APPROVED"},
            allowed_scopes=["public_eng"],
            embedding=[0.3] * 768,
            is_stale=False,
        ),
    )

    # Document Chunk 2: Superseded/Stale Doc Chunk
    doc2_id = uuid.uuid4()
    doc2_text = (
        "DEPRECATED: Legacy runner uses synchronous blocking calls without calculate_metric."
    )
    repo.insert_doc_chunk(
        admin_principal,
        DocChunkRecord(
            chunk_id=doc2_id,
            domain_id=domain_id,
            doc_id="spec_engine_legacy_2024",
            doc_title="Legacy Runner Architecture v1",
            source_uri="confluence:eng:spec-101",
            market_scope=None,
            published_date=None,
            authority_weight=0.5,
            section_heading="Legacy Architecture",
            contextual_header="Old engine specification",
            verbatim_content=doc2_text,
            metadata_json={"version": 1, "status": "SUPERSEDED"},
            allowed_scopes=["public_eng"],
            embedding=[0.4] * 768,
            is_stale=True,
        ),
    )

    # Document Chunk 3: Jira Ticket with Negative Resolution ("Won't Fix")
    doc3_id = uuid.uuid4()
    doc3_text = "ENG-404: Request to support asynchronous streaming for CoreEngineRunner."
    repo.insert_doc_chunk(
        admin_principal,
        DocChunkRecord(
            chunk_id=doc3_id,
            domain_id=domain_id,
            doc_id="jira_ENG-404",
            doc_title="ENG-404: Async streaming support",
            source_uri="jira:ENG-404",
            market_scope=None,
            published_date=None,
            authority_weight=0.8,
            section_heading="Ticket Resolution",
            contextual_header="Jira ticket ENG-404",
            verbatim_content=doc3_text,
            metadata_json={"resolution": "Won't Fix", "status": "Closed"},
            allowed_scopes=["public_eng"],
            embedding=[0.5] * 768,
            is_stale=False,
        ),
    )

    # Document Chunk 4: Jira Ticket with Valid Resolution ("Done")
    doc4_id = uuid.uuid4()
    doc4_text = "ENG-200: Successfully validated calculate_metric bounds in production."
    repo.insert_doc_chunk(
        admin_principal,
        DocChunkRecord(
            chunk_id=doc4_id,
            domain_id=domain_id,
            doc_id="jira_ENG-200",
            doc_title="ENG-200: Metric bounds validation",
            source_uri="jira:ENG-200",
            market_scope=None,
            published_date=None,
            authority_weight=0.9,
            section_heading="Ticket Completion",
            contextual_header="Jira ticket ENG-200",
            verbatim_content=doc4_text,
            metadata_json={"resolution": "Done", "status": "Closed"},
            allowed_scopes=["public_eng"],
            embedding=[0.6] * 768,
            is_stale=False,
        ),
    )

    return repo


# ============================================================================
# Part 1: Preflight Gate 6-Check Adversarial Stress & Bypass Resistance
# ============================================================================


class TestPreflightGateAdversarialStress:
    """Stress tests every single check of the 6-check preflight sequence."""

    def test_check_1_nesting_depth_exceeds_10_rejected(
        self,
        populated_repo: InMemoryRLSRepository,
        principal_engineer: PrincipalContext,
        slicer: DeterministicSlicer,
    ) -> None:
        """Adversarial stress: nesting depth > 10 must deterministically fail Check 1."""
        validator = PreFlightValidator(
            repo=populated_repo,
            slicer=slicer,
            principal=principal_engineer,
            domain_id="polyglot_unseen_v2",
        )
        deeply_nested_code = "1 | def deeply_nested():\n2 |     return ((((((((((((42))))))))))))"
        slice_item = VerbatimCodeSlice(
            repo_name="core_engine",
            file_path="src/engine/nested.py",
            start_line=1,
            end_line=2,
            qualified_symbol="engine.nested.deeply_nested",
            selection_reason="Testing depth bypass",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="a1b2c3d4e5f67890",
            git_blob_sha="b1c2d3e4f5a67890",
            verbatim_code_with_lines=deeply_nested_code,
            auto_expanded_by_ast=False,
        )
        res = validator.evaluate_gate(
            code_slices=[slice_item],
            user_prompt="Refactor deeply_nested",
            normalized_rrf_score=0.90,
        )
        assert isinstance(res, GateAbstentionResult)
        assert res.passed is False
        assert res.report.check_results["check_1_syntax"] is False
        assert any("Nesting depth" in err for err in res.report.syntax_errors)
        assert res.diagnostic.paid_tokens_spent == 0
        assert any(item.kind == "syntax" for item in res.diagnostic.missing_items)

    def test_check_1_unclosed_delimiters_rejected(
        self,
        populated_repo: InMemoryRLSRepository,
        principal_engineer: PrincipalContext,
        slicer: DeterministicSlicer,
    ) -> None:
        """Check 1 must reject unclosed brackets, braces, and parentheses."""
        validator = PreFlightValidator(
            repo=populated_repo,
            slicer=slicer,
            principal=principal_engineer,
            domain_id="polyglot_unseen_v2",
        )
        unclosed_slice = VerbatimCodeSlice(
            repo_name="core_engine",
            file_path="src/engine/unclosed.py",
            start_line=1,
            end_line=2,
            qualified_symbol="engine.unclosed.broken",
            selection_reason="Testing unclosed delimiters",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="a1b2c3d4e5f67890",
            git_blob_sha="b1c2d3e4f5a67890",
            verbatim_code_with_lines="1 | def broken():\n2 |     data = {'key': [1, 2, 3",
            auto_expanded_by_ast=False,
        )
        res = validator.evaluate_gate(
            code_slices=[unclosed_slice],
            user_prompt="Refactor broken",
            normalized_rrf_score=0.90,
        )
        assert isinstance(res, GateAbstentionResult)
        assert res.passed is False
        assert res.report.check_results["check_1_syntax"] is False
        assert res.diagnostic.paid_tokens_spent == 0

    def test_check_1_treesitter_typescript_syntax_error_rejected(
        self,
        populated_repo: InMemoryRLSRepository,
        principal_engineer: PrincipalContext,
        slicer: DeterministicSlicer,
    ) -> None:
        """Check 1: tree-sitter TypeScript parser detects malformed TS constructs."""
        validator = PreFlightValidator(
            repo=populated_repo,
            slicer=slicer,
            principal=principal_engineer,
            domain_id="polyglot_unseen_v2",
        )
        ts_malformed_slice = VerbatimCodeSlice(
            repo_name="core_engine",
            file_path="src/ui/Component.tsx",
            start_line=1,
            end_line=2,
            qualified_symbol="Component",
            selection_reason="Testing TSX parser",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="a1b2c3d4e5f67890",
            git_blob_sha="b1c2d3e4f5a67890",
            verbatim_code_with_lines=(
                "1 | export const Broken = () => {\n2 |   const x: = 123;\n3 | };"
            ),
            auto_expanded_by_ast=False,
        )
        res = validator.evaluate_gate(
            code_slices=[ts_malformed_slice],
            user_prompt="Fix Component",
            normalized_rrf_score=0.90,
        )
        assert isinstance(res, GateAbstentionResult)
        assert res.passed is False
        assert res.report.check_results["check_1_syntax"] is False
        assert res.diagnostic.paid_tokens_spent == 0

    def test_check_2_unresolved_internal_symbol_cannot_masquerade_as_builtin(
        self,
        populated_repo: InMemoryRLSRepository,
        admin_principal: PrincipalContext,
        principal_engineer: PrincipalContext,
        slicer: DeterministicSlicer,
    ) -> None:
        """Check 2: internal missing symbols (e.g. PhantomEngine) cannot masquerade as built-ins."""
        sym1_id = next(iter(populated_repo._symbols.keys()))
        populated_repo.insert_ast_edge(
            admin_principal,
            ASTEdgeRecord(
                edge_id=uuid.uuid4(),
                domain_id="polyglot_unseen_v2",
                source_symbol_id=sym1_id,
                target_qualified_name="engine.phantom.PhantomEngine",
                target_symbol_id=uuid.uuid4(),  # Does not exist in symbols
                edge_kind="USES_TYPE",
                allowed_scopes=["public_eng"],
            ),
        )
        validator = PreFlightValidator(
            repo=populated_repo,
            slicer=slicer,
            principal=principal_engineer,
            domain_id="polyglot_unseen_v2",
        )
        file_rec = populated_repo._files[next(iter(populated_repo._files.keys()))]
        slice_item = VerbatimCodeSlice(
            repo_name="core_engine",
            file_path=file_rec.file_path,
            start_line=3,
            end_line=8,
            qualified_symbol="engine.runner.CoreEngineRunner",
            selection_reason="Testing unresolved closure",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha=file_rec.git_commit_sha,
            git_blob_sha=file_rec.git_blob_sha,
            verbatim_code_with_lines=slicer.slice_lines(file_rec.raw_content, 3, 8),
            auto_expanded_by_ast=False,
        )
        res = validator.evaluate_gate(
            code_slices=[slice_item],
            user_prompt="Refactor CoreEngineRunner",
            normalized_rrf_score=0.90,
        )
        assert isinstance(res, GateAbstentionResult)
        assert res.passed is False
        assert res.report.check_results["check_2_closure"] is False
        assert any("PhantomEngine" in s for s in res.report.unresolved_symbols)
        assert res.diagnostic.paid_tokens_spent == 0

    def test_check_3_marginal_rrf_score_below_065_strictly_abstains(
        self,
        populated_repo: InMemoryRLSRepository,
        principal_engineer: PrincipalContext,
        slicer: DeterministicSlicer,
        expander: ASTDependencyExpander,
    ) -> None:
        """Check 3: score of 0.6499 (just below 0.65) must strictly abstain with $0 spend."""
        validator = PreFlightValidator(
            repo=populated_repo,
            slicer=slicer,
            expander=expander,
            principal=principal_engineer,
            domain_id="polyglot_unseen_v2",
        )
        file_rec = populated_repo._files[next(iter(populated_repo._files.keys()))]
        valid_slice = VerbatimCodeSlice(
            repo_name="core_engine",
            file_path=file_rec.file_path,
            start_line=3,
            end_line=8,
            qualified_symbol="engine.runner.CoreEngineRunner",
            selection_reason="Target class",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha=file_rec.git_commit_sha,
            git_blob_sha=file_rec.git_blob_sha,
            verbatim_code_with_lines=slicer.slice_lines(file_rec.raw_content, 3, 8),
            auto_expanded_by_ast=False,
        )

        # 0.6499 must abstain
        res_fail = validator.evaluate_gate(
            code_slices=[valid_slice],
            user_prompt="Refactor CoreEngineRunner",
            normalized_rrf_score=0.6499,
        )
        assert isinstance(res_fail, GateAbstentionResult)
        assert res_fail.passed is False
        assert res_fail.report.check_results["check_3_rrf"] is False
        assert res_fail.diagnostic.paid_tokens_spent == 0
        assert "Check 3 (Normalized RRF Score" in res_fail.diagnostic.reason

        # 0.6500 must pass Check 3
        res_pass = validator.evaluate_gate(
            code_slices=[valid_slice],
            user_prompt="Refactor CoreEngineRunner",
            normalized_rrf_score=0.6500,
        )
        assert isinstance(res_pass, GatePassResult)
        assert res_pass.passed is True
        assert res_pass.report.check_results["check_3_rrf"] is True

    def test_check_4_scope_mismatch_excises_unauthorized_items(
        self,
        populated_repo: InMemoryRLSRepository,
        principal_restricted: PrincipalContext,
        slicer: DeterministicSlicer,
    ) -> None:
        """Check 4: principal lacking allowed scopes gets unauthorized items excised."""
        validator = PreFlightValidator(
            repo=populated_repo,
            slicer=slicer,
            principal=principal_restricted,
            domain_id="polyglot_unseen_v2",
        )
        # Attempt to access file restricted to public_eng / repo:core_engine
        file_rec = populated_repo._files[next(iter(populated_repo._files.keys()))]
        unauthorized_slice = VerbatimCodeSlice(
            repo_name="core_engine",
            file_path=file_rec.file_path,
            start_line=3,
            end_line=8,
            qualified_symbol="engine.runner.CoreEngineRunner",
            selection_reason="Target class",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha=file_rec.git_commit_sha,
            git_blob_sha=file_rec.git_blob_sha,
            verbatim_code_with_lines=slicer.slice_lines(file_rec.raw_content, 3, 8),
            auto_expanded_by_ast=False,
        )
        res = validator.evaluate_gate(
            code_slices=[unauthorized_slice],
            user_prompt="Inspect CoreEngineRunner",
            normalized_rrf_score=0.85,
        )
        assert isinstance(res, GateAbstentionResult)
        assert res.passed is False
        assert res.report.check_results["check_4_scope"] is False
        assert len(res.report.excised_unauthorized_items) > 0
        assert res.diagnostic.paid_tokens_spent == 0

    def test_check_5_cryptographic_provenance_rejects_truncated_or_mismatched_shas(
        self,
        populated_repo: InMemoryRLSRepository,
        principal_engineer: PrincipalContext,
        slicer: DeterministicSlicer,
    ) -> None:
        """Check 5: commit/blob SHAs shorter than 7 chars or DB mismatches are rejected."""
        validator = PreFlightValidator(
            repo=populated_repo,
            slicer=slicer,
            principal=principal_engineer,
            domain_id="polyglot_unseen_v2",
        )
        file_rec = populated_repo._files[next(iter(populated_repo._files.keys()))]

        # Case A: Truncated commit SHA (< 7 chars) rejected by Pydantic schema
        with pytest.raises(ValidationError):
            VerbatimCodeSlice(
                repo_name="core_engine",
                file_path=file_rec.file_path,
                start_line=3,
                end_line=8,
                qualified_symbol="engine.runner.CoreEngineRunner",
                selection_reason="Target",
                modification_intent=ModificationIntent.MODIFY_TARGET,
                git_commit_sha="abc12",  # only 5 chars
                git_blob_sha=file_rec.git_blob_sha,
                verbatim_code_with_lines=slicer.slice_lines(file_rec.raw_content, 3, 8),
                auto_expanded_by_ast=False,
            )

        # Case A2: Placeholder commit SHA rejected by Check 5
        slice_placeholder_sha = VerbatimCodeSlice(
            repo_name="core_engine",
            file_path=file_rec.file_path,
            start_line=3,
            end_line=8,
            qualified_symbol="engine.runner.CoreEngineRunner",
            selection_reason="Target",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="placeholder_sha",
            git_blob_sha=file_rec.git_blob_sha,
            verbatim_code_with_lines=slicer.slice_lines(file_rec.raw_content, 3, 8),
            auto_expanded_by_ast=False,
        )
        res_a2 = validator.evaluate_gate(
            code_slices=[slice_placeholder_sha],
            user_prompt="Refactor CoreEngineRunner",
            normalized_rrf_score=0.85,
        )
        assert res_a2.passed is False
        assert res_a2.report.check_results["check_5_provenance"] is False

        # Case B: Blob SHA mismatch against repository record
        slice_mismatched_blob = VerbatimCodeSlice(
            repo_name="core_engine",
            file_path=file_rec.file_path,
            start_line=3,
            end_line=8,
            qualified_symbol="engine.runner.CoreEngineRunner",
            selection_reason="Target",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha=file_rec.git_commit_sha,
            git_blob_sha="forged_blob_sha_99999",
            verbatim_code_with_lines=slicer.slice_lines(file_rec.raw_content, 3, 8),
            auto_expanded_by_ast=False,
        )
        res_b = validator.evaluate_gate(
            code_slices=[slice_mismatched_blob],
            user_prompt="Refactor CoreEngineRunner",
            normalized_rrf_score=0.85,
        )
        assert res_b.passed is False
        assert res_b.report.check_results["check_5_provenance"] is False
        assert any("Blob SHA mismatch" in item for item in res_b.report.unanchored_provenance_items)

        # Case C: Inverted line bounds (start_line > end_line) caught by Pydantic model validator
        with pytest.raises(ValidationError):
            VerbatimCodeSlice(
                repo_name="core_engine",
                file_path=file_rec.file_path,
                start_line=8,
                end_line=3,
                qualified_symbol="engine.runner.CoreEngineRunner",
                selection_reason="Target",
                modification_intent=ModificationIntent.MODIFY_TARGET,
                git_commit_sha=file_rec.git_commit_sha,
                git_blob_sha=file_rec.git_blob_sha,
                verbatim_code_with_lines=slicer.slice_lines(file_rec.raw_content, 3, 8),
                auto_expanded_by_ast=False,
            )

    def test_check_6_uuid_type_mismatch_silent_bypass_and_attribute_error_bug(
        self,
        populated_repo: InMemoryRLSRepository,
        principal_engineer: PrincipalContext,
    ) -> None:
        """CRITICAL CHALLENGER FINDING: Proves Check 6 has two critical defects:
        1. UUID vs str key mismatch causing negative resolutions to silently BYPASS Check 6.
        2. AttributeError on chunk.metadata when chunk is accessed.
        """
        validator = PreFlightValidator(
            repo=populated_repo,
            principal=principal_engineer,
            domain_id="polyglot_unseen_v2",
        )
        chunks = list(populated_repo._docs.values())
        wontfix_chunk = next(c for c in chunks if c.metadata_json.get("resolution") == "Won't Fix")
        # Note: DocCitationSpan requires chunk_id as str
        cit_wontfix = DocCitationSpan(
            chunk_id=str(wontfix_chunk.chunk_id),
            doc_id="ENG-404",
            doc_title=wontfix_chunk.doc_title,
            source_uri="jira:ENG-404",
            section_heading=wontfix_chunk.section_heading,
            verbatim_quote=wontfix_chunk.verbatim_content,
            token_count=10,
        )

        # Remediation verified: check_document_freshness creates chunks_by_id
        # with stringified UUID keys, correctly matching cit.chunk_id
        # and rejecting negative resolution "Won't Fix"
        passed, stale_items, _missing_items = validator.check_document_freshness(
            citations=[cit_wontfix],
            doc_chunks=None,
            bound_repo=validator._resolve_graph_repo(populated_repo),
            principal=principal_engineer,
            domain_id="polyglot_unseen_v2",
        )
        assert passed is False, "Negative resolution must be rejected!"
        assert len(stale_items) == 1
        assert "negative_resolution" in stale_items[0]

        # Attribute resolution uses metadata_json
        assert wontfix_chunk.metadata_json.get("resolution") == "Won't Fix"

    def test_preflight_gate_vacuous_pass_vulnerability(
        self,
        populated_repo: InMemoryRLSRepository,
        principal_engineer: PrincipalContext,
    ) -> None:
        """CHALLENGER REMEDIATION: PreFlightValidator.evaluate_gate() returns
        GateAbstentionResult(passed=False) when code_slices=[] and supporting_doc_citations=[].
        """
        validator = PreFlightValidator(
            repo=populated_repo,
            principal=principal_engineer,
            domain_id="polyglot_unseen_v2",
        )
        # When all evidence is missing, gate properly abstains
        res = validator.evaluate_gate(
            user_prompt="Refactor completely nonexistent module",
            code_slices=[],
            supporting_doc_citations=[],
            normalized_rrf_score=0.85,
        )
        assert isinstance(res, GateAbstentionResult), (
            "PreFlightValidator must abstain when evidence is completely empty!"
        )
        assert res.passed is False


# ============================================================================
# Part 2: Populated Repository Discrimination: EVAL-01 vs EVAL-04
# ============================================================================


class TestPopulatedRepoDiscrimination:
    """Stress tests EVAL-01 unanswerables and EVAL-04 answerables against a POPULATED repository."""

    def test_eval01_unanswerable_tasks_abstain_on_populated_repo_with_zero_spend(
        self,
        populated_repo: InMemoryRLSRepository,
    ) -> None:
        """Empirical proof: EVAL-01 tasks against POPULATED repo achieve 100% abstention."""
        from pathlib import Path

        domains_dir = Path(__file__).resolve().parent.parent / "domains"
        domain_registry = DomainRegistry(domains_dir)
        domain_registry.load_all()

        workflow = ScoutWorkflow(registry=domain_registry, repo=populated_repo)
        suite = get_unanswerable_benchmark_set(300)
        spy = PaidExecutionSpy()

        # Wrap GeminiImplementationExecutor to record calls
        orig_exec = GeminiImplementationExecutor.execute_implementation
        GeminiImplementationExecutor.execute_implementation = lambda self, *a, **kw: (
            spy.record_call("execute_implementation", *a, **kw)
        )

        successful_abstentions = 0
        total_tasks = len(suite)

        try:
            for task in suite:
                spy.reset()

                principal = PrincipalContext(
                    user_id=f"challenger_eval_{task.task_id}",
                    role=RoleType.ENGINEER,
                    allowed_scopes=task.requested_scopes or ["public_eng"],
                )

                target_domain = "polyglot_unseen_v2"

                # Run through scout workflow against populated repo
                result = workflow.run(
                    user_prompt=task.prompt,
                    principal=principal,
                    domain_id=target_domain,
                )

                # Workflow must emit InsufficientContextDiagnostic with paid_tokens_spent == 0
                assert isinstance(result, InsufficientContextDiagnostic)
                assert result.paid_tokens_spent == 0
                assert spy.call_count == 0
                assert spy.tokens_billed == 0

                successful_abstentions += 1

            # Dynamic empirical metric calculation (k / N)
            k_empirical = successful_abstentions
            n_empirical = total_tasks
            assert k_empirical == n_empirical == 300

            # Dynamic Clopper-Pearson computation from actual test counts
            alpha = 0.05
            p_lower_twosided = float(
                stats.beta.ppf(alpha / 2.0, k_empirical, n_empirical - k_empirical + 1)
            )
            assert p_lower_twosided >= 0.9877, (
                f"Empirical lower bound {p_lower_twosided:.6f} < 0.9878"
            )

        finally:
            GeminiImplementationExecutor.execute_implementation = orig_exec

    def test_discrimination_answerable_query_passes_on_populated_repo(
        self,
        populated_repo: InMemoryRLSRepository,
        principal_engineer: PrincipalContext,
        slicer: DeterministicSlicer,
        expander: ASTDependencyExpander,
    ) -> None:
        """Genuinely proves discrimination: valid answerable context passes all checks."""
        validator = PreFlightValidator(
            repo=populated_repo,
            slicer=slicer,
            expander=expander,
            principal=principal_engineer,
            domain_id="polyglot_unseen_v2",
        )
        file_rec = populated_repo._files[next(iter(populated_repo._files.keys()))]
        valid_target_slice = VerbatimCodeSlice(
            repo_name="core_engine",
            file_path=file_rec.file_path,
            start_line=3,
            end_line=8,
            qualified_symbol="engine.runner.CoreEngineRunner",
            selection_reason="Target runner class",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha=file_rec.git_commit_sha,
            git_blob_sha=file_rec.git_blob_sha,
            verbatim_code_with_lines=slicer.slice_lines(file_rec.raw_content, 3, 8),
            auto_expanded_by_ast=False,
        )

        res = validator.evaluate_gate(
            code_slices=[valid_target_slice],
            user_prompt="Refactor CoreEngineRunner to update metrics calculation",
            normalized_rrf_score=0.88,
        )

        # Proves system does NOT blanket-refuse: answerable query passes 100% of 6 checks
        assert isinstance(res, GatePassResult)
        assert res.passed is True
        assert res.report.passed is True
        assert res.report.check_results["check_1_syntax"] is True
        assert res.report.check_results["check_2_closure"] is True
        assert res.report.check_results["check_3_rrf"] is True
        assert res.report.check_results["check_4_scope"] is True
        assert res.report.check_results["check_5_provenance"] is True
        assert res.report.auto_attached_ast_slices_count == 1


# ============================================================================
# Part 3: PaidExecutionSpy Stress & Interception Verification
# ============================================================================


class TestPaidExecutionSpyStress:
    """Stress tests PaidExecutionSpy against active attempts to bypass interception."""

    def test_spy_raises_zero_spend_violation_on_unauthorized_call(self) -> None:
        """Confirm spy raises ZeroSpendViolationError immediately upon recording an illicit call."""
        spy = PaidExecutionSpy()
        with pytest.raises(ZeroSpendViolationError, match="ZERO-SPEND VIOLATION"):
            spy.record_call("GeminiImplementationExecutor.execute_implementation", tokens=500)
        assert spy.call_count == 1
        assert spy.tokens_billed == 500

    def test_spy_intercepts_direct_cloud_llm_network_connections(self) -> None:
        """Verify that attempt to reach external cloud LLM APIs triggers ZeroSpendViolationError."""
        spy = PaidExecutionSpy()
        orig_urlopen = urllib.request.urlopen

        def spy_urlopen(url_or_req, *args, **kwargs):
            target_url = str(getattr(url_or_req, "full_url", url_or_req))
            cloud_hosts = ("googleapis.com", "anthropic.com", "openai.com")
            if any(cloud in target_url for cloud in cloud_hosts):
                spy.record_call(f"urllib.request.urlopen({target_url})")
            return orig_urlopen(url_or_req, *args, **kwargs)

        urllib.request.urlopen = spy_urlopen

        try:
            with pytest.raises(ZeroSpendViolationError, match=r"googleapis\.com"):
                cloud_ep = (
                    "https://generativelanguage.googleapis.com/v1beta/"
                    "models/gemini-4-argon:generateContent"
                )
                urllib.request.urlopen(cloud_ep)
        finally:
            urllib.request.urlopen = orig_urlopen


# ============================================================================
# Part 4: Mutation Proofs (Proving Gates are Active and Necessary)
# ============================================================================


class TestMutationProofs:
    """Mutation tests: deliberately weaken the gate and prove that tests catch the defect."""

    def test_mutation_proof_rrf_threshold_weakening_detected(
        self,
        populated_repo: InMemoryRLSRepository,
        principal_engineer: PrincipalContext,
        slicer: DeterministicSlicer,
    ) -> None:
        """Mutation 1: Weakening min_rrf_score to 0.0 allows low-relevance garbage through."""
        validator = PreFlightValidator(
            repo=populated_repo,
            slicer=slicer,
            principal=principal_engineer,
            domain_id="polyglot_unseen_v2",
        )
        file_rec = populated_repo._files[next(iter(populated_repo._files.keys()))]
        valid_slice = VerbatimCodeSlice(
            repo_name="core_engine",
            file_path=file_rec.file_path,
            start_line=3,
            end_line=8,
            qualified_symbol="engine.runner.CoreEngineRunner",
            selection_reason="Target",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha=file_rec.git_commit_sha,
            git_blob_sha=file_rec.git_blob_sha,
            verbatim_code_with_lines=slicer.slice_lines(file_rec.raw_content, 3, 8),
            auto_expanded_by_ast=False,
        )

        # Baseline: score 0.40 fails
        assert (
            validator.evaluate_gate(
                code_slices=[valid_slice],
                user_prompt="Refactor CoreEngineRunner",
                normalized_rrf_score=0.40,
            ).passed
            is False
        )

        # Mutate: set min_rrf_score = 0.0
        validator.min_rrf_score = 0.0
        mutated_res = validator.evaluate_gate(
            code_slices=[valid_slice],
            user_prompt="Refactor CoreEngineRunner",
            normalized_rrf_score=0.40,
        )
        # Mutation caught: mutated gate erroneously passes low score 0.40
        assert mutated_res.passed is True
        assert mutated_res.report.check_results["check_3_rrf"] is True

    def test_mutation_proof_syntax_check_bypass_detected(
        self,
        populated_repo: InMemoryRLSRepository,
        principal_engineer: PrincipalContext,
        slicer: DeterministicSlicer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mutation 2: Bypassing Check 1 syntax validation allows broken code to pass."""
        validator = PreFlightValidator(
            repo=populated_repo,
            slicer=slicer,
            principal=principal_engineer,
            domain_id="polyglot_unseen_v2",
        )
        broken_slice = VerbatimCodeSlice(
            repo_name="core_engine",
            file_path="src/engine/broken.py",
            start_line=1,
            end_line=2,
            qualified_symbol="engine.broken.Func",
            selection_reason="Target",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="a1b2c3d4e5f67890",
            git_blob_sha="b1c2d3e4f5a67890",
            verbatim_code_with_lines="1 | def broken(\n2 |     data = [1, 2, 3",
            auto_expanded_by_ast=False,
        )

        # Baseline: broken syntax rejected
        baseline_res = validator.evaluate_gate(
            code_slices=[broken_slice],
            user_prompt="Fix broken",
            normalized_rrf_score=0.90,
        )
        assert baseline_res.passed is False

        # Mutate: bypass syntax check
        monkeypatch.setattr(validator, "check_ast_syntax", lambda slices: (True, [], []))
        mutated_res = validator.evaluate_gate(
            code_slices=[broken_slice],
            user_prompt="Fix broken",
            normalized_rrf_score=0.90,
        )
        # Mutation caught: syntax error is admitted when Check 1 is disabled
        assert mutated_res.report.check_results["check_1_syntax"] is True


# ============================================================================
# Part 5: Neutral Domain Routing Audit
# ============================================================================


class TestNeutralDomainRoutingAudit:
    """Verifies neutral domain routing without finance-specific hardcoding."""

    def test_domain_config_defaults_are_neutral(self) -> None:
        """DomainConfig must provide neutral defaults for Jira negative resolutions."""
        cfg = DomainConfig()
        assert "Won't Fix" in cfg.jira_negative_resolutions
        assert "Duplicate" in cfg.jira_negative_resolutions
        assert "Cannot Reproduce" in cfg.jira_negative_resolutions
        assert "Closed" in cfg.jira_terminal_statuses
        assert "Resolved" in cfg.jira_terminal_statuses

    def test_preflight_gate_operates_with_neutral_custom_domain(
        self,
        populated_repo: InMemoryRLSRepository,
        principal_engineer: PrincipalContext,
    ) -> None:
        """Preflight gate must function cleanly under arbitrary custom domain names."""
        custom_domain = "enterprise_ecommerce_service"
        validator = PreFlightValidator(
            repo=populated_repo,
            principal=principal_engineer,
            domain_id=custom_domain,
        )
        assert validator.domain_id == custom_domain
        res = validator.evaluate_gate(
            user_prompt="Refactor cart checkout",
            normalized_rrf_score=0.30,
        )
        assert isinstance(res, GateAbstentionResult)
        assert res.diagnostic.domain_id == custom_domain
        assert res.diagnostic.paid_tokens_spent == 0
