"""Comprehensive unit and adversarial test suite for PreFlightValidator (Milestone 5 R2).

Verifies the 4 deterministic quality checks:
1. Internal symbol resolution via AST indexing and 1-hop auto-closure.
2. Prompt identifier partitioning (VAL-01, reference vs novel symbols).
3. Normalized RRF threshold (>=0.55 or cross-encoder >=0.60).
4. Verbatim citation verification (VAL-02, >=0.98 overlap with NFKC normalization).
5. Detailed $0 bounce-back feedback generation and InsufficientContextDiagnostic terminal exit.
"""

import uuid
from typing import Final

import pytest

from context_scout.agents.preflight_gate import (
    ContextPackValidator,
    PreFlightValidator,
)
from context_scout.ast_engine.dependency_expander import ASTDependencyExpander
from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
    InMemoryRLSRepository,
)
from context_scout.schemas.contracts import (
    DocCitationSpan,
    DocumentScope,
    GateAbstentionResult,
    GatePassResult,
    ImplementationContextPack,
    InsufficientContextDiagnostic,
    ModificationIntent,
    PreFlightValidationReport,
    PrincipalContext,
    RoleType,
    VerbatimCodeSlice,
)

DOMAIN_ID: Final[str] = "code_architecture_scout"
REPO_NAME: Final[str] = "trading-execution"


@pytest.fixture
def principal() -> PrincipalContext:
    return PrincipalContext(
        user_id="lead_quant_dev",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "repo:code_architecture_scout", "corpus:enterprise_docs"],
    )


@pytest.fixture
def slicer() -> DeterministicSlicer:
    return DeterministicSlicer()


@pytest.fixture
def expander() -> ASTDependencyExpander:
    return ASTDependencyExpander(max_direct_callers=3, max_expansion_token_budget=1500)


@pytest.fixture
def test_repo(principal: PrincipalContext) -> InMemoryRLSRepository:
    repo = InMemoryRLSRepository()

    # File 1: OrderFlowImbalanceAnalyzer
    file1_id = uuid.uuid4()
    content1 = (
        "from .models import OrderBookSnapshot, ExecutionReceipt\n"
        "\n"
        "class OrderFlowImbalanceAnalyzer:\n"
        '    """Calculates order flow imbalance across market levels."""\n'
        "\n"
        "    def calculate_ofi(self, snapshot: OrderBookSnapshot) -> float:\n"
        "        spread = calc_spread_floor(snapshot)\n"
        "        return spread * 1.5\n"
    )
    repo.insert_code_file(
        principal,
        CodeFileRecord(
            file_id=file1_id,
            domain_id=DOMAIN_ID,
            repo_name=REPO_NAME,
            file_path="src/service_core/order_flow_imbalance.py",
            git_commit_sha="commit_ofi_123",
            git_blob_sha="blob_ofi_456",
            language="python",
            allowed_scopes=["public_eng"],
            raw_content=content1,
            line_count=8,
        ),
    )

    sym1_id = uuid.uuid4()
    repo.insert_ast_symbol(
        principal,
        ASTSymbolRecord(
            symbol_id=sym1_id,
            file_id=file1_id,
            domain_id=DOMAIN_ID,
            qualified_name="service_core.order_flow_imbalance.OrderFlowImbalanceAnalyzer",
            symbol_kind="class",
            start_line=3,
            end_line=8,
            signature_skeleton="class OrderFlowImbalanceAnalyzer:",
            contextual_summary="Calculates OFI metrics",
            allowed_scopes=["public_eng"],
            embedding=[0.1] * 768,
        ),
    )

    # File 2: Models (OrderBookSnapshot)
    file2_id = uuid.uuid4()
    content2 = (
        "class OrderBookSnapshot:\n"
        '    """Verbatim order book snapshot."""\n'
        "    best_bid: float\n"
        "    best_ask: float\n"
    )
    repo.insert_code_file(
        principal,
        CodeFileRecord(
            file_id=file2_id,
            domain_id=DOMAIN_ID,
            repo_name=REPO_NAME,
            file_path="src/service_core/models.py",
            git_commit_sha="commit_models_123",
            git_blob_sha="blob_models_456",
            language="python",
            allowed_scopes=["public_eng"],
            raw_content=content2,
            line_count=4,
        ),
    )

    sym2_id = uuid.uuid4()
    repo.insert_ast_symbol(
        principal,
        ASTSymbolRecord(
            symbol_id=sym2_id,
            file_id=file2_id,
            domain_id=DOMAIN_ID,
            qualified_name="service_core.models.OrderBookSnapshot",
            symbol_kind="class",
            start_line=1,
            end_line=4,
            signature_skeleton="class OrderBookSnapshot:",
            contextual_summary="Order book snapshot schema",
            allowed_scopes=["public_eng"],
            embedding=[0.2] * 768,
        ),
    )

    # Edge: OrderFlowImbalanceAnalyzer USES_TYPE OrderBookSnapshot
    repo.insert_ast_edge(
        principal,
        ASTEdgeRecord(
            edge_id=uuid.uuid4(),
            domain_id=DOMAIN_ID,
            source_symbol_id=sym1_id,
            target_qualified_name="service_core.models.OrderBookSnapshot",
            target_symbol_id=sym2_id,
            edge_kind="USES_TYPE",
            allowed_scopes=["public_eng"],
        ),
    )

    # Research Doc Chunk for Citation Verification
    chunk_id = uuid.uuid4()
    chunk_text = (
        "Across the German Xetra DAX mid-cap universe, order flow imbalance exhibits "
        "a persistent 10.5 bps mean reversion over 300 millisecond horizons."
    )
    repo.insert_doc_chunk(
        principal,
        DocChunkRecord(
            chunk_id=chunk_id,
            domain_id=DOMAIN_ID,
            doc_id="doc_ofi_eu_2026",
            doc_title="High-Frequency Order Flow Imbalance in European Equities",
            source_uri="arxiv:2603.99999",
            market_scope=DocumentScope.REGIONAL_EU,
            published_date=None,
            authority_weight=1.0,
            section_heading="Empirical Findings",
            contextual_header="Empirical analysis of OFI on Xetra",
            verbatim_content=chunk_text,
            metadata_json={"pages": [4, 5]},
            allowed_scopes=["corpus:enterprise_docs"],
            embedding=[0.3] * 768,
        ),
    )

    return repo


class TestPreFlightValidator:
    """Test suite for 4-check deterministic validation gate."""

    def test_alias_and_initialization(self) -> None:
        """Verifies ContextPackValidator alias and clean initialization."""
        assert ContextPackValidator is PreFlightValidator
        validator = PreFlightValidator()
        assert validator.min_rrf_score == 0.65
        assert validator.min_citation_overlap == 1.0

    def test_check_1_symbol_resolution_and_auto_closure(
        self,
        test_repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        slicer: DeterministicSlicer,
        expander: ASTDependencyExpander,
    ) -> None:
        """Check 1: target slice auto-attaches missing 1-hop dependencies and passes."""
        validator = PreFlightValidator(
            repo=test_repo,
            slicer=slicer,
            expander=expander,
            principal=principal,
            domain_id=DOMAIN_ID,
        )

        # Initially, pack ONLY contains the target slice
        file_rec = test_repo._files[next(iter(test_repo._files.keys()))]
        target_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service_core/order_flow_imbalance.py",
            start_line=3,
            end_line=8,
            qualified_symbol="service_core.order_flow_imbalance.OrderFlowImbalanceAnalyzer",
            selection_reason="Target class to modify",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="commit_ofi_123",
            git_blob_sha="blob_ofi_456",
            verbatim_code_with_lines=slicer.slice_lines(file_rec.raw_content, 3, 8),
            auto_expanded_by_ast=False,
        )

        code_slices = [target_slice]
        report = validator.validate(
            code_slices=code_slices,
            user_prompt="Refactor OrderFlowImbalanceAnalyzer",
            normalized_rrf_score=0.75,
        )

        assert report.passed is True
        assert len(report.unresolved_symbols) == 0
        assert report.auto_attached_ast_slices_count == 1
        # Check that OrderBookSnapshot was auto-attached into code_slices
        assert len(code_slices) == 2
        attached = code_slices[1]
        assert "OrderBookSnapshot" in attached.qualified_symbol
        assert attached.auto_expanded_by_ast is True

    def test_check_1_fails_when_internal_dependency_unresolved(
        self,
        test_repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        slicer: DeterministicSlicer,
    ) -> None:
        """Check 1: fails when an internal symbol referenced by edge does not exist in repo."""
        # Add an edge to a nonexistent internal symbol
        focus_sym_id = next(iter(test_repo._symbols.keys()))
        test_repo.insert_ast_edge(
            principal,
            ASTEdgeRecord(
                edge_id=uuid.uuid4(),
                domain_id=DOMAIN_ID,
                source_symbol_id=focus_sym_id,
                target_qualified_name="service_core.internal_secret.SecretPricingEngine",
                target_symbol_id=uuid.uuid4(),  # Does not exist in _symbols
                edge_kind="USES_TYPE",
                allowed_scopes=["public_eng"],
            ),
        )

        validator = PreFlightValidator(
            repo=test_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_ID,
        )

        file_rec = test_repo._files[next(iter(test_repo._files.keys()))]
        target_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service_core/order_flow_imbalance.py",
            start_line=3,
            end_line=8,
            qualified_symbol="service_core.order_flow_imbalance.OrderFlowImbalanceAnalyzer",
            selection_reason="Target class to modify",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="commit_ofi_123",
            git_blob_sha="blob_ofi_456",
            verbatim_code_with_lines=slicer.slice_lines(file_rec.raw_content, 3, 8),
            auto_expanded_by_ast=False,
        )

        report = validator.validate(
            code_slices=[target_slice],
            user_prompt="Refactor OrderFlowImbalanceAnalyzer",
            normalized_rrf_score=0.70,
        )

        assert report.passed is False
        assert "SecretPricingEngine" in str(report.unresolved_symbols)
        assert report.bounce_back_feedback is not None
        assert "Check 2 (Internal Symbol Resolution)" in report.bounce_back_feedback

    def test_check_2_prompt_identifier_partitioning_val_01(
        self,
        test_repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        slicer: DeterministicSlicer,
    ) -> None:
        """Check 2: VAL-01 partitions existing reference vs novel target symbols."""
        validator = PreFlightValidator(
            repo=test_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_ID,
        )

        file_rec = test_repo._files[next(iter(test_repo._files.keys()))]
        target_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service_core/order_flow_imbalance.py",
            start_line=3,
            end_line=8,
            qualified_symbol="service_core.order_flow_imbalance.OrderFlowImbalanceAnalyzer",
            selection_reason="Target class to modify",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="commit_ofi_123",
            git_blob_sha="blob_ofi_456",
            verbatim_code_with_lines=slicer.slice_lines(file_rec.raw_content, 3, 8),
            auto_expanded_by_ast=False,
        )

        # Prompt mentions:
        # - Existing reference symbol: OrderFlowImbalanceAnalyzer (in pack)
        # - Existing reference file: src/service_core/order_flow_imbalance.py (in pack)
        # - Novel target symbols to create: AdaptiveDriftRouter, calc_spread_floor (NOT in repo)
        prompt = (
            "Refactor `OrderFlowImbalanceAnalyzer` in "
            "`src/service_core/order_flow_imbalance.py` "
            "to add new class `AdaptiveDriftRouter` and helper method `calc_spread_floor`."
        )

        report = validator.validate(
            code_slices=[target_slice],
            user_prompt=prompt,
            normalized_rrf_score=0.80,
        )

        assert report.passed is True
        assert len(report.missing_reference_identifiers) == 0
        # Crucially, novel symbols are partitioned and DO NOT trigger failure
        assert "AdaptiveDriftRouter" in report.novel_target_identifiers
        assert "calc_spread_floor" in report.novel_target_identifiers

    def test_check_2_fails_when_reference_symbol_missing_from_pack(
        self,
        test_repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        slicer: DeterministicSlicer,
    ) -> None:
        """Check 2: fails when prompt references an existing repo symbol missing from pack."""
        validator = PreFlightValidator(
            repo=test_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_ID,
        )

        # Only pass OrderBookSnapshot in slice, NOT OrderFlowImbalanceAnalyzer
        file2_rec = next(f for f in test_repo._files.values() if "models.py" in f.file_path)
        model_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service_core/models.py",
            start_line=1,
            end_line=4,
            qualified_symbol="service_core.models.OrderBookSnapshot",
            selection_reason="Model slice",
            modification_intent=ModificationIntent.REFERENCE_ONLY,
            git_commit_sha="commit_models_123",
            git_blob_sha="blob_models_456",
            verbatim_code_with_lines=slicer.slice_lines(file2_rec.raw_content, 1, 4),
            auto_expanded_by_ast=False,
        )

        prompt = "Update `OrderFlowImbalanceAnalyzer` to support order book snapshot caching."
        report = validator.validate(
            code_slices=[model_slice],
            user_prompt=prompt,
            normalized_rrf_score=0.80,
        )

        assert report.passed is False
        assert "OrderFlowImbalanceAnalyzer" in report.missing_reference_identifiers
        assert report.bounce_back_feedback is not None
        assert "Check 2 (Prompt Identifier Partitioning, VAL-01)" in report.bounce_back_feedback

    def test_check_3_normalized_rrf_threshold_blk_01(
        self,
        test_repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        slicer: DeterministicSlicer,
    ) -> None:
        """Check 3 enforces Normalized RRF >= 0.55 (or cross-encoder >= 0.60)."""
        validator = PreFlightValidator(
            repo=test_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_ID,
        )

        file_rec = test_repo._files[next(iter(test_repo._files.keys()))]
        target_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service_core/order_flow_imbalance.py",
            start_line=3,
            end_line=8,
            qualified_symbol="service_core.order_flow_imbalance.OrderFlowImbalanceAnalyzer",
            selection_reason="Target class",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="commit_ofi_123",
            git_blob_sha="blob_ofi_456",
            verbatim_code_with_lines=slicer.slice_lines(file_rec.raw_content, 3, 8),
            auto_expanded_by_ast=False,
        )

        # 1. Below 0.55 threshold -> Fails
        report_low = validator.validate(
            code_slices=[target_slice],
            user_prompt="Refactor OrderFlowImbalanceAnalyzer",
            normalized_rrf_score=0.48,
        )
        assert report_low.passed is False
        assert report_low.normalized_rrf_score == 0.48
        assert "Check 3 (Normalized RRF Score)" in (report_low.bounce_back_feedback or "")

        # 2. At or above 0.65 threshold -> Passes
        report_pass = validator.validate(
            code_slices=[target_slice],
            user_prompt="Refactor OrderFlowImbalanceAnalyzer",
            normalized_rrf_score=0.65,
        )
        assert report_pass.passed is True

        # 3. Low RRF but high cross-encoder score (>= 0.60) -> Passes
        report_cross = validator.validate(
            code_slices=[target_slice],
            user_prompt="Refactor OrderFlowImbalanceAnalyzer",
            normalized_rrf_score=0.42,
            cross_encoder_score=0.68,
        )
        assert report_cross.passed is True

    def test_check_4_verbatim_citation_verification_val_02(
        self,
        test_repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        slicer: DeterministicSlicer,
    ) -> None:
        """Check 4: VAL-02 enforces 100% verbatim quote match (>=0.98 overlap, NFKC normalized)."""
        validator = PreFlightValidator(
            repo=test_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_ID,
        )

        chunk_rec = next(iter(test_repo._docs.values()))
        file_rec = next(iter(test_repo._files.values()))

        target_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service_core/order_flow_imbalance.py",
            start_line=3,
            end_line=8,
            qualified_symbol="service_core.order_flow_imbalance.OrderFlowImbalanceAnalyzer",
            selection_reason="Target class",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="commit_ofi_123",
            git_blob_sha="blob_ofi_456",
            verbatim_code_with_lines=slicer.slice_lines(file_rec.raw_content, 3, 8),
            auto_expanded_by_ast=False,
        )

        # 1. 100% Verbatim Quote -> Passes
        valid_citation = DocCitationSpan(
            chunk_id=str(chunk_rec.chunk_id),
            doc_id=chunk_rec.doc_id,
            doc_title=chunk_rec.doc_title,
            source_uri=chunk_rec.source_uri,
            section_heading=chunk_rec.section_heading,
            verbatim_quote=(
                "Across the German Xetra DAX mid-cap universe, order flow imbalance exhibits "
                "a persistent 10.5 bps mean reversion over 300 millisecond horizons."
            ),
        )

        report_valid = validator.validate(
            code_slices=[target_slice],
            user_prompt="Refactor OrderFlowImbalanceAnalyzer",
            supporting_doc_citations=[valid_citation],
            normalized_rrf_score=0.90,
        )
        assert report_valid.passed is True
        assert report_valid.citations_verified_100_pct is True

        # 2. Fabricated / Hallucinated quote -> Fails
        hallucinated_citation = DocCitationSpan(
            chunk_id=str(chunk_rec.chunk_id),
            doc_id=chunk_rec.doc_id,
            doc_title=chunk_rec.doc_title,
            source_uri=chunk_rec.source_uri,
            section_heading=chunk_rec.section_heading,
            verbatim_quote="Xetra order flow yields 45.0 bps risk-free arbitrage on all trades.",
        )

        report_invalid = validator.validate(
            code_slices=[target_slice],
            user_prompt="Refactor OrderFlowImbalanceAnalyzer",
            supporting_doc_citations=[hallucinated_citation],
            normalized_rrf_score=0.90,
        )
        assert report_invalid.passed is False
        assert report_invalid.citations_verified_100_pct is False
        assert "Check 6 (Citation Verification)" in (report_invalid.bounce_back_feedback or "")

    def test_pure_code_task_defaults_citation_verified(
        self,
        test_repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        slicer: DeterministicSlicer,
    ) -> None:
        """In pure code tasks without citations, citations_verified_100_pct is True."""
        validator = PreFlightValidator(
            repo=test_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_ID,
        )
        file_rec = next(iter(test_repo._files.values()))
        target_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service_core/order_flow_imbalance.py",
            start_line=3,
            end_line=8,
            qualified_symbol="service_core.order_flow_imbalance.OrderFlowImbalanceAnalyzer",
            selection_reason="Target class",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="commit_ofi_123",
            git_blob_sha="blob_ofi_456",
            verbatim_code_with_lines=slicer.slice_lines(file_rec.raw_content, 3, 8),
            auto_expanded_by_ast=False,
        )

        report = validator.validate(
            code_slices=[target_slice],
            user_prompt="Refactor OrderFlowImbalanceAnalyzer",
            supporting_doc_citations=[],
        )
        assert report.passed is True
        assert report.citations_verified_100_pct is True

    def test_terminal_insufficient_context_diagnostic(
        self,
        test_repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        slicer: DeterministicSlicer,
    ) -> None:
        """Verifies InsufficientContextDiagnostic enforces paid_tokens_spent == 0."""
        validator = PreFlightValidator(
            repo=test_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_ID,
        )

        # Trigger failure on Check 3
        report = validator.validate(
            code_slices=[],
            user_prompt="Refactor OrderFlowImbalanceAnalyzer",
            normalized_rrf_score=0.30,
        )
        assert report.passed is False

        diagnostic: InsufficientContextDiagnostic = validator.create_diagnostic(
            trace_id="trace_gate_test_001",
            domain_id=DOMAIN_ID,
            report=report,
            partial_candidate_files=["src/service_core/order_flow_imbalance.py"],
        )

        assert diagnostic.trace_id == "trace_gate_test_001"
        assert diagnostic.domain_id == DOMAIN_ID
        assert diagnostic.paid_tokens_spent == 0
        assert "OrderFlowImbalanceAnalyzer" in diagnostic.missing_symbols
        assert diagnostic.partial_candidate_files == ["src/service_core/order_flow_imbalance.py"]

    def test_full_pack_integration(
        self,
        test_repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        slicer: DeterministicSlicer,
    ) -> None:
        """Verifies validating a full ImplementationContextPack modifies it in-place."""
        validator = PreFlightValidator(
            repo=test_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_ID,
        )
        file_rec = next(iter(test_repo._files.values()))

        target_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service_core/order_flow_imbalance.py",
            start_line=3,
            end_line=8,
            qualified_symbol="service_core.order_flow_imbalance.OrderFlowImbalanceAnalyzer",
            selection_reason="Target class",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="commit_ofi_123",
            git_blob_sha="blob_ofi_456",
            verbatim_code_with_lines=slicer.slice_lines(file_rec.raw_content, 3, 8),
            auto_expanded_by_ast=False,
        )

        dummy_report = PreFlightValidationReport(
            passed=False,
            unresolved_symbols=[],
            missing_reference_identifiers=[],
            novel_target_identifiers=[],
            normalized_rrf_score=0.0,
            citations_verified_100_pct=False,
            auto_attached_ast_slices_count=0,
        )

        pack = ImplementationContextPack(
            trace_id="trace_pack_001",
            domain_id=DOMAIN_ID,
            user_task="Refactor OrderFlowImbalanceAnalyzer with adaptive bounds",
            architectural_execution_plan=["Step 1: hydrate OFI", "Step 2: attach models"],
            code_slices=[target_slice],
            supporting_doc_citations=[],
            preflight_report=dummy_report,
            local_tokens_consumed=450,
            pack_token_count=1200,
        )

        report = validator.validate(pack=pack, normalized_rrf_score=0.88)

        assert report.passed is True
        assert pack.preflight_report.passed is True
        assert pack.preflight_report.auto_attached_ast_slices_count == 1
        assert len(pack.code_slices) == 2

    def test_check_1_fails_on_syntax_error(
        self,
        test_repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        slicer: DeterministicSlicer,
    ) -> None:
        """Check 1: AST syntax check rejects unclosed delimiters or invalid code."""
        validator = PreFlightValidator(
            repo=test_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_ID,
        )
        malformed_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service_core/order_flow_imbalance.py",
            start_line=1,
            end_line=2,
            qualified_symbol="service_core.broken.BrokenSyntax",
            selection_reason="Broken slice",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="commit_ofi_123",
            git_blob_sha="blob_ofi_456",
            verbatim_code_with_lines="1 | def unclosed_call(\n2 |     data = [1, 2, 3",
            auto_expanded_by_ast=False,
        )
        report = validator.validate(
            code_slices=[malformed_slice],
            user_prompt="Refactor BrokenSyntax",
            normalized_rrf_score=0.90,
        )
        assert report.passed is False
        assert report.check_results["check_1_syntax"] is False
        assert any(
            "syntax" in err.lower() or "order_flow_imbalance.py" in err
            for err in report.syntax_errors
        ), f"Expected syntax error diagnostic, got {report.syntax_errors}"

    def test_check_5_fails_on_unanchored_git_provenance(
        self,
        test_repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        slicer: DeterministicSlicer,
    ) -> None:
        """Check 5: Rejects slices with placeholder or empty git commit/blob SHAs."""
        validator = PreFlightValidator(
            repo=test_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_ID,
        )
        file_rec = next(iter(test_repo._files.values()))
        unanchored_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service_core/order_flow_imbalance.py",
            start_line=3,
            end_line=8,
            qualified_symbol="service_core.order_flow_imbalance.OrderFlowImbalanceAnalyzer",
            selection_reason="Target class",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="placeholder_sha",
            git_blob_sha="blob_ofi_456",
            verbatim_code_with_lines=slicer.slice_lines(file_rec.raw_content, 3, 8),
            auto_expanded_by_ast=False,
        )
        report = validator.validate(
            code_slices=[unanchored_slice],
            user_prompt="Refactor OrderFlowImbalanceAnalyzer",
            normalized_rrf_score=0.90,
        )
        assert report.passed is False
        assert report.check_results["check_5_provenance"] is False
        assert any(
            "order_flow_imbalance.py" in item or "placeholder_sha" in item
            for item in report.unanchored_provenance_items
        ), f"Expected provenance item, got {report.unanchored_provenance_items}"

    def test_evaluate_gate_returns_gate_pass_and_gate_abstention(
        self,
        test_repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        slicer: DeterministicSlicer,
        expander: ASTDependencyExpander,
    ) -> None:
        """Verifies evaluate_gate emits GatePassResult or GateAbstentionResult with $0 spend."""
        validator = PreFlightValidator(
            repo=test_repo,
            slicer=slicer,
            expander=expander,
            principal=principal,
            domain_id=DOMAIN_ID,
        )
        file_rec = next(iter(test_repo._files.values()))
        target_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service_core/order_flow_imbalance.py",
            start_line=3,
            end_line=8,
            qualified_symbol="service_core.order_flow_imbalance.OrderFlowImbalanceAnalyzer",
            selection_reason="Target class",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="commit_ofi_123",
            git_blob_sha="blob_ofi_456",
            verbatim_code_with_lines=slicer.slice_lines(file_rec.raw_content, 3, 8),
            auto_expanded_by_ast=False,
        )

        # 1. GatePassResult when score is >= 0.65 and all checks pass
        res_pass = validator.evaluate_gate(
            code_slices=[target_slice],
            user_prompt="Refactor OrderFlowImbalanceAnalyzer",
            normalized_rrf_score=0.85,
        )
        assert isinstance(res_pass, GatePassResult)
        assert res_pass.passed is True
        assert res_pass.report.passed is True

        # 2. GateAbstentionResult when score is below 0.65 threshold
        res_fail = validator.evaluate_gate(
            code_slices=[target_slice],
            user_prompt="Refactor OrderFlowImbalanceAnalyzer",
            normalized_rrf_score=0.45,
        )
        assert isinstance(res_fail, GateAbstentionResult)
        assert res_fail.passed is False
        assert res_fail.diagnostic.paid_tokens_spent == 0
        assert "Check 3 (Normalized RRF Score" in res_fail.diagnostic.reason

    @pytest.mark.parametrize(
        "intent",
        [ModificationIntent.DIRECT_CALLER, ModificationIntent.IMPORTED_TYPE],
    )
    def test_check_1_fails_on_unclosed_delimiters_in_partial_windows(
        self,
        test_repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        slicer: DeterministicSlicer,
        intent: ModificationIntent,
    ) -> None:
        """Check 1: DIRECT_CALLER and IMPORTED_TYPE slices fail when delimiters are unclosed."""
        validator = PreFlightValidator(
            repo=test_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_ID,
        )
        bad_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service_core/order_flow_imbalance.py",
            start_line=1,
            end_line=2,
            qualified_symbol=f"service_core.test.{intent.value}",
            selection_reason="Partial slice with unclosed delimiter",
            modification_intent=intent,
            git_commit_sha="commit_ofi_123",
            git_blob_sha="blob_ofi_456",
            verbatim_code_with_lines="1 | def snippet():\n2 |     data = (1 + 2",
            auto_expanded_by_ast=False,
        )
        # 1. Direct check_ast_syntax call
        passed, errors, missing = validator.check_ast_syntax([bad_slice])
        assert passed is False
        assert len(errors) == 1
        assert "Unclosed syntax delimiters in code slice" in errors[0]
        assert any(m.kind == "syntax" for m in missing)

        # 2. End-to-end validate() call
        report = validator.validate(
            code_slices=[bad_slice],
            user_prompt=f"Refactor {bad_slice.qualified_symbol}",
            normalized_rrf_score=0.90,
        )
        assert report.passed is False
        assert report.check_results["check_1_syntax"] is False
        assert any("Unclosed syntax delimiters" in e for e in report.syntax_errors)

    @pytest.mark.parametrize(
        "intent",
        [ModificationIntent.DIRECT_CALLER, ModificationIntent.IMPORTED_TYPE],
    )
    def test_check_1_passes_balanced_partial_windows_without_module_parse_errors(
        self,
        test_repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        slicer: DeterministicSlicer,
        intent: ModificationIntent,
    ) -> None:
        """Check 1: DIRECT_CALLER and IMPORTED_TYPE pass when delimiters are balanced,
        even if the partial window is not a valid whole-module AST.
        """
        validator = PreFlightValidator(
            repo=test_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_ID,
        )
        # In TypeScript, a lone interface property `readonly x: (number | string);` fails
        # whole-module TS parser, but is balanced and must pass Check 1 partial window exemption.
        balanced_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service_core/types.ts",
            start_line=5,
            end_line=7,
            qualified_symbol=f"service_core.types.{intent.value}",
            selection_reason="Balanced partial window",
            modification_intent=intent,
            git_commit_sha="commit_ofi_123",
            git_blob_sha="blob_ofi_456",
            verbatim_code_with_lines=(
                "5 |   readonly x: (number | string);\n6 |   readonly y: boolean;"
            ),
            auto_expanded_by_ast=True,
        )
        passed, errors, missing = validator.check_ast_syntax([balanced_slice])
        assert passed is True
        assert len(errors) == 0
        assert len(missing) == 0

    def test_check_1_fails_on_nesting_depth_exceeded_in_partial_window(
        self,
        test_repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        slicer: DeterministicSlicer,
    ) -> None:
        """Check 1: DIRECT_CALLER fails when parentheses nesting depth exceeds 10 levels."""
        validator = PreFlightValidator(
            repo=test_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_ID,
        )
        deep_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service_core/order_flow_imbalance.py",
            start_line=1,
            end_line=2,
            qualified_symbol="service_core.deep.DeepCaller",
            selection_reason="Deep nesting caller",
            modification_intent=ModificationIntent.DIRECT_CALLER,
            git_commit_sha="commit_ofi_123",
            git_blob_sha="blob_ofi_456",
            verbatim_code_with_lines=(
                "1 | def deep_caller():\n2 |     return ((((((((((((42))))))))))))"
            ),
            auto_expanded_by_ast=False,
        )
        passed, errors, missing = validator.check_ast_syntax([deep_slice])
        assert passed is False
        assert len(errors) == 1
        assert "Nesting depth (12) exceeds maximum limit (10)" in errors[0]
        assert any(m.kind == "syntax" for m in missing)


def test_preflight_check3_and_check4_strict_threshold_boundaries(
    principal: PrincipalContext,
) -> None:
    """PR-4 (ECS-100): Verify strict Check 3 (0.65 RRF) and Check 4 (1.0 overlap) fail-closed."""
    from uuid import uuid4

    from context_scout.db.schema import DocChunkRecord
    from context_scout.schemas.contracts import DocCitationSpan, DocumentScope

    repo = InMemoryRLSRepository()
    validator = PreFlightValidator(repo=repo, principal=principal, domain_id=DOMAIN_ID)
    assert validator.min_rrf_score == pytest.approx(0.65)
    assert validator.min_citation_overlap == pytest.approx(1.0)

    for score in (0.55, 0.60, 0.6499):
        ok_rrf, _ = validator.check_rrf_threshold(normalized_rrf_score=score)
        assert ok_rrf is False, f"Expected RRF score {score} < 0.65 to fail Check 3"
    ok_pass, _ = validator.check_rrf_threshold(normalized_rrf_score=0.65)
    assert ok_pass is True

    # Check 4: overlap in [0.98, 1.0) must fail strict min_overlap_ratio=1.0
    cid = uuid4()
    words = [f"token{i}" for i in range(60)]
    chunk_text = " ".join(words)
    partial_quote = " ".join(words[:59]) + " wrongtoken"
    repo.insert_doc_chunk(
        principal,
        DocChunkRecord(
            chunk_id=cid,
            domain_id=DOMAIN_ID,
            doc_id="doc_thresh",
            doc_title="Threshold Spec",
            source_uri="urn:doc_thresh",
            market_scope=DocumentScope.ENTERPRISE_DOC,
            published_date=None,
            authority_weight=1.0,
            section_heading="Body",
            contextual_header="Threshold Spec - Body",
            verbatim_content=chunk_text,
            metadata_json={},
            allowed_scopes=list(principal.allowed_scopes),
            embedding=[0.1] * 768,
        ),
    )
    cit = DocCitationSpan(
        chunk_id=str(cid),
        doc_id="doc_thresh",
        doc_title="Threshold Spec",
        source_uri="urn:doc_thresh",
        section_heading="Body",
        verbatim_quote=partial_quote,
    )
    cit_rep = validator.citation_gate.verify_citations(
        [cit],
        corpus_chunks={str(cid): chunk_text},
        require_citations=True,
    )
    assert cit_rep.passed is False


def test_conflict_detector_distinguishes_qualified_parameters() -> None:
    """PR-4 (ECS-023): connect_timeout vs read_timeout or l1_ttl vs l2_ttl must not collide."""
    from dataclasses import replace

    from context_scout.agents.conflict_detector import ConflictDetector
    from context_scout.db.schema import DocChunkRecord
    from context_scout.schemas.contracts import DocumentScope

    detector = ConflictDetector()
    doc = DocChunkRecord(
        chunk_id=uuid.uuid4(),
        domain_id=DOMAIN_ID,
        doc_id="doc_cfg",
        doc_title="Config Spec",
        source_uri="urn:doc_cfg",
        source_type="enterprise_doc",
        market_scope=DocumentScope.ENTERPRISE_DOC,
        published_date=None,
        authority_weight=1.0,
        section_heading="Configuration",
        contextual_header="Config Spec - Configuration",
        verbatim_content=(
            "Configuration guide:\nread_timeout = 30s\nl2_ttl = 120s\nmetrics_port = 9090\n"
        ),
        metadata_json={},
        allowed_scopes=["public_eng"],
        embedding=[0.1] * 768,
    )
    slc = VerbatimCodeSlice(
        repo_name=REPO_NAME,
        file_path="src/cfg.py",
        start_line=1,
        end_line=3,
        qualified_symbol="cfg.settings",
        selection_reason="Config settings",
        modification_intent=ModificationIntent.REFERENCE_ONLY,
        git_commit_sha="abc1234",
        git_blob_sha="def5678",
        verbatim_code_with_lines=(
            "1 | connect_timeout = 5000\n2 | l1_ttl = 10\n3 | admin_port = 8080\n"
        ),
    )
    conflicts = detector.detect_code_vs_doc_conflicts([slc], [doc])
    assert conflicts == [], f"False parameter collision detected: {conflicts}"

    doc_match = replace(doc, verbatim_content="Configuration guide:\nconnect_timeout = 30s\n")
    slc_match = slc.model_copy(
        update={
            "verbatim_code_with_lines": "1 | connect_timeout_ms = 5000\n2 | l1_ttl = 10\n",
        }
    )
    real_conflicts = detector.detect_code_vs_doc_conflicts([slc_match], [doc_match])
    assert len(real_conflicts) == 1
