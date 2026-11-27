"""Tier 5 White-Box Adversarial Stress Testing & Coverage Hardening (Milestone 6).

Adversarial stress verification across 5 core components:
1. TestPreFlightGateAdversarial:
   - AST syntax check: nesting depth limits (>10), unclosed string quotes,
     trailing comma multi-line call truncation handling.
   - Principal scope mirroring: admin vs restricted scopes, domain prefix resolution,
     missing slices excision.
   - Cryptographic provenance: short/invalid SHA anchors, line bounds validation,
     phantom files.
   - Document freshness: negative Jira resolutions (CANCELLED/WON'T FIX) and stale chunks.
2. TestClaimVerifierAdversarial:
   - Token overlap exact matching: case sensitivity, punctuation stripping, negation.
   - Homoglyph detection: mixed-script lookalikes rejected; authentic non-Latin Unicode accepted.
   - Candidate source polymorphism: dicts, tuples, strings, DocChunkRecord.
   - Policy matrix: strict abstention zero-spend invariant across all policies.
3. TestGeminiExecutorSandboxAdversarial:
   - GitWorktreeSandbox: SEARCH/REPLACE block parsing with line number stripping,
     unified diff line drift matching.
   - Environment sanitization: stripping sensitive env vars.
   - Confirmation gate: unconfirmed merge attempts block git commit.
   - Inactive sandbox operations: raise RuntimeError.
4. TestCostGovernanceAdversarial:
   - Boundary token conditions: tiny tokens (1-2 tokens) never invert low <= expected <= high.
   - October 2026 pricing tables: exact rates for gemini-4-argon, gemini-3.8-flash,
     claude-opus-5.5, gpt-5.6-sol.
   - SessionBudgetTracker kill-switch: 250k token exploration limit and $25.00 spend cap.
   - Concurrency: thread-safe reservation under concurrent requests.
5. TestOpenTelemetryAdversarial:
   - 4-phase spans: retrieval, slicing, gating, execution lifecycle and attributes.
   - Abstention invariant: scout.is_abstention=True, scout.paid_tokens_spent=0, scout.cost_usd=0.0.
   - W3C context propagation: valid traceparent injection/extraction, corrupted fallback.
"""

from __future__ import annotations

import concurrent.futures
import os
import subprocess
import uuid
from pathlib import Path
from typing import Final
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import ReadableSpan

from context_scout.agents.claim_verifier import (
    ClaimFailureCode,
    ClaimVerificationPolicy,
    ClaimVerifier,
    detect_homoglyph_spoofing,
)
from context_scout.agents.gemini_executor import (
    GitWorktreeSandbox,
)
from context_scout.agents.preflight_gate import (
    PreFlightValidator,
)
from context_scout.config import ClaimVerifierConfig, DomainConfig
from context_scout.db.repositories import (
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
    InMemoryRLSRepository,
)
from context_scout.schemas.contracts import (
    DocCitationSpan,
    InsufficientContextDiagnostic,
    ModificationIntent,
    PreFlightValidationReport,
    PrincipalContext,
    RoleType,
    VerbatimCodeSlice,
)
from context_scout.telemetry import (
    ATTR_SCOUT_COST_USD,
    ATTR_SCOUT_IS_ABSTENTION,
    ATTR_SCOUT_PAID_TOKENS_SPENT,
    ATTR_SCOUT_PHASE,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    PHASE_EXECUTION,
    PHASE_GATING,
    PHASE_RETRIEVAL,
    PHASE_SLICING,
    ScoutTelemetryTracer,
    extract_trace_context,
    inject_trace_context,
)
from context_scout.telemetry.cost_governance import (
    DEFAULT_EXPLORATION_TOKEN_KILL_SWITCH,
    DEFAULT_SESSION_SPEND_CEILING_USD,
    OCTOBER_2026_PRICING,
    BudgetExceededException,
    KillSwitchTriggeredException,
    PaidComputeCostEstimator,
    SessionBudgetTracker,
)

DOMAIN_ID: Final[str] = "code_architecture_scout"
REPO_NAME: Final[str] = "trading-execution"


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def base_principal() -> PrincipalContext:
    """Standard authenticated engineer principal."""
    return PrincipalContext(
        user_id="adv_challenger",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "repo:trading-execution"],
        max_session_paid_tokens=25_000,
    )


@pytest.fixture
def in_memory_repo(base_principal: PrincipalContext) -> InMemoryRLSRepository:
    """Populated InMemoryRLSRepository for preflight gate tests."""
    repo = InMemoryRLSRepository()
    file_id = uuid.uuid4()
    code_content = "class OrderBook:\n    def get_spread(self) -> float:\n        return 0.5\n"
    repo.insert_code_file(
        base_principal,
        CodeFileRecord(
            file_id=file_id,
            domain_id=DOMAIN_ID,
            repo_name=REPO_NAME,
            file_path="src/order_book.py",
            git_commit_sha="a1b2c3d4e5f6",
            git_blob_sha="f6e5d4c3b2a1",
            language="python",
            allowed_scopes=["public_eng", "repo:trading-execution"],
            raw_content=code_content,
            line_count=3,
        ),
    )
    repo.insert_ast_symbol(
        base_principal,
        ASTSymbolRecord(
            symbol_id=uuid.uuid4(),
            file_id=file_id,
            domain_id=DOMAIN_ID,
            qualified_name="trading_execution.order_book.OrderBook",
            symbol_kind="class",
            start_line=1,
            end_line=3,
            signature_skeleton="class OrderBook:",
            contextual_summary="Verbatim OrderBook structure",
            allowed_scopes=["public_eng", "repo:trading-execution"],
            embedding=[0.1] * 768,
        ),
    )
    return repo


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Creates a temporary initialized git repository with initial commit."""
    repo_dir = tmp_path / "sandbox_adversarial_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Adversarial Challenger"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "challenger@scout.internal"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )

    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    target_file = src_dir / "module.py"
    target_file.write_text(
        "def compute_fee(amount: float) -> float:\n"
        "    base_rate = 0.001\n"
        "    return amount * base_rate\n",
        encoding="utf-8",
    )

    subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    return repo_dir


# =============================================================================
# 1. TestPreFlightGateAdversarial
# =============================================================================


class TestPreFlightGateAdversarial:
    """Adversarial stress testing of PreFlightValidator quality and safety checks."""

    def test_ast_syntax_nesting_depth_limit_boundary(self) -> None:
        """Nesting depth > 10 levels must trigger MALFORMED_SYNTAX and fail check."""
        validator = PreFlightValidator()

        # 11 levels of nested parentheses
        deeply_nested_code = "1 | x = (((((((((((42)))))))))))\n"
        deep_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/nested.py",
            start_line=1,
            end_line=1,
            qualified_symbol="trading.nested.x",
            selection_reason="Test depth",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="a1b2c3d4e5f6",
            git_blob_sha="f6e5d4c3b2a1",
            verbatim_code_with_lines=deeply_nested_code,
        )

        passed, errors, missing = validator.check_ast_syntax([deep_slice])
        assert passed is False
        assert any("Nesting depth" in err and "exceeds maximum limit" in err for err in errors)
        assert len(missing) == 1
        assert missing[0].kind == "syntax"

        # 10 levels of nested parentheses must pass the nesting limit check
        valid_depth_code = "1 | x = ((((((((((42))))))))))\n"
        valid_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/nested_valid.py",
            start_line=1,
            end_line=1,
            qualified_symbol="trading.nested.x_ok",
            selection_reason="Test depth valid",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="a1b2c3d4e5f6",
            git_blob_sha="f6e5d4c3b2a1",
            verbatim_code_with_lines=valid_depth_code,
        )
        passed_valid, errors_valid, _ = validator.check_ast_syntax([valid_slice])
        assert passed_valid is True
        assert len(errors_valid) == 0

    def test_ast_syntax_unclosed_string_quote_rejected(self) -> None:
        """Unclosed string literal in code slice must be detected and rejected."""
        validator = PreFlightValidator()
        malformed_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/query.py",
            start_line=1,
            end_line=2,
            qualified_symbol="trading.query.sql",
            selection_reason="Unclosed quote test",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="a1b2c3d4e5f6",
            git_blob_sha="f6e5d4c3b2a1",
            verbatim_code_with_lines='1 | sql = "SELECT * FROM orders\n2 | execute(sql)\n',
        )
        passed, errors, missing = validator.check_ast_syntax([malformed_slice])
        assert passed is False
        assert any("Unclosed syntax delimiters" in err for err in errors)
        assert len(missing) == 1

    def test_ast_syntax_trailing_comma_multiline_call_truncation(self) -> None:
        """Valid partial caller window ending in trailing comma passes syntax check."""
        validator = PreFlightValidator()
        partial_caller_code = (
            "10 | response = service.execute_order(\n"
            "11 |     order_id='ord_123',\n"
            "12 |     amount=100.0,\n"
        )
        caller_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/client.py",
            start_line=10,
            end_line=12,
            qualified_symbol="trading.client.execute",
            selection_reason="Direct caller window",
            modification_intent=ModificationIntent.DIRECT_CALLER,
            git_commit_sha="a1b2c3d4e5f6",
            git_blob_sha="f6e5d4c3b2a1",
            verbatim_code_with_lines=partial_caller_code,
        )
        # As DIRECT_CALLER with trailing comma, partial window truncation logic accepts it
        passed, errors, _ = validator.check_ast_syntax([caller_slice])
        assert passed is True
        assert len(errors) == 0

        # As MODIFY_TARGET (not a caller/type window), unclosed delimiter must fail
        target_slice = caller_slice.model_copy(
            update={"modification_intent": ModificationIntent.MODIFY_TARGET}
        )
        passed_target, errors_target, _ = validator.check_ast_syntax([target_slice])
        assert passed_target is False
        assert any("Unclosed syntax delimiters" in err for err in errors_target)

    def test_principal_scope_mirroring_admin_vs_restricted(
        self, in_memory_repo: InMemoryRLSRepository
    ) -> None:
        """Privileged principal accesses restricted files while unprivileged gets excised."""
        validator = PreFlightValidator()
        bound_repo = validator._resolve_graph_repo(in_memory_repo)

        # Create restricted file requiring 'scope:restricted_admin'
        restricted_file = CodeFileRecord(
            file_id=uuid.uuid4(),
            domain_id=DOMAIN_ID,
            repo_name=REPO_NAME,
            file_path="src/restricted/vault.py",
            git_commit_sha="c0ffee123456",
            git_blob_sha="deadbeef1234",
            language="python",
            allowed_scopes=["scope:restricted_admin"],
            raw_content="SECRET_KEY = 'super_secret'\n",
            line_count=1,
        )
        inserting_principal = PrincipalContext(
            user_id="sec_admin",
            role=RoleType.ENGINEER,
            allowed_scopes=["scope:restricted_admin"],
        )
        in_memory_repo.insert_code_file(inserting_principal, restricted_file)

        slice_item = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/restricted/vault.py",
            start_line=1,
            end_line=1,
            qualified_symbol="vault.SECRET_KEY",
            selection_reason="Restricted slice",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="c0ffee123456",
            git_blob_sha="deadbeef1234",
            verbatim_code_with_lines="1 | SECRET_KEY = 'super_secret'\n",
        )

        # 1. Unprivileged principal lacking scope
        unprivileged = PrincipalContext(
            user_id="guest_dev",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        passed_unp, permitted_unp, _, excised_unp, missing_unp = validator.check_principal_scopes(
            code_slices=[slice_item],
            citations=[],
            bound_repo=bound_repo,
            principal=unprivileged,
            domain_id=DOMAIN_ID,
        )
        assert passed_unp is False
        assert len(permitted_unp) == 0
        assert len(excised_unp) == 1
        assert "src/restricted/vault.py" in excised_unp[0]
        assert missing_unp[0].kind == "scope"

        # 2. Privileged principal with matching scope
        (
            passed_priv,
            permitted_priv,
            _,
            excised_priv,
            missing_priv,
        ) = validator.check_principal_scopes(
            code_slices=[slice_item],
            citations=[],
            bound_repo=bound_repo,
            principal=inserting_principal,
            domain_id=DOMAIN_ID,
        )
        assert passed_priv is True
        assert len(permitted_priv) == 1
        assert len(excised_priv) == 0
        assert len(missing_priv) == 0

    def test_cryptographic_provenance_sha_and_line_bounds(
        self, in_memory_repo: InMemoryRLSRepository, base_principal: PrincipalContext
    ) -> None:
        """Short SHAs, placeholder SHAs, inverted line bounds, and phantom files fail provenance."""
        validator = PreFlightValidator()
        bound_repo = validator._resolve_graph_repo(in_memory_repo)

        # 1. Short commit SHA (< 7 chars) constructed via model_construct or placeholder
        short_sha_slice = VerbatimCodeSlice.model_construct(
            repo_name=REPO_NAME,
            file_path="src/order_book.py",
            start_line=1,
            end_line=3,
            qualified_symbol="trading.order_book.OrderBook",
            selection_reason="Short sha",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="abc",
            git_blob_sha="f6e5d4c3b2a1",
            verbatim_code_with_lines="1 | code\n",
        )
        passed_sha, unanchored_sha, _ = validator.check_cryptographic_provenance(
            [short_sha_slice], bound_repo, base_principal, DOMAIN_ID
        )
        assert passed_sha is False
        assert any("Invalid commit SHA" in err for err in unanchored_sha)

        # 2. Placeholder commit SHA
        placeholder_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/order_book.py",
            start_line=1,
            end_line=3,
            qualified_symbol="trading.order_book.OrderBook",
            selection_reason="Placeholder sha",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="placeholder_sha",
            git_blob_sha="f6e5d4c3b2a1",
            verbatim_code_with_lines="1 | code\n",
        )
        passed_ph, unanchored_ph, _ = validator.check_cryptographic_provenance(
            [placeholder_slice], bound_repo, base_principal, DOMAIN_ID
        )
        assert passed_ph is False
        assert any("Invalid commit SHA" in err for err in unanchored_ph)

        # 3. Inverted line bounds (start_line > end_line)
        inverted_bounds_slice = placeholder_slice.model_copy(
            update={
                "git_commit_sha": "a1b2c3d4e5f6",
                "start_line": 10,
                "end_line": 5,
            }
        )
        passed_bounds, unanchored_bounds, _ = validator.check_cryptographic_provenance(
            [inverted_bounds_slice], bound_repo, base_principal, DOMAIN_ID
        )
        assert passed_bounds is False
        assert any("Invalid line bounds 10-5" in err for err in unanchored_bounds)

        # 4. Phantom file not in repository
        phantom_slice = placeholder_slice.model_copy(
            update={
                "git_commit_sha": "a1b2c3d4e5f6",
                "file_path": "src/phantom_ghost.py",
                "start_line": 1,
                "end_line": 2,
            }
        )
        passed_phantom, unanchored_phantom, _ = validator.check_cryptographic_provenance(
            [phantom_slice], bound_repo, base_principal, DOMAIN_ID
        )
        assert passed_phantom is False
        assert any("Phantom file not found" in err for err in unanchored_phantom)

    def test_document_freshness_negative_jira_and_stale_chunks(
        self, in_memory_repo: InMemoryRLSRepository, base_principal: PrincipalContext
    ) -> None:
        """Negative Jira resolutions and stale chunks trigger freshness invalidation."""
        domain_cfg = DomainConfig(
            domain_id=DOMAIN_ID,
            jira_negative_resolutions=["cancelled", "won't fix", "duplicate"],
        )
        validator = PreFlightValidator(domain_config=domain_cfg)
        bound_repo = validator._resolve_graph_repo(in_memory_repo)

        # Stale chunk
        stale_id = uuid.uuid4()
        in_memory_repo.insert_doc_chunk(
            base_principal,
            DocChunkRecord(
                chunk_id=stale_id,
                domain_id=DOMAIN_ID,
                doc_id="doc_stale",
                doc_title="Stale Architecture",
                source_uri="confluence://spaces/ARCH/pages/100",
                market_scope=None,
                published_date=None,
                authority_weight=1.0,
                section_heading="Overview",
                contextual_header="Stale header",
                verbatim_content="Old outdated design specification",
                metadata_json={},
                allowed_scopes=["public_eng"],
                embedding=[0.0] * 768,
                is_stale=True,
            ),
        )

        # Cancelled ticket chunk
        cancelled_id = uuid.uuid4()
        in_memory_repo.insert_doc_chunk(
            base_principal,
            DocChunkRecord(
                chunk_id=cancelled_id,
                domain_id=DOMAIN_ID,
                doc_id="ticket_cancelled",
                doc_title="SCOUT-999 Cancelled Feature",
                source_uri="jira://browse/SCOUT-999",
                market_scope=None,
                published_date=None,
                authority_weight=1.0,
                section_heading="Resolution",
                contextual_header="Ticket header",
                verbatim_content="This task was cancelled by product management",
                metadata_json={"resolution": "Cancelled"},
                allowed_scopes=["public_eng"],
                embedding=[0.0] * 768,
                is_stale=False,
            ),
        )

        stale_citation = DocCitationSpan(
            chunk_id=str(stale_id),
            doc_id="doc_stale",
            doc_title="Stale Architecture",
            source_uri="confluence://spaces/ARCH/pages/100",
            section_heading="Overview",
            verbatim_quote="Old outdated design specification",
        )
        cancelled_citation = DocCitationSpan(
            chunk_id=str(cancelled_id),
            doc_id="ticket_cancelled",
            doc_title="SCOUT-999 Cancelled Feature",
            source_uri="jira://browse/SCOUT-999",
            section_heading="Resolution",
            verbatim_quote="This task was cancelled by product management",
        )

        passed, stale_items, missing = validator.check_document_freshness(
            citations=[stale_citation, cancelled_citation],
            doc_chunks=None,
            bound_repo=bound_repo,
            principal=base_principal,
            domain_id=DOMAIN_ID,
        )
        assert passed is False
        assert any(f"stale_chunk:{stale_id}" in item for item in stale_items)
        assert any(f"negative_resolution:{cancelled_id}:cancelled" in item for item in stale_items)
        assert len(missing) == 2
        assert all(m.kind == "freshness" for m in missing)


# =============================================================================
# 2. TestClaimVerifierAdversarial
# =============================================================================


class TestClaimVerifierAdversarial:
    """Adversarial stress testing of exact claim verification and homoglyph defense."""

    def test_token_overlap_exact_matching_substitutions_and_negations(self) -> None:
        """Casing changes, punctuation stripping, and negations must fail exact matching."""
        source = (
            "Under NO circumstances should the gateway allow unauthenticated fallback mode. "
            "The authentication timeout is set to 500 ms, strictly enforced."
        )
        verifier = ClaimVerifier()

        # 1. Exact match passes
        rec_exact = verifier.verify_single_claim(
            "The authentication timeout is set to 500 ms, strictly enforced.", [source]
        )
        assert rec_exact.is_verified is True
        assert rec_exact.failure_code == ClaimFailureCode.EXACT_MATCH_PASSED

        # 2. Case sensitivity mismatch fails
        rec_case = verifier.verify_single_claim(
            "The Authentication timeout is set to 500 ms, strictly enforced.", [source]
        )
        assert rec_case.is_verified is False
        assert rec_case.failure_code == ClaimFailureCode.NOT_IN_SOURCE

        # 3. Punctuation stripping fails (comma removed between '500 ms' and 'strictly')
        rec_punct = verifier.verify_single_claim(
            "The authentication timeout is set to 500 ms strictly enforced.", [source]
        )
        assert rec_punct.is_verified is False
        assert rec_punct.failure_code == ClaimFailureCode.NOT_IN_SOURCE

        # 4. Negation / substitution fails
        rec_neg = verifier.verify_single_claim(
            "Under NO circumstances should the gateway disallow unauthenticated fallback mode.",
            [source],
        )
        assert rec_neg.is_verified is False
        assert rec_neg.failure_code == ClaimFailureCode.NOT_IN_SOURCE

    def test_homoglyph_detection_lookalike_rejection_and_unicode_acceptance(self) -> None:
        """Mixed-script Cyrillic/Greek lookalikes rejected; authentic German and CJK accepted."""
        # 1. Mixed-script spoofing: Latin 'password' with Cyrillic small letter a (U+0430)
        spoofed_text = "Verify the p\u0430ssword token."
        is_hg, hg_msg = detect_homoglyph_spoofing(spoofed_text)
        assert is_hg is True
        assert "Mixed-script homoglyph" in hg_msg

        verifier = ClaimVerifier()
        rec_spoof = verifier.verify_single_claim(spoofed_text, ["Verify the password token."])
        assert rec_spoof.is_verified is False
        assert rec_spoof.failure_code == ClaimFailureCode.HOMOGLYPH_SPOOFING
        assert rec_spoof.is_homoglyph_attack is True

        # 2. Authentic non-Latin Unicode (German Umlauts & CJK)
        german_quote = "Überwachung der Frankfurter Börse."
        is_hg_de, _ = detect_homoglyph_spoofing(german_quote)
        assert is_hg_de is False
        rec_de = verifier.verify_single_claim(german_quote, [german_quote])
        assert rec_de.is_verified is True

        cjk_quote = "東京証券取引所のリアルタイムデータ"
        is_hg_cjk, _ = detect_homoglyph_spoofing(cjk_quote)
        assert is_hg_cjk is False
        rec_cjk = verifier.verify_single_claim(cjk_quote, [cjk_quote])
        assert rec_cjk.is_verified is True

    def test_candidate_source_polymorphism(self) -> None:
        """ClaimVerifier handles candidate sources as dict, tuple, string, or DocChunkRecord."""
        claim = "Order flow imbalance exhibits mean reversion."
        verifier = ClaimVerifier()

        # Dict[str, str]
        src_dict = {"chunk_1": f"Prefix {claim} Suffix"}
        assert verifier.verify_single_claim(claim, src_dict).is_verified is True

        # List[tuple[str, str]]
        src_tuples = [("chunk_1", f"Prefix {claim} Suffix")]
        assert verifier.verify_single_claim(claim, src_tuples).is_verified is True

        # List[str]
        src_strings = [f"Prefix {claim} Suffix"]
        assert verifier.verify_single_claim(claim, src_strings).is_verified is True

        # List[DocChunkRecord]
        chunk_rec = DocChunkRecord(
            chunk_id=uuid.uuid4(),
            domain_id=DOMAIN_ID,
            doc_id="doc_poly",
            doc_title="Polymorphic Doc",
            source_uri="uri://poly",
            market_scope=None,
            published_date=None,
            authority_weight=1.0,
            section_heading="Heading",
            contextual_header="Header",
            verbatim_content=f"Prefix {claim} Suffix",
            metadata_json={},
            allowed_scopes=["public_eng"],
            embedding=[0.0] * 768,
        )
        assert verifier.verify_single_claim(claim, [chunk_rec]).is_verified is True

    def test_policy_matrix_strict_abstention_zero_spend_invariant(self) -> None:
        """All verification policies enforce paid_tokens_spent=0 on abstention."""
        sources = ["Alpha generation requires non-linear features."]
        claims = [
            "Alpha generation requires non-linear features.",  # valid
            "Beta hedging is completely optional.",  # hallucinated / not in source
        ]

        # 1. STRICT_ABSTAIN_ON_ANY_UNVERIFIED
        cfg_strict = ClaimVerifierConfig(
            policy=ClaimVerificationPolicy.STRICT_ABSTAIN_ON_ANY_UNVERIFIED
        )
        res_strict = ClaimVerifier(cfg_strict).verify_claims(claims, sources)
        assert res_strict.passed is False
        assert res_strict.paid_tokens_spent == 0
        assert res_strict.diagnostic is not None
        assert res_strict.diagnostic.paid_tokens_spent == 0
        assert res_strict.pruned_count == 1

        # 2. PRUNE_AND_MARK_LOW_TRUST (with min_verified_claims = 3)
        cfg_prune = ClaimVerifierConfig(
            policy=ClaimVerificationPolicy.PRUNE_AND_MARK_LOW_TRUST,
            min_verified_claims=3,
        )
        res_prune = ClaimVerifier(cfg_prune).verify_claims(claims, sources)
        assert res_prune.passed is False
        assert res_prune.paid_tokens_spent == 0
        assert res_prune.diagnostic is not None
        assert res_prune.diagnostic.paid_tokens_spent == 0

        # 3. MIN_VERIFIED_COUNT (with min_verified_claims = 2)
        cfg_min = ClaimVerifierConfig(
            policy=ClaimVerificationPolicy.MIN_VERIFIED_COUNT,
            min_verified_claims=2,
        )
        res_min = ClaimVerifier(cfg_min).verify_claims(claims, sources)
        assert res_min.passed is False
        assert res_min.paid_tokens_spent == 0
        assert res_min.diagnostic is not None
        assert res_min.diagnostic.paid_tokens_spent == 0


# =============================================================================
# 3. TestGeminiExecutorSandboxAdversarial
# =============================================================================


class TestGeminiExecutorSandboxAdversarial:
    """Adversarial stress testing of isolated GitWorktreeSandbox and patch engine."""

    def test_search_replace_block_parsing_with_line_number_stripping(
        self, temp_git_repo: Path
    ) -> None:
        """SEARCH/REPLACE block with line number prefixes is cleanly parsed and applied."""
        sr_patch = (
            "### File: src/module.py\n"
            "<<<<<<< SEARCH\n"
            " 1 | def compute_fee(amount: float) -> float:\n"
            " 2 |     base_rate = 0.001\n"
            "=======\n"
            "def compute_fee(amount: float) -> float:\n"
            "    base_rate = 0.0025\n"
            ">>>>>>> REPLACE\n"
        )
        with GitWorktreeSandbox(repo_root=temp_git_repo, require_confirmation=False) as sandbox:
            assert sandbox.worktree_dir is not None
            applied, msg = sandbox.apply_patch(sr_patch)
            assert applied is True
            assert "SEARCH/REPLACE" in msg

            # Verify file content updated
            target = sandbox.worktree_dir / "src" / "module.py"
            updated_content = target.read_text(encoding="utf-8")
            assert "base_rate = 0.0025" in updated_content

    def test_unified_diff_line_drift_matching(self, temp_git_repo: Path) -> None:
        """Unified diff with drifted line numbers applies via hunk-content matching."""
        # Intentionally wrong @@ line numbers (-50,2 +50,2 instead of 2,2)
        drifted_patch = (
            "--- a/src/module.py\n"
            "+++ b/src/module.py\n"
            "@@ -50,2 +50,2 @@\n"
            "-    base_rate = 0.001\n"
            "-    return amount * base_rate\n"
            "+    base_rate = 0.0050\n"
            "+    return amount * base_rate * 1.1\n"
        )
        with GitWorktreeSandbox(repo_root=temp_git_repo, require_confirmation=False) as sandbox:
            assert sandbox.worktree_dir is not None
            applied, msg = sandbox.apply_patch(drifted_patch)
            assert applied is True
            assert "hunk" in msg.lower() or "applied" in msg.lower()

            target = sandbox.worktree_dir / "src" / "module.py"
            updated_content = target.read_text(encoding="utf-8")
            assert "base_rate = 0.0050" in updated_content
            assert "return amount * base_rate * 1.1" in updated_content

    def test_environment_sanitization_strips_sensitive_keys(self, temp_git_repo: Path) -> None:
        """Verification commands use strict allowlist so secrets/tokens never leak (ECS-105)."""
        injected_env = {
            "SCOUT_SUPER_SECRET_API_KEY": "compromised_token_12345",
            "SECRET_ADMIN_PASSWORD": "hunter2",
            "SCOUT_DB_APP_URL": "postgresql://user:pass@localhost/db",
            "SCOUT_REDIS_URL": "rediss://:redis-pass@localhost:6379/0",
            "GITHUB_TOKEN": "ghp_dummy_secret_token",
            "CI_JOB_TOKEN": "glcbt-dummy-job-token",
            "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/fake-sa.json",
        }
        os.environ.update(injected_env)

        try:
            with GitWorktreeSandbox(repo_root=temp_git_repo, require_confirmation=False) as sandbox:
                test_cmd = [
                    "python3",
                    "-c",
                    (
                        "import os, sys\n"
                        "forbidden_exact = [\n"
                        "    'SCOUT_SUPER_SECRET_API_KEY',\n"
                        "    'SECRET_ADMIN_PASSWORD',\n"
                        "    'SCOUT_DB_APP_URL',\n"
                        "    'SCOUT_REDIS_URL',\n"
                        "    'GITHUB_TOKEN',\n"
                        "    'CI_JOB_TOKEN',\n"
                        "    'GOOGLE_APPLICATION_CREDENTIALS',\n"
                        "]\n"
                        "leaked = [k for k in forbidden_exact if k in os.environ]\n"
                        "if leaked:\n"
                        "    sys.stderr.write(f'Leaked env keys: {leaked}')\n"
                        "    sys.exit(1)\n"
                        "sys.exit(0)\n"
                    ),
                ]
                lint_cmd = ["python3", "-c", "import sys; sys.exit(0)"]
                lint_ok, test_ok, details = sandbox.run_verification(
                    lint_cmd=lint_cmd, test_cmd=test_cmd
                )
                assert lint_ok is True
                assert test_ok is True, f"Sanitization failed: {details}"

                # unittest & shell wrapper bypasses must also be blocked fail-closed (ECS-017)
                for blocked_cmd in (
                    ["python3", "-m", "unittest"],
                    ["python3", "-munittest", "discover"],
                    ["python3", "-c", "import unittest; unittest.main()"],
                    ["sh", "-c", "pytest tests/"],
                    ["/bin/bash", "-lc", "python -m unittest"],
                ):
                    with pytest.raises(PermissionError):
                        sandbox.run_verification(lint_cmd=lint_cmd, test_cmd=blocked_cmd)
        finally:
            for key in injected_env:
                os.environ.pop(key, None)

    def test_confirmation_gate_blocks_unconfirmed_commit(self, temp_git_repo: Path) -> None:
        """confirm_and_merge(confirmed=False) must block commit and leave branch uncommitted."""
        patch = (
            "--- a/src/module.py\n"
            "+++ b/src/module.py\n"
            "@@ -2,2 +2,2 @@\n"
            "-    base_rate = 0.001\n"
            "+    base_rate = 0.0099\n"
        )
        with GitWorktreeSandbox(repo_root=temp_git_repo, require_confirmation=True) as sandbox:
            sandbox.apply_patch(patch)

            # Block merge without human confirmation
            ok, msg = sandbox.confirm_and_merge(confirmed=False)
            assert ok is False
            assert "Human confirmation required" in msg

            # Approve merge with confirmation
            ok_conf, msg_conf = sandbox.confirm_and_merge(confirmed=True)
            assert ok_conf is True
            assert "committed" in msg_conf.lower()

    def test_inactive_sandbox_operations_raise_runtime_error(self, temp_git_repo: Path) -> None:
        """Calling sandbox operations outside the context manager must raise RuntimeError."""
        sandbox = GitWorktreeSandbox(repo_root=temp_git_repo)
        with pytest.raises(RuntimeError, match="Sandbox is not active"):
            sandbox.apply_patch("diff --git a/file b/file")

        with pytest.raises(RuntimeError, match="Sandbox is not active"):
            sandbox.run_verification()

        with pytest.raises(RuntimeError, match="Sandbox is not active"):
            sandbox.confirm_and_merge(confirmed=True)


# =============================================================================
# 4. TestCostGovernanceAdversarial
# =============================================================================


class TestCostGovernanceAdversarial:
    """Adversarial stress testing of live pricing, monotonicity, kill-switches, and concurrency."""

    def test_boundary_token_conditions_monotonicity(self) -> None:
        """Boundary token counts (1-2 tokens) never invert low <= expected <= high."""
        estimator = PaidComputeCostEstimator()

        models_to_test = ["gemini-3.8-flash", "gemini-4-argon", "claude-opus-5.5", "gpt-5.6-sol"]
        token_cases = [(0, 1), (1, 1), (2, 1), (1, 100), (10, 1)]

        for model in models_to_test:
            for in_tokens, max_out in token_cases:
                rep = estimator.calculate_cost_estimate(
                    model_id=model,
                    estimated_input_tokens=in_tokens,
                    max_output_tokens=max_out,
                )
                assert rep.cost_low_usd >= 0.0
                assert rep.cost_expected_usd >= rep.cost_low_usd, (
                    f"Inversion for {model} at in={in_tokens}, out={max_out}: "
                    f"low={rep.cost_low_usd} > exp={rep.cost_expected_usd}"
                )
                assert rep.cost_high_usd >= rep.cost_expected_usd, (
                    f"Inversion for {model} at in={in_tokens}, out={max_out}: "
                    f"exp={rep.cost_expected_usd} > high={rep.cost_high_usd}"
                )

    def test_october_2026_pricing_table_exact_rates(self) -> None:
        """Live October 2026 model pricing tables adhere strictly to audited standards."""
        # 1. gemini-4-argon
        argon = OCTOBER_2026_PRICING["gemini-4-argon"]
        assert argon.input_usd_per_1m_low == 2.00
        assert argon.input_usd_per_1m_expected == 3.00
        assert argon.input_usd_per_1m_high == 4.00
        assert argon.output_usd_per_1m_low == 10.00
        assert argon.output_usd_per_1m_expected == 15.00
        assert argon.output_usd_per_1m_high == 20.00
        assert argon.is_zero_cost is False

        # 2. gemini-3.8-flash
        flash = OCTOBER_2026_PRICING["gemini-3.8-flash"]
        assert flash.input_usd_per_1m_low == 0.75
        assert flash.input_usd_per_1m_expected == 1.00
        assert flash.input_usd_per_1m_high == 1.50
        assert flash.output_usd_per_1m_low == 3.75
        assert flash.output_usd_per_1m_expected == 5.00
        assert flash.output_usd_per_1m_high == 7.50
        assert flash.is_zero_cost is False

        # 3. claude-opus-5.5
        opus = OCTOBER_2026_PRICING["claude-opus-5.5"]
        assert opus.input_usd_per_1m_low == 4.00
        assert opus.input_usd_per_1m_expected == 4.00
        assert opus.input_usd_per_1m_high == 4.00
        assert opus.output_usd_per_1m_low == 20.00
        assert opus.output_usd_per_1m_expected == 20.00
        assert opus.output_usd_per_1m_high == 20.00

        # 4. gpt-5.6-sol
        sol = OCTOBER_2026_PRICING["gpt-5.6-sol"]
        assert sol.input_usd_per_1m_low == 4.00
        assert sol.input_usd_per_1m_expected == 4.00
        assert sol.input_usd_per_1m_high == 4.00
        assert sol.output_usd_per_1m_low == 20.00
        assert sol.output_usd_per_1m_expected == 20.00
        assert sol.output_usd_per_1m_high == 20.00

        # 5. Local Gemma 4 MoE ($0 cost)
        gemma = OCTOBER_2026_PRICING["gemma-4-26b-a4b-it"]
        assert gemma.is_zero_cost is True
        assert gemma.input_usd_per_1m_expected == 0.0
        assert gemma.output_usd_per_1m_expected == 0.0

    def test_session_budget_tracker_kill_switch_boundaries(self) -> None:
        """250k token exploration cap and $25 spend ceiling trigger KillSwitchTriggeredException."""
        tracker = SessionBudgetTracker(spend_ceiling_usd=DEFAULT_SESSION_SPEND_CEILING_USD)
        session_id = f"sess_{uuid.uuid4().hex[:6]}"

        # 1. Exactly 250,000 exploration tokens succeeds
        tracker.record_exploration_tokens(
            DEFAULT_EXPLORATION_TOKEN_KILL_SWITCH, session_id=session_id
        )

        # 250,001 tokens breaches ceiling
        with pytest.raises(
            KillSwitchTriggeredException, match=r"Automated exploration kill-switch triggered"
        ):
            tracker.record_exploration_tokens(1, session_id=session_id)

        # 2. Spend ceiling check
        principal = PrincipalContext(
            user_id="spend_tester",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
            max_session_paid_tokens=500_000,
        )
        # $24.90 spend reservation succeeds
        tracker.check_and_reserve(principal, projected_tokens=10_000, estimated_cost_usd=24.90)
        tracker.record_consumption(principal, input_tokens=10_000, output_tokens=0, cost_usd=24.90)

        # Additional $0.20 brings projected spend to $25.10 > $25.00 ceiling
        with pytest.raises(KillSwitchTriggeredException, match=r"Automated kill-switch triggered"):
            tracker.check_and_reserve(principal, projected_tokens=1_000, estimated_cost_usd=0.20)

    def test_thread_safe_token_reservation_concurrency(self) -> None:
        """Concurrent requests competing for budget serialize safely without race condition."""
        tracker = SessionBudgetTracker()
        principal = PrincipalContext(
            user_id="concurrency_tester",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
            max_session_paid_tokens=10_000,
        )

        successes = 0
        failures = 0

        def attempt_reservation() -> bool:
            try:
                tracker.check_and_reserve(principal, projected_tokens=1_000)
                tracker.record_consumption(principal, input_tokens=1_000)
                return True
            except BudgetExceededException:
                return False

        # Launch 25 concurrent threads attempting to reserve 1,000 tokens each against 10,000 budget
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(attempt_reservation) for _ in range(25)]
            for fut in concurrent.futures.as_completed(futures):
                if fut.result():
                    successes += 1
                else:
                    failures += 1

        # Exactly 10 reservations must succeed and 15 must fail
        assert successes == 10
        assert failures == 15
        assert (
            tracker.get_consumption(principal.user_id, principal.tenant_id)["total_tokens"]
            == 10_000
        )


# =============================================================================
# 5. TestOpenTelemetryAdversarial
# =============================================================================


class TestOpenTelemetryAdversarial:
    """Adversarial stress testing of OpenTelemetry 4-phase spans, abstention, and W3C."""

    def test_4_phase_spans_lifecycle_and_attributes(self) -> None:
        """4 pipeline phases (retrieval, slicing, gating, execution) correctly populate spans."""
        tracer = ScoutTelemetryTracer.reset_instance()
        tracer.exporter.clear()

        # Phase 1: Retrieval
        with tracer.trace_retrieval(domain_id=DOMAIN_ID, query="OFI calculation") as r_ctx:
            r_ctx.record_retrieval(doc_chunks_count=3, ast_symbols_count=7, query="OFI calculation")

        # Phase 2: Slicing
        with tracer.trace_slicing(domain_id=DOMAIN_ID, file_path="src/order_book.py") as s_ctx:
            s_ctx.record_slicing(file_path="src/order_book.py", lines_sliced=25, syntax_valid=True)

        # Phase 3: Gating
        with tracer.trace_gating(domain_id=DOMAIN_ID) as g_ctx:
            mock_report = PreFlightValidationReport(
                passed=True,
                normalized_rrf_score=0.85,
                citations_verified_100_pct=True,
                auto_attached_ast_slices_count=2,
            )
            g_ctx.record_report(mock_report)

        # Phase 4: Execution
        with tracer.trace_execution(domain_id=DOMAIN_ID) as e_ctx:
            mock_result = MagicMock(
                token_count=150, verification_passed=True, patch_applied=True, success=True
            )
            mock_telemetry = MagicMock(
                paid_tokens_saved=3_500,
                estimated_usd_saved=0.035,
                zero_touch_sufficient=True,
                context_pack_tokens_sent=300,
                preflight_passed_first_try=True,
            )
            e_ctx.record_result(mock_result, mock_telemetry)

        finished_spans = tracer.exporter.get_finished_spans()
        assert len(finished_spans) == 4

        span_phases = [
            s.attributes.get(ATTR_SCOUT_PHASE) if s.attributes is not None else None
            for s in finished_spans
        ]
        assert span_phases == [
            PHASE_RETRIEVAL,
            PHASE_SLICING,
            PHASE_GATING,
            PHASE_EXECUTION,
        ]

    def test_abstention_invariant_zero_spend_attributes(self) -> None:
        """Span with InsufficientContextDiagnostic strictly enforces zero-spend attributes."""
        tracer = ScoutTelemetryTracer.reset_instance()
        tracer.exporter.clear()

        diagnostic = InsufficientContextDiagnostic(
            trace_id="trace_zero_spend_test",
            domain_id=DOMAIN_ID,
            reason="Normalized RRF score below threshold (0.42 < 0.65)",
            missing_symbols=["OrderBook.get_depth"],
            partial_candidate_files=[],
            paid_tokens_spent=0,
        )

        with tracer.trace_workflow(trace_id="trace_zero_spend_test", domain_id=DOMAIN_ID) as wf_ctx:
            wf_ctx.record_abstention(diagnostic)

        finished_spans = tracer.exporter.get_finished_spans()
        assert len(finished_spans) == 1
        span = finished_spans[0]
        assert span.attributes is not None

        # Invariant checks: zero tokens and zero dollars
        assert span.attributes[ATTR_SCOUT_IS_ABSTENTION] is True
        assert span.attributes[ATTR_SCOUT_PAID_TOKENS_SPENT] == 0
        assert span.attributes[ATTR_SCOUT_COST_USD] == 0.0
        assert span.attributes[GEN_AI_USAGE_INPUT_TOKENS] == 0
        assert span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS] == 0
        assert "Normalized RRF" in str(span.attributes.get("scout.rejection_reason"))

    def test_w3c_traceparent_injection_extraction_and_corrupted_fallback(self) -> None:
        """W3C traceparent propagates parentage and gracefully falls back on corrupted header."""
        tracer = ScoutTelemetryTracer.reset_instance()

        # 1. Valid injection and extraction
        carrier: dict[str, str] = {}
        with tracer.tracer.start_as_current_span("root_span") as root_span:
            inject_trace_context(carrier)
            assert "traceparent" in carrier
            traceparent = carrier["traceparent"]
            assert traceparent.startswith("00-")
            assert len(traceparent.split("-")) == 4

        extracted_context = extract_trace_context(carrier)
        with tracer.tracer.start_as_current_span(
            "child_span", context=extracted_context
        ) as child_span:
            # Child span must link to the root span context
            assert isinstance(child_span, ReadableSpan)
            assert child_span.parent is not None
            assert child_span.parent.span_id == root_span.get_span_context().span_id

        # 2. Corrupted traceparent fallback
        corrupted_carrier = {"traceparent": "corrupted;invalid;traceparent;token"}
        # Must not raise an exception; returns fallback context
        fallback_context = extract_trace_context(corrupted_carrier)
        with tracer.tracer.start_as_current_span(
            "fallback_span", context=fallback_context
        ) as fb_span:
            # Span executes cleanly as a root span without parent
            assert fb_span.is_recording()
