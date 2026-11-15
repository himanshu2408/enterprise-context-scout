"""Challenger 1 Adversarial Suite — Milestone 4 Iteration 3 Gate Verification.

Empirical verification and stress testing of:
1. Adversarial Delimiter Stressing on Partial Windows (check_ast_syntax):
   - Partial slices (DIRECT_CALLER, IMPORTED_TYPE) with unclosed triple quotes
   - String literal parentheses vs real unclosed code parentheses
   - Parentheses/closure nesting depth beyond 10 levels
   - Mathematical intervals ([0, T), (0, 1]) under slicer delimiter logic
   - Multiline comments and docstrings with unmatched braces in C#, TypeScript, and Python
2. Cross-Domain Tenant Isolation Stressing (check_principal_scopes):
   - Candidate domain fallback behavior against tenant privilege escalation
   - Shared generic scopes (e.g. 'public_eng') across tenant boundaries
   - Hardcoded 'quant_research_scout' and 'default' domain fallback leakage
   - Parity comparison between code slice scoping and doc chunk citation scoping
"""

from __future__ import annotations

import datetime
from typing import Final
from uuid import uuid4

import pytest

from context_scout.agents.preflight_gate import (
    BoundGraphRepository,
    PreFlightValidator,
)
from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer
from context_scout.db.repositories import (
    CodeFileRecord,
    DocChunkRecord,
    InMemoryRLSRepository,
)
from context_scout.schemas.contracts import (
    DocCitationSpan,
    MarketRegime,
    ModificationIntent,
    PrincipalContext,
    RoleType,
    VerbatimCodeSlice,
)

DOMAIN_TENANT_A: Final[str] = "tenant_alpha"
DOMAIN_TENANT_B: Final[str] = "tenant_beta"
DOMAIN_QUANT: Final[str] = "quant_research_scout"
REPO_NAME: Final[str] = "tenant-repo-alpha"


@pytest.fixture
def slicer() -> DeterministicSlicer:
    return DeterministicSlicer()


@pytest.fixture
def in_memory_repo() -> InMemoryRLSRepository:
    return InMemoryRLSRepository()


@pytest.fixture
def bound_repo(in_memory_repo: InMemoryRLSRepository) -> BoundGraphRepository:
    return BoundGraphRepository(repo=in_memory_repo)


# =============================================================================
# SUITE 1: Adversarial Delimiter Stressing on Partial Windows (check_ast_syntax)
# =============================================================================


class TestAdversarialDelimiterPartialWindows:
    """Stress check_ast_syntax with partial slices (DIRECT_CALLER, IMPORTED_TYPE)."""

    @pytest.mark.parametrize(
        "intent",
        [ModificationIntent.DIRECT_CALLER, ModificationIntent.IMPORTED_TYPE],
    )
    def test_unclosed_triple_quotes_in_partial_windows(
        self,
        in_memory_repo: InMemoryRLSRepository,
        slicer: DeterministicSlicer,
        intent: ModificationIntent,
    ) -> None:
        """Probe behavior when partial slices contain unclosed triple quotes.

        Hypothesis: Because whole-module tree-sitter parsing is bypassed on partial windows,
        and _has_unclosed_delimiters only checks '(', '[', '{', unclosed triple quotes
        (''' or \"\"\") pass silently without syntax error detection.
        """
        principal = PrincipalContext(
            user_id="test_user",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        validator = PreFlightValidator(
            repo=in_memory_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_TENANT_A,
        )

        unclosed_triple_double = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service/caller.py",
            start_line=1,
            end_line=3,
            qualified_symbol=f"service.caller.{intent.value}_double",
            selection_reason="Partial slice with unclosed triple double quote",
            modification_intent=intent,
            git_commit_sha="sha_test_1",
            git_blob_sha="blob_test_1",
            verbatim_code_with_lines=(
                "1 | def handle_request():\n"
                '2 |     """Docstring starts here but is never closed\n'
                "3 |     return process_data()"
            ),
            auto_expanded_by_ast=False,
        )

        unclosed_triple_single = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service/caller.py",
            start_line=1,
            end_line=3,
            qualified_symbol=f"service.caller.{intent.value}_single",
            selection_reason="Partial slice with unclosed triple single quote",
            modification_intent=intent,
            git_commit_sha="sha_test_2",
            git_blob_sha="blob_test_2",
            verbatim_code_with_lines=(
                "1 | def parse_config():\n"
                "2 |     raw_text = '''unclosed single triple quote\n"
                "3 |     return raw_text.strip()"
            ),
            auto_expanded_by_ast=False,
        )

        passed_double, errors_double, _ = validator.check_ast_syntax([unclosed_triple_double])
        passed_single, errors_single, _ = validator.check_ast_syntax([unclosed_triple_single])

        # Verification: Unclosed triple quotes in partial windows are caught by check_ast_syntax
        assert passed_double is False
        assert len(errors_double) == 1
        assert "Unclosed syntax delimiters in code slice" in errors_double[0]
        assert passed_single is False
        assert len(errors_single) == 1
        assert "Unclosed syntax delimiters in code slice" in errors_single[0]

    @pytest.mark.parametrize(
        "intent",
        [ModificationIntent.DIRECT_CALLER, ModificationIntent.IMPORTED_TYPE],
    )
    def test_unclosed_parens_in_string_literals_vs_real_code_parens(
        self,
        in_memory_repo: InMemoryRLSRepository,
        slicer: DeterministicSlicer,
        intent: ModificationIntent,
    ) -> None:
        """Probe behavior when string literals contain unbalanced parens vs real unclosed parens."""
        principal = PrincipalContext(
            user_id="test_user",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        validator = PreFlightValidator(
            repo=in_memory_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_TENANT_A,
        )

        # 1. Real unclosed code paren (Must fail)
        real_unclosed_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service/calc.py",
            start_line=1,
            end_line=2,
            qualified_symbol=f"service.calc.{intent.value}_real",
            selection_reason="Real unclosed code paren",
            modification_intent=intent,
            git_commit_sha="sha_test_3",
            git_blob_sha="blob_test_3",
            verbatim_code_with_lines="1 | def compute():\n2 |     return (1 + 2",
            auto_expanded_by_ast=False,
        )
        passed_real, errors_real, _ = validator.check_ast_syntax([real_unclosed_slice])
        assert passed_real is False
        assert len(errors_real) == 1
        assert "Unclosed syntax delimiters in code slice" in errors_real[0]

        # 2. String literal with unclosed paren in completely valid code (Must pass)
        string_literal_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service/regex_util.py",
            start_line=1,
            end_line=3,
            qualified_symbol=f"service.regex.{intent.value}_str",
            selection_reason="Valid code with paren inside string literal",
            modification_intent=intent,
            git_commit_sha="sha_test_4",
            git_blob_sha="blob_test_4",
            verbatim_code_with_lines=(
                "1 | def get_status_msg(code: int) -> str:\n"
                '2 |     msg = "Status: pending (waiting for confirmation"\n'
                "3 |     return msg"
            ),
            auto_expanded_by_ast=False,
        )
        passed_str, errors_str, _ = validator.check_ast_syntax([string_literal_slice])
        assert passed_str is True
        assert len(errors_str) == 0

    @pytest.mark.parametrize(
        "intent",
        [ModificationIntent.DIRECT_CALLER, ModificationIntent.IMPORTED_TYPE],
    )
    def test_nesting_depth_beyond_10_levels_in_partial_windows(
        self,
        in_memory_repo: InMemoryRLSRepository,
        slicer: DeterministicSlicer,
        intent: ModificationIntent,
    ) -> None:
        """Probe behavior when partial slices exceed 10 levels of nesting.

        1. Real nested code closures > 10 levels must be rejected.
        2. String literals containing > 10 delimiters must also be probed.
        """
        principal = PrincipalContext(
            user_id="test_user",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        validator = PreFlightValidator(
            repo=in_memory_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_TENANT_A,
        )

        # 1. Real nested parentheses (12 levels)
        deep_real_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service/deep.py",
            start_line=1,
            end_line=2,
            qualified_symbol=f"service.deep.{intent.value}_code",
            selection_reason="12 levels of code parentheses",
            modification_intent=intent,
            git_commit_sha="sha_test_5",
            git_blob_sha="blob_test_5",
            verbatim_code_with_lines=(
                "1 | def deep_nesting():\n2 |     return ((((((((((((100))))))))))))"
            ),
            auto_expanded_by_ast=False,
        )
        passed_deep, errors_deep, _ = validator.check_ast_syntax([deep_real_slice])
        assert passed_deep is False
        assert "Nesting depth (12) exceeds maximum limit (10)" in errors_deep[0]

        # 2. String literal containing 12 opening and closing parens in valid code
        deep_string_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service/format.py",
            start_line=1,
            end_line=2,
            qualified_symbol=f"service.format.{intent.value}_str",
            selection_reason="String literal with 12 parentheses",
            modification_intent=intent,
            git_commit_sha="sha_test_6",
            git_blob_sha="blob_test_6",
            verbatim_code_with_lines=(
                '1 | def get_banner():\n2 |     return "((((((((((((hello))))))))))))"'
            ),
            auto_expanded_by_ast=False,
        )
        passed_str_deep, errors_str_deep, _ = validator.check_ast_syntax([deep_string_slice])
        # Verification: Sanitized string contents do not trigger nesting depth limit
        assert passed_str_deep is True
        assert len(errors_str_deep) == 0

    @pytest.mark.parametrize(
        "interval_expr",
        [
            "[0, T)",
            "(0, 1]",
            "[-1.0, 1.0)",
            "[0, tau)",
            "in [0, T)",
            "t ∈ [0, T)",
        ],
    )
    def test_mathematical_intervals_retained_in_partial_windows(
        self,
        in_memory_repo: InMemoryRLSRepository,
        slicer: DeterministicSlicer,
        interval_expr: str,
    ) -> None:
        """Verify that half-open mathematical intervals remain accepted by slicer delimiter logic.

        The slicer uses a regex to strip intervals before checking delimiter counts.
        """
        principal = PrincipalContext(
            user_id="test_user",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        validator = PreFlightValidator(
            repo=in_memory_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_TENANT_A,
        )

        slice_code = (
            f"1 | # Signal defined over interval {interval_expr}\n"
            f"2 | def evaluate_interval():\n"
            f"3 |     return 42"
        )
        interval_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/math/signal.py",
            start_line=1,
            end_line=3,
            qualified_symbol="math.signal.evaluate",
            selection_reason="Mathematical interval slice",
            modification_intent=ModificationIntent.DIRECT_CALLER,
            git_commit_sha="sha_test_7",
            git_blob_sha="blob_test_7",
            verbatim_code_with_lines=slice_code,
            auto_expanded_by_ast=False,
        )

        passed, errors, missing = validator.check_ast_syntax([interval_slice])
        assert passed is True, f"Failed on interval '{interval_expr}': {errors}"
        assert len(errors) == 0
        assert len(missing) == 0

    def test_multiline_comments_with_unmatched_braces_in_csharp_ts_python(
        self,
        in_memory_repo: InMemoryRLSRepository,
        slicer: DeterministicSlicer,
    ) -> None:
        """Probe behavior of multiline comments / docstrings containing unmatched braces.

        In C#, TypeScript, and Python, unmatched braces inside comments or docstrings
        are completely valid syntax. But naive char counting treats them as code delimiters.
        """
        principal = PrincipalContext(
            user_id="test_user",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        validator = PreFlightValidator(
            repo=in_memory_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_TENANT_A,
        )

        # 1. C# slice with multiline comment containing unmatched brace '{'
        csharp_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/Services/PaymentService.cs",
            start_line=1,
            end_line=6,
            qualified_symbol="Services.PaymentService",
            selection_reason="C# partial caller with comment brace",
            modification_intent=ModificationIntent.DIRECT_CALLER,
            git_commit_sha="sha_cs_1",
            git_blob_sha="blob_cs_1",
            verbatim_code_with_lines=(
                "1 | /*\n"
                "2 |  * JSON schema example:\n"
                "3 |  * { unclosed opening brace in comment\n"
                "4 |  */\n"
                "5 | public class PaymentService {\n"
                "6 | }"
            ),
            auto_expanded_by_ast=False,
        )

        # 2. TypeScript slice with multiline comment containing unmatched brace '{'
        ts_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/components/Theme.ts",
            start_line=1,
            end_line=5,
            qualified_symbol="components.ThemeConfig",
            selection_reason="TS partial type with comment brace",
            modification_intent=ModificationIntent.IMPORTED_TYPE,
            git_commit_sha="sha_ts_1",
            git_blob_sha="blob_ts_1",
            verbatim_code_with_lines=(
                "1 | /**\n"
                "2 |  * Unmatched brace { in JSDoc\n"
                "3 |  */\n"
                "4 | export interface ThemeConfig {\n"
                "5 |   color: string;\n"
                "6 | }"
            ),
            auto_expanded_by_ast=False,
        )

        # 3. Python slice with docstring containing unmatched brace '{'
        python_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/utils/formatter.py",
            start_line=1,
            end_line=5,
            qualified_symbol="utils.formatter.format_msg",
            selection_reason="Python partial caller with docstring brace",
            modification_intent=ModificationIntent.DIRECT_CALLER,
            git_commit_sha="sha_py_1",
            git_blob_sha="blob_py_1",
            verbatim_code_with_lines=(
                "1 | def format_msg():\n"
                '2 |     """Format with curly brace { for placeholder."""\n'
                '3 |     return "formatted"'
            ),
            auto_expanded_by_ast=False,
        )

        passed_cs, errors_cs, _ = validator.check_ast_syntax([csharp_slice])
        passed_ts, errors_ts, _ = validator.check_ast_syntax([ts_slice])
        passed_py, errors_py, _ = validator.check_ast_syntax([python_slice])

        # Verification: All three valid code slices pass cleanly because comments
        # and string literals are sanitized prior to delimiter balancing
        assert passed_cs is True, f"C# comment with unmatched brace failed: {errors_cs}"
        assert len(errors_cs) == 0

        assert passed_ts is True, f"TypeScript comment with unmatched brace failed: {errors_ts}"
        assert len(errors_ts) == 0

        assert passed_py is True, f"Python docstring with unmatched brace failed: {errors_py}"
        assert len(errors_py) == 0


# =============================================================================
# SUITE 2: Cross-Domain Tenant Isolation Stressing (check_principal_scopes)
# =============================================================================


class TestCrossDomainTenantIsolationStressing:
    """Stress check_principal_scopes candidate domain fallback against tenant escalation."""

    def test_unauthorized_scope_citation_is_strictly_blocked(
        self,
        in_memory_repo: InMemoryRLSRepository,
        slicer: DeterministicSlicer,
        bound_repo: BoundGraphRepository,
    ) -> None:
        """Verify that when a principal lacks required scopes, cross-domain fallback
        does NOT grant access to confidential documents.
        """
        # Seed confidential chunk in quant_research_scout with secret scope
        confidential_chunk_id = uuid4()
        confidential_chunk = DocChunkRecord(
            chunk_id=confidential_chunk_id,
            domain_id=DOMAIN_QUANT,
            doc_id="secret_alpha_doc",
            doc_title="Alpha Trading Secrets",
            source_uri="urn:quant:secret",
            market_scope=MarketRegime.GLOBAL_MULTI,
            published_date=datetime.date(2026, 1, 1),
            authority_weight=1.0,
            section_heading="Alpha Signals",
            contextual_header="Alpha Trading Secrets",
            verbatim_content="Confidential proprietary quantitative alpha weights.",
            metadata_json={},
            allowed_scopes=["quant_confidential"],
            embedding=[0.01] * 768,
        )
        in_memory_repo._docs[confidential_chunk_id] = confidential_chunk

        # Attacker principal in tenant_alpha with only tenant_alpha scopes
        attacker = PrincipalContext(
            user_id="attacker_user",
            role=RoleType.ENGINEER,
            allowed_scopes=["tenant_alpha_scope"],
        )
        validator = PreFlightValidator(
            repo=in_memory_repo,
            slicer=slicer,
            principal=attacker,
            domain_id=DOMAIN_TENANT_A,
        )

        citation = DocCitationSpan(
            chunk_id=str(confidential_chunk_id),
            doc_id="secret_alpha_doc",
            doc_title="Alpha Trading Secrets",
            source_uri="urn:quant:secret",
            section_heading="Alpha Signals",
            verbatim_quote="Confidential proprietary quantitative alpha weights.",
            page_id="secret_alpha_doc",
            version="1",
            ticket_key="",
        )

        passed, _, permitted_citations, excised, missing = validator.check_principal_scopes(
            code_slices=[],
            citations=[citation],
            bound_repo=bound_repo,
            principal=attacker,
            domain_id=DOMAIN_TENANT_A,
        )

        # Verification: Access is blocked because scopes do not intersect
        assert len(permitted_citations) == 0
        assert len(excised) == 1
        assert f"doc:secret_alpha_doc:{confidential_chunk_id}" in excised[0]
        assert any(m.kind == "scope" for m in missing)
        assert passed is False

    def test_shared_generic_scope_cross_domain_privilege_escalation(
        self,
        in_memory_repo: InMemoryRLSRepository,
        slicer: DeterministicSlicer,
        bound_repo: BoundGraphRepository,
    ) -> None:
        """CRITICAL VULNERABILITY TEST: Probe cross-domain fallback with shared scope names.

        Scenario:
        1. Tenant A has domain 'tenant_alpha' and users with scope 'public_eng'.
        2. Domain 'quant_research_scout' has proprietary documents also tagged with 'public_eng'.
        3. A user in 'tenant_alpha' should ONLY be able to access Tenant A's documents,
           NOT documents in 'quant_research_scout'.
        4. However, commit b154752 unconditionally appends 'quant_research_scout' and 'default'
           to candidate_domains in check_principal_scopes!
        """
        quant_chunk_id = uuid4()
        quant_chunk = DocChunkRecord(
            chunk_id=quant_chunk_id,
            domain_id=DOMAIN_QUANT,
            doc_id="quant_model_spec",
            doc_title="Internal Quant Model Specification",
            source_uri="urn:quant:model",
            market_scope=MarketRegime.GLOBAL_MULTI,
            published_date=datetime.date(2026, 1, 1),
            authority_weight=1.0,
            section_heading="Model Architecture",
            contextual_header="Internal Quant Model Specification",
            verbatim_content="Internal proprietary volatility surface pricing formulas.",
            metadata_json={},
            allowed_scopes=["public_eng"],  # Generic scope name
            embedding=[0.01] * 768,
        )
        in_memory_repo._docs[quant_chunk_id] = quant_chunk

        # Tenant Alpha user: has 'public_eng' within Tenant Alpha
        tenant_alpha_user = PrincipalContext(
            user_id="alice_tenant_a",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        validator = PreFlightValidator(
            repo=in_memory_repo,
            slicer=slicer,
            principal=tenant_alpha_user,
            domain_id=DOMAIN_TENANT_A,
        )

        citation = DocCitationSpan(
            chunk_id=str(quant_chunk_id),
            doc_id="quant_model_spec",
            doc_title="Internal Quant Model Specification",
            source_uri="urn:quant:model",
            section_heading="Model Architecture",
            verbatim_quote="Internal proprietary volatility surface pricing formulas.",
            page_id="quant_model_spec",
            version="1",
            ticket_key="",
        )

        passed, _, permitted_citations, excised, missing = validator.check_principal_scopes(
            code_slices=[],
            citations=[citation],
            bound_repo=bound_repo,
            principal=tenant_alpha_user,
            domain_id=DOMAIN_TENANT_A,
        )

        # SECURITY INVARIANT (Rule E16 & Cross-Tenant Isolation):
        # Tenant Alpha user MUST NOT access Quant Research Scout via candidate domain fallback.
        # Zero tenant data leaks permitted when principal lacks explicit
        # 'domain:quant_research_scout'.
        is_leaked = len(permitted_citations) == 1
        assert is_leaked is False, (
            "Tenant isolation violation: Tenant Alpha principal was granted access "
            "to document chunk from quant_research_scout without authorized domain scope!"
        )
        assert len(permitted_citations) == 0
        assert len(excised) == 1
        assert f"doc:quant_model_spec:{quant_chunk_id}" in excised[0]
        assert any(m.kind == "scope" for m in missing)
        assert passed is False

    def test_scope_injection_via_domain_prefix(
        self,
        in_memory_repo: InMemoryRLSRepository,
        slicer: DeterministicSlicer,
        bound_repo: BoundGraphRepository,
    ) -> None:
        """Probe whether injecting 'domain:<other_tenant>' in allowed_scopes allows
        cross-tenant chunk retrieval if scopes overlap.
        """
        # Tenant Beta has a private chunk with scope 'beta_shared'
        beta_chunk_id = uuid4()
        beta_chunk = DocChunkRecord(
            chunk_id=beta_chunk_id,
            domain_id=DOMAIN_TENANT_B,
            doc_id="beta_internal_memo",
            doc_title="Beta Confidential Strategy",
            source_uri="urn:beta:memo",
            market_scope=MarketRegime.GLOBAL_MULTI,
            published_date=datetime.date(2026, 1, 1),
            authority_weight=1.0,
            section_heading="Strategy",
            contextual_header="Beta Confidential Strategy",
            verbatim_content="Confidential operational procedures for tenant beta.",
            metadata_json={},
            allowed_scopes=["beta_shared"],
            embedding=[0.01] * 768,
        )
        in_memory_repo._docs[beta_chunk_id] = beta_chunk

        # Malicious user in Tenant Alpha crafts allowed_scopes with 'domain:tenant_beta'
        crafted_principal = PrincipalContext(
            user_id="mallory",
            role=RoleType.ENGINEER,
            allowed_scopes=[f"domain:{DOMAIN_TENANT_B}", "beta_shared"],
        )
        validator = PreFlightValidator(
            repo=in_memory_repo,
            slicer=slicer,
            principal=crafted_principal,
            domain_id=DOMAIN_TENANT_A,
        )

        citation = DocCitationSpan(
            chunk_id=str(beta_chunk_id),
            doc_id="beta_internal_memo",
            doc_title="Beta Confidential Strategy",
            source_uri="urn:beta:memo",
            section_heading="Strategy",
            verbatim_quote="Confidential operational procedures for tenant beta.",
            page_id="beta_internal_memo",
            version="1",
            ticket_key="",
        )

        passed, _, permitted_citations, excised, _ = validator.check_principal_scopes(
            code_slices=[],
            citations=[citation],
            bound_repo=bound_repo,
            principal=crafted_principal,
            domain_id=DOMAIN_TENANT_A,
        )

        # Check if domain: prefix expands candidate_domains
        assert len(permitted_citations) == 1
        assert len(excised) == 0
        assert passed is True

    def test_code_slice_scoping_strictness_has_no_domain_fallback(
        self,
        in_memory_repo: InMemoryRLSRepository,
        slicer: DeterministicSlicer,
        bound_repo: BoundGraphRepository,
    ) -> None:
        """Verify that unlike document citations, code slices have NO candidate domain fallback.

        Code slices are strictly bounded to domain_id via list_visible_code_files.
        A code slice from another domain is always excised.
        """
        # Register code file under quant_research_scout
        code_file_id = uuid4()
        code_file = CodeFileRecord(
            file_id=code_file_id,
            domain_id=DOMAIN_QUANT,
            repo_name="quant-core",
            file_path="src/quant/pricing.py",
            git_commit_sha="commit_quant_1",
            git_blob_sha="blob_quant_1",
            language="python",
            allowed_scopes=["public_eng"],
            raw_content="def calc():\n    return 42\n",
            line_count=2,
        )
        in_memory_repo._files[code_file_id] = code_file

        # User in Tenant Alpha submits code slice from quant-core
        user_alpha = PrincipalContext(
            user_id="user_alpha",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        validator = PreFlightValidator(
            repo=in_memory_repo,
            slicer=slicer,
            principal=user_alpha,
            domain_id=DOMAIN_TENANT_A,
        )

        code_slice = VerbatimCodeSlice(
            repo_name="quant-core",
            file_path="src/quant/pricing.py",
            start_line=1,
            end_line=2,
            qualified_symbol="quant.pricing.calc",
            selection_reason="Cross-domain code slice",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="commit_quant_1",
            git_blob_sha="blob_quant_1",
            verbatim_code_with_lines="1 | def calc():\n2 |     return 42",
            auto_expanded_by_ast=False,
        )

        passed, permitted_slices, _, excised, _ = validator.check_principal_scopes(
            code_slices=[code_slice],
            citations=[],
            bound_repo=bound_repo,
            principal=user_alpha,
            domain_id=DOMAIN_TENANT_A,
        )

        # Code slice MUST be excised because it is not in Tenant Alpha's visible files
        assert len(permitted_slices) == 0
        assert len(excised) == 1
        assert "code:quant-core/src/quant/pricing.py" in excised[0]
        assert passed is False
