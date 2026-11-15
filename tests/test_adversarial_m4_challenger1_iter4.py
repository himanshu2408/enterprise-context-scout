"""Challenger 1 Adversarial Suite — Milestone 4 Iteration 4 Gate Verification.

Empirical verification and stress testing of:
1. Delimiter balancing, string/comment stripping, and unclosed quote detection:
   - Valid code with parens inside strings ("Status: pending (waiting") -> PASS
   - Multiline comments/docstrings with unmatched braces (/* { */) in C#, TS, Python -> PASS
   - Unclosed triple quotes (\"\"\" / ''') in partial slices -> FAIL check_1_syntax
   - Unclosed single-line expressions (data = (1 + 2) in partial slices -> FAIL check_1_syntax
   - Single-line truncated expressions ending in comma (data = (1 + 2,) -> FAIL check_1_syntax
   - Valid truncated multi-line caller windows ending in a comma ',' -> PASS check_1_syntax
   - Nesting depth (>10 levels) in strings/comments vs real code parens -> PASS vs FAIL
2. Candidate domain tenant isolation:
   - Unprivileged tenant principal cannot cite doc chunks from another tenant without scope
   - Shared generic scope names ('public_eng') across tenant boundaries cannot leak
   - Explicit signed domain scope ('domain:<target>') with required chunk scope permits access
   - Explicit signed domain scope without matching chunk scope is rejected
   - Hardcoded fallback domains ('default', 'quant_research_scout') are completely absent
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

DOMAIN_TENANT_ALPHA: Final[str] = "tenant_alpha"
DOMAIN_TENANT_BETA: Final[str] = "tenant_beta"
DOMAIN_TENANT_GAMMA: Final[str] = "tenant_gamma"
DOMAIN_QUANT: Final[str] = "quant_research_scout"
REPO_NAME: Final[str] = "enterprise-service-repo"


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
# SUITE 1: Delimiter Balancing, String/Comment Stripping & Quote Detection
# =============================================================================


class TestAdversarialDelimiterAndQuoteStripping:
    """Empirical verification of delimiter balancing, comments/strings, and quote checks."""

    @pytest.mark.parametrize(
        "intent",
        [ModificationIntent.DIRECT_CALLER, ModificationIntent.IMPORTED_TYPE],
    )
    def test_probe_valid_code_with_parens_inside_string_literals(
        self,
        in_memory_repo: InMemoryRLSRepository,
        slicer: DeterministicSlicer,
        intent: ModificationIntent,
    ) -> None:
        """Probe valid code containing opening/closing parentheses inside string literals.

        Requirement: Valid code with parens inside string literals must PASS check_1_syntax.
        """
        principal = PrincipalContext(
            user_id="alice",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        validator = PreFlightValidator(
            repo=in_memory_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_TENANT_ALPHA,
        )

        # 1. Double-quoted string literal containing unclosed '('
        double_quote_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service/status.py",
            start_line=1,
            end_line=3,
            qualified_symbol=f"service.status.{intent.value}_double",
            selection_reason="Valid code with paren inside double-quoted string",
            modification_intent=intent,
            git_commit_sha="sha_test_101",
            git_blob_sha="blob_test_101",
            verbatim_code_with_lines=(
                "1 | def format_status(code: int) -> str:\n"
                '2 |     msg = "Status: pending (waiting for confirmation"\n'
                "3 |     return msg"
            ),
            auto_expanded_by_ast=False,
        )

        # 2. Single-quoted string literal containing unclosed ')'
        single_quote_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service/status.py",
            start_line=4,
            end_line=6,
            qualified_symbol=f"service.status.{intent.value}_single",
            selection_reason="Valid code with paren inside single-quoted string",
            modification_intent=intent,
            git_commit_sha="sha_test_102",
            git_blob_sha="blob_test_102",
            verbatim_code_with_lines=(
                "4 | def get_closing_bracket() -> str:\n"
                "5 |     bracket_text = 'closing symbol: ) - do not parse'\n"
                "6 |     return bracket_text"
            ),
            auto_expanded_by_ast=False,
        )

        # 3. TypeScript backtick template literal containing unbalanced parens
        ts_template_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/ui/banner.ts",
            start_line=1,
            end_line=4,
            qualified_symbol=f"ui.banner.{intent.value}_template",
            selection_reason="Valid TS template string with paren inside backticks",
            modification_intent=intent,
            git_commit_sha="sha_test_103",
            git_blob_sha="blob_test_103",
            verbatim_code_with_lines=(
                "1 | export function renderBanner(title: string): string {\n"
                "2 |   const prefix = `Header [App (v2.1)] - banner: (`;\n"
                "3 |   return `${prefix} ${title}`;\n"
                "4 | }"
            ),
            auto_expanded_by_ast=False,
        )

        for s in [double_quote_slice, single_quote_slice, ts_template_slice]:
            passed, errors, missing = validator.check_ast_syntax([s])
            assert passed is True, f"Failed on slice {s.qualified_symbol}: {errors}"
            assert len(errors) == 0
            assert len(missing) == 0

    def test_probe_multiline_comments_with_braces_in_csharp_ts_python(
        self,
        in_memory_repo: InMemoryRLSRepository,
        slicer: DeterministicSlicer,
    ) -> None:
        """Probe multiline comments with braces (/* { */) in C#, TS, Python.

        Requirement: Multiline comments with braces must PASS check_1_syntax.
        """
        principal = PrincipalContext(
            user_id="bob",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        validator = PreFlightValidator(
            repo=in_memory_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_TENANT_ALPHA,
        )

        # C# multiline block comment /* { */
        csharp_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/Domain/OrderValidator.cs",
            start_line=1,
            end_line=6,
            qualified_symbol="Domain.OrderValidator",
            selection_reason="C# block comment with braces",
            modification_intent=ModificationIntent.DIRECT_CALLER,
            git_commit_sha="sha_cs_201",
            git_blob_sha="blob_cs_201",
            verbatim_code_with_lines=(
                "1 | /* { Block comment opening brace without matching close\n"
                "2 |  * Example payload: { key: 123 }\n"
                "3 |  */\n"
                "4 | public class OrderValidator {\n"
                "5 |     public bool IsValid() => true;\n"
                "6 | }"
            ),
            auto_expanded_by_ast=False,
        )

        # TypeScript multiline JSDoc comment /** { */
        ts_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/types/schema.ts",
            start_line=1,
            end_line=5,
            qualified_symbol="types.schema.IConfig",
            selection_reason="TS multiline comment with braces",
            modification_intent=ModificationIntent.IMPORTED_TYPE,
            git_commit_sha="sha_ts_202",
            git_blob_sha="blob_ts_202",
            verbatim_code_with_lines=(
                "1 | /**\n"
                "2 |  * Config map schema: { [key: string]: any\n"
                "3 |  */\n"
                "4 | export interface IConfig {\n"
                "5 |   active: boolean;\n"
                "6 | }"
            ),
            auto_expanded_by_ast=False,
        )

        # Python multiline docstring with braces
        py_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/parser/json_helper.py",
            start_line=1,
            end_line=5,
            qualified_symbol="parser.json_helper.parse_payload",
            selection_reason="Python docstring with braces",
            modification_intent=ModificationIntent.DIRECT_CALLER,
            git_commit_sha="sha_py_203",
            git_blob_sha="blob_py_203",
            verbatim_code_with_lines=(
                "1 | def parse_payload(raw: str):\n"
                '2 |     """Extracts payload { unclosed brace in docstring."""\n'
                "3 |     import json\n"
                "4 |     return json.loads(raw)"
            ),
            auto_expanded_by_ast=False,
        )

        passed_cs, errors_cs, _ = validator.check_ast_syntax([csharp_slice])
        passed_ts, errors_ts, _ = validator.check_ast_syntax([ts_slice])
        passed_py, errors_py, _ = validator.check_ast_syntax([py_slice])

        assert passed_cs is True, f"C# multiline comment failed: {errors_cs}"
        assert passed_ts is True, f"TypeScript multiline comment failed: {errors_ts}"
        assert passed_py is True, f"Python multiline docstring failed: {errors_py}"

    @pytest.mark.parametrize(
        "quote_token",
        ['"""', "'''"],
    )
    def test_probe_unclosed_triple_quotes_in_partial_slices(
        self,
        in_memory_repo: InMemoryRLSRepository,
        slicer: DeterministicSlicer,
        quote_token: str,
    ) -> None:
        """Probe unclosed triple quotes (\"\"\" / ''') in partial slices.

        Requirement: Partial slices with unclosed triple quotes must FAIL check_1_syntax.
        """
        principal = PrincipalContext(
            user_id="charlie",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        validator = PreFlightValidator(
            repo=in_memory_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_TENANT_ALPHA,
        )

        unclosed_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service/worker.py",
            start_line=1,
            end_line=3,
            qualified_symbol="service.worker.process",
            selection_reason="Partial slice with unclosed triple quote",
            modification_intent=ModificationIntent.DIRECT_CALLER,
            git_commit_sha="sha_py_301",
            git_blob_sha="blob_py_301",
            verbatim_code_with_lines=(
                f"1 | def process():\n"
                f"2 |     {quote_token}Unclosed triple quote block\n"
                f"3 |     return None"
            ),
            auto_expanded_by_ast=False,
        )

        passed, errors, missing = validator.check_ast_syntax([unclosed_slice])
        assert passed is False
        assert len(errors) == 1
        assert "Unclosed syntax delimiters in code slice" in errors[0]
        assert len(missing) == 1
        assert missing[0].kind == "syntax"

    def test_probe_unclosed_single_line_expressions_in_partial_slices(
        self,
        in_memory_repo: InMemoryRLSRepository,
        slicer: DeterministicSlicer,
    ) -> None:
        """Probe unclosed single-line expressions (data = (1 + 2) in partial slices.

        Requirement: Partial slices with broken single-line delimiters must FAIL check_1_syntax.
        """
        principal = PrincipalContext(
            user_id="david",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        validator = PreFlightValidator(
            repo=in_memory_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_TENANT_ALPHA,
        )

        # 1. Unclosed paren on single line
        paren_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service/calc.py",
            start_line=1,
            end_line=2,
            qualified_symbol="service.calc.compute",
            selection_reason="Broken single-line paren",
            modification_intent=ModificationIntent.DIRECT_CALLER,
            git_commit_sha="sha_calc_401",
            git_blob_sha="blob_calc_401",
            verbatim_code_with_lines=("1 | def compute():\n2 |     data = (1 + 2"),
            auto_expanded_by_ast=False,
        )

        # 2. Unclosed bracket on single line
        bracket_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service/calc.py",
            start_line=3,
            end_line=4,
            qualified_symbol="service.calc.build_list",
            selection_reason="Broken single-line bracket",
            modification_intent=ModificationIntent.IMPORTED_TYPE,
            git_commit_sha="sha_calc_402",
            git_blob_sha="blob_calc_402",
            verbatim_code_with_lines=("3 | def build_list():\n4 |     items = [1, 2, 3"),
            auto_expanded_by_ast=False,
        )

        # 3. Unclosed brace on single line
        brace_slice = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service/calc.py",
            start_line=5,
            end_line=6,
            qualified_symbol="service.calc.build_dict",
            selection_reason="Broken single-line brace",
            modification_intent=ModificationIntent.DIRECT_CALLER,
            git_commit_sha="sha_calc_403",
            git_blob_sha="blob_calc_403",
            verbatim_code_with_lines=("5 | def build_dict():\n6 |     mapping = {'a': 1, 'b': 2"),
            auto_expanded_by_ast=False,
        )

        for s in [paren_slice, bracket_slice, brace_slice]:
            passed, errors, missing = validator.check_ast_syntax([s])
            assert passed is False, f"Broken slice {s.qualified_symbol} unexpectedly passed!"
            assert len(errors) == 1
            assert "Unclosed syntax delimiters in code slice" in errors[0]
            assert len(missing) == 1

    def test_probe_valid_truncated_multiline_caller_windows_ending_in_comma(
        self,
        in_memory_repo: InMemoryRLSRepository,
        slicer: DeterministicSlicer,
    ) -> None:
        """Probe valid truncated multi-line caller windows ending in a comma.

        Requirement: Valid truncated caller window ending in comma must PASS.
        Adversarial counter-probe: Single-line expression with unclosed paren
        ending in comma must FAIL.
        """
        principal = PrincipalContext(
            user_id="eve",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        validator = PreFlightValidator(
            repo=in_memory_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_TENANT_ALPHA,
        )

        # 1. Multi-line caller window (length >= 2) ending cleanly on a comma
        valid_multiline_caller = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service/pipeline.py",
            start_line=80,
            end_line=83,
            qualified_symbol="service.pipeline.execute_task",
            selection_reason="Multi-line caller parameter window ending in comma",
            modification_intent=ModificationIntent.DIRECT_CALLER,
            git_commit_sha="sha_pipe_501",
            git_blob_sha="blob_pipe_501",
            verbatim_code_with_lines=(
                "80 | def execute_task(\n"
                "81 |     task_id: str,\n"
                "82 |     timeout_seconds: int = 30,\n"
                "83 |     retry_count: int = 3,"
            ),
            auto_expanded_by_ast=False,
        )

        passed_valid, errors_valid, _ = validator.check_ast_syntax([valid_multiline_caller])
        assert passed_valid is True, f"Valid multi-line truncated window failed: {errors_valid}"
        assert len(errors_valid) == 0

        # 2. ADVERSARIAL COUNTER-PROBE: Single-line broken snippet ending in a comma
        # Must NOT be excused by the multi-line window exemption!
        single_line_comma = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service/pipeline.py",
            start_line=90,
            end_line=90,
            qualified_symbol="service.pipeline.bad_expression",
            selection_reason="Single-line broken expression ending in comma",
            modification_intent=ModificationIntent.DIRECT_CALLER,
            git_commit_sha="sha_pipe_502",
            git_blob_sha="blob_pipe_502",
            verbatim_code_with_lines="90 | data = (1 + 2,",
            auto_expanded_by_ast=False,
        )

        passed_bad, errors_bad, _ = validator.check_ast_syntax([single_line_comma])
        assert passed_bad is False, "Single-line expr ending in comma should fail check_1_syntax!"
        assert len(errors_bad) == 1
        assert "Unclosed syntax delimiters in code slice" in errors_bad[0]

    def test_probe_nesting_depth_inside_strings_comments_vs_code_delimiters(
        self,
        in_memory_repo: InMemoryRLSRepository,
        slicer: DeterministicSlicer,
    ) -> None:
        """Probe nesting depth (>10 levels) inside string literals & comments vs real code parens.

        Requirements:
        - Real code with >10 nesting levels must FAIL check_1_syntax.
        - String literals with >10 nesting levels in valid code must PASS check_1_syntax.
        - Comments with >10 nesting levels in valid code must PASS check_1_syntax.
        """
        principal = PrincipalContext(
            user_id="frank",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        validator = PreFlightValidator(
            repo=in_memory_repo,
            slicer=slicer,
            principal=principal,
            domain_id=DOMAIN_TENANT_ALPHA,
        )

        # 1. Real code parentheses nested 14 levels deep (exceeds limit 10)
        code_14_parens = "((((((((((((((42))))))))))))))"
        real_nested_code = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service/deep_math.py",
            start_line=1,
            end_line=2,
            qualified_symbol="service.deep_math.compute_deep",
            selection_reason="14 levels of real code parens",
            modification_intent=ModificationIntent.DIRECT_CALLER,
            git_commit_sha="sha_nest_601",
            git_blob_sha="blob_nest_601",
            verbatim_code_with_lines=(f"1 | def compute_deep():\n2 |     return {code_14_parens}"),
            auto_expanded_by_ast=False,
        )
        passed_real, errors_real, missing_real = validator.check_ast_syntax([real_nested_code])
        assert passed_real is False
        assert len(errors_real) == 1
        assert "Nesting depth (14) exceeds maximum limit (10)" in errors_real[0]
        assert missing_real[0].kind == "syntax"

        # 2. String literal with 15 nested parens in valid code
        str_15_parens = '"(((((((((((((((nested string)))))))))))))))"'
        string_nested = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service/display.py",
            start_line=1,
            end_line=2,
            qualified_symbol="service.display.banner",
            selection_reason="15 parens inside string literal",
            modification_intent=ModificationIntent.DIRECT_CALLER,
            git_commit_sha="sha_nest_602",
            git_blob_sha="blob_nest_602",
            verbatim_code_with_lines=(f"1 | def banner():\n2 |     return {str_15_parens}"),
            auto_expanded_by_ast=False,
        )
        passed_str, errors_str, _ = validator.check_ast_syntax([string_nested])
        assert passed_str is True, f"String literal with parens was falsely flagged: {errors_str}"
        assert len(errors_str) == 0

        # 3. Comment with 15 nested parens in valid code
        comment_nested = VerbatimCodeSlice(
            repo_name=REPO_NAME,
            file_path="src/service/display.py",
            start_line=3,
            end_line=5,
            qualified_symbol="service.display.render",
            selection_reason="15 parens inside comment",
            modification_intent=ModificationIntent.DIRECT_CALLER,
            git_commit_sha="sha_nest_603",
            git_blob_sha="blob_nest_603",
            verbatim_code_with_lines=(
                "3 | # (((((((((((((((nested comment note)))))))))))))))\n"
                "4 | def render():\n"
                "5 |     return 'rendered'"
            ),
            auto_expanded_by_ast=False,
        )
        passed_cmt, errors_cmt, _ = validator.check_ast_syntax([comment_nested])
        assert passed_cmt is True, f"Comment with parens was falsely flagged: {errors_cmt}"
        assert len(errors_cmt) == 0


# =============================================================================
# SUITE 2: Candidate Domain Tenant Isolation Stressing
# =============================================================================


class TestAdversarialCandidateDomainTenantIsolation:
    """Empirical verification of cross-tenant boundary isolation and candidate domains."""

    def test_unprivileged_tenant_principal_cannot_cite_other_tenant_chunk_with_shared_scope(
        self,
        in_memory_repo: InMemoryRLSRepository,
        slicer: DeterministicSlicer,
        bound_repo: BoundGraphRepository,
    ) -> None:
        """Prove that an unprivileged tenant principal cannot cite doc chunks from another tenant.

        # Even when principal and foreign chunk share generic scope 'public_eng',
        # candidate domains must NOT include foreign tenant or fallbacks.
        """
        # Foreign tenant doc chunk in DOMAIN_TENANT_BETA
        secret_chunk_id = uuid4()
        secret_chunk = DocChunkRecord(
            chunk_id=secret_chunk_id,
            domain_id=DOMAIN_TENANT_BETA,
            doc_id="beta_secret_runbook",
            doc_title="Beta Incident Runbook",
            source_uri="urn:beta:runbook",
            market_scope=MarketRegime.GLOBAL_MULTI,
            published_date=datetime.date(2026, 1, 1),
            authority_weight=1.0,
            section_heading="Key Vault Access",
            contextual_header="Beta Incident Runbook",
            verbatim_content="Root credentials for Beta production key vault cluster.",
            metadata_json={},
            allowed_scopes=["public_eng"],  # Generic scope shared across companies
            embedding=[0.05] * 768,
        )
        in_memory_repo._docs[secret_chunk_id] = secret_chunk

        # Unprivileged principal in DOMAIN_TENANT_ALPHA with only 'public_eng'
        alice_tenant_a = PrincipalContext(
            user_id="alice_tenant_a",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        validator = PreFlightValidator(
            repo=in_memory_repo,
            slicer=slicer,
            principal=alice_tenant_a,
            domain_id=DOMAIN_TENANT_ALPHA,
        )

        citation = DocCitationSpan(
            chunk_id=str(secret_chunk_id),
            doc_id="beta_secret_runbook",
            doc_title="Beta Incident Runbook",
            source_uri="urn:beta:runbook",
            section_heading="Key Vault Access",
            verbatim_quote="Root credentials for Beta production key vault cluster.",
            page_id="beta_secret_runbook",
            version="1",
            ticket_key="",
        )

        passed, _, permitted_citations, excised, missing = validator.check_principal_scopes(
            code_slices=[],
            citations=[citation],
            bound_repo=bound_repo,
            principal=alice_tenant_a,
            domain_id=DOMAIN_TENANT_ALPHA,
        )

        # Verification: Chunk is from tenant_beta. Alice has no domain:tenant_beta scope.
        # It MUST be excised, zero permitted citations, passed is False.
        assert passed is False
        assert len(permitted_citations) == 0
        assert len(excised) == 1
        assert f"doc:beta_secret_runbook:{secret_chunk_id}" in excised[0]
        assert any(m.kind == "scope" for m in missing)
        assert any(
            f"lacks visibility into document chunk '{secret_chunk_id}'" in m.reason for m in missing
        )

    def test_unprivileged_tenant_principal_cannot_access_quant_research_scout(
        self,
        in_memory_repo: InMemoryRLSRepository,
        slicer: DeterministicSlicer,
        bound_repo: BoundGraphRepository,
    ) -> None:
        """Verify that hardcoded fallback to quant_research_scout is completely removed.

        A tenant principal without explicit 'domain:quant_research_scout' cannot cite chunks
        from quant_research_scout, even with generic scope names.
        """
        quant_chunk_id = uuid4()
        quant_chunk = DocChunkRecord(
            chunk_id=quant_chunk_id,
            domain_id=DOMAIN_QUANT,
            doc_id="quant_alpha_factor",
            doc_title="Quant Alpha Factor",
            source_uri="urn:quant:factor",
            market_scope=MarketRegime.GLOBAL_MULTI,
            published_date=datetime.date(2026, 1, 1),
            authority_weight=1.0,
            section_heading="Factor Definition",
            contextual_header="Quant Alpha Factor",
            verbatim_content="Proprietary statistical arbitrage alpha weights matrix.",
            metadata_json={},
            allowed_scopes=["public_eng", "quant_research"],
            embedding=[0.02] * 768,
        )
        in_memory_repo._docs[quant_chunk_id] = quant_chunk

        # Principal in tenant_gamma
        gamma_user = PrincipalContext(
            user_id="gamma_developer",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        validator = PreFlightValidator(
            repo=in_memory_repo,
            slicer=slicer,
            principal=gamma_user,
            domain_id=DOMAIN_TENANT_GAMMA,
        )

        citation = DocCitationSpan(
            chunk_id=str(quant_chunk_id),
            doc_id="quant_alpha_factor",
            doc_title="Quant Alpha Factor",
            source_uri="urn:quant:factor",
            section_heading="Factor Definition",
            verbatim_quote="Proprietary statistical arbitrage alpha weights matrix.",
            page_id="quant_alpha_factor",
            version="1",
            ticket_key="",
        )

        passed, _, permitted, excised, _missing = validator.check_principal_scopes(
            code_slices=[],
            citations=[citation],
            bound_repo=bound_repo,
            principal=gamma_user,
            domain_id=DOMAIN_TENANT_GAMMA,
        )

        assert passed is False
        assert len(permitted) == 0
        assert len(excised) == 1
        assert f"doc:quant_alpha_factor:{quant_chunk_id}" in excised[0]

    def test_signed_domain_scope_allows_access_only_with_matching_chunk_scope(
        self,
        in_memory_repo: InMemoryRLSRepository,
        slicer: DeterministicSlicer,
        bound_repo: BoundGraphRepository,
    ) -> None:
        """Prove that explicit signed domain scope ('domain:<target>') allows cross-domain citation
        IF AND ONLY IF chunk allowed_scopes intersect with principal allowed_scopes.
        """
        partner_chunk_id = uuid4()
        partner_chunk = DocChunkRecord(
            chunk_id=partner_chunk_id,
            domain_id=DOMAIN_TENANT_BETA,
            doc_id="shared_api_spec",
            doc_title="Partner API Spec",
            source_uri="urn:beta:partner_api",
            market_scope=MarketRegime.GLOBAL_MULTI,
            published_date=datetime.date(2026, 1, 1),
            authority_weight=1.0,
            section_heading="Endpoints",
            contextual_header="Partner API Spec",
            verbatim_content="Federated OAuth token exchange specifications for partner.",
            metadata_json={},
            allowed_scopes=["partner_integration_scope"],
            embedding=[0.03] * 768,
        )
        in_memory_repo._docs[partner_chunk_id] = partner_chunk

        # Case A: Principal has signed domain scope BUT lacks the required chunk scope
        unauthorized_principal = PrincipalContext(
            user_id="authorized_domain_only",
            role=RoleType.ENGINEER,
            allowed_scopes=[f"domain:{DOMAIN_TENANT_BETA}", "internal_only_scope"],
        )
        val_a = PreFlightValidator(
            repo=in_memory_repo,
            slicer=slicer,
            principal=unauthorized_principal,
            domain_id=DOMAIN_TENANT_ALPHA,
        )

        citation = DocCitationSpan(
            chunk_id=str(partner_chunk_id),
            doc_id="shared_api_spec",
            doc_title="Partner API Spec",
            source_uri="urn:beta:partner_api",
            section_heading="Endpoints",
            verbatim_quote="Federated OAuth token exchange specifications for partner.",
            page_id="shared_api_spec",
            version="1",
            ticket_key="",
        )

        passed_a, _, permitted_a, excised_a, missing_a = val_a.check_principal_scopes(
            code_slices=[],
            citations=[citation],
            bound_repo=bound_repo,
            principal=unauthorized_principal,
            domain_id=DOMAIN_TENANT_ALPHA,
        )
        # Even with domain scope, lack of chunk scope in RLS repository blocks visibility
        assert passed_a is False
        assert len(permitted_a) == 0
        assert len(excised_a) == 1
        assert any("lacks visibility into document chunk" in m.reason for m in missing_a)

        # Case A2: Pre-loaded chunk metadata without matching scope triggers explicit scope check
        passed_a2, _, permitted_a2, excised_a2, missing_a2 = val_a.check_principal_scopes(
            code_slices=[],
            citations=[citation],
            bound_repo=bound_repo,
            principal=unauthorized_principal,
            domain_id=DOMAIN_TENANT_ALPHA,
            doc_chunks=[partner_chunk],
        )
        assert passed_a2 is False
        assert len(permitted_a2) == 0
        assert len(excised_a2) == 1
        assert any("lacks required scopes" in m.reason for m in missing_a2)

        # Case B: Principal has BOTH signed domain scope AND matching chunk scope
        authorized_principal = PrincipalContext(
            user_id="fully_authorized_partner",
            role=RoleType.ENGINEER,
            allowed_scopes=[
                f"domain:{DOMAIN_TENANT_BETA}",
                "partner_integration_scope",
            ],
        )
        val_b = PreFlightValidator(
            repo=in_memory_repo,
            slicer=slicer,
            principal=authorized_principal,
            domain_id=DOMAIN_TENANT_ALPHA,
        )

        passed_b, _, permitted_b, excised_b, missing_b = val_b.check_principal_scopes(
            code_slices=[],
            citations=[citation],
            bound_repo=bound_repo,
            principal=authorized_principal,
            domain_id=DOMAIN_TENANT_ALPHA,
        )
        # Fully authorized: permitted
        assert passed_b is True
        assert len(permitted_b) == 1
        assert len(excised_b) == 0
        assert len(missing_b) == 0

    def test_check_citations_candidate_domain_isolation(
        self,
        in_memory_repo: InMemoryRLSRepository,
        slicer: DeterministicSlicer,
        bound_repo: BoundGraphRepository,
    ) -> None:
        """Verify check_citations in PreFlightValidator enforces candidate domain isolation.

        When check_citations runs to verify verbatim quotes of missing chunk IDs,
        it must NOT query quant_research_scout or default unless explicitly authorized.
        """
        isolated_chunk_id = uuid4()
        isolated_chunk = DocChunkRecord(
            chunk_id=isolated_chunk_id,
            domain_id=DOMAIN_QUANT,
            doc_id="quant_private_formula",
            doc_title="Private Formula",
            source_uri="urn:quant:private",
            market_scope=MarketRegime.GLOBAL_MULTI,
            published_date=datetime.date(2026, 1, 1),
            authority_weight=1.0,
            section_heading="Formula",
            contextual_header="Private Formula",
            verbatim_content="Exact mathematical closed form solution for alpha vector pricing.",
            metadata_json={},
            allowed_scopes=["public_eng"],
            embedding=[0.01] * 768,
        )
        in_memory_repo._docs[isolated_chunk_id] = isolated_chunk

        # Principal in tenant_alpha
        tenant_user = PrincipalContext(
            user_id="tenant_user_1",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        validator = PreFlightValidator(
            repo=in_memory_repo,
            slicer=slicer,
            principal=tenant_user,
            domain_id=DOMAIN_TENANT_ALPHA,
        )

        citation = DocCitationSpan(
            chunk_id=str(isolated_chunk_id),
            doc_id="quant_private_formula",
            doc_title="Private Formula",
            source_uri="urn:quant:private",
            section_heading="Formula",
            verbatim_quote="Exact mathematical closed form solution for alpha vector pricing.",
            page_id="quant_private_formula",
            version="1",
            ticket_key="",
        )

        # check_citations is called directly
        passed, failed_reasons = validator.check_citations(
            citations=[citation],
            bound_repo=bound_repo,
            principal=tenant_user,
            domain_id=DOMAIN_TENANT_ALPHA,
            doc_chunks=None,  # Forces bound_repo lookup
        )

        # Since candidate_domains does not include quant_research_scout, the chunk is not retrieved
        # and verification fails.
        assert passed is False
        assert len(failed_reasons) == 1
        assert "not in corpus" in failed_reasons[0].lower() or "failed" in failed_reasons[0].lower()
