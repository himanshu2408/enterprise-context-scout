"""Adversarial Challenger 2 Stress Test Suite (Milestone 4).

Empirically challenges:
1. Exact normalized-substring matching with zero tolerance on realistic populated text:
   - 1-word substitutions in 50+ word quotes
   - Negation reversals ("must" -> "must not", "allow" -> "deny")
   - Numeric mutations (0.95 -> 0.96, port 8080 -> 8081)
   - Intra-sentence omissions
   - Whitespace permutations (tabs, newlines, multiple spaces, NBSP \\u00A0)
2. Homoglyph spoofing rejection vs. Authentic internationalized Unicode acceptance:
   - Cyrillic/Greek lookalike injection in Latin strings
   - Authentic Russian documentation
   - Greek mathematical symbols (\\alpha, \\beta, \\lambda)
   - German umlauts (ä, ö, ü, ß)
   - French accents (é, è, ç)
   - CJK characters
3. Populated repository benchmark validity:
   - Demonstrating failure modes when unanswerable tasks run against non-empty repositories.
"""

from __future__ import annotations

import pathlib
import uuid
from typing import Final

import pytest

from context_scout.agents.citation_gate import (
    CitationFailureCode,
    DocumentCitationGate,
)
from context_scout.agents.claim_verifier import (
    ClaimFailureCode,
    ClaimVerifier,
    detect_homoglyph_spoofing,
    normalize_claim_text,
)
from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.scout_workflow import ScoutWorkflow
from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer
from context_scout.ast_engine.models import SlicerErrorCode
from context_scout.db.repositories import (
    ASTSymbolRecord,
    CodeFileRecord,
    InMemoryRLSRepository,
)
from context_scout.schemas.contracts import (
    DocCitationSpan,
    InsufficientContextDiagnostic,
    PrincipalContext,
    RoleType,
)

# Realistic 75-word enterprise specification source text
REALISTIC_SPEC_TEXT: Final[str] = (
    "The enterprise authentication gateway must strictly enforce signed JSON Web Tokens "
    "for all incoming RPC requests across cluster nodes. Under no circumstances should the "
    "edge proxy allow unauthenticated fallback mode or anonymous bypass. The idle session "
    "timeout is fixed at 500 milliseconds and the replication factor across the distributed "
    "ledger must be maintained at 3 replicas. Any signature mismatch or malformed credential "
    "must immediately trigger a security audit event and terminate the connection."
)


class TestAdversarialNormalizedSubstring:
    """Empirical verification of exact NFKC normalized-substring matching."""

    @pytest.fixture
    def slicer(self) -> DeterministicSlicer:
        return DeterministicSlicer()

    @pytest.fixture
    def verifier(self) -> ClaimVerifier:
        return ClaimVerifier()

    @pytest.fixture
    def citation_gate(self) -> DocumentCitationGate:
        return DocumentCitationGate(strict_freshness=True)

    def test_verbatim_quote_passes_cleanly(
        self,
        slicer: DeterministicSlicer,
        verifier: ClaimVerifier,
        citation_gate: DocumentCitationGate,
    ) -> None:
        quote = (
            "The enterprise authentication gateway must strictly enforce signed JSON Web Tokens "
            "for all incoming RPC requests across cluster nodes. Under no circumstances should the "
            "edge proxy allow unauthenticated fallback mode or anonymous bypass."
        )
        assert len(quote.split()) >= 30

        s_res = slicer.verify_quote(quote, REALISTIC_SPEC_TEXT)
        assert s_res.is_valid is True
        assert s_res.token_overlap_ratio == 1.0

        c_rec = verifier.verify_single_claim(quote, [REALISTIC_SPEC_TEXT])
        assert c_rec.is_verified is True
        assert c_rec.failure_code == ClaimFailureCode.EXACT_MATCH_PASSED

        cit = DocCitationSpan(
            chunk_id="chunk_real_01",
            doc_id="spec_auth_v1",
            doc_title="Auth Spec",
            source_uri="https://docs.enterprise/auth",
            section_heading="Gateway Security",
            verbatim_quote=quote,
        )
        corpus = {cit.chunk_id: REALISTIC_SPEC_TEXT}
        gate_res = citation_gate.verify_citations([cit], corpus, require_citations=True)
        assert gate_res.passed is True
        assert gate_res.verified_count == 1

    def test_whitespace_variance_collapses_cleanly(
        self,
        slicer: DeterministicSlicer,
        verifier: ClaimVerifier,
    ) -> None:
        quote_with_whitespace_chaos = (
            "The\tenterprise\n\nauthentication   gateway\u00a0must\tstrictly\u00a0\u00a0enforce\n"
            "signed\tJSON Web\t\tTokens for all   incoming RPC requests across cluster nodes."
        )
        norm = normalize_claim_text(quote_with_whitespace_chaos)
        assert "\u00a0" not in norm
        assert "\t" not in norm
        assert "\n" not in norm
        assert "  " not in norm

        s_res = slicer.verify_quote(quote_with_whitespace_chaos, REALISTIC_SPEC_TEXT)
        assert s_res.is_valid is True
        assert s_res.token_overlap_ratio == 1.0

        c_rec = verifier.verify_single_claim(quote_with_whitespace_chaos, [REALISTIC_SPEC_TEXT])
        assert c_rec.is_verified is True
        assert c_rec.failure_code == ClaimFailureCode.EXACT_MATCH_PASSED

    @pytest.mark.parametrize(
        "original_word,substituted_word",
        [
            ("enforce", "require"),
            ("strictly", "loosely"),
            ("allow", "permit"),
            ("unauthenticated", "unauthorized"),
            ("audit", "logging"),
            ("terminate", "close"),
        ],
    )
    def test_single_word_substitutions_rejected_100_percent(
        self,
        slicer: DeterministicSlicer,
        verifier: ClaimVerifier,
        citation_gate: DocumentCitationGate,
        original_word: str,
        substituted_word: str,
    ) -> None:
        base_quote = REALISTIC_SPEC_TEXT
        assert len(base_quote.split()) >= 50
        assert original_word in base_quote

        tampered_quote = base_quote.replace(original_word, substituted_word, 1)
        assert tampered_quote != base_quote

        s_res = slicer.verify_quote(tampered_quote, REALISTIC_SPEC_TEXT)
        assert s_res.is_valid is False
        assert s_res.token_overlap_ratio == 0.0
        assert s_res.error_code == SlicerErrorCode.HALLUCINATED_QUOTE

        c_rec = verifier.verify_single_claim(tampered_quote, [REALISTIC_SPEC_TEXT])
        assert c_rec.is_verified is False
        assert c_rec.failure_code == ClaimFailureCode.NOT_IN_SOURCE

        cit = DocCitationSpan(
            chunk_id="chunk_tamper_01",
            doc_id="spec_auth_v1",
            doc_title="Auth Spec",
            source_uri="https://docs.enterprise/auth",
            section_heading="Gateway Security",
            verbatim_quote=tampered_quote,
        )
        gate_res = citation_gate.verify_citations(
            [cit], {cit.chunk_id: REALISTIC_SPEC_TEXT}, require_citations=True
        )
        assert gate_res.passed is False
        assert gate_res.failed_count == 1
        assert gate_res.details[0].error_code == CitationFailureCode.HALLUCINATED_QUOTE

    @pytest.mark.parametrize(
        "negation_tamper",
        [
            ("must strictly enforce", "must not strictly enforce"),
            ("should the edge proxy allow", "should the edge proxy deny"),
            ("Under no circumstances", "Under any circumstances"),
            ("unauthenticated fallback mode", "authenticated fallback mode"),
        ],
    )
    def test_negation_reversals_rejected_100_percent(
        self,
        slicer: DeterministicSlicer,
        verifier: ClaimVerifier,
        citation_gate: DocumentCitationGate,
        negation_tamper: tuple[str, str],
    ) -> None:
        original_phrase, altered_phrase = negation_tamper
        quote = (
            "The enterprise authentication gateway must strictly enforce signed JSON Web Tokens "
            "for all incoming RPC requests across cluster nodes. Under no circumstances should the "
            "edge proxy allow unauthenticated fallback mode or anonymous bypass."
        )
        assert original_phrase in quote
        tampered_quote = quote.replace(original_phrase, altered_phrase)

        s_res = slicer.verify_quote(tampered_quote, REALISTIC_SPEC_TEXT)
        assert s_res.is_valid is False
        assert s_res.token_overlap_ratio == 0.0

        c_rec = verifier.verify_single_claim(tampered_quote, [REALISTIC_SPEC_TEXT])
        assert c_rec.is_verified is False
        assert c_rec.failure_code == ClaimFailureCode.NOT_IN_SOURCE

        cit = DocCitationSpan(
            chunk_id="chunk_neg_01",
            doc_id="spec_auth_v1",
            doc_title="Auth Spec",
            source_uri="https://docs.enterprise/auth",
            section_heading="Gateway Security",
            verbatim_quote=tampered_quote,
        )
        gate_res = citation_gate.verify_citations(
            [cit], {cit.chunk_id: REALISTIC_SPEC_TEXT}, require_citations=True
        )
        assert gate_res.passed is False

    @pytest.mark.parametrize(
        "numeric_tamper",
        [
            ("500 milliseconds", "501 milliseconds"),
            ("500 milliseconds", "5000 milliseconds"),
            ("3 replicas", "2 replicas"),
            ("3 replicas", "5 replicas"),
        ],
    )
    def test_numeric_mutations_rejected_100_percent(
        self,
        slicer: DeterministicSlicer,
        verifier: ClaimVerifier,
        citation_gate: DocumentCitationGate,
        numeric_tamper: tuple[str, str],
    ) -> None:
        orig_num, altered_num = numeric_tamper
        quote = (
            "The idle session timeout is fixed at 500 milliseconds and the replication factor "
            "across the distributed ledger must be maintained at 3 replicas."
        )
        tampered_quote = quote.replace(orig_num, altered_num)

        s_res = slicer.verify_quote(tampered_quote, REALISTIC_SPEC_TEXT)
        assert s_res.is_valid is False

        c_rec = verifier.verify_single_claim(tampered_quote, [REALISTIC_SPEC_TEXT])
        assert c_rec.is_verified is False

        cit = DocCitationSpan(
            chunk_id="chunk_num_01",
            doc_id="spec_auth_v1",
            doc_title="Auth Spec",
            source_uri="https://docs.enterprise/auth",
            section_heading="Gateway Security",
            verbatim_quote=tampered_quote,
        )
        gate_res = citation_gate.verify_citations(
            [cit], {cit.chunk_id: REALISTIC_SPEC_TEXT}, require_citations=True
        )
        assert gate_res.passed is False

    @pytest.mark.parametrize(
        "dropped_token",
        [
            "strictly ",
            "incoming ",
            "distributed ",
            "security ",
        ],
    )
    def test_intra_sentence_omissions_rejected_100_percent(
        self,
        slicer: DeterministicSlicer,
        verifier: ClaimVerifier,
        citation_gate: DocumentCitationGate,
        dropped_token: str,
    ) -> None:
        quote = (
            "The enterprise authentication gateway must strictly enforce signed JSON Web Tokens "
            "for all incoming RPC requests across cluster nodes. The idle session timeout is fixed "
            "at 500 milliseconds and the replication factor across the distributed ledger must "
            "be maintained at 3 replicas. Any signature mismatch or malformed credential must "
            "immediately trigger a security audit event and terminate the connection."
        )
        assert dropped_token in quote
        tampered_quote = quote.replace(dropped_token, "", 1)

        s_res = slicer.verify_quote(tampered_quote, REALISTIC_SPEC_TEXT)
        assert s_res.is_valid is False

        c_rec = verifier.verify_single_claim(tampered_quote, [REALISTIC_SPEC_TEXT])
        assert c_rec.is_verified is False

        cit = DocCitationSpan(
            chunk_id="chunk_omiss_01",
            doc_id="spec_auth_v1",
            doc_title="Auth Spec",
            source_uri="https://docs.enterprise/auth",
            section_heading="Gateway Security",
            verbatim_quote=tampered_quote,
        )
        gate_res = citation_gate.verify_citations(
            [cit], {cit.chunk_id: REALISTIC_SPEC_TEXT}, require_citations=True
        )
        assert gate_res.passed is False


class TestAdversarialHomoglyphsAndUnicode:
    """Empirical challenge of homoglyph spoofing detector vs authentic Unicode."""

    @pytest.fixture
    def verifier(self) -> ClaimVerifier:
        return ClaimVerifier()

    @pytest.fixture
    def citation_gate(self) -> DocumentCitationGate:
        return DocumentCitationGate()

    @pytest.mark.parametrize(
        "attack_name,spoofed_text,target_word",
        [
            (
                "Cyrillic a in allow",
                "The gateway should \u0430llow fallback mode.",
                "\u0430llow",
            ),
            (
                "Greek omicron in token",
                "Extract the authentication t\u03bfken from headers.",
                "t\u03bfken",
            ),
            (
                "Cyrillic e in service",
                "Connect to the billing s\u0435rvice endpoint.",
                "s\u0435rvice",
            ),
            (
                "Cyrillic o in root",
                "Access granted to r\u043e\u043et directory.",
                "r\u043e\u043et",
            ),
            (
                "Greek nu in event",
                "Trigger a security audit e\u03bde\u03bdt.",
                "e\u03bde\u03bdt",
            ),
            (
                "Greek tau in target",
                "Dispatch request to the \u03c4arget host.",
                "\u03c4arget",
            ),
            (
                "Pure Cyrillic lookalike token in Latin context",
                "Parameter \u0430 must be configured.",
                "\u0430",
            ),
            (
                "Pure Greek lookalike token in Latin context",
                "Variable \u03bf is reserved for indexing.",
                "\u03bf",
            ),
        ],
    )
    def test_adversarial_homoglyphs_rejected_100_percent(
        self,
        verifier: ClaimVerifier,
        citation_gate: DocumentCitationGate,
        attack_name: str,
        spoofed_text: str,
        target_word: str,
    ) -> None:
        is_hg, reason = detect_homoglyph_spoofing(spoofed_text)
        assert is_hg is True, f"Failed to detect {attack_name}"
        assert ("Mixed-script" in reason) or ("Pure homoglyph" in reason)

        source = spoofed_text.replace(target_word, "target")
        c_rec = verifier.verify_single_claim(spoofed_text, [source])
        assert c_rec.is_verified is False
        assert c_rec.failure_code == ClaimFailureCode.HOMOGLYPH_SPOOFING
        assert c_rec.is_homoglyph_attack is True

        cit = DocCitationSpan(
            chunk_id="chk_hg_01",
            doc_id="doc_hg",
            doc_title="Security Doc",
            source_uri="https://docs.enterprise/sec",
            section_heading="Token Security",
            verbatim_quote=spoofed_text,
        )
        gate_res = citation_gate.verify_citations(
            [cit], {cit.chunk_id: source}, require_citations=True
        )
        assert gate_res.passed is False
        assert gate_res.details[0].error_code == CitationFailureCode.HALLUCINATED_QUOTE
        assert "homoglyph spoofing" in (gate_res.details[0].error_reason or "")

    @pytest.mark.parametrize(
        "case_name,authentic_text",
        [
            (
                "Full Russian documentation",
                "Конфигурация шлюза безопасности требует обязательной проверки цифровой подписи.",
            ),
            (
                "Greek mathematical symbols",
                "Empirical loss: L = \u03b1 * L_{rec} + \u03b2 * L_{reg} + \u03bb * L_{aux}.",
            ),
            (
                "German technical umlauts",
                "Überprüfung der Rückgabewerte und Berechtigungen läuft reibungslos (ä, ö, ü, ß).",
            ),
            (
                "French security policy",
                "Système de vérification des identifiants et des accès sécurisés pour les usagers.",
            ),
            (
                "CJK multilingual specification",
                "企业上下文搜索与确定性验证引擎 (日本語: インデックス作成, 한국어: 결정적 검증).",
            ),
        ],
    )
    def test_authentic_internationalized_unicode_accepted_0_false_rejections(
        self,
        verifier: ClaimVerifier,
        citation_gate: DocumentCitationGate,
        case_name: str,
        authentic_text: str,
    ) -> None:
        is_hg, reason = detect_homoglyph_spoofing(authentic_text)
        assert is_hg is False, f"False positive on {case_name}: {reason}"

        c_rec = verifier.verify_single_claim(authentic_text, [authentic_text])
        assert c_rec.is_verified is True
        assert c_rec.failure_code == ClaimFailureCode.EXACT_MATCH_PASSED

        cit = DocCitationSpan(
            chunk_id=f"chk_intl_{case_name[:6]}",
            doc_id="doc_intl",
            doc_title="International Spec",
            source_uri="https://docs.enterprise/intl",
            section_heading="Localization",
            verbatim_quote=authentic_text,
        )
        gate_res = citation_gate.verify_citations(
            [cit], {cit.chunk_id: authentic_text}, require_citations=True
        )
        assert gate_res.passed is True
        assert gate_res.verified_count == 1


class TestPopulatedRepositoryAbstentionChallenge:
    """Empirical proof of the empty repository benchmark vulnerability."""

    @pytest.fixture
    def domain_registry(self) -> DomainRegistry:
        domains_dir = pathlib.Path(__file__).resolve().parent.parent / "domains"
        reg = DomainRegistry(domains_dir)
        reg.load_all()
        return reg

    def test_unanswerable_query_on_empty_repo_abstains_due_to_empty_context(
        self,
        domain_registry: DomainRegistry,
    ) -> None:
        empty_repo = InMemoryRLSRepository()
        workflow = ScoutWorkflow(registry=domain_registry, repo=empty_repo)
        principal = PrincipalContext(
            user_id="auditor",
            role=RoleType.ENGINEER,
            allowed_scopes=["repo:enterprise/billing-service"],
        )
        prompt = "Refactor class NonExistentSymbol_V000Handler in billing to use async lock."
        result = workflow.run(prompt, principal, "trading_code_scout")

        assert isinstance(result, InsufficientContextDiagnostic)
        assert result.paid_tokens_spent == 0
        assert "domain:trading_code_scout:no_verifiable_context" in result.missing_items

    def test_unanswerable_query_on_populated_repo_reveals_partitioning_leak(
        self,
        domain_registry: DomainRegistry,
    ) -> None:
        populated_repo = InMemoryRLSRepository()
        principal = PrincipalContext(
            user_id="auditor",
            role=RoleType.ENGINEER,
            allowed_scopes=["repo:enterprise/billing-service"],
        )

        fid = uuid.uuid4()
        code = (
            "class ExistingBillingManager:\n"
            "    def process_invoice(self, invoice_id: str):\n"
            "        pass\n"
        )
        f_rec = CodeFileRecord(
            file_id=fid,
            domain_id="trading_code_scout",
            repo_name="enterprise/billing-service",
            file_path="src/billing.py",
            git_commit_sha="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
            git_blob_sha="b1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
            language="python",
            allowed_scopes=["repo:enterprise/billing-service"],
            raw_content=code,
            line_count=3,
        )
        populated_repo.insert_code_file(principal, f_rec)
        populated_repo.insert_ast_symbol(
            principal,
            ASTSymbolRecord(
                symbol_id=uuid.uuid4(),
                file_id=fid,
                domain_id="trading_code_scout",
                qualified_name="enterprise/billing-service:src/billing.py:ExistingBillingManager",
                symbol_kind="class",
                start_line=1,
                end_line=3,
                signature_skeleton="class ExistingBillingManager:",
                contextual_summary="Billing class",
                allowed_scopes=["repo:enterprise/billing-service"],
            ),
        )

        workflow = ScoutWorkflow(registry=domain_registry, repo=populated_repo)
        prompt = "Refactor class NonExistentSymbol_V000Handler in billing to use async lock."
        result = workflow.run(prompt, principal, "trading_code_scout")

        # Empirically verify remediation:
        # Prompt requested nonexistent 'NonExistentSymbol_V000Handler', workflow properly abstains!
        assert isinstance(result, InsufficientContextDiagnostic)
        assert result.paid_tokens_spent == 0
        assert any("NonExistentSymbol_V000Handler" in str(item) for item in result.missing_items)
