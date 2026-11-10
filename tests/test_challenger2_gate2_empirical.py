# ruff: noqa: RUF001
"""Empirical Challenger 2 Suite — Milestone 4 Iteration 2 Gate 2.

Comprehensive adversarial verification covering:
1. Populated repo discrimination: unanswerable queries on absent target symbols
   (PascalCase, camelCase, snake_case, file paths, quoted tokens).
2. Bidirectional homoglyph spoofing rejection & authentic international Unicode acceptance:
   - Cyrillic/Greek in Latin
   - Latin in Cyrillic/Greek
   - Pure lookalikes in Latin context
   - Authentic Russian, Ukrainian, Greek (prose and math), German, French, Spanish, CJK.
3. Exact Unicode NFKC substring matching with zero tolerance:
   - Case sensitivity
   - 1-char mutations
   - Punctuation drift
   - Numeric drift
   - Negation insertion
   - Omission
4. Marginal RRF threshold score verification (0.6499 vs 0.6500):
   - Direct gate check
   - End-to-end PreFlightGate context pack validation
"""

from __future__ import annotations

import pathlib
import uuid

import pytest

from context_scout.agents.claim_verifier import (
    ClaimFailureCode,
    ClaimVerifier,
    detect_homoglyph_spoofing,
)
from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.preflight_gate import (
    GateAbstentionResult,
    PreFlightValidator,
)
from context_scout.agents.scout_workflow import ScoutWorkflow
from context_scout.db.repositories import (
    ASTSymbolRecord,
    CodeFileRecord,
    InMemoryRLSRepository,
)
from context_scout.schemas.contracts import (
    InsufficientContextDiagnostic,
    ModificationIntent,
    PrincipalContext,
    RoleType,
    VerbatimCodeSlice,
)


@pytest.fixture
def domain_registry() -> DomainRegistry:
    domains_dir = pathlib.Path(__file__).resolve().parent.parent / "domains"
    reg = DomainRegistry(domains_dir)
    reg.load_all()
    return reg


@pytest.fixture
def populated_repo() -> InMemoryRLSRepository:
    repo = InMemoryRLSRepository()
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
        "\n"
        "def calculate_tax(amount: float) -> float:\n"
        "    return amount * 0.19\n"
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
        line_count=6,
    )
    repo.insert_code_file(principal, f_rec)

    repo.insert_ast_symbol(
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
    repo.insert_ast_symbol(
        principal,
        ASTSymbolRecord(
            symbol_id=uuid.uuid4(),
            file_id=fid,
            domain_id="trading_code_scout",
            qualified_name="enterprise/billing-service:src/billing.py:calculate_tax",
            symbol_kind="function",
            start_line=5,
            end_line=6,
            signature_skeleton="def calculate_tax(amount: float) -> float:",
            contextual_summary="Tax calculation function",
            allowed_scopes=["repo:enterprise/billing-service"],
        ),
    )
    return repo


# =============================================================================
# 1. Populated Repository Discrimination on Absent Symbols
# =============================================================================


class TestPopulatedRepoDiscrimination:
    """Stress-test unanswerable queries targeting absent symbols in populated repo."""

    @pytest.mark.parametrize(
        "prompt,expected_missing",
        [
            (
                "Refactor class NonExistentSymbol_V000Handler in billing to use async lock.",
                "NonExistentSymbol_V000Handler",
            ),
            (
                "Modify method processRefundTransaction in billing to support partial refund.",
                "processRefundTransaction",
            ),
            (
                "Debug function compute_compound_interest_v2 in billing.",
                "compute_compound_interest_v2",
            ),
            (
                "Inspect file src/reconciliation_worker.py in billing service.",
                "src/reconciliation_worker.py",
            ),
            (
                "Update `UnseenPaymentGateway` in billing to log telemetry.",
                "UnseenPaymentGateway",
            ),
        ],
    )
    def test_absent_target_symbols_trigger_insufficient_context_naming_symbol(
        self,
        domain_registry: DomainRegistry,
        populated_repo: InMemoryRLSRepository,
        prompt: str,
        expected_missing: str,
    ) -> None:
        workflow = ScoutWorkflow(registry=domain_registry, repo=populated_repo)
        principal = PrincipalContext(
            user_id="auditor",
            role=RoleType.ENGINEER,
            allowed_scopes=["repo:enterprise/billing-service"],
        )

        result = workflow.run(prompt, principal, "trading_code_scout")

        assert isinstance(result, InsufficientContextDiagnostic), (
            f"Expected InsufficientContextDiagnostic, got {type(result)}"
        )
        assert result.paid_tokens_spent == 0
        assert any(expected_missing in str(item) for item in result.missing_items), (
            f"Expected {expected_missing} in missing_items: {result.missing_items}"
        )


# =============================================================================
# 2. Exact Unicode NFKC Substring Matching & Tampering Rejection
# =============================================================================


class TestExactUnicodeNFKCSubstrings:
    """Adversarially challenge zero-tolerance exact substring matching."""

    SOURCE_TEXT = (
        "Enterprise cluster nodes must enforce strict TLS 1.3 encryption on port 8443. "
        "Any unverified certificate authority must be rejected with error code SEC-007."
    )

    @pytest.fixture
    def verifier(self) -> ClaimVerifier:
        return ClaimVerifier()

    def test_exact_quote_passes(self, verifier: ClaimVerifier) -> None:
        quote = "Enterprise cluster nodes must enforce strict TLS 1.3 encryption on port 8443."
        res = verifier.verify_single_claim(quote, [self.SOURCE_TEXT])
        assert res.is_verified is True
        assert res.failure_code == ClaimFailureCode.EXACT_MATCH_PASSED

    def test_case_drift_rejected_zero_tolerance(self, verifier: ClaimVerifier) -> None:
        # 'Must' instead of 'must'
        tampered = "Enterprise cluster nodes Must enforce strict TLS 1.3 encryption on port 8443."
        res = verifier.verify_single_claim(tampered, [self.SOURCE_TEXT])
        assert res.is_verified is False
        assert res.failure_code == ClaimFailureCode.NOT_IN_SOURCE

    def test_punctuation_tamper_rejected_zero_tolerance(self, verifier: ClaimVerifier) -> None:
        # Semicolon instead of period
        tampered = "Enterprise cluster nodes must enforce strict TLS 1.3 encryption on port 8443;"
        res = verifier.verify_single_claim(tampered, [self.SOURCE_TEXT])
        assert res.is_verified is False
        assert res.failure_code == ClaimFailureCode.NOT_IN_SOURCE

    def test_numeric_tamper_rejected_zero_tolerance(self, verifier: ClaimVerifier) -> None:
        # 8444 instead of 8443
        tampered = "Enterprise cluster nodes must enforce strict TLS 1.3 encryption on port 8444."
        res = verifier.verify_single_claim(tampered, [self.SOURCE_TEXT])
        assert res.is_verified is False
        assert res.failure_code == ClaimFailureCode.NOT_IN_SOURCE

    def test_negation_tamper_rejected_zero_tolerance(self, verifier: ClaimVerifier) -> None:
        # 'must not' instead of 'must'
        tampered = (
            "Enterprise cluster nodes must not enforce strict TLS 1.3 encryption on port 8443."
        )
        res = verifier.verify_single_claim(tampered, [self.SOURCE_TEXT])
        assert res.is_verified is False
        assert res.failure_code == ClaimFailureCode.NOT_IN_SOURCE

    def test_nfkc_ligature_normalization(self, verifier: ClaimVerifier) -> None:
        # Ligature 'ﬁ' (U+FB01) normalizes to 'fi' under NFKC
        ligature_source = "Deﬁnition of security context."
        plain_claim = "Definition of security context."
        res = verifier.verify_single_claim(plain_claim, [ligature_source])
        assert res.is_verified is True
        assert res.failure_code == ClaimFailureCode.EXACT_MATCH_PASSED


# =============================================================================
# 3. Bidirectional Homoglyph Detection & Authentic International Unicode
# =============================================================================


class TestBidirectionalHomoglyphs:
    """Stress-test bidirectional homoglyph rejection and international Unicode acceptance."""

    @pytest.fixture
    def verifier(self) -> ClaimVerifier:
        return ClaimVerifier()

    # Direction A: Lookalikes injected into Latin text
    @pytest.mark.parametrize(
        "spoofed_token,word",
        [
            ("p\u0430ssword", "Latin with Cyrillic a"),
            ("t\u03bfken", "Latin with Greek omicron"),
            ("s\u0435rvice", "Latin with Cyrillic e"),
            ("r\u043e\u043et", "Latin with Cyrillic o"),
            ("c\u0435rt", "Latin with Cyrillic e"),
        ],
    )
    def test_lookalikes_in_latin_rejected(
        self, verifier: ClaimVerifier, spoofed_token: str, word: str
    ) -> None:
        claim = f"Validate user {spoofed_token} against auth provider."
        is_hg, reason = detect_homoglyph_spoofing(claim)
        assert is_hg is True, f"Failed detection for {word}: {claim}"
        assert "Mixed-script" in reason

        res = verifier.verify_single_claim(claim, ["Validate user password against auth provider."])
        assert res.is_verified is False
        assert res.failure_code == ClaimFailureCode.HOMOGLYPH_SPOOFING
        assert res.is_homoglyph_attack is True

    # Direction B: Latin character injected into Cyrillic (Russian) text
    @pytest.mark.parametrize(
        "spoofed_word,clean_word",
        [
            ("шлюзa", "шлюза"),  # Latin 'a' at end
            ("парольo", "пароль"),  # Latin 'o' at end
            ("пoдпись", "подпись"),  # Latin 'o' in middle
        ],
    )
    def test_latin_in_cyrillic_rejected_bidirectional(
        self, verifier: ClaimVerifier, spoofed_word: str, clean_word: str
    ) -> None:
        claim = f"Электронная {spoofed_word} проверена успешно."
        source = f"Электронная {clean_word} проверена успешно."
        is_hg, reason = detect_homoglyph_spoofing(claim)
        assert is_hg is True, f"Failed to detect Latin letter in Cyrillic: {claim}"
        assert "Mixed-script" in reason

        res = verifier.verify_single_claim(claim, [source])
        assert res.is_verified is False
        assert res.failure_code == ClaimFailureCode.HOMOGLYPH_SPOOFING
        assert res.is_homoglyph_attack is True

    # Direction C: Pure lookalike token in Latin context
    def test_pure_lookalike_token_in_latin_context_rejected(self, verifier: ClaimVerifier) -> None:
        # Cyrillic small letter a (\u0430) as a standalone variable name in Latin sentence
        claim = "Set variable \u0430 to zero on initialization."
        is_hg, reason = detect_homoglyph_spoofing(claim)
        assert is_hg is True
        assert "Pure homoglyph" in reason

        res = verifier.verify_single_claim(claim, ["Set variable a to zero on initialization."])
        assert res.is_verified is False
        assert res.failure_code == ClaimFailureCode.HOMOGLYPH_SPOOFING

    # Authentic International Unicode Acceptance
    @pytest.mark.parametrize(
        "lang,text",
        [
            ("Russian", "Параметры безопасности и аутентификация пользователя."),
            ("Ukrainian", "Перевірка автентифікації та цифрового підпису."),
            ("Greek Prose", "Το σύστημα ασφαλείας ελέγχει τα διαπιστευτήρια."),
            ("Greek Math", "L_total = \u03b1 * L_recon + \u03b2 * L_kl + \u03bb * L_reg"),
            ("German", "Überprüfung der Schlüssel und Zertifikate: Größe, Länge, Gültigkeit."),
            ("French", "Système d'accès sécurisé avec contrôle des identifiants et rôles."),
            ("Spanish", "Autenticación de contraseñas y validación de parámetros de diseño."),
            ("Japanese", "認証ゲートウェイと暗号化プロトコルの設定仕様書。"),
            ("Chinese", "企业上下文检索与确定性代码切片引擎。"),
            ("Korean", "결정적 컨텍스트 탐색 및 암호화 검증 시스템."),
        ],
    )
    def test_authentic_international_unicode_accepted(
        self, verifier: ClaimVerifier, lang: str, text: str
    ) -> None:
        is_hg, reason = detect_homoglyph_spoofing(text)
        assert is_hg is False, f"False positive homoglyph alert for authentic {lang}: {reason}"

        res = verifier.verify_single_claim(text, [text])
        assert res.is_verified is True
        assert res.failure_code == ClaimFailureCode.EXACT_MATCH_PASSED


# =============================================================================
# 4. Marginal RRF Threshold Score (0.6499 vs 0.6500)
# =============================================================================


class TestMarginalRRFThreshold:
    """Stress-test boundary conditions for marginal RRF score around 0.65."""

    @pytest.fixture
    def gate(self) -> PreFlightValidator:
        repo = InMemoryRLSRepository()
        return PreFlightValidator(repo=repo, min_rrf_score=0.65)

    def test_rrf_below_threshold_fails_strict(self, gate: PreFlightValidator) -> None:
        # 0.6499 must strictly fail Check 3
        passed, score = gate.check_rrf_threshold(normalized_rrf_score=0.6499)
        assert passed is False
        assert score == 0.6499

    def test_rrf_micro_below_threshold_fails_strict(self, gate: PreFlightValidator) -> None:
        # 0.649999 must strictly fail Check 3
        passed, score = gate.check_rrf_threshold(normalized_rrf_score=0.649999)
        assert passed is False
        assert score == 0.649999

    def test_rrf_at_threshold_passes(self, gate: PreFlightValidator) -> None:
        # 0.6500 must pass Check 3
        passed, score = gate.check_rrf_threshold(normalized_rrf_score=0.6500)
        assert passed is True
        assert score == 0.65

    def test_rrf_above_threshold_passes(self, gate: PreFlightValidator) -> None:
        # 0.6501 must pass Check 3
        passed, score = gate.check_rrf_threshold(normalized_rrf_score=0.6501)
        assert passed is True
        assert score == 0.6501

    def test_end_to_end_preflight_gate_abstains_on_0_6499(self, gate: PreFlightValidator) -> None:
        principal = PrincipalContext(
            user_id="auditor",
            role=RoleType.ENGINEER,
            allowed_scopes=["repo:test"],
        )
        slice_item = VerbatimCodeSlice(
            file_path="src/app.py",
            repo_name="test",
            git_commit_sha="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
            git_blob_sha="b1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
            start_line=1,
            end_line=2,
            verbatim_code_with_lines="1: x = 1\n2: y = 2\n",
            language="python",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            qualified_symbol="app.main",
            selection_reason="direct_hit",
            allowed_scopes=["repo:test"],
        )
        res = gate.evaluate_gate(
            code_slices=[slice_item],
            user_prompt="Refactor some code.",
            normalized_rrf_score=0.6499,
            principal=principal,
            domain_id="trading_code_scout",
        )
        assert isinstance(res, GateAbstentionResult)
        assert res.passed is False
        assert res.diagnostic.paid_tokens_spent == 0
        assert "0.6499 is below threshold 0.65" in res.report.bounce_back_feedback

    def test_rrf_marginal_with_cross_encoder_rescue(self, gate: PreFlightValidator) -> None:
        # 0.6499 RRF alone fails
        passed, _ = gate.check_rrf_threshold(normalized_rrf_score=0.6499, cross_encoder_score=0.59)
        assert passed is False

        # 0.6499 RRF rescued by cross_encoder >= 0.60
        passed, _ = gate.check_rrf_threshold(normalized_rrf_score=0.6499, cross_encoder_score=0.60)
        assert passed is True

        # 0.6500 RRF passes even if cross_encoder is low
        passed, _ = gate.check_rrf_threshold(normalized_rrf_score=0.6500, cross_encoder_score=0.10)
        assert passed is True


class TestDeepStressHomoglyphsAndMultiRepo:
    """Extra deep adversarial tests for uppercase lookalikes and multi-symbol prompts."""

    @pytest.fixture
    def verifier(self) -> ClaimVerifier:
        return ClaimVerifier()

    @pytest.mark.parametrize(
        "spoofed_upper,desc",
        [
            ("JWT_T\u039fKEN", "Greek capital omicron in JWT_TOKEN"),
            ("\u0410DMIN_GATEWAY", "Cyrillic capital A in ADMIN_GATEWAY"),
            ("S\u0415RVER_PORT", "Cyrillic capital E in SERVER_PORT"),
            ("R\u041e\u041eT_DIR", "Cyrillic capital O in ROOT_DIR"),
            ("AU\u0422H_HEADER", "Cyrillic capital T in AUTH_HEADER"),
        ],
    )
    def test_uppercase_homoglyph_lookalikes_rejected(
        self, verifier: ClaimVerifier, spoofed_upper: str, desc: str
    ) -> None:
        claim = f"Check header {spoofed_upper} on cluster gateway."
        is_hg, reason = detect_homoglyph_spoofing(claim)
        assert is_hg is True, f"Failed uppercase homoglyph detection for {desc}"
        assert "Mixed-script" in reason

        res = verifier.verify_single_claim(claim, ["Check header on cluster gateway."])
        assert res.is_verified is False
        assert res.failure_code == ClaimFailureCode.HOMOGLYPH_SPOOFING

    def test_greek_prose_with_latin_lookalike_rejected(self, verifier: ClaimVerifier) -> None:
        # Greek word 'βάση' with Latin 'a' substituted
        spoofed_greek = "Σφάλμα στη βaση δεδομένων."
        clean_greek = "Σφάλμα στη βάση δεδομένων."
        is_hg, reason = detect_homoglyph_spoofing(spoofed_greek)
        assert is_hg is True
        assert "Mixed-script" in reason

        res = verifier.verify_single_claim(spoofed_greek, [clean_greek])
        assert res.is_verified is False
        assert res.failure_code == ClaimFailureCode.HOMOGLYPH_SPOOFING

    def test_existing_symbol_with_absent_collaborator_abstains_naming_absent(
        self, domain_registry: DomainRegistry, populated_repo: InMemoryRLSRepository
    ) -> None:
        # ExistingBillingManager exists, but MissingAuditSink does not!
        prompt = "Refactor ExistingBillingManager to dispatch events to MissingAuditSink."
        workflow = ScoutWorkflow(registry=domain_registry, repo=populated_repo)
        principal = PrincipalContext(
            user_id="auditor",
            role=RoleType.ENGINEER,
            allowed_scopes=["repo:enterprise/billing-service"],
        )
        res = workflow.run(prompt, principal, "trading_code_scout")
        assert isinstance(res, InsufficientContextDiagnostic)
        assert res.paid_tokens_spent == 0
        assert any("MissingAuditSink" in str(item) for item in res.missing_items)
