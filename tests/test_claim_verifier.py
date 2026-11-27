"""Tests for Exact Unicode NFKC Normalized-Substring Claim Verifier (Milestone 4).

Validates:
1. Exact substring matching with zero tolerance (100% precision).
2. Complete rejection of substitutions, negations, omissions, and numeric drift.
3. Homoglyph spoofing detection (Cyrillic and Greek lookalikes rejected;
   authentic international accepted).
4. Configurable ClaimVerificationPolicy behaviors.
5. Zero-spend diagnostic emission on unverified abstention.
"""

from context_scout.agents.claim_verifier import (
    ClaimFailureCode,
    ClaimVerifier,
    detect_homoglyph_spoofing,
    normalize_claim_text,
)
from context_scout.config import ClaimVerificationPolicy, ClaimVerifierConfig


def test_normalize_claim_text_collapses_whitespace() -> None:
    raw = "The  authentication\n    timeout\tis  set\t\tto  500 ms."
    norm = normalize_claim_text(raw)
    assert norm == "The authentication timeout is set to 500 ms."


def test_exact_substring_match_success() -> None:
    source = (
        "Under NO circumstances should the gateway allow unauthenticated fallback mode. "
        "The authentication timeout is set to 500 ms. "
        "Order flow imbalance exhibits mean reversion over sub-second horizons."
    )
    verifier = ClaimVerifier()
    rec = verifier.verify_single_claim("The authentication timeout is set to 500 ms.", [source])
    assert rec.is_verified is True
    assert rec.failure_code == ClaimFailureCode.EXACT_MATCH_PASSED
    assert rec.matched_segment == "The authentication timeout is set to 500 ms."


def test_whitespace_variation_passes_under_nfkc() -> None:
    source = "The authentication timeout is set to 500 ms."
    verifier = ClaimVerifier()
    rec = verifier.verify_single_claim(
        "The authentication\n    timeout is set\tto 500 ms.", [source]
    )
    assert rec.is_verified is True
    assert rec.failure_code == ClaimFailureCode.EXACT_MATCH_PASSED


def test_single_word_substitution_rejected() -> None:
    source = "Under NO circumstances should the gateway allow unauthenticated fallback mode."
    verifier = ClaimVerifier()
    # 'allow' -> 'deny'
    rec = verifier.verify_single_claim(
        "Under NO circumstances should the gateway deny unauthenticated fallback mode.",
        [source],
    )
    assert rec.is_verified is False
    assert rec.failure_code == ClaimFailureCode.NOT_IN_SOURCE


def test_negation_insertion_rejected() -> None:
    source = "Under NO circumstances should the gateway allow unauthenticated fallback mode."
    verifier = ClaimVerifier()
    # 'allow' -> 'not allow'
    rec = verifier.verify_single_claim(
        "Under NO circumstances should the gateway not allow unauthenticated fallback mode.",
        [source],
    )
    assert rec.is_verified is False
    assert rec.failure_code == ClaimFailureCode.NOT_IN_SOURCE


def test_critical_token_omission_rejected() -> None:
    source = "Under NO circumstances should the gateway allow unauthenticated fallback mode."
    verifier = ClaimVerifier()
    # 'NO' dropped
    rec = verifier.verify_single_claim(
        "Under circumstances should the gateway allow unauthenticated fallback mode.",
        [source],
    )
    assert rec.is_verified is False
    assert rec.failure_code == ClaimFailureCode.NOT_IN_SOURCE


def test_numeric_alteration_rejected() -> None:
    source = "The authentication timeout is set to 500 ms."
    verifier = ClaimVerifier()
    # '500' -> '5000'
    rec = verifier.verify_single_claim(
        "The authentication timeout is set to 5000 ms.",
        [source],
    )
    assert rec.is_verified is False
    assert rec.failure_code == ClaimFailureCode.NOT_IN_SOURCE


def test_homoglyph_spoofing_detection() -> None:
    # Cyrillic small letter a (\u0430) injected into 'allow'
    spoofed = "Under NO circumstances should the gateway \u0430llow fallback."
    is_hg, reason = detect_homoglyph_spoofing(spoofed)
    assert is_hg is True
    assert "Mixed-script" in reason

    source = "Under NO circumstances should the gateway allow fallback."
    verifier = ClaimVerifier()
    rec = verifier.verify_single_claim(spoofed, [source])
    assert rec.is_verified is False
    assert rec.failure_code == ClaimFailureCode.HOMOGLYPH_SPOOFING
    assert rec.is_homoglyph_attack is True


def test_authentic_international_unicode_accepted() -> None:
    verifier = ClaimVerifier()

    # Authentic Russian
    rus_source = "Параметр аутентификации обязателен для всех пользователей."
    rec_rus = verifier.verify_single_claim(
        "Параметр аутентификации обязателен",
        [rus_source],
    )
    assert rec_rus.is_verified is True

    # Authentic Greek math
    math_source = "Total loss calculation: loss = \u03b1 * loss_primary + \u03b2 * loss_auxiliary."
    rec_math = verifier.verify_single_claim(
        "loss = \u03b1 * loss_primary + \u03b2 * loss_auxiliary",
        [math_source],
    )
    assert rec_math.is_verified is True

    # Authentic German umlauts
    ger_source = "Überprüfung der Rückgabewerte und Berechtigungen der Schnittstelle."
    rec_ger = verifier.verify_single_claim(
        "Überprüfung der Rückgabewerte",
        [ger_source],
    )
    assert rec_ger.is_verified is True


def test_strict_abstain_policy() -> None:
    source = (
        "Under NO circumstances should the gateway allow unauthenticated fallback mode. "
        "The authentication timeout is set to 500 ms."
    )
    verifier = ClaimVerifier(
        ClaimVerifierConfig(policy=ClaimVerificationPolicy.STRICT_ABSTAIN_ON_ANY_UNVERIFIED)
    )

    claims = [
        "The authentication timeout is set to 500 ms.",
        "Under NO circumstances should the gateway deny unauthenticated fallback mode.",
    ]
    result = verifier.verify_claims(claims, [source])
    assert result.passed is False
    assert result.verified_count == 1
    assert result.pruned_count == 1
    assert result.paid_tokens_spent == 0
    assert "Claim Verification Abstention" in (result.diagnostic_reason or "")


def test_prune_and_mark_policy() -> None:
    source = (
        "Under NO circumstances should the gateway allow unauthenticated fallback mode. "
        "The authentication timeout is set to 500 ms."
    )
    verifier = ClaimVerifier(
        ClaimVerifierConfig(
            policy=ClaimVerificationPolicy.PRUNE_AND_MARK_LOW_TRUST,
            min_verified_claims=1,
        )
    )

    claims = [
        "The authentication timeout is set to 500 ms.",
        "Under NO circumstances should the gateway deny unauthenticated fallback mode.",
    ]
    result = verifier.verify_claims(claims, [source])
    assert result.passed is True
    assert result.verified_count == 1
    assert result.pruned_count == 1
    assert len(result.verified_claims) == 1
    assert result.verified_claims[0] == "The authentication timeout is set to 500 ms."


def test_min_verified_count_policy() -> None:
    source = "The authentication timeout is set to 500 ms."
    verifier = ClaimVerifier(
        ClaimVerifierConfig(
            policy=ClaimVerificationPolicy.MIN_VERIFIED_COUNT,
            min_verified_claims=2,
        )
    )

    claims = [
        "The authentication timeout is set to 500 ms.",
        "Invalid non-existent claim.",
    ]
    result = verifier.verify_claims(claims, [source])
    assert result.passed is False
    assert result.verified_count == 1
    assert result.paid_tokens_spent == 0


def test_empty_claim_and_empty_source() -> None:
    verifier = ClaimVerifier()
    rec_empty_claim = verifier.verify_single_claim("   ", ["source"])
    assert rec_empty_claim.is_verified is False
    assert rec_empty_claim.failure_code == ClaimFailureCode.EMPTY_CLAIM

    rec_empty_src = verifier.verify_single_claim("Valid claim", [])
    assert rec_empty_src.is_verified is False
    assert rec_empty_src.failure_code == ClaimFailureCode.EMPTY_SOURCE


def test_homoglyph_spoofing_boundary_cases() -> None:
    """Verifies punctuation-only, pure-script, and pure-homoglyph lookalike boundaries."""
    assert detect_homoglyph_spoofing("!!!") == (False, "")
    assert detect_homoglyph_spoofing("") == (False, "")

    hg_greek_mix, _ = detect_homoglyph_spoofing("Verify s\u03bfme")
    assert hg_greek_mix is True

    hg_pure_cyr, reason_cyr = detect_homoglyph_spoofing("Valid Latin \u0430\u0441")
    assert hg_pure_cyr is True
    assert "Pure homoglyph" in reason_cyr

    hg_pure_gr, _ = detect_homoglyph_spoofing("Valid Latin \u03bf\u03bd")
    assert hg_pure_gr is True

    assert (
        detect_homoglyph_spoofing("\u041f\u0440\u0438\u0432\u0435\u0442 123 \u043c\u0438\u0440")[0]
        is False
    )
    assert detect_homoglyph_spoofing("\u0391\u03b2\u03b3 \u03b4\u03b5\u03b6")[0] is False


def test_claim_verifier_rejects_nfkc_compatibility_homoglyphs_before_normalization() -> None:
    """PR-4 (ECS-011): Fullwidth/mathematical/enclosed NFKC homoglyphs must be rejected."""
    verifier = ClaimVerifier(ClaimVerifierConfig(reject_homoglyphs=True))
    source = "The API timeout is set to 500 ms."

    for spoofed_claim in (
        "The \uff21PI timeout is set to 500 ms.",  # fullwidth capital A
        "The \U0001d400PI timeout is set to 500 ms.",  # mathematical bold capital A
        "Version \u2161 timeout is set to 500 ms.",  # compatibility Roman numeral II
    ):
        rec = verifier.verify_single_claim(spoofed_claim, [source])
        assert rec.is_verified is False, f"Accepted NFKC spoofed claim: {spoofed_claim!r}"
        assert rec.failure_code == ClaimFailureCode.HOMOGLYPH_SPOOFING
        assert rec.is_homoglyph_attack is True
