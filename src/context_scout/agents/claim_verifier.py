"""Exact Unicode NFKC Normalized-Substring Claim Verifier (Milestone 4, F21).

Production claim and quote verification engine implementing:
1. Zero tolerance exact Unicode NFKC-normalized substring matching (no SequenceMatcher).
2. Complete rejection of single-word substitutions, omissions, and negations.
3. Bidirectional homoglyph detection: Cyrillic/Greek lookalikes in Latin text rejected 100%;
   authentic internationalized Unicode source code accepted.
4. Configurable ClaimVerificationPolicy (STRICT_ABSTAIN, PRUNE_AND_MARK_LOW_TRUST, MIN_COUNT).
5. Deterministic InsufficientContextDiagnostic emission with paid_tokens_spent=0 on abstention.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal

from context_scout.config import (
    ClaimVerificationPolicy,
    ClaimVerifierConfig,
)
from context_scout.schemas.contracts import (
    InsufficientContextDiagnostic,
)

if TYPE_CHECKING:
    from context_scout.db.repositories import DocChunkRecord
    from context_scout.ingestion.doc_parser import DocumentChunk

logger = logging.getLogger(__name__)

# Cyrillic lookalikes that visually collide with Latin characters
CYRILLIC_LOOKALIKES: Final[set[int]] = {
    0x0430,
    0x0441,
    0x0435,
    0x0456,
    0x0458,
    0x043E,
    0x0440,
    0x0455,
    0x0445,
    0x0443,
    0x0410,
    0x0412,
    0x0421,
    0x0415,
    0x041D,
    0x0406,
    0x0408,
    0x041A,
    0x041C,
    0x041E,
    0x0420,
    0x0422,
    0x0425,
    0x04AE,
}

# Greek lookalikes that visually collide with Latin characters
GREEK_LOOKALIKES: Final[set[int]] = {
    0x03BF,
    0x03BD,
    0x03C1,
    0x03BA,
    0x03C4,
    0x03C7,
    0x0391,
    0x0392,
    0x0395,
    0x0397,
    0x0399,
    0x039A,
    0x039C,
    0x039D,
    0x039F,
    0x03A1,
    0x03A4,
    0x03A7,
    0x03A5,
    0x0396,
}

ALL_LOOKALIKES: Final[set[int]] = CYRILLIC_LOOKALIKES | GREEK_LOOKALIKES
_NFKC_HOMOGLYPH_RE: Final[re.Pattern[str]] = re.compile(
    r"[\uFF01-\uFF5E\u2160-\u217F\u2460-\u24FF\U0001D400-\U0001D7FF]"
)

CandidateSources = (
    Sequence[tuple[str, str]]
    | dict[str, str]
    | Sequence["DocChunkRecord"]
    | Sequence["DocumentChunk"]
    | Sequence[str]
)


def normalize_claim_text(text: str) -> str:
    """Applies Unicode NFKC normalization and collapses contiguous whitespace.

    Zero-tolerance: Does not lowercase (case-sensitive by default), does not strip punctuation,
    does not allow token reordering or omission.
    """
    if not text:
        return ""
    nfkc = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", nfkc).strip()


def detect_homoglyph_spoofing(text: str) -> tuple[bool, str]:
    """Detects Cyrillic and Greek homoglyph forgery in Latin context.

    Rules:
    1. Mixed-script tokens (e.g. Latin word containing Cyrillic or Greek lookalikes) are rejected.
    2. In predominantly Latin context, isolated tokens composed purely of lookalikes are rejected.
    3. Authentic internationalized text (pure Cyrillic Russian, pure Greek, Greek mathematical
       symbols like alpha/beta/lambda, German umlauts, French accents, CJK) is accepted.
    """
    nf_match = _NFKC_HOMOGLYPH_RE.search(text)
    if nf_match:
        ch = nf_match.group(0)
        return (
            True,
            f"NFKC compatibility homoglyph character detected: {ch!r} (U+{ord(ch):04X})",
        )

    tokens = re.findall(r"\b\w+\b", text)
    if not tokens:
        return False, ""

    mixed_tokens: list[str] = []
    latin_tokens = 0
    non_latin_tokens = 0

    for w in tokens:
        scripts: set[str] = set()
        for ch in w:
            if ch.isalpha():
                name = unicodedata.name(ch, "")
                if "LATIN" in name:
                    scripts.add("LATIN")
                elif "CYRILLIC" in name:
                    scripts.add("CYRILLIC")
                elif "GREEK" in name:
                    scripts.add("GREEK")
        if "LATIN" in scripts and ("CYRILLIC" in scripts or "GREEK" in scripts):
            mixed_tokens.append(w)
        elif "LATIN" in scripts:
            latin_tokens += 1
        elif "CYRILLIC" in scripts or "GREEK" in scripts:
            non_latin_tokens += 1

    if mixed_tokens:
        return True, f"Mixed-script homoglyph token detected: {mixed_tokens[0]!r}"

    # In predominantly Latin context, flag tokens composed exclusively of lookalikes
    if latin_tokens > 0 and latin_tokens >= non_latin_tokens:
        for w in tokens:
            alphas = [ch for ch in w if ch.isalpha()]
            if not alphas:
                continue
            scripts = {
                "LATIN"
                if "LATIN" in unicodedata.name(ch, "")
                else "CYRILLIC"
                if "CYRILLIC" in unicodedata.name(ch, "")
                else "GREEK"
                if "GREEK" in unicodedata.name(ch, "")
                else "OTHER"
                for ch in alphas
            }
            is_non_latin = ("CYRILLIC" in scripts or "GREEK" in scripts) and "LATIN" not in scripts
            if is_non_latin and all(ord(ch) in ALL_LOOKALIKES for ch in alphas):
                return True, f"Pure homoglyph lookalike token in Latin context: {w!r}"

    return False, ""


class ClaimFailureCode(StrEnum):
    """Categorized root causes for claim verification rejections."""

    EXACT_MATCH_PASSED = "EXACT_MATCH_PASSED"
    NOT_IN_SOURCE = "NOT_IN_SOURCE"
    HOMOGLYPH_SPOOFING = "HOMOGLYPH_SPOOFING"
    EMPTY_CLAIM = "EMPTY_CLAIM"
    EMPTY_SOURCE = "EMPTY_SOURCE"


@dataclass(frozen=True)
class ClaimVerificationRecord:
    """Detailed audit trail for a single claim or quote evaluation."""

    claim_text: str
    is_verified: bool
    failure_code: ClaimFailureCode
    failure_reason: str | None = None
    matched_source_id: str | None = None
    matched_segment: str | None = None
    is_homoglyph_attack: bool = False


@dataclass(frozen=True)
class ClaimVerificationResult:
    """Comprehensive evaluation payload from ClaimVerifier."""

    passed: bool
    policy_applied: ClaimVerificationPolicy
    total_claims: int
    verified_count: int
    pruned_count: int
    verified_claims: list[str]
    pruned_claims: list[str]
    records: list[ClaimVerificationRecord]
    diagnostic: InsufficientContextDiagnostic | None = None
    reason: str = ""
    diagnostic_reason: str | None = None
    paid_tokens_spent: Literal[0] = 0


class ClaimVerifier:
    """Exact Unicode NFKC normalized-substring claim verification engine (M4, F21).

    Enforces:
    1. Zero tolerance exact Unicode NFKC-normalized substring matching (SequenceMatcher removed).
    2. Zero tolerance for single-word substitutions, omissions, and negations.
    3. Bidirectional homoglyph checks (Cyrillic/Greek lookalikes rejected; authentic Unicode ok).
    4. Configurable ClaimVerificationPolicy.
    5. $0 paid token spend on abstention.
    """

    def __init__(self, config: ClaimVerifierConfig | None = None) -> None:
        self.config: Final[ClaimVerifierConfig] = config or ClaimVerifierConfig()

    def normalize_text(self, text: str) -> str:
        """Applies Unicode NFKC normalization and collapses contiguous whitespace."""
        return normalize_claim_text(text)

    def verify_single_claim(
        self,
        claim: str,
        sources: CandidateSources,
    ) -> ClaimVerificationRecord:
        """Verifies a single claim against candidate sources using exact NFKC substring matching."""
        if not claim.strip():
            return ClaimVerificationRecord(
                claim_text=claim,
                is_verified=False,
                failure_code=ClaimFailureCode.EMPTY_CLAIM,
                failure_reason="Claim text is empty or whitespace",
            )

        if self.config.reject_homoglyphs:
            is_hg, hg_reason = detect_homoglyph_spoofing(claim)
            if is_hg:
                return ClaimVerificationRecord(
                    claim_text=claim,
                    is_verified=False,
                    failure_code=ClaimFailureCode.HOMOGLYPH_SPOOFING,
                    failure_reason=f"Homoglyph spoofing detected: {hg_reason}",
                    is_homoglyph_attack=True,
                )

        norm_claim = self.normalize_text(claim)
        if not norm_claim:
            return ClaimVerificationRecord(
                claim_text=claim,
                is_verified=False,
                failure_code=ClaimFailureCode.EMPTY_CLAIM,
                failure_reason="Claim text is empty or whitespace",
            )

        # Normalize candidate sources
        normalized_sources: list[tuple[str, str]] = []
        if isinstance(sources, dict):
            for s_id, s_text in sources.items():
                normalized_sources.append((str(s_id), self.normalize_text(str(s_text))))
        else:
            for idx, item in enumerate(sources):
                if isinstance(item, tuple) and len(item) == 2:
                    normalized_sources.append((str(item[0]), self.normalize_text(str(item[1]))))
                elif hasattr(item, "verbatim_content"):
                    cid = str(getattr(item, "chunk_id", f"chunk_{idx}"))
                    normalized_sources.append(
                        (cid, self.normalize_text(str(item.verbatim_content)))
                    )
                elif isinstance(item, str):
                    normalized_sources.append((f"source_{idx}", self.normalize_text(item)))

        if not normalized_sources:
            return ClaimVerificationRecord(
                claim_text=claim,
                is_verified=False,
                failure_code=ClaimFailureCode.EMPTY_SOURCE,
                failure_reason="No source text provided for verification",
            )

        # Exact normalized-substring check (Zero tolerance)
        for s_id, s_norm in normalized_sources:
            if norm_claim in s_norm:
                return ClaimVerificationRecord(
                    claim_text=claim,
                    is_verified=True,
                    failure_code=ClaimFailureCode.EXACT_MATCH_PASSED,
                    matched_source_id=s_id,
                    matched_segment=norm_claim,
                )

        return ClaimVerificationRecord(
            claim_text=claim,
            is_verified=False,
            failure_code=ClaimFailureCode.NOT_IN_SOURCE,
            failure_reason="Claim is not an exact normalized-substring of any retrieved source",
        )

    def verify_claims(
        self,
        claims: Sequence[str],
        sources: CandidateSources,
        *,
        trace_id: str = "trace_claim_verifier",
        domain_id: str = "enterprise_scout",
    ) -> ClaimVerificationResult:
        """Verifies a collection of claims under the configured ClaimVerificationPolicy."""
        records: list[ClaimVerificationRecord] = []
        verified_claims: list[str] = []
        pruned_claims: list[str] = []

        for claim in claims:
            rec = self.verify_single_claim(claim, sources)
            records.append(rec)
            if rec.is_verified:
                verified_claims.append(claim)
            else:
                pruned_claims.append(claim)

        total = len(claims)
        v_count = len(verified_claims)
        p_count = len(pruned_claims)

        policy = self.config.policy
        passed = False
        diag_reason: str | None = None
        reasons_list: list[str] = []

        if policy == ClaimVerificationPolicy.STRICT_ABSTAIN_ON_ANY_UNVERIFIED:
            if p_count == 0 and v_count == total and total > 0:
                passed = True
                reasons_list.append(f"100% of {total} claims verified verbatim (STRICT_ABSTAIN)")
            else:
                passed = False
                unverified_details = [
                    f"'{r.claim_text}' ({r.failure_reason})" for r in records if not r.is_verified
                ]
                diag_reason = (
                    f"Claim Verification Abstention ($0 Spend): {p_count} of {total} claims "
                    f"unverified under STRICT_ABSTAIN: {'; '.join(unverified_details[:3])}"
                )
                reasons_list.append(diag_reason)

        elif policy == ClaimVerificationPolicy.PRUNE_AND_MARK_LOW_TRUST:
            if v_count >= self.config.min_verified_claims:
                passed = True
                reasons_list.append(
                    f"{v_count} claims verified ({p_count} pruned under PRUNE_AND_MARK_LOW_TRUST)"
                )
            else:
                passed = False
                diag_reason = (
                    f"Claim Verification Abstention ($0 Spend): only {v_count} verified claims "
                    f"remain after pruning (min required: {self.config.min_verified_claims})"
                )
                reasons_list.append(diag_reason)

        elif policy == ClaimVerificationPolicy.MIN_VERIFIED_COUNT:
            if v_count >= self.config.min_verified_claims:
                passed = True
                reasons_list.append(
                    f"{v_count} claims verified >= min_count={self.config.min_verified_claims}"
                )
            else:
                passed = False
                diag_reason = (
                    f"Claim Verification Abstention ($0 Spend): verified {v_count} claims "
                    f"is below min_verified_claims={self.config.min_verified_claims}"
                )
                reasons_list.append(diag_reason)

        diagnostic = None
        if not passed:
            missing_syms = [
                f"unverified_claim:{r.failure_code}:{r.claim_text[:40]}"
                for r in records
                if not r.is_verified
            ]
            diagnostic = InsufficientContextDiagnostic(
                trace_id=trace_id,
                domain_id=domain_id,
                reason=diag_reason or "Claim Verification FAILED",
                missing_symbols=missing_syms,
                partial_candidate_files=[],
                paid_tokens_spent=0,
            )

        return ClaimVerificationResult(
            passed=passed,
            policy_applied=policy,
            total_claims=total,
            verified_count=v_count,
            pruned_count=p_count,
            verified_claims=verified_claims,
            pruned_claims=pruned_claims,
            records=records,
            diagnostic=diagnostic,
            reason=" | ".join(reasons_list),
            diagnostic_reason=diag_reason,
            paid_tokens_spent=0,
        )


__all__ = [
    "ALL_LOOKALIKES",
    "CYRILLIC_LOOKALIKES",
    "GREEK_LOOKALIKES",
    "ClaimFailureCode",
    "ClaimVerificationPolicy",
    "ClaimVerificationRecord",
    "ClaimVerificationResult",
    "ClaimVerifier",
    "ClaimVerifierConfig",
    "detect_homoglyph_spoofing",
    "normalize_claim_text",
]
