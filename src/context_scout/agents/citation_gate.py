"""Domain-Agnostic Document Citation Gate (VAL-02, F18).

Decoupled citation verification engine validating 100% of document citations
across any enterprise source (Confluence, Jira, technical specs, arXiv, local docs)
with strict provenance checks, verbatim quote grounding (NFKC + SequenceMatcher >=0.98),
staleness detection, and deterministic InsufficientContextDiagnostic production.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer
from context_scout.db.repositories import DocChunkRecord
from context_scout.schemas.contracts import (
    DocCitationSpan,
    InsufficientContextDiagnostic,
)

if TYPE_CHECKING:
    from context_scout.ingestion.doc_parser import DocumentChunk

logger = logging.getLogger(__name__)

# Standard Jira project-ticket key pattern (e.g. ENG-4092, CORE-101, SEC-12)
JIRA_KEY_REGEX: Final[re.Pattern[str]] = re.compile(r"^[A-Z][A-Z0-9]+-[0-9]+$")


class CitationFailureCode(StrEnum):
    """Categorized root causes for citation gate rejections."""

    MISSING_CITATIONS = "MISSING_CITATIONS"
    CHUNK_NOT_IN_CORPUS = "CHUNK_NOT_IN_CORPUS"
    HALLUCINATED_QUOTE = "HALLUCINATED_QUOTE"
    TRUNCATED_QUOTE = "TRUNCATED_QUOTE"
    QUOTE_MISMATCH = "QUOTE_MISMATCH"
    MISSING_PROVENANCE = "MISSING_PROVENANCE"
    INVALID_DOC_ID = "INVALID_DOC_ID"
    INVALID_TICKET_KEY = "INVALID_TICKET_KEY"
    STALE_SUPERSEDED_VERSION = "STALE_SUPERSEDED_VERSION"
    STALE_TICKET_STATUS = "STALE_TICKET_STATUS"


@dataclass(frozen=True)
class CitationVerificationDetail:
    """Detailed audit trail for a single citation evaluation."""

    citation: DocCitationSpan
    is_valid: bool
    overlap_ratio: float
    matched_chunk_id: str | None
    error_code: CitationFailureCode | None = None
    error_reason: str | None = None
    is_stale: bool = False
    stale_reason: str | None = None


@dataclass(frozen=True)
class DocumentCitationGateResult:
    """Comprehensive evaluation payload from DocumentCitationGate."""

    passed: bool
    total_citations: int
    verified_count: int
    failed_count: int
    details: list[CitationVerificationDetail]
    failed_citations: list[tuple[DocCitationSpan, str]]
    diagnostic: InsufficientContextDiagnostic | None = None
    reason: str = ""


class DocumentCitationGate:
    """Domain-agnostic citation verification gate (VAL-02, F18).

    Validates:
    1. Citation presence (when documentation grounding is required)
    2. Enterprise provenance (page_id + version for Confluence; ticket_key for Jira)
    3. Verbatim quote grounding via DeterministicSlicer (NFKC normalization, delimiter check)
    4. Staleness / obsolescence flags
    5. Deterministic InsufficientContextDiagnostic generation on failure
    """

    DEFAULT_MIN_OVERLAP_RATIO: Final[float] = 0.98

    def __init__(
        self,
        min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
        require_citations_default: bool = False,
        strict_freshness: bool = True,
    ) -> None:
        self.min_overlap_ratio: Final[float] = min_overlap_ratio
        self.require_citations_default: Final[bool] = require_citations_default
        self.strict_freshness: Final[bool] = strict_freshness
        self.slicer = DeterministicSlicer()

    def _validate_confluence_provenance(
        self,
        citation: DocCitationSpan,
        chunk_metadata: dict[str, Any] | None = None,
        chunk_doc_id: str | None = None,
    ) -> tuple[bool, CitationFailureCode | None, str | None]:
        """Validates exact Confluence page ID and version."""
        page_id = citation.page_id or (citation.doc_id if citation.doc_id.isdigit() else None)
        if not page_id:
            return (
                False,
                CitationFailureCode.MISSING_PROVENANCE,
                f"Confluence citation '{citation.doc_title}' is missing a valid numeric page_id",
            )

        if not citation.version or citation.version.strip() == "":
            return (
                False,
                CitationFailureCode.MISSING_PROVENANCE,
                (
                    f"Confluence citation '{citation.doc_title}' (page_id={page_id}) "
                    "is missing version"
                ),
            )

        # Normalize and validate version string (strip leading ^[vV]\.?, parse positive integer)
        raw_version = str(citation.version).strip()
        norm_version_str = re.sub(r"^[vV]\.?", "", raw_version).strip()
        try:
            if "." in norm_version_str:
                f_val = float(norm_version_str)
                if not f_val.is_integer() or f_val <= 0:
                    raise ValueError(f"Non-integer or non-positive version: {norm_version_str}")
                c_ver = int(f_val)
            else:
                c_ver = int(norm_version_str)
            if c_ver <= 0:
                raise ValueError(f"Non-positive version: {c_ver}")
        except (ValueError, TypeError):
            return (
                False,
                CitationFailureCode.MISSING_PROVENANCE,
                (
                    f"Confluence page {page_id} citation contains invalid/non-numeric version "
                    f"'{citation.version}' (expected positive integer version, e.g. 1, 2, v1)"
                ),
            )

        # Cross-document check: verify page_id against chunk metadata and doc_id
        if chunk_metadata:
            expected_page_id = str(chunk_metadata.get("page_id") or "")
            if expected_page_id and str(page_id) != expected_page_id:
                return (
                    False,
                    CitationFailureCode.INVALID_DOC_ID,
                    f"Citation page_id '{page_id}' does not match chunk page_id "
                    f"'{expected_page_id}'",
                )

        if chunk_doc_id:
            clean_chunk_doc = chunk_doc_id.replace("confluence:", "")
            clean_cit_doc = citation.doc_id.replace("confluence:", "")
            if clean_cit_doc != clean_chunk_doc and clean_chunk_doc.isdigit():
                return (
                    False,
                    CitationFailureCode.INVALID_DOC_ID,
                    f"Citation doc_id '{citation.doc_id}' does not match chunk doc_id "
                    f"'{chunk_doc_id}'",
                )

        if chunk_metadata:
            latest_version = (
                chunk_metadata.get("version")
                or chunk_metadata.get("page_version")
                or chunk_metadata.get("version_str")
            )
            if latest_version is not None:
                norm_latest_str = re.sub(r"^[vV]\.?", "", str(latest_version)).strip()
                try:
                    if "." in norm_latest_str:
                        f_latest = float(norm_latest_str)
                        l_ver = int(f_latest) if f_latest.is_integer() else None
                    else:
                        l_ver = int(norm_latest_str)
                except (ValueError, TypeError):
                    l_ver = None

                if l_ver is not None:
                    if c_ver < l_ver:
                        return (
                            False,
                            CitationFailureCode.STALE_SUPERSEDED_VERSION,
                            f"Confluence page {page_id} version {c_ver} is superseded by "
                            f"latest indexed version {l_ver}",
                        )
                    if c_ver > l_ver:
                        return (
                            False,
                            CitationFailureCode.STALE_SUPERSEDED_VERSION,
                            f"Confluence page {page_id} version {c_ver} exceeds latest "
                            f"indexed version {l_ver} (unindexed version)",
                        )

        return True, None, None

    def _validate_jira_provenance(
        self,
        citation: DocCitationSpan,
        chunk_metadata: dict[str, Any] | None = None,
        chunk_doc_id: str | None = None,
    ) -> tuple[bool, CitationFailureCode | None, str | None]:
        """Validates exact Jira ticket key pattern and active status."""
        ticket_key = citation.ticket_key or (
            citation.doc_id if "-" in citation.doc_id else None
        )
        if not ticket_key or not JIRA_KEY_REGEX.match(ticket_key):
            return (
                False,
                CitationFailureCode.INVALID_TICKET_KEY,
                f"Jira citation contains invalid ticket key '{ticket_key}' "
                "(expected format e.g. ENG-4092)",
            )

        if chunk_metadata:
            # Reject cross-source spoofing: chunk from Confluence cannot be cited as Jira ticket
            if "page_id" in chunk_metadata and "issue_key" not in chunk_metadata:
                return (
                    False,
                    CitationFailureCode.INVALID_DOC_ID,
                    f"Cross-source citation spoofing: Confluence chunk cited as Jira ticket "
                    f"'{ticket_key}'",
                )

            expected_ticket_key = str(
                chunk_metadata.get("issue_key") or chunk_metadata.get("ticket_key") or ""
            )
            if expected_ticket_key and ticket_key != expected_ticket_key:
                return (
                    False,
                    CitationFailureCode.INVALID_TICKET_KEY,
                    f"Citation ticket_key '{ticket_key}' does not match chunk issue_key "
                    f"'{expected_ticket_key}'",
                )

        if chunk_doc_id:
            clean_chunk_doc = chunk_doc_id.replace("jira:", "")
            clean_cit_doc = citation.doc_id.replace("jira:", "")
            if clean_cit_doc != clean_chunk_doc and "-" in clean_chunk_doc:
                return (
                    False,
                    CitationFailureCode.INVALID_DOC_ID,
                    f"Citation doc_id '{citation.doc_id}' does not match chunk doc_id "
                    f"'{chunk_doc_id}'",
                )

        if chunk_metadata and self.strict_freshness:
            raw_status = str(
                chunk_metadata.get("status") or chunk_metadata.get("ticket_status") or ""
            ).strip().lower()
            norm_status = raw_status.replace(" ", "").replace("'", "").replace("-", "")

            raw_resolution = str(chunk_metadata.get("resolution") or "").strip().lower()
            norm_resolution = raw_resolution.replace(" ", "").replace("'", "").replace("-", "")

            jira_stale_statuses = {
                "closed", "archived", "resolved (obsolete)", "obsolete",
                "rejected", "cancelled", "won't fix", "wont fix", "wontfix",
                "won't do", "wont do", "wontdo", "abandoned",
            }
            jira_stale_resolutions = {
                "won't fix", "wont fix", "wontfix", "duplicate", "incomplete", "cannot reproduce",
                "rejected", "cancelled", "declined", "obsolete",
                "won't do", "wont do", "wontdo", "abandoned",
            }

            is_stale_status = (
                raw_status in jira_stale_statuses or norm_status in jira_stale_statuses
            )
            is_stale_resolution = (
                raw_resolution in jira_stale_resolutions
                or norm_resolution in jira_stale_resolutions
            )

            if is_stale_status or is_stale_resolution:
                flag = raw_status if is_stale_status else f"resolution: {raw_resolution}"
                return (
                    False,
                    CitationFailureCode.STALE_TICKET_STATUS,
                    f"Jira ticket {ticket_key} has obsolete/terminal status '{flag}' "
                    "under strict freshness",
                )

        return True, None, None

    def _validate_generic_provenance(
        self,
        citation: DocCitationSpan,
    ) -> tuple[bool, CitationFailureCode | None, str | None]:
        """Validates doc_id, version, source_uri, and section_heading."""
        if not citation.doc_id or not citation.doc_id.strip():
            return (
                False,
                CitationFailureCode.MISSING_PROVENANCE,
                "Citation is missing doc_id",
            )
        if not citation.source_uri or not citation.source_uri.strip():
            return (
                False,
                CitationFailureCode.MISSING_PROVENANCE,
                f"Citation '{citation.doc_id}' is missing source_uri",
            )
        if not citation.section_heading or not citation.section_heading.strip():
            return (
                False,
                CitationFailureCode.MISSING_PROVENANCE,
                f"Citation '{citation.doc_id}' is missing section_heading",
            )
        return True, None, None

    def verify_citations(
        self,
        citations: Sequence[DocCitationSpan],
        corpus_chunks: list[DocChunkRecord] | list[DocumentChunk] | dict[str, str] | None = None,
        *,
        domain_id: str = "enterprise_scout",
        trace_id: str = "trace_citation_gate",
        require_citations: bool | None = None,
    ) -> DocumentCitationGateResult:
        """Verifies 100% of document citations across any enterprise domain."""
        must_require = (
            require_citations
            if require_citations is not None
            else self.require_citations_default
        )

        # 1. Citation Presence Check
        if not citations:
            if must_require:
                diag = InsufficientContextDiagnostic(
                    trace_id=trace_id,
                    domain_id=domain_id,
                    reason=(
                        "Citation Gate Abstention: Zero document citations provided "
                        "for documentation-grounded task"
                    ),
                    missing_symbols=["supporting_doc_citations"],
                    partial_candidate_files=[],
                    paid_tokens_spent=0,
                )
                return DocumentCitationGateResult(
                    passed=False,
                    total_citations=0,
                    verified_count=0,
                    failed_count=0,
                    details=[],
                    failed_citations=[],
                    diagnostic=diag,
                    reason=diag.reason,
                )
            return DocumentCitationGateResult(
                passed=True,
                total_citations=0,
                verified_count=0,
                failed_count=0,
                details=[],
                failed_citations=[],
                diagnostic=None,
                reason="Zero citations evaluated (citations optional)",
            )

        # 2. Index corpus chunks for O(1) retrieval
        chunk_by_id: dict[str, tuple[str, str, dict[str, Any], bool]] = {}
        chunks_by_doc_id: dict[str, list[tuple[str, str, dict[str, Any], bool]]] = defaultdict(list)

        if corpus_chunks is not None:
            if isinstance(corpus_chunks, dict):
                for cid, text in corpus_chunks.items():
                    chunk_by_id[str(cid)] = ("", str(text), {}, False)
            else:
                for c in corpus_chunks:
                    cid = str(c.chunk_id)
                    text = c.verbatim_content
                    meta = dict(getattr(c, "metadata_json", {}) or {})
                    v_str = getattr(c, "version_str", None)
                    if v_str is not None:
                        if "version" not in meta:
                            meta["version"] = v_str
                        if "version_str" not in meta:
                            meta["version_str"] = v_str
                    stale_reason_attr = getattr(c, "stale_reason", None)
                    if stale_reason_attr and "stale_reason" not in meta:
                        meta["stale_reason"] = stale_reason_attr
                    source_type_attr = getattr(c, "source_type", None)
                    if source_type_attr and "source_type" not in meta:
                        meta["source_type"] = source_type_attr
                    stale = bool(getattr(c, "is_stale", False) or meta.get("is_stale", False))
                    chunk_by_id[cid] = (c.doc_id, text, meta, stale)
                    chunks_by_doc_id[c.doc_id].append((cid, text, meta, stale))
                    clean_doc = c.doc_id.replace("confluence:", "").replace("jira:", "")
                    if clean_doc != c.doc_id:
                        chunks_by_doc_id[clean_doc].append((cid, text, meta, stale))

        # 3. Evaluate each citation
        details: list[CitationVerificationDetail] = []
        failed_citations: list[tuple[DocCitationSpan, str]] = []
        verified_count = 0

        for cit in citations:
            is_confluence = "confluence" in cit.source_uri.lower() or cit.page_id is not None
            is_jira = "jira" in cit.source_uri.lower() or cit.ticket_key is not None

            # Find matching chunk candidates
            candidates: list[tuple[str, str, str, dict[str, Any], bool]] = []
            exact_match = chunk_by_id.get(str(cit.chunk_id))
            if exact_match:
                candidates.append((
                    str(cit.chunk_id),
                    exact_match[0],
                    exact_match[1],
                    exact_match[2],
                    exact_match[3],
                ))

            for cand in chunks_by_doc_id.get(cit.doc_id, []):
                if cand[0] != str(cit.chunk_id):
                    candidates.append((cand[0], cit.doc_id, cand[1], cand[2], cand[3]))

            if not candidates:
                reason = (
                    f"Chunk ID '{cit.chunk_id}' (doc_id='{cit.doc_id}') not in corpus "
                    "(not found in corpus)"
                )
                details.append(
                    CitationVerificationDetail(
                        citation=cit,
                        is_valid=False,
                        overlap_ratio=0.0,
                        matched_chunk_id=None,
                        error_code=CitationFailureCode.CHUNK_NOT_IN_CORPUS,
                        error_reason=reason,
                    )
                )
                failed_citations.append((cit, reason))
                continue

            # Provenance Check against candidate metadata
            matched_chunk_id, matched_doc_id, _chunk_text, chunk_meta, is_stale = candidates[0]
            prov_valid = True
            prov_code: CitationFailureCode | None = None
            prov_reason: str | None = None

            if is_confluence:
                prov_valid, prov_code, prov_reason = self._validate_confluence_provenance(
                    cit, chunk_meta, matched_doc_id
                )
            elif is_jira:
                prov_valid, prov_code, prov_reason = self._validate_jira_provenance(
                    cit, chunk_meta, matched_doc_id
                )
            else:
                prov_valid, prov_code, prov_reason = self._validate_generic_provenance(cit)

            if not prov_valid:
                assert prov_reason is not None
                details.append(
                    CitationVerificationDetail(
                        citation=cit,
                        is_valid=False,
                        overlap_ratio=0.0,
                        matched_chunk_id=matched_chunk_id,
                        error_code=prov_code,
                        error_reason=prov_reason,
                    )
                )
                failed_citations.append((cit, prov_reason))
                continue

            # Verbatim Quote Grounding Check (VAL-02)
            v_res = None
            for c_id, _c_doc_id, c_text, _c_meta, c_stale in candidates:
                v_res = self.slicer.verify_doc_citation(
                    citation=cit,
                    chunk_content=c_text,
                    min_overlap_ratio=self.min_overlap_ratio,
                )
                if v_res.is_valid:
                    matched_chunk_id = c_id
                    is_stale = c_stale
                    break

            assert v_res is not None
            if not v_res.is_valid:
                err_code = (
                    CitationFailureCode.TRUNCATED_QUOTE
                    if v_res.error_code == "TRUNCATED_QUOTE"
                    else CitationFailureCode.HALLUCINATED_QUOTE
                )
                reason = (
                    f"Quote from chunk '{cit.chunk_id}' failed verbatim verification: "
                    f"Quote failed NFKC overlap check: overlap={v_res.token_overlap_ratio:.4f} "
                    f"(min={self.min_overlap_ratio:.2f}), reason={v_res.error_reason}"
                )
                details.append(
                    CitationVerificationDetail(
                        citation=cit,
                        is_valid=False,
                        overlap_ratio=v_res.token_overlap_ratio,
                        matched_chunk_id=matched_chunk_id,
                        error_code=err_code,
                        error_reason=reason,
                    )
                )
                failed_citations.append((cit, reason))
                continue

            # Staleness check
            stale_reason = None
            if is_stale:
                stale_reason = chunk_meta.get("stale_reason", "Marked stale in document catalog")
                if self.strict_freshness:
                    reason = f"Citation from chunk '{cit.chunk_id}' rejected: {stale_reason}"
                    is_jira_citation = (
                        is_jira
                        or cit.ticket_key is not None
                        or chunk_meta.get("source_type") == "jira"
                        or "issue_key" in chunk_meta
                        or "ticket_key" in chunk_meta
                        or "jira:" in cit.doc_id
                        or "jira" in cit.source_uri.lower()
                    )
                    error_code = (
                        CitationFailureCode.STALE_TICKET_STATUS
                        if is_jira_citation
                        else CitationFailureCode.STALE_SUPERSEDED_VERSION
                    )
                    details.append(
                        CitationVerificationDetail(
                            citation=cit,
                            is_valid=False,
                            overlap_ratio=v_res.token_overlap_ratio,
                            matched_chunk_id=matched_chunk_id,
                            error_code=error_code,
                            error_reason=reason,
                            is_stale=True,
                            stale_reason=stale_reason,
                        )
                    )
                    failed_citations.append((cit, reason))
                    continue

            details.append(
                CitationVerificationDetail(
                    citation=cit,
                    is_valid=True,
                    overlap_ratio=v_res.token_overlap_ratio,
                    matched_chunk_id=matched_chunk_id,
                    is_stale=is_stale,
                    stale_reason=stale_reason,
                )
            )
            verified_count += 1

        passed = len(failed_citations) == 0 and verified_count == len(citations)

        diagnostic = None
        reasons_list: list[str] = []
        if not passed:
            missing_items = [
                f"{cit.chunk_id}:{reason}" for cit, reason in failed_citations
            ]
            reasons_list.append(
                f"{len(failed_citations)} of {len(citations)} citations failed verification"
            )
            diag_reason = (
                f"Citation Gate Validation FAILED ($0 Abstention): "
                f"{'; '.join(r for _, r in failed_citations)}"
            )
            diagnostic = InsufficientContextDiagnostic(
                trace_id=trace_id,
                domain_id=domain_id,
                reason=diag_reason,
                missing_symbols=missing_items,
                partial_candidate_files=[cit.doc_id for cit, _ in failed_citations],
                paid_tokens_spent=0,
            )
        else:
            reasons_list.append(
                f"100% of {verified_count} document citations verified verbatim (VAL-02)"
            )

        return DocumentCitationGateResult(
            passed=passed,
            total_citations=len(citations),
            verified_count=verified_count,
            failed_count=len(failed_citations),
            details=details,
            failed_citations=failed_citations,
            diagnostic=diagnostic,
            reason=" | ".join(reasons_list),
        )


__all__ = [
    "JIRA_KEY_REGEX",
    "CitationFailureCode",
    "CitationVerificationDetail",
    "DocumentCitationGate",
    "DocumentCitationGateResult",
]
