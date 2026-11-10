"""Unit and integration tests for Domain-Agnostic DocumentCitationGate (F18, VAL-02).

Verifies citation gating across:
1. Confluence spaces, pages, and versions.
2. Jira projects, ticket keys, and statuses.
3. Technical specs, arXiv papers, and markdown documentation.
4. Deterministic InsufficientContextDiagnostic production on abstention.
5. Composition with FinanceDomainPlugin.
"""

from __future__ import annotations

import uuid

from context_scout.agents.citation_gate import (
    CitationFailureCode,
    DocumentCitationGate,
)
from context_scout.ingestion.doc_parser import DocumentChunk
from context_scout.plugins.domains.finance import (
    FinanceDomainPlugin,
)
from context_scout.schemas.contracts import (
    DocCitationSpan,
    FrictionVerdict,
    MarketRegime,
    QuantStrategyBrief,
)


def _make_chunk(
    doc_id: str,
    title: str,
    content: str,
    source_uri: str,
    section: str = "Overview",
    metadata: dict | None = None,
    is_stale: bool = False,
) -> DocumentChunk:
    meta = dict(metadata or {})
    if is_stale:
        meta["is_stale"] = True
    return DocumentChunk(
        chunk_id=uuid.uuid4(),
        doc_id=doc_id,
        doc_title=title,
        source_uri=source_uri,
        market_scope=MarketRegime.GLOBAL_MULTI,
        section_heading=section,
        verbatim_content=content,
        formulas_latex=[],
        tables_markdown=[],
        token_count_approx=len(content.split()),
        metadata_json=meta,
    )


# =============================================================================
# 1. Confluence Verification Tests
# =============================================================================


def test_confluence_valid_citation() -> None:
    gate = DocumentCitationGate()
    quote = (
        "All microservices must implement OAuth 2.0 PKCE flow with 15-minute token rotation."
    )
    chunk = _make_chunk(
        doc_id="10928374",
        title="Authentication Architecture RFC-82",
        content=f"Engineering Standards. {quote} Refresh tokens must be stored in secure storage.",
        source_uri="https://wiki.corp.internal/spaces/ENG/pages/10928374",
        section="Security Requirements",
        metadata={"version": 3, "space_key": "ENG"},
    )
    citation = DocCitationSpan(
        chunk_id=str(chunk.chunk_id),
        doc_id=chunk.doc_id,
        doc_title=chunk.doc_title,
        source_uri=chunk.source_uri,
        section_heading=chunk.section_heading,
        verbatim_quote=quote,
        version="3",
        page_id="10928374",
    )
    res = gate.verify_citations([citation], [chunk])
    assert res.passed is True
    assert res.verified_count == 1
    assert res.failed_count == 0
    assert res.details[0].overlap_ratio == 1.0


def test_confluence_missing_page_id_fails() -> None:
    gate = DocumentCitationGate()
    quote = "Database migrations must run idempotently."
    chunk = _make_chunk(
        doc_id="non_numeric_doc",
        title="DB Runbook",
        content=quote,
        source_uri="https://wiki.corp.internal/confluence/pages/view",
        section="Database Runbook",
    )
    citation = DocCitationSpan(
        chunk_id=str(chunk.chunk_id),
        doc_id="non_numeric_doc",
        doc_title="DB Runbook",
        source_uri=chunk.source_uri,
        section_heading=chunk.section_heading,
        verbatim_quote=quote,
        version="1.0",
        page_id=None,
    )
    res = gate.verify_citations([citation], [chunk])
    assert res.passed is False
    assert res.details[0].error_code == CitationFailureCode.MISSING_PROVENANCE


def test_confluence_superseded_version_fails() -> None:
    gate = DocumentCitationGate()
    quote = "Service uses PostgreSQL 14 with PgBouncer."
    chunk = _make_chunk(
        doc_id="10928374",
        title="Infra Specification",
        content=quote,
        source_uri="https://wiki.corp.internal/confluence/10928374",
        metadata={"version": 5},
    )
    # Citation references outdated version 2
    citation = DocCitationSpan(
        chunk_id=str(chunk.chunk_id),
        doc_id=chunk.doc_id,
        doc_title=chunk.doc_title,
        source_uri=chunk.source_uri,
        section_heading=chunk.section_heading,
        verbatim_quote=quote,
        version="2",
        page_id="10928374",
    )
    res = gate.verify_citations([citation], [chunk])
    assert res.passed is False
    assert res.details[0].error_code == CitationFailureCode.STALE_SUPERSEDED_VERSION


def test_confluence_hallucinated_quote_fails() -> None:
    gate = DocumentCitationGate()
    chunk = _make_chunk(
        doc_id="10928374",
        title="Infra RFC",
        content="Cache TTL is strictly 300 seconds for session tokens.",
        source_uri="https://wiki.corp.internal/confluence/10928374",
        metadata={"version": 1},
    )
    citation = DocCitationSpan(
        chunk_id=str(chunk.chunk_id),
        doc_id=chunk.doc_id,
        doc_title=chunk.doc_title,
        source_uri=chunk.source_uri,
        section_heading=chunk.section_heading,
        # Hallucinated number
        verbatim_quote="Cache TTL is strictly 86400 seconds for session tokens.",
        version="1",
        page_id="10928374",
    )
    res = gate.verify_citations([citation], [chunk])
    assert res.passed is False
    assert res.details[0].error_code == CitationFailureCode.HALLUCINATED_QUOTE


# =============================================================================
# 2. Jira Verification Tests
# =============================================================================


def test_jira_valid_ticket_citation() -> None:
    gate = DocumentCitationGate()
    quote = "Fix NullReferenceException in PaymentGateway by null-checking client secret."
    chunk = _make_chunk(
        doc_id="ENG-4092",
        title="Checkout Error Resolution",
        content=f"Resolution details: {quote}",
        source_uri="https://jira.corp.internal/browse/ENG-4092",
        section="Resolution Summary",
        metadata={"ticket_status": "Done", "project_key": "ENG"},
    )
    citation = DocCitationSpan(
        chunk_id=str(chunk.chunk_id),
        doc_id=chunk.doc_id,
        doc_title=chunk.doc_title,
        source_uri=chunk.source_uri,
        section_heading=chunk.section_heading,
        verbatim_quote=quote,
        ticket_key="ENG-4092",
    )
    res = gate.verify_citations([citation], [chunk])
    assert res.passed is True
    assert res.verified_count == 1


def test_jira_invalid_ticket_key_format_fails() -> None:
    gate = DocumentCitationGate()
    quote = "Fix memory leak in background worker."
    chunk = _make_chunk(
        doc_id="invalid_key",
        title="Memory Leak",
        content=quote,
        source_uri="https://jira.corp.internal/browse/invalid_key",
    )
    citation = DocCitationSpan(
        chunk_id=str(chunk.chunk_id),
        doc_id="invalid_key",
        doc_title=chunk.doc_title,
        source_uri=chunk.source_uri,
        section_heading=chunk.section_heading,
        verbatim_quote=quote,
        ticket_key="invalid_key",
    )
    res = gate.verify_citations([citation], [chunk])
    assert res.passed is False
    assert res.details[0].error_code == CitationFailureCode.INVALID_TICKET_KEY


def test_jira_obsolete_status_under_strict_freshness() -> None:
    gate = DocumentCitationGate(strict_freshness=True)
    quote = "Deprecated legacy endpoint /api/v1/auth remains active until migration."
    chunk = _make_chunk(
        doc_id="CORE-101",
        title="Legacy Deprecation",
        content=quote,
        source_uri="https://jira.corp.internal/browse/CORE-101",
        metadata={"ticket_status": "Closed", "project_key": "CORE"},
    )
    citation = DocCitationSpan(
        chunk_id=str(chunk.chunk_id),
        doc_id=chunk.doc_id,
        doc_title=chunk.doc_title,
        source_uri=chunk.source_uri,
        section_heading=chunk.section_heading,
        verbatim_quote=quote,
        ticket_key="CORE-101",
    )
    res = gate.verify_citations([citation], [chunk])
    assert res.passed is False
    assert res.details[0].error_code == CitationFailureCode.STALE_TICKET_STATUS


# =============================================================================
# 3. Technical Specs / arXiv Tests
# =============================================================================


def test_arxiv_valid_citation() -> None:
    gate = DocumentCitationGate()
    quote = "Limit order book imbalance signal provides predictable alpha over 500ms horizons."
    chunk = _make_chunk(
        doc_id="arxiv:2403.01182",
        title="High-Frequency Microstructure",
        content=f"Abstract: {quote} Backtested across liquid equities.",
        source_uri="https://arxiv.org/abs/2403.01182",
        section="Abstract",
    )
    citation = DocCitationSpan(
        chunk_id=str(chunk.chunk_id),
        doc_id=chunk.doc_id,
        doc_title=chunk.doc_title,
        source_uri=chunk.source_uri,
        section_heading=chunk.section_heading,
        verbatim_quote=quote,
    )
    res = gate.verify_citations([citation], [chunk])
    assert res.passed is True


def test_arxiv_truncated_quote_with_unclosed_delimiter_fails() -> None:
    gate = DocumentCitationGate()
    chunk = _make_chunk(
        doc_id="arxiv:2401.05432",
        title="Formula Spec",
        content="The pricing model is P(t) = exp(alpha * (x + y)) where alpha > 0.",
        source_uri="https://arxiv.org/abs/2401.05432",
    )
    citation = DocCitationSpan(
        chunk_id=str(chunk.chunk_id),
        doc_id=chunk.doc_id,
        doc_title=chunk.doc_title,
        source_uri=chunk.source_uri,
        section_heading=chunk.section_heading,
        verbatim_quote="pricing model is P(t) = exp(alpha * (x + y",  # Unclosed parenthesis
    )
    res = gate.verify_citations([citation], [chunk])
    assert res.passed is False
    assert res.details[0].error_code == CitationFailureCode.TRUNCATED_QUOTE


def test_chunk_not_in_corpus_fails_with_diagnostic() -> None:
    gate = DocumentCitationGate()
    citation = DocCitationSpan(
        chunk_id=str(uuid.uuid4()),
        doc_id="arxiv:9999.99999",
        doc_title="Phantom Paper",
        source_uri="https://arxiv.org/abs/9999.99999",
        section_heading="Phantom Section",
        verbatim_quote="This quote does not exist in the corpus index.",
    )
    res = gate.verify_citations([citation], [])
    assert res.passed is False
    assert res.details[0].error_code == CitationFailureCode.CHUNK_NOT_IN_CORPUS
    assert res.diagnostic is not None
    assert res.diagnostic.paid_tokens_spent == 0


def test_empty_citations_when_required_abstains() -> None:
    gate = DocumentCitationGate(require_citations_default=True)
    res = gate.verify_citations([], [], require_citations=True)
    assert res.passed is False
    assert res.diagnostic is not None
    assert res.diagnostic.paid_tokens_spent == 0
    assert "supporting_doc_citations" in res.diagnostic.missing_symbols


# =============================================================================
# 4. Finance Domain Plug-in Composition Tests
# =============================================================================


def test_finance_plugin_composition_realistic() -> None:
    plugin = FinanceDomainPlugin()
    quote = "Securities Transaction Tax requires 15.0 bps friction on Indian delivery equity."
    chunk = _make_chunk(
        doc_id="arxiv:2403.01182",
        title="NSE STT Market Dynamics",
        content=f"Statutory rules: {quote}",
        source_uri="https://arxiv.org/abs/2403.01182",
    )
    brief = QuantStrategyBrief(
        strategy_name="NSE Delivery Momentum",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        is_intraday_futures=False,
        mathematical_signal_latex="S_t = \\sum R_{t-k}",
        entry_exit_rules=["Rebalance monthly"],
        sample_period_years="2018-2023",
        reported_sharpe_ratio=1.5,
        reported_max_drawdown_pct=-15.0,
        transaction_cost_bps_assumed=15.0,
        friction_verdict=FrictionVerdict.REALISTIC_FRICTION,
        market_specific_caveats=["Includes 15 bps statutory STT"],
        known_failure_regimes=["Sideways market"],
        verified_citations=[
            DocCitationSpan(
                chunk_id=str(chunk.chunk_id),
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                source_uri=chunk.source_uri,
                section_heading=chunk.section_heading,
                verbatim_quote=quote,
            )
        ],
    )
    res = plugin.evaluate_strategy_brief(brief, [chunk])
    assert res.passed is True
    assert res.friction_verdict == FrictionVerdict.REALISTIC_FRICTION
    assert res.citations_valid is True
    assert res.effective_friction_floor_bps == 15.0


def test_finance_plugin_unrealistic_friction_fails_while_citation_passes() -> None:
    plugin = FinanceDomainPlugin()
    quote = "Execution cost is approximately 8 bps on Indian delivery equity."
    chunk = _make_chunk(
        doc_id="arxiv:2403.01182",
        title="NSE STT Market Dynamics",
        content=quote,
        source_uri="https://arxiv.org/abs/2403.01182",
    )
    brief = QuantStrategyBrief(
        strategy_name="NSE Sub-Floor Strategy",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        is_intraday_futures=False,
        mathematical_signal_latex="S_t = 1.0",
        entry_exit_rules=["Rebalance monthly"],
        sample_period_years="2018-2023",
        reported_sharpe_ratio=1.5,
        reported_max_drawdown_pct=-15.0,
        transaction_cost_bps_assumed=8.0,  # Below 15.0 bps floor
        friction_verdict=FrictionVerdict.UNVERIFIED_FRICTION_WARNING,
        market_specific_caveats=["Omits statutory STT"],
        known_failure_regimes=["Sideways market"],
        verified_citations=[
            DocCitationSpan(
                chunk_id=str(chunk.chunk_id),
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                source_uri=chunk.source_uri,
                section_heading=chunk.section_heading,
                verbatim_quote=quote,
            )
        ],
    )
    res = plugin.evaluate_strategy_brief(brief, [chunk])
    assert res.passed is False
    assert res.citations_valid is True  # Quote was verbatim
    assert res.friction_verdict == FrictionVerdict.UNVERIFIED_FRICTION_WARNING
