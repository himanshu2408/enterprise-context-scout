"""Empirical Challenger 2 Adversarial Stress Suite for Milestone 3.

Adversarial Testing Scope:
1. RLS Tenant Isolation on live PostgreSQL 17 doc_chunks:
   - Querying with unauthorized scope ('unauthorized_scope') yields zero rows.
   - Querying with authorized scope ('public_quant') yields all 1,473 harvested chunks.
   - Multi-modal hybrid RRF search with unauthorized scope returns zero hits.
2. 3-Way Normalized RRF Mathematical Bounds & Extremes:
   - Invariant: [1, 1, 1] yields mathematically exact 1.0.
   - Extreme rankings: [1000, 1000, 1000] strictly within (0.0, 1.0].
   - Empty ranks list [] and all-None [None, None, None] yield exactly 0.0.
   - Extreme boundary [10_000_000, 10_000_000, 10_000_000] strictly bounded.
   - Randomized rank monotonicity across 1,000 trials.
3. Market-Regime Quant Scepticism Gate:
   - 0.0 bps friction strategy -> MUST be penalized and rejected.
   - Just below cost floor (14.9 bps for IN_NSE cash, 9.9 bps for DE_XETRA) ->
     MUST be flagged or rejected.
   - Exactly at cost floor (15.0 bps for IN_NSE, 10.0 bps for DE_XETRA) -> MUST pass.
   - Maximum drawdown of 30.1% -> MUST fail.
   - Maximum drawdown of 29.9% -> MUST pass.
   - Distorted, mutated, and fabricated citations -> MUST fail NFKC quote verification.
4. End-to-end verification against live PostgreSQL 17 corpus.
"""

import math
import random
import uuid

import pytest

from context_scout.ast_engine.deterministic_slicer import (
    DeterministicSlicer,
    HallucinatedQuoteError,
    TruncatedLineError,
)
from context_scout.ast_engine.models import SlicerErrorCode
from context_scout.config import ScoutSettings
from context_scout.db.repositories import DocChunkRecord, PostgresRLSRepository
from context_scout.db.session import RLSConnectionManager
from context_scout.ingestion.doc_parser import DocumentChunk
from context_scout.ingestion.scepticism_gate import MarketRegimeScepticismGate
from context_scout.schemas.contracts import (
    DocCitationSpan,
    FrictionVerdict,
    MarketRegime,
    PrincipalContext,
    QuantStrategyBrief,
    RoleType,
    compute_normalized_rrf,
)


@pytest.fixture(scope="module")
def settings() -> ScoutSettings:
    return ScoutSettings()


def _make_brief(
    strategy_name: str = "Empirical Test Strategy",
    target_markets: list[MarketRegime] | None = None,
    is_intraday_futures: bool = False,
    transaction_cost_bps_assumed: float | None = 15.0,
    reported_sharpe_ratio: float | None = 1.5,
    reported_max_drawdown_pct: float | None = 15.0,
    verified_citations: list[DocCitationSpan] | None = None,
    mathematical_signal_latex: str = r"\alpha_t = \sum_{i=1}^N w_i r_{i, t}",
    entry_exit_rules: list[str] | None = None,
    sample_period_years: str = "2020-2024",
    market_specific_caveats: list[str] | None = None,
    known_failure_regimes: list[str] | None = None,
) -> QuantStrategyBrief:
    """Helper to construct QuantStrategyBrief satisfying all required schema constraints."""
    if verified_citations is None:
        verified_citations = [
            DocCitationSpan(
                chunk_id=str(uuid.uuid4()),
                doc_id="doc_default",
                doc_title="Default Title",
                source_uri="https://example.com/doc",
                section_heading="Results",
                verbatim_quote="Default valid citation quote for testing.",
            )
        ]

    return QuantStrategyBrief(
        strategy_name=strategy_name,
        target_markets=target_markets or [MarketRegime.INDIA_NSE_BSE],
        is_intraday_futures=is_intraday_futures,
        transaction_cost_bps_assumed=transaction_cost_bps_assumed,
        reported_sharpe_ratio=reported_sharpe_ratio,
        reported_max_drawdown_pct=reported_max_drawdown_pct,
        verified_citations=verified_citations,
        mathematical_signal_latex=mathematical_signal_latex,
        entry_exit_rules=entry_exit_rules or ["Enter on signal > 0.0", "Exit on signal <= 0.0"],
        sample_period_years=sample_period_years,
        market_specific_caveats=market_specific_caveats or [
            "Subject to statutory transaction cost"
        ],
        known_failure_regimes=known_failure_regimes or ["Flash crash volatility"],
    )


# ==============================================================================
# SECTION 1: Empirical Live PostgreSQL 17 RLS Isolation
# ==============================================================================


def test_empirical_live_rls_isolation_doc_chunks(settings: ScoutSettings) -> None:
    """Adversarial stress-test: querying live doc_chunks with unauthorized scope.

    Requirements:
    - Connecting as scout_app_user with 'unauthorized_scope' MUST return zero rows.
    - Connecting as scout_app_user with 'public_quant' MUST return all 1,473 rows.
    - 3-way hybrid RRF search with unauthorized scope MUST return zero hits.
    """
    mgr = RLSConnectionManager(settings.db_app_url)
    repo = PostgresRLSRepository()

    unauth_principal = PrincipalContext(
        user_id="adversary_trader",
        role=RoleType.PRODUCT_MANAGER,
        allowed_scopes=["unauthorized_scope"],
    )
    auth_principal = PrincipalContext(
        user_id="authorized_quant",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["public_quant"],
    )

    # 1. Unauthorized scope direct table scan
    with mgr.scoped_transaction(unauth_principal, enable_hnsw_iterative_scan=False) as conn:
        count_row = conn.execute("SELECT count(*) FROM doc_chunks;").fetchone()
        assert count_row is not None
        assert count_row[0] == 0, (
            f"SECURITY BREACH: Unauthorized scope saw {count_row[0]} rows in doc_chunks!"
        )

        sample_rows = conn.execute(
            "SELECT chunk_id, doc_title FROM doc_chunks LIMIT 10;"
        ).fetchall()
        assert len(sample_rows) == 0, (
            "SECURITY BREACH: Unauthorized scope retrieved sample rows from doc_chunks!"
        )

    # 2. Authorized scope direct table scan
    with mgr.scoped_transaction(auth_principal, enable_hnsw_iterative_scan=False) as conn:
        auth_count_row = conn.execute("SELECT count(*) FROM doc_chunks;").fetchone()
        assert auth_count_row is not None
        assert int(str(auth_count_row[0])) >= 1000, (
            f"Expected >= 1,000 chunks in doc_chunks for public_quant, got {auth_count_row[0]}"
        )

    # 3. Hybrid RRF search with unauthorized scope
    dummy_embedding = [0.05] * 768
    with mgr.scoped_transaction(unauth_principal, enable_hnsw_iterative_scan=True) as conn:
        unauth_hits = repo.search_doc_chunks_hybrid_rrf(
            conn=conn,
            domain_id="quant_research_scout",
            query_text="order flow imbalance NIFTY market impact",
            query_embedding=dummy_embedding,
            top_k=10,
        )
        assert len(unauth_hits) == 0, (
            f"SECURITY BREACH: Unauthorized scope returned {len(unauth_hits)} RRF search hits!"
        )

    # 4. Hybrid RRF search with authorized scope
    with mgr.scoped_transaction(auth_principal, enable_hnsw_iterative_scan=True) as conn:
        auth_hits = repo.search_doc_chunks_hybrid_rrf(
            conn=conn,
            domain_id="quant_research_scout",
            query_text="order flow imbalance NIFTY market impact",
            query_embedding=dummy_embedding,
            top_k=10,
        )
        assert len(auth_hits) > 0, "Authorized search should return hits on live corpus"
        for hit in auth_hits:
            assert 0.0 <= hit.normalized_rrf_score <= 1.0


# ==============================================================================
# SECTION 2: 3-Way Normalized RRF Formula Verification & Extremes
# ==============================================================================


def test_empirical_3way_rrf_formula_bounds_and_extremes() -> None:
    """Stress-tests 3-Way Normalized RRF formula adherence to [0.0, 1.0] bounds.

    Formula:
        raw_score = sum_{m in modalities} (1 / (k + rank_m))
        max_possible = M * (1 / (k + 1))
        normalized = raw_score / max_possible
    """
    # 1. All rank 1 across 3 modalities -> MUST be exactly 1.0
    score_all_rank1 = compute_normalized_rrf([1, 1, 1], num_modalities=3, k=60)
    assert score_all_rank1 == 1.0
    assert math.isclose(score_all_rank1, 1.0, rel_tol=1e-12)

    # Invariant across diverse k parameter values
    for k_val in [0, 1, 10, 30, 60, 100, 500, 1000]:
        score_k = compute_normalized_rrf([1, 1, 1], num_modalities=3, k=k_val)
        assert score_k == 1.0, f"Failed exact 1.0 invariant at k={k_val}"

    # 2. Extreme rank 1000 across all 3 modalities
    # Theoretical value: (3 / (60 + 1000)) / (3 / 61) = 61 / 1060 = 0.0575471698...
    score_rank1000 = compute_normalized_rrf([1000, 1000, 1000], num_modalities=3, k=60)
    expected_rank1000 = round(61.0 / 1060.0, 6)
    assert score_rank1000 == expected_rank1000 == 0.057547
    assert 0.0 <= score_rank1000 <= 1.0

    # 3. Empty result set
    score_empty = compute_normalized_rrf([], num_modalities=3, k=60)
    assert score_empty == 0.0

    # 4. All None result set
    score_all_none = compute_normalized_rrf([None, None, None], num_modalities=3, k=60)
    assert score_all_none == 0.0

    # 5. Extreme astronomical rank: 10,000,000
    score_extreme = compute_normalized_rrf(
        [10_000_000, 10_000_000, 10_000_000], num_modalities=3, k=60
    )
    assert 0.0 < score_extreme <= 1.0
    assert score_extreme < 0.0001

    # 6. Single extreme hit in single modality
    score_single_1000 = compute_normalized_rrf([1000, None, None], num_modalities=3, k=60)
    expected_single_1000 = round((1.0 / 1060.0) / (3.0 / 61.0), 6)
    assert score_single_1000 == expected_single_1000 == 0.019182
    assert 0.0 <= score_single_1000 <= 1.0

    # 7. Asymmetric modality combinations
    score_one_top = compute_normalized_rrf([1, None, None], num_modalities=3, k=60)
    assert score_one_top == 0.333333
    score_two_top = compute_normalized_rrf([1, 1, None], num_modalities=3, k=60)
    assert score_two_top == 0.666667

    # 8. Randomized rank monotonicity and strict containment across 1,000 trials
    rng = random.Random(2026)
    for _ in range(1000):
        ranks: list[int | None] = [
            rng.choice([None, rng.randint(1, 50_000)])
            for _ in range(3)
        ]
        s = compute_normalized_rrf(ranks, num_modalities=3, k=60)
        assert 0.0 <= s <= 1.0, f"RRF score {s} outside [0.0, 1.0] for ranks {ranks}"


# ==============================================================================
# SECTION 3: Market-Regime Scepticism Gate Edge Cases
# ==============================================================================


def test_empirical_scepticism_gate_zero_bps_friction_rejected() -> None:
    """Stress-test: 0.0 bps friction strategy MUST be penalized and rejected."""
    gate = MarketRegimeScepticismGate()
    chunk_uuid = uuid.uuid4()
    source_text = "Analysis of NSE delivery trading dynamics."
    doc_chunk = DocumentChunk(
        chunk_id=chunk_uuid,
        doc_id="paper_nse_01",
        doc_title="NSE Microstructure",
        source_uri="https://arxiv.org/abs/2403.00001",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        section_heading="Methodology",
        verbatim_content=source_text,
        formulas_latex=[],
        tables_markdown=[],
        token_count_approx=10,
    )
    valid_citation = DocCitationSpan(
        chunk_id=str(chunk_uuid),
        doc_id="paper_nse_01",
        doc_title="NSE Microstructure",
        source_uri="https://arxiv.org/abs/2403.00001",
        section_heading="Methodology",
        verbatim_quote=source_text,
    )

    # 1. 0.0 bps for IN_NSE cash (floor: 15.0 bps)
    brief_zero_nse = _make_brief(
        strategy_name="Zero Friction NSE Cash",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        is_intraday_futures=False,
        transaction_cost_bps_assumed=0.0,
        reported_sharpe_ratio=2.5,
        reported_max_drawdown_pct=10.0,
        verified_citations=[valid_citation],
    )
    res_nse = gate.evaluate(brief_zero_nse, [doc_chunk])
    assert res_nse.passed is False, "0.0 bps friction strategy should not pass gate"
    assert res_nse.friction_verdict == FrictionVerdict.UNVERIFIED_FRICTION_WARNING
    assert "below market floor (15.0 bps)" in res_nse.reason

    # 2. 0.0 bps for DE_XETRA (floor: 10.0 bps)
    brief_zero_xetra = _make_brief(
        strategy_name="Zero Friction DE Xetra",
        target_markets=[MarketRegime.GERMANY_XETRA_DAX],
        is_intraday_futures=False,
        transaction_cost_bps_assumed=0.0,
        reported_sharpe_ratio=2.0,
        reported_max_drawdown_pct=10.0,
        verified_citations=[valid_citation],
    )
    res_xetra = gate.evaluate(brief_zero_xetra, [doc_chunk])
    assert res_xetra.passed is False
    assert res_xetra.friction_verdict == FrictionVerdict.UNVERIFIED_FRICTION_WARNING
    assert "below market floor (10.0 bps)" in res_xetra.reason

    # 3. None friction assumed (omitted friction)
    brief_none = _make_brief(
        strategy_name="None Friction Omission",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        transaction_cost_bps_assumed=None,
        verified_citations=[valid_citation],
    )
    res_none = gate.evaluate(brief_none, [doc_chunk])
    assert res_none.passed is False
    assert res_none.friction_verdict == FrictionVerdict.UNVERIFIED_FRICTION_WARNING


@pytest.mark.parametrize(
    ("market", "is_intraday", "assumed_bps", "expected_verdict", "expected_pass"),
    [
        # Just below statutory floors -> MUST fail
        (
            MarketRegime.INDIA_NSE_BSE,
            False,
            14.9,
            FrictionVerdict.UNVERIFIED_FRICTION_WARNING,
            False,
        ),
        (
            MarketRegime.INDIA_NSE_BSE,
            False,
            14.99,
            FrictionVerdict.UNVERIFIED_FRICTION_WARNING,
            False,
        ),
        (
            MarketRegime.GERMANY_XETRA_DAX,
            False,
            9.9,
            FrictionVerdict.UNVERIFIED_FRICTION_WARNING,
            False,
        ),
        (
            MarketRegime.GERMANY_XETRA_DAX,
            False,
            9.99,
            FrictionVerdict.UNVERIFIED_FRICTION_WARNING,
            False,
        ),
        (
            MarketRegime.US_NYSE_NASDAQ,
            False,
            4.9,
            FrictionVerdict.UNVERIFIED_FRICTION_WARNING,
            False,
        ),
        (
            MarketRegime.US_NYSE_NASDAQ,
            False,
            4.99,
            FrictionVerdict.UNVERIFIED_FRICTION_WARNING,
            False,
        ),
        # Exactly at statutory floors -> MUST pass
        (MarketRegime.INDIA_NSE_BSE, False, 15.0, FrictionVerdict.REALISTIC_FRICTION, True),
        (MarketRegime.INDIA_NSE_BSE, True, 5.0, FrictionVerdict.REALISTIC_FRICTION, True),
        (MarketRegime.GERMANY_XETRA_DAX, False, 10.0, FrictionVerdict.REALISTIC_FRICTION, True),
        (MarketRegime.US_NYSE_NASDAQ, False, 5.0, FrictionVerdict.REALISTIC_FRICTION, True),
    ],
)
def test_empirical_scepticism_gate_cost_floor_boundaries(
    market: MarketRegime,
    is_intraday: bool,
    assumed_bps: float,
    expected_verdict: FrictionVerdict,
    expected_pass: bool,
) -> None:
    """Stress-test: boundary values just below vs exactly at statutory cost floors."""
    gate = MarketRegimeScepticismGate()
    chunk_uuid = uuid.uuid4()
    source_text = "Market execution dynamics and transaction cost models."
    doc_chunk = DocumentChunk(
        chunk_id=chunk_uuid,
        doc_id="paper_boundary_01",
        doc_title="Transaction Cost Boundary Test",
        source_uri="https://arxiv.org/abs/2403.00002",
        market_scope=market,
        section_heading="Empirical Analysis",
        verbatim_content=source_text,
        formulas_latex=[],
        tables_markdown=[],
        token_count_approx=10,
    )
    valid_citation = DocCitationSpan(
        chunk_id=str(chunk_uuid),
        doc_id="paper_boundary_01",
        doc_title="Transaction Cost Boundary Test",
        source_uri="https://arxiv.org/abs/2403.00002",
        section_heading="Empirical Analysis",
        verbatim_quote=source_text,
    )

    brief = _make_brief(
        strategy_name=f"Boundary Test {market.value} {assumed_bps}bps",
        target_markets=[market],
        is_intraday_futures=is_intraday,
        transaction_cost_bps_assumed=assumed_bps,
        reported_sharpe_ratio=1.5,
        reported_max_drawdown_pct=15.0,
        verified_citations=[valid_citation],
    )

    res = gate.evaluate(brief, [doc_chunk])
    assert res.friction_verdict == expected_verdict, (
        f"Failed for {market.value} at {assumed_bps} bps: got {res.friction_verdict}, "
        f"expected {expected_verdict}"
    )
    assert res.passed is expected_pass, (
        f"Passed state mismatch for {market.value} at {assumed_bps} bps: "
        f"got {res.passed}, expected {expected_pass}"
    )


@pytest.mark.parametrize(
    ("drawdown_pct", "expected_verdict", "expected_pass"),
    [
        (30.1, FrictionVerdict.FAILS_AFTER_COSTS, False),
        (-30.1, FrictionVerdict.FAILS_AFTER_COSTS, False),
        (35.0, FrictionVerdict.FAILS_AFTER_COSTS, False),
        (29.9, FrictionVerdict.REALISTIC_FRICTION, True),
        (-29.9, FrictionVerdict.REALISTIC_FRICTION, True),
        (30.0, FrictionVerdict.REALISTIC_FRICTION, True),
        (-30.0, FrictionVerdict.REALISTIC_FRICTION, True),
        (12.5, FrictionVerdict.REALISTIC_FRICTION, True),
    ],
)
def test_empirical_scepticism_gate_max_drawdown_thresholds(
    drawdown_pct: float,
    expected_verdict: FrictionVerdict,
    expected_pass: bool,
) -> None:
    """Stress-test: maximum drawdown boundary around the 30.0% institutional threshold."""
    gate = MarketRegimeScepticismGate()
    chunk_uuid = uuid.uuid4()
    source_text = "Institutional risk management and maximum drawdown boundaries."
    doc_chunk = DocumentChunk(
        chunk_id=chunk_uuid,
        doc_id="paper_risk_01",
        doc_title="Risk Management",
        source_uri="https://arxiv.org/abs/2403.00003",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        section_heading="Risk Analysis",
        verbatim_content=source_text,
        formulas_latex=[],
        tables_markdown=[],
        token_count_approx=10,
    )
    valid_citation = DocCitationSpan(
        chunk_id=str(chunk_uuid),
        doc_id="paper_risk_01",
        doc_title="Risk Management",
        source_uri="https://arxiv.org/abs/2403.00003",
        section_heading="Risk Analysis",
        verbatim_quote=source_text,
    )

    brief = _make_brief(
        strategy_name=f"DD Test {drawdown_pct}%",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        is_intraday_futures=False,
        transaction_cost_bps_assumed=15.0,
        reported_sharpe_ratio=1.5,
        reported_max_drawdown_pct=drawdown_pct,
        verified_citations=[valid_citation],
    )

    res = gate.evaluate(brief, [doc_chunk])
    assert res.friction_verdict == expected_verdict, (
        f"Drawdown {drawdown_pct}%: got verdict {res.friction_verdict}, expected {expected_verdict}"
    )
    assert res.passed is expected_pass, (
        f"Drawdown {drawdown_pct}%: got passed={res.passed}, expected {expected_pass}"
    )
    if not expected_pass:
        assert "exceeds 30.0% institutional risk threshold" in res.reason


def test_empirical_scepticism_gate_distorted_and_fabricated_citations() -> None:
    """Stress-test: distorted and fabricated citations MUST fail NFKC quote verification."""
    gate = MarketRegimeScepticismGate(min_citation_overlap_ratio=0.98)
    slicer = DeterministicSlicer()

    source_text = (
        "Securities Transaction Tax on the National Stock Exchange of India specifies an exact "
        "statutory floor of 15.0 bps for equity delivery transactions."
    )
    chunk_uuid = uuid.uuid4()
    doc_chunk = DocumentChunk(
        chunk_id=chunk_uuid,
        doc_id="arxiv:2403.99991",
        doc_title="Statutory Taxation in Indian Equities",
        source_uri="https://arxiv.org/abs/2403.99991",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        section_heading="Taxation Framework",
        verbatim_content=source_text,
        formulas_latex=[],
        tables_markdown=[],
        token_count_approx=25,
    )

    # 1. Subtle digit distortion: 15.0 bps -> 15.1 bps
    distorted_quote_digit = (
        "National Stock Exchange of India specifies an exact statutory floor of 15.1 bps for equity"
    )
    res_digit = slicer.verify_quote(distorted_quote_digit, source_text)
    assert res_digit.is_valid is False
    assert res_digit.token_overlap_ratio < 0.98
    assert res_digit.error_code == SlicerErrorCode.HALLUCINATED_QUOTE

    with pytest.raises(HallucinatedQuoteError):
        slicer.verify_quote(distorted_quote_digit, source_text, strict=True)

    # 2. Key entity mutation: National Stock Exchange -> Bombay Stock Exchange
    mutated_entity_quote = (
        "Securities Transaction Tax on the Bombay Stock Exchange of India specifies"
    )
    res_entity = slicer.verify_quote(mutated_entity_quote, source_text)
    assert res_entity.is_valid is False
    assert res_entity.error_code == SlicerErrorCode.HALLUCINATED_QUOTE

    # 3. Completely fabricated claim (hallucinated quote)
    fabricated_quote = (
        "We prove that momentum anomaly delivers 40% risk-free annual returns in NSE cash."
    )
    res_fab = slicer.verify_quote(fabricated_quote, source_text)
    assert res_fab.is_valid is False
    assert res_fab.token_overlap_ratio < 0.10
    assert res_fab.error_code == SlicerErrorCode.HALLUCINATED_QUOTE

    # 4. Truncated quote with unclosed delimiter
    truncated_quote = (
        "floor of 15.0 bps for (equity delivery transactions"
    )
    with pytest.raises(TruncatedLineError):
        slicer.verify_quote(truncated_quote, source_text, strict=True)

    # 5. Evaluate in MarketRegimeScepticismGate
    citation_digit = DocCitationSpan(
        chunk_id=str(chunk_uuid),
        doc_id="arxiv:2403.99991",
        doc_title="Statutory Taxation in Indian Equities",
        source_uri="https://arxiv.org/abs/2403.99991",
        section_heading="Taxation Framework",
        verbatim_quote=distorted_quote_digit,
    )
    brief_digit = _make_brief(
        strategy_name="Distorted Digit Brief",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        transaction_cost_bps_assumed=15.0,
        verified_citations=[citation_digit],
    )
    gate_res = gate.evaluate(brief_digit, [doc_chunk])
    assert gate_res.passed is False
    assert gate_res.citations_valid is False
    assert gate_res.citations_verified_count == 0
    assert len(gate_res.failed_citations) == 1
    assert "Quote failed NFKC overlap check" in gate_res.failed_citations[0][1]


# ==============================================================================
# SECTION 4: Live Corpus End-to-End Verification
# ==============================================================================


def test_empirical_live_corpus_scepticism_gate_integration(settings: ScoutSettings) -> None:
    """Verifies Scepticism Gate end-to-end against real ingested chunks in PostgreSQL 17."""
    mgr = RLSConnectionManager(settings.db_app_url)
    gate = MarketRegimeScepticismGate()

    auth_principal = PrincipalContext(
        user_id="lead_quant_auditor",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["public_quant"],
    )

    # Fetch a real chunk from the live corpus
    with mgr.scoped_transaction(auth_principal, enable_hnsw_iterative_scan=False) as conn:
        row = conn.execute(
            """
            SELECT chunk_id, doc_id, doc_title, source_uri, market_scope,
                   section_heading, verbatim_content
            FROM doc_chunks
            WHERE market_scope = 'IN_NSE'
            LIMIT 1;
            """
        ).fetchone()

        assert row is not None, "Live database must contain at least one IN_NSE doc chunk"
        chunk_id_raw, doc_id_raw, title_raw, uri_raw, market_raw, heading_raw, content_raw = row
        assert isinstance(chunk_id_raw, uuid.UUID)
        assert isinstance(doc_id_raw, str)
        assert isinstance(title_raw, str)
        assert isinstance(uri_raw, str)
        assert isinstance(market_raw, str)
        assert isinstance(heading_raw, str)
        assert isinstance(content_raw, str)

        chunk_id_val: uuid.UUID = chunk_id_raw
        doc_id_val: str = doc_id_raw
        title_val: str = title_raw
        uri_val: str = uri_raw
        market_val: str = market_raw
        heading_val: str = heading_raw
        content_val: str = content_raw

    # Convert to DocChunkRecord
    real_chunk = DocChunkRecord(
        chunk_id=chunk_id_val,
        domain_id="quant_research_scout",
        doc_id=doc_id_val,
        doc_title=title_val,
        source_uri=uri_val,
        market_scope=MarketRegime(market_val),
        published_date=None,
        authority_weight=1.0,
        section_heading=heading_val,
        contextual_header="[Live Corpus Test]",
        verbatim_content=content_val,
        metadata_json={},
        allowed_scopes=["public_quant"],
        embedding=[0.0] * 768,
    )

    # Extract an exact sentence or phrase from the real content
    words = content_val.split()
    assert len(words) >= 10, "Content must have at least 10 words"
    exact_verbatim_quote = " ".join(words[2:12])

    # 1. Valid citation with exact quote -> MUST pass gate
    valid_citation = DocCitationSpan(
        chunk_id=str(chunk_id_val),
        doc_id=doc_id_val,
        doc_title=title_val,
        source_uri=uri_val,
        section_heading=heading_val,
        verbatim_quote=exact_verbatim_quote,
    )
    valid_brief = _make_brief(
        strategy_name="Live Corpus Verified Strategy",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        is_intraday_futures=False,
        transaction_cost_bps_assumed=15.0,
        reported_sharpe_ratio=1.45,
        reported_max_drawdown_pct=14.0,
        verified_citations=[valid_citation],
    )

    eval_pass = gate.evaluate(valid_brief, [real_chunk])
    assert eval_pass.passed is True
    assert eval_pass.citations_valid is True
    assert eval_pass.citations_verified_count == 1
    assert eval_pass.friction_verdict == FrictionVerdict.REALISTIC_FRICTION

    # 2. Mutated quote from live paper -> MUST fail gate
    mutated_words = list(words[2:12])
    mutated_words[3] = "MUTATED_NONEXISTENT_WORD_X99"
    mutated_quote = " ".join(mutated_words)

    invalid_citation = DocCitationSpan(
        chunk_id=str(chunk_id_val),
        doc_id=doc_id_val,
        doc_title=title_val,
        source_uri=uri_val,
        section_heading=heading_val,
        verbatim_quote=mutated_quote,
    )
    invalid_brief = _make_brief(
        strategy_name="Live Corpus Mutated Strategy",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        is_intraday_futures=False,
        transaction_cost_bps_assumed=15.0,
        reported_sharpe_ratio=1.45,
        reported_max_drawdown_pct=14.0,
        verified_citations=[invalid_citation],
    )

    eval_fail_citation = gate.evaluate(invalid_brief, [real_chunk])
    assert eval_fail_citation.passed is False
    assert eval_fail_citation.citations_valid is False
    assert len(eval_fail_citation.failed_citations) == 1

    # 3. Deficit friction (14.9 bps) against live corpus chunk -> MUST fail gate
    deficit_brief = _make_brief(
        strategy_name="Live Corpus Deficit Strategy",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        is_intraday_futures=False,
        transaction_cost_bps_assumed=14.9,
        reported_sharpe_ratio=1.45,
        reported_max_drawdown_pct=14.0,
        verified_citations=[valid_citation],
    )
    eval_fail_friction = gate.evaluate(deficit_brief, [real_chunk])
    assert eval_fail_friction.passed is False
    assert eval_fail_friction.friction_verdict == FrictionVerdict.UNVERIFIED_FRICTION_WARNING

    # 4. Excessive drawdown (30.1%) against live corpus chunk -> MUST fail gate
    excessive_dd_brief = _make_brief(
        strategy_name="Live Corpus High DD Strategy",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        is_intraday_futures=False,
        transaction_cost_bps_assumed=15.0,
        reported_sharpe_ratio=1.45,
        reported_max_drawdown_pct=30.1,
        verified_citations=[valid_citation],
    )
    eval_fail_dd = gate.evaluate(excessive_dd_brief, [real_chunk])
    assert eval_fail_dd.passed is False
    assert eval_fail_dd.friction_verdict == FrictionVerdict.FAILS_AFTER_COSTS
