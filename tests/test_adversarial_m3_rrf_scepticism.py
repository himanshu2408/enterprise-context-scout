"""Empirical Adversarial Stress Tests for Milestone 3 (RRF BLK-01 & Scepticism Gate FIN-01).

Challenger 2 Empirical Verification:
1. 3-Way Normalized RRF (BLK-01):
   - Invariant: Rank #1 across all 3 modalities yields mathematically exact 1.0.
   - Extreme & boundary ranks: 0 hits, single hit in 1 modality, 2 modalities, symmetry.
   - Monotonicity: Component-wise monotonic decrease across rank space.
   - Score bounds: Strict containment in [0.0, 1.0] across all inputs.
   - Parameter validation: ValueError on negative/zero ranks, invalid k, non-positive modalities.
   - Multi-tenant RLS isolation: Forced isolation across InMemoryRLS and live PostgreSQL 17.
2. Market-Regime Quant Scepticism Gate (FIN-01):
   - Boundary checks at exact floor:
     - IN_NSE delivery: 14.99 bps (warning) vs 15.00 bps (pass) vs 15.01 bps (pass).
     - IN_NSE intraday futures: 4.99 bps (warning) vs 5.00 bps (pass) vs 5.01 bps (pass).
     - DE_XETRA: 9.99 bps (warning) vs 10.00 bps (pass) vs 10.01 bps (pass).
     - US_EQUITY: 4.99 bps (warning) vs 5.00 bps (pass) vs 5.01 bps (pass).
     - Multi-market combinations: max-floor aggregation.
   - Drawdown boundaries: 30.00% (pass) vs 30.01% (fail after costs).
   - Adversarial citation verification (VAL-02):
     - Single-character/digit mutation (15.2 -> 15.3 bps) triggers
       HallucinatedQuoteError in strict mode.
     - Missing doc chunk ID, empty citations, and truncated quotes.
"""

import math
import random
import uuid
from datetime import date

import pytest
from pydantic import ValidationError

from context_scout.ast_engine.deterministic_slicer import (
    DeterministicSlicer,
    HallucinatedQuoteError,
    TruncatedLineError,
)
from context_scout.ast_engine.models import SlicerErrorCode
from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    DocChunkRecord,
    InMemoryRLSRepository,
    PostgresRLSRepository,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.ingestion.doc_parser import DocumentChunk
from context_scout.ingestion.git_merkle_indexer import generate_deterministic_embedding
from context_scout.ingestion.scepticism_gate import (
    MarketRegimeScepticismGate,
    calculate_effective_friction_floor,
)
from context_scout.schemas.contracts import (
    DocCitationSpan,
    FrictionVerdict,
    MarketRegime,
    PrincipalContext,
    QuantStrategyBrief,
    RoleType,
    compute_normalized_rrf,
    evaluate_market_friction_floor,
)

# ==============================================================================
# SECTION 1: 3-Way Normalized RRF Stress Testing (BLK-01)
# ==============================================================================


def test_normalized_rrf_rank_one_across_all_modalities_equals_one() -> None:
    """Stress-tests invariant that rank #1 across all M modalities yields exactly 1.0.

    Mathematical Invariant:
        raw_score = M / (k + 1)
        max_possible = M / (k + 1)
        normalized = raw_score / max_possible = 1.0
    """
    # Standard 3-way RRF with default k=60
    score_default = compute_normalized_rrf([1, 1, 1], num_modalities=3, k=60)
    assert score_default == 1.0, f"Expected exactly 1.0, got {score_default}"
    assert math.isclose(score_default, 1.0, rel_tol=1e-12)

    # Invariant must hold across diverse k values (0, 1, 10, 20, 60, 100, 1000)
    for k_val in [0, 1, 10, 20, 60, 100, 1000]:
        score_k = compute_normalized_rrf([1, 1, 1], num_modalities=3, k=k_val)
        assert score_k == 1.0, f"Failed at k={k_val}: got {score_k}"

    # Invariant must hold across arbitrary modality counts (1, 2, 3, 4, 5, 10)
    for num_mods in [1, 2, 3, 4, 5, 10]:
        ranks: list[int | None] = [1] * num_mods
        score_m = compute_normalized_rrf(ranks, num_modalities=num_mods, k=60)
        assert score_m == 1.0, f"Failed at num_modalities={num_mods}: got {score_m}"


def test_normalized_rrf_zero_and_sparse_hits() -> None:
    """Stress-tests sparse modality hits: 0 hits, single hit in 1 modality, 2 modalities."""
    # 0 hits (all modalities None)
    score_zero = compute_normalized_rrf([None, None, None], num_modalities=3, k=60)
    assert score_zero == 0.0

    # Empty ranks list
    score_empty = compute_normalized_rrf([], num_modalities=3, k=60)
    assert score_empty == 0.0

    # Single hit in 1 modality: must be exactly 1/3 (0.333333) regardless of which modality
    score_mod0 = compute_normalized_rrf([1, None, None], num_modalities=3, k=60)
    score_mod1 = compute_normalized_rrf([None, 1, None], num_modalities=3, k=60)
    score_mod2 = compute_normalized_rrf([None, None, 1], num_modalities=3, k=60)

    assert score_mod0 == 0.333333
    assert score_mod1 == 0.333333
    assert score_mod2 == 0.333333
    # Symmetric property across modalities
    assert score_mod0 == score_mod1 == score_mod2

    # Two hits in 2 modalities: must be exactly 2/3 (0.666667)
    score_two_01 = compute_normalized_rrf([1, 1, None], num_modalities=3, k=60)
    score_two_02 = compute_normalized_rrf([1, None, 1], num_modalities=3, k=60)
    score_two_12 = compute_normalized_rrf([None, 1, 1], num_modalities=3, k=60)

    assert score_two_01 == 0.666667
    assert score_two_02 == 0.666667
    assert score_two_12 == 0.666667
    assert score_two_01 == score_two_02 == score_two_12

    # Lower rank in single modality: rank 2 -> (1/62) / (3/61) = 61 / 186 = 0.327957
    score_rank2 = compute_normalized_rrf([2, None, None], num_modalities=3, k=60)
    expected_rank2 = round((1.0 / 62.0) / (3.0 / 61.0), 6)
    assert score_rank2 == expected_rank2 == 0.327957


def test_normalized_rrf_monotonicity_and_bounds() -> None:
    """Empirically stress-tests strict monotonicity and boundary containment in [0.0, 1.0]."""
    # 1. Monotonic decrease along single modality
    prev_score = 1.0
    for r in range(1, 100):
        current_score = compute_normalized_rrf([r, r, r], num_modalities=3, k=60)
        assert 0.0 <= current_score <= 1.0
        assert current_score < prev_score or r == 1, (
            f"Monotonicity violated at rank {r}: {current_score} >= {prev_score}"
        )
        prev_score = current_score

    # 2. Random permutation component-wise monotonicity test (500 trials)
    rng = random.Random(42)
    for _ in range(500):
        r1 = rng.randint(1, 200)
        r2 = rng.randint(1, 200)
        r3 = rng.randint(1, 200)

        # Deteriorated ranks (higher number = worse rank)
        delta1 = rng.randint(0, 50)
        delta2 = rng.randint(0, 50)
        delta3 = rng.randint(0, 50)
        if delta1 == 0 and delta2 == 0 and delta3 == 0:
            delta1 = 1

        score_better = compute_normalized_rrf([r1, r2, r3], num_modalities=3, k=60)
        score_worse = compute_normalized_rrf(
            [r1 + delta1, r2 + delta2, r3 + delta3], num_modalities=3, k=60
        )

        assert 0.0 <= score_better <= 1.0
        assert 0.0 <= score_worse <= 1.0
        assert score_better > score_worse, (
            f"Monotonicity failure: ({r1},{r2},{r3}) -> {score_better} "
            f"vs ({r1 + delta1},{r2 + delta2},{r3 + delta3}) -> {score_worse}"
        )

    # 3. Extreme boundary ranks
    extreme_score = compute_normalized_rrf(
        [10_000_000, 10_000_000, 10_000_000], num_modalities=3, k=60
    )
    assert 0.0 < extreme_score <= 1.0
    assert extreme_score < 0.0001


def test_normalized_rrf_invalid_inputs_rejected() -> None:
    """Verifies that invalid ranks, non-positive modalities, or negative k raise ValueError."""
    with pytest.raises(ValueError, match="Rank must be >= 1"):
        compute_normalized_rrf([0, 1, 1], num_modalities=3, k=60)

    with pytest.raises(ValueError, match="Rank must be >= 1"):
        compute_normalized_rrf([-1, 1, 1], num_modalities=3, k=60)

    with pytest.raises(ValueError, match="num_modalities must be >= 1"):
        compute_normalized_rrf([1, 1, 1], num_modalities=0, k=60)

    with pytest.raises(ValueError, match="num_modalities must be >= 1"):
        compute_normalized_rrf([1, 1, 1], num_modalities=-3, k=60)

    with pytest.raises(ValueError, match="k must be >= 0"):
        compute_normalized_rrf([1, 1, 1], num_modalities=3, k=-1)


def test_inmemory_hybrid_rrf_multi_tenant_rls_isolation() -> None:
    """Stress-tests InMemoryRLSRepository hybrid RRF for multi-tenant scope isolation."""
    repo = InMemoryRLSRepository()

    tenant_a_principal = PrincipalContext(
        user_id="trader_alpha",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["tenant_alpha_restricted"],
    )
    tenant_b_principal = PrincipalContext(
        user_id="trader_beta",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["tenant_beta_restricted"],
    )
    public_principal = PrincipalContext(
        user_id="public_viewer",
        role=RoleType.PRODUCT_MANAGER,
        allowed_scopes=["public_guest"],
    )

    id_a = uuid.uuid4()
    id_b = uuid.uuid4()
    emb = [0.1] * 768

    chunk_a = DocChunkRecord(
        chunk_id=id_a,
        domain_id="quant_research_scout",
        doc_id="alpha_doc_01",
        doc_title="Alpha Momentum Strategy",
        source_uri="https://internal/alpha_01",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        published_date=date(2025, 1, 1),
        authority_weight=1.0,
        section_heading="Alpha Strategy",
        contextual_header="[Context: alpha_01 | Market: IN_NSE | Tenant Alpha]",
        verbatim_content="Confidential alpha momentum formula with STT 15 bps delivery.",
        metadata_json={"tenant": "alpha"},
        allowed_scopes=["tenant_alpha_restricted"],
        embedding=emb,
    )
    chunk_b = DocChunkRecord(
        chunk_id=id_b,
        domain_id="quant_research_scout",
        doc_id="beta_doc_01",
        doc_title="Beta Arbitrage Strategy",
        source_uri="https://internal/beta_01",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        published_date=date(2025, 1, 1),
        authority_weight=1.0,
        section_heading="Beta Strategy",
        contextual_header="[Context: beta_01 | Market: IN_NSE | Tenant Beta]",
        verbatim_content="Confidential beta arbitrage formula with STT 15 bps delivery.",
        metadata_json={"tenant": "beta"},
        allowed_scopes=["tenant_beta_restricted"],
        embedding=emb,
    )

    repo.insert_doc_chunk(tenant_a_principal, chunk_a)
    repo.insert_doc_chunk(tenant_b_principal, chunk_b)

    # 1. Tenant Alpha searches: MUST NOT see Tenant Beta's chunk
    hits_a = repo.search_doc_chunks_hybrid_rrf(
        principal=tenant_a_principal,
        domain_id="quant_research_scout",
        query_text="arbitrage formula STT",
        query_embedding=emb,
    )
    assert len(hits_a) == 1
    assert hits_a[0].item_id == id_a
    assert all(h.item_id != id_b for h in hits_a)
    # Since only 1 chunk is visible to tenant A, it is rank 1 in all 3 modalities -> score == 1.0
    assert hits_a[0].normalized_rrf_score == 1.0

    # 2. Tenant Beta searches: MUST NOT see Tenant Alpha's chunk
    hits_b = repo.search_doc_chunks_hybrid_rrf(
        principal=tenant_b_principal,
        domain_id="quant_research_scout",
        query_text="momentum formula STT",
        query_embedding=emb,
    )
    assert len(hits_b) == 1
    assert hits_b[0].item_id == id_b
    assert all(h.item_id != id_a for h in hits_b)
    assert hits_b[0].normalized_rrf_score == 1.0

    # 3. Public principal searches: MUST see 0 chunks
    hits_public = repo.search_doc_chunks_hybrid_rrf(
        principal=public_principal,
        domain_id="quant_research_scout",
        query_text="formula STT",
        query_embedding=emb,
    )
    assert len(hits_public) == 0


def test_postgres_live_forced_rls_multi_tenant_isolation() -> None:
    """Stress-tests forced RLS multi-tenant boundary on live PostgreSQL 17 daemon."""
    settings = ScoutSettings()
    mgr = RLSConnectionManager(settings.db_app_url)
    admin_mgr = RLSConnectionManager(settings.db_admin_url)
    repo = PostgresRLSRepository()

    admin_principal = PrincipalContext(
        user_id="admin_sec_auditor",
        role=RoleType.ENGINEER,
        allowed_scopes=["tenant_x_isolated", "tenant_y_isolated"],
    )

    # Ensure domain pack exists
    with admin_mgr.scoped_transaction(admin_principal, enable_hnsw_iterative_scan=False) as conn:
        conn.execute(
            """
            INSERT INTO domain_packs (
                domain_id, display_name, lora_adapter_id,
                primary_truth_source, freshness_half_life_days, config_json
            )
            VALUES (
                'quant_research_scout', 'Quant Research Scout', 'gemma4_26b_quant_scout',
                'document', 180, '{"m3_adversarial": true}'::jsonb
            )
            ON CONFLICT (domain_id) DO NOTHING;
            """
        )

    principal_x = PrincipalContext(
        user_id="user_tenant_x",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["tenant_x_isolated"],
    )
    principal_y = PrincipalContext(
        user_id="user_tenant_y",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["tenant_y_isolated"],
    )
    principal_unauthorized = PrincipalContext(
        user_id="user_unauthorized",
        role=RoleType.PRODUCT_MANAGER,
        allowed_scopes=["public_guest_only"],
    )

    id_x = uuid.uuid4()
    id_y = uuid.uuid4()
    emb = generate_deterministic_embedding("multi-tenant rls test vector", 768)

    chunk_x = DocChunkRecord(
        chunk_id=id_x,
        domain_id="quant_research_scout",
        doc_id="tenant_x_doc",
        doc_title="Tenant X Proprietary Flow",
        source_uri="https://tenant_x.internal/doc",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        published_date=date(2025, 2, 1),
        authority_weight=1.0,
        section_heading="Tenant X Execution",
        contextual_header="[Context: tenant_x | Market: IN_NSE]",
        verbatim_content="Securities Transaction Tax 15 bps delivery on NSE for tenant X.",
        metadata_json={"tenant": "x"},
        allowed_scopes=["tenant_x_isolated"],
        embedding=emb,
    )
    chunk_y = DocChunkRecord(
        chunk_id=id_y,
        domain_id="quant_research_scout",
        doc_id="tenant_y_doc",
        doc_title="Tenant Y Proprietary Flow",
        source_uri="https://tenant_y.internal/doc",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        published_date=date(2025, 2, 1),
        authority_weight=1.0,
        section_heading="Tenant Y Execution",
        contextual_header="[Context: tenant_y | Market: IN_NSE]",
        verbatim_content="Securities Transaction Tax 15 bps delivery on NSE for tenant Y.",
        metadata_json={"tenant": "y"},
        allowed_scopes=["tenant_y_isolated"],
        embedding=emb,
    )

    # Insert under respective tenant connections
    with mgr.scoped_transaction(principal_x, enable_hnsw_iterative_scan=False) as conn:
        repo.insert_doc_chunk(conn, chunk_x)

    with mgr.scoped_transaction(principal_y, enable_hnsw_iterative_scan=False) as conn:
        repo.insert_doc_chunk(conn, chunk_y)

    try:
        # Principal X searches with exact text that matches both documents
        with mgr.scoped_transaction(principal_x, enable_hnsw_iterative_scan=True) as conn:
            hits_x = repo.search_doc_chunks_hybrid_rrf(
                conn=conn,
                domain_id="quant_research_scout",
                query_text="Securities Transaction Tax 15 bps delivery",
                query_embedding=emb,
                market_scope=MarketRegime.INDIA_NSE_BSE,
                top_k=10,
            )
            assert any(h.item_id == id_x for h in hits_x), "Tenant X should see its own chunk"
            assert not any(h.item_id == id_y for h in hits_x), (
                "CRITICAL SECURITY VIOLATION: Tenant X saw Tenant Y's isolated row in RRF search!"
            )

        # Principal Y searches with exact text
        with mgr.scoped_transaction(principal_y, enable_hnsw_iterative_scan=True) as conn:
            hits_y = repo.search_doc_chunks_hybrid_rrf(
                conn=conn,
                domain_id="quant_research_scout",
                query_text="Securities Transaction Tax 15 bps delivery",
                query_embedding=emb,
                market_scope=MarketRegime.INDIA_NSE_BSE,
                top_k=10,
            )
            assert any(h.item_id == id_y for h in hits_y), "Tenant Y should see its own chunk"
            assert not any(h.item_id == id_x for h in hits_y), (
                "CRITICAL SECURITY VIOLATION: Tenant Y saw Tenant X's isolated row in RRF search!"
            )

        # Unauthorized Principal searches
        with mgr.scoped_transaction(
            principal_unauthorized, enable_hnsw_iterative_scan=True
        ) as conn:
            hits_unauth = repo.search_doc_chunks_hybrid_rrf(
                conn=conn,
                domain_id="quant_research_scout",
                query_text="Securities Transaction Tax 15 bps delivery",
                query_embedding=emb,
                market_scope=MarketRegime.INDIA_NSE_BSE,
                top_k=10,
            )
            assert len(hits_unauth) == 0, (
                "CRITICAL SECURITY VIOLATION: Unauthorized user saw isolated rows!"
            )

    finally:
        with admin_mgr.scoped_transaction(
            admin_principal, enable_hnsw_iterative_scan=False
        ) as conn:
            conn.execute("DELETE FROM doc_chunks WHERE chunk_id IN (%s, %s);", (id_x, id_y))


# ==============================================================================
# SECTION 2: Market-Regime Scepticism Gate Stress Testing (FIN-01 & VAL-02)
# ==============================================================================


@pytest.mark.parametrize(
    ("market", "is_intraday", "assumed_bps", "expected_verdict"),
    [
        # IN_NSE Delivery (Statutory floor: 15.00 bps)
        (MarketRegime.INDIA_NSE_BSE, False, 14.99, FrictionVerdict.UNVERIFIED_FRICTION_WARNING),
        (MarketRegime.INDIA_NSE_BSE, False, 15.00, FrictionVerdict.REALISTIC_FRICTION),
        (MarketRegime.INDIA_NSE_BSE, False, 15.01, FrictionVerdict.REALISTIC_FRICTION),
        (MarketRegime.INDIA_NSE_BSE, False, 0.0, FrictionVerdict.UNVERIFIED_FRICTION_WARNING),
        (MarketRegime.INDIA_NSE_BSE, False, None, FrictionVerdict.UNVERIFIED_FRICTION_WARNING),
        # IN_NSE Intraday Futures (Floor: 5.00 bps)
        (MarketRegime.INDIA_NSE_BSE, True, 4.99, FrictionVerdict.UNVERIFIED_FRICTION_WARNING),
        (MarketRegime.INDIA_NSE_BSE, True, 5.00, FrictionVerdict.REALISTIC_FRICTION),
        (MarketRegime.INDIA_NSE_BSE, True, 5.01, FrictionVerdict.REALISTIC_FRICTION),
        # DE_XETRA (MiFID II execution floor: 10.00 bps)
        (MarketRegime.GERMANY_XETRA_DAX, False, 9.99, FrictionVerdict.UNVERIFIED_FRICTION_WARNING),
        (MarketRegime.GERMANY_XETRA_DAX, False, 10.00, FrictionVerdict.REALISTIC_FRICTION),
        (MarketRegime.GERMANY_XETRA_DAX, False, 10.01, FrictionVerdict.REALISTIC_FRICTION),
        # US_EQUITY (SEC + FINRA + spread floor: 5.00 bps)
        (MarketRegime.US_NYSE_NASDAQ, False, 4.99, FrictionVerdict.UNVERIFIED_FRICTION_WARNING),
        (MarketRegime.US_NYSE_NASDAQ, False, 5.00, FrictionVerdict.REALISTIC_FRICTION),
        (MarketRegime.US_NYSE_NASDAQ, False, 5.01, FrictionVerdict.REALISTIC_FRICTION),
    ],
)
def test_market_regime_scepticism_friction_floor_boundaries(
    market: MarketRegime,
    is_intraday: bool,
    assumed_bps: float | None,
    expected_verdict: FrictionVerdict,
) -> None:
    """Stress-tests exact boundary checks around statutory transaction cost floors."""
    verdict = evaluate_market_friction_floor(
        target_markets=[market],
        assumed_bps=assumed_bps,
        is_intraday_futures=is_intraday,
    )
    assert verdict == expected_verdict, (
        f"Failed for {market.value} (intraday={is_intraday}) with {assumed_bps} bps: "
        f"got {verdict.value}, expected {expected_verdict.value}"
    )


def test_market_regime_multi_market_floor_aggregation() -> None:
    """Stress-tests multi-market portfolio friction floor aggregation."""
    # 1. Combined IN_NSE (15 bps) + US_EQUITY (5 bps) -> effective floor is max(15, 5) = 15.0 bps
    floor_in_us = calculate_effective_friction_floor(
        [MarketRegime.INDIA_NSE_BSE, MarketRegime.US_NYSE_NASDAQ],
        is_intraday_futures=False,
    )
    assert floor_in_us == 15.0

    assert (
        evaluate_market_friction_floor(
            [MarketRegime.INDIA_NSE_BSE, MarketRegime.US_NYSE_NASDAQ],
            assumed_bps=14.99,
            is_intraday_futures=False,
        )
        == FrictionVerdict.UNVERIFIED_FRICTION_WARNING
    )
    assert (
        evaluate_market_friction_floor(
            [MarketRegime.INDIA_NSE_BSE, MarketRegime.US_NYSE_NASDAQ],
            assumed_bps=15.00,
            is_intraday_futures=False,
        )
        == FrictionVerdict.REALISTIC_FRICTION
    )

    # 2. Combined DE_XETRA (10 bps) + US_EQUITY (5 bps) -> effective floor is 10.0 bps
    floor_de_us = calculate_effective_friction_floor(
        [MarketRegime.GERMANY_XETRA_DAX, MarketRegime.US_NYSE_NASDAQ],
        is_intraday_futures=False,
    )
    assert floor_de_us == 10.0

    assert (
        evaluate_market_friction_floor(
            [MarketRegime.GERMANY_XETRA_DAX, MarketRegime.US_NYSE_NASDAQ],
            assumed_bps=9.99,
        )
        == FrictionVerdict.UNVERIFIED_FRICTION_WARNING
    )
    assert (
        evaluate_market_friction_floor(
            [MarketRegime.GERMANY_XETRA_DAX, MarketRegime.US_NYSE_NASDAQ],
            assumed_bps=10.00,
        )
        == FrictionVerdict.REALISTIC_FRICTION
    )


def test_scepticism_gate_adversarial_single_digit_mutation() -> None:
    """Stress-tests citation verification against adversarial single-character/digit mutation.

    Adversarial Scenario:
        Source: "...friction of 15.2 bps produces..."
        Mutated: "...friction of 15.3 bps produces..."
        Requirement:
            - Strict mode verify_quote MUST raise HallucinatedQuoteError.
            - Gate evaluation MUST fail verification and mark citations_valid=False.
    """
    slicer = DeterministicSlicer()
    source_chunk = (
        "Empirical backtesting across the National Stock Exchange of India shows that a delivery "
        "friction of 15.2 bps produces an annual Sharpe ratio of 1.45."
    )
    valid_quote = (
        "National Stock Exchange of India shows that a delivery friction of 15.2 bps produces"
    )
    # Adversarial single-digit alteration: 15.2 -> 15.3 bps
    mutated_quote = (
        "National Stock Exchange of India shows that a delivery friction of 15.3 bps produces"
    )

    # 1. Valid quote verifies with 1.0 overlap
    v_valid = slicer.verify_quote(valid_quote, source_chunk)
    assert v_valid.is_valid is True
    assert v_valid.token_overlap_ratio == 1.0

    # 2. Mutated quote fails overlap check and returns HALLUCINATED_QUOTE
    v_mutated = slicer.verify_quote(mutated_quote, source_chunk)
    assert v_mutated.is_valid is False
    assert v_mutated.token_overlap_ratio < 0.98
    assert v_mutated.error_code == SlicerErrorCode.HALLUCINATED_QUOTE

    # 3. Strict mode MUST raise HallucinatedQuoteError
    with pytest.raises(HallucinatedQuoteError, match="Quote failed verification"):
        slicer.verify_quote(mutated_quote, source_chunk, strict=True)

    # 4. Integrate with MarketRegimeScepticismGate
    gate = MarketRegimeScepticismGate(min_citation_overlap_ratio=0.98)
    chunk_uuid = uuid.uuid4()
    chunk_id = str(chunk_uuid)
    doc_chunk = DocumentChunk(
        chunk_id=chunk_uuid,
        doc_id="arxiv:2403.01182",
        doc_title="Momentum with Indian STT",
        source_uri="https://arxiv.org/abs/2403.01182",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        section_heading="Empirical Results",
        verbatim_content=source_chunk,
        formulas_latex=[],
        tables_markdown=[],
        token_count_approx=35,
    )

    # Test with valid citation
    valid_citation = DocCitationSpan(
        chunk_id=chunk_id,
        doc_id="arxiv:2403.01182",
        doc_title="Momentum with Indian STT",
        source_uri="https://arxiv.org/abs/2403.01182",
        section_heading="Empirical Results",
        verbatim_quote=valid_quote,
    )
    valid_brief = _make_brief(
        strategy_name="NSE Momentum Alpha",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        is_intraday_futures=False,
        transaction_cost_bps_assumed=15.2,
        reported_sharpe_ratio=1.45,
        reported_max_drawdown_pct=14.0,
        verified_citations=[valid_citation],
    )
    res_valid = gate.evaluate(valid_brief, [doc_chunk])
    assert res_valid.passed is True
    assert res_valid.citations_valid is True
    assert res_valid.citations_verified_count == 1
    assert len(res_valid.failed_citations) == 0

    # Test with mutated citation (15.2 -> 15.3)
    mutated_citation = DocCitationSpan(
        chunk_id=chunk_id,
        doc_id="arxiv:2403.01182",
        doc_title="Momentum with Indian STT",
        source_uri="https://arxiv.org/abs/2403.01182",
        section_heading="Empirical Results",
        verbatim_quote=mutated_quote,
    )
    adversarial_brief = _make_brief(
        strategy_name="NSE Momentum Alpha",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        is_intraday_futures=False,
        transaction_cost_bps_assumed=15.2,
        reported_sharpe_ratio=1.45,
        reported_max_drawdown_pct=14.0,
        verified_citations=[mutated_citation],
    )
    res_adversarial = gate.evaluate(adversarial_brief, [doc_chunk])
    assert res_adversarial.passed is False
    assert res_adversarial.citations_valid is False
    assert res_adversarial.citations_verified_count == 0
    assert len(res_adversarial.failed_citations) == 1
    assert "Quote failed NFKC overlap check" in res_adversarial.failed_citations[0][1]


def _make_brief(
    strategy_name: str = "Test Strategy",
    target_markets: list[MarketRegime] | None = None,
    is_intraday_futures: bool = False,
    transaction_cost_bps_assumed: float | None = 15.0,
    reported_sharpe_ratio: float | None = 1.5,
    reported_max_drawdown_pct: float | None = 10.0,
    verified_citations: list[DocCitationSpan] | None = None,
    mathematical_signal_latex: str = r"S_t = \frac{P_t - \mu}{\sigma}",
    entry_exit_rules: list[str] | None = None,
    sample_period_years: str = "2018-2023",
    market_specific_caveats: list[str] | None = None,
    known_failure_regimes: list[str] | None = None,
) -> QuantStrategyBrief:
    """Helper to construct QuantStrategyBrief satisfying all required contract fields."""
    return QuantStrategyBrief(
        strategy_name=strategy_name,
        target_markets=target_markets or [MarketRegime.INDIA_NSE_BSE],
        is_intraday_futures=is_intraday_futures,
        transaction_cost_bps_assumed=transaction_cost_bps_assumed,
        reported_sharpe_ratio=reported_sharpe_ratio,
        reported_max_drawdown_pct=reported_max_drawdown_pct,
        verified_citations=verified_citations if verified_citations is not None else [],
        mathematical_signal_latex=mathematical_signal_latex,
        entry_exit_rules=entry_exit_rules or ["Enter when S_t > 2.0", "Exit when S_t < 0.0"],
        sample_period_years=sample_period_years,
        market_specific_caveats=market_specific_caveats or ["Subject to STT delivery tax"],
        known_failure_regimes=known_failure_regimes or ["High volatility regime"],
    )


def test_scepticism_gate_adversarial_citation_failure_modes() -> None:
    """Stress-tests missing chunk IDs, 0 citations, and truncated quotes in Scepticism Gate."""
    gate = MarketRegimeScepticismGate()
    chunk_uuid = uuid.uuid4()
    doc_chunk = DocumentChunk(
        chunk_id=chunk_uuid,
        doc_id="arxiv:2403.01182",
        doc_title="Momentum with Indian STT",
        source_uri="https://arxiv.org/abs/2403.01182",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        section_heading="Empirical Results",
        verbatim_content="Delivery friction of 15.0 bps satisfies institutional requirements.",
        formulas_latex=[],
        tables_markdown=[],
        token_count_approx=20,
    )

    # 1. Missing chunk ID in corpus
    missing_id_citation = DocCitationSpan(
        chunk_id=str(uuid.uuid4()),
        doc_id="arxiv:9999.99999",
        doc_title="Ghost Document",
        source_uri="https://arxiv.org/abs/9999.99999",
        section_heading="Ghosts",
        verbatim_quote="Delivery friction of 15.0 bps satisfies",
    )
    brief_missing = _make_brief(
        strategy_name="Ghost Brief",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        transaction_cost_bps_assumed=15.0,
        reported_sharpe_ratio=1.5,
        reported_max_drawdown_pct=10.0,
        verified_citations=[missing_id_citation],
    )
    res_missing = gate.evaluate(brief_missing, [doc_chunk])
    assert res_missing.passed is False
    assert res_missing.citations_valid is False
    assert "not in corpus" in res_missing.failed_citations[0][1]

    # 2. Empty citations list (cannot pass institutional audit without citations)
    # The Pydantic model contract rejects verified_citations=[] at the boundary
    with pytest.raises(ValidationError):
        _make_brief(verified_citations=[])

    # 3. Truncated citation with unclosed delimiter
    slicer = DeterministicSlicer()
    truncated_source = "Friction formula is f(x) = (cost + spread) / volume"
    truncated_quote = "formula is f(x) = (cost + spread"
    with pytest.raises(TruncatedLineError, match=r"Truncated quote|unclosed delimiters"):
        slicer.verify_quote(truncated_quote, truncated_source, strict=True)


def test_scepticism_gate_risk_threshold_boundaries() -> None:
    """Stress-tests institutional risk boundaries: 30.00% max drawdown and net Sharpe penalty."""
    gate = MarketRegimeScepticismGate()
    chunk_uuid = uuid.uuid4()
    chunk_id = str(chunk_uuid)
    doc_chunk = DocumentChunk(
        chunk_id=chunk_uuid,
        doc_id="arxiv:2403.01182",
        doc_title="Valid Title",
        source_uri="https://arxiv.org/abs/2403.01182",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        section_heading="Empirical Results",
        verbatim_content="Delivery friction of 15.0 bps satisfies institutional requirements.",
        formulas_latex=[],
        tables_markdown=[],
        token_count_approx=20,
    )
    valid_citation = DocCitationSpan(
        chunk_id=chunk_id,
        doc_id="arxiv:2403.01182",
        doc_title="Valid Title",
        source_uri="https://arxiv.org/abs/2403.01182",
        section_heading="Empirical Results",
        verbatim_quote="Delivery friction of 15.0 bps satisfies",
    )

    # 1. Exact 30.0% drawdown passes
    brief_dd_30 = _make_brief(
        strategy_name="Boundary DD 30",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        transaction_cost_bps_assumed=15.0,
        reported_sharpe_ratio=1.5,
        reported_max_drawdown_pct=30.0,
        verified_citations=[valid_citation],
    )
    res_30 = gate.evaluate(brief_dd_30, [doc_chunk])
    assert res_30.passed is True
    assert res_30.friction_verdict == FrictionVerdict.REALISTIC_FRICTION

    # 2. 30.01% drawdown fails institutional threshold
    brief_dd_30_01 = _make_brief(
        strategy_name="Boundary DD 30.01",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        transaction_cost_bps_assumed=15.0,
        reported_sharpe_ratio=1.5,
        reported_max_drawdown_pct=30.01,
        verified_citations=[valid_citation],
    )
    res_30_01 = gate.evaluate(brief_dd_30_01, [doc_chunk])
    assert res_30_01.passed is False
    assert res_30_01.friction_verdict == FrictionVerdict.FAILS_AFTER_COSTS
    assert "exceeds 30.0% institutional risk threshold" in res_30_01.reason

    # 3. Negative drawdown notation: -30.01% also fails
    brief_dd_neg_30_01 = _make_brief(
        strategy_name="Boundary DD -30.01",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        transaction_cost_bps_assumed=15.0,
        reported_sharpe_ratio=1.5,
        reported_max_drawdown_pct=-30.01,
        verified_citations=[valid_citation],
    )
    res_neg_30_01 = gate.evaluate(brief_dd_neg_30_01, [doc_chunk])
    assert res_neg_30_01.passed is False
    assert res_neg_30_01.friction_verdict == FrictionVerdict.FAILS_AFTER_COSTS

    # 4. Net Sharpe penalty: gross Sharpe 0.55 with friction deficit collapses net Sharpe <= 0.5
    brief_low_sharpe = _make_brief(
        strategy_name="Low Sharpe with Deficit",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        transaction_cost_bps_assumed=5.0,  # 10 bps deficit vs 15 bps floor
        reported_sharpe_ratio=0.55,
        reported_max_drawdown_pct=15.0,
        verified_citations=[valid_citation],
    )
    res_low_sharpe = gate.evaluate(brief_low_sharpe, [doc_chunk])
    assert res_low_sharpe.passed is False
    # Deficit triggers warning or failure after costs
    assert res_low_sharpe.friction_verdict in (
        FrictionVerdict.UNVERIFIED_FRICTION_WARNING,
        FrictionVerdict.FAILS_AFTER_COSTS,
    )
