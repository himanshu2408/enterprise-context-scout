"""Milestone 1 deterministic test suite: Foundation, RLS security, RRF math & Contracts."""

from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from context_scout.agents.domain_registry import AdapterPool, DomainRegistry
from context_scout.config import ScoutSettings, validate_google_model_id
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
    InMemoryRLSRepository,
)
from context_scout.schemas.contracts import (
    CodeSlicePointer,
    DocCitationSpan,
    FrictionVerdict,
    MarketRegime,
    ModificationIntent,
    PrincipalContext,
    QuantStrategyBrief,
    RoleType,
    compute_normalized_rrf,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_google_only_model_policy_enforcement() -> None:
    """Rule 2: Only Oct 2026 Google models are allowed; non-Google & stale models fail."""
    assert validate_google_model_id("google/gemma-4-26B-A4B-it") == "google/gemma-4-26B-A4B-it"
    assert validate_google_model_id("google/gemma-4-E4B-it") == "google/gemma-4-E4B-it"
    assert validate_google_model_id("google/embeddinggemma-2") == "google/embeddinggemma-2"
    assert validate_google_model_id("gemini-3.8-flash") == "gemini-3.8-flash"

    for prohibited in [
        "gpt-4o",
        "claude-3-5-sonnet",
        "Qwen/Qwen3-Coder-30B-A3B",
        "text-embedding-004",
        "google/gemma-2-9b-it",
        "google/gemma-3-12b-it",
        "gemini-ultra",
    ]:
        with pytest.raises(ValueError):
            validate_google_model_id(prohibited)

    settings = ScoutSettings()
    assert settings.primary_model_id == "google/gemma-4-26B-A4B-it"
    assert settings.local_max_kv_tokens == 32_768


def test_blk01_normalized_rrf_score_range_and_threshold() -> None:
    """BLK-01: Normalized RRF scales rank #1 across 3 modalities to 1.0 (passing >=0.55 gate)."""
    top_score = compute_normalized_rrf([1, 1, 1], num_modalities=3, k=60)
    assert top_score == 1.0
    assert top_score >= 0.55

    rank_10_all = compute_normalized_rrf([10, 10, 10], num_modalities=3, k=60)
    assert 0.85 <= rank_10_all < 1.0

    single_modality_rank_1 = compute_normalized_rrf([1, None, None], num_modalities=3, k=60)
    assert pytest.approx(single_modality_rank_1, rel=1e-4) == 1.0 / 3.0
    assert single_modality_rank_1 < 0.55


def test_fin01_market_regime_quant_scepticism_gate() -> None:
    """FIN-01: Enforces market friction floors (IN_NSE >=15 bps delivery, DE_XETRA >=10 bps)."""
    citation = DocCitationSpan(
        chunk_id="chk-1",
        doc_id="arxiv:2502.08192",
        doc_title="Order Flow and Momentum in Indian Equities",
        source_uri="https://arxiv.org/abs/2502.08192",
        section_heading="4. Empirical Backtest & Transaction Costs",
        verbatim_quote="Net Sharpe ratio after 18 bps round-trip brokerage and STT is 1.42.",
    )

    # 1. India NSE equity delivery with only 8 bps (< 15 bps floor) -> forced warning
    under_cost_nse = QuantStrategyBrief(
        strategy_name="NSE Opening Range Momentum",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        is_intraday_futures=False,
        mathematical_signal_latex=r"S_t = \frac{P_t - P_{open}}{\sigma_{20}}",
        entry_exit_rules=["Buy when S_t > 1.5", "Exit at 15:15 IST"],
        sample_period_years="2018-2025",
        transaction_cost_bps_assumed=8.0,
        friction_verdict=FrictionVerdict.REALISTIC_FRICTION,
        market_specific_caveats=["Ignores 10 bps delivery STT per leg"],
        known_failure_regimes=["Low-volume monsoon sideways sessions"],
        verified_citations=[citation],
    )
    assert under_cost_nse.friction_verdict == FrictionVerdict.UNVERIFIED_FRICTION_WARNING

    # 2. India NSE equity delivery with 18 bps (>= 15 bps floor) -> passes as realistic
    realistic_nse = under_cost_nse.model_copy(
        update={
            "transaction_cost_bps_assumed": 18.0,
            "friction_verdict": FrictionVerdict.REALISTIC_FRICTION,
        }
    )
    validated_nse = QuantStrategyBrief.model_validate(realistic_nse.model_dump())
    assert validated_nse.friction_verdict == FrictionVerdict.REALISTIC_FRICTION

    # 3. Germany Xetra with 6 bps (< 10 bps floor) -> forced warning
    under_cost_xetra = QuantStrategyBrief(
        strategy_name="DAX Closing Auction Imbalance",
        target_markets=[MarketRegime.GERMANY_XETRA_DAX],
        mathematical_signal_latex=r"I_t = \frac{V_{bid} - V_{ask}}{V_{bid} + V_{ask}}",
        entry_exit_rules=["Enter 5m before Xetra closing auction"],
        sample_period_years="2019-2025",
        transaction_cost_bps_assumed=6.0,
        friction_verdict=FrictionVerdict.REALISTIC_FRICTION,
        market_specific_caveats=["Xetra MDAX spread wider than 6 bps"],
        known_failure_regimes=["ECB rate announcement days"],
        verified_citations=[citation],
    )
    assert under_cost_xetra.friction_verdict == FrictionVerdict.UNVERIFIED_FRICTION_WARNING


def test_domain_registry_and_adapter_pool_isolation() -> None:
    """ARC-02 & HW-01: Verifies declarative YAML loading, AdapterPool pinning, and 32K KV cap."""
    with pytest.raises(ValueError, match="exceeds Apple M5 Pro 48GB safety cap"):
        AdapterPool(max_kv_tokens=65_536)

    registry = DomainRegistry(REPO_ROOT / "domains")
    manifests = registry.load_all()
    assert set(manifests.keys()) == {"trading_code_scout", "quant_research_scout"}

    code_slot = registry.adapter_pool.get_slot("trading_code_scout")
    quant_slot = registry.adapter_pool.get_slot("quant_research_scout")
    assert code_slot.adapter_id == "gemma4_26b_code_scout"
    assert quant_slot.adapter_id == "gemma4_26b_quant_research"
    assert code_slot.max_kv_tokens == 32_768

    assert registry.is_mcp_tool_allowed("trading_code_scout", "code_repo_mcp:search_ast_symbols")
    assert not registry.is_mcp_tool_allowed(
        "trading_code_scout", "quant_corpus_mcp:search_papers_hybrid"
    )


def test_sql_schema_enforces_rls_on_all_tables_including_ast_edges() -> None:
    """SEC-01 & SEC-02: Verifies 001_init_schema.sql has FORCE RLS & scout_app_user."""
    sql_text = (REPO_ROOT / "sql" / "001_init_schema.sql").read_text(encoding="utf-8")

    for table in ("code_files", "ast_symbols", "ast_edges", "doc_chunks"):
        assert f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;" in sql_text
        assert f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY;" in sql_text

    assert "CREATE ROLE scout_app_user" in sql_text
    assert "NOSUPERUSER" in sql_text
    assert "CREATE POLICY rls_ast_edges_scope ON ast_edges" in sql_text


def test_rls_zero_leakage_and_hybrid_rrf_retrieval() -> None:
    """Verifies product_manager cannot leak quant_restricted files, symbols, edges, or docs."""
    repo = InMemoryRLSRepository()

    quant_principal = PrincipalContext(
        user_id="usr_quant_1",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["public_eng", "quant_restricted"],
    )
    pm_principal = PrincipalContext(
        user_id="usr_pm_1",
        role=RoleType.PRODUCT_MANAGER,
        allowed_scopes=["public_eng"],
    )

    pub_file_id = uuid4()
    sec_file_id = uuid4()
    pub_sym_id = uuid4()
    sec_sym_id = uuid4()

    repo.insert_code_file(
        quant_principal,
        CodeFileRecord(
            file_id=pub_file_id,
            domain_id="trading_code_scout",
            repo_name="trading-core",
            file_path="src/execution/order_router.py",
            git_commit_sha="a1b2c3d4",
            git_blob_sha="f0e1d2c3",
            language="python",
            allowed_scopes=["public_eng"],
            raw_content="def route_order(order): pass\n",
            line_count=1,
        ),
    )
    repo.insert_code_file(
        quant_principal,
        CodeFileRecord(
            file_id=sec_file_id,
            domain_id="trading_code_scout",
            repo_name="trading-core",
            file_path="src/alpha/proprietary_stat_arb.py",
            git_commit_sha="a1b2c3d4",
            git_blob_sha="99887766",
            language="python",
            allowed_scopes=["quant_restricted"],
            raw_content="def compute_secret_alpha_weights(): pass\n",
            line_count=1,
        ),
    )

    emb_router = [1.0, 0.2, 0.0, 0.0] + [0.0] * 764
    emb_secret = [0.0, 1.0, 0.0, 0.0] + [0.0] * 764

    repo.insert_ast_symbol(
        quant_principal,
        ASTSymbolRecord(
            symbol_id=pub_sym_id,
            file_id=pub_file_id,
            domain_id="trading_code_scout",
            qualified_name="trading.execution.order_router.route_order",
            symbol_kind="function",
            start_line=1,
            end_line=1,
            signature_skeleton="def route_order(order: Order) -> ExecutionReceipt:",
            contextual_summary="Routes standard orders to exchange adapters.",
            allowed_scopes=["public_eng"],
            embedding=emb_router,
        ),
    )
    repo.insert_ast_symbol(
        quant_principal,
        ASTSymbolRecord(
            symbol_id=sec_sym_id,
            file_id=sec_file_id,
            domain_id="trading_code_scout",
            qualified_name="trading.alpha.proprietary_stat_arb.compute_secret_alpha_weights",
            symbol_kind="function",
            start_line=1,
            end_line=1,
            signature_skeleton="def compute_secret_alpha_weights() -> dict[str, float]:",
            contextual_summary="Proprietary intraday stat-arb signal weights.",
            allowed_scopes=["quant_restricted"],
            embedding=emb_secret,
        ),
    )

    # SEC-02: Insert restricted AST call edge
    repo.insert_ast_edge(
        quant_principal,
        ASTEdgeRecord(
            edge_id=uuid4(),
            domain_id="trading_code_scout",
            source_symbol_id=sec_sym_id,
            target_qualified_name="trading.alpha.secret_kelly.HiddenKellyMatrix",
            target_symbol_id=None,
            edge_kind="USES_TYPE",
            allowed_scopes=["quant_restricted"],
        ),
    )

    # Insert restricted DocChunk
    repo.insert_doc_chunk(
        quant_principal,
        DocChunkRecord(
            chunk_id=uuid4(),
            domain_id="quant_research_scout",
            doc_id="internal:alpha-memo-01",
            doc_title="Proprietary NSE Intraday Alpha Weights",
            source_uri="internal://alpha-memo-01",
            market_scope=MarketRegime.INDIA_NSE_BSE,
            published_date=date(2026, 9, 1),
            authority_weight=1.0,
            section_heading="1. Confidential Signal Weights",
            contextual_header="[Context: Restricted internal quant note]",
            verbatim_content="Proprietary weight vector w = [0.42, -0.31, 0.19]",
            metadata_json={"restricted": True},
            allowed_scopes=["quant_restricted"],
            embedding=emb_secret,
        ),
    )

    # Product manager must see ONLY public_eng files, 0 restricted edges, and 0 restricted docs
    pm_files = repo.list_visible_code_files(pm_principal, "trading_code_scout")
    assert [f.file_path for f in pm_files] == ["src/execution/order_router.py"]
    assert repo.list_visible_ast_edges(pm_principal, "trading_code_scout") == []
    assert repo.list_visible_doc_chunks(pm_principal, "quant_research_scout") == []

    # Even if PM searches directly for "compute_secret_alpha_weights", 0 restricted hits leak
    pm_hits = repo.search_symbols_hybrid_rrf(
        principal=pm_principal,
        domain_id="trading_code_scout",
        query_text="compute_secret_alpha_weights",
        query_embedding=emb_secret,
    )
    assert len(pm_hits) == 1
    assert "secret" not in pm_hits[0].qualified_name_or_title

    # Quant researcher with quant_restricted scope gets the restricted symbol at Rank #1 (RRF=1.0)
    quant_hits = repo.search_symbols_hybrid_rrf(
        principal=quant_principal,
        domain_id="trading_code_scout",
        query_text="compute_secret_alpha_weights",
        query_embedding=emb_secret,
    )
    assert quant_hits[0].qualified_name_or_title.endswith("compute_secret_alpha_weights")
    assert quant_hits[0].normalized_rrf_score == 1.0


def test_code_slice_pointer_validation() -> None:
    """Verifies CodeSlicePointer rejects inverted line ranges."""
    with pytest.raises(ValidationError):
        CodeSlicePointer(
            repo_name="trading-core",
            file_path="src/engine.py",
            start_line=50,
            end_line=20,
            qualified_symbol="engine.run",
            selection_reason="Target function",
            modification_intent=ModificationIntent.MODIFY_TARGET,
        )
