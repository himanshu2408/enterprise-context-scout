"""Unit tests for Milestone 3 (Quant Research Harvester - Model 2 quant_research_scout).

Verifies paper harvesting, LaTeX/Markdown lossless parsing, 50-token situated headers,
3-way normalized RRF retrieval, market friction floors, and verbatim citation verification.
"""

import uuid
from datetime import date

import pytest

from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer
from context_scout.db.repositories import (
    DocChunkRecord,
    InMemoryRLSRepository,
)
from context_scout.ingestion.contextualizer import (
    DeterministicContextualizer,
    build_situated_context_prompt,
    clean_situated_header,
)
from context_scout.ingestion.doc_parser import (
    DocumentChunk,
    LatexDocParser,
    MarkdownTableChunker,
)
from context_scout.ingestion.git_merkle_indexer import generate_deterministic_embedding
from context_scout.ingestion.paper_harvester import (
    MARKET_QUERY_PRESETS,
    HarvesterSource,
    MockPaperHarvesterClient,
    build_arxiv_query_url,
    build_openalex_query_url,
    parse_arxiv_atom_feed,
    parse_openalex_response,
)
from context_scout.ingestion.scepticism_gate import (
    MarketRegimeScepticismGate,
)
from context_scout.schemas.contracts import (
    DocCitationSpan,
    FrictionVerdict,
    MarketRegime,
    PrincipalContext,
    QuantStrategyBrief,
    RoleType,
)


def test_paper_harvester_market_queries() -> None:
    """Verifies query construction, URL formatting, and mock fixtures across IN, DE, and US."""
    # 1. Indian NSE query preset
    nse_query = MARKET_QUERY_PRESETS[MarketRegime.INDIA_NSE_BSE]
    assert nse_query.market_scope == MarketRegime.INDIA_NSE_BSE
    assert "q-fin.TR" in nse_query.arxiv_categories
    assert "National Stock Exchange" in nse_query.keywords
    arxiv_url = build_arxiv_query_url(nse_query)
    assert "export.arxiv.org/api/query" in arxiv_url
    assert "cat%3Aq-fin.TR" in arxiv_url or "cat:q-fin.TR" in arxiv_url

    openalex_url = build_openalex_query_url(nse_query)
    assert "api.openalex.org/works" in openalex_url
    assert "from_publication_date" in openalex_url

    # 2. German Xetra query preset
    de_query = MARKET_QUERY_PRESETS[MarketRegime.GERMANY_XETRA_DAX]
    assert "Xetra" in de_query.keywords
    assert "Deutsche Börse" in de_query.openalex_concepts

    # 3. US equity query preset
    us_query = MARKET_QUERY_PRESETS[MarketRegime.US_NYSE_NASDAQ]
    assert "NYSE" in us_query.keywords
    assert "Almgren-Chriss" in us_query.keywords

    # 4. Mock client delivery
    client = MockPaperHarvesterClient()
    nse_papers = client.fetch_papers(nse_query)
    assert len(nse_papers) >= 1
    assert nse_papers[0].market_scope == MarketRegime.INDIA_NSE_BSE
    assert nse_papers[0].source == HarvesterSource.ARXIV
    assert "STT" in nse_papers[0].abstract or "Securities Transaction Tax" in nse_papers[0].title


def test_parse_arxiv_atom_feed_xml() -> None:
    """Verifies robust parsing of arXiv Atom XML responses."""
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2502.08192v2</id>
        <title> Deep Order Flow Dynamics in Modern Equity Exchanges </title>
        <summary> Comprehensive empirical analysis of limit order book mechanics. </summary>
        <published>2025-02-12T18:00:00Z</published>
        <author><name>Vikram Sethi</name></author>
        <author><name>Elena Rostova</name></author>
        <link href="http://arxiv.org/pdf/2502.08192v2" title="pdf" type="application/pdf"/>
      </entry>
    </feed>
    """
    papers = parse_arxiv_atom_feed(sample_xml, MarketRegime.INDIA_NSE_BSE)
    assert len(papers) == 1
    paper = papers[0]
    assert paper.doc_id == "arxiv:2502.08192"  # Version stripped
    assert paper.title == "Deep Order Flow Dynamics in Modern Equity Exchanges"
    assert paper.authors == ["Vikram Sethi", "Elena Rostova"]
    assert paper.published_date == date(2025, 2, 12)
    assert paper.market_scope == MarketRegime.INDIA_NSE_BSE


def test_parse_openalex_response_json() -> None:
    """Verifies parsing of OpenAlex REST API inverted index abstracts."""
    data: dict[str, object] = {
        "results": [
            {
                "id": "https://openalex.org/W4389201",
                "title": "Cross-Impact and Order Imbalance in German Equity Markets",
                "publication_date": "2024-05-18",
                "authorships": [
                    {"author": {"display_name": "Maximilian Bauer"}},
                    {"author": {"display_name": "Sophia Becker"}},
                ],
                "abstract_inverted_index": {
                    "We": [0],
                    "study": [1],
                    "liquidity": [2],
                    "on": [3],
                    "Xetra.": [4],
                },
                "doi": "https://doi.org/10.1016/j.jfineco.2024.103",
            }
        ]
    }
    papers = parse_openalex_response(data, MarketRegime.GERMANY_XETRA_DAX)
    assert len(papers) == 1
    p = papers[0]
    assert p.doc_id == "openalex:W4389201"
    assert p.title == "Cross-Impact and Order Imbalance in German Equity Markets"
    assert p.abstract == "We study liquidity on Xetra."
    assert p.authors == ["Maximilian Bauer", "Sophia Becker"]


def test_latex_parser_formula_and_table_preservation() -> None:
    """Verifies that LaTeX equations and tables are never split across chunk boundaries."""
    raw_tex = r"""\documentclass{article}
\begin{document}
\section{1. Microstructure and Signal}
Consider the continuous price process $P_t$ with midpoint quote $M_t$.
The Almgren-Chriss execution cost equation is specified as:
\begin{equation}
x_k = x_0 - \sum_{j=1}^{k} n_j, \quad \text{with} \quad \tau_j = t_j - t_{j-1}
\end{equation}
This formula models the deterministic liquidation trajectory under transient price impact.

\subsection{1.1 Order Book Multi-Line Invariants}
We also formulate the cross-asset spread imbalance:
\begin{align}
\Delta M_{i,t} &= \alpha_i \text{OFI}_{i,t}
+ \sum_{j \ne i} \gamma_{ij} \text{OFI}_{j,t} + \epsilon_{i,t} \\
\mathbb{E}[\epsilon_{i,t}] &= 0
\end{align}
Notice that multi-line align environments must remain completely contiguous in the output chunk.

\section{2. Empirical Transaction Cost Schedule}
Table 1 specifies the transaction cost components for institutional equity orders.
\begin{table}[h]
\begin{tabular}{|l|r|r|}
\hline
Fee Category & Delivery (bps) & Futures (bps) \\
\hline
Securities Transaction Tax & 10.0 & 1.25 \\
Exchange Turnover Fee & 3.25 & 1.90 \\
GST and Stamp Duty & 1.75 & 0.85 \\
\hline
Total Statutory Floor & 15.0 & 4.00 \\
\hline
\end{tabular}
\end{table}
The tabular schedule confirms the 15 bps statutory threshold for Indian cash delivery.
\end{document}
"""
    parser = LatexDocParser()
    sections = parser.parse_latex(raw_tex)
    assert len(sections) >= 2

    # Chunk the document into 400-800 token chunks
    chunks = parser.chunk_paper(
        doc_id="arxiv:2403.01182",
        doc_title="Almgren-Chriss Execution under Indian Market Friction",
        source_uri="https://arxiv.org/abs/2403.01182",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        sections=sections,
        min_chunk_tokens=50,
        max_chunk_tokens=300,
    )

    assert len(chunks) >= 2
    for chunk in chunks:
        # Verify no equation is split
        if r"\begin{equation}" in chunk.verbatim_content:
            assert r"\end{equation}" in chunk.verbatim_content
        if r"\begin{align}" in chunk.verbatim_content:
            assert r"\end{align}" in chunk.verbatim_content
        if r"\begin{table}" in chunk.verbatim_content:
            assert r"\end{table}" in chunk.verbatim_content
        if r"\begin{tabular}" in chunk.verbatim_content:
            assert r"\end{tabular}" in chunk.verbatim_content

    # Breadcrumb verification
    sec_headings = [c.section_heading for c in chunks]
    assert any("1. Microstructure and Signal" in h for h in sec_headings)


def test_markdown_table_chunker_keeps_tables_intact() -> None:
    """Verifies that Markdown pipe tables are never split mid-table."""
    md_text = """# Indian Equities Execution Model

Empirical execution analysis across liquid equities.

| Market | Asset | Spread (bps) | STT (bps) | Total Floor (bps) |
|---|---|---|---|---|
| NSE | NIFTY50 | 2.5 | 10.0 | 15.0 |
| BSE | SENSEX | 3.0 | 10.0 | 15.5 |
| Eurex | DAX Futures | 1.0 | 0.0 | 4.0 |
| NYSE | SPY ETF | 0.5 | 0.0 | 5.0 |

The above matrix defines institutional trading minimums.
"""
    chunker = MarkdownTableChunker()
    chunks = chunker.parse_and_chunk(
        doc_id="doc:md_matrix",
        doc_title="Market Execution Matrix",
        source_uri="https://example.com/matrix",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        markdown_content=md_text,
        min_chunk_tokens=20,
        max_chunk_tokens=100,
    )
    assert len(chunks) >= 1
    # Check that table exists intact in one chunk
    table_chunk = next(c for c in chunks if "| NIFTY50 |" in c.verbatim_content)
    assert "| SENSEX |" in table_chunk.verbatim_content
    assert "| SPY ETF |" in table_chunk.verbatim_content


def test_contextual_retrieval_situated_headers() -> None:
    """Verifies 50-token situated context header generation and prompt formatting."""
    chunk = DocumentChunk(
        chunk_id=uuid.uuid4(),
        doc_id="arxiv:2403.01182",
        doc_title="Momentum and Reversal Dynamics in Indian Equities with STT",
        source_uri="https://arxiv.org/abs/2403.01182",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        section_heading="4. Empirical Results > 4.2 Impact of Indian STT",
        verbatim_content=(
            "Gross Sharpe is 1.45, but factoring in 15 bps STT delivery drops Sharpe to 0.92."
        ),
        formulas_latex=[],
        tables_markdown=[],
        token_count_approx=450,
        metadata_json={
            "year": 2024,
            "sample_period": "2018-2023",
            "strategy_type": "Cross-Sectional Momentum",
            "assumed_friction_bps": 15.0,
        },
    )

    contextualizer = DeterministicContextualizer()
    header = contextualizer.generate_header(doc_abstract="Abstract content...", chunk=chunk)

    assert header.startswith("[Context:")
    assert "arxiv:2403.01182" in header
    assert "IN_NSE" in header
    assert "2018-2023" in header
    assert "15.0 bps floor" in header or "15" in header

    # Test prompt formatter for Gemma 4 turn template
    prompt = build_situated_context_prompt(
        doc_title=chunk.doc_title,
        doc_id=chunk.doc_id,
        doc_abstract="Abstract...",
        market_scope="IN_NSE",
        section_heading=chunk.section_heading,
        verbatim_content=chunk.verbatim_content,
    )
    assert "<start_of_turn>user" in prompt
    assert "<start_of_turn>model" in prompt
    assert "google/gemma-4-E4B-it" not in prompt  # Model ID not in prompt text

    # Test cleaner
    dirty = (
        "   Here is the header:\n[Context: arxiv:1234 (2024) | Market: IN_NSE]   \n"
        "Hope this helps! "
    )
    cleaned = clean_situated_header(dirty)
    assert cleaned == "[Context: arxiv:1234 (2024) | Market: IN_NSE]"


def test_doc_chunks_3way_normalized_rrf_scoring() -> None:
    """Verifies that 3-way RRF fuses BM25, dense vector, and trigram/authority scores."""
    principal = PrincipalContext(
        user_id="quant_trader_1",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["quant_public", "quant_in"],
    )
    repo = InMemoryRLSRepository()

    emb_1 = generate_deterministic_embedding("Indian STT securities transaction tax momentum", 768)
    emb_2 = generate_deterministic_embedding("German DAX Xetra liquidity auction tick size", 768)
    emb_3 = generate_deterministic_embedding("US limit order book Almgren Chriss execution", 768)

    chunk_1 = DocChunkRecord(
        chunk_id=uuid.uuid4(),
        domain_id="quant_research_scout",
        doc_id="arxiv:2403.01182",
        doc_title="Momentum with Indian STT",
        source_uri="https://arxiv.org/abs/2403.01182",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        published_date=date(2024, 3, 15),
        authority_weight=1.5,
        section_heading="Methodology",
        contextual_header=(
            "[Context: arxiv:2403.01182 (2024) | Market: IN_NSE | Friction: 15 bps]"
        ),
        verbatim_content=(
            "Securities Transaction Tax (STT) requires 15 bps delivery friction on NSE."
        ),
        metadata_json={"sample_period": "2018-2023"},
        allowed_scopes=["quant_public"],
        embedding=emb_1,
    )
    chunk_2 = DocChunkRecord(
        chunk_id=uuid.uuid4(),
        domain_id="quant_research_scout",
        doc_id="arxiv:2311.08941",
        doc_title="Xetra DAX Liquidity",
        source_uri="https://arxiv.org/abs/2311.08941",
        market_scope=MarketRegime.GERMANY_XETRA_DAX,
        published_date=date(2023, 11, 20),
        authority_weight=1.2,
        section_heading="Spread Estimation",
        contextual_header=(
            "[Context: arxiv:2311.08941 (2023) | Market: DE_XETRA | Friction: 10 bps]"
        ),
        verbatim_content="Deutsche Börse Xetra closing auction dynamics under MiFID II tick sizes.",
        metadata_json={"sample_period": "2019-2023"},
        allowed_scopes=["quant_public"],
        embedding=emb_2,
    )
    chunk_3 = DocChunkRecord(
        chunk_id=uuid.uuid4(),
        domain_id="quant_research_scout",
        doc_id="arxiv:2401.05432",
        doc_title="US Order Flow Imbalance",
        source_uri="https://arxiv.org/abs/2401.05432",
        market_scope=MarketRegime.US_NYSE_NASDAQ,
        published_date=date(2024, 1, 10),
        authority_weight=1.0,
        section_heading="Order Flow Imbalance",
        contextual_header=(
            "[Context: arxiv:2401.05432 (2024) | Market: US_EQUITY | Friction: 5 bps]"
        ),
        verbatim_content=(
            "Almgren-Chriss institutional order execution across NYSE and NASDAQ equities."
        ),
        metadata_json={"sample_period": "2020-2023"},
        allowed_scopes=["quant_public"],
        embedding=emb_3,
    )

    repo.insert_doc_chunks_batch(principal, [chunk_1, chunk_2, chunk_3])

    # Search with Indian query
    query_text = "Securities Transaction Tax STT delivery NSE"
    hits = repo.search_doc_chunks_hybrid_rrf(
        principal=principal,
        domain_id="quant_research_scout",
        query_text=query_text,
        query_embedding=emb_1,
        market_scope=MarketRegime.INDIA_NSE_BSE,
        top_k=3,
    )

    assert len(hits) == 1
    hit = hits[0]
    assert hit.item_id == chunk_1.chunk_id
    # Mathematical range check [0.0, 1.0]
    assert 0.0 <= hit.normalized_rrf_score <= 1.0
    assert hit.normalized_rrf_score >= 0.55  # Passes pre-flight gate threshold
    assert hit.bm25_rank == 1
    assert hit.vector_rank == 1
    assert hit.trigram_rank == 1
    assert hit.normalized_rrf_score == 1.0  # Rank #1 across all 3 yields 1.0


def test_market_regime_scepticism_gate_nse_friction() -> None:
    """Verifies Indian equity friction gate: 8 bps flagged vs 15 bps realistic."""
    gate = MarketRegimeScepticismGate()
    chunk_content = (
        "Securities Transaction Tax (STT) requires at least 15 bps round-trip friction. "
        "Gross returns of 22.4% reduce to 14.8% net after factoring in statutory STT and GST."
    )
    chunk = DocumentChunk(
        chunk_id=uuid.uuid4(),
        doc_id="arxiv:2403.01182",
        doc_title="Indian Momentum with STT",
        source_uri="https://arxiv.org/abs/2403.01182",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        section_heading="Empirical Results",
        verbatim_content=chunk_content,
        formulas_latex=[],
        tables_markdown=[],
        token_count_approx=100,
    )

    # 1. Unrealistic 8 bps delivery friction -> UNVERIFIED_FRICTION_WARNING
    unrealistic_brief = QuantStrategyBrief(
        strategy_name="NSE Momentum",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        is_intraday_futures=False,
        mathematical_signal_latex="S_{i,t} = \\sum_{k=2}^{12} R_{i,t-k}",
        entry_exit_rules=["Enter top decile at month end", "Rebalance monthly"],
        sample_period_years="2018-2023",
        reported_sharpe_ratio=1.45,
        reported_max_drawdown_pct=-18.2,
        transaction_cost_bps_assumed=8.0,  # Below 15.0 bps floor
        friction_verdict=FrictionVerdict.UNVERIFIED_FRICTION_WARNING,
        market_specific_caveats=["Omits STT 10 bps per delivery leg"],
        known_failure_regimes=["High volatility sideways markets"],
        verified_citations=[
            DocCitationSpan(
                chunk_id=str(chunk.chunk_id),
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                source_uri=chunk.source_uri,
                section_heading=chunk.section_heading,
                verbatim_quote=(
                    "Securities Transaction Tax (STT) requires at least "
                    "15 bps round-trip friction."
                ),
            )
        ],
    )
    res_unrealistic = gate.evaluate(unrealistic_brief, [chunk])
    assert not res_unrealistic.passed
    assert res_unrealistic.friction_verdict == FrictionVerdict.UNVERIFIED_FRICTION_WARNING
    assert res_unrealistic.effective_friction_floor_bps == 15.0
    assert res_unrealistic.citations_valid is True

    # 2. Realistic 15 bps delivery friction -> REALISTIC_FRICTION
    realistic_brief = QuantStrategyBrief(
        strategy_name="NSE Momentum",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        is_intraday_futures=False,
        mathematical_signal_latex="S_{i,t} = \\sum_{k=2}^{12} R_{i,t-k}",
        entry_exit_rules=["Enter top decile at month end", "Rebalance monthly"],
        sample_period_years="2018-2023",
        reported_sharpe_ratio=1.45,
        reported_max_drawdown_pct=-18.2,
        transaction_cost_bps_assumed=15.0,  # Meets 15.0 bps floor
        friction_verdict=FrictionVerdict.REALISTIC_FRICTION,
        market_specific_caveats=["Includes 15 bps STT, turnover charges, and GST"],
        known_failure_regimes=["High volatility sideways markets"],
        verified_citations=[
            DocCitationSpan(
                chunk_id=str(chunk.chunk_id),
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                source_uri=chunk.source_uri,
                section_heading=chunk.section_heading,
                verbatim_quote=(
                    "Securities Transaction Tax (STT) requires at least "
                    "15 bps round-trip friction."
                ),
            )
        ],
    )
    res_realistic = gate.evaluate(realistic_brief, [chunk])
    assert res_realistic.passed is True
    assert res_realistic.friction_verdict == FrictionVerdict.REALISTIC_FRICTION

    # 3. Intraday futures at 5 bps -> REALISTIC_FRICTION
    intraday_brief = QuantStrategyBrief(
        strategy_name="NSE Futures Intraday",
        target_markets=[MarketRegime.INDIA_NSE_BSE],
        is_intraday_futures=True,
        mathematical_signal_latex="S_{t} = OFI_t",
        entry_exit_rules=["Square off before 15:15 IST"],
        sample_period_years="2020-2023",
        reported_sharpe_ratio=1.8,
        reported_max_drawdown_pct=-12.0,
        transaction_cost_bps_assumed=5.0,  # 5 bps floor for futures
        friction_verdict=FrictionVerdict.REALISTIC_FRICTION,
        market_specific_caveats=["Intraday futures STT 1.25 bps"],
        known_failure_regimes=["Opening gap up/down"],
        verified_citations=[
            DocCitationSpan(
                chunk_id=str(chunk.chunk_id),
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                source_uri=chunk.source_uri,
                section_heading=chunk.section_heading,
                verbatim_quote=(
                    "Gross returns of 22.4% reduce to 14.8% net after factoring in statutory "
                    "STT and GST."
                ),
            )
        ],
    )
    res_intraday = gate.evaluate(intraday_brief, [chunk])
    assert res_intraday.passed is True
    assert res_intraday.friction_verdict == FrictionVerdict.REALISTIC_FRICTION
    assert res_intraday.effective_friction_floor_bps == 5.0


def test_market_regime_scepticism_gate_de_xetra_friction() -> None:
    """Verifies German Xetra friction gate: 6 bps flagged vs 10 bps realistic."""
    gate = MarketRegimeScepticismGate()
    chunk_content = (
        "Xetra order execution incurs an effective round-trip rebalancing friction of 10 bps."
    )
    chunk = DocumentChunk(
        chunk_id=uuid.uuid4(),
        doc_id="arxiv:2311.08941",
        doc_title="Xetra Microstructure",
        source_uri="https://arxiv.org/abs/2311.08941",
        market_scope=MarketRegime.GERMANY_XETRA_DAX,
        section_heading="Market Microstructure",
        verbatim_content=chunk_content,
        formulas_latex=[],
        tables_markdown=[],
        token_count_approx=80,
    )

    # 6 bps assumes too little spread/clearing on Xetra DAX
    brief_6bps = QuantStrategyBrief(
        strategy_name="DAX Imbalance",
        target_markets=[MarketRegime.GERMANY_XETRA_DAX],
        mathematical_signal_latex="S_{eff} = \\frac{|P - M|}{M}",
        entry_exit_rules=["Enter auction imbalance"],
        sample_period_years="2019-2023",
        reported_sharpe_ratio=1.2,
        reported_max_drawdown_pct=-14.0,
        transaction_cost_bps_assumed=6.0,  # Below 10 bps floor
        friction_verdict=FrictionVerdict.UNVERIFIED_FRICTION_WARNING,
        market_specific_caveats=["Omits MiFID II tick size clearing and short borrow costs"],
        known_failure_regimes=["Low liquidity auctions"],
        verified_citations=[
            DocCitationSpan(
                chunk_id=str(chunk.chunk_id),
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                source_uri=chunk.source_uri,
                section_heading=chunk.section_heading,
                verbatim_quote=chunk_content,
            )
        ],
    )
    res = gate.evaluate(brief_6bps, [chunk])
    assert not res.passed
    assert res.friction_verdict == FrictionVerdict.UNVERIFIED_FRICTION_WARNING
    assert res.effective_friction_floor_bps == 10.0


def test_market_regime_scepticism_gate_us_equity_friction() -> None:
    """Verifies US equity friction gate: 3 bps flagged vs 5 bps realistic."""
    gate = MarketRegimeScepticismGate()
    chunk_content = "Execution friction in liquid US large caps incurs effective cost of 5.0 bps."
    chunk = DocumentChunk(
        chunk_id=uuid.uuid4(),
        doc_id="arxiv:2401.05432",
        doc_title="US OFI",
        source_uri="https://arxiv.org/abs/2401.05432",
        market_scope=MarketRegime.US_NYSE_NASDAQ,
        section_heading="Order Flow Imbalance",
        verbatim_content=chunk_content,
        formulas_latex=[],
        tables_markdown=[],
        token_count_approx=60,
    )

    # 3 bps assumes sub-spread execution with zero regulatory fee
    brief_3bps = QuantStrategyBrief(
        strategy_name="US OFI",
        target_markets=[MarketRegime.US_NYSE_NASDAQ],
        mathematical_signal_latex="OFI_t = q_t^b - q_t^a",
        entry_exit_rules=["Cross spread on signal"],
        sample_period_years="2020-2023",
        reported_sharpe_ratio=1.6,
        reported_max_drawdown_pct=-11.0,
        transaction_cost_bps_assumed=3.0,  # Below 5 bps floor
        friction_verdict=FrictionVerdict.UNVERIFIED_FRICTION_WARNING,
        market_specific_caveats=["Omits SEC Section 31 and FINRA TAF"],
        known_failure_regimes=["Market maker withdrawal"],
        verified_citations=[
            DocCitationSpan(
                chunk_id=str(chunk.chunk_id),
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                source_uri=chunk.source_uri,
                section_heading=chunk.section_heading,
                verbatim_quote=chunk_content,
            )
        ],
    )
    res = gate.evaluate(brief_3bps, [chunk])
    assert not res.passed
    assert res.effective_friction_floor_bps == 5.0


def test_scepticism_gate_fails_after_costs_drawdown() -> None:
    """Verifies that excessive drawdown (> 30%) triggers FAILS_AFTER_COSTS."""
    gate = MarketRegimeScepticismGate()
    chunk_content = "Strategy achieves 1.1 Sharpe with 34.5% peak-to-trough max drawdown."
    chunk = DocumentChunk(
        chunk_id=uuid.uuid4(),
        doc_id="arxiv:2401.05432",
        doc_title="US Drawdown Test",
        source_uri="https://arxiv.org/abs/2401.05432",
        market_scope=MarketRegime.US_NYSE_NASDAQ,
        section_heading="Results",
        verbatim_content=chunk_content,
        formulas_latex=[],
        tables_markdown=[],
        token_count_approx=50,
    )

    brief_high_dd = QuantStrategyBrief(
        strategy_name="High DD Strategy",
        target_markets=[MarketRegime.US_NYSE_NASDAQ],
        mathematical_signal_latex="S_t = 1.0",
        entry_exit_rules=["Hold always"],
        sample_period_years="2018-2023",
        reported_sharpe_ratio=1.1,
        reported_max_drawdown_pct=-34.5,  # Exceeds 30% limit
        transaction_cost_bps_assumed=5.0,
        friction_verdict=FrictionVerdict.FAILS_AFTER_COSTS,
        market_specific_caveats=["Severe tail risk"],
        known_failure_regimes=["Bear markets"],
        verified_citations=[
            DocCitationSpan(
                chunk_id=str(chunk.chunk_id),
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                source_uri=chunk.source_uri,
                section_heading=chunk.section_heading,
                verbatim_quote=chunk_content,
            )
        ],
    )
    res = gate.evaluate(brief_high_dd, [chunk])
    assert not res.passed
    assert res.friction_verdict == FrictionVerdict.FAILS_AFTER_COSTS


def test_verbatim_doc_citation_verification_val02() -> None:
    """Verifies citation quote verification via DeterministicSlicer enforces VAL-02."""
    slicer = DeterministicSlicer()
    source_chunk = (
        "In our empirical analysis of Nifty 50 constituents, gross annual returns of 22.4% "
        "and Sharpe ratio 1.45 decline to 14.8% and 0.92 net Sharpe after subtracting 15 bps STT."
    )

    # 1. Byte-exact verbatim quote -> Pass (overlap 1.0)
    exact_span = DocCitationSpan(
        chunk_id="chunk-1",
        doc_id="arxiv:2403.01182",
        doc_title="NSE STT",
        source_uri="https://arxiv.org/abs/2403.01182",
        section_heading="Results",
        verbatim_quote=(
            "gross annual returns of 22.4% and Sharpe ratio 1.45 decline to "
            "14.8% and 0.92 net Sharpe"
        ),
    )
    res_exact = slicer.verify_doc_citation(exact_span, source_chunk)
    assert res_exact.is_valid is True
    assert res_exact.token_overlap_ratio >= 0.98

    # 2. Fabricated or altered metrics -> Fail
    altered_span = DocCitationSpan(
        chunk_id="chunk-1",
        doc_id="arxiv:2403.01182",
        doc_title="NSE STT",
        source_uri="https://arxiv.org/abs/2403.01182",
        section_heading="Results",
        verbatim_quote=(
            "gross annual returns of 44.8% and Sharpe ratio 2.90 decline to "
            "29.6% and 1.84 net Sharpe"
        ),
    )
    res_altered = slicer.verify_doc_citation(altered_span, source_chunk)
    assert res_altered.is_valid is False
    assert res_altered.token_overlap_ratio < 0.98


def test_in_memory_rls_isolation_on_doc_chunks() -> None:
    """Verifies that InMemoryRLSRepository strictly isolates doc_chunks by tenant scopes."""
    repo = InMemoryRLSRepository()
    emb = [0.0] * 768

    public_chunk = DocChunkRecord(
        chunk_id=uuid.uuid4(),
        domain_id="quant_research_scout",
        doc_id="arxiv:public_1",
        doc_title="Public Quant Research",
        source_uri="https://arxiv.org/abs/public_1",
        market_scope=MarketRegime.US_NYSE_NASDAQ,
        published_date=date(2024, 1, 1),
        authority_weight=1.0,
        section_heading="Intro",
        contextual_header="[Context: public]",
        verbatim_content="Public market research.",
        metadata_json={},
        allowed_scopes=["quant_public"],
        embedding=emb,
    )
    restricted_chunk = DocChunkRecord(
        chunk_id=uuid.uuid4(),
        domain_id="quant_research_scout",
        doc_id="arxiv:internal_1",
        doc_title="Restricted Alpha Paper",
        source_uri="https://internal/alpha_1",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        published_date=date(2024, 2, 1),
        authority_weight=2.0,
        section_heading="Alpha",
        contextual_header="[Context: restricted]",
        verbatim_content="Proprietary statistical arbitrage alpha.",
        metadata_json={},
        allowed_scopes=["quant_restricted"],
        embedding=emb,
    )

    admin_principal = PrincipalContext(
        user_id="admin",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["quant_public", "quant_restricted"],
    )
    repo.insert_doc_chunks_batch(admin_principal, [public_chunk, restricted_chunk])

    # Public principal MUST see only 1 chunk
    public_principal = PrincipalContext(
        user_id="junior_quant",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["quant_public"],
    )
    visible_to_public = repo.list_visible_doc_chunks(public_principal, "quant_research_scout")
    assert len(visible_to_public) == 1
    assert visible_to_public[0].doc_id == "arxiv:public_1"

    # Restricted principal sees both
    visible_to_admin = repo.list_visible_doc_chunks(admin_principal, "quant_research_scout")
    assert len(visible_to_admin) == 2

    # Unauthorized insertion raises PermissionError
    unauthorized_chunk = DocChunkRecord(
        chunk_id=uuid.uuid4(),
        domain_id="quant_research_scout",
        doc_id="arxiv:hack",
        doc_title="Unauthorized",
        source_uri="https://bad",
        market_scope=MarketRegime.GLOBAL_MULTI,
        published_date=None,
        authority_weight=1.0,
        section_heading="Sec",
        contextual_header="[Context: bad]",
        verbatim_content="Hacked content",
        metadata_json={},
        allowed_scopes=["secret_scope"],
        embedding=emb,
    )
    with pytest.raises(PermissionError):
        repo.insert_doc_chunk(public_principal, unauthorized_chunk)
