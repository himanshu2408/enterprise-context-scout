"""Quantitative Research Corpus Enterprise MCP Server (Model 2: quant_research_scout).

Implements typed read tools backed by 14,318 full-text research chunks across
IN_NSE, DE_XETRA, and US_EQUITY in PostgreSQL 17, LocalSemanticEmbedder,
3-Way Normalized RRF, and the Market-Regime Quant Scepticism Gate.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Final
from uuid import UUID

from mcp.server.mcpserver import Context, MCPServer

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.config import ScoutSettings
from context_scout.db.repositories import PostgresRLSRepository
from context_scout.db.session import RLSConnectionManager
from context_scout.ingestion.scepticism_gate import (
    MarketRegimeScepticismGate,
    calculate_effective_friction_floor,
)
from context_scout.mcp_servers.common import (
    MAX_PAYLOAD_ITEMS,
    bound_payload_dict,
)
from context_scout.retrieval.embedder import LocalSemanticEmbedder
from context_scout.schemas.contracts import (
    FrictionVerdict,
    MarketRegime,
)
from context_scout.security.rbac import (
    DomainAuthorizationGate,
    get_current_principal,
    scoped_db_session,
    validate_tool_call_arguments,
)

logger = logging.getLogger(__name__)

DOMAIN_ID: Final[str] = "quant_research_scout"

# Math and table extraction patterns
MATH_DISPLAY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(\\begin\{(?:equation|align|gather|multline|eqnarray)\*?\}[\s\S]*?\\end\{(?:equation|align|gather|multline|eqnarray)\*?\}|\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\])",
    re.MULTILINE,
)
TABLE_LATEX_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(\\begin\{(?:table|tabular)\*?\}[\s\S]*?\\end\{(?:table|tabular)\*?\})",
    re.MULTILINE,
)
TABLE_MARKDOWN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(\|.*?\|\n\|[-:\s|]+\|\n(?:\|.*?\|\n?)+)",
    re.MULTILINE,
)

MARKET_MICROSTRUCTURE_PROFILES: Final[dict[str, dict[str, str]]] = {
    "IN_NSE": {
        "regulatory_statutes": (
            "Securities Transaction Tax (STT) under Finance (No. 2) Act 2024: 0.1% delivery, "
            "0.02% equity futures, 0.125% options exercise. SEBI turnover fees, GST (18%), "
            "stamp duty (0.015% delivery)."
        ),
        "microstructure_mechanisms": (
            "Pre-open auction (09:00-09:08 IST), continuous order matching (09:15-15:30 IST), "
            "closing call auction (15:30-15:40 IST). Dynamic tick size framework (₹0.01 / ₹0.05)."
        ),
    },
    "DE_XETRA": {
        "regulatory_statutes": (
            "MiFID II RTS 27/28 best execution disclosures. Eurex clearing fees. Article 23 "
            "Trading Obligation for shares (STO). German FTT draft guidelines."
        ),
        "microstructure_mechanisms": (
            "Opening call auction (08:50-09:00 CET), midday auction (13:00 CET), and closing "
            "auction (17:30 CET). MiFID II liquidity band tick size schedule."
        ),
    },
    "US_EQUITY": {
        "regulatory_statutes": (
            "SEC Section 31 fee ($27.80/million). FINRA TAF ($0.000166/share). "
            "Reg NMS Rule 611 (Order Protection), SEC Rule 605/606 execution quality."
        ),
        "microstructure_mechanisms": (
            "Continuous market maker and electronic communication networks (ECNs). Maker-taker and "
            "inverted maker-taker rebate models. Sub-penny quoting reforms (Rule 612)."
        ),
    },
}


def _resolve_market_scope(market: str | None) -> MarketRegime | None:
    if not market:
        return None
    normalized = market.strip().upper()
    for regime in MarketRegime:
        if regime.value == normalized or regime.name == normalized:
            return regime
    if "NSE" in normalized or "INDIA" in normalized:
        return MarketRegime.INDIA_NSE_BSE
    if "XETRA" in normalized or "GERMANY" in normalized or "DAX" in normalized:
        return MarketRegime.GERMANY_XETRA_DAX
    if "US" in normalized or "NYSE" in normalized or "NASDAQ" in normalized:
        return MarketRegime.US_NYSE_NASDAQ
    return None


def create_quant_corpus_mcp_server(
    settings: ScoutSettings | None = None,
    connection_manager: RLSConnectionManager | None = None,
    embedder: LocalSemanticEmbedder | None = None,
) -> MCPServer:
    """Factory creating the quant_corpus_mcp MCP Server with live PostgreSQL 17 RLS integration."""
    st = settings or ScoutSettings()
    conn_mgr = connection_manager or RLSConnectionManager(st.db_app_url)
    repo = PostgresRLSRepository()
    registry = DomainRegistry(st.domains_dir)
    registry.load_all()
    auth_gate = DomainAuthorizationGate(registry)
    scepticism_gate = MarketRegimeScepticismGate()
    local_embedder = embedder or LocalSemanticEmbedder(
        settings=st,
        allow_deterministic_fallback=True,
    )

    server = MCPServer("quant_corpus_mcp")

    # -------------------------------------------------------------------------
    # Canonical Tool 1: search_quant_papers_hybrid
    # -------------------------------------------------------------------------
    @server.tool(
        name="search_quant_papers_hybrid",
        description="Search quant corpus using 3-Way Normalized RRF across 14,318 papers.",
    )
    async def search_quant_papers_hybrid(
        query: str,
        market: str | None = None,
        top_k: int = 5,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Executes 3-way Normalized RRF across live doc_chunks in PostgreSQL 17."""
        validate_tool_call_arguments({"query": query, "market": market, "top_k": top_k})
        principal = get_current_principal()
        auth_gate.authorize_tool_call(
            DOMAIN_ID, "quant_corpus_mcp:search_quant_papers_hybrid", principal
        )

        clamped_k = min(max(1, top_k), MAX_PAYLOAD_ITEMS)
        market_scope = _resolve_market_scope(market)

        # Generate 768-dim semantic embedding vector
        try:
            query_vector = local_embedder.embed_text(query, is_query=True)
        except Exception as err:
            logger.warning("Local embedder failed (%s); using deterministic fallback", err)
            from context_scout.ingestion.git_merkle_indexer import generate_deterministic_embedding

            query_vector = generate_deterministic_embedding(query, dim=768)

        with scoped_db_session(conn_mgr, principal) as conn:
            ranked_hits = repo.search_doc_chunks_hybrid_rrf(
                conn=conn,
                domain_id=DOMAIN_ID,
                query_text=query,
                query_embedding=query_vector,
                market_scope=market_scope,
                top_k=clamped_k,
            )

            hits_data: list[dict[str, Any]] = []
            if ranked_hits:
                chunk_ids = [h.item_id for h in ranked_hits]
                rows = conn.execute(
                    """
                    SELECT chunk_id, doc_id, doc_title, source_uri, market_scope,
                           section_heading, contextual_header, verbatim_content, authority_weight
                    FROM doc_chunks
                    WHERE chunk_id = ANY(%s);
                    """,
                    (chunk_ids,),
                ).fetchall()

                lookup = {UUID(str(r[0])): r for r in rows}
                for hit in ranked_hits:
                    if hit.item_id in lookup:
                        r = lookup[hit.item_id]
                        hits_data.append(
                            {
                                "chunk_id": str(r[0]),
                                "doc_id": str(r[1]),
                                "doc_title": str(r[2]),
                                "source_uri": str(r[3]),
                                "market_scope": str(r[4]),
                                "section_heading": str(r[5]),
                                "contextual_header": str(r[6]),
                                "verbatim_content": str(r[7]),
                                "authority_weight": float(str(r[8])),
                                "normalized_rrf_score": round(hit.normalized_rrf_score, 4),
                                "trigram_rank": hit.trigram_rank,
                                "fts_rank": hit.bm25_rank,
                                "vector_rank": hit.vector_rank,
                            }
                        )

        payload = {
            "query": query,
            "market_filter": market_scope.value if market_scope else None,
            "total_hits": len(hits_data),
            "hits": hits_data,
        }
        return bound_payload_dict(payload)

    # -------------------------------------------------------------------------
    # Canonical Tool 2: extract_paper_equations_and_tables
    # -------------------------------------------------------------------------
    @server.tool(
        name="extract_paper_equations_and_tables",
        description="Extract LaTeX display formulas and tables from research papers under RLS.",
    )
    async def extract_paper_equations_and_tables(
        doc_id: str,
        section_filter: str | None = None,
        max_items: int = 5,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Extracts mathematical formulas and tabular structures from live doc_chunks."""
        validate_tool_call_arguments(
            {"doc_id": doc_id, "section_filter": section_filter, "max_items": max_items}
        )
        principal = get_current_principal()
        auth_gate.authorize_tool_call(
            DOMAIN_ID, "quant_corpus_mcp:extract_paper_equations_and_tables", principal
        )

        clamped_max = min(max(1, max_items), MAX_PAYLOAD_ITEMS)
        clean_doc = doc_id.strip()

        with scoped_db_session(conn_mgr, principal) as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, section_heading, verbatim_content
                FROM doc_chunks
                WHERE domain_id = %s
                  AND doc_id = %s
                ORDER BY chunk_id;
                """,
                (DOMAIN_ID, clean_doc),
            ).fetchall()

            equations: list[dict[str, Any]] = []
            tables: list[dict[str, Any]] = []

            for r in rows:
                section = str(r[1])
                content = str(r[2])

                if section_filter and section_filter.lower() not in section.lower():
                    continue

                # Extract LaTeX math blocks
                for math_match in MATH_DISPLAY_PATTERN.finditer(content):
                    if len(equations) < clamped_max:
                        eq_text = math_match.group(1).strip()
                        equations.append(
                            {
                                "section_heading": section,
                                "formula_latex": eq_text,
                                "char_length": len(eq_text),
                            }
                        )

                # Extract LaTeX tables
                for tbl_match in TABLE_LATEX_PATTERN.finditer(content):
                    if len(tables) < clamped_max:
                        tbl_text = tbl_match.group(1).strip()
                        tables.append(
                            {
                                "section_heading": section,
                                "format": "latex",
                                "table_content": tbl_text,
                            }
                        )

                # Extract Markdown tables
                for md_match in TABLE_MARKDOWN_PATTERN.finditer(content):
                    if len(tables) < clamped_max:
                        md_text = md_match.group(1).strip()
                        tables.append(
                            {
                                "section_heading": section,
                                "format": "markdown",
                                "table_content": md_text,
                            }
                        )

        payload = {
            "doc_id": clean_doc,
            "total_chunks_inspected": len(rows),
            "equations_extracted": len(equations),
            "tables_extracted": len(tables),
            "equations": equations,
            "tables": tables,
        }
        return bound_payload_dict(payload)

    # -------------------------------------------------------------------------
    # Canonical Tool 3: verify_market_friction_assumptions
    # -------------------------------------------------------------------------
    @server.tool(
        name="verify_market_friction_assumptions",
        description="Verify transaction cost realism against regulatory friction floors (FIN-01).",
    )
    async def verify_market_friction_assumptions(
        target_markets: list[str],
        assumed_cost_bps: float,
        is_intraday_futures: bool = False,
        reported_sharpe: float | None = None,
        reported_max_drawdown_pct: float | None = None,
        turnover_annual: float = 12.0,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Evaluates strategy viability using institutional friction floors and Sharpe drag."""
        validate_tool_call_arguments(
            {
                "target_markets": target_markets,
                "assumed_cost_bps": assumed_cost_bps,
                "is_intraday_futures": is_intraday_futures,
                "reported_sharpe": reported_sharpe,
                "reported_max_drawdown_pct": reported_max_drawdown_pct,
                "turnover_annual": turnover_annual,
            }
        )
        principal = get_current_principal()
        auth_gate.authorize_tool_call(
            DOMAIN_ID, "quant_corpus_mcp:verify_market_friction_assumptions", principal
        )

        regimes: list[MarketRegime] = []
        for m in target_markets:
            r = _resolve_market_scope(m)
            if r:
                regimes.append(r)

        if not regimes:
            regimes = [MarketRegime.INDIA_NSE_BSE]

        floor_bps = calculate_effective_friction_floor(
            target_markets=regimes,
            is_intraday_futures=is_intraday_futures,
        )

        friction_deficit_bps = max(0.0, floor_bps - assumed_cost_bps)
        annual_drag = (friction_deficit_bps * turnover_annual) / 10000.0
        net_sharpe = scepticism_gate.estimate_net_sharpe(
            reported_sharpe=reported_sharpe,
            assumed_bps=assumed_cost_bps,
            floor_bps=floor_bps,
            turnover_annual=turnover_annual,
        )

        # Verdict assignment
        if (net_sharpe is not None and net_sharpe <= 0.5) or (
            reported_max_drawdown_pct is not None and abs(reported_max_drawdown_pct) > 30.0
        ):
            verdict = FrictionVerdict.FAILS_AFTER_COSTS
            reason = (
                f"Net Sharpe ratio ({net_sharpe}) after institutional cost drag of "
                f"{floor_bps:.1f} bps falls <= 0.5 survival hurdle or drawdown exceeds 30%."
            )
        elif assumed_cost_bps < floor_bps:
            verdict = FrictionVerdict.UNVERIFIED_FRICTION_WARNING
            reason = (
                f"Assumed cost of {assumed_cost_bps:.1f} bps is below institutional minimum floor "
                f"of {floor_bps:.1f} bps for {[reg.value for reg in regimes]}."
            )
        else:
            verdict = FrictionVerdict.REALISTIC_FRICTION
            reason = (
                f"Assumed cost ({assumed_cost_bps:.1f} bps) meets or exceeds institutional floor "
                f"({floor_bps:.1f} bps); strategy remains viable after friction."
            )

        citations: list[dict[str, str]] = []
        for reg in regimes:
            if reg.value in MARKET_MICROSTRUCTURE_PROFILES:
                prof = MARKET_MICROSTRUCTURE_PROFILES[reg.value]
                citations.append(
                    {
                        "market": reg.value,
                        "statutory_basis": prof["regulatory_statutes"],
                        "microstructure_rule": prof["microstructure_mechanisms"],
                    }
                )

        payload = {
            "target_markets": [r.value for r in regimes],
            "assumed_cost_bps": assumed_cost_bps,
            "institutional_floor_bps": floor_bps,
            "friction_deficit_bps": round(friction_deficit_bps, 2),
            "estimated_annual_cost_drag": round(annual_drag, 6),
            "reported_sharpe": reported_sharpe,
            "estimated_net_sharpe": net_sharpe,
            "reported_max_drawdown_pct": reported_max_drawdown_pct,
            "turnover_annual": turnover_annual,
            "is_intraday_futures": is_intraday_futures,
            "friction_verdict": verdict.value,
            "rationale": reason,
            "regulatory_citations": citations,
        }
        return bound_payload_dict(payload)

    # -------------------------------------------------------------------------
    # Canonical Tool 4: compare_cross_market_regimes
    # -------------------------------------------------------------------------
    @server.tool(
        name="compare_cross_market_regimes",
        description="Cross-market comparative analysis across IN_NSE, DE_XETRA, and US_EQUITY.",
    )
    async def compare_cross_market_regimes(
        strategy_concept: str,
        markets: list[str] | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Compares strategy feasibility across market regimes with empirical evidence."""
        validate_tool_call_arguments({"strategy_concept": strategy_concept, "markets": markets})
        principal = get_current_principal()
        auth_gate.authorize_tool_call(
            DOMAIN_ID, "quant_corpus_mcp:compare_cross_market_regimes", principal
        )

        requested_markets = markets or ["IN_NSE", "DE_XETRA", "US_EQUITY"]
        comparison: list[dict[str, Any]] = []

        with scoped_db_session(conn_mgr, principal) as conn:
            for mkt_str in requested_markets:
                regime = _resolve_market_scope(mkt_str)
                if not regime:
                    continue

                floor_bps = calculate_effective_friction_floor([regime])
                profile = MARKET_MICROSTRUCTURE_PROFILES.get(regime.value, {})

                # Find empirical findings in live corpus
                rows = conn.execute(
                    """
                    SELECT doc_id, doc_title, section_heading, verbatim_content
                    FROM doc_chunks
                    WHERE domain_id = %s
                      AND market_scope = %s
                      AND (
                          fts_doc @@ plainto_tsquery('english', %s)
                          OR verbatim_content ILIKE %s
                      )
                    ORDER BY authority_weight DESC
                    LIMIT 2;
                    """,
                    (DOMAIN_ID, regime.value, strategy_concept, f"%{strategy_concept}%"),
                ).fetchall()

                findings: list[dict[str, str]] = []
                for r in rows:
                    findings.append(
                        {
                            "doc_id": str(r[0]),
                            "title": str(r[1]),
                            "section": str(r[2]),
                            "empirical_excerpt": str(r[3])[:300],
                        }
                    )

                comparison.append(
                    {
                        "market": regime.value,
                        "institutional_friction_floor_bps": floor_bps,
                        "regulatory_statutes": profile.get("regulatory_statutes", "Standard rules"),
                        "microstructure_mechanisms": profile.get(
                            "microstructure_mechanisms", "Standard matching"
                        ),
                        "empirical_evidence_count": len(findings),
                        "top_empirical_findings": findings,
                    }
                )

        payload = {
            "strategy_concept": strategy_concept,
            "compared_markets_count": len(comparison),
            "comparison": comparison,
        }
        return bound_payload_dict(payload)

    # -------------------------------------------------------------------------
    # Legacy Tool Aliases
    # -------------------------------------------------------------------------
    @server.tool(
        name="search_papers_hybrid",
        description="Legacy alias for search_quant_papers_hybrid.",
    )
    async def search_papers_hybrid(
        query: str,
        market: str | None = None,
        top_k: int = 5,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return await search_quant_papers_hybrid(query=query, market=market, top_k=top_k, ctx=ctx)

    @server.tool(name="get_paper_section", description="Retrieve specific paper section by doc_id.")
    async def get_paper_section(
        doc_id: str,
        section_heading: str | None = None,
        limit: int = 5,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        validate_tool_call_arguments(
            {"doc_id": doc_id, "section_heading": section_heading, "limit": limit}
        )
        principal = get_current_principal()
        auth_gate.authorize_tool_call(DOMAIN_ID, "quant_corpus_mcp:get_paper_section", principal)

        clamped_limit = min(max(1, limit), MAX_PAYLOAD_ITEMS)
        clean_doc = doc_id.strip()

        with scoped_db_session(conn_mgr, principal) as conn:
            sql = """
                SELECT chunk_id, doc_id, doc_title, source_uri, market_scope,
                       section_heading, contextual_header, verbatim_content
                FROM doc_chunks
                WHERE domain_id = %s
                  AND doc_id = %s
                  AND (%s::text IS NULL OR section_heading ILIKE %s)
                ORDER BY chunk_id
                LIMIT %s;
            """
            s_filter = f"%{section_heading.strip()}%" if section_heading else None
            rows = conn.execute(
                sql, (DOMAIN_ID, clean_doc, s_filter, s_filter, clamped_limit)
            ).fetchall()

            sections_data: list[dict[str, Any]] = []
            for r in rows:
                sections_data.append(
                    {
                        "chunk_id": str(r[0]),
                        "doc_id": str(r[1]),
                        "doc_title": str(r[2]),
                        "source_uri": str(r[3]),
                        "market_scope": str(r[4]),
                        "section_heading": str(r[5]),
                        "contextual_header": str(r[6]),
                        "verbatim_content": str(r[7]),
                    }
                )

        payload = {
            "doc_id": clean_doc,
            "section_filter": section_heading,
            "total_sections": len(sections_data),
            "items": sections_data,
        }
        return bound_payload_dict(payload)

    @server.tool(
        name="compare_market_frictions",
        description="Legacy alias for verify_market_friction_assumptions.",
    )
    async def compare_market_frictions(
        target_markets: list[str],
        assumed_cost_bps: float,
        is_intraday_futures: bool = False,
        reported_sharpe: float | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return await verify_market_friction_assumptions(
            target_markets=target_markets,
            assumed_cost_bps=assumed_cost_bps,
            is_intraday_futures=is_intraday_futures,
            reported_sharpe=reported_sharpe,
            ctx=ctx,
        )

    orig_call_tool = server.call_tool

    async def secure_call_tool(
        name: str,
        arguments: dict[str, Any] | None = None,
        context: Context[Any, Any] | None = None,
    ) -> Any:
        if arguments:
            validate_tool_call_arguments(arguments)
        return await orig_call_tool(
            name, arguments=arguments if arguments is not None else {}, context=context
        )

    server.call_tool = secure_call_tool  # type: ignore[method-assign]

    return server


__all__ = ["create_quant_corpus_mcp_server"]
