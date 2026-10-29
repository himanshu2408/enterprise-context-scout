"""Comprehensive Tests for MCP Servers (R3, R4, R5).

Verifies out-of-band context injection (SEC-04), payload bounding (SEC-03),
live code repo tool execution against PostgreSQL 17, and live quant corpus
tool execution against 14,318 research chunks under Forced RLS.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
from mcp.types import CallToolResult

from context_scout.config import ScoutSettings
from context_scout.db.session import RLSConnectionManager
from context_scout.mcp_servers.code_repo_mcp import create_code_repo_mcp_server
from context_scout.mcp_servers.common import (
    MAX_PAYLOAD_BYTES,
    TRUNCATION_INDICATOR,
    bound_payload_dict,
)
from context_scout.mcp_servers.quant_corpus_mcp import create_quant_corpus_mcp_server
from context_scout.mcp_servers.seed_code_corpus import seed_trading_codebase_and_issues
from context_scout.schemas.contracts import (
    FrictionVerdict,
    PrincipalContext,
    RoleType,
)
from context_scout.security.rbac import (
    AuthenticationRequiredError,
    DomainToolAccessDeniedError,
    SecurityError,
    bind_session_principal,
    scoped_db_session,
)


def _extract_tool_payload(res: object) -> dict[str, Any]:
    assert isinstance(res, CallToolResult)
    assert not res.is_error
    content = res.structured_content
    assert isinstance(content, dict)
    return content


@pytest.fixture(scope="module", autouse=True)
def ensure_code_corpus_seeded() -> None:
    """Ensures trading strategies and GitHub issues are persisted in live PostgreSQL 17."""
    seed_trading_codebase_and_issues()


@pytest.fixture
def eng_principal() -> PrincipalContext:
    return PrincipalContext(
        user_id="test_m4_eng_lead",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )


@pytest.fixture
def quant_principal() -> PrincipalContext:
    return PrincipalContext(
        user_id="test_m4_quant_lead",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["public_quant"],
    )


# =============================================================================
# 1. Schema Hygiene & Out-of-Band Security Isolation Tests (SEC-04)
# =============================================================================


@pytest.mark.asyncio
async def test_code_repo_mcp_schema_excludes_security_parameters() -> None:
    """Verifies no tool in code_repo_mcp exposes PrincipalContext or scopes in input schema."""
    server = create_code_repo_mcp_server()
    tools = await server.list_tools()

    tool_names = [t.name for t in tools]
    assert "search_code_trigram" in tool_names
    assert "lookup_ast_symbol" in tool_names
    assert "expand_ast_1hop_closure" in tool_names
    assert "slice_file_lines" in tool_names
    assert "search_github_issues" in tool_names

    forbidden_schema_props = {
        "principal",
        "principal_context",
        "allowed_scopes",
        "scopes",
        "ctx",
        "user_id",
        "role",
    }
    for t in tools:
        props = set(t.input_schema.get("properties", {}).keys())
        leakage = props.intersection(forbidden_schema_props)
        assert not leakage, f"Security parameter leakage in tool {t.name}: {leakage}"


@pytest.mark.asyncio
async def test_quant_corpus_mcp_schema_excludes_security_parameters() -> None:
    """Verifies no tool in quant_corpus_mcp exposes PrincipalContext or scopes in input schema."""
    server = create_quant_corpus_mcp_server()
    tools = await server.list_tools()

    tool_names = [t.name for t in tools]
    assert "search_quant_papers_hybrid" in tool_names
    assert "extract_paper_equations_and_tables" in tool_names
    assert "verify_market_friction_assumptions" in tool_names
    assert "compare_cross_market_regimes" in tool_names

    forbidden_schema_props = {
        "principal",
        "principal_context",
        "allowed_scopes",
        "scopes",
        "ctx",
        "user_id",
        "role",
    }
    for t in tools:
        props = set(t.input_schema.get("properties", {}).keys())
        leakage = props.intersection(forbidden_schema_props)
        assert not leakage, f"Security parameter leakage in tool {t.name}: {leakage}"


@pytest.mark.asyncio
async def test_tool_execution_unbound_session_raises_auth_required() -> None:
    """Verifies executing MCP tools without out-of-band principal binding fails."""
    from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

    code_server = create_code_repo_mcp_server()
    with pytest.raises((AuthenticationRequiredError, UnexpectedToolError, ToolError)) as exc_c:
        await code_server.call_tool("search_code_trigram", {"query": "OrderFlowImbalance"})
    cause_c = exc_c.value.__cause__ or exc_c.value
    assert isinstance(cause_c, AuthenticationRequiredError)

    quant_server = create_quant_corpus_mcp_server()
    with pytest.raises((AuthenticationRequiredError, UnexpectedToolError, ToolError)) as exc_q:
        await quant_server.call_tool("search_quant_papers_hybrid", {"query": "microstructure"})
    cause_q = exc_q.value.__cause__ or exc_q.value
    assert isinstance(cause_q, AuthenticationRequiredError)


@pytest.mark.asyncio
async def test_tool_execution_with_parameter_injection_rejected(
    eng_principal: PrincipalContext,
) -> None:
    """Verifies adversarial attempts to supply 'allowed_scopes' or 'principal' in args fail."""
    from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

    server = create_code_repo_mcp_server()
    with bind_session_principal(eng_principal), pytest.raises(
        (SecurityError, UnexpectedToolError, ToolError)
    ) as exc_info:
        await server.call_tool(
            "search_code_trigram",
            {"query": "Kalman", "allowed_scopes": ["quant_restricted"]},
        )
    cause = exc_info.value.__cause__ or exc_info.value
    assert isinstance(cause, SecurityError)
    assert "Security parameter injection detected" in str(cause)


def test_domain_authorization_gate_rejects_unallowed_tool(
    eng_principal: PrincipalContext,
) -> None:
    """Verifies DomainAuthorizationGate rejects tool calls outside the active domain manifest."""
    from context_scout.agents.domain_registry import DomainRegistry
    from context_scout.security.rbac import DomainAuthorizationGate

    gate = DomainAuthorizationGate(DomainRegistry(ScoutSettings().domains_dir))
    with pytest.raises(DomainToolAccessDeniedError):
        gate.authorize_tool_call(
            "trading_code_scout",
            "quant_corpus_mcp:search_quant_papers_hybrid",
            eng_principal,
        )


# =============================================================================
# 2. Hard Payload Caps and Truncation Guardrail Tests (SEC-03)
# =============================================================================


def test_bound_payload_dict_item_clamping() -> None:
    """Verifies collection outputs are clamped to <= 10 items."""
    oversized = {"matches": [{"id": i, "val": f"match_{i}"} for i in range(25)]}
    bounded = bound_payload_dict(oversized, max_items=10)
    assert len(bounded["matches"]) == 10
    assert bounded["matches"][0]["val"] == "match_0"
    assert bounded["matches"][9]["val"] == "match_9"


def test_bound_payload_dict_byte_capping_and_truncation_notice() -> None:
    """Verifies oversized serialized payloads are truncated to <= 32 KB with notice."""
    huge_text = "X" * 40_000
    oversized = {
        "query": "stress_test",
        "matches": [
            {"file_path": "foo.py", "verbatim_content": huge_text},
            {"file_path": "bar.py", "verbatim_content": huge_text},
        ],
    }
    bounded = bound_payload_dict(oversized, max_bytes=MAX_PAYLOAD_BYTES)
    serialized_bytes = len(json.dumps(bounded, default=str).encode("utf-8"))

    assert serialized_bytes <= MAX_PAYLOAD_BYTES
    assert bounded.get("payload_truncated") is True
    assert TRUNCATION_INDICATOR in str(bounded)


# =============================================================================
# 3. Live Code Repo Enterprise MCP Server Integration Tests (R3)
# =============================================================================


@pytest.mark.asyncio
async def test_live_search_code_trigram(eng_principal: PrincipalContext) -> None:
    """Tests search_code_trigram queries live code_files in PostgreSQL 17 under RLS."""
    server = create_code_repo_mcp_server()
    with bind_session_principal(eng_principal):
        res = await server.call_tool(
            "search_code_trigram",
            {"query": "OrderFlowImbalanceSignal", "limit": 5},
        )
        data = _extract_tool_payload(res)
        assert data["total_matches"] >= 1
        matches = data["matches"]
        assert any("order_flow_imbalance.py" in m["file_path"] for m in matches)
        assert all("match_line" in m and "snippet" in m for m in matches)


@pytest.mark.asyncio
async def test_live_lookup_ast_symbol(eng_principal: PrincipalContext) -> None:
    """Tests lookup_ast_symbol retrieves live symbol records from ast_symbols table."""
    server = create_code_repo_mcp_server()
    with bind_session_principal(eng_principal):
        res = await server.call_tool(
            "lookup_ast_symbol",
            {"symbol_name": "KalmanHedgeEstimator.update", "limit": 5},
        )
        data = _extract_tool_payload(res)
        assert data["total_matches"] >= 1
        sym = data["symbols"][0]
        assert "KalmanHedgeEstimator.update" in sym["qualified_name"]
        assert sym["symbol_kind"] in ("method", "function")
        assert sym["start_line"] > 0
        assert sym["end_line"] >= sym["start_line"]
        assert "pairs_trading_kalman.py" in sym["file_path"]


@pytest.mark.asyncio
async def test_live_expand_ast_1hop_closure(eng_principal: PrincipalContext) -> None:
    """Tests expand_ast_1hop_closure bounded traversal under SEC-03 token budget."""
    server = create_code_repo_mcp_server()
    with bind_session_principal(eng_principal):
        res = await server.call_tool(
            "expand_ast_1hop_closure",
            {"symbol_name": "PairsSpreadSignal"},
        )
        data = _extract_tool_payload(res)
        assert "focus_symbol" in data
        assert data["accumulated_tokens"] <= 1500
        assert data["direct_callers_count"] <= 3


@pytest.mark.asyncio
async def test_live_slice_file_lines_valid_and_bounds_errors(
    eng_principal: PrincipalContext,
) -> None:
    """Tests slice_file_lines with 1-indexed bounds, SHA provenance, and off-by-one errors."""
    server = create_code_repo_mcp_server()
    with bind_session_principal(eng_principal):
        # 1. Valid slice
        res = await server.call_tool(
            "slice_file_lines",
            {
                "file_path": "src/context_scout/trading_strategies/volatility_garch.py",
                "start_line": 1,
                "end_line": 20,
            },
        )
        data = _extract_tool_payload(res)
        assert data["slice_line_count"] == 20
        assert len(data["lines"]) == 20
        assert data["lines"][0]["line_number"] == 1
        assert len(data["git_commit_sha"]) >= 7
        assert len(data["git_blob_sha"]) >= 7

        # 2. Off-by-one: 0-indexed start_line must fail
        with pytest.raises((ValueError, UnexpectedToolError, ToolError)) as err_zero:
            await server.call_tool(
                "slice_file_lines",
                {
                    "file_path": "src/context_scout/trading_strategies/volatility_garch.py",
                    "start_line": 0,
                    "end_line": 10,
                },
            )
        cause_zero = err_zero.value.__cause__ or err_zero.value
        assert "0-indexed" in str(cause_zero) or "Invalid line bounds" in str(cause_zero)

        # 3. Inverted bounds: end_line < start_line must fail
        with pytest.raises((ValueError, UnexpectedToolError, ToolError)) as err_inv:
            await server.call_tool(
                "slice_file_lines",
                {
                    "file_path": "src/context_scout/trading_strategies/volatility_garch.py",
                    "start_line": 25,
                    "end_line": 10,
                },
            )
        cause_inv = err_inv.value.__cause__ or err_inv.value
        assert "cannot be less than" in str(cause_inv) or "Invalid line bounds" in str(cause_inv)

        # 4. Exceeding 200 line cap must fail
        with pytest.raises((ValueError, UnexpectedToolError, ToolError)) as err_cap:
            await server.call_tool(
                "slice_file_lines",
                {
                    "file_path": "src/context_scout/trading_strategies/volatility_garch.py",
                    "start_line": 1,
                    "end_line": 250,
                },
            )
        cause_cap = err_cap.value.__cause__ or err_cap.value
        assert "exceeds maximum allowed 200 lines" in str(cause_cap)


@pytest.mark.asyncio
async def test_live_search_github_issues(eng_principal: PrincipalContext) -> None:
    """Tests search_github_issues retrieves seeded issues from doc_chunks table."""
    server = create_code_repo_mcp_server()
    with bind_session_principal(eng_principal):
        res = await server.call_tool(
            "search_github_issues",
            {"query": "Kalman covariance instability", "limit": 5},
        )
        data = _extract_tool_payload(res)
        assert data["total_matches"] >= 1
        issue = data["issues"][0]
        assert issue["doc_id"] == "gh-issue:101"
        assert "Kalman" in issue["title"]
        assert issue["state"] == "open"
        assert "Joseph form" in issue["excerpt"]

        # Filter by state
        res_closed = await server.call_tool(
            "search_github_issues",
            {"query": "Almgren", "state": "closed"},
        )
        data_closed = _extract_tool_payload(res_closed)
        assert data_closed["total_matches"] >= 1
        assert data_closed["issues"][0]["state"] == "closed"


# =============================================================================
# 4. Live Quant Research Corpus MCP Server Integration Tests (R4)
# =============================================================================


@pytest.mark.asyncio
async def test_live_search_quant_papers_hybrid_rrf(quant_principal: PrincipalContext) -> None:
    """Tests search_quant_papers_hybrid executes 3-way RRF over 14,318 live chunks."""
    server = create_quant_corpus_mcp_server()
    with bind_session_principal(quant_principal):
        res = await server.call_tool(
            "search_quant_papers_hybrid",
            {"query": "intraday momentum microstructure friction", "market": "IN_NSE", "top_k": 5},
        )
        data = _extract_tool_payload(res)
        assert data["market_filter"] == "IN_NSE"
        assert len(data["hits"]) >= 1
        top_hit = data["hits"][0]
        assert top_hit["market_scope"] == "IN_NSE"
        assert top_hit["normalized_rrf_score"] > 0.0
        assert "doc_title" in top_hit
        assert "verbatim_content" in top_hit
        assert len(top_hit["verbatim_content"]) > 10


@pytest.mark.asyncio
async def test_live_extract_paper_equations_and_tables(
    quant_principal: PrincipalContext,
) -> None:
    """Tests extract_paper_equations_and_tables finds formulas and tables in live papers."""
    server = create_quant_corpus_mcp_server()
    settings = ScoutSettings()
    conn_mgr = RLSConnectionManager(settings.db_app_url)

    # Find a paper id that contains mathematical notation
    with bind_session_principal(quant_principal), scoped_db_session(conn_mgr) as conn:
        sample_doc = conn.execute(
            """
            SELECT doc_id FROM doc_chunks
            WHERE domain_id = 'quant_research_scout'
              AND (verbatim_content LIKE '%%$$%%' OR verbatim_content LIKE '%%\\begin{equation%%')
            LIMIT 1;
            """
        ).fetchone()

    assert sample_doc is not None
    doc_id = str(sample_doc[0])

    with bind_session_principal(quant_principal):
        res = await server.call_tool(
            "extract_paper_equations_and_tables",
            {"doc_id": doc_id, "max_items": 5},
        )
        data = _extract_tool_payload(res)
        assert data["doc_id"] == doc_id
        assert data["total_chunks_inspected"] > 0
        assert data["equations_extracted"] > 0
        assert len(data["equations"]) >= 1
        assert "formula_latex" in data["equations"][0]


@pytest.mark.asyncio
async def test_live_verify_market_friction_assumptions_verdicts(
    quant_principal: PrincipalContext,
) -> None:
    """Tests Scepticism Gate verdicts across friction deficits and regulatory citations."""
    server = create_quant_corpus_mcp_server()
    with bind_session_principal(quant_principal):
        # 1. Below floor -> UNVERIFIED_FRICTION_WARNING
        res_warn = await server.call_tool(
            "verify_market_friction_assumptions",
            {
                "target_markets": ["IN_NSE"],
                "assumed_cost_bps": 5.0,  # Floor is 15.0 bps
                "reported_sharpe": 2.1,
            },
        )
        data_warn = _extract_tool_payload(res_warn)
        assert data_warn["friction_verdict"] == FrictionVerdict.UNVERIFIED_FRICTION_WARNING.value
        assert data_warn["institutional_floor_bps"] == 15.0
        assert any(
            "Securities Transaction Tax" in c["statutory_basis"]
            for c in data_warn["regulatory_citations"]
        )

        # 2. High drag causes net Sharpe <= 0.5 -> FAILS_AFTER_COSTS
        res_fail = await server.call_tool(
            "verify_market_friction_assumptions",
            {
                "target_markets": ["IN_NSE"],
                "assumed_cost_bps": 2.0,
                "reported_sharpe": 0.6,
                "turnover_annual": 24.0,
            },
        )
        data_fail = _extract_tool_payload(res_fail)
        assert data_fail["friction_verdict"] == FrictionVerdict.FAILS_AFTER_COSTS.value
        assert data_fail["estimated_net_sharpe"] <= 0.5

        # 3. Meets or exceeds floor -> REALISTIC_FRICTION
        res_pass = await server.call_tool(
            "verify_market_friction_assumptions",
            {
                "target_markets": ["IN_NSE"],
                "assumed_cost_bps": 18.0,
                "reported_sharpe": 1.9,
            },
        )
        data_pass = _extract_tool_payload(res_pass)
        assert data_pass["friction_verdict"] == FrictionVerdict.REALISTIC_FRICTION.value
        assert data_pass["friction_deficit_bps"] == 0.0

        # 4. Intraday futures floor on IN_NSE drops to 5.0 bps
        res_fut = await server.call_tool(
            "verify_market_friction_assumptions",
            {
                "target_markets": ["IN_NSE"],
                "assumed_cost_bps": 6.0,
                "is_intraday_futures": True,
                "reported_sharpe": 1.6,
            },
        )
        data_fut = _extract_tool_payload(res_fut)
        assert data_fut["institutional_floor_bps"] == 5.0
        assert data_fut["friction_verdict"] == FrictionVerdict.REALISTIC_FRICTION.value


@pytest.mark.asyncio
async def test_live_compare_cross_market_regimes(quant_principal: PrincipalContext) -> None:
    """Tests compare_cross_market_regimes compares IN_NSE, DE_XETRA, and US_EQUITY."""
    server = create_quant_corpus_mcp_server()
    with bind_session_principal(quant_principal):
        res = await server.call_tool(
            "compare_cross_market_regimes",
            {
                "strategy_concept": "momentum order book",
                "markets": ["IN_NSE", "DE_XETRA", "US_EQUITY"],
            },
        )
        data = _extract_tool_payload(res)
        assert data["compared_markets_count"] == 3
        comp = data["comparison"]
        markets_found = {c["market"] for c in comp}
        assert "IN_NSE" in markets_found
        assert "DE_XETRA" in markets_found
        assert "US_EQUITY" in markets_found

        nse_item = next(c for c in comp if c["market"] == "IN_NSE")
        assert nse_item["institutional_friction_floor_bps"] == 15.0
        assert "STT" in nse_item["regulatory_statutes"]

        xetra_item = next(c for c in comp if c["market"] == "DE_XETRA")
        assert xetra_item["institutional_friction_floor_bps"] == 10.0
        assert "MiFID II" in xetra_item["regulatory_statutes"]

        us_item = next(c for c in comp if c["market"] == "US_EQUITY")
        assert us_item["institutional_friction_floor_bps"] == 5.0
        assert "Reg NMS" in us_item["regulatory_statutes"]


# =============================================================================
# 5. Legacy Tool Aliases Verification
# =============================================================================


@pytest.mark.asyncio
async def test_code_repo_legacy_tool_aliases(eng_principal: PrincipalContext) -> None:
    """Verifies all legacy tool aliases in code_repo_mcp execute successfully."""
    server = create_code_repo_mcp_server()
    with bind_session_principal(eng_principal):
        # grep_repo -> search_code_trigram
        res_grep = await server.call_tool("grep_repo", {"query": "Kalman"})
        data_grep = _extract_tool_payload(res_grep)
        assert "matches" in data_grep

        # search_ast_symbols -> lookup_ast_symbol
        res_sym = await server.call_tool("search_ast_symbols", {"symbol_name": "GARCH11Model"})
        data_sym = _extract_tool_payload(res_sym)
        assert "symbols" in data_sym

        # list_github_issues -> search_github_issues
        res_iss = await server.call_tool("list_github_issues", {"query": "Kalman"})
        data_iss = _extract_tool_payload(res_iss)
        assert "issues" in data_iss


@pytest.mark.asyncio
async def test_quant_corpus_legacy_tool_aliases(quant_principal: PrincipalContext) -> None:
    """Verifies all legacy tool aliases in quant_corpus_mcp execute successfully."""
    server = create_quant_corpus_mcp_server()
    with bind_session_principal(quant_principal):
        # search_papers_hybrid -> search_quant_papers_hybrid
        res_search = await server.call_tool(
            "search_papers_hybrid", {"query": "momentum", "top_k": 2}
        )
        data_search = _extract_tool_payload(res_search)
        assert "hits" in data_search

        # compare_market_frictions -> verify_market_friction_assumptions
        res_comp = await server.call_tool(
            "compare_market_frictions",
            {"target_markets": ["DE_XETRA"], "assumed_cost_bps": 12.0},
        )
        data_comp = _extract_tool_payload(res_comp)
        assert data_comp["friction_verdict"] == FrictionVerdict.REALISTIC_FRICTION.value
