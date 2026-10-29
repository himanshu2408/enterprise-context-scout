"""Code Repository Enterprise MCP Server (Model 1: trading_code_scout).

Implements typed read tools backed by live PostgreSQL 17 Forced RLS,
ASTDependencyExpander, and DeterministicSlicer under out-of-band context (SEC-04).
"""

from __future__ import annotations

import logging
from typing import Any, Final
from uuid import UUID

from mcp.server.mcpserver import Context, MCPServer

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.ast_engine.dependency_expander import ASTDependencyExpander
from context_scout.ast_engine.deterministic_slicer import (
    DeterministicSlicer,
    InvalidLineBoundsError,
    OffByOneBoundsError,
)
from context_scout.config import ScoutSettings
from context_scout.db.repositories import ASTSymbolRecord
from context_scout.db.session import RLSConnectionManager
from context_scout.mcp_servers.common import (
    MAX_PAYLOAD_ITEMS,
    PostgresScoutGraphAdapter,
    bound_payload_dict,
)
from context_scout.security.rbac import (
    DomainAuthorizationGate,
    get_current_principal,
    scoped_db_session,
    validate_tool_call_arguments,
)

logger = logging.getLogger(__name__)

DOMAIN_ID: Final[str] = "trading_code_scout"
MAX_SLICE_LINES: Final[int] = 200


def create_code_repo_mcp_server(
    settings: ScoutSettings | None = None,
    connection_manager: RLSConnectionManager | None = None,
) -> MCPServer:
    """Factory creating the code_repo_mcp MCP Server with out-of-band security controls."""
    st = settings or ScoutSettings()
    conn_mgr = connection_manager or RLSConnectionManager(st.db_app_url)
    registry = DomainRegistry(st.domains_dir)
    registry.load_all()
    auth_gate = DomainAuthorizationGate(registry)
    graph_adapter = PostgresScoutGraphAdapter(conn_mgr, st)
    slicer = DeterministicSlicer()
    expander = ASTDependencyExpander(
        max_direct_callers=3,
        max_expansion_token_budget=1500,
    )

    server = MCPServer("code_repo_mcp")

    # -------------------------------------------------------------------------
    # Canonical Tool 1: search_code_trigram
    # -------------------------------------------------------------------------
    @server.tool(
        name="search_code_trigram",
        description="Search codebase via pg_trgm substring similarity under Forced RLS.",
    )
    async def search_code_trigram(
        query: str,
        file_pattern: str | None = None,
        limit: int = 10,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Searches code_files under live PostgreSQL 17 RLS."""
        validate_tool_call_arguments({"query": query, "file_pattern": file_pattern, "limit": limit})
        principal = get_current_principal()
        auth_gate.authorize_tool_call(DOMAIN_ID, "code_repo_mcp:search_code_trigram", principal)

        clamped_limit = min(max(1, limit), MAX_PAYLOAD_ITEMS)
        clean_pattern = f"%{file_pattern.strip()}%" if file_pattern else None

        with scoped_db_session(conn_mgr, principal) as conn:
            sql = """
                SELECT file_path, repo_name, raw_content,
                       similarity(raw_content, %s) AS sim
                FROM code_files
                WHERE domain_id = %s
                  AND (%s::text IS NULL OR file_path ILIKE %s)
                  AND (raw_content %% %s OR raw_content ILIKE %s)
                ORDER BY sim DESC, file_path ASC
                LIMIT %s;
            """
            search_str = f"%{query}%"
            rows = conn.execute(
                sql,
                (query, DOMAIN_ID, clean_pattern, clean_pattern, query, search_str, clamped_limit),
            ).fetchall()

            matches: list[dict[str, Any]] = []
            for r in rows:
                f_path = str(r[0])
                r_name = str(r[1])
                content = str(r[2])
                sim = float(str(r[3]))

                # Find line number of matching query snippet
                match_line = 1
                snippet = ""
                content_lines = content.splitlines()
                q_lower = query.lower()
                for idx, line in enumerate(content_lines, start=1):
                    if q_lower in line.lower():
                        match_line = idx
                        snippet = line.strip()
                        break
                if not snippet and content_lines:
                    snippet = content_lines[0].strip()

                matches.append(
                    {
                        "file_path": f_path,
                        "repo_name": r_name,
                        "match_line": match_line,
                        "snippet": snippet,
                        "similarity": round(sim, 4),
                    }
                )

        payload = {
            "query": query,
            "file_pattern": file_pattern,
            "total_matches": len(matches),
            "matches": matches,
        }
        return bound_payload_dict(payload)

    # -------------------------------------------------------------------------
    # Canonical Tool 2: lookup_ast_symbol
    # -------------------------------------------------------------------------
    @server.tool(
        name="lookup_ast_symbol",
        description="Lookup symbol definition, signature skeleton, and bounds under RLS.",
    )
    async def lookup_ast_symbol(
        symbol_name: str,
        limit: int = 5,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Resolves symbol in ast_symbols JOIN code_files under live PostgreSQL 17 RLS."""
        validate_tool_call_arguments({"symbol_name": symbol_name, "limit": limit})
        principal = get_current_principal()
        auth_gate.authorize_tool_call(DOMAIN_ID, "code_repo_mcp:lookup_ast_symbol", principal)

        clamped_limit = min(max(1, limit), MAX_PAYLOAD_ITEMS)
        clean_name = symbol_name.strip()
        wildcard_name = f"%{clean_name}%"

        with scoped_db_session(conn_mgr, principal) as conn:
            sql = """
                SELECT s.symbol_id, s.qualified_name, s.symbol_kind, f.file_path,
                       s.start_line, s.end_line, s.signature_skeleton, s.contextual_summary,
                       similarity(s.qualified_name, %s) AS sim
                FROM ast_symbols s
                JOIN code_files f ON s.file_id = f.file_id
                WHERE s.domain_id = %s
                  AND (
                      s.qualified_name = %s
                      OR s.qualified_name ILIKE %s
                      OR s.qualified_name %% %s
                      OR s.fts_doc @@ plainto_tsquery('english', %s)
                  )
                ORDER BY (s.qualified_name = %s) DESC, sim DESC, s.qualified_name ASC
                LIMIT %s;
            """
            rows = conn.execute(
                sql,
                (
                    clean_name,
                    DOMAIN_ID,
                    clean_name,
                    wildcard_name,
                    clean_name,
                    clean_name,
                    clean_name,
                    clamped_limit,
                ),
            ).fetchall()

            symbols: list[dict[str, Any]] = []
            for r in rows:
                symbols.append(
                    {
                        "symbol_id": str(r[0]),
                        "qualified_name": str(r[1]),
                        "symbol_kind": str(r[2]),
                        "file_path": str(r[3]),
                        "start_line": int(str(r[4])),
                        "end_line": int(str(r[5])),
                        "signature_skeleton": str(r[6]),
                        "contextual_summary": str(r[7]),
                        "similarity": round(float(str(r[8])), 4),
                    }
                )

        payload = {
            "symbol_name": clean_name,
            "total_matches": len(symbols),
            "symbols": symbols,
        }
        return bound_payload_dict(payload)

    # -------------------------------------------------------------------------
    # Canonical Tool 3: expand_ast_1hop_closure
    # -------------------------------------------------------------------------
    @server.tool(
        name="expand_ast_1hop_closure",
        description="Expand bounded 1-hop AST callers and types under 1500 tokens (SEC-03).",
    )
    async def expand_ast_1hop_closure(
        symbol_name: str,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Expands 1-hop AST closure using ASTDependencyExpander and PostgresScoutGraphAdapter."""
        validate_tool_call_arguments({"symbol_name": symbol_name})
        principal = get_current_principal()
        auth_gate.authorize_tool_call(DOMAIN_ID, "code_repo_mcp:expand_ast_1hop_closure", principal)

        clean_name = symbol_name.strip()

        with scoped_db_session(conn_mgr, principal) as conn:
            row = conn.execute(
                """
                SELECT symbol_id, file_id, domain_id, qualified_name, symbol_kind,
                       start_line, end_line, signature_skeleton, contextual_summary,
                       allowed_scopes, embedding
                FROM ast_symbols
                WHERE domain_id = %s
                  AND (qualified_name = %s OR qualified_name ILIKE %s)
                ORDER BY (qualified_name = %s) DESC
                LIMIT 1;
                """,
                (DOMAIN_ID, clean_name, f"%{clean_name}%", clean_name),
            ).fetchone()

            if not row:
                return bound_payload_dict(
                    {
                        "symbol_name": clean_name,
                        "error": f"Symbol '{clean_name}' not found under active RLS scopes",
                        "total_callers": 0,
                        "items": [],
                    }
                )

            raw_emb = row[10]
            if isinstance(raw_emb, str):
                emb_floats = [float(x.strip()) for x in raw_emb.strip("[]").split(",") if x.strip()]
            elif hasattr(raw_emb, "tolist"):
                emb_floats = [float(x) for x in raw_emb.tolist()]
            elif isinstance(raw_emb, (list, tuple)):
                emb_floats = [float(x) for x in raw_emb]
            else:
                emb_floats = []

            scopes_raw = row[9]
            scopes_list = (
                [str(s) for s in scopes_raw] if isinstance(scopes_raw, (list, tuple)) else []
            )

            focus_record = ASTSymbolRecord(
                symbol_id=UUID(str(row[0])),
                file_id=UUID(str(row[1])),
                domain_id=str(row[2]),
                qualified_name=str(row[3]),
                symbol_kind=str(row[4]),  # type: ignore[arg-type]
                start_line=int(str(row[5])),
                end_line=int(str(row[6])),
                signature_skeleton=str(row[7]),
                contextual_summary=str(row[8]),
                allowed_scopes=scopes_list,
                embedding=emb_floats,
            )

        expansion_result = expander.expand_1hop(
            focus_symbol=focus_record,
            repo=graph_adapter,
            principal=principal,
            domain_id=DOMAIN_ID,
        )

        direct_callers: list[dict[str, Any]] = [
            {
                "symbol_name": c.qualified_symbol,
                "file_path": c.file_path,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "selection_reason": c.selection_reason,
                "modification_intent": str(c.modification_intent),
            }
            for c in expansion_result.caller_slices
        ]

        imported_types: list[dict[str, Any]] = [
            {
                "symbol_name": t.qualified_symbol,
                "file_path": t.file_path,
                "start_line": t.start_line,
                "end_line": t.end_line,
                "selection_reason": t.selection_reason,
                "modification_intent": str(t.modification_intent),
            }
            for t in expansion_result.imported_type_slices
        ]

        test_fixtures: list[dict[str, Any]] = [
            {
                "symbol_name": f.qualified_symbol,
                "file_path": f.file_path,
                "start_line": f.start_line,
                "end_line": f.end_line,
                "selection_reason": f.selection_reason,
                "modification_intent": str(f.modification_intent),
            }
            for f in expansion_result.test_fixture_slices
        ]

        payload = {
            "focus_symbol": focus_record.qualified_name,
            "accumulated_tokens": expansion_result.total_expansion_tokens,
            "token_budget_exhausted": expansion_result.budget_exhausted,
            "direct_callers_count": len(direct_callers),
            "direct_callers": direct_callers,
            "imported_types_count": len(imported_types),
            "imported_types": imported_types,
            "test_fixtures_count": len(test_fixtures),
            "test_fixtures": test_fixtures,
        }
        return bound_payload_dict(payload)

    # -------------------------------------------------------------------------
    # Canonical Tool 4: slice_file_lines
    # -------------------------------------------------------------------------
    @server.tool(
        name="slice_file_lines",
        description="Extract byte-exact source lines with Git commit and blob provenance.",
    )
    async def slice_file_lines(
        file_path: str,
        start_line: int,
        end_line: int,
        repo_name: str = "enterprise-context-scout",
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Slices verbatim code lines using DeterministicSlicer with bounds checks."""
        validate_tool_call_arguments(
            {
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "repo_name": repo_name,
            }
        )
        principal = get_current_principal()
        auth_gate.authorize_tool_call(DOMAIN_ID, "code_repo_mcp:slice_file_lines", principal)

        clean_path = file_path.strip().lstrip("/")
        requested_count = end_line - start_line + 1
        if requested_count > MAX_SLICE_LINES:
            raise ValueError(
                f"Requested slice of {requested_count} lines exceeds "
                f"maximum allowed {MAX_SLICE_LINES} lines"
            )

        with scoped_db_session(conn_mgr, principal) as conn:
            row = conn.execute(
                """
                SELECT git_commit_sha, git_blob_sha, raw_content, line_count
                FROM code_files
                WHERE domain_id = %s
                  AND repo_name = %s
                  AND (file_path = %s OR file_path ILIKE %s)
                LIMIT 1;
                """,
                (DOMAIN_ID, repo_name, clean_path, f"%{clean_path}"),
            ).fetchone()

            if not row:
                raise FileNotFoundError(
                    f"File '{file_path}' not found in repo '{repo_name}' under active RLS scopes"
                )

            commit_sha = str(row[0])
            blob_sha = str(row[1])
            raw_content = str(row[2])
            total_lines = int(str(row[3]))

        # Validate bounds (raises InvalidLineBoundsError or OffByOneBoundsError on violation)
        try:
            slicer.validate_bounds(start_line, end_line, total_lines)
        except (InvalidLineBoundsError, OffByOneBoundsError) as err:
            raise ValueError(f"Invalid line bounds [{start_line}, {end_line}]: {err}") from err

        sliced_content = slicer.slice_lines(raw_content, start_line, end_line)
        lines_data: list[dict[str, Any]] = []
        sliced_lines = sliced_content.splitlines()
        for line_no, text in zip(range(start_line, end_line + 1), sliced_lines, strict=False):
            lines_data.append({"line_number": line_no, "content": text})

        payload = {
            "file_path": clean_path,
            "repo_name": repo_name,
            "git_commit_sha": commit_sha,
            "git_blob_sha": blob_sha,
            "start_line": start_line,
            "end_line": end_line,
            "total_file_lines": total_lines,
            "slice_line_count": len(lines_data),
            "lines": lines_data,
        }
        return bound_payload_dict(payload)

    # -------------------------------------------------------------------------
    # Canonical Tool 5: search_github_issues
    # -------------------------------------------------------------------------
    @server.tool(
        name="search_github_issues",
        description="Search architectural GitHub issues and resolution logs under Forced RLS.",
    )
    async def search_github_issues(
        query: str,
        state: str | None = None,
        limit: int = 5,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Searches GitHub issues stored in doc_chunks under domain trading_code_scout."""
        validate_tool_call_arguments({"query": query, "state": state, "limit": limit})
        principal = get_current_principal()
        auth_gate.authorize_tool_call(DOMAIN_ID, "code_repo_mcp:search_github_issues", principal)

        clamped_limit = min(max(1, limit), MAX_PAYLOAD_ITEMS)
        clean_state = state.strip().lower() if state else None

        with scoped_db_session(conn_mgr, principal) as conn:
            sql = """
                SELECT doc_id, doc_title, source_uri, section_heading,
                       verbatim_content, metadata_json,
                       ts_rank_cd(fts_doc, plainto_tsquery('english', %s)) AS fts_score,
                       similarity(verbatim_content, %s) AS trgm_score
                FROM doc_chunks
                WHERE domain_id = %s
                  AND doc_id LIKE 'gh-issue:%%'
                  AND (%s::text IS NULL OR metadata_json->>'state' = %s)
                  AND (
                      fts_doc @@ plainto_tsquery('english', %s)
                      OR to_tsvector('english', doc_title || ' ' || verbatim_content)
                         @@ plainto_tsquery('english', %s)
                      OR verbatim_content %% %s
                      OR doc_title ILIKE %s
                      OR verbatim_content ILIKE %s
                  )
                ORDER BY (doc_title ILIKE %s) DESC, fts_score DESC, trgm_score DESC
                LIMIT %s;
            """
            wildcard = f"%{query}%"
            rows = conn.execute(
                sql,
                (
                    query,
                    query,
                    DOMAIN_ID,
                    clean_state,
                    clean_state,
                    query,
                    query,
                    query,
                    wildcard,
                    wildcard,
                    wildcard,
                    clamped_limit,
                ),
            ).fetchall()

            issues: list[dict[str, Any]] = []
            for r in rows:
                doc_id = str(r[0])
                title = str(r[1])
                source_uri = str(r[2])
                section = str(r[3])
                content = str(r[4])
                meta = r[5] if isinstance(r[5], dict) else {}

                issues.append(
                    {
                        "doc_id": doc_id,
                        "title": title,
                        "source_uri": source_uri,
                        "section_heading": section,
                        "state": meta.get("state", "unknown"),
                        "labels": meta.get("labels", []),
                        "excerpt": content[:400],
                    }
                )

        payload = {
            "query": query,
            "state_filter": clean_state,
            "total_matches": len(issues),
            "issues": issues,
        }
        return bound_payload_dict(payload)

    # -------------------------------------------------------------------------
    # Legacy Tool Aliases (for backward compatibility)
    # -------------------------------------------------------------------------
    @server.tool(name="search_ast_symbols", description="Legacy alias for lookup_ast_symbol.")
    async def search_ast_symbols(
        symbol_name: str,
        limit: int = 5,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return await lookup_ast_symbol(symbol_name=symbol_name, limit=limit, ctx=ctx)

    @server.tool(name="get_symbol_definition", description="Legacy alias for lookup_ast_symbol.")
    async def get_symbol_definition(
        symbol_name: str,
        limit: int = 1,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return await lookup_ast_symbol(symbol_name=symbol_name, limit=limit, ctx=ctx)

    @server.tool(name="get_symbol_callers", description="Legacy alias for expand_ast_1hop_closure.")
    async def get_symbol_callers(
        symbol_name: str,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return await expand_ast_1hop_closure(symbol_name=symbol_name, ctx=ctx)

    @server.tool(name="grep_repo", description="Legacy alias for search_code_trigram.")
    async def grep_repo(
        query: str,
        file_pattern: str | None = None,
        limit: int = 10,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return await search_code_trigram(
            query=query, file_pattern=file_pattern, limit=limit, ctx=ctx
        )

    @server.tool(name="list_github_issues", description="Legacy alias for search_github_issues.")
    async def list_github_issues(
        query: str,
        state: str | None = None,
        limit: int = 5,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return await search_github_issues(query=query, state=state, limit=limit, ctx=ctx)

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


__all__ = ["create_code_repo_mcp_server"]
