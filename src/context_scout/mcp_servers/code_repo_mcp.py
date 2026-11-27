"""Code Repository Enterprise MCP Server.

Implements typed read tools backed by live PostgreSQL 17 Forced RLS,
ASTDependencyExpander, and DeterministicSlicer under out-of-band context (SEC-04).
"""

from __future__ import annotations

import difflib
import logging
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Any, Final
from uuid import UUID

import anyio
from mcp.server._otel import OpenTelemetryMiddleware
from mcp.server.mcpserver import Context, MCPServer
from starlette.types import ASGIApp

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.ast_engine.dependency_expander import ASTDependencyExpander
from context_scout.ast_engine.deterministic_slicer import (
    DeterministicSlicer,
    InvalidLineBoundsError,
    OffByOneBoundsError,
)
from context_scout.config import ScoutSettings
from context_scout.db.repositories import ASTSymbolRecord, PostgresRLSRepository
from context_scout.db.session import RLSConnectionManager
from context_scout.mcp_servers.common import (
    MAX_PAYLOAD_ITEMS,
    PostgresScoutGraphAdapter,
    bound_payload_dict,
    create_stateless_mcp_http_app,
)
from context_scout.plugins.registry import PluginRegistry
from context_scout.security.rbac import (
    DomainAuthorizationGate,
    get_current_principal,
    scoped_db_session,
    validate_tool_call_arguments,
)

logger = logging.getLogger(__name__)

MAX_SLICE_LINES: Final[int] = 200


def _validate_clean_file_path(raw_path: str) -> str:
    """Validates and normalizes repository relative paths, blocking traversal and NUL bytes."""
    if "\x00" in raw_path or "\\" in raw_path:
        raise ValueError(f"Unsafe file_path rejected (ASI02): {raw_path!r}")
    stripped = raw_path.strip().lstrip("/")
    if not stripped:
        raise ValueError("file_path must not be empty.")
    parts = PurePosixPath(stripped).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Path traversal segment rejected in file_path (ASI02): {raw_path!r}")
    return "/".join(parts)


def _escape_sql_like(value: str) -> str:
    """Escapes SQL LIKE/ILIKE wildcards (\\, %, _) for ESCAPE '\\' queries."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _resolve_default_code_domain_id(
    settings: ScoutSettings,
    domain_id: str | None = None,
) -> str:
    """Resolves target code domain ID from override, settings, or registered plugins."""
    if domain_id:
        return domain_id
    if settings.default_domain_id and settings.default_domain_id != "default":
        return settings.default_domain_id
    primary_plugin = PluginRegistry.get_primary_domain_plugin()
    if primary_plugin is not None:
        return primary_plugin.get_role_domain_ids().get("code", settings.default_domain_id)
    return settings.default_domain_id


def create_code_repo_mcp_server(
    settings: ScoutSettings | None = None,
    connection_manager: RLSConnectionManager | None = None,
    domain_id: str | None = None,
) -> MCPServer:
    """Factory creating the code_repo_mcp MCP Server with out-of-band security controls."""
    st = settings or ScoutSettings()
    default_domain_id = _resolve_default_code_domain_id(st, domain_id=domain_id)
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

    server = MCPServer("code_repo_mcp", middleware=[OpenTelemetryMiddleware()])

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

        def _sync() -> dict[str, Any]:
            validate_tool_call_arguments(
                {"query": query, "file_pattern": file_pattern, "limit": limit}
            )
            principal = get_current_principal()
            auth_gate.authorize_tool_call(
                default_domain_id, "code_repo_mcp:search_code_trigram", principal
            )

            clamped_limit = min(max(1, limit), MAX_PAYLOAD_ITEMS)
            clean_pattern = (
                f"%{_escape_sql_like(file_pattern.strip())}%"
                if file_pattern and file_pattern.strip()
                else None
            )

            with scoped_db_session(conn_mgr, principal) as conn:
                sql = """
                    SELECT file_path, repo_name, raw_content,
                           (similarity(raw_content, %s) * COALESCE(ranking_weight, 1.0)) AS sim
                    FROM code_files
                    WHERE domain_id = %s
                      AND (%s::text IS NULL OR file_path ILIKE %s ESCAPE '\\')
                      AND (raw_content %% %s OR raw_content ILIKE %s ESCAPE '\\')
                    ORDER BY sim DESC, file_path ASC
                    LIMIT %s;
                """
                search_str = f"%{_escape_sql_like(query)}%"
                rows = conn.execute(
                    sql,
                    (
                        query,
                        default_domain_id,
                        clean_pattern,
                        clean_pattern,
                        query,
                        search_str,
                        clamped_limit,
                    ),
                ).fetchall()

                matches: list[dict[str, Any]] = []
                for r in rows:
                    f_path = str(r[0])
                    r_name = str(r[1])
                    content = str(r[2])
                    sim = float(str(r[3]))

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

            return bound_payload_dict(
                {
                    "query": query,
                    "file_pattern": file_pattern,
                    "total_matches": len(matches),
                    "matches": matches,
                }
            )

        return await anyio.to_thread.run_sync(_sync)

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

        def _sync() -> dict[str, Any]:
            validate_tool_call_arguments({"symbol_name": symbol_name, "limit": limit})
            principal = get_current_principal()
            auth_gate.authorize_tool_call(
                default_domain_id, "code_repo_mcp:lookup_ast_symbol", principal
            )

            clamped_limit = min(max(1, limit), MAX_PAYLOAD_ITEMS)
            clean_name = symbol_name.strip()
            wildcard_name = f"%{_escape_sql_like(clean_name)}%"

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
                          OR s.qualified_name ILIKE %s ESCAPE '\\'
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
                        default_domain_id,
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

            return bound_payload_dict(
                {
                    "symbol_name": clean_name,
                    "total_matches": len(symbols),
                    "symbols": symbols,
                }
            )

        return await anyio.to_thread.run_sync(_sync)

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

        def _sync() -> dict[str, Any]:
            validate_tool_call_arguments({"symbol_name": symbol_name})
            principal = get_current_principal()
            auth_gate.authorize_tool_call(
                default_domain_id, "code_repo_mcp:expand_ast_1hop_closure", principal
            )

            clean_name = symbol_name.strip()
            wildcard_name = f"%{_escape_sql_like(clean_name)}%"

            with scoped_db_session(conn_mgr, principal) as conn:
                row = conn.execute(
                    """
                    SELECT symbol_id, file_id, domain_id, qualified_name, symbol_kind,
                           start_line, end_line, signature_skeleton, contextual_summary,
                           allowed_scopes, embedding
                    FROM ast_symbols
                    WHERE domain_id = %s
                      AND (qualified_name = %s OR qualified_name ILIKE %s ESCAPE '\\')
                    ORDER BY (qualified_name = %s) DESC
                    LIMIT 1;
                    """,
                    (default_domain_id, clean_name, wildcard_name, clean_name),
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
                    emb_floats = [
                        float(x.strip()) for x in raw_emb.strip("[]").split(",") if x.strip()
                    ]
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
                    symbol_kind=str(row[4]),
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
                domain_id=default_domain_id,
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

            return bound_payload_dict(
                {
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
            )

        return await anyio.to_thread.run_sync(_sync)

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

        def _sync() -> dict[str, Any]:
            validate_tool_call_arguments(
                {
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "repo_name": repo_name,
                }
            )
            principal = get_current_principal()
            auth_gate.authorize_tool_call(
                default_domain_id, "code_repo_mcp:slice_file_lines", principal
            )

            clean_path = _validate_clean_file_path(file_path)
            like_pattern = f"%{_escape_sql_like(clean_path)}"
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
                      AND (file_path = %s OR file_path ILIKE %s ESCAPE '\\')
                    LIMIT 1;
                    """,
                    (default_domain_id, repo_name, clean_path, like_pattern),
                ).fetchone()

                if not row:
                    raise FileNotFoundError(
                        f"File '{file_path}' not found in repo '{repo_name}' under RLS scopes"
                    )

                commit_sha = str(row[0])
                blob_sha = str(row[1])
                raw_content = str(row[2])
                total_lines = int(str(row[3]))

            try:
                slicer.validate_bounds(start_line, end_line, total_lines)
            except (InvalidLineBoundsError, OffByOneBoundsError) as err:
                raise ValueError(f"Invalid line bounds [{start_line}, {end_line}]: {err}") from err

            sliced_content = slicer.slice_lines(raw_content, start_line, end_line)
            lines_data: list[dict[str, Any]] = []
            sliced_lines = sliced_content.splitlines()
            for line_no, text in zip(range(start_line, end_line + 1), sliced_lines, strict=False):
                lines_data.append({"line_number": line_no, "content": text})

            return bound_payload_dict(
                {
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
            )

        return await anyio.to_thread.run_sync(_sync)

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
        """Searches GitHub issues stored in doc_chunks under the configured code domain."""

        def _sync() -> dict[str, Any]:
            validate_tool_call_arguments({"query": query, "state": state, "limit": limit})
            principal = get_current_principal()
            auth_gate.authorize_tool_call(
                default_domain_id, "code_repo_mcp:search_github_issues", principal
            )

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
                      AND is_stale = FALSE
                      AND doc_id LIKE 'gh-issue:%%'
                      AND (%s::text IS NULL OR metadata_json->>'state' = %s)
                      AND (
                          fts_doc @@ plainto_tsquery('english', %s)
                          OR to_tsvector('english', doc_title || ' ' || verbatim_content)
                             @@ plainto_tsquery('english', %s)
                          OR verbatim_content %% %s
                          OR doc_title ILIKE %s ESCAPE '\\'
                          OR verbatim_content ILIKE %s ESCAPE '\\'
                      )
                    ORDER BY (doc_title ILIKE %s ESCAPE '\\') DESC, fts_score DESC, trgm_score DESC
                    LIMIT %s;
                """
                wildcard = f"%{_escape_sql_like(query)}%"
                rows = conn.execute(
                    sql,
                    (
                        query,
                        query,
                        default_domain_id,
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

            return bound_payload_dict(
                {
                    "query": query,
                    "state_filter": clean_state,
                    "total_matches": len(issues),
                    "issues": issues,
                }
            )

        return await anyio.to_thread.run_sync(_sync)

    # -------------------------------------------------------------------------
    # Canonical Tool 6: assemble_verified_context_pack (High-Level IDE Tool)
    # -------------------------------------------------------------------------
    @server.tool(
        name="assemble_verified_context_pack",
        description=(
            "Run the complete 4-stage Enterprise Context Scout DAG (live Git/working-tree "
            "incremental sync, 3-way hybrid retrieval, 1-hop AST dependency closure, and "
            "4-check PreFlightValidator with $0 local bounce-back) and return a verified "
            "ImplementationContextPack in a single call for IDE agents."
        ),
    )
    async def assemble_verified_context_pack(
        query: str,
        domain_id: str | None = None,
        max_bounce_backs: int = 1,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Assembles a verified ImplementationContextPack under out-of-band PrincipalContext."""

        def _sync() -> dict[str, Any]:
            target_domain_id = domain_id or default_domain_id
            validate_tool_call_arguments(
                {
                    "query": query,
                    "domain_id": target_domain_id,
                    "max_bounce_backs": max_bounce_backs,
                }
            )
            principal = get_current_principal()
            registry.load_all()
            auth_gate.authorize_tool_call(
                target_domain_id, "code_repo_mcp:assemble_verified_context_pack", principal
            )

            from context_scout.agents.scout_workflow import ScoutWorkflow

            clamped_bounces = min(max(0, max_bounce_backs), 3)
            workflow = ScoutWorkflow(
                registry=registry,
                repo=graph_adapter,
                slicer=slicer,
                expander=expander,
                auth_gate=auth_gate,
            )
            pack_or_diag = workflow.run(
                user_prompt=query,
                principal=principal,
                domain_id=target_domain_id,
                max_bounce_backs=clamped_bounces,
            )
            dumped: dict[str, Any] = pack_or_diag.model_dump(mode="json")
            return dumped

        return await anyio.to_thread.run_sync(_sync)

    # -------------------------------------------------------------------------
    # Canonical Tool 7: list_repo_branches_and_mrs (M5-02 IDE Branch/MR Tools)
    # -------------------------------------------------------------------------
    @server.tool(
        name="list_repo_branches_and_mrs",
        description="List indexed repository branch and Merge/Pull Request snapshots under RLS.",
    )
    async def list_repo_branches_and_mrs(
        repo_name: str | None = None,
        state: str | None = None,
        limit: int = 20,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Lists Copy-on-Write ref_snapshots visible to principal under Forced RLS."""

        def _sync() -> dict[str, Any]:
            validate_tool_call_arguments({"repo_name": repo_name, "state": state, "limit": limit})
            principal = get_current_principal()
            auth_gate.authorize_tool_call(
                default_domain_id, "code_repo_mcp:list_repo_branches_and_mrs", principal
            )
            clamped_limit = min(max(1, limit), MAX_PAYLOAD_ITEMS)
            clean_repo = repo_name.strip() if repo_name and repo_name.strip() else None
            clean_state = state.strip().lower() if state and state.strip() else None

            pg_repo = PostgresRLSRepository()
            with scoped_db_session(conn_mgr, principal) as conn:
                snaps = pg_repo.list_ref_snapshots(
                    conn,
                    default_domain_id,
                    repo_name=clean_repo,
                    state=clean_state,
                )

            snapshots_out = [
                {
                    "snapshot_id": str(s.snapshot_id),
                    "repo_name": s.repo_name,
                    "ref_name": s.ref_name,
                    "ref_kind": s.ref_kind,
                    "mr_iid": s.mr_iid,
                    "base_commit_sha": s.base_commit_sha,
                    "head_commit_sha": s.head_commit_sha,
                    "state": s.state,
                    "updated_at": s.updated_at.isoformat() if s.updated_at else "",
                }
                for s in snaps[:clamped_limit]
            ]
            return bound_payload_dict(
                {
                    "domain_id": default_domain_id,
                    "repo_name": clean_repo,
                    "state": clean_state,
                    "count": len(snapshots_out),
                    "snapshots": snapshots_out,
                }
            )

        return await anyio.to_thread.run_sync(_sync)

    # -------------------------------------------------------------------------
    # Canonical Tool 8: inspect_mr_overlay_diff (M5-02 CoW MR Diff Tool)
    # -------------------------------------------------------------------------
    @server.tool(
        name="inspect_mr_overlay_diff",
        description="Inspect Copy-on-Write branch/MR overlay unified diffs & changed symbols.",
    )
    async def inspect_mr_overlay_diff(
        repo_name: str,
        mr_iid: int | None = None,
        ref_name: str | None = None,
        file_path: str | None = None,
        context_lines: int = 3,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Computes bounded unified diffs across CoW overlay_file_deltas under Forced RLS."""

        def _sync() -> dict[str, Any]:
            validate_tool_call_arguments(
                {
                    "repo_name": repo_name,
                    "mr_iid": mr_iid,
                    "ref_name": ref_name,
                    "file_path": file_path,
                    "context_lines": context_lines,
                }
            )
            principal = get_current_principal()
            auth_gate.authorize_tool_call(
                default_domain_id, "code_repo_mcp:inspect_mr_overlay_diff", principal
            )
            clean_repo = repo_name.strip()
            clean_ref = ref_name.strip() if ref_name and ref_name.strip() else None
            clean_fp = (
                _validate_clean_file_path(file_path)
                if file_path is not None and file_path.strip()
                else None
            )
            if mr_iid is None and clean_ref is None:
                raise ValueError(
                    "Specify either mr_iid or ref_name to inspect MR/branch overlay diff."
                )

            ctx_n = max(0, min(int(context_lines), 10))
            pg_repo = PostgresRLSRepository()
            with scoped_db_session(conn_mgr, principal) as conn:
                snap = pg_repo.get_ref_snapshot(
                    conn,
                    default_domain_id,
                    repo_name=clean_repo,
                    ref_name=clean_ref,
                    mr_iid=str(mr_iid) if mr_iid is not None else None,
                )
                if snap is None:
                    return bound_payload_dict(
                        {
                            "domain_id": default_domain_id,
                            "repo_name": clean_repo,
                            "mr_iid": mr_iid,
                            "ref_name": clean_ref,
                            "found": False,
                            "count": 0,
                            "deltas": [],
                        }
                    )
                raw_deltas = pg_repo.list_overlay_file_deltas(
                    conn, default_domain_id, snap.snapshot_id
                )
                if clean_fp is not None:
                    raw_deltas = [
                        d
                        for d in raw_deltas
                        if d.file_path == clean_fp or d.previous_file_path == clean_fp
                    ]

                deltas_out: list[dict[str, Any]] = []
                for d in raw_deltas[:MAX_PAYLOAD_ITEMS]:
                    old_path = d.previous_file_path or d.file_path
                    base_text = ""
                    if d.change_type != "added":
                        base_row = conn.execute(
                            """
                            SELECT raw_content
                            FROM code_files
                            WHERE domain_id = %s AND repo_name = %s AND file_path = %s
                            LIMIT 1;
                            """,
                            (default_domain_id, clean_repo, old_path),
                        ).fetchone()
                        if base_row is not None:
                            base_text = str(base_row[0] or "")
                    new_text = "" if d.change_type == "deleted" else (d.raw_content or "")
                    diff_text = "".join(
                        difflib.unified_diff(
                            base_text.splitlines(keepends=True),
                            new_text.splitlines(keepends=True),
                            fromfile=f"a/{old_path}",
                            tofile=f"b/{d.file_path}",
                            n=ctx_n,
                        )
                    )
                    changed_symbols = [
                        str(s.get("qualified_name"))
                        for s in d.symbols_json
                        if isinstance(s, dict) and s.get("qualified_name")
                    ]
                    deltas_out.append(
                        {
                            "delta_id": str(d.delta_id),
                            "file_path": d.file_path,
                            "previous_file_path": d.previous_file_path,
                            "change_type": d.change_type,
                            "language": d.language,
                            "base_blob_sha": d.base_blob_sha,
                            "overlay_blob_sha": d.overlay_blob_sha,
                            "line_count": d.line_count,
                            "changed_symbols": changed_symbols,
                            "unified_diff": diff_text,
                        }
                    )

            return bound_payload_dict(
                {
                    "found": True,
                    "domain_id": default_domain_id,
                    "snapshot_id": str(snap.snapshot_id),
                    "repo_name": snap.repo_name,
                    "ref_name": snap.ref_name,
                    "ref_kind": snap.ref_kind,
                    "mr_iid": snap.mr_iid,
                    "base_commit_sha": snap.base_commit_sha,
                    "head_commit_sha": snap.head_commit_sha,
                    "count": len(deltas_out),
                    "deltas": deltas_out,
                }
            )

        return await anyio.to_thread.run_sync(_sync)

    # -------------------------------------------------------------------------
    # Canonical Tool 9: slice_branch_or_mr_file (M5-02 CoW Overlay File Slicer)
    # -------------------------------------------------------------------------
    @server.tool(
        name="slice_branch_or_mr_file",
        description="Slice verbatim lines from a branch or MR CoW overlay snapshot under RLS.",
    )
    async def slice_branch_or_mr_file(
        repo_name: str,
        file_path: str,
        start_line: int,
        end_line: int,
        mr_iid: int | None = None,
        ref_name: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Resolves effective CoW overlay file and slices exact line range."""

        def _sync() -> dict[str, Any]:
            validate_tool_call_arguments(
                {
                    "repo_name": repo_name,
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "mr_iid": mr_iid,
                    "ref_name": ref_name,
                }
            )
            principal = get_current_principal()
            auth_gate.authorize_tool_call(
                default_domain_id, "code_repo_mcp:slice_branch_or_mr_file", principal
            )
            clean_repo = repo_name.strip()
            clean_fp = _validate_clean_file_path(file_path)
            clean_ref = ref_name.strip() if ref_name and ref_name.strip() else None
            if mr_iid is None and clean_ref is None:
                raise ValueError("Specify either mr_iid or ref_name to slice branch/MR file.")
            requested_count = end_line - start_line + 1
            if requested_count > MAX_SLICE_LINES:
                raise ValueError(
                    f"Requested slice of {requested_count} lines exceeds "
                    f"maximum allowed {MAX_SLICE_LINES} lines"
                )

            pg_repo = PostgresRLSRepository()
            with scoped_db_session(conn_mgr, principal) as conn:
                snap = pg_repo.get_ref_snapshot(
                    conn,
                    default_domain_id,
                    repo_name=clean_repo,
                    ref_name=clean_ref,
                    mr_iid=str(mr_iid) if mr_iid is not None else None,
                )
                if snap is None:
                    raise FileNotFoundError(
                        f"Branch/MR snapshot not found in repo '{clean_repo}' under RLS scopes."
                    )
                resolved = pg_repo.get_overlay_file(
                    conn, default_domain_id, snap.snapshot_id, clean_fp
                )
                if resolved is None:
                    raise FileNotFoundError(
                        f"File '{clean_fp}' not found on snapshot '{snap.ref_name}' under RLS."
                    )

            try:
                slicer.validate_bounds(start_line, end_line, resolved.line_count)
            except (InvalidLineBoundsError, OffByOneBoundsError) as err:
                raise ValueError(f"Invalid line bounds [{start_line}, {end_line}]: {err}") from err

            sliced_content = slicer.slice_lines(resolved.raw_content, start_line, end_line)
            lines_data: list[dict[str, Any]] = [
                {"line_number": ln, "content": txt}
                for ln, txt in zip(
                    range(start_line, end_line + 1), sliced_content.splitlines(), strict=False
                )
            ]
            return bound_payload_dict(
                {
                    "file_path": resolved.file_path,
                    "repo_name": resolved.repo_name,
                    "ref_name": snap.ref_name,
                    "mr_iid": snap.mr_iid,
                    "source_layer": resolved.source_layer,
                    "git_commit_sha": resolved.git_commit_sha,
                    "git_blob_sha": resolved.git_blob_sha,
                    "start_line": start_line,
                    "end_line": end_line,
                    "total_file_lines": resolved.line_count,
                    "slice_line_count": len(lines_data),
                    "lines": lines_data,
                }
            )

        return await anyio.to_thread.run_sync(_sync)

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


def create_code_repo_mcp_http_app(
    settings: ScoutSettings | None = None,
    connection_manager: RLSConnectionManager | None = None,
    domain_id: str | None = None,
    *,
    streamable_http_path: str = "/mcp",
    allowed_hosts: Sequence[str] | None = None,
    require_auth: bool = True,
) -> ASGIApp:
    """Builds a stateless MCP 2026-07-28 Streamable HTTP ASGI app for code_repo_mcp."""
    st = settings or ScoutSettings()
    server = create_code_repo_mcp_server(
        settings=st,
        connection_manager=connection_manager,
        domain_id=domain_id,
    )
    return create_stateless_mcp_http_app(
        server,
        streamable_http_path=streamable_http_path,
        json_response=True,
        allowed_hosts=allowed_hosts,
        require_auth=require_auth,
        settings=st,
    )


__all__ = ["create_code_repo_mcp_http_app", "create_code_repo_mcp_server"]
