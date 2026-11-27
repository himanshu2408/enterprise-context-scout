"""Shared MCP Utilities: Out-of-band context, payload caps, and Postgres graph adapter."""

from __future__ import annotations

import base64
import copy
import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, Final
from uuid import UUID

from fastapi import HTTPException
from mcp.server.mcpserver import MCPServer
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from mcp_types.version import HANDSHAKE_PROTOCOL_VERSIONS, MODERN_PROTOCOL_VERSIONS
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTNeighborhoodRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
    JiraIssueRecord,
    ScoutSessionRecord,
    SideEffectRecord,
    _parse_market_scope,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.schemas.contracts import DocumentScope, PrincipalContext
from context_scout.security.betrvg_privacy import pseudonymize_actor_if_enabled
from context_scout.security.rbac import (
    FORBIDDEN_TOOL_ARGUMENTS,
    bind_session_principal,
    get_current_principal,
    scoped_db_session,
    validate_tool_call_arguments,
)
from context_scout.telemetry import (
    ATTR_SCOUT_PRINCIPAL_ID,
    ScoutTelemetryTracer,
    extract_trace_context,
    inject_trace_context,
)

logger = logging.getLogger(__name__)

MAX_PAYLOAD_ITEMS: Final[int] = 25
MAX_PAYLOAD_BYTES: Final[int] = 131_072  # 128 KB hard serialized JSON cap
TRUNCATION_INDICATOR: Final[str] = "[PAYLOAD_TRUNCATED]"


def _as_uuid(val: object) -> UUID:
    return val if isinstance(val, UUID) else UUID(str(val))


def _serialized_byte_len(data: dict[str, Any]) -> int:
    return len(json.dumps(data, default=str).encode("utf-8"))


def bound_payload_dict(
    payload: dict[str, Any],
    max_items: int = MAX_PAYLOAD_ITEMS,
    max_bytes: int = MAX_PAYLOAD_BYTES,
) -> dict[str, Any]:
    """Clamps item counts to <= max_items and enforces <= max_bytes serialized JSON limit."""
    data = copy.deepcopy(payload)

    # 1. Clamp any collection fields to max_items
    collection_keys = (
        "items",
        "matches",
        "symbols",
        "hits",
        "issues",
        "equations",
        "tables",
        "comparison",
        "snapshots",
        "deltas",
        "files",
        "branches",
    )
    for key in collection_keys:
        if key in data and isinstance(data[key], list) and len(data[key]) > max_items:
            data[key] = data[key][:max_items]

    # 2. Check byte size
    if _serialized_byte_len(data) <= max_bytes:
        return data

    data["payload_truncated"] = True
    data["truncation_notice"] = TRUNCATION_INDICATOR

    # 3. Drop trailing list items cleanly first so retained items remain un-truncated
    for key in collection_keys:
        if key in data and isinstance(data[key], list) and len(data[key]) > 1:
            while len(data[key]) > 1:
                data[key].pop()
                if _serialized_byte_len(data) <= max_bytes:
                    return data

    # 4. Only if a single item alone still exceeds max_bytes, slice its largest text field(s)
    # proportionally to fit max_bytes (never hardcoded [:150] mutilation).
    text_fields = (
        "verbatim_content",
        "raw_content",
        "snippet",
        "excerpt",
        "contextual_summary",
        "signature_skeleton",
        "content",
    )
    suffix = f"... {TRUNCATION_INDICATOR}"
    for key in collection_keys:
        if key in data and isinstance(data[key], list) and data[key]:
            single_item = data[key][0]
            if isinstance(single_item, dict):
                while _serialized_byte_len(data) > max_bytes:
                    largest_field: str | None = None
                    largest_len = 0
                    for tf in text_fields:
                        val = single_item.get(tf)
                        if (
                            isinstance(val, str)
                            and not val.endswith(TRUNCATION_INDICATOR)
                            and len(val) > largest_len
                        ):
                            largest_len = len(val)
                            largest_field = tf
                    if largest_field is None or largest_len == 0:
                        break
                    orig_val = str(single_item[largest_field])
                    single_item[largest_field] = suffix
                    base_bytes = _serialized_byte_len(data)
                    available_bytes = max_bytes - base_bytes
                    if available_bytes <= 0:
                        break
                    # Binary-search largest prefix of orig_val that fits within max_bytes
                    lo, hi = 0, len(orig_val)
                    best = 0
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        single_item[largest_field] = orig_val[:mid] + suffix
                        if _serialized_byte_len(data) <= max_bytes:
                            best = mid
                            lo = mid + 1
                        else:
                            hi = mid - 1
                    single_item[largest_field] = orig_val[:best] + suffix

    # Ensure final JSON is strictly within max_bytes
    if _serialized_byte_len(data) > max_bytes:
        # Ultimate guarantee: minimal truncated payload
        return {
            "payload_truncated": True,
            "truncation_notice": TRUNCATION_INDICATOR,
            "error": (
                f"Payload exceeded {max_bytes} byte threshold "
                "and was truncated to minimal representation"
            ),
            "retained_metadata": {
                k: v for k, v in data.items() if not isinstance(v, list) and len(str(v)) < 100
            },
        }

    return data


class PostgresScoutGraphAdapter:
    """Live PostgreSQL 17 graph repository adapter conforming to ScoutGraphRepository (SEC-03)."""

    def __init__(
        self,
        connection_manager: RLSConnectionManager,
        settings: ScoutSettings | None = None,
    ) -> None:
        self.connection_manager = connection_manager
        self.settings = settings or ScoutSettings()

    def list_visible_code_files(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        limit: int = 5_000,
        include_raw_content: bool = True,
    ) -> list[CodeFileRecord]:
        """Returns visible code files for the principal under Forced RLS (bounded by limit)."""
        safe_limit = max(1, min(limit, 5000))
        content_col = "raw_content" if include_raw_content else "'' AS raw_content"
        with scoped_db_session(self.connection_manager, principal) as conn:
            rows = conn.execute(
                f"""
                SELECT file_id, domain_id, repo_name, file_path, git_commit_sha,
                       git_blob_sha, language, allowed_scopes, {content_col}, line_count
                FROM code_files
                WHERE domain_id = %s
                ORDER BY file_path
                LIMIT %s;
                """,
                (domain_id, safe_limit),
            ).fetchall()
            results: list[CodeFileRecord] = []
            for r in rows:
                scopes_raw = r[7]
                scopes_list = (
                    [str(s) for s in scopes_raw] if isinstance(scopes_raw, (list, tuple)) else []
                )
                results.append(
                    CodeFileRecord(
                        file_id=_as_uuid(r[0]),
                        domain_id=str(r[1]),
                        repo_name=str(r[2]),
                        file_path=str(r[3]),
                        git_commit_sha=str(r[4]),
                        git_blob_sha=str(r[5]),
                        language=str(r[6]),
                        allowed_scopes=scopes_list,
                        raw_content=str(r[8]),
                        line_count=int(str(r[9])),
                        tenant_id=principal.tenant_id,
                    )
                )
            return results

    def list_visible_ast_symbols(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        include_embeddings: bool = False,
    ) -> list[ASTSymbolRecord]:
        """Returns all AST symbols visible to the principal under Forced RLS."""
        cols = (
            "symbol_id, file_id, domain_id, qualified_name, symbol_kind, "
            "start_line, end_line, signature_skeleton, contextual_summary, "
            "allowed_scopes"
        )
        if include_embeddings:
            cols += ", embedding"
        with scoped_db_session(self.connection_manager, principal) as conn:
            rows = conn.execute(
                f"""
                SELECT {cols}
                FROM ast_symbols
                WHERE domain_id = %s
                ORDER BY qualified_name;
                """,
                (domain_id,),
            ).fetchall()
            results: list[ASTSymbolRecord] = []
            for r in rows:
                emb_floats: list[float] = []
                if include_embeddings and len(r) > 10:
                    raw_emb = r[10]
                    if isinstance(raw_emb, str):
                        emb_floats = [
                            float(x.strip()) for x in raw_emb.strip("[]").split(",") if x.strip()
                        ]
                    elif hasattr(raw_emb, "tolist"):
                        emb_floats = [float(x) for x in raw_emb.tolist()]
                    elif isinstance(raw_emb, (list, tuple)):
                        emb_floats = [float(x) for x in raw_emb]

                scopes_raw = r[9]
                scopes_list = (
                    [str(s) for s in scopes_raw] if isinstance(scopes_raw, (list, tuple)) else []
                )
                kind_str = str(r[4])
                results.append(
                    ASTSymbolRecord(
                        symbol_id=_as_uuid(r[0]),
                        file_id=_as_uuid(r[1]),
                        domain_id=str(r[2]),
                        qualified_name=str(r[3]),
                        symbol_kind=kind_str,
                        start_line=int(str(r[5])),
                        end_line=int(str(r[6])),
                        signature_skeleton=str(r[7]),
                        contextual_summary=str(r[8]),
                        allowed_scopes=scopes_list,
                        embedding=emb_floats,
                    )
                )
            return results

    def search_visible_ast_symbols(
        self,
        principal: PrincipalContext,
        domain_id: str,
        query_text: str,
        limit: int = 50,
    ) -> list[ASTSymbolRecord]:
        """Searches visible AST symbols via SQL pushdown (ILIKE + LIMIT) under Forced RLS."""
        import re

        words = [
            w.lower()
            for w in re.findall(r"[A-Za-z0-9_]{3,}", query_text)
            if w.lower()
            not in {
                "the",
                "and",
                "for",
                "with",
                "how",
                "what",
                "where",
                "when",
                "from",
                "into",
                "this",
                "that",
            }
        ]
        if not words:
            return self.list_visible_ast_symbols(principal, domain_id)[:limit]

        score_parts: list[str] = []
        where_parts: list[str] = []
        score_params: list[object] = []
        where_params: list[object] = []
        for w in words[:8]:
            pat = f"%{w}%"
            score_parts.append(
                "(CASE WHEN qualified_name ILIKE %s THEN 6 ELSE 0 END + "
                "CASE WHEN signature_skeleton ILIKE %s THEN 3 ELSE 0 END + "
                "CASE WHEN contextual_summary ILIKE %s THEN 1 ELSE 0 END)"
            )
            score_params.extend([pat, pat, pat])
            where_parts.append(
                "(qualified_name ILIKE %s OR signature_skeleton ILIKE %s "
                "OR contextual_summary ILIKE %s)"
            )
            where_params.extend([pat, pat, pat])

        score_expr = " + ".join(score_parts)
        where_expr = " OR ".join(where_parts)
        sql = f"""
            SELECT symbol_id, file_id, domain_id, qualified_name, symbol_kind,
                   start_line, end_line, signature_skeleton, contextual_summary,
                   allowed_scopes
            FROM ast_symbols
            WHERE domain_id = %s
              AND ({where_expr})
            ORDER BY ({score_expr}) DESC, qualified_name ASC
            LIMIT %s;
        """
        full_params: list[object] = [domain_id, *where_params, *score_params, limit]
        with scoped_db_session(self.connection_manager, principal) as conn:
            rows = conn.execute(sql, tuple(full_params)).fetchall()
            results: list[ASTSymbolRecord] = []
            for r in rows:
                scopes_raw = r[9]
                scopes_list = (
                    [str(s) for s in scopes_raw] if isinstance(scopes_raw, (list, tuple)) else []
                )
                results.append(
                    ASTSymbolRecord(
                        symbol_id=_as_uuid(r[0]),
                        file_id=_as_uuid(r[1]),
                        domain_id=str(r[2]),
                        qualified_name=str(r[3]),
                        symbol_kind=str(r[4]),
                        start_line=int(str(r[5])),
                        end_line=int(str(r[6])),
                        signature_skeleton=str(r[7]),
                        contextual_summary=str(r[8]),
                        allowed_scopes=scopes_list,
                        embedding=[],
                    )
                )
            return results

    def get_code_files_by_ids(
        self,
        principal: PrincipalContext,
        domain_id: str,
        file_ids: list[UUID | str],
    ) -> list[CodeFileRecord]:
        """Fetches specific code files by UUID under Forced RLS without scanning whole table."""
        valid_uuids: list[str] = []
        for fid in file_ids:
            try:
                valid_uuids.append(str(UUID(str(fid))))
            except ValueError as exc:
                logger.debug(
                    "Skipping invalid code file UUID %r (%s): %s",
                    fid,
                    type(exc).__name__,
                    exc,
                )
                continue
        if not valid_uuids:
            return []
        with scoped_db_session(self.connection_manager, principal) as conn:
            rows = conn.execute(
                """
                SELECT file_id, domain_id, repo_name, file_path, git_commit_sha,
                       git_blob_sha, language, allowed_scopes, raw_content, line_count
                FROM code_files
                WHERE domain_id = %s
                  AND file_id = ANY(%s::uuid[])
                ORDER BY file_path;
                """,
                (domain_id, valid_uuids),
            ).fetchall()
            results: list[CodeFileRecord] = []
            for r in rows:
                scopes_raw = r[7]
                scopes_list = (
                    [str(s) for s in scopes_raw] if isinstance(scopes_raw, (list, tuple)) else []
                )
                results.append(
                    CodeFileRecord(
                        file_id=_as_uuid(r[0]),
                        domain_id=str(r[1]),
                        repo_name=str(r[2]),
                        file_path=str(r[3]),
                        git_commit_sha=str(r[4]),
                        git_blob_sha=str(r[5]),
                        language=str(r[6]),
                        allowed_scopes=scopes_list,
                        raw_content=str(r[8]),
                        line_count=int(str(r[9])),
                    )
                )
            return results

    def list_visible_ast_edges(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[ASTEdgeRecord]:
        """Returns all directed AST call/import edges visible to the principal under Forced RLS."""
        with scoped_db_session(self.connection_manager, principal) as conn:
            rows = conn.execute(
                """
                SELECT edge_id, domain_id, source_symbol_id, target_qualified_name,
                       target_symbol_id, edge_kind, allowed_scopes
                FROM ast_edges
                WHERE domain_id = %s
                ORDER BY target_qualified_name;
                """,
                (domain_id,),
            ).fetchall()
            results: list[ASTEdgeRecord] = []
            for r in rows:
                t_sym_id = _as_uuid(r[4]) if r[4] is not None else None
                scopes_raw = r[6]
                scopes_list = (
                    [str(s) for s in scopes_raw] if isinstance(scopes_raw, (list, tuple)) else []
                )
                results.append(
                    ASTEdgeRecord(
                        edge_id=_as_uuid(r[0]),
                        domain_id=str(r[1]),
                        source_symbol_id=_as_uuid(r[2]),
                        target_qualified_name=str(r[3]),
                        target_symbol_id=t_sym_id,
                        edge_kind=str(r[5]),
                        allowed_scopes=scopes_list,
                    )
                )
            return results

    @staticmethod
    def _row_to_doc_chunk(r: tuple[object, ...]) -> DocChunkRecord:
        meta = r[11]
        if isinstance(meta, str):
            meta_dict = json.loads(meta)
        elif isinstance(meta, dict):
            meta_dict = meta
        else:
            meta_dict = {}

        pub_date = r[6]
        if isinstance(pub_date, str):
            pub_date = date.fromisoformat(pub_date)
        elif not isinstance(pub_date, date):
            pub_date = None

        raw_scopes = r[12]
        scopes_list = [str(s) for s in raw_scopes] if isinstance(raw_scopes, (list, tuple)) else []
        return DocChunkRecord(
            chunk_id=_as_uuid(r[0]),
            domain_id=str(r[1]),
            doc_id=str(r[2]),
            doc_title=str(r[3]),
            source_uri=str(r[4]),
            market_scope=_parse_market_scope(r[5]),
            published_date=pub_date,
            authority_weight=float(str(r[7])),
            section_heading=str(r[8]),
            contextual_header=str(r[9]),
            verbatim_content=str(r[10]),
            metadata_json=meta_dict,
            allowed_scopes=scopes_list,
            embedding=[],
        )

    def list_visible_doc_chunks(
        self,
        principal: PrincipalContext,
        domain_id: str,
        market_scope: DocumentScope | str | None = None,
    ) -> list[DocChunkRecord]:
        """Returns visible document chunks under Forced RLS (bounded to 100 rows)."""
        with scoped_db_session(self.connection_manager, principal) as conn:
            query = """
                SELECT chunk_id, domain_id, doc_id, doc_title, source_uri,
                       market_scope, published_date, authority_weight,
                       section_heading, contextual_header, verbatim_content,
                       metadata_json, allowed_scopes
                FROM doc_chunks
                WHERE domain_id = %s
                  AND is_stale = FALSE
            """
            params: list[object] = [domain_id]
            if market_scope is not None:
                query += " AND market_scope = %s"
                params.append(
                    market_scope.value if hasattr(market_scope, "value") else str(market_scope)
                )
            query += " ORDER BY doc_id, chunk_id LIMIT 100;"
            rows = conn.execute(query, tuple(params)).fetchall()
            return [self._row_to_doc_chunk(r) for r in rows]

    def count_visible_doc_chunks(
        self,
        principal: PrincipalContext,
        domain_id: str,
    ) -> tuple[int, int]:
        """Returns (active_non_stale_count, stale_count) visible to principal under RLS."""
        with scoped_db_session(self.connection_manager, principal) as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE NOT COALESCE(is_stale, FALSE)),
                    COUNT(*) FILTER (WHERE COALESCE(is_stale, FALSE))
                FROM doc_chunks
                WHERE domain_id = %s;
                """,
                (domain_id,),
            ).fetchone()
            if row is None:
                return (0, 0)
            return (int(str(row[0] or 0)), int(str(row[1] or 0)))

    def search_visible_doc_chunks(
        self,
        principal: PrincipalContext,
        domain_id: str,
        query_text: str,
        market_scope: DocumentScope | str | None = None,
        limit: int = 50,
    ) -> list[DocChunkRecord]:
        """Searches visible document chunks ranked by RRF, keyword, and market scope under RLS."""
        if (
            query_text.strip()
            and os.environ.get("SCOUT_USE_HASH_EMBEDDINGS_FOR_TESTS", "").strip() != "1"
            and (
                not os.environ.get("PYTEST_CURRENT_TEST")
                or os.environ.get("SCOUT_ENABLE_HYBRID_RRF_IN_PYTEST", "").strip() == "1"
            )
        ):
            try:
                from context_scout.agents.preflight_gate import _embed_query_for_hybrid_rrf
                from context_scout.db.repositories import PostgresRLSRepository

                query_vec = _embed_query_for_hybrid_rrf(query_text)
                pg_repo = PostgresRLSRepository()
                with scoped_db_session(self.connection_manager, principal) as conn:
                    hits = pg_repo.search_doc_chunks_hybrid_rrf(
                        conn=conn,
                        domain_id=domain_id,
                        query_text=query_text,
                        query_embedding=query_vec,
                        market_scope=market_scope,
                        top_k=limit,
                    )
                if hits:
                    ordered_ids = [str(h.item_id) for h in hits]
                    fetched = self.get_doc_chunks_by_ids(principal, domain_id, ordered_ids)
                    by_id = {str(c.chunk_id): c for c in fetched}
                    ranked_hits = [by_id[cid] for cid in ordered_ids if cid in by_id]
                    if ranked_hits:
                        return ranked_hits[:limit]
            except Exception as exc:
                logger.debug(
                    "PG17 hybrid RRF doc search fallback (%s): %s",
                    type(exc).__name__,
                    exc,
                )

        words = [
            w.lower()
            for w in re.findall(r"[A-Za-z0-9]{3,}", query_text)
            if w.lower()
            not in {
                "the",
                "and",
                "for",
                "with",
                "best",
                "way",
                "how",
                "what",
                "does",
                "stand",
                "stands",
                "mean",
                "means",
                "define",
                "definition",
                "implement",
                "implementing",
                "implementation",
                "strategy",
                "strategies",
                "market",
                "markets",
                "stock",
                "stocks",
                "using",
                "from",
                "into",
            }
        ]
        with scoped_db_session(self.connection_manager, principal) as conn:
            if not words:
                return self.list_visible_doc_chunks(
                    principal=principal,
                    domain_id=domain_id,
                    market_scope=market_scope,
                )

            score_parts: list[str] = []
            where_parts: list[str] = []
            score_params: list[object] = []
            where_params: list[object] = []

            for w in words[:8]:
                pat = f"%{w}%"
                score_parts.append(
                    "(CASE WHEN doc_title ILIKE %s THEN 6 ELSE 0 END + "
                    "CASE WHEN section_heading ILIKE %s THEN 3 ELSE 0 END + "
                    "CASE WHEN verbatim_content ILIKE %s THEN 1 ELSE 0 END)"
                )
                score_params.extend([pat, pat, pat])
                where_parts.append(
                    "(doc_title ILIKE %s OR section_heading ILIKE %s OR verbatim_content ILIKE %s)"
                )
                where_params.extend([pat, pat, pat])

            market_bonus_sql = "0"
            market_order_prefix = ""
            market_bonus_params: list[object] = []
            if market_scope is not None:
                m_val = market_scope.value if hasattr(market_scope, "value") else str(market_scope)
                market_bonus_sql = "(CASE WHEN market_scope = %s THEN 5 ELSE 0 END)"
                market_order_prefix = f"{market_bonus_sql} DESC,"
                market_bonus_params.append(m_val)

            score_expr = " + ".join(score_parts) + f" + {market_bonus_sql}"
            tsquery_str = " | ".join(words[:8])

            fts_sql = f"""
                WITH q AS (
                    SELECT to_tsquery('english', %s) AS tsq
                ),
                candidates AS (
                    SELECT chunk_id, domain_id, doc_id, doc_title, source_uri,
                           market_scope, published_date, authority_weight,
                           section_heading, contextual_header, verbatim_content,
                           metadata_json, allowed_scopes
                    FROM doc_chunks, q
                    WHERE domain_id = %s
                      AND is_stale = FALSE
                      AND fts_doc @@ q.tsq
                    ORDER BY {market_order_prefix}
                             ts_rank_cd(fts_doc, q.tsq) DESC,
                             authority_weight DESC
                    LIMIT 250
                )
                SELECT chunk_id, domain_id, doc_id, doc_title, source_uri,
                       market_scope, published_date, authority_weight,
                       section_heading, contextual_header, verbatim_content,
                       metadata_json, allowed_scopes
                FROM candidates
                ORDER BY ({score_expr}) DESC, authority_weight DESC, doc_id ASC, chunk_id ASC
                LIMIT %s;
            """
            fts_params: list[object] = [
                tsquery_str,
                domain_id,
                *market_bonus_params,
                *score_params,
                *market_bonus_params,
                limit,
            ]
            rows = conn.execute(fts_sql, tuple(fts_params)).fetchall()
            if rows:
                return [self._row_to_doc_chunk(r) for r in rows]

            where_expr = " OR ".join(where_parts)

            sql = f"""
                SELECT chunk_id, domain_id, doc_id, doc_title, source_uri,
                       market_scope, published_date, authority_weight,
                       section_heading, contextual_header, verbatim_content,
                       metadata_json, allowed_scopes
                FROM doc_chunks
                WHERE domain_id = %s
                  AND is_stale = FALSE
                  AND ({where_expr})
                ORDER BY ({score_expr}) DESC, authority_weight DESC, doc_id ASC, chunk_id ASC
                LIMIT %s;
            """
            full_params: list[object] = [
                domain_id,
                *where_params,
                *score_params,
                *market_bonus_params,
                limit,
            ]
            rows = conn.execute(sql, tuple(full_params)).fetchall()
            if rows:
                return [self._row_to_doc_chunk(r) for r in rows]

            return self.list_visible_doc_chunks(
                principal=principal,
                domain_id=domain_id,
                market_scope=market_scope,
            )

    def get_doc_chunks_by_ids(
        self,
        principal: PrincipalContext,
        domain_id: str,
        chunk_ids: list[str],
    ) -> list[DocChunkRecord]:
        """Fetches specific document chunks by UUID under Forced RLS for citation verification."""
        valid_uuids: list[str] = []
        for cid in chunk_ids:
            try:
                valid_uuids.append(str(UUID(str(cid))))
            except ValueError as exc:
                logger.debug(
                    "Skipping invalid doc chunk UUID %r (%s): %s",
                    cid,
                    type(exc).__name__,
                    exc,
                )
                continue
        if not valid_uuids:
            return []
        with scoped_db_session(self.connection_manager, principal) as conn:
            sql = """
                SELECT chunk_id, domain_id, doc_id, doc_title, source_uri,
                       market_scope, published_date, authority_weight,
                       section_heading, contextual_header, verbatim_content,
                       metadata_json, allowed_scopes
                FROM doc_chunks
                WHERE domain_id = %s
                  AND chunk_id = ANY(%s::uuid[])
                  AND is_stale = FALSE;
            """
            rows = conn.execute(sql, (domain_id, valid_uuids)).fetchall()
            return [self._row_to_doc_chunk(r) for r in rows]

    def traverse_1hop_neighborhood(
        self,
        principal: PrincipalContext,
        domain_id: str,
        focus_symbol_id: UUID | str,
        max_direct_callers: int = 3,
    ) -> list[ASTNeighborhoodRecord]:
        """Executes memory-bounded recursive CTE 1-hop AST closure directly in PostgreSQL 17."""
        from context_scout.db.repositories import PostgresRLSRepository

        repo = PostgresRLSRepository()
        with scoped_db_session(self.connection_manager, principal) as conn:
            return repo.traverse_1hop_neighborhood(
                conn=conn,
                domain_id=domain_id,
                focus_symbol_id=focus_symbol_id,
                max_direct_callers=max_direct_callers,
            )

    def list_visible_side_effects(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        effect_category: str | None = None,
        repo_name: str | None = None,
        file_path: str | None = None,
    ) -> list[SideEffectRecord]:
        """Lists side effects visible to principal under Forced RLS."""
        from context_scout.db.repositories import PostgresRLSRepository

        repo = PostgresRLSRepository()
        with scoped_db_session(self.connection_manager, principal) as conn:
            return repo.list_visible_side_effects(
                conn=conn,
                domain_id=domain_id,
                effect_category=effect_category,
                repo_name=repo_name,
                file_path=file_path,
            )

    def list_visible_jira_issues(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        project_key: str | None = None,
    ) -> list[JiraIssueRecord]:
        """Lists Jira issues visible to principal under Forced RLS."""
        from context_scout.db.repositories import PostgresRLSRepository

        repo = PostgresRLSRepository()
        with scoped_db_session(self.connection_manager, principal) as conn:
            return repo.list_jira_issues(conn=conn, domain_id=domain_id, project_key=project_key)

    def get_scout_session(
        self,
        principal: PrincipalContext,
        session_id: str,
    ) -> ScoutSessionRecord | None:
        """Loads a multi-turn scout_sessions record for principal under Forced RLS."""
        from context_scout.db.repositories import PostgresRLSRepository

        repo = PostgresRLSRepository()
        with scoped_db_session(self.connection_manager, principal) as conn:
            return repo.get_scout_session(
                conn=conn,
                session_id=session_id,
                principal_id=principal.user_id,
            )

    def upsert_scout_session(
        self,
        principal: PrincipalContext,
        record: ScoutSessionRecord,
    ) -> ScoutSessionRecord:
        """Upserts a multi-turn scout_sessions record for principal under Forced RLS."""
        from context_scout.db.repositories import PostgresRLSRepository

        repo = PostgresRLSRepository()
        with scoped_db_session(self.connection_manager, principal) as conn:
            return repo.upsert_scout_session(conn=conn, record=record)


MCP_PROTOCOL_VERSION_2026_07_28: Final[str] = "2026-07-28"
SUPPORTED_MCP_PROTOCOL_VERSIONS: Final[tuple[str, ...]] = (
    *MODERN_PROTOCOL_VERSIONS,
    *reversed(HANDSHAKE_PROTOCOL_VERSIONS),
)
_CLOUD_TRACE_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*([0-9a-fA-F]{32})/([0-9]+)(?:;o=([01]))?\s*$"
)
_DEFAULT_ALLOWED_MCP_HOSTS: Final[list[str]] = [
    "127.0.0.1",
    "127.0.0.1:*",
    "localhost",
    "localhost:*",
    "[::1]",
    "[::1]:*",
    "testserver",
    "testserver:*",
    "context-scout-api",
    "context-scout-api:*",
    "*.svc.cluster.local",
    "*.svc.cluster.local:*",
]
_DEFAULT_ALLOWED_MCP_ORIGINS: Final[list[str]] = [
    "http://127.0.0.1",
    "http://127.0.0.1:*",
    "http://localhost",
    "http://localhost:*",
    "http://[::1]",
    "http://[::1]:*",
    "http://testserver",
    "http://context-scout-api",
    "http://context-scout-api:*",
]


def _host_matches_pattern(host: str | None, allowed_hosts: Sequence[str]) -> bool:
    """Validates Host header against exact, `:*` port, and leading `*.` wildcard patterns."""
    if not host:
        return False
    clean_host = host.strip().lower()
    if not clean_host:
        return False

    def _split_host_port(val: str) -> tuple[str, str | None]:
        if val.startswith("["):
            end_br = val.find("]")
            if end_br != -1:
                h_part = val[: end_br + 1]
                rest = val[end_br + 1 :]
                return h_part, rest[1:] if rest.startswith(":") else None
        if ":" in val:
            h_part, p_part = val.rsplit(":", 1)
            return h_part, p_part
        return val, None

    req_name, req_port = _split_host_port(clean_host)
    for raw_pat in allowed_hosts:
        pat = raw_pat.strip().lower()
        if not pat:
            continue
        if pat == clean_host:
            return True
        pat_name, pat_port = _split_host_port(pat)
        port_ok = (pat_port == "*" and req_port is not None) or (pat_port == req_port)
        if not port_ok:
            continue
        if pat_name == req_name:
            return True
        if pat_name.startswith("*.") and len(pat_name) > 2:
            suffix = pat_name[1:]  # '.svc.cluster.local'
            if req_name.endswith(suffix) and len(req_name) > len(suffix):
                return True
    return False


def _origin_matches_pattern(origin: str | None, allowed_origins: Sequence[str]) -> bool:
    """Validates Origin header against exact and `:*` port patterns (empty Origin allowed)."""
    if not origin:
        return True
    clean = origin.strip().lower()
    for raw_pat in allowed_origins:
        pat = raw_pat.strip().lower()
        if not pat:
            continue
        if pat == clean:
            return True
        if pat.endswith(":*") and clean.startswith(pat[:-2] + ":"):
            return True
    return False


def _request_origin_from_scope(scope: Scope) -> str:
    scheme = str(scope.get("scheme") or "http")
    host = ""
    for k, v in scope.get("headers", ()):
        if k.lower() == b"host":
            host = v.decode("latin-1").strip()
            break
    if not host:
        server_info = scope.get("server")
        if isinstance(server_info, tuple) and server_info:
            srv_host = str(server_info[0])
            srv_port = server_info[1] if len(server_info) > 1 else None
            host = f"{srv_host}:{srv_port}" if srv_port else srv_host
        else:
            host = "127.0.0.1"
    return f"{scheme}://{host}"


def _extract_bearer_token_claims(headers: Mapping[str, str]) -> dict[str, Any]:
    """Decodes JSON claims from an already-verified Bearer token (JWT or HMAC payload.sig)."""
    auth = ""
    for k, v in headers.items():
        if k.lower() == "authorization":
            auth = v.strip()
            break
    if not auth.lower().startswith("bearer "):
        return {}
    token = auth[7:].strip()
    if not token:
        return {}
    parts = token.split(".")
    payload_b64 = parts[1] if len(parts) == 3 else (parts[0] if len(parts) == 2 else "")
    if not payload_b64:
        return {}
    try:
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        return decoded if isinstance(decoded, dict) else {}
    except Exception:
        return {}


def _token_matches_expected_resource(claims: Mapping[str, Any], expected_resource: str) -> bool:
    """Checks RFC 8707 `resource` or `aud` claim matches the configured expected resource URI."""
    target = expected_resource.strip().rstrip("/")
    if not target:
        return True
    for claim_key in ("resource", "aud"):
        val = claims.get(claim_key)
        if isinstance(val, str) and val.strip().rstrip("/") == target:
            return True
        if (
            isinstance(val, Sequence)
            and not isinstance(val, (str, bytes))
            and any(isinstance(item, str) and item.strip().rstrip("/") == target for item in val)
        ):
            return True
    return False


def build_oauth_protected_resource_metadata(
    request_origin: str,
    *,
    settings: ScoutSettings | None = None,
    resource_path: str = "/mcp",
) -> dict[str, Any]:
    """Builds RFC 9728 OAuth 2.0 Protected Resource Metadata for stateless MCP endpoints."""
    cfg = settings or ScoutSettings()
    expected = (cfg.mcp_expected_resource or "").strip()
    resource_id = expected or f"{request_origin.rstrip('/')}{resource_path}"
    issuer = (
        os.environ.get("SCOUT_OIDC_ISSUER", "").strip()
        or getattr(cfg, "oidc_issuer", None)
        or request_origin.rstrip("/")
    )
    return {
        "resource": resource_id,
        "authorization_servers": [str(issuer).rstrip("/")],
        "bearer_methods_supported": ["header"],
        "scopes_supported": ["mcp:tools:read", "public_eng", "public_docs"],
    }


def parse_cloud_trace_to_traceparent(cloud_trace: str | None) -> str | None:
    """Converts GCP X-Cloud-Trace-Context (<trace_hex>/<span_dec>;o=<0|1>) to W3C traceparent."""
    if not cloud_trace or not cloud_trace.strip():
        return None
    m = _CLOUD_TRACE_RE.match(cloud_trace)
    if m is None:
        return None
    trace_hex = m.group(1).lower()
    span_dec = int(m.group(2))
    if span_dec <= 0 or trace_hex == "0" * 32:
        return None
    span_hex = f"{span_dec & 0xFFFFFFFFFFFFFFFF:016x}"
    if span_hex == "0" * 16:
        span_hex = "0000000000000001"
    sampled = "01" if (m.group(3) in (None, "1")) else "00"
    return f"00-{trace_hex}-{span_hex}-{sampled}"


def resolve_traceparent_from_headers(
    headers: Mapping[str, str],
    body_payload: Mapping[str, Any] | None = None,
) -> str | None:
    """Extracts W3C traceparent from HTTP headers, Cloud Trace header, or JSON-RPC _meta."""
    for k, v in headers.items():
        if k.lower() == "traceparent" and v.strip():
            return v.strip()
    for k, v in headers.items():
        if k.lower() == "x-cloud-trace-context":
            converted = parse_cloud_trace_to_traceparent(v)
            if converted:
                return converted
    if isinstance(body_payload, Mapping):
        params = body_payload.get("params")
        if isinstance(params, Mapping):
            meta = params.get("_meta")
            if isinstance(meta, Mapping) and isinstance(meta.get("traceparent"), str):
                tp = str(meta["traceparent"]).strip()
                if tp:
                    return tp
    return None


def build_mcp_well_known_metadata(
    server: MCPServer,
    *,
    endpoint_path: str = "/mcp",
    server_name: str | None = None,
    include_tool_schemas: bool = True,
    resource_metadata_url: str = "/.well-known/oauth-protected-resource",
) -> dict[str, Any]:
    """Builds SEP-2243 /.well-known/mcp discovery metadata for stateless MCP 2026-07-28."""
    resolved_name = server_name or getattr(server, "name", None) or "enterprise-context-scout"
    tools_desc: list[dict[str, Any]] = []
    tool_names: list[str] = []
    tool_mgr = getattr(server, "_tool_manager", None)
    if tool_mgr is not None and hasattr(tool_mgr, "list_tools"):
        for tool in tool_mgr.list_tools():
            tool_names.append(tool.name)
            if not include_tool_schemas:
                continue
            raw_props = (
                dict(tool.parameters.get("properties", {}))
                if isinstance(getattr(tool, "parameters", None), dict)
                else {}
            )
            sanitized_props = {
                k: v for k, v in raw_props.items() if k.lower() not in FORBIDDEN_TOOL_ARGUMENTS
            }
            tools_desc.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": {
                        "type": "object",
                        "properties": sanitized_props,
                    },
                }
            )
    payload: dict[str, Any] = {
        "server": resolved_name,
        "protocolVersion": MCP_PROTOCOL_VERSION_2026_07_28,
        "supportedProtocolVersions": list(SUPPORTED_MCP_PROTOCOL_VERSIONS),
        "transport": {
            "type": "streamable-http",
            "stateless": True,
            "endpoint": endpoint_path,
            "sep": "SEP-2575",
        },
        "discovery": {"sep": "SEP-2243"},
        "capabilities": {
            "tools": {"listChanged": False},
            "tracing": {
                "sep": "SEP-2663",
                "w3cTraceContext": True,
                "cloudTraceCompatible": True,
            },
        },
        "authentication": {
            "outOfBandOnly": True,
            "schemes": ["CloudIAP", "OIDC_JWKS_Bearer", "HMAC_Bearer"],
            "resourceMetadataUrl": resource_metadata_url,
        },
        "schemasRedacted": not include_tool_schemas,
        "toolCount": len(tool_names),
        "tools": tools_desc,
    }
    if not include_tool_schemas:
        payload["toolNames"] = tool_names
    return payload


_SINGLE_AUTH_HEADERS: Final[frozenset[str]] = frozenset(
    {"authorization", "x-goog-iap-jwt-assertion"}
)
_SINGLE_CONTEXT_HEADERS: Final[frozenset[str]] = frozenset(
    {
        "x-principal-tenant-id",
        "x-scout-tenant-id",
        "x-principal-user-id",
        "x-principal-role",
        "x-principal-scopes",
        "mcp-session-id",
        "mcp-protocol-version",
    }
)


def _rewrite_headers(
    raw_headers: Sequence[tuple[bytes, bytes]],
    updates: Mapping[str, str],
) -> list[tuple[bytes, bytes]]:
    lower_updates = {k.lower().encode("latin-1"): v.encode("latin-1") for k, v in updates.items()}
    out: list[tuple[bytes, bytes]] = [
        (k, v) for k, v in raw_headers if k.lower() not in lower_updates
    ]
    for k_bytes, v_bytes in lower_updates.items():
        out.append((k_bytes, v_bytes))
    return out


class _StatelessMCPHTTPASGIApp:
    """ASGI wrapper enforcing SEP-2575 stateless MCP 2026-07-28, SEP-2243, SEP-2663, and SEC-04."""

    def __init__(
        self,
        server: MCPServer,
        *,
        streamable_http_path: str = "/mcp",
        json_response: bool = True,
        allowed_hosts: Sequence[str] | None = None,
        allowed_origins: Sequence[str] | None = None,
        require_auth: bool = True,
        settings: ScoutSettings | None = None,
    ) -> None:
        self.server = server
        self.streamable_http_path = streamable_http_path
        self.json_response = json_response
        self.require_auth = require_auth
        self._explicit_settings = settings
        cfg = settings or ScoutSettings()
        resolved_hosts = (
            list(allowed_hosts)
            if allowed_hosts is not None
            else (list(cfg.mcp_allowed_hosts) or list(_DEFAULT_ALLOWED_MCP_HOSTS))
        )
        resolved_origins = (
            list(allowed_origins)
            if allowed_origins is not None
            else (list(cfg.mcp_allowed_origins) or list(_DEFAULT_ALLOWED_MCP_ORIGINS))
        )
        self._allowed_hosts_patterns = resolved_hosts
        self._allowed_origins_patterns = resolved_origins
        self.security_settings = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=resolved_hosts,
            allowed_origins=resolved_origins,
        )
        self.starlette_app = server.streamable_http_app(
            streamable_http_path=streamable_http_path,
            json_response=json_response,
            stateless_http=True,
            transport_security=self.security_settings,
        )

    def _active_settings(self) -> ScoutSettings:
        return self._explicit_settings if self._explicit_settings is not None else ScoutSettings()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.starlette_app(scope, receive, send)
            return

        path = str(scope.get("path") or "").rstrip("/") or "/"
        method = str(scope.get("method") or "GET").upper()
        req_origin = _request_origin_from_scope(scope)
        resource_meta_url = f"{req_origin}/.well-known/oauth-protected-resource"
        www_auth_value = f'Bearer resource_metadata="{resource_meta_url}"'

        raw_headers = list(scope.get("headers", ()))
        folded_headers: dict[str, str] = {}
        seen_keys: set[str] = set()
        for raw_k, raw_v in raw_headers:
            k_lower = raw_k.decode("latin-1").lower()
            v_str = raw_v.decode("latin-1")
            if k_lower in _SINGLE_AUTH_HEADERS and k_lower in seen_keys:
                err_resp = JSONResponse(
                    {"error": f"Duplicate authentication header '{k_lower}' is not permitted."},
                    status_code=401,
                    headers={"WWW-Authenticate": www_auth_value},
                )
                await err_resp(scope, receive, send)
                return
            if k_lower in _SINGLE_CONTEXT_HEADERS and k_lower in seen_keys:
                err_resp = JSONResponse(
                    {"error": f"Duplicate header '{k_lower}' is not permitted."},
                    status_code=400,
                )
                await err_resp(scope, receive, send)
                return
            seen_keys.add(k_lower)
            folded_headers.setdefault(k_lower, v_str)

        # Enforce configurable Host/Origin protection (ECS-001) across exact, :*, and *.suffix rules
        req_host = folded_headers.get("host")
        if self.security_settings.enable_dns_rebinding_protection:
            if not _host_matches_pattern(req_host, self._allowed_hosts_patterns):
                resp421 = Response("Invalid Host header", status_code=421)
                await resp421(scope, receive, send)
                return
            req_origin_hdr = folded_headers.get("origin")
            if not _origin_matches_pattern(req_origin_hdr, self._allowed_origins_patterns):
                resp403 = Response("Invalid Origin header", status_code=403)
                await resp403(scope, receive, send)
                return

        active_settings = self._active_settings()

        # RFC 9728 OAuth 2.0 Protected Resource Metadata
        if method == "GET" and (
            path == "/.well-known/oauth-protected-resource"
            or "/.well-known/oauth-protected-resource" in path
        ):
            prm = build_oauth_protected_resource_metadata(
                req_origin,
                settings=active_settings,
                resource_path=self.streamable_http_path,
            )
            resp = JSONResponse(prm, headers={"Cache-Control": "no-store"})
            await resp(scope, receive, send)
            return

        # SEP-2243 capability discovery (schema exposure gated on auth or setting)
        if method == "GET" and (
            path in {"/.well-known/mcp", "/.well-known/mcp.json"}
            or path.endswith("/.well-known/mcp")
            or path.endswith("/.well-known/mcp.json")
        ):
            env_pub = os.environ.get("SCOUT_MCP_PUBLIC_TOOL_SCHEMAS", "").strip().lower()
            include_schemas = bool(active_settings.mcp_public_tool_schemas) or env_pub in {
                "1",
                "true",
                "yes",
            }
            if not include_schemas and (
                "authorization" in folded_headers or "x-goog-iap-jwt-assertion" in folded_headers
            ):
                from context_scout.api.dependencies import authenticate_headers

                try:
                    authenticate_headers(folded_headers)
                    include_schemas = True
                except HTTPException:
                    include_schemas = False
            metadata = build_mcp_well_known_metadata(
                self.server,
                endpoint_path=self.streamable_http_path,
                include_tool_schemas=include_schemas,
                resource_metadata_url="/.well-known/oauth-protected-resource",
            )
            resp = JSONResponse(metadata, headers={"Cache-Control": "no-store"})
            await resp(scope, receive, send)
            return

        # SEP-2575 strict stateless check: prohibit sticky session state
        if "mcp-session-id" in folded_headers:
            err_resp = JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32600,
                        "message": (
                            "Stateless MCP 2026-07-28 (SEP-2575) prohibits sticky "
                            "Mcp-Session-Id headers."
                        ),
                    },
                },
                status_code=400,
            )
            await err_resp(scope, receive, send)
            return

        # Out-of-band auth (Cloud IAP / OIDC JWKS / HMAC Bearer) + RFC 8707 aud/resource check
        principal: PrincipalContext | None = None
        if self.require_auth:
            from context_scout.api.dependencies import authenticate_headers

            try:
                principal = authenticate_headers(folded_headers)
            except HTTPException as exc:
                hdrs = {"WWW-Authenticate": www_auth_value} if exc.status_code == 401 else {}
                err_resp = JSONResponse(
                    {"error": str(exc.detail)},
                    status_code=exc.status_code,
                    headers=hdrs,
                )
                await err_resp(scope, receive, send)
                return

            expected_res = (
                os.environ.get("SCOUT_MCP_EXPECTED_RESOURCE")
                or active_settings.mcp_expected_resource
                or ""
            ).strip()
            if expected_res:
                claims = _extract_bearer_token_claims(folded_headers)
                if not _token_matches_expected_resource(claims, expected_res):
                    err_resp = JSONResponse(
                        {
                            "error": (
                                "Bearer token aud/resource claim does not match "
                                f"expected MCP resource '{expected_res}'."
                            )
                        },
                        status_code=401,
                        headers={"WWW-Authenticate": www_auth_value},
                    )
                    await err_resp(scope, receive, send)
                    return

        if method == "GET":
            err_resp = JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32600,
                        "message": (
                            "Method Not Allowed: Stateless MCP 2026-07-28 (SEP-2575) does not "
                            "support standalone GET SSE streams; use POST."
                        ),
                    },
                },
                status_code=405,
                headers={"Allow": "POST"},
            )
            await err_resp(scope, receive, send)
            return

        # Read HTTP body once for SEP-2663 _meta traceparent inspection & replay
        body_chunks: list[bytes] = []
        if method == "POST":
            while True:
                msg = await receive()
                if msg["type"] != "http.request":
                    break
                body_chunks.append(msg.get("body", b""))
                if not msg.get("more_body", False):
                    break
        body_bytes = b"".join(body_chunks)

        decoded_body: dict[str, Any] | None = None
        if body_bytes:
            try:
                parsed = json.loads(body_bytes.decode("utf-8"))
                if isinstance(parsed, dict):
                    decoded_body = parsed
            except (UnicodeDecodeError, ValueError, RecursionError):
                decoded_body = None

        inbound_tp = resolve_traceparent_from_headers(folded_headers, decoded_body)
        parent_ctx = extract_trace_context({"traceparent": inbound_tp}) if inbound_tp else None

        rpc_method = str(decoded_body.get("method") or "") if isinstance(decoded_body, dict) else ""
        rpc_params = decoded_body.get("params") if isinstance(decoded_body, dict) else None
        tool_name = (
            str(rpc_params.get("name") or "")
            if isinstance(rpc_params, dict) and rpc_params.get("name")
            else ""
        )
        req_proto = folded_headers.get("mcp-protocol-version") or (
            str(rpc_params.get("_meta", {}).get("io.modelcontextprotocol/protocolVersion") or "")
            if isinstance(rpc_params, dict) and isinstance(rpc_params.get("_meta"), dict)
            else ""
        )

        tracer = ScoutTelemetryTracer.get_instance()
        span_name = f"mcp.http.{rpc_method or method.lower()}"
        with tracer.tracer.start_as_current_span(span_name, context=parent_ctx) as span:
            span.set_attribute("mcp.stateless", True)
            if rpc_method:
                span.set_attribute("mcp.method.name", rpc_method)
            if req_proto:
                span.set_attribute("mcp.protocol.version", req_proto)
            if tool_name:
                span.set_attribute("gen_ai.tool.name", tool_name)
            if principal is not None:
                safe_pid = pseudonymize_actor_if_enabled(principal.user_id, principal.tenant_id)
                span.set_attribute(ATTR_SCOUT_PRINCIPAL_ID, safe_pid)

            carrier = inject_trace_context({})
            active_tp = carrier.get("traceparent") or inbound_tp or ""

            # Propagate traceparent into body params._meta for mcp.server._otel middleware
            forwarded_body = body_bytes
            if (
                active_tp
                and isinstance(decoded_body, dict)
                and isinstance(decoded_body.get("params"), dict)
                and isinstance(decoded_body["params"].get("_meta"), dict)
                and not decoded_body["params"]["_meta"].get("traceparent")
            ):
                decoded_body["params"]["_meta"]["traceparent"] = active_tp
                forwarded_body = json.dumps(decoded_body).encode("utf-8")

            header_updates: dict[str, str] = {}
            if active_tp:
                header_updates["traceparent"] = active_tp
            if forwarded_body is not body_bytes:
                header_updates["content-length"] = str(len(forwarded_body))

            modified_scope = dict(scope)
            if header_updates:
                modified_scope["headers"] = _rewrite_headers(raw_headers, header_updates)

            body_sent = False

            async def replay_receive() -> Message:
                nonlocal body_sent
                if not body_sent:
                    body_sent = True
                    return {"type": "http.request", "body": forwarded_body, "more_body": False}
                return await receive()

            active_receive = replay_receive if method == "POST" else receive

            async def wrapped_send(message: Message) -> None:
                if message["type"] == "http.response.start":
                    resp_headers = list(message.get("headers", ()))
                    extra: dict[str, str] = {}
                    if active_tp:
                        extra["traceparent"] = active_tp
                    if req_proto == MCP_PROTOCOL_VERSION_2026_07_28:
                        extra["mcp-protocol-version"] = MCP_PROTOCOL_VERSION_2026_07_28
                    if extra:
                        message = {
                            **message,
                            "headers": _rewrite_headers(resp_headers, extra),
                        }
                await send(message)

            async def _dispatch() -> None:
                effective_security = self.security_settings
                if req_host and req_host not in effective_security.allowed_hosts:
                    effective_security = TransportSecuritySettings(
                        enable_dns_rebinding_protection=effective_security.enable_dns_rebinding_protection,
                        allowed_hosts=[*effective_security.allowed_hosts, req_host],
                        allowed_origins=list(effective_security.allowed_origins),
                    )
                session_mgr = getattr(self.server._lowlevel_server, "_session_manager", None)
                if (
                    session_mgr is not None
                    and getattr(session_mgr, "_task_group", None) is not None
                    and effective_security is self.security_settings
                ):
                    await self.starlette_app(modified_scope, active_receive, wrapped_send)
                    return
                # Stateless fallback when ASGI runs without lifespan startup or wildcard host
                per_req_mgr = StreamableHTTPSessionManager(
                    app=self.server._lowlevel_server,
                    json_response=self.json_response,
                    stateless=True,
                    security_settings=effective_security,
                )
                async with per_req_mgr.run():
                    await per_req_mgr.handle_request(modified_scope, active_receive, wrapped_send)

            if principal is not None:
                with bind_session_principal(principal):
                    await _dispatch()
            else:
                await _dispatch()


def create_stateless_mcp_http_app(
    server: MCPServer,
    *,
    streamable_http_path: str = "/mcp",
    json_response: bool = True,
    allowed_hosts: Sequence[str] | None = None,
    allowed_origins: Sequence[str] | None = None,
    require_auth: bool = True,
    settings: ScoutSettings | None = None,
) -> ASGIApp:
    """Wraps an MCPServer in stateless Streamable HTTP 2026-07-28 ASGI transport."""
    return _StatelessMCPHTTPASGIApp(
        server,
        streamable_http_path=streamable_http_path,
        json_response=json_response,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
        require_auth=require_auth,
        settings=settings,
    )


__all__ = [
    "MAX_PAYLOAD_BYTES",
    "MAX_PAYLOAD_ITEMS",
    "MCP_PROTOCOL_VERSION_2026_07_28",
    "SUPPORTED_MCP_PROTOCOL_VERSIONS",
    "TRUNCATION_INDICATOR",
    "PostgresScoutGraphAdapter",
    "bound_payload_dict",
    "build_mcp_well_known_metadata",
    "build_oauth_protected_resource_metadata",
    "create_stateless_mcp_http_app",
    "get_current_principal",
    "parse_cloud_trace_to_traceparent",
    "resolve_traceparent_from_headers",
    "scoped_db_session",
    "validate_tool_call_arguments",
]
