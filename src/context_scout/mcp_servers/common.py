"""Shared MCP Utilities: Out-of-band context, payload caps, and Postgres graph adapter."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from datetime import date
from typing import Any, Final
from uuid import UUID

from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTNeighborhoodRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
    _parse_market_scope,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.schemas.contracts import MarketRegime, PrincipalContext
from context_scout.security.rbac import (
    get_current_principal,
    scoped_db_session,
    validate_tool_call_arguments,
)

logger = logging.getLogger(__name__)

MAX_PAYLOAD_ITEMS: Final[int] = 10
MAX_PAYLOAD_BYTES: Final[int] = 32768  # 32 KB hard serialized JSON cap
TRUNCATION_INDICATOR: Final[str] = "[PAYLOAD_TRUNCATED_AT_32KB]"


def _as_uuid(val: object) -> UUID:
    return val if isinstance(val, UUID) else UUID(str(val))


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
    )
    for key in collection_keys:
        if key in data and isinstance(data[key], list) and len(data[key]) > max_items:
            data[key] = data[key][:max_items]

    # 2. Check byte size
    raw_json = json.dumps(data, default=str)
    raw_bytes = len(raw_json.encode("utf-8"))
    if raw_bytes <= max_bytes:
        return data

    # 3. Truncate long text fields within collections
    text_fields = (
        "verbatim_content",
        "raw_content",
        "snippet",
        "excerpt",
        "contextual_summary",
        "signature_skeleton",
        "content",
    )
    for key in collection_keys:
        if key in data and isinstance(data[key], list):
            items_list = data[key]
            for item in items_list:
                if isinstance(item, dict):
                    for tf in text_fields:
                        if tf in item and isinstance(item[tf], str) and len(item[tf]) > 300:
                            item[tf] = item[tf][:300] + "... [TRUNCATED]"

    data["payload_truncated"] = True
    data["truncation_notice"] = TRUNCATION_INDICATOR

    raw_json = json.dumps(data, default=str)
    raw_bytes = len(raw_json.encode("utf-8"))
    if raw_bytes <= max_bytes:
        return data

    # 4. Progressively shed items from list until payload satisfies limit
    for key in collection_keys:
        if key in data and isinstance(data[key], list) and len(data[key]) > 1:
            while len(data[key]) > 1:
                data[key].pop()
                if len(json.dumps(data, default=str).encode("utf-8")) <= max_bytes:
                    return data

    # 5. Fallback: if single item exceeds limit, aggressively truncate its text fields
    for key in collection_keys:
        if key in data and isinstance(data[key], list) and data[key]:
            single_item = data[key][0]
            if isinstance(single_item, dict):
                for tf in text_fields:
                    if tf in single_item and isinstance(single_item[tf], str):
                        single_item[tf] = single_item[tf][:150] + "... " + TRUNCATION_INDICATOR

    # Ensure final JSON is strictly within max_bytes
    final_json = json.dumps(data, default=str)
    if len(final_json.encode("utf-8")) > max_bytes:
        # Ultimate guarantee: minimal truncated payload
        return {
            "payload_truncated": True,
            "truncation_notice": TRUNCATION_INDICATOR,
            "error": "Payload exceeded 32 KB threshold and was truncated to minimal representation",
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
        self, principal: PrincipalContext, domain_id: str
    ) -> list[CodeFileRecord]:
        """Returns all code files visible to the principal under Forced RLS."""
        with scoped_db_session(self.connection_manager, principal) as conn:
            rows = conn.execute(
                """
                SELECT file_id, domain_id, repo_name, file_path, git_commit_sha,
                       git_blob_sha, language, allowed_scopes, raw_content, line_count
                FROM code_files
                WHERE domain_id = %s
                ORDER BY file_path;
                """,
                (domain_id,),
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
        market_scope: MarketRegime | str | None = None,
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

    def search_visible_doc_chunks(
        self,
        principal: PrincipalContext,
        domain_id: str,
        query_text: str,
        market_scope: MarketRegime | str | None = None,
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
                  AND chunk_id = ANY(%s::uuid[]);
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


__all__ = [
    "MAX_PAYLOAD_BYTES",
    "MAX_PAYLOAD_ITEMS",
    "TRUNCATION_INDICATOR",
    "PostgresScoutGraphAdapter",
    "bound_payload_dict",
    "get_current_principal",
    "scoped_db_session",
    "validate_tool_call_arguments",
]
