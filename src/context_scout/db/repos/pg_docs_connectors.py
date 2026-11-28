"""PostgreSQL RLS repository mixin: Confluence, Jira, doc_chunks hybrid RRF, and watermarks."""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any, cast
from uuid import UUID

from psycopg import Connection

from context_scout.db.repos.types_and_helpers import (
    CalibratedRankedSearchHit,
    _apply_authority_score,
    _as_dt,
    _as_opt_dt,
    _compute_calibrated_relevance,
    _extract_substantive_tokens,
    _format_db_vector_and_halfvec,
    _infer_market_scopes_from_query,
    _market_scope_aliases,
    _parse_market_scope,
    _serialize_market_scope,
)
from context_scout.db.schema import (
    ConfluencePageRecord,
    ConnectorWatermarkRecord,
    DocChunkRecord,
    JiraIssueRecord,
    RankedSearchHit,
)
from context_scout.schemas.contracts import (
    DocumentScope,
    compute_normalized_rrf,
)

logger = logging.getLogger(__name__)


class _PgDocsAndConnectorsMixin:
    def insert_confluence_page(
        self, conn: Connection[tuple[object, ...]], record: ConfluencePageRecord
    ) -> None:
        conn.execute(
            """
            INSERT INTO confluence_pages (
                page_id, tenant_id, domain_id, confluence_page_id, space_key,
                title, version_number, author_username, source_url,
                allowed_scopes, last_modified_at, indexed_at,
                status, is_deleted
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                COALESCE(%s, NOW()), 'current', FALSE
            )
            ON CONFLICT (tenant_id, domain_id, confluence_page_id) DO UPDATE SET
                space_key = EXCLUDED.space_key,
                title = EXCLUDED.title,
                version_number = EXCLUDED.version_number,
                author_username = EXCLUDED.author_username,
                source_url = EXCLUDED.source_url,
                allowed_scopes = EXCLUDED.allowed_scopes,
                last_modified_at = EXCLUDED.last_modified_at,
                indexed_at = NOW(),
                status = 'current',
                is_deleted = FALSE;
            """,
            (
                record.page_id,
                record.tenant_id,
                record.domain_id,
                record.confluence_page_id,
                record.space_key,
                record.title,
                record.version_number,
                record.author_username,
                record.source_url,
                record.allowed_scopes,
                record.last_modified_at,
                record.indexed_at,
            ),
        )

    def get_confluence_page(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        confluence_page_id: str,
        include_deleted: bool = False,
    ) -> ConfluencePageRecord | None:
        query = """
            SELECT page_id, domain_id, confluence_page_id, space_key,
                   title, version_number, author_username, source_url,
                   allowed_scopes, last_modified_at, indexed_at
            FROM confluence_pages
            WHERE domain_id = %s AND confluence_page_id = %s
        """
        if not include_deleted:
            query += " AND NOT is_deleted"
        query += ";"
        row = conn.execute(query, (domain_id, confluence_page_id)).fetchone()
        if not row:
            return None
        return ConfluencePageRecord(
            page_id=UUID(str(row[0])),
            domain_id=str(row[1]),
            confluence_page_id=str(row[2]),
            space_key=str(row[3]),
            title=str(row[4]),
            version_number=int(str(row[5])),
            author_username=str(row[6]),
            source_url=str(row[7]),
            allowed_scopes=[str(s) for s in row[8]] if isinstance(row[8], list) else [],
            last_modified_at=_as_dt(row[9]),
            indexed_at=_as_opt_dt(row[10]),
        )

    def list_confluence_pages(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        space_key: str | None = None,
        include_deleted: bool = False,
    ) -> list[ConfluencePageRecord]:
        query = """
            SELECT page_id, domain_id, confluence_page_id, space_key,
                   title, version_number, author_username, source_url,
                   allowed_scopes, last_modified_at, indexed_at
            FROM confluence_pages
            WHERE domain_id = %s
        """
        params: list[object] = [domain_id]
        if space_key:
            query += " AND space_key = %s"
            params.append(space_key)
        if not include_deleted:
            query += " AND NOT is_deleted"
        query += " ORDER BY title;"
        rows = conn.execute(query, tuple(params)).fetchall()
        return [
            ConfluencePageRecord(
                page_id=UUID(str(row[0])),
                domain_id=str(row[1]),
                confluence_page_id=str(row[2]),
                space_key=str(row[3]),
                title=str(row[4]),
                version_number=int(str(row[5])),
                author_username=str(row[6]),
                source_url=str(row[7]),
                allowed_scopes=[str(s) for s in row[8]] if isinstance(row[8], list) else [],
                last_modified_at=_as_dt(row[9]),
                indexed_at=_as_opt_dt(row[10]),
            )
            for row in rows
        ]

    def tombstone_confluence_pages(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        confluence_page_ids: list[str],
        reason: str = "Page deleted or archived in Confluence",
        status: str = "deleted",
    ) -> tuple[int, int]:
        """Soft-deletes Confluence pages and marks their active doc_chunks stale."""
        clean_ids = [pid for pid in confluence_page_ids if pid]
        if not clean_ids:
            return (0, 0)
        doc_ids = [f"confluence:{pid}" for pid in clean_ids]
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE confluence_pages
                SET is_deleted = TRUE,
                    status = %s
                WHERE domain_id = %s
                  AND confluence_page_id = ANY(%s::text[])
                  AND (NOT is_deleted OR status <> %s);
                """,
                (status, domain_id, clean_ids, status),
            )
            pages_tombstoned = max(cur.rowcount or 0, 0)
            cur.execute(
                """
                UPDATE doc_chunks
                SET is_stale = TRUE,
                    stale_reason = %s
                WHERE domain_id = %s
                  AND doc_id = ANY(%s::text[])
                  AND NOT is_stale;
                """,
                (reason, domain_id, doc_ids),
            )
            chunks_flagged = max(cur.rowcount or 0, 0)
        return (pages_tombstoned, chunks_flagged)

    def update_confluence_allowed_scopes(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        confluence_page_id: str,
        allowed_scopes: list[str],
    ) -> int:
        """Synchronizes changed allowed_scopes across confluence_pages and active doc_chunks."""
        doc_id = f"confluence:{confluence_page_id}"
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE confluence_pages
                SET allowed_scopes = %s
                WHERE domain_id = %s AND confluence_page_id = %s;
                """,
                (allowed_scopes, domain_id, confluence_page_id),
            )
            cur.execute(
                """
                UPDATE doc_chunks
                SET allowed_scopes = %s
                WHERE domain_id = %s AND doc_id = %s AND NOT is_stale;
                """,
                (allowed_scopes, domain_id, doc_id),
            )
            return max(cur.rowcount or 0, 0)

    def insert_jira_issue(
        self, conn: Connection[tuple[object, ...]], record: JiraIssueRecord
    ) -> None:
        conn.execute(
            """
            INSERT INTO jira_issues (
                issue_id, tenant_id, domain_id, issue_key, project_key,
                summary, issue_type, status, resolution,
                assignee_username, source_url, allowed_scopes,
                updated_at, indexed_at,
                epic_key, parent_issue_key, linked_issues_json, linked_mrs_json
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, NOW()),
                %s, %s, %s::jsonb, %s::jsonb
            )
            ON CONFLICT (tenant_id, domain_id, issue_key) DO UPDATE SET
                project_key = EXCLUDED.project_key,
                summary = EXCLUDED.summary,
                issue_type = EXCLUDED.issue_type,
                status = EXCLUDED.status,
                resolution = EXCLUDED.resolution,
                assignee_username = EXCLUDED.assignee_username,
                source_url = EXCLUDED.source_url,
                allowed_scopes = EXCLUDED.allowed_scopes,
                updated_at = EXCLUDED.updated_at,
                indexed_at = NOW(),
                epic_key = EXCLUDED.epic_key,
                parent_issue_key = EXCLUDED.parent_issue_key,
                linked_issues_json = EXCLUDED.linked_issues_json,
                linked_mrs_json = EXCLUDED.linked_mrs_json;
            """,
            (
                record.issue_id,
                record.tenant_id,
                record.domain_id,
                record.issue_key,
                record.project_key,
                record.summary,
                record.issue_type,
                record.status,
                record.resolution,
                record.assignee_username,
                record.source_url,
                record.allowed_scopes,
                record.updated_at,
                record.indexed_at,
                record.epic_key,
                record.parent_issue_key,
                json.dumps(record.linked_issues),
                json.dumps(record.linked_mrs),
            ),
        )

    @staticmethod
    def _row_to_jira_issue(row: tuple[object, ...]) -> JiraIssueRecord:
        linked_issues_raw = (
            row[13] if isinstance(row[13], list) else json.loads(str(row[13] or "[]"))
        )
        linked_mrs_raw = row[14] if isinstance(row[14], list) else json.loads(str(row[14] or "[]"))
        return JiraIssueRecord(
            issue_id=UUID(str(row[0])),
            domain_id=str(row[1]),
            issue_key=str(row[2]),
            project_key=str(row[3]),
            summary=str(row[4]),
            issue_type=str(row[5]),
            status=str(row[6]),
            resolution=str(row[7]) if row[7] is not None else None,
            assignee_username=str(row[8]) if row[8] is not None else None,
            source_url=str(row[9]),
            allowed_scopes=[str(s) for s in row[10]] if isinstance(row[10], list) else [],
            updated_at=_as_dt(row[11]),
            indexed_at=_as_opt_dt(row[12]),
            epic_key=str(row[15]) if row[15] is not None else None,
            parent_issue_key=str(row[16]) if row[16] is not None else None,
            linked_issues=[dict(item) for item in linked_issues_raw if isinstance(item, dict)],
            linked_mrs=[str(item) for item in linked_mrs_raw],
            tenant_id=UUID(str(row[17])),
        )

    def get_jira_issue(
        self, conn: Connection[tuple[object, ...]], domain_id: str, issue_key: str
    ) -> JiraIssueRecord | None:
        row = conn.execute(
            """
            SELECT issue_id, domain_id, issue_key, project_key,
                   summary, issue_type, status, resolution,
                   assignee_username, source_url, allowed_scopes,
                   updated_at, indexed_at,
                   linked_issues_json, linked_mrs_json,
                   epic_key, parent_issue_key, tenant_id
            FROM jira_issues
            WHERE domain_id = %s AND issue_key = %s;
            """,
            (domain_id, issue_key),
        ).fetchone()
        if not row:
            return None
        return self._row_to_jira_issue(row)

    def list_jira_issues(
        self, conn: Connection[tuple[object, ...]], domain_id: str, project_key: str | None = None
    ) -> list[JiraIssueRecord]:
        if project_key:
            rows = conn.execute(
                """
                SELECT issue_id, domain_id, issue_key, project_key,
                       summary, issue_type, status, resolution,
                       assignee_username, source_url, allowed_scopes,
                       updated_at, indexed_at,
                       linked_issues_json, linked_mrs_json,
                       epic_key, parent_issue_key, tenant_id
                FROM jira_issues
                WHERE domain_id = %s AND project_key = %s
                ORDER BY issue_key;
                """,
                (domain_id, project_key),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT issue_id, domain_id, issue_key, project_key,
                       summary, issue_type, status, resolution,
                       assignee_username, source_url, allowed_scopes,
                       updated_at, indexed_at,
                       linked_issues_json, linked_mrs_json,
                       epic_key, parent_issue_key, tenant_id
                FROM jira_issues
                WHERE domain_id = %s
                ORDER BY issue_key;
                """,
                (domain_id,),
            ).fetchall()
        return [self._row_to_jira_issue(row) for row in rows]

    def insert_doc_chunk(
        self, conn: Connection[tuple[object, ...]], record: DocChunkRecord
    ) -> None:
        market_val = _serialize_market_scope(record.market_scope)
        doc_title = record.doc_title.replace("\x00", "")
        section_heading = record.section_heading.replace("\x00", "")
        contextual_header = record.contextual_header.replace("\x00", "")
        verbatim_content = record.verbatim_content.replace("\x00", "")
        stale_reason = (
            record.stale_reason.replace("\x00", "") if record.stale_reason is not None else None
        )
        vec_literal, halfvec_literal = _format_db_vector_and_halfvec(record.embedding)
        conn.execute(
            """
            INSERT INTO doc_chunks (
                chunk_id, tenant_id, domain_id, doc_id, doc_title, source_uri,
                market_scope, published_date, authority_weight,
                section_heading, contextual_header, verbatim_content,
                metadata_json, allowed_scopes, embedding,
                embedding_space_id, embedding_halfvec,
                source_type, version_str, is_stale, stale_reason
            )
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s::jsonb, %s, %s::vector,
                %s, %s::halfvec,
                %s, %s, %s, %s
            )
            """,
            (
                record.chunk_id,
                record.tenant_id,
                record.domain_id,
                record.doc_id,
                doc_title,
                record.source_uri,
                market_val,
                record.published_date,
                record.authority_weight,
                section_heading,
                contextual_header,
                verbatim_content,
                json.dumps(record.metadata_json),
                record.allowed_scopes,
                vec_literal,
                record.embedding_space_id,
                halfvec_literal,
                record.source_type,
                record.version_str,
                record.is_stale,
                stale_reason,
            ),
        )

    def insert_doc_chunks_batch(
        self, conn: Connection[tuple[object, ...]], records: list[DocChunkRecord]
    ) -> int:
        if not records:
            return 0
        sql = """
            INSERT INTO doc_chunks (
                chunk_id, tenant_id, domain_id, doc_id, doc_title, source_uri,
                market_scope, published_date, authority_weight,
                section_heading, contextual_header, verbatim_content,
                metadata_json, allowed_scopes, embedding,
                embedding_space_id, embedding_halfvec,
                source_type, version_str, is_stale, stale_reason
            )
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s::jsonb, %s, %s::vector,
                %s, %s::halfvec,
                %s, %s, %s, %s
            )
        """
        params_seq = []
        for record in records:
            vec_lit, hv_lit = _format_db_vector_and_halfvec(record.embedding)
            params_seq.append(
                (
                    record.chunk_id,
                    record.tenant_id,
                    record.domain_id,
                    record.doc_id,
                    record.doc_title.replace("\x00", ""),
                    record.source_uri,
                    _serialize_market_scope(record.market_scope),
                    record.published_date,
                    record.authority_weight,
                    record.section_heading.replace("\x00", ""),
                    record.contextual_header.replace("\x00", ""),
                    record.verbatim_content.replace("\x00", ""),
                    json.dumps(record.metadata_json),
                    record.allowed_scopes,
                    vec_lit,
                    record.embedding_space_id,
                    hv_lit,
                    record.source_type,
                    record.version_str,
                    record.is_stale,
                    record.stale_reason.replace("\x00", "")
                    if record.stale_reason is not None
                    else None,
                )
            )
        with conn.cursor() as cur:
            cur.executemany(sql, params_seq)
        return len(records)

    def list_visible_doc_chunks(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        market_scope: DocumentScope | str | None = None,
        include_stale: bool = False,
    ) -> list[DocChunkRecord]:
        query = """
            SELECT chunk_id, domain_id, doc_id, doc_title, source_uri,
                   market_scope, published_date, authority_weight,
                   section_heading, contextual_header, verbatim_content,
                   metadata_json, allowed_scopes,
                   COALESCE(embedding_halfvec::text, embedding::text) AS active_vec,
                   source_type, version_str, is_stale, stale_reason,
                   COALESCE(embedding_space_id, 'embeddinggemma-2-768') AS space_id
            FROM doc_chunks
            WHERE domain_id = %s
        """
        params: list[object] = [domain_id]
        if not include_stale:
            query += " AND NOT is_stale"
        serialized_market = _serialize_market_scope(market_scope)
        if serialized_market is not None:
            query += " AND market_scope = %s"
            params.append(serialized_market)
        query += " ORDER BY doc_id, chunk_id;"

        rows = conn.execute(query, tuple(params)).fetchall()
        results: list[DocChunkRecord] = []
        for r in rows:
            raw_emb = r[13]
            if isinstance(raw_emb, str):
                emb_floats = [float(x.strip()) for x in raw_emb.strip("[]").split(",") if x.strip()]
            elif hasattr(raw_emb, "tolist"):
                emb_floats = [float(x) for x in raw_emb.tolist()]
            elif isinstance(raw_emb, (list, tuple)):
                emb_floats = [float(x) for x in raw_emb]
            else:
                emb_floats = []

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
            scopes_list = (
                [str(s) for s in raw_scopes] if isinstance(raw_scopes, (list, tuple)) else []
            )
            results.append(
                DocChunkRecord(
                    chunk_id=UUID(str(r[0])),
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
                    embedding=emb_floats,
                    source_type=str(r[14]) if r[14] is not None else "enterprise_doc",
                    version_str=str(r[15]) if r[15] is not None else "1",
                    is_stale=bool(r[16]),
                    stale_reason=str(r[17]) if r[17] is not None else None,
                    embedding_space_id=str(r[18]) if r[18] is not None else "embeddinggemma-2-768",
                )
            )
        return results

    def search_doc_chunks_hybrid_rrf(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        query_text: str,
        query_embedding: list[float],
        market_scope: DocumentScope | str | None = None,
        top_k: int = 5,
        include_stale: bool = False,
        min_calibrated_score: float = 0.0,
    ) -> list[RankedSearchHit]:
        q_dim = len(query_embedding)
        if q_dim < 128 or q_dim > 4096:
            raise ValueError(f"Query embedding dimension must be in [128, 4096], got {q_dim}")
        vec_literal = "[" + ",".join(f"{v:.6f}" for v in query_embedding) + "]"
        has_null_hv = False
        if q_dim == 768:
            null_row = conn.execute(
                "SELECT 1 FROM doc_chunks "
                "WHERE domain_id = %s AND embedding_halfvec IS NULL LIMIT 1;",
                (domain_id,),
            ).fetchone()
            has_null_hv = null_row is not None
        use_halfvec = (q_dim != 768) or (not has_null_hv)
        if use_halfvec:
            vec_dist_expr = f"(embedding_halfvec::halfvec({q_dim}) <=> %s::halfvec({q_dim}))"
            outer_dist_expr = f"(d.embedding_halfvec::halfvec({q_dim}) <=> %s::halfvec({q_dim}))"
            vec_dim_clause = f"AND vector_dims(embedding_halfvec) = {q_dim}"
        else:
            vec_dist_expr = "(embedding <=> %s::vector)"
            outer_dist_expr = "(d.embedding <=> %s::vector)"
            vec_dim_clause = ""
        market_str = _serialize_market_scope(market_scope)
        candidate_pool = max(100, top_k * 10)
        penalty_rank = candidate_pool + 1
        # Never allow parenthesized glossary expansion suffixes or explicit disjunctive
        # OR-recall queries to poison conjunctive plainto_tsquery (AND) matching.
        stripped_q = query_text.strip()
        if stripped_q.endswith(")"):
            open_idx = stripped_q.rfind("(")
            if open_idx != -1 and 3 <= (len(stripped_q) - open_idx - 2) <= 200:
                raw_base_query = stripped_q[:open_idx].strip()
            else:
                raw_base_query = stripped_q
        else:
            raw_base_query = stripped_q
        is_disjunctive_query = " OR " in query_text
        base_tokens = _extract_substantive_tokens(raw_base_query or query_text)
        content_tokens = _extract_substantive_tokens(query_text)
        and_query = (
            (base_tokens[0] if is_disjunctive_query else " ".join(base_tokens))
            if base_tokens
            else raw_base_query or query_text.strip()
        )
        or_query = " OR ".join(content_tokens) if content_tokens else query_text.strip()
        trgm_query = (
            " ".join(base_tokens) if base_tokens else raw_base_query
        ) or query_text.strip()
        if market_str is None:
            inferred_markets, inferred_alias_str, market_tokens_set = (
                _infer_market_scopes_from_query(query_text, domain_id=domain_id)
            )
        else:
            inferred_markets, inferred_alias_str, market_tokens_set = [], "", set()
        non_market_tokens = [t for t in content_tokens if t not in market_tokens_set]
        non_market_or = " OR ".join(non_market_tokens) if non_market_tokens else or_query
        market_only_query = bool(inferred_markets and not non_market_tokens)
        title_fts_expr = (
            "setweight(to_tsvector('english', "
            "COALESCE(doc_title, '') || ' ' || COALESCE(section_heading, '')), 'A')"
        )
        meta_scope_expr = (
            "COALESCE(market_scope::text, '') || ' ' || "
            "REPLACE(COALESCE(market_scope::text, ''), '_', ' ') || ' ' || "
            "CASE WHEN sh.markets <> '{}'::text[] AND market_scope::text = ANY(sh.markets) "
            "THEN sh.aliases ELSE '' END"
        )
        trgm_small_expr = (
            "COALESCE(doc_title, '') || ' ' || COALESCE(section_heading, '') || ' ' || "
            f"{meta_scope_expr} || ' ' || LEFT(verbatim_content, 1200)"
        )
        trgm_title_expr = (
            "COALESCE(doc_title, '') || ' ' || COALESCE(section_heading, '') || ' ' || "
            f"{meta_scope_expr}"
        )
        fts_market_clause = (
            """
                          OR (
                              sh.markets <> '{}'::text[]
                              AND market_scope::text = ANY(sh.markets)
                          )
            """
            if market_only_query
            else ""
        )
        query = f"""
            WITH domain_size AS (
                SELECT COUNT(*) AS cnt
                FROM doc_chunks
                WHERE domain_id = %s
            ),
            scope_hint AS (
                SELECT %s::text[] AS markets, %s::text AS aliases, %s::text AS nm_or
            ),
            fts_ranked AS (
                SELECT chunk_id,
                       fts_score,
                       ROW_NUMBER() OVER (
                           ORDER BY fts_score DESC, chunk_id
                       ) AS fts_rank
                FROM (
                    SELECT chunk_id,
                           (
                               ts_rank_cd(fts_doc, plainto_tsquery('english', %s)) * 3.0
                               + ts_rank_cd(fts_doc, websearch_to_tsquery('english', %s))
                               + ts_rank_cd(
                                   {title_fts_expr},
                                   websearch_to_tsquery('english', sh.nm_or)
                               ) * 2.5
                               + CASE
                                   WHEN sh.markets <> '{{}}'::text[]
                                        AND market_scope::text = ANY(sh.markets)
                                        AND to_tsvector(
                                            'english',
                                            COALESCE(doc_title, '')
                                            || ' ' || COALESCE(section_heading, '')
                                        ) @@ websearch_to_tsquery('english', sh.nm_or)
                                   THEN 0.50
                                   WHEN sh.markets <> '{{}}'::text[]
                                        AND market_scope::text = ANY(sh.markets)
                                   THEN 0.25
                                   ELSE 0.0
                                 END
                           ) * authority_weight AS fts_score
                    FROM doc_chunks
                    CROSS JOIN scope_hint sh
                    WHERE domain_id = %s
                      AND (%s::text IS NULL OR market_scope = %s)
                      AND authority_weight > 0.0
                      AND (%s OR NOT is_stale)
                      AND (
                          fts_doc @@ websearch_to_tsquery('english', %s)
                          {fts_market_clause}
                      )
                      AND (
                          (SELECT cnt FROM domain_size) <= 50
                          OR (
                              LENGTH(TRIM(verbatim_content)) >= 45
                              AND LEFT(LTRIM(verbatim_content), 5) <> (CHR(37) || ' ---')
                              AND LEFT(LTRIM(verbatim_content), 11) <> '## See also'
                          )
                      )
                    ORDER BY fts_score DESC, chunk_id
                    LIMIT %s
                ) fts_sub
            ),
            vec_ranked AS (
                SELECT chunk_id,
                       vec_dist,
                       ROW_NUMBER() OVER (
                           ORDER BY vec_dist ASC, chunk_id
                       ) AS vec_rank
                FROM (
                    SELECT chunk_id, {vec_dist_expr} AS vec_dist
                    FROM doc_chunks
                    WHERE domain_id = %s
                      AND (%s::text IS NULL OR market_scope = %s)
                      AND authority_weight > 0.0
                      AND (%s OR NOT is_stale)
                      {vec_dim_clause}
                      AND (
                          (SELECT cnt FROM domain_size) <= 50
                          OR (
                              LENGTH(TRIM(verbatim_content)) >= 45
                              AND LEFT(LTRIM(verbatim_content), 5) <> (CHR(37) || ' ---')
                              AND LEFT(LTRIM(verbatim_content), 11) <> '## See also'
                          )
                      )
                    ORDER BY {vec_dist_expr} ASC
                    LIMIT %s
                ) vec_sub
            ),
            trgm_ranked AS (
                SELECT chunk_id,
                       trgm_score,
                       ROW_NUMBER() OVER (
                           ORDER BY trgm_score DESC, chunk_id
                       ) AS trgm_rank
                FROM (
                    SELECT chunk_id,
                           (
                               word_similarity(
                                   %s,
                                   CASE
                                       WHEN (SELECT cnt FROM domain_size) <= 50
                                       THEN {trgm_small_expr}
                                       ELSE {trgm_title_expr}
                                   END
                               ) * authority_weight
                           ) AS trgm_score
                    FROM doc_chunks
                    CROSS JOIN scope_hint sh
                    WHERE domain_id = %s
                      AND (%s::text IS NULL OR market_scope = %s)
                      AND authority_weight > 0.0
                      AND (%s OR NOT is_stale)
                      AND word_similarity(
                              %s,
                              CASE
                                  WHEN (SELECT cnt FROM domain_size) <= 50
                                  THEN {trgm_small_expr}
                                  ELSE {trgm_title_expr}
                              END
                          ) >= 0.20
                      AND (
                          (SELECT cnt FROM domain_size) <= 50
                          OR (
                              LENGTH(TRIM(verbatim_content)) >= 45
                              AND LEFT(LTRIM(verbatim_content), 5) <> (CHR(37) || ' ---')
                              AND LEFT(LTRIM(verbatim_content), 11) <> '## See also'
                          )
                      )
                    ORDER BY trgm_score DESC, chunk_id
                    LIMIT %s
                ) trgm_sub
            ),
            candidate_ids AS (
                SELECT chunk_id FROM fts_ranked
                UNION
                SELECT chunk_id FROM vec_ranked
                UNION
                SELECT chunk_id FROM trgm_ranked
            )
            SELECT d.chunk_id, d.doc_title, d.section_heading,
                   COALESCE(f.fts_rank, %s) AS fts_rank,
                   COALESCE(v.vec_rank, %s) AS vec_rank,
                   COALESCE(t.trgm_rank, %s) AS trgm_rank,
                   d.authority_weight,
                   COALESCE(f.fts_score, 0.0) AS fts_score,
                   CASE
                       WHEN COALESCE(v.vec_dist, {outer_dist_expr}) = 'NaN'::float8
                       THEN 0.0
                       ELSE GREATEST(
                           0.0, 1.0 - COALESCE(v.vec_dist, {outer_dist_expr})
                       )
                   END AS vec_cos,
                   COALESCE(t.trgm_score, 0.0) AS trgm_score,
                   LEFT(d.verbatim_content, 1800) AS content_preview,
                   (SELECT cnt FROM domain_size) AS domain_cnt,
                   d.market_scope::text AS market_scope_val
            FROM candidate_ids c
            JOIN doc_chunks d ON d.chunk_id = c.chunk_id
            LEFT JOIN fts_ranked f ON d.chunk_id = f.chunk_id
            LEFT JOIN vec_ranked v ON d.chunk_id = v.chunk_id
            LEFT JOIN trgm_ranked t ON d.chunk_id = t.chunk_id
            WHERE d.domain_id = %s
              AND (%s::text IS NULL OR d.market_scope = %s)
              AND d.authority_weight > 0.0
              AND (%s OR NOT d.is_stale);
        """
        params = (
            domain_id,
            inferred_markets,
            inferred_alias_str,
            non_market_or,
            and_query,
            or_query,
            domain_id,
            market_str,
            market_str,
            include_stale,
            or_query,
            candidate_pool,
            vec_literal,
            domain_id,
            market_str,
            market_str,
            include_stale,
            vec_literal,
            candidate_pool,
            trgm_query,
            domain_id,
            market_str,
            market_str,
            include_stale,
            trgm_query,
            candidate_pool,
            penalty_rank,
            penalty_rank,
            penalty_rank,
            vec_literal,
            vec_literal,
            domain_id,
            market_str,
            market_str,
            include_stale,
        )
        rows = conn.execute(query, params).fetchall()
        scored_hits: list[tuple[float, float, float, CalibratedRankedSearchHit]] = []
        for r in rows:
            chunk_id = UUID(str(r[0]))
            doc_title = str(r[1])
            section_heading = str(r[2])
            fts_rank = int(str(r[3]))
            vec_rank = int(str(r[4]))
            trgm_rank = int(str(r[5]))
            auth_weight = float(str(r[6]))
            fts_score = float(str(r[7]))
            vec_cos = float(str(r[8]))
            trgm_score = float(str(r[9]))
            content_preview = str(r[10] or "")
            domain_cnt = int(str(r[11]))
            row_market_scope = str(r[12]) if r[12] is not None else None
            is_scope_mismatch = bool(
                market_str is None
                and inferred_markets
                and row_market_scope
                and row_market_scope not in inferred_markets
                and _market_scope_aliases(row_market_scope, domain_id=domain_id)
            )

            base_rrf = compute_normalized_rrf(
                [trgm_rank, fts_rank, vec_rank], num_modalities=3, k=60
            )
            weighted_score = _apply_authority_score(base_rrf, auth_weight)
            calibrated_score, lexical_cov, is_boiler = _compute_calibrated_relevance(
                query_text=query_text,
                title=doc_title,
                section=section_heading,
                content=content_preview,
                cosine_similarity=vec_cos,
                bm25_score=fts_score,
                trigram_score=trgm_score,
                raw_rrf_score=base_rrf,
                authority_weight=auth_weight,
                market_scope=row_market_scope,
                domain_id=domain_id,
                scope_mismatch=is_scope_mismatch,
            )

            if is_boiler and domain_cnt > 50:
                continue
            if domain_cnt > 50:
                passes_floor = (
                    (vec_cos >= 0.46 and (lexical_cov > 0.0 or not content_tokens))
                    or (
                        lexical_cov >= 0.34
                        and (vec_cos >= 0.18 or fts_score > 0.0 or trgm_score >= 0.18)
                    )
                    or (fts_score >= 0.10 and lexical_cov > 0.0)
                )
                if not passes_floor:
                    continue
            if min_calibrated_score > 0.0 and calibrated_score < min_calibrated_score:
                continue

            hit = CalibratedRankedSearchHit(
                item_id=chunk_id,
                qualified_name_or_title=f"{doc_title} > {section_heading}",
                normalized_rrf_score=weighted_score,
                trigram_rank=trgm_rank,
                bm25_rank=fts_rank,
                vector_rank=vec_rank,
                cosine_similarity=round(vec_cos, 6),
                bm25_score=round(fts_score, 6),
                trigram_score=round(trgm_score, 6),
                raw_rrf_score=round(base_rrf, 6),
                calibrated_score=calibrated_score,
                domain_id=domain_id,
                lexical_coverage=lexical_cov,
                is_boilerplate=is_boiler,
            )
            rank_key = calibrated_score if domain_cnt > 50 else weighted_score
            scored_hits.append((rank_key, weighted_score, auth_weight, hit))
        scored_hits.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [item[3] for item in scored_hits[:top_k]]

    @staticmethod
    def _row_to_watermark(row: tuple[object, ...]) -> ConnectorWatermarkRecord:
        meta_raw = row[10] if isinstance(row[10], dict) else json.loads(str(row[10] or "{}"))
        tier_str = str(row[5] or "watermark")
        tier = cast(
            Any,
            tier_str
            if tier_str in {"webhook", "watermark", "reconcile", "bundle"}
            else "watermark",
        )
        return ConnectorWatermarkRecord(
            tenant_id=UUID(str(row[0])),
            domain_id=str(row[1]),
            source_id=str(row[2]),
            container_key=str(row[3]),
            watermark_cursor=str(row[4] or ""),
            sync_tier=tier,
            items_synced=int(str(row[6] or 0)),
            last_webhook_at=_as_opt_dt(row[7]),
            last_watermark_sync_at=_as_opt_dt(row[8]),
            last_reconciliation_at=_as_opt_dt(row[9]),
            metadata_json=dict(meta_raw),
            updated_at=_as_opt_dt(row[11]),
        )

    def upsert_connector_watermark(
        self,
        conn: Connection[tuple[object, ...]],
        record: ConnectorWatermarkRecord,
    ) -> None:
        conn.execute(
            """
            INSERT INTO connector_sync_watermarks (
                tenant_id, domain_id, source_id, container_key,
                watermark_cursor, sync_tier, items_synced,
                last_webhook_at, last_watermark_sync_at, last_reconciliation_at,
                metadata_json, updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, COALESCE(%s, NOW())
            )
            ON CONFLICT (tenant_id, domain_id, source_id, container_key) DO UPDATE SET
                watermark_cursor = CASE
                    WHEN EXCLUDED.watermark_cursor <> '' THEN EXCLUDED.watermark_cursor
                    ELSE connector_sync_watermarks.watermark_cursor
                END,
                sync_tier = EXCLUDED.sync_tier,
                items_synced = EXCLUDED.items_synced,
                last_webhook_at = COALESCE(
                    EXCLUDED.last_webhook_at,
                    connector_sync_watermarks.last_webhook_at
                ),
                last_watermark_sync_at = COALESCE(
                    EXCLUDED.last_watermark_sync_at,
                    connector_sync_watermarks.last_watermark_sync_at
                ),
                last_reconciliation_at = COALESCE(
                    EXCLUDED.last_reconciliation_at,
                    connector_sync_watermarks.last_reconciliation_at
                ),
                metadata_json = EXCLUDED.metadata_json,
                updated_at = NOW();
            """,
            (
                record.tenant_id,
                record.domain_id,
                record.source_id,
                record.container_key,
                record.watermark_cursor,
                record.sync_tier,
                record.items_synced,
                record.last_webhook_at,
                record.last_watermark_sync_at,
                record.last_reconciliation_at,
                json.dumps(record.metadata_json),
                record.updated_at,
            ),
        )

    def reset_connector_watermark(
        self,
        conn: Connection[tuple[object, ...]],
        record: ConnectorWatermarkRecord,
    ) -> None:
        """Unconditionally rewinds watermark_cursor (including to empty) after PITR."""
        conn.execute(
            """
            INSERT INTO connector_sync_watermarks (
                tenant_id, domain_id, source_id, container_key,
                watermark_cursor, sync_tier, items_synced,
                last_webhook_at, last_watermark_sync_at, last_reconciliation_at,
                metadata_json, updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, NOW()), %s::jsonb, NOW()
            )
            ON CONFLICT (tenant_id, domain_id, source_id, container_key) DO UPDATE SET
                watermark_cursor = EXCLUDED.watermark_cursor,
                sync_tier = EXCLUDED.sync_tier,
                items_synced = EXCLUDED.items_synced,
                last_reconciliation_at = COALESCE(
                    EXCLUDED.last_reconciliation_at,
                    NOW()
                ),
                metadata_json = EXCLUDED.metadata_json,
                updated_at = NOW();
            """,
            (
                record.tenant_id,
                record.domain_id,
                record.source_id,
                record.container_key,
                record.watermark_cursor,
                record.sync_tier,
                record.items_synced,
                record.last_webhook_at,
                record.last_watermark_sync_at,
                record.last_reconciliation_at,
                json.dumps(record.metadata_json),
            ),
        )

    def get_connector_watermark(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        source_id: str,
        container_key: str,
    ) -> ConnectorWatermarkRecord | None:
        row = conn.execute(
            """
            SELECT tenant_id, domain_id, source_id, container_key,
                   watermark_cursor, sync_tier, items_synced,
                   last_webhook_at, last_watermark_sync_at, last_reconciliation_at,
                   metadata_json, updated_at
            FROM connector_sync_watermarks
            WHERE domain_id = %s AND source_id = %s AND container_key = %s;
            """,
            (domain_id, source_id, container_key),
        ).fetchone()
        return self._row_to_watermark(row) if row is not None else None

    def list_connector_watermarks(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        source_id: str | None = None,
    ) -> list[ConnectorWatermarkRecord]:
        if source_id:
            rows = conn.execute(
                """
                SELECT tenant_id, domain_id, source_id, container_key,
                       watermark_cursor, sync_tier, items_synced,
                       last_webhook_at, last_watermark_sync_at, last_reconciliation_at,
                       metadata_json, updated_at
                FROM connector_sync_watermarks
                WHERE domain_id = %s AND source_id = %s
                ORDER BY source_id, container_key;
                """,
                (domain_id, source_id),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT tenant_id, domain_id, source_id, container_key,
                       watermark_cursor, sync_tier, items_synced,
                       last_webhook_at, last_watermark_sync_at, last_reconciliation_at,
                       metadata_json, updated_at
                FROM connector_sync_watermarks
                WHERE domain_id = %s
                ORDER BY source_id, container_key;
                """,
                (domain_id,),
            ).fetchall()
        return [self._row_to_watermark(row) for row in rows]
