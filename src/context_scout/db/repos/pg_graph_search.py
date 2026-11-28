"""PostgreSQL RLS repository mixin: 1-hop/2-hop AST CTE and hybrid RRF symbol search."""

from __future__ import annotations

import logging
import math
import re
from typing import Final
from uuid import UUID

from psycopg import Connection

from context_scout.db.repos.types_and_helpers import (
    CalibratedRankedSearchHit,
    _apply_authority_score,
    _compute_calibrated_relevance,
    _extract_substantive_tokens,
)
from context_scout.db.schema import (
    ASTNeighborhoodRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    RankedSearchHit,
)
from context_scout.schemas.contracts import (
    compute_normalized_rrf,
)

logger = logging.getLogger(__name__)


class _PgGraphSearchMixin:
    TRAVERSE_1HOP_NEIGHBORHOOD_CTE_SQL: Final[str] = """
        WITH
        target_sym AS (
            SELECT s.symbol_id, s.file_id, s.domain_id, s.qualified_name, s.symbol_kind,
                   s.start_line, s.end_line, s.signature_skeleton, s.contextual_summary,
                   f.repo_name, f.file_path, f.git_commit_sha, f.git_blob_sha, f.raw_content
            FROM ast_symbols s
            JOIN code_files f ON f.file_id = s.file_id
            WHERE s.domain_id = %s AND s.symbol_id = %s
        ),
        focus_family AS (
            SELECT symbol_id, qualified_name, file_id
            FROM target_sym
            UNION
            SELECT s.symbol_id, s.qualified_name, s.file_id
            FROM ast_symbols s
            JOIN target_sym ts ON s.domain_id = ts.domain_id AND s.file_id = ts.file_id
            WHERE ts.symbol_kind = 'class'
              AND s.qualified_name LIKE (ts.qualified_name || '.' || CHR(37))
        ),
        hop1_outward AS (
            -- Hop 1: Outward edges (USES_TYPE, IMPORTS, INHERITS, and same-file CALLS)
            SELECT
                e.edge_id,
                e.edge_kind,
                e.target_symbol_id AS related_symbol_id,
                1 AS depth,
                'imported_type' AS intent_role
            FROM ast_edges e
            JOIN focus_family ff ON e.source_symbol_id = ff.symbol_id
            JOIN target_sym ts ON TRUE
            WHERE e.domain_id = %s
              AND e.target_symbol_id IS NOT NULL
              AND e.target_symbol_id NOT IN (
                  SELECT ff2.symbol_id FROM focus_family ff2 WHERE ff2.symbol_id <> ts.symbol_id
              )
              AND (
                  e.edge_kind IN (
                      'USES_TYPE', 'IMPORTS', 'INHERITS',
                      'PROVISIONED_BY', 'DISPATCHES', 'SELECTS', 'DOCUMENTS_COMPONENT',
                      'OWNED_BY_TEAM', 'PART_OF_SYSTEM',
                      'PUBLISHES_EVENT', 'SUBSCRIBES_EVENT', 'READS_TABLE',
                      'WRITES_TABLE', 'RENDERS_TEMPLATE', 'IMPLEMENTS_SPEC',
                      'IMPLEMENTS_ISSUE', 'BINDS_SERVICE_ACCOUNT', 'ACCESSES_SECRET'
                  )
                  OR (e.edge_kind = 'CALLS' AND EXISTS (
                      SELECT 1 FROM ast_symbols s2
                      WHERE s2.symbol_id = e.target_symbol_id AND s2.file_id = ts.file_id
                  ))
              )
        ),
        hop2_outward AS (
            -- Hop 2: Transitive schema/type/inheritance/import closure within the same repository
            SELECT
                e2.edge_id,
                e2.edge_kind,
                e2.target_symbol_id AS related_symbol_id,
                2 AS depth,
                'imported_type' AS intent_role
            FROM hop1_outward h1
            JOIN ast_edges e2 ON e2.source_symbol_id = h1.related_symbol_id
            JOIN target_sym ts ON TRUE
            JOIN ast_symbols s_hop1 ON s_hop1.symbol_id = h1.related_symbol_id
            JOIN code_files f_hop1 ON f_hop1.file_id = s_hop1.file_id
            WHERE h1.edge_kind IN ('USES_TYPE', 'INHERITS', 'IMPORTS')
              AND e2.domain_id = %s
              AND e2.target_symbol_id IS NOT NULL
              AND e2.edge_kind IN ('USES_TYPE', 'INHERITS', 'IMPORTS')
              AND f_hop1.repo_name = ts.repo_name
        ),
        outward_closure AS (
            SELECT * FROM hop1_outward
            UNION
            SELECT * FROM hop2_outward
        ),
        inward_caller_edges AS (
            SELECT e.edge_id, e.edge_kind, e.source_symbol_id
            FROM ast_edges e
            JOIN focus_family ff ON e.target_symbol_id = ff.symbol_id
            WHERE e.domain_id = %s
              AND (
                  e.edge_kind = 'CALLS'
                  OR (
                      e.is_cross_repo = TRUE
                      AND e.edge_kind IN (
                          'USES_TYPE', 'PROVISIONED_BY', 'DOCUMENTS_COMPONENT',
                          'PUBLISHES_EVENT', 'SUBSCRIBES_EVENT', 'READS_TABLE',
                          'WRITES_TABLE', 'RENDERS_TEMPLATE', 'IMPLEMENTS_SPEC',
                          'IMPLEMENTS_ISSUE', 'BINDS_SERVICE_ACCOUNT', 'ACCESSES_SECRET'
                      )
                  )
              )
              AND e.source_symbol_id NOT IN (SELECT symbol_id FROM focus_family)
            UNION
            SELECT e.edge_id, e.edge_kind, e.source_symbol_id
            FROM ast_edges e
            JOIN focus_family ff ON e.target_qualified_name = ff.qualified_name
            WHERE e.domain_id = %s
              AND (
                  e.edge_kind = 'CALLS'
                  OR (
                      e.is_cross_repo = TRUE
                      AND e.edge_kind IN (
                          'USES_TYPE', 'PROVISIONED_BY', 'DOCUMENTS_COMPONENT',
                          'PUBLISHES_EVENT', 'SUBSCRIBES_EVENT', 'READS_TABLE',
                          'WRITES_TABLE', 'RENDERS_TEMPLATE', 'IMPLEMENTS_SPEC',
                          'IMPLEMENTS_ISSUE', 'BINDS_SERVICE_ACCOUNT', 'ACCESSES_SECRET'
                      )
                  )
              )
              AND e.source_symbol_id NOT IN (SELECT symbol_id FROM focus_family)
        ),
        inward_callers AS (
            -- Direct callers ranked with diversity across caller files
            SELECT
                ranked.edge_id,
                ranked.edge_kind,
                ranked.source_symbol_id AS related_symbol_id,
                1 AS depth,
                'direct_caller' AS intent_role
            FROM (
                SELECT
                    e.edge_id,
                    e.edge_kind,
                    e.source_symbol_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY s_caller.file_id
                        ORDER BY s_caller.start_line, e.edge_id
                    ) AS file_call_rank,
                    ROW_NUMBER() OVER (
                        ORDER BY (s_caller.file_id = ts.file_id) ASC,
                                 s_caller.file_id, s_caller.start_line
                    ) AS overall_rank
                FROM inward_caller_edges e
                CROSS JOIN target_sym ts
                JOIN ast_symbols s_caller ON s_caller.symbol_id = e.source_symbol_id
            ) ranked
            WHERE ranked.file_call_rank = 1 AND ranked.overall_rank <= %s
        ),
        test_fixtures AS (
            -- Direct TESTED_BY edge resolution (UNION of index-friendly branches)
            SELECT edge_id, edge_kind, related_symbol_id, 1 AS depth, 'test_fixture' AS intent_role
            FROM (
                SELECT e.edge_id, e.edge_kind, e.target_symbol_id AS related_symbol_id
                FROM ast_edges e
                JOIN focus_family ff ON e.source_symbol_id = ff.symbol_id
                WHERE e.domain_id = %s
                  AND e.edge_kind = 'TESTED_BY'
                  AND e.target_symbol_id IS NOT NULL
                UNION
                SELECT e.edge_id, e.edge_kind, e.source_symbol_id AS related_symbol_id
                FROM ast_edges e
                JOIN focus_family ff ON e.target_symbol_id = ff.symbol_id
                WHERE e.domain_id = %s
                  AND e.edge_kind = 'TESTED_BY'
                UNION
                SELECT e.edge_id, e.edge_kind, e.source_symbol_id AS related_symbol_id
                FROM ast_edges e
                JOIN focus_family ff ON e.target_qualified_name = ff.qualified_name
                WHERE e.domain_id = %s
                  AND e.edge_kind = 'TESTED_BY'
            ) tf
            LIMIT 2
        ),
        combined_neighborhood AS (
            SELECT * FROM outward_closure
            UNION ALL
            SELECT * FROM inward_callers
            UNION ALL
            SELECT * FROM test_fixtures
        )
        SELECT DISTINCT ON (cn.related_symbol_id, cn.intent_role)
            cn.intent_role,
            cn.edge_kind,
            cn.depth,
            s.symbol_id,
            s.domain_id,
            s.qualified_name,
            s.symbol_kind,
            s.start_line,
            s.end_line,
            s.signature_skeleton,
            s.contextual_summary,
            f.file_id,
            f.repo_name,
            f.file_path,
            f.git_commit_sha,
            f.git_blob_sha,
            f.language,
            f.line_count,
            f.raw_content
        FROM combined_neighborhood cn
        JOIN ast_symbols s ON s.symbol_id = cn.related_symbol_id
        JOIN code_files f ON f.file_id = s.file_id
        WHERE s.domain_id = %s
        ORDER BY cn.related_symbol_id, cn.intent_role, cn.depth ASC;
        """

    def traverse_1hop_neighborhood(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        focus_symbol_id: UUID | str,
        max_direct_callers: int = 3,
    ) -> list[ASTNeighborhoodRecord]:
        """Executes memory-bounded recursive CTE 1-hop AST closure directly in PostgreSQL 17.

        Replaces unbounded Python heap iteration with a parameterized CTE running under
        PostgreSQL 17 Forced RLS. Returns only relevant symbols in O(k) memory.
        """
        focus_uuid = UUID(str(focus_symbol_id))
        cte_sql = self.TRAVERSE_1HOP_NEIGHBORHOOD_CTE_SQL
        params = (
            domain_id,
            focus_uuid,
            domain_id,
            domain_id,
            domain_id,
            domain_id,
            max_direct_callers,
            domain_id,
            domain_id,
            domain_id,
            domain_id,
        )
        rows = conn.execute(cte_sql, params).fetchall()
        neighbors: list[ASTNeighborhoodRecord] = []
        for r in rows:
            intent_role = r[0]
            edge_kind = str(r[1])
            depth = int(str(r[2]))
            sym_id = UUID(str(r[3]))
            sym_domain = str(r[4])
            qual_name = str(r[5])
            sym_kind = str(r[6])
            start_l = int(str(r[7]))
            end_l = int(str(r[8]))
            sig = str(r[9])
            ctx_summary = str(r[10])
            file_id = UUID(str(r[11]))
            repo_name = str(r[12])
            file_path = str(r[13])
            commit_sha = str(r[14])
            blob_sha = str(r[15])
            lang = str(r[16])
            line_cnt = int(str(r[17]))
            raw_content = str(r[18])

            file_rec = CodeFileRecord(
                file_id=file_id,
                domain_id=sym_domain,
                repo_name=repo_name,
                file_path=file_path,
                git_commit_sha=commit_sha,
                git_blob_sha=blob_sha,
                language=lang,
                allowed_scopes=[],  # populated by RLS session
                raw_content=raw_content,
                line_count=line_cnt,
            )
            sym_rec = ASTSymbolRecord(
                symbol_id=sym_id,
                file_id=file_id,
                domain_id=sym_domain,
                qualified_name=qual_name,
                symbol_kind=sym_kind,
                start_line=start_l,
                end_line=end_l,
                signature_skeleton=sig,
                contextual_summary=ctx_summary,
                allowed_scopes=[],
                embedding=[],
            )
            neighbors.append(
                ASTNeighborhoodRecord(
                    intent_role=intent_role,  # type: ignore[arg-type]
                    edge_kind=edge_kind,
                    depth=depth,
                    symbol=sym_rec,
                    file=file_rec,
                )
            )
        return neighbors

    def search_symbols_hybrid_rrf(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 5,
        min_calibrated_score: float = 0.0,
    ) -> list[RankedSearchHit]:
        """Executes 3-way Hybrid RRF over PostgreSQL ast_symbols (GIN trigram + FTS + HNSW)."""
        q_dim = len(query_embedding)
        if q_dim < 128 or q_dim > 4096:
            raise ValueError(f"Query embedding dimension must be in [128, 4096], got {q_dim}")
        vec_literal = "[" + ",".join(f"{v:.6f}" for v in query_embedding) + "]"
        has_null_hv = False
        if q_dim == 768:
            null_row = conn.execute(
                "SELECT 1 FROM ast_symbols "
                "WHERE domain_id = %s AND embedding_halfvec IS NULL LIMIT 1;",
                (domain_id,),
            ).fetchone()
            has_null_hv = null_row is not None
        use_halfvec = (q_dim != 768) or (not has_null_hv)
        if use_halfvec:
            vec_dist_expr = f"(embedding_halfvec::halfvec({q_dim}) <=> %s::halfvec({q_dim}))"
            outer_dist_expr = f"(s.embedding_halfvec::halfvec({q_dim}) <=> %s::halfvec({q_dim}))"
            vec_dim_clause = f"AND vector_dims(embedding_halfvec) = {q_dim}"
        else:
            vec_dist_expr = "(embedding <=> %s::vector)"
            outer_dist_expr = "(s.embedding <=> %s::vector)"
            vec_dim_clause = ""
        candidate_pool = max(100, top_k * 10)
        penalty_rank = candidate_pool + 1
        content_tokens = _extract_substantive_tokens(query_text)
        fts_query = " OR ".join(content_tokens) if content_tokens else query_text.strip()
        query = f"""
            WITH domain_size AS (
                SELECT COUNT(*) AS cnt
                FROM ast_symbols
                WHERE domain_id = %s
            ),
            fts_ranked AS (
                SELECT symbol_id,
                       fts_score,
                       ROW_NUMBER() OVER (
                           ORDER BY fts_score DESC, symbol_id
                       ) AS fts_rank
                FROM (
                    SELECT symbol_id,
                           (
                               ts_rank_cd(fts_doc, plainto_tsquery('english', %s)) * 2.0
                               + ts_rank_cd(fts_doc, websearch_to_tsquery('english', %s))
                           ) AS fts_score
                    FROM ast_symbols
                    WHERE domain_id = %s
                      AND fts_doc @@ websearch_to_tsquery('english', %s)
                    ORDER BY fts_score DESC, symbol_id
                    LIMIT %s
                ) fts_sub
            ),
            vec_ranked AS (
                SELECT symbol_id,
                       vec_dist,
                       ROW_NUMBER() OVER (
                           ORDER BY vec_dist ASC, symbol_id
                       ) AS vec_rank
                FROM (
                    SELECT symbol_id, {vec_dist_expr} AS vec_dist
                    FROM ast_symbols
                    WHERE domain_id = %s
                      {vec_dim_clause}
                    ORDER BY {vec_dist_expr} ASC
                    LIMIT %s
                ) vec_sub
            ),
            trgm_ranked AS (
                SELECT symbol_id,
                       trgm_score,
                       ROW_NUMBER() OVER (
                           ORDER BY trgm_score DESC, symbol_id
                       ) AS trgm_rank
                FROM (
                    SELECT symbol_id,
                           GREATEST(
                               similarity(qualified_name, %s),
                               similarity(signature_skeleton, %s)
                           ) AS trgm_score
                    FROM ast_symbols
                    WHERE domain_id = %s
                    ORDER BY trgm_score DESC, symbol_id
                    LIMIT %s
                ) trgm_sub
            ),
            candidate_ids AS (
                SELECT symbol_id FROM fts_ranked
                UNION
                SELECT symbol_id FROM vec_ranked
                UNION
                SELECT symbol_id FROM trgm_ranked
            )
            SELECT s.symbol_id, s.qualified_name,
                   COALESCE(f.fts_rank, %s) AS fts_rank,
                   COALESCE(v.vec_rank, %s) AS vec_rank,
                   COALESCE(t.trgm_rank, %s) AS trgm_rank,
                   COALESCE(f.fts_score, 0.0) AS fts_score,
                   CASE
                       WHEN COALESCE(v.vec_dist, {outer_dist_expr}) = 'NaN'::float8
                       THEN 0.0
                       ELSE GREATEST(
                           0.0, 1.0 - COALESCE(v.vec_dist, {outer_dist_expr})
                       )
                   END AS vec_cos,
                   COALESCE(t.trgm_score, 0.0) AS trgm_score,
                   s.signature_skeleton,
                   LEFT(COALESCE(s.contextual_summary, ''), 600) AS summary_preview,
                   (SELECT cnt FROM domain_size) AS domain_cnt,
                   COALESCE(cf.ranking_weight, 1.0) AS ranking_weight
            FROM candidate_ids c
            JOIN ast_symbols s ON s.symbol_id = c.symbol_id
            LEFT JOIN code_files cf ON cf.file_id = s.file_id
            LEFT JOIN fts_ranked f ON s.symbol_id = f.symbol_id
            LEFT JOIN vec_ranked v ON s.symbol_id = v.symbol_id
            LEFT JOIN trgm_ranked t ON s.symbol_id = t.symbol_id
            WHERE s.domain_id = %s;
        """
        params = (
            domain_id,
            query_text,
            fts_query,
            domain_id,
            fts_query,
            candidate_pool,
            vec_literal,
            domain_id,
            vec_literal,
            candidate_pool,
            query_text,
            query_text,
            domain_id,
            candidate_pool,
            penalty_rank,
            penalty_rank,
            penalty_rank,
            vec_literal,
            vec_literal,
            domain_id,
        )
        rows = conn.execute(query, params).fetchall()
        hits: list[CalibratedRankedSearchHit] = []
        for r in rows:
            sym_id = UUID(str(r[0]))
            qual_name = str(r[1])
            fts_rank = int(str(r[2]))
            vec_rank = int(str(r[3]))
            trgm_rank = int(str(r[4]))
            fts_score = float(str(r[5]))
            vec_cos = float(str(r[6]))
            trgm_score = float(str(r[7]))
            sig_skeleton = str(r[8] or "")
            summary_preview = str(r[9] or "")
            domain_cnt = int(str(r[10]))
            ranking_weight = float(str(r[11]))

            norm_rrf = compute_normalized_rrf(
                [trgm_rank, fts_rank, vec_rank], num_modalities=3, k=60
            )
            calibrated_score, lexical_cov, _ = _compute_calibrated_relevance(
                query_text=query_text,
                title=qual_name,
                section=sig_skeleton,
                content=summary_preview,
                cosine_similarity=vec_cos,
                bm25_score=fts_score,
                trigram_score=trgm_score,
                raw_rrf_score=norm_rrf,
                authority_weight=1.0,
                is_code_symbol=True,
            )
            if not math.isclose(ranking_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
                norm_rrf = _apply_authority_score(norm_rrf, ranking_weight)
                calibrated_score = round(min(1.0, max(0.0, calibrated_score * ranking_weight)), 6)
            if domain_cnt > 50:
                passes_floor = (
                    (vec_cos >= 0.48 and (lexical_cov > 0.0 or trgm_score >= 0.25))
                    or (
                        lexical_cov >= 0.34
                        and (vec_cos >= 0.18 or fts_score > 0.0 or trgm_score >= 0.18)
                    )
                    or (fts_score >= 0.10 and lexical_cov > 0.0)
                    or (trgm_score >= 0.32 and lexical_cov > 0.0)
                )
                if not passes_floor:
                    continue
            if min_calibrated_score > 0.0 and calibrated_score < min_calibrated_score:
                continue

            hits.append(
                CalibratedRankedSearchHit(
                    item_id=sym_id,
                    qualified_name_or_title=qual_name,
                    normalized_rrf_score=norm_rrf,
                    trigram_rank=trgm_rank,
                    bm25_rank=fts_rank,
                    vector_rank=vec_rank,
                    cosine_similarity=round(vec_cos, 6),
                    bm25_score=round(fts_score, 6),
                    trigram_score=round(trgm_score, 6),
                    raw_rrf_score=round(norm_rrf, 6),
                    calibrated_score=calibrated_score,
                    domain_id=domain_id,
                    lexical_coverage=lexical_cov,
                )
            )
        hits.sort(
            key=lambda h: (h.calibrated_score, h.normalized_rrf_score),
            reverse=True,
        )
        return list(hits[:top_k])

    def search_visible_ast_symbols(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        query_text: str,
        limit: int = 50,
    ) -> list[ASTSymbolRecord]:
        """Searches visible AST symbols under RLS with SQL pushdown and LIMIT."""

        words = [
            w.lower()
            for w in re.findall(r"[A-Za-z0-9_]{3,}", query_text)
            if w.lower() not in {"the", "and", "for", "with", "how", "what", "where", "from"}
        ]
        safe_limit = max(1, min(limit, 500))
        if not words:
            rows = conn.execute(
                """
                SELECT symbol_id, file_id, domain_id, qualified_name, symbol_kind,
                       start_line, end_line, signature_skeleton, contextual_summary,
                       allowed_scopes
                FROM ast_symbols
                WHERE domain_id = %s
                ORDER BY qualified_name
                LIMIT %s;
                """,
                (domain_id, safe_limit),
            ).fetchall()
        else:
            patterns = [f"%{w}%" for w in words[:8]]
            rows = conn.execute(
                """
                SELECT symbol_id, file_id, domain_id, qualified_name, symbol_kind,
                       start_line, end_line, signature_skeleton, contextual_summary,
                       allowed_scopes
                FROM ast_symbols
                WHERE domain_id = %s
                  AND (
                      qualified_name ILIKE ANY(%s)
                      OR signature_skeleton ILIKE ANY(%s)
                      OR contextual_summary ILIKE ANY(%s)
                  )
                ORDER BY qualified_name
                LIMIT %s;
                """,
                (domain_id, patterns, patterns, patterns, safe_limit),
            ).fetchall()
        return [
            ASTSymbolRecord(
                symbol_id=UUID(str(r[0])),
                file_id=UUID(str(r[1])),
                domain_id=str(r[2]),
                qualified_name=str(r[3]),
                symbol_kind=str(r[4]),
                start_line=int(str(r[5])),
                end_line=int(str(r[6])),
                signature_skeleton=str(r[7]),
                contextual_summary=str(r[8]),
                allowed_scopes=[str(s) for s in r[9]] if isinstance(r[9], list) else [],
                embedding=[],
            )
            for r in rows
        ]
