"""PostgreSQL RLS repository mixin: embedding spaces, telemetry, and domain artifact purge."""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg import Connection

from context_scout.db.repos.types_and_helpers import (
    _as_opt_dt,
    normalize_purge_selectors,
)
from context_scout.db.schema import (
    EmbeddingSpaceRecord,
)
from context_scout.retrieval.embedder import validate_non_zero_embedding
from context_scout.schemas.contracts import (
    ExecutionTelemetryRecord,
)

logger = logging.getLogger(__name__)


class _PgEmbeddingAndPurgeMixin:
    def register_embedding_space(
        self,
        conn: Connection[tuple[object, ...]],
        record: EmbeddingSpaceRecord,
    ) -> EmbeddingSpaceRecord:
        """Registers or updates an embedding space in PG17 under RLS."""
        if record.dimensions < 128 or record.dimensions > 4096:
            raise ValueError(
                f"Embedding space dimensions must be in [128, 4096], got {record.dimensions}"
            )
        conn.execute(
            "SELECT pg_advisory_xact_lock("
            "hashtextextended(COALESCE(current_setting('app.tenant_id', true), '') || ':' || %s, 0)"
            ");",
            (record.domain_id,),
        )
        effective_status = "active" if record.is_active else record.status
        if record.is_active:
            conn.execute(
                """
                UPDATE embedding_spaces
                SET is_active = FALSE,
                    status = CASE WHEN status = 'active' THEN 'retired' ELSE status END
                WHERE domain_id = %s AND space_id <> %s AND is_active = TRUE;
                """,
                (record.domain_id, record.space_id),
            )
        row = conn.execute(
            """
            INSERT INTO embedding_spaces (
                tenant_id, domain_id, space_id, model_id, provider,
                dimensions, quantization, status, is_active,
                index_operator_class, created_at, activated_at
            )
            VALUES (
                COALESCE(
                    NULLIF(current_setting('app.tenant_id', true), '')::uuid,
                    %s
                ),
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                COALESCE(%s, NOW()),
                CASE WHEN %s THEN COALESCE(%s, NOW()) ELSE %s END
            )
            ON CONFLICT (tenant_id, domain_id, space_id) DO UPDATE SET
                model_id = EXCLUDED.model_id,
                provider = EXCLUDED.provider,
                dimensions = EXCLUDED.dimensions,
                quantization = EXCLUDED.quantization,
                status = EXCLUDED.status,
                is_active = EXCLUDED.is_active,
                index_operator_class = EXCLUDED.index_operator_class,
                activated_at = CASE
                    WHEN EXCLUDED.is_active THEN COALESCE(EXCLUDED.activated_at, NOW())
                    ELSE embedding_spaces.activated_at
                END
            RETURNING tenant_id, domain_id, space_id, model_id, provider,
                      dimensions, quantization, status, is_active,
                      index_operator_class, created_at, activated_at;
            """,
            (
                record.tenant_id,
                record.domain_id,
                record.space_id,
                record.model_id,
                record.provider,
                record.dimensions,
                record.quantization,
                effective_status,
                record.is_active,
                record.index_operator_class,
                record.created_at,
                record.is_active,
                record.activated_at,
                record.activated_at,
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Failed to upsert embedding_space '{record.space_id}'")
        return EmbeddingSpaceRecord(
            tenant_id=UUID(str(row[0])),
            domain_id=str(row[1]),
            space_id=str(row[2]),
            model_id=str(row[3]),
            provider=str(row[4]),
            dimensions=int(str(row[5])),
            quantization=str(row[6]),  # type: ignore[arg-type]
            status=str(row[7]),  # type: ignore[arg-type]
            is_active=bool(row[8]),
            index_operator_class=str(row[9]),
            created_at=_as_opt_dt(row[10]),
            activated_at=_as_opt_dt(row[11]),
        )

    def get_active_embedding_space(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
    ) -> EmbeddingSpaceRecord | None:
        """Returns the currently active EmbeddingSpaceRecord for domain_id under RLS."""
        row = conn.execute(
            """
            SELECT tenant_id, domain_id, space_id, model_id, provider,
                   dimensions, quantization, status, is_active,
                   index_operator_class, created_at, activated_at
            FROM embedding_spaces
            WHERE domain_id = %s AND is_active = TRUE
            LIMIT 1;
            """,
            (domain_id,),
        ).fetchone()
        if row is None:
            return None
        return EmbeddingSpaceRecord(
            tenant_id=UUID(str(row[0])),
            domain_id=str(row[1]),
            space_id=str(row[2]),
            model_id=str(row[3]),
            provider=str(row[4]),
            dimensions=int(str(row[5])),
            quantization=str(row[6]),  # type: ignore[arg-type]
            status=str(row[7]),  # type: ignore[arg-type]
            is_active=bool(row[8]),
            index_operator_class=str(row[9]),
            created_at=_as_opt_dt(row[10]),
            activated_at=_as_opt_dt(row[11]),
        )

    def list_embedding_spaces(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
    ) -> list[EmbeddingSpaceRecord]:
        """Lists all registered embedding spaces for domain_id under RLS."""
        rows = conn.execute(
            """
            SELECT tenant_id, domain_id, space_id, model_id, provider,
                   dimensions, quantization, status, is_active,
                   index_operator_class, created_at, activated_at
            FROM embedding_spaces
            WHERE domain_id = %s
            ORDER BY is_active DESC, space_id ASC;
            """,
            (domain_id,),
        ).fetchall()
        return [
            EmbeddingSpaceRecord(
                tenant_id=UUID(str(r[0])),
                domain_id=str(r[1]),
                space_id=str(r[2]),
                model_id=str(r[3]),
                provider=str(r[4]),
                dimensions=int(str(r[5])),
                quantization=str(r[6]),  # type: ignore[arg-type]
                status=str(r[7]),  # type: ignore[arg-type]
                is_active=bool(r[8]),
                index_operator_class=str(r[9]),
                created_at=_as_opt_dt(r[10]),
                activated_at=_as_opt_dt(r[11]),
            )
            for r in rows
        ]

    def stage_shadow_embeddings(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        shadow_space_id: str,
        *,
        symbol_embeddings: Mapping[UUID, Sequence[float]] | None = None,
        chunk_embeddings: Mapping[UUID, Sequence[float]] | None = None,
    ) -> dict[str, int]:
        """Stages blue/green shadow halfvec(D) embeddings without disrupting active queries."""
        space_row = conn.execute(
            """
            SELECT dimensions
            FROM embedding_spaces
            WHERE domain_id = %s AND space_id = %s
            LIMIT 1;
            """,
            (domain_id, shadow_space_id),
        ).fetchone()
        if space_row is None:
            raise KeyError(
                f"Shadow embedding_space '{shadow_space_id}' not registered for '{domain_id}'"
            )
        expected_dim = int(str(space_row[0]))

        symbols_staged = 0
        if symbol_embeddings:
            sym_params: list[tuple[str, str, str, UUID]] = []
            for sym_id, vec in symbol_embeddings.items():
                checked = validate_non_zero_embedding(vec, expected_dim=expected_dim)
                hv_lit = "[" + ",".join(f"{v:.6f}" for v in checked) + "]"
                sym_params.append((shadow_space_id, hv_lit, domain_id, sym_id))
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    UPDATE ast_symbols
                    SET shadow_embedding_space_id = %s,
                        shadow_embedding_halfvec = %s::halfvec
                    WHERE domain_id = %s AND symbol_id = %s;
                    """,
                    sym_params,
                )
                symbols_staged = max(cur.rowcount or 0, 0)

        chunks_staged = 0
        if chunk_embeddings:
            chk_params: list[tuple[str, str, str, UUID]] = []
            for chunk_id, vec in chunk_embeddings.items():
                checked = validate_non_zero_embedding(vec, expected_dim=expected_dim)
                hv_lit = "[" + ",".join(f"{v:.6f}" for v in checked) + "]"
                chk_params.append((shadow_space_id, hv_lit, domain_id, chunk_id))
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    UPDATE doc_chunks
                    SET shadow_embedding_space_id = %s,
                        shadow_embedding_halfvec = %s::halfvec
                    WHERE domain_id = %s AND chunk_id = %s;
                    """,
                    chk_params,
                )
                chunks_staged = max(cur.rowcount or 0, 0)

        return {
            "symbols_staged": symbols_staged,
            "chunks_staged": chunks_staged,
        }

    def reindex_domain_shadow_space(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        shadow_space: EmbeddingSpaceRecord,
        embedder: Any,
    ) -> dict[str, int]:
        """Builds shadow halfvec(D) embeddings for all visible symbols & chunks in domain_id."""
        staged_record = dataclasses.replace(shadow_space, is_active=False, status="shadow")
        self.register_embedding_space(conn, staged_record)

        sym_rows = conn.execute(
            """
            SELECT symbol_id, qualified_name, signature_skeleton, contextual_summary
            FROM ast_symbols
            WHERE domain_id = %s
            ORDER BY symbol_id;
            """,
            (domain_id,),
        ).fetchall()
        sym_map: dict[UUID, Sequence[float]] = {}
        if sym_rows:
            sym_texts = [f"{r[1]} {r[2]} {r[3]}".strip() for r in sym_rows]
            sym_vecs = embedder.embed_batch(sym_texts, is_query=False)
            for r, vec in zip(sym_rows, sym_vecs, strict=True):
                sym_map[UUID(str(r[0]))] = vec

        chk_rows = conn.execute(
            """
            SELECT chunk_id, doc_title, section_heading, contextual_header, verbatim_content
            FROM doc_chunks
            WHERE domain_id = %s AND NOT is_stale
            ORDER BY chunk_id;
            """,
            (domain_id,),
        ).fetchall()
        chk_map: dict[UUID, Sequence[float]] = {}
        if chk_rows:
            chk_texts = [f"{r[1]} {r[2]} {r[3]} {r[4]}".strip() for r in chk_rows]
            chk_vecs = embedder.embed_batch(chk_texts, is_query=False)
            for r, vec in zip(chk_rows, chk_vecs, strict=True):
                chk_map[UUID(str(r[0]))] = vec

        return self.stage_shadow_embeddings(
            conn,
            domain_id,
            shadow_space.space_id,
            symbol_embeddings=sym_map,
            chunk_embeddings=chk_map,
        )

    def activate_embedding_space(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        target_space_id: str,
        *,
        require_complete_shadow: bool = True,
    ) -> EmbeddingSpaceRecord:
        """Atomically swaps active embedding_space pointer and promotes staged shadow halfvecs."""
        conn.execute(
            "SELECT pg_advisory_xact_lock("
            "hashtextextended(COALESCE(current_setting('app.tenant_id', true), '') || ':' || %s, 0)"
            ");",
            (domain_id,),
        )
        space_row = conn.execute(
            """
            SELECT dimensions
            FROM embedding_spaces
            WHERE domain_id = %s AND space_id = %s
            LIMIT 1;
            """,
            (domain_id, target_space_id),
        ).fetchone()
        if space_row is None:
            raise KeyError(
                f"Target embedding_space '{target_space_id}' not registered for '{domain_id}'"
            )

        if require_complete_shadow:
            missing_sym = conn.execute(
                """
                SELECT COUNT(*)
                FROM ast_symbols
                WHERE domain_id = %s
                  AND COALESCE(embedding_space_id, 'embeddinggemma-2-768') <> %s
                  AND (
                      shadow_embedding_space_id IS DISTINCT FROM %s
                      OR shadow_embedding_halfvec IS NULL
                  );
                """,
                (domain_id, target_space_id, target_space_id),
            ).fetchone()
            missing_chk = conn.execute(
                """
                SELECT COUNT(*)
                FROM doc_chunks
                WHERE domain_id = %s
                  AND NOT is_stale
                  AND COALESCE(embedding_space_id, 'embeddinggemma-2-768') <> %s
                  AND (
                      shadow_embedding_space_id IS DISTINCT FROM %s
                      OR shadow_embedding_halfvec IS NULL
                  );
                """,
                (domain_id, target_space_id, target_space_id),
            ).fetchone()
            unready_syms = int(str((missing_sym or (0,))[0]))
            unready_chks = int(str((missing_chk or (0,))[0]))
            if unready_syms > 0 or unready_chks > 0:
                raise RuntimeError(
                    f"Cannot activate embedding_space '{target_space_id}' for '{domain_id}': "
                    f"{unready_syms} ast_symbols and {unready_chks} doc_chunks lack staged "
                    "shadow halfvec embeddings."
                )

        conn.execute(
            """
            UPDATE ast_symbols
            SET embedding_space_id = shadow_embedding_space_id,
                embedding_halfvec = shadow_embedding_halfvec,
                shadow_embedding_space_id = NULL,
                shadow_embedding_halfvec = NULL
            WHERE domain_id = %s
              AND shadow_embedding_space_id = %s
              AND shadow_embedding_halfvec IS NOT NULL;
            """,
            (domain_id, target_space_id),
        )
        conn.execute(
            """
            UPDATE doc_chunks
            SET embedding_space_id = shadow_embedding_space_id,
                embedding_halfvec = shadow_embedding_halfvec,
                shadow_embedding_space_id = NULL,
                shadow_embedding_halfvec = NULL
            WHERE domain_id = %s
              AND shadow_embedding_space_id = %s
              AND shadow_embedding_halfvec IS NOT NULL;
            """,
            (domain_id, target_space_id),
        )
        conn.execute(
            """
            UPDATE embedding_spaces
            SET is_active = FALSE,
                status = CASE WHEN status = 'active' THEN 'retired' ELSE status END
            WHERE domain_id = %s AND space_id <> %s AND is_active = TRUE;
            """,
            (domain_id, target_space_id),
        )
        conn.execute(
            """
            UPDATE embedding_spaces
            SET is_active = TRUE,
                status = 'active',
                activated_at = NOW()
            WHERE domain_id = %s AND space_id = %s;
            """,
            (domain_id, target_space_id),
        )
        active = self.get_active_embedding_space(conn, domain_id)
        if active is None or active.space_id != target_space_id:
            raise RuntimeError(
                f"Atomic pointer swap failed for embedding_space '{target_space_id}'"
            )
        return active

    def record_execution_telemetry(
        self, conn: Connection[tuple[object, ...]], record: ExecutionTelemetryRecord
    ) -> None:
        mode_val = str(record.execution_mode)
        missing_json = json.dumps(record.missing_context_requests)
        conn.execute(
            """
            INSERT INTO execution_telemetry (
                trace_id, tenant_id, domain_id, principal_id, principal_role,
                execution_mode, local_tokens_read, context_pack_tokens_sent,
                paid_tokens_saved, estimated_usd_saved, preflight_passed_first_try,
                zero_touch_sufficient, missing_context_requests, latency_ms,
                created_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, COALESCE(%s, NOW())
            )
            ON CONFLICT (trace_id) DO UPDATE SET
                local_tokens_read = EXCLUDED.local_tokens_read,
                context_pack_tokens_sent = EXCLUDED.context_pack_tokens_sent,
                paid_tokens_saved = EXCLUDED.paid_tokens_saved,
                estimated_usd_saved = EXCLUDED.estimated_usd_saved,
                preflight_passed_first_try = EXCLUDED.preflight_passed_first_try,
                zero_touch_sufficient = EXCLUDED.zero_touch_sufficient,
                missing_context_requests = EXCLUDED.missing_context_requests,
                latency_ms = EXCLUDED.latency_ms;
            """,
            (
                record.trace_id,
                record.tenant_id,
                record.domain_id,
                record.principal_id,
                record.principal_role,
                mode_val,
                record.local_tokens_read,
                record.context_pack_tokens_sent,
                record.paid_tokens_saved,
                record.estimated_usd_saved,
                record.preflight_passed_first_try,
                record.zero_touch_sufficient,
                missing_json,
                record.latency_ms,
                record.created_at,
            ),
        )

    def get_execution_telemetry(
        self, conn: Connection[tuple[object, ...]], trace_id: UUID
    ) -> ExecutionTelemetryRecord | None:
        row = conn.execute(
            """
            SELECT
                trace_id, domain_id, principal_id, principal_role,
                execution_mode, local_tokens_read, context_pack_tokens_sent,
                paid_tokens_saved, estimated_usd_saved, preflight_passed_first_try,
                zero_touch_sufficient, missing_context_requests, latency_ms,
                created_at, tenant_id
            FROM execution_telemetry
            WHERE trace_id = %s;
            """,
            (trace_id,),
        ).fetchone()
        if row is None:
            return None
        raw_missing = row[11]
        if isinstance(raw_missing, str):
            missing_requests = json.loads(raw_missing)
        elif isinstance(raw_missing, list):
            missing_requests = raw_missing
        else:
            missing_requests = []
        zero_touch_val: bool | None = bool(row[10]) if row[10] is not None else None
        created_at_val: datetime | None = row[13] if isinstance(row[13], datetime) else None
        return ExecutionTelemetryRecord(
            trace_id=UUID(str(row[0])),
            domain_id=str(row[1]),
            principal_id=str(row[2]),
            principal_role=str(row[3]),
            execution_mode=str(row[4]),  # type: ignore[arg-type]
            local_tokens_read=int(str(row[5])),
            context_pack_tokens_sent=int(str(row[6])),
            paid_tokens_saved=int(str(row[7])),
            estimated_usd_saved=float(str(row[8])),
            preflight_passed_first_try=bool(row[9]),
            zero_touch_sufficient=zero_touch_val,
            missing_context_requests=missing_requests,
            latency_ms=int(str(row[12])),
            created_at=created_at_val,
            tenant_id=UUID(str(row[14])),
        )

    def get_principal_paid_tokens_consumed(
        self,
        conn: Connection[tuple[object, ...]],
        principal_id: str,
        since_hours: int = 24,
    ) -> int:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(context_pack_tokens_sent), 0)
            FROM execution_telemetry
            WHERE principal_id = %s
              AND execution_mode = 'gemini_handoff'
              AND created_at >= NOW() - (%s || ' hours')::interval;
            """,
            (principal_id, str(since_hours)),
        ).fetchone()
        return int(str(row[0])) if row and row[0] is not None else 0

    def purge_domain_artifacts(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        *,
        doc_id: str | None = None,
        confluence_page_id: str | None = None,
        jira_issue_key: str | None = None,
        repo_name: str | None = None,
        file_path: str | None = None,
        commit_sha: str | None = None,
        blob_sha: str | None = None,
        include_derived_wiki: bool = True,
    ) -> dict[str, int]:
        """Hard-deletes matching artifacts across PostgreSQL 17 tables under Forced RLS."""
        norm = normalize_purge_selectors(
            domain_id,
            doc_id=doc_id,
            confluence_page_id=confluence_page_id,
            jira_issue_key=jira_issue_key,
            repo_name=repo_name,
            file_path=file_path,
            commit_sha=commit_sha,
            blob_sha=blob_sha,
        )
        domain_id = norm.domain_id
        doc_id = norm.doc_id
        confluence_page_id = norm.confluence_page_id
        jira_issue_key = norm.jira_issue_key
        repo_name = norm.repo_name
        file_path = norm.file_path
        commit_sha = norm.commit_sha
        blob_sha = norm.blob_sha
        counts: dict[str, int] = {
            "side_effects": 0,
            "overlay_file_deltas": 0,
            "ref_snapshots": 0,
            "ast_edges": 0,
            "ast_symbols": 0,
            "code_files": 0,
            "repo_commits": 0,
            "repositories": 0,
            "doc_chunks": 0,
            "confluence_pages": 0,
            "jira_issues": 0,
        }
        target_doc_id = doc_id
        if target_doc_id is None and confluence_page_id:
            target_doc_id = f"confluence:{confluence_page_id}"
        elif target_doc_id is None and jira_issue_key:
            target_doc_id = f"jira:{jira_issue_key}"

        if target_doc_id:
            cur = conn.execute(
                "DELETE FROM doc_chunks WHERE domain_id = %s AND doc_id = %s;",
                (domain_id, target_doc_id),
            )
            counts["doc_chunks"] += max(cur.rowcount or 0, 0)

        if confluence_page_id:
            cur = conn.execute(
                "DELETE FROM confluence_pages WHERE domain_id = %s AND confluence_page_id = %s;",
                (domain_id, confluence_page_id),
            )
            counts["confluence_pages"] += max(cur.rowcount or 0, 0)

        if jira_issue_key:
            cur = conn.execute(
                "DELETE FROM jira_issues WHERE domain_id = %s AND issue_key = %s;",
                (domain_id, jira_issue_key),
            )
            counts["jira_issues"] += max(cur.rowcount or 0, 0)

        target_file_ids: list[UUID] = []
        if repo_name and file_path:
            clean_repo = repo_name.replace("/", "_")
            rows = conn.execute(
                """
                SELECT file_id FROM code_files
                WHERE domain_id = %s AND (repo_name = %s OR repo_name = %s) AND file_path = %s;
                """,
                (domain_id, repo_name, clean_repo, file_path),
            ).fetchall()
            target_file_ids = [UUID(str(r[0])) for r in rows]
            cur_ov = conn.execute(
                """
                DELETE FROM overlay_file_deltas
                WHERE domain_id = %s AND (repo_name = %s OR repo_name = %s)
                  AND (file_path = %s OR previous_file_path = %s);
                """,
                (domain_id, repo_name, clean_repo, file_path, file_path),
            )
            counts["overlay_file_deltas"] += max(cur_ov.rowcount or 0, 0)
            cur_se = conn.execute(
                """
                DELETE FROM side_effects
                WHERE domain_id = %s AND (repo_name = %s OR repo_name = %s) AND file_path = %s;
                """,
                (domain_id, repo_name, clean_repo, file_path),
            )
            counts["side_effects"] += max(cur_se.rowcount or 0, 0)
        elif blob_sha:
            rows = conn.execute(
                "SELECT file_id FROM code_files WHERE domain_id = %s AND git_blob_sha = %s;",
                (domain_id, blob_sha),
            ).fetchall()
            target_file_ids = [UUID(str(r[0])) for r in rows]
            cur_ov = conn.execute(
                """
                DELETE FROM overlay_file_deltas
                WHERE domain_id = %s AND (overlay_blob_sha = %s OR base_blob_sha = %s);
                """,
                (domain_id, blob_sha, blob_sha),
            )
            counts["overlay_file_deltas"] += max(cur_ov.rowcount or 0, 0)
        elif commit_sha:
            rows = conn.execute(
                "SELECT file_id FROM code_files WHERE domain_id = %s AND git_commit_sha = %s;",
                (domain_id, commit_sha),
            ).fetchall()
            target_file_ids = [UUID(str(r[0])) for r in rows]
            cur_rc = conn.execute(
                """
                DELETE FROM repo_commits
                WHERE commit_sha = %s
                  AND repo_id IN (SELECT repo_id FROM repositories WHERE domain_id = %s);
                """,
                (commit_sha, domain_id),
            )
            counts["repo_commits"] += max(cur_rc.rowcount or 0, 0)
            cur_ov = conn.execute(
                """
                DELETE FROM overlay_file_deltas
                WHERE snapshot_id IN (
                    SELECT snapshot_id FROM ref_snapshots
                    WHERE domain_id = %s
                      AND (head_commit_sha = %s OR base_commit_sha = %s)
                );
                """,
                (domain_id, commit_sha, commit_sha),
            )
            counts["overlay_file_deltas"] += max(cur_ov.rowcount or 0, 0)
            cur_rs = conn.execute(
                """
                DELETE FROM ref_snapshots
                WHERE domain_id = %s AND (head_commit_sha = %s OR base_commit_sha = %s);
                """,
                (domain_id, commit_sha, commit_sha),
            )
            counts["ref_snapshots"] += max(cur_rs.rowcount or 0, 0)
        elif repo_name:
            clean_repo = repo_name.replace("/", "_")
            rows = conn.execute(
                """
                SELECT file_id FROM code_files
                WHERE domain_id = %s AND (repo_name = %s OR repo_name = %s);
                """,
                (domain_id, repo_name, clean_repo),
            ).fetchall()
            target_file_ids = [UUID(str(r[0])) for r in rows]
            cur_se = conn.execute(
                """
                DELETE FROM side_effects
                WHERE domain_id = %s AND (repo_name = %s OR repo_name = %s);
                """,
                (domain_id, repo_name, clean_repo),
            )
            counts["side_effects"] += max(cur_se.rowcount or 0, 0)
            cur_ov = conn.execute(
                """
                DELETE FROM overlay_file_deltas
                WHERE domain_id = %s AND (repo_name = %s OR repo_name = %s);
                """,
                (domain_id, repo_name, clean_repo),
            )
            counts["overlay_file_deltas"] += max(cur_ov.rowcount or 0, 0)
            cur_rs = conn.execute(
                """
                DELETE FROM ref_snapshots
                WHERE domain_id = %s AND (repo_name = %s OR repo_name = %s);
                """,
                (domain_id, repo_name, clean_repo),
            )
            counts["ref_snapshots"] += max(cur_rs.rowcount or 0, 0)
            cur_rc = conn.execute(
                """
                DELETE FROM repo_commits
                WHERE repo_id IN (
                    SELECT repo_id FROM repositories
                    WHERE domain_id = %s AND (repo_name = %s OR repo_name = %s)
                );
                """,
                (domain_id, repo_name, clean_repo),
            )
            counts["repo_commits"] += max(cur_rc.rowcount or 0, 0)
            cur_rep = conn.execute(
                """
                DELETE FROM repositories
                WHERE domain_id = %s AND (repo_name = %s OR repo_name = %s);
                """,
                (domain_id, repo_name, clean_repo),
            )
            counts["repositories"] += max(cur_rep.rowcount or 0, 0)

        if target_file_ids:
            sym_rows = conn.execute(
                "SELECT symbol_id FROM ast_symbols WHERE file_id = ANY(%s);",
                (target_file_ids,),
            ).fetchall()
            sym_ids = [UUID(str(r[0])) for r in sym_rows]
            cur_se_f = conn.execute(
                "DELETE FROM side_effects WHERE file_id = ANY(%s);",
                (target_file_ids,),
            )
            counts["side_effects"] += max(cur_se_f.rowcount or 0, 0)
            if sym_ids:
                cur_ed = conn.execute(
                    """
                    DELETE FROM ast_edges
                    WHERE domain_id = %s
                      AND (source_symbol_id = ANY(%s) OR target_symbol_id = ANY(%s));
                    """,
                    (domain_id, sym_ids, sym_ids),
                )
                counts["ast_edges"] += max(cur_ed.rowcount or 0, 0)
            cur_sy = conn.execute(
                "DELETE FROM ast_symbols WHERE file_id = ANY(%s);",
                (target_file_ids,),
            )
            counts["ast_symbols"] += max(cur_sy.rowcount or 0, 0)
            cur_cf = conn.execute(
                "DELETE FROM code_files WHERE file_id = ANY(%s);",
                (target_file_ids,),
            )
            counts["code_files"] += max(cur_cf.rowcount or 0, 0)

        if include_derived_wiki and repo_name:
            clean_repo = repo_name.replace("/", "_")
            cur_wiki = conn.execute(
                """
                DELETE FROM doc_chunks
                WHERE domain_id = %s
                  AND source_type = 'repo_wiki'
                  AND (
                      doc_id = %s OR doc_id LIKE %s
                      OR doc_id = %s OR doc_id LIKE %s
                  );
                """,
                (
                    domain_id,
                    f"repo_wiki:{repo_name}",
                    f"repo_wiki:{repo_name}:%",
                    f"repo_wiki:{clean_repo}",
                    f"repo_wiki:{clean_repo}:%",
                ),
            )
            counts["doc_chunks"] += max(cur_wiki.rowcount or 0, 0)

        return counts
