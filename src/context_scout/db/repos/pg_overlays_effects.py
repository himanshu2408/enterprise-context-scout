"""PostgreSQL RLS repository mixin: ref snapshots, CoW overlay deltas, ownership, side effects."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import cast
from uuid import UUID

from psycopg import Connection

from context_scout.db.repos.types_and_helpers import (
    OwnershipEdgeRecord,
    _as_opt_dt,
)
from context_scout.db.schema import (
    ASTSymbolRecord,
    EffectCategory,
    OverlayFileDeltaRecord,
    RefSnapshotRecord,
    ResolvedOverlayFileRecord,
    SideEffectRecord,
    SideEffectSeverity,
)

logger = logging.getLogger(__name__)


class _PgOverlaysAndEffectsMixin:
    @staticmethod
    def _row_to_ref_snapshot(row: tuple[object, ...]) -> RefSnapshotRecord:
        meta_raw = row[13]
        meta = meta_raw if isinstance(meta_raw, dict) else json.loads(str(meta_raw or "{}"))
        return RefSnapshotRecord(
            snapshot_id=UUID(str(row[0])),
            tenant_id=UUID(str(row[1])),
            domain_id=str(row[2]),
            repo_id=UUID(str(row[3])) if row[3] is not None else None,
            repo_name=str(row[4]),
            ref_name=str(row[5]),
            ref_kind=row[6],  # type: ignore[arg-type]
            mr_iid=str(row[7]) if row[7] is not None else None,
            base_snapshot_id=UUID(str(row[8])) if row[8] is not None else None,
            base_commit_sha=str(row[9]),
            head_commit_sha=str(row[10]),
            state=row[11],  # type: ignore[arg-type]
            allowed_scopes=[str(s) for s in row[12]] if isinstance(row[12], list) else [],
            metadata_json=meta,
            created_at=_as_opt_dt(row[14]),
            updated_at=_as_opt_dt(row[15]),
        )

    @staticmethod
    def _row_to_overlay_delta(row: tuple[object, ...]) -> OverlayFileDeltaRecord:
        syms_raw = row[15]
        edges_raw = row[16]
        syms = syms_raw if isinstance(syms_raw, list) else json.loads(str(syms_raw or "[]"))
        edges = edges_raw if isinstance(edges_raw, list) else json.loads(str(edges_raw or "[]"))
        return OverlayFileDeltaRecord(
            delta_id=UUID(str(row[0])),
            tenant_id=UUID(str(row[1])),
            snapshot_id=UUID(str(row[2])),
            domain_id=str(row[3]),
            repo_name=str(row[4]),
            file_path=str(row[5]),
            previous_file_path=str(row[6]) if row[6] is not None else None,
            change_type=row[7],  # type: ignore[arg-type]
            base_blob_sha=str(row[8]) if row[8] is not None else None,
            overlay_blob_sha=str(row[9]) if row[9] is not None else None,
            language=str(row[10]),
            raw_content=str(row[11]) if row[11] is not None else None,
            line_count=int(str(row[12])),
            ranking_weight=float(str(row[13])),
            is_generated=bool(row[14]),
            is_vendored=bool(row[17]),
            symbols_json=syms,
            edges_json=edges,
            allowed_scopes=[str(s) for s in row[18]] if isinstance(row[18], list) else [],
            indexed_at=_as_opt_dt(row[19]),
        )

    def upsert_ref_snapshot(
        self,
        conn: Connection[tuple[object, ...]],
        record: RefSnapshotRecord,
    ) -> RefSnapshotRecord:
        row = conn.execute(
            """
            INSERT INTO ref_snapshots (
                snapshot_id, tenant_id, domain_id, repo_id, repo_name, ref_name,
                ref_kind, mr_iid, base_snapshot_id, base_commit_sha, head_commit_sha,
                state, allowed_scopes, metadata_json, created_at, updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb,
                COALESCE(%s, NOW()), COALESCE(%s, NOW())
            )
            ON CONFLICT (tenant_id, domain_id, repo_name, ref_name) DO UPDATE SET
                repo_id = EXCLUDED.repo_id,
                ref_kind = EXCLUDED.ref_kind,
                mr_iid = EXCLUDED.mr_iid,
                base_snapshot_id = EXCLUDED.base_snapshot_id,
                base_commit_sha = EXCLUDED.base_commit_sha,
                head_commit_sha = EXCLUDED.head_commit_sha,
                state = EXCLUDED.state,
                allowed_scopes = EXCLUDED.allowed_scopes,
                metadata_json = EXCLUDED.metadata_json,
                updated_at = NOW()
            RETURNING snapshot_id, tenant_id, domain_id, repo_id, repo_name, ref_name,
                      ref_kind, mr_iid, base_snapshot_id, base_commit_sha, head_commit_sha,
                      state, allowed_scopes, metadata_json, created_at, updated_at;
            """,
            (
                record.snapshot_id,
                record.tenant_id,
                record.domain_id,
                record.repo_id,
                record.repo_name,
                record.ref_name,
                record.ref_kind,
                record.mr_iid,
                record.base_snapshot_id,
                record.base_commit_sha,
                record.head_commit_sha,
                record.state,
                record.allowed_scopes,
                json.dumps(record.metadata_json),
                record.created_at,
                record.updated_at,
            ),
        ).fetchone()
        assert row is not None
        return self._row_to_ref_snapshot(row)

    def get_ref_snapshot(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        *,
        snapshot_id: UUID | None = None,
        repo_name: str | None = None,
        ref_name: str | None = None,
        mr_iid: str | None = None,
    ) -> RefSnapshotRecord | None:
        if snapshot_id is not None:
            row = conn.execute(
                """
                SELECT snapshot_id, tenant_id, domain_id, repo_id, repo_name, ref_name,
                       ref_kind, mr_iid, base_snapshot_id, base_commit_sha, head_commit_sha,
                       state, allowed_scopes, metadata_json, created_at, updated_at
                FROM ref_snapshots
                WHERE domain_id = %s AND snapshot_id = %s;
                """,
                (domain_id, snapshot_id),
            ).fetchone()
        elif repo_name is not None and ref_name is not None:
            row = conn.execute(
                """
                SELECT snapshot_id, tenant_id, domain_id, repo_id, repo_name, ref_name,
                       ref_kind, mr_iid, base_snapshot_id, base_commit_sha, head_commit_sha,
                       state, allowed_scopes, metadata_json, created_at, updated_at
                FROM ref_snapshots
                WHERE domain_id = %s AND repo_name = %s AND ref_name = %s;
                """,
                (domain_id, repo_name, ref_name),
            ).fetchone()
        elif repo_name is not None and mr_iid is not None:
            row = conn.execute(
                """
                SELECT snapshot_id, tenant_id, domain_id, repo_id, repo_name, ref_name,
                       ref_kind, mr_iid, base_snapshot_id, base_commit_sha, head_commit_sha,
                       state, allowed_scopes, metadata_json, created_at, updated_at
                FROM ref_snapshots
                WHERE domain_id = %s AND repo_name = %s AND mr_iid = %s
                ORDER BY updated_at DESC
                LIMIT 1;
                """,
                (domain_id, repo_name, mr_iid),
            ).fetchone()
        else:
            raise ValueError("Specify snapshot_id, (repo_name, ref_name), or (repo_name, mr_iid)")
        return self._row_to_ref_snapshot(row) if row is not None else None

    def list_ref_snapshots(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        *,
        repo_name: str | None = None,
        state: str | None = None,
    ) -> list[RefSnapshotRecord]:
        rows = conn.execute(
            """
            SELECT snapshot_id, tenant_id, domain_id, repo_id, repo_name, ref_name,
                   ref_kind, mr_iid, base_snapshot_id, base_commit_sha, head_commit_sha,
                   state, allowed_scopes, metadata_json, created_at, updated_at
            FROM ref_snapshots
            WHERE domain_id = %s
              AND (%s::text IS NULL OR repo_name = %s)
              AND (%s::text IS NULL OR state = %s)
            ORDER BY updated_at DESC, ref_name ASC;
            """,
            (domain_id, repo_name, repo_name, state, state),
        ).fetchall()
        return [self._row_to_ref_snapshot(r) for r in rows]

    def delete_ref_snapshot(self, conn: Connection[tuple[object, ...]], snapshot_id: UUID) -> int:
        cur = conn.execute("DELETE FROM ref_snapshots WHERE snapshot_id = %s;", (snapshot_id,))
        return max(cur.rowcount or 0, 0)

    def upsert_overlay_file_delta(
        self,
        conn: Connection[tuple[object, ...]],
        record: OverlayFileDeltaRecord,
    ) -> OverlayFileDeltaRecord:
        row = conn.execute(
            """
            INSERT INTO overlay_file_deltas (
                delta_id, tenant_id, snapshot_id, domain_id, repo_name, file_path,
                previous_file_path, change_type, base_blob_sha, overlay_blob_sha,
                language, raw_content, line_count, ranking_weight, is_generated,
                is_vendored, symbols_json, edges_json, allowed_scopes, indexed_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, COALESCE(%s, NOW())
            )
            ON CONFLICT (tenant_id, snapshot_id, file_path) DO UPDATE SET
                previous_file_path = EXCLUDED.previous_file_path,
                change_type = EXCLUDED.change_type,
                base_blob_sha = EXCLUDED.base_blob_sha,
                overlay_blob_sha = EXCLUDED.overlay_blob_sha,
                language = EXCLUDED.language,
                raw_content = EXCLUDED.raw_content,
                line_count = EXCLUDED.line_count,
                ranking_weight = EXCLUDED.ranking_weight,
                is_generated = EXCLUDED.is_generated,
                is_vendored = EXCLUDED.is_vendored,
                symbols_json = EXCLUDED.symbols_json,
                edges_json = EXCLUDED.edges_json,
                allowed_scopes = EXCLUDED.allowed_scopes,
                indexed_at = NOW()
            RETURNING delta_id, tenant_id, snapshot_id, domain_id, repo_name, file_path,
                      previous_file_path, change_type, base_blob_sha, overlay_blob_sha,
                      language, raw_content, line_count, ranking_weight, is_generated,
                      symbols_json, edges_json, is_vendored, allowed_scopes, indexed_at;
            """,
            (
                record.delta_id,
                record.tenant_id,
                record.snapshot_id,
                record.domain_id,
                record.repo_name,
                record.file_path,
                record.previous_file_path,
                record.change_type,
                record.base_blob_sha,
                record.overlay_blob_sha,
                record.language,
                record.raw_content,
                record.line_count,
                record.ranking_weight,
                record.is_generated,
                record.is_vendored,
                json.dumps(record.symbols_json),
                json.dumps(record.edges_json),
                record.allowed_scopes,
                record.indexed_at,
            ),
        ).fetchone()
        assert row is not None
        return self._row_to_overlay_delta(row)

    def list_overlay_file_deltas(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        snapshot_id: UUID,
    ) -> list[OverlayFileDeltaRecord]:
        rows = conn.execute(
            """
            SELECT delta_id, tenant_id, snapshot_id, domain_id, repo_name, file_path,
                   previous_file_path, change_type, base_blob_sha, overlay_blob_sha,
                   language, raw_content, line_count, ranking_weight, is_generated,
                   symbols_json, edges_json, is_vendored, allowed_scopes, indexed_at
            FROM overlay_file_deltas
            WHERE domain_id = %s AND snapshot_id = %s
            ORDER BY file_path ASC;
            """,
            (domain_id, snapshot_id),
        ).fetchall()
        return [self._row_to_overlay_delta(r) for r in rows]

    def resolve_overlay_files(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        snapshot_id: UUID,
        *,
        file_path: str | None = None,
    ) -> list[ResolvedOverlayFileRecord]:
        """Resolves effective files for an MR/branch via COALESCE(overlay, base_snapshot)."""
        path_filter_delta = "AND d.file_path = %s" if file_path is not None else ""
        path_filter_base = "AND cf.file_path = %s" if file_path is not None else ""
        sql = f"""
            WITH RECURSIVE target_snap AS (
                SELECT snapshot_id, tenant_id, domain_id, repo_name,
                       base_snapshot_id, head_commit_sha
                FROM ref_snapshots
                WHERE domain_id = %s AND snapshot_id = %s
            ),
            snapshot_chain AS (
                SELECT snapshot_id, base_snapshot_id, 0 AS depth
                FROM target_snap
                UNION ALL
                SELECT parent.snapshot_id, parent.base_snapshot_id, sc.depth + 1
                FROM ref_snapshots parent
                JOIN snapshot_chain sc ON parent.snapshot_id = sc.base_snapshot_id
                WHERE sc.depth < 16
            ),
            effective_deltas AS (
                SELECT DISTINCT ON (d.file_path)
                       d.delta_id, d.snapshot_id, d.domain_id, d.repo_name, d.file_path,
                       d.previous_file_path, d.change_type, d.overlay_blob_sha, d.language,
                       d.raw_content, d.line_count, d.ranking_weight, d.is_generated,
                       d.is_vendored, d.allowed_scopes, d.tenant_id
                FROM overlay_file_deltas d
                JOIN snapshot_chain sc ON sc.snapshot_id = d.snapshot_id
                WHERE d.domain_id = %s {path_filter_delta}
                ORDER BY d.file_path, sc.depth ASC
            ),
            renamed_sources AS (
                SELECT DISTINCT d2.previous_file_path AS old_path
                FROM overlay_file_deltas d2
                JOIN snapshot_chain sc2 ON sc2.snapshot_id = d2.snapshot_id
                WHERE d2.previous_file_path IS NOT NULL
                  AND d2.previous_file_path <> d2.file_path
            ),
            base_files AS (
                SELECT cf.file_id, cf.domain_id, cf.repo_name, cf.file_path, cf.git_commit_sha,
                       cf.git_blob_sha, cf.language, cf.allowed_scopes, cf.raw_content,
                       cf.line_count, COALESCE(cf.ranking_weight, 1.0) AS ranking_weight,
                       COALESCE(cf.is_generated, FALSE) AS is_generated,
                       COALESCE(cf.is_vendored, FALSE) AS is_vendored, cf.tenant_id
                FROM code_files cf
                JOIN target_snap ts ON cf.domain_id = ts.domain_id AND cf.repo_name = ts.repo_name
                WHERE NOT EXISTS (
                    SELECT 1 FROM renamed_sources rs WHERE rs.old_path = cf.file_path
                )
                {path_filter_base}
            )
            SELECT
                COALESCE(d.domain_id, b.domain_id) AS domain_id,
                COALESCE(d.repo_name, b.repo_name) AS repo_name,
                COALESCE(d.file_path, b.file_path) AS file_path,
                CASE
                    WHEN d.delta_id IS NOT NULL THEN ts.head_commit_sha
                    ELSE b.git_commit_sha
                END AS git_commit_sha,
                COALESCE(d.overlay_blob_sha, b.git_blob_sha) AS git_blob_sha,
                COALESCE(d.language, b.language) AS language,
                COALESCE(d.allowed_scopes, b.allowed_scopes) AS allowed_scopes,
                COALESCE(d.raw_content, b.raw_content) AS raw_content,
                COALESCE(d.line_count, b.line_count) AS line_count,
                CASE
                    WHEN d.delta_id IS NOT NULL THEN 'overlay'
                    ELSE 'base_snapshot'
                END AS source_layer,
                ts.snapshot_id,
                b.file_id AS base_file_id,
                d.delta_id,
                d.change_type,
                COALESCE(d.ranking_weight, b.ranking_weight, 1.0) AS ranking_weight,
                COALESCE(d.is_generated, b.is_generated, FALSE) AS is_generated,
                COALESCE(d.is_vendored, b.is_vendored, FALSE) AS is_vendored,
                COALESCE(d.tenant_id, b.tenant_id) AS tenant_id
            FROM effective_deltas d
            FULL OUTER JOIN base_files b ON d.file_path = b.file_path
            CROSS JOIN target_snap ts
            WHERE COALESCE(d.change_type, 'base') <> 'deleted'
            ORDER BY COALESCE(d.file_path, b.file_path) ASC;
        """
        params: list[object] = [domain_id, snapshot_id, domain_id]
        if file_path is not None:
            params.extend([file_path, file_path])
        rows = conn.execute(sql, tuple(params)).fetchall()
        return [
            ResolvedOverlayFileRecord(
                domain_id=str(r[0]),
                repo_name=str(r[1]),
                file_path=str(r[2]),
                git_commit_sha=str(r[3]),
                git_blob_sha=str(r[4]),
                language=str(r[5]),
                allowed_scopes=[str(s) for s in r[6]] if isinstance(r[6], list) else [],
                raw_content=str(r[7]),
                line_count=int(str(r[8])),
                source_layer=r[9],  # type: ignore[arg-type]
                snapshot_id=UUID(str(r[10])) if r[10] is not None else None,
                base_file_id=UUID(str(r[11])) if r[11] is not None else None,
                delta_id=UUID(str(r[12])) if r[12] is not None else None,
                change_type=str(r[13]) if r[13] is not None else None,
                ranking_weight=float(str(r[14])),
                is_generated=bool(r[15]),
                is_vendored=bool(r[16]),
                tenant_id=UUID(str(r[17])),
            )
            for r in rows
        ]

    def get_overlay_file(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        snapshot_id: UUID,
        file_path: str,
    ) -> ResolvedOverlayFileRecord | None:
        items = self.resolve_overlay_files(conn, domain_id, snapshot_id, file_path=file_path)
        return items[0] if items else None

    def resolve_overlay_symbols(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        snapshot_id: UUID,
    ) -> list[ASTSymbolRecord]:
        """Resolves effective AST symbols for an overlay snapshot without mutating base symbols."""
        snap = self.get_ref_snapshot(conn, domain_id, snapshot_id=snapshot_id)
        if snap is None:
            return []

        deltas_rows = conn.execute(
            """
            WITH RECURSIVE snapshot_chain AS (
                SELECT snapshot_id, base_snapshot_id, 0 AS depth
                FROM ref_snapshots
                WHERE domain_id = %s AND snapshot_id = %s
                UNION ALL
                SELECT parent.snapshot_id, parent.base_snapshot_id, sc.depth + 1
                FROM ref_snapshots parent
                JOIN snapshot_chain sc ON parent.snapshot_id = sc.base_snapshot_id
                WHERE sc.depth < 16
            ),
            effective_deltas AS (
                SELECT DISTINCT ON (d.file_path)
                       d.delta_id, d.domain_id, d.file_path, d.previous_file_path,
                       d.change_type, d.symbols_json, d.allowed_scopes, d.tenant_id
                FROM overlay_file_deltas d
                JOIN snapshot_chain sc ON sc.snapshot_id = d.snapshot_id
                WHERE d.domain_id = %s
                ORDER BY d.file_path, sc.depth ASC
            )
            SELECT delta_id, domain_id, file_path, previous_file_path,
                   change_type, symbols_json, allowed_scopes, tenant_id
            FROM effective_deltas;
            """,
            (domain_id, snapshot_id, domain_id),
        ).fetchall()

        renamed_old_rows = conn.execute(
            """
            WITH RECURSIVE snapshot_chain AS (
                SELECT snapshot_id, base_snapshot_id, 0 AS depth
                FROM ref_snapshots
                WHERE domain_id = %s AND snapshot_id = %s
                UNION ALL
                SELECT parent.snapshot_id, parent.base_snapshot_id, sc.depth + 1
                FROM ref_snapshots parent
                JOIN snapshot_chain sc ON parent.snapshot_id = sc.base_snapshot_id
                WHERE sc.depth < 16
            )
            SELECT DISTINCT d.previous_file_path
            FROM overlay_file_deltas d
            JOIN snapshot_chain sc ON sc.snapshot_id = d.snapshot_id
            WHERE d.previous_file_path IS NOT NULL;
            """,
            (domain_id, snapshot_id),
        ).fetchall()

        suppressed_paths: set[str] = {str(r[0]) for r in renamed_old_rows if r[0] is not None}
        overlay_symbols: list[ASTSymbolRecord] = []

        for r in deltas_rows:
            delta_id = UUID(str(r[0]))
            f_path = str(r[2])
            prev_path = str(r[3]) if r[3] is not None else None
            change_type = str(r[4])
            suppressed_paths.add(f_path)
            if prev_path:
                suppressed_paths.add(prev_path)
            if change_type == "deleted":
                continue
            syms_raw = r[5] if isinstance(r[5], list) else json.loads(str(r[5] or "[]"))
            scopes = [str(s) for s in r[6]] if isinstance(r[6], list) else []
            tenant_id = UUID(str(r[7]))
            for sym_dict in syms_raw:
                if not isinstance(sym_dict, dict):
                    continue
                overlay_symbols.append(
                    ASTSymbolRecord(
                        symbol_id=UUID(str(sym_dict["symbol_id"])),
                        file_id=delta_id,
                        domain_id=str(r[1]),
                        qualified_name=str(sym_dict["qualified_name"]),
                        symbol_kind=str(sym_dict.get("symbol_kind", "function")),
                        start_line=int(sym_dict.get("start_line", 1)),
                        end_line=int(sym_dict.get("end_line", 1)),
                        signature_skeleton=str(sym_dict.get("signature_skeleton", "")),
                        contextual_summary=str(sym_dict.get("contextual_summary", "")),
                        allowed_scopes=scopes,
                        embedding=[],
                        repo_qualified_id=sym_dict.get("repo_qualified_id"),
                        bare_name=sym_dict.get("bare_name"),
                        tenant_id=tenant_id,
                    )
                )

        base_rows = conn.execute(
            """
            SELECT s.symbol_id, s.file_id, s.domain_id, s.qualified_name, s.symbol_kind,
                   s.start_line, s.end_line, s.signature_skeleton, s.contextual_summary,
                   s.allowed_scopes, s.repo_qualified_id, s.bare_name, s.tenant_id
            FROM ast_symbols s
            JOIN code_files f ON f.file_id = s.file_id
            WHERE s.domain_id = %s
              AND f.repo_name = %s
              AND NOT (f.file_path = ANY(%s::text[]))
            ORDER BY s.qualified_name ASC;
            """,
            (domain_id, snap.repo_name, list(suppressed_paths)),
        ).fetchall()

        base_symbols = [
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
                repo_qualified_id=str(r[10]) if r[10] is not None else None,
                bare_name=str(r[11]) if r[11] is not None else None,
                tenant_id=UUID(str(r[12])),
            )
            for r in base_rows
        ]
        combined = base_symbols + overlay_symbols
        combined.sort(key=lambda s: s.qualified_name)
        return combined

    def list_ownership_edges(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        *,
        edge_kinds: Sequence[str] = ("OWNED_BY_TEAM", "PART_OF_SYSTEM"),
    ) -> list[OwnershipEdgeRecord]:
        rows = conn.execute(
            """
            SELECT e.edge_id, e.edge_kind, s_src.qualified_name, s_src.symbol_kind,
                   e.target_qualified_name, COALESCE(s_tgt.symbol_kind, '')
            FROM ast_edges e
            JOIN ast_symbols s_src ON s_src.symbol_id = e.source_symbol_id
            LEFT JOIN ast_symbols s_tgt ON s_tgt.symbol_id = e.target_symbol_id
            WHERE e.domain_id = %s AND e.edge_kind = ANY(%s::text[])
            ORDER BY s_src.qualified_name ASC, e.edge_kind ASC, e.target_qualified_name ASC;
            """,
            (domain_id, list(edge_kinds)),
        ).fetchall()
        return [
            OwnershipEdgeRecord(
                edge_id=UUID(str(r[0])),
                edge_kind=str(r[1]),
                source_qualified_name=str(r[2]),
                source_symbol_kind=str(r[3]),
                target_qualified_name=str(r[4]),
                target_symbol_kind=str(r[5]),
            )
            for r in rows
        ]

    @staticmethod
    def _row_to_side_effect(row: tuple[object, ...]) -> SideEffectRecord:
        meta_raw = row[16] if isinstance(row[16], dict) else json.loads(str(row[16] or "{}"))
        return SideEffectRecord(
            side_effect_id=UUID(str(row[0])),
            tenant_id=UUID(str(row[1])),
            domain_id=str(row[2]),
            repo_name=str(row[3]),
            file_id=UUID(str(row[4])) if row[4] is not None else None,
            symbol_id=UUID(str(row[5])) if row[5] is not None else None,
            file_path=str(row[6]),
            qualified_symbol=str(row[7]),
            start_line=int(str(row[8])),
            end_line=int(str(row[9])),
            effect_category=cast(EffectCategory, str(row[10])),
            target_identifier=str(row[11]),
            provider_or_driver=str(row[12] or ""),
            trigger_condition=str(row[13] or ""),
            template_or_schema_ref=str(row[14]) if row[14] is not None else None,
            severity_tier=cast(SideEffectSeverity, str(row[15] or "external_io")),
            metadata_json=dict(meta_raw),
            confidence=float(str(row[17])),
            allowed_scopes=[str(s) for s in row[18]] if isinstance(row[18], list) else [],
            indexed_at=_as_opt_dt(row[19]),
        )

    def upsert_side_effects_batch(
        self,
        conn: Connection[tuple[object, ...]],
        records: Sequence[SideEffectRecord],
    ) -> int:
        if not records:
            return 0
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO side_effects (
                    side_effect_id, tenant_id, domain_id, repo_name, file_id, symbol_id,
                    file_path, qualified_symbol, start_line, end_line,
                    effect_category, target_identifier, provider_or_driver,
                    trigger_condition, template_or_schema_ref, severity_tier,
                    confidence, metadata_json, allowed_scopes, indexed_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, COALESCE(%s, NOW())
                )
                ON CONFLICT (
                    tenant_id, domain_id, repo_name, file_path,
                    qualified_symbol, effect_category, target_identifier, start_line
                )
                DO UPDATE SET
                    file_id = EXCLUDED.file_id,
                    symbol_id = EXCLUDED.symbol_id,
                    end_line = EXCLUDED.end_line,
                    provider_or_driver = EXCLUDED.provider_or_driver,
                    trigger_condition = EXCLUDED.trigger_condition,
                    template_or_schema_ref = EXCLUDED.template_or_schema_ref,
                    severity_tier = EXCLUDED.severity_tier,
                    confidence = EXCLUDED.confidence,
                    metadata_json = EXCLUDED.metadata_json,
                    allowed_scopes = EXCLUDED.allowed_scopes,
                    indexed_at = NOW();
                """,
                [
                    (
                        r.side_effect_id,
                        r.tenant_id,
                        r.domain_id,
                        r.repo_name,
                        r.file_id,
                        r.symbol_id,
                        r.file_path,
                        r.qualified_symbol,
                        r.start_line,
                        r.end_line,
                        r.effect_category,
                        r.target_identifier,
                        r.provider_or_driver,
                        r.trigger_condition,
                        r.template_or_schema_ref,
                        r.severity_tier,
                        r.confidence,
                        json.dumps(r.metadata_json),
                        r.allowed_scopes,
                        r.indexed_at,
                    )
                    for r in records
                ],
            )
        return len(records)

    def list_visible_side_effects(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        *,
        effect_category: str | None = None,
        repo_name: str | None = None,
        file_path: str | None = None,
    ) -> list[SideEffectRecord]:
        rows = conn.execute(
            """
            SELECT side_effect_id, tenant_id, domain_id, repo_name, file_id, symbol_id,
                   file_path, qualified_symbol, start_line, end_line,
                   effect_category, target_identifier, provider_or_driver,
                   trigger_condition, template_or_schema_ref, severity_tier,
                   metadata_json, confidence, allowed_scopes, indexed_at
            FROM side_effects
            WHERE domain_id = %s
              AND (%s::text IS NULL OR effect_category = %s)
              AND (%s::text IS NULL OR repo_name = %s)
              AND (%s::text IS NULL OR file_path = %s)
            ORDER BY repo_name ASC, file_path ASC, start_line ASC, effect_category ASC;
            """,
            (
                domain_id,
                effect_category,
                effect_category,
                repo_name,
                repo_name,
                file_path,
                file_path,
            ),
        ).fetchall()
        return [self._row_to_side_effect(row) for row in rows]

    def delete_side_effects_for_file(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        repo_name: str,
        file_path: str,
    ) -> int:
        cur = conn.execute(
            "DELETE FROM side_effects WHERE domain_id = %s AND repo_name = %s AND file_path = %s;",
            (domain_id, repo_name, file_path),
        )
        return max(cur.rowcount or 0, 0)

    def delete_side_effects_for_repo(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        repo_name: str,
    ) -> int:
        cur = conn.execute(
            "DELETE FROM side_effects WHERE domain_id = %s AND repo_name = %s;",
            (domain_id, repo_name),
        )
        return max(cur.rowcount or 0, 0)
