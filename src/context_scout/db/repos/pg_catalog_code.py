"""PostgreSQL RLS repository mixin: domain packs, repositories, code files, and AST records."""

from __future__ import annotations

import json
import logging
from uuid import UUID

from psycopg import Connection

from context_scout.db.repos.types_and_helpers import (
    _as_dt,
    _as_opt_dt,
    _format_db_vector_and_halfvec,
)
from context_scout.db.schema import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    DomainSyncManifestRecord,
    RepoCommitRecord,
    RepositoryRecord,
)
from context_scout.schemas.contracts import (
    DomainPackManifest,
)

logger = logging.getLogger(__name__)


class _PgCatalogAndCodeMixin:
    def upsert_domain_pack(
        self,
        conn: Connection[tuple[object, ...]],
        manifest: DomainPackManifest,
        *,
        tenant_id: UUID | None = None,
        s3_bundle_uri: str | None = None,
        bundle_sha256: str | None = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO domain_packs (
                tenant_id, domain_id, display_name, lora_adapter_id,
                primary_truth_source, freshness_half_life_days, config_json,
                s3_bundle_uri, bundle_sha256, updated_at
            )
            VALUES (
                COALESCE(
                    %s,
                    NULLIF(current_setting('app.tenant_id', true), '')::uuid,
                    '00000000-0000-0000-0000-000000000001'::uuid
                ),
                %s, %s, %s, %s, %s, %s::jsonb, %s, %s, NOW()
            )
            ON CONFLICT (tenant_id, domain_id) DO UPDATE SET
                display_name = EXCLUDED.display_name,
                lora_adapter_id = EXCLUDED.lora_adapter_id,
                primary_truth_source = EXCLUDED.primary_truth_source,
                freshness_half_life_days = EXCLUDED.freshness_half_life_days,
                config_json = EXCLUDED.config_json,
                s3_bundle_uri = COALESCE(EXCLUDED.s3_bundle_uri, domain_packs.s3_bundle_uri),
                bundle_sha256 = COALESCE(EXCLUDED.bundle_sha256, domain_packs.bundle_sha256),
                updated_at = NOW()
            """,
            (
                tenant_id,
                manifest.domain_id,
                manifest.display_name,
                manifest.lora_adapter_id,
                manifest.authority_rules.primary_truth_source,
                manifest.authority_rules.freshness_half_life_days,
                manifest.model_dump_json(),
                s3_bundle_uri,
                bundle_sha256,
            ),
        )

    def get_domain_pack(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
    ) -> DomainPackManifest | None:
        row = conn.execute(
            """
            SELECT config_json
            FROM domain_packs
            WHERE domain_id = %s
            LIMIT 1;
            """,
            (domain_id,),
        ).fetchone()
        if not row or row[0] is None:
            return None
        raw = row[0] if isinstance(row[0], dict) else json.loads(str(row[0]))
        if not raw or not isinstance(raw, dict) or "base_model_id" not in raw:
            return None
        return DomainPackManifest.model_validate(raw)

    def list_domain_packs(
        self,
        conn: Connection[tuple[object, ...]],
        *,
        include_persistent_flag: bool = False,
    ) -> list[DomainPackManifest]:
        rows = conn.execute(
            """
            SELECT config_json, s3_bundle_uri
            FROM domain_packs
            ORDER BY domain_id;
            """
        ).fetchall()
        manifests: list[DomainPackManifest] = []
        for row in rows:
            raw = row[0] if isinstance(row[0], dict) else json.loads(str(row[0]))
            if isinstance(raw, dict) and "base_model_id" in raw:
                if include_persistent_flag and row[1] is None:
                    continue
                manifests.append(DomainPackManifest.model_validate(raw))
        return manifests

    def list_domain_pack_entries(
        self,
        conn: Connection[tuple[object, ...]],
    ) -> list[tuple[DomainPackManifest, bool]]:
        """Returns (manifest, has_persistent_s3_bundle) tuples visible under RLS."""
        rows = conn.execute(
            """
            SELECT config_json, (s3_bundle_uri IS NOT NULL) AS is_persistent
            FROM domain_packs
            ORDER BY domain_id;
            """
        ).fetchall()
        entries: list[tuple[DomainPackManifest, bool]] = []
        for row in rows:
            raw = row[0] if isinstance(row[0], dict) else json.loads(str(row[0]))
            if isinstance(raw, dict) and "base_model_id" in raw:
                entries.append((DomainPackManifest.model_validate(raw), bool(row[1])))
        return entries

    def delete_domain_pack(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
    ) -> int:
        cur = conn.execute("DELETE FROM domain_packs WHERE domain_id = %s;", (domain_id,))
        return max(cur.rowcount or 0, 0)

    def upsert_domain_sync_manifest(
        self,
        conn: Connection[tuple[object, ...]],
        record: DomainSyncManifestRecord,
    ) -> None:
        conn.execute(
            """
            INSERT INTO domain_sync_manifests (
                tenant_id, domain_id, kind, source_dir,
                repo_commit_shas, empty_doc_shas,
                code_files_indexed, ast_symbols_indexed,
                ast_edges_indexed, doc_chunks_indexed,
                s3_manifest_uri, manifest_sha256, synced_at
            )
            VALUES (
                %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                %s, %s, %s, %s, %s, %s, COALESCE(%s, NOW())
            )
            ON CONFLICT (tenant_id, domain_id) DO UPDATE SET
                kind = EXCLUDED.kind,
                source_dir = EXCLUDED.source_dir,
                repo_commit_shas = EXCLUDED.repo_commit_shas,
                empty_doc_shas = EXCLUDED.empty_doc_shas,
                code_files_indexed = EXCLUDED.code_files_indexed,
                ast_symbols_indexed = EXCLUDED.ast_symbols_indexed,
                ast_edges_indexed = EXCLUDED.ast_edges_indexed,
                doc_chunks_indexed = EXCLUDED.doc_chunks_indexed,
                s3_manifest_uri = COALESCE(
                    EXCLUDED.s3_manifest_uri, domain_sync_manifests.s3_manifest_uri
                ),
                manifest_sha256 = COALESCE(
                    EXCLUDED.manifest_sha256, domain_sync_manifests.manifest_sha256
                ),
                synced_at = NOW();
            """,
            (
                record.tenant_id,
                record.domain_id,
                record.kind,
                record.source_dir,
                json.dumps(record.repo_commit_shas),
                json.dumps(record.empty_doc_shas),
                record.code_files_indexed,
                record.ast_symbols_indexed,
                record.ast_edges_indexed,
                record.doc_chunks_indexed,
                record.s3_manifest_uri,
                record.manifest_sha256,
                record.synced_at,
            ),
        )

    def get_domain_sync_manifest(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
    ) -> DomainSyncManifestRecord | None:
        row = conn.execute(
            """
            SELECT tenant_id, domain_id, kind, source_dir,
                   repo_commit_shas, empty_doc_shas,
                   code_files_indexed, ast_symbols_indexed,
                   ast_edges_indexed, doc_chunks_indexed,
                   s3_manifest_uri, manifest_sha256, synced_at
            FROM domain_sync_manifests
            WHERE domain_id = %s
            LIMIT 1;
            """,
            (domain_id,),
        ).fetchone()
        if row is None:
            return None
        raw_commits = row[4] if isinstance(row[4], dict) else json.loads(str(row[4] or "{}"))
        raw_empty = row[5] if isinstance(row[5], dict) else json.loads(str(row[5] or "{}"))
        return DomainSyncManifestRecord(
            tenant_id=UUID(str(row[0])),
            domain_id=str(row[1]),
            kind=str(row[2]),
            source_dir=str(row[3] or ""),
            repo_commit_shas={str(k): str(v) for k, v in raw_commits.items()},
            empty_doc_shas={str(k): str(v) for k, v in raw_empty.items()},
            code_files_indexed=int(str(row[6] or 0)),
            ast_symbols_indexed=int(str(row[7] or 0)),
            ast_edges_indexed=int(str(row[8] or 0)),
            doc_chunks_indexed=int(str(row[9] or 0)),
            s3_manifest_uri=str(row[10]) if row[10] is not None else None,
            manifest_sha256=str(row[11]) if row[11] is not None else None,
            synced_at=_as_opt_dt(row[12]),
        )

    def insert_code_file(
        self, conn: Connection[tuple[object, ...]], record: CodeFileRecord
    ) -> None:
        conn.execute(
            """
            INSERT INTO code_files (
                file_id, tenant_id, domain_id, repo_name, file_path,
                git_commit_sha, git_blob_sha, language,
                allowed_scopes, raw_content, line_count,
                ranking_weight, is_generated, is_vendored
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                record.file_id,
                record.tenant_id,
                record.domain_id,
                record.repo_name,
                record.file_path,
                record.git_commit_sha,
                record.git_blob_sha,
                record.language,
                record.allowed_scopes,
                record.raw_content,
                record.line_count,
                record.ranking_weight,
                record.is_generated,
                record.is_vendored,
            ),
        )

    def insert_ast_symbol(
        self, conn: Connection[tuple[object, ...]], record: ASTSymbolRecord
    ) -> None:
        vec_literal, halfvec_literal = _format_db_vector_and_halfvec(record.embedding)
        conn.execute(
            """
            INSERT INTO ast_symbols (
                symbol_id, tenant_id, file_id, domain_id, qualified_name,
                symbol_kind, start_line, end_line,
                signature_skeleton, contextual_summary,
                allowed_scopes, embedding, embedding_space_id, embedding_halfvec,
                repo_qualified_id, bare_name
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s::halfvec, %s, %s)
            """,
            (
                record.symbol_id,
                record.tenant_id,
                record.file_id,
                record.domain_id,
                record.qualified_name,
                record.symbol_kind,
                record.start_line,
                record.end_line,
                record.signature_skeleton,
                record.contextual_summary,
                record.allowed_scopes,
                vec_literal,
                record.embedding_space_id,
                halfvec_literal,
                record.repo_qualified_id,
                record.bare_name,
            ),
        )

    def insert_ast_edge(self, conn: Connection[tuple[object, ...]], record: ASTEdgeRecord) -> None:
        conn.execute(
            """
            INSERT INTO ast_edges (
                edge_id, tenant_id, domain_id, source_symbol_id,
                target_qualified_name, target_symbol_id,
                edge_kind, allowed_scopes, target_repo_qualified_id, is_cross_repo
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                record.edge_id,
                record.tenant_id,
                record.domain_id,
                record.source_symbol_id,
                record.target_qualified_name,
                record.target_symbol_id,
                record.edge_kind,
                record.allowed_scopes,
                record.target_repo_qualified_id,
                record.is_cross_repo,
            ),
        )

    def list_visible_code_files(
        self, conn: Connection[tuple[object, ...]], domain_id: str
    ) -> list[str]:
        rows = conn.execute(
            "SELECT file_path FROM code_files WHERE domain_id = %s ORDER BY file_path",
            (domain_id,),
        ).fetchall()
        return [str(r[0]) for r in rows]

    def list_visible_ast_symbols(
        self, conn: Connection[tuple[object, ...]], domain_id: str
    ) -> list[str]:
        rows = conn.execute(
            "SELECT qualified_name FROM ast_symbols WHERE domain_id = %s ORDER BY qualified_name",
            (domain_id,),
        ).fetchall()
        return [str(r[0]) for r in rows]

    def list_visible_ast_edges(
        self, conn: Connection[tuple[object, ...]], domain_id: str
    ) -> list[str]:
        rows = conn.execute(
            "SELECT target_qualified_name FROM ast_edges "
            "WHERE domain_id = %s ORDER BY target_qualified_name",
            (domain_id,),
        ).fetchall()
        return [str(r[0]) for r in rows]

    def get_file_blob_hashes(
        self, conn: Connection[tuple[object, ...]], domain_id: str, repo_name: str
    ) -> dict[str, tuple[UUID, str]]:
        rows = conn.execute(
            """
            SELECT file_path, file_id, git_blob_sha
            FROM code_files
            WHERE domain_id = %s AND repo_name = %s
            ORDER BY file_path;
            """,
            (domain_id, repo_name),
        ).fetchall()
        return {
            str(r[0]): (r[1] if isinstance(r[1], UUID) else UUID(str(r[1])), str(r[2]))
            for r in rows
        }

    def delete_code_file(self, conn: Connection[tuple[object, ...]], file_id: UUID) -> None:
        conn.execute("DELETE FROM code_files WHERE file_id = %s;", (file_id,))

    def delete_code_file_and_symbols(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        repo_name: str,
        file_path: str,
    ) -> None:
        """Deletes a code file by domain, repo, and file_path, cascading symbol deletions."""
        conn.execute(
            """
            DELETE FROM code_files
            WHERE domain_id = %s AND (repo_name = %s OR repo_name = %s) AND file_path = %s;
            """,
            (domain_id, repo_name, repo_name.replace("/", "_"), file_path),
        )

    def insert_repository(
        self, conn: Connection[tuple[object, ...]], record: RepositoryRecord
    ) -> None:
        conn.execute(
            """
            INSERT INTO repositories (
                repo_id, tenant_id, domain_id, repo_name, source_type, remote_url,
                default_branch, head_commit_sha, allowed_scopes, metadata_json,
                is_active, created_at, updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s,
                COALESCE(%s, NOW()), COALESCE(%s, NOW())
            )
            ON CONFLICT (tenant_id, domain_id, repo_name) DO UPDATE SET
                source_type = EXCLUDED.source_type,
                remote_url = EXCLUDED.remote_url,
                default_branch = EXCLUDED.default_branch,
                head_commit_sha = EXCLUDED.head_commit_sha,
                allowed_scopes = EXCLUDED.allowed_scopes,
                metadata_json = EXCLUDED.metadata_json,
                is_active = EXCLUDED.is_active,
                updated_at = NOW();
            """,
            (
                record.repo_id,
                record.tenant_id,
                record.domain_id,
                record.repo_name,
                record.source_type,
                record.remote_url,
                record.default_branch,
                record.head_commit_sha,
                record.allowed_scopes,
                json.dumps(record.metadata_json),
                record.is_active,
                record.created_at,
                record.updated_at,
            ),
        )

    def upsert_repository(
        self, conn: Connection[tuple[object, ...]], record: RepositoryRecord
    ) -> None:
        self.insert_repository(conn, record)

    def get_repository(
        self, conn: Connection[tuple[object, ...]], domain_id: str, repo_name: str
    ) -> RepositoryRecord | None:
        row = conn.execute(
            """
            SELECT repo_id, domain_id, repo_name, source_type, remote_url,
                   default_branch, head_commit_sha, allowed_scopes, metadata_json,
                   is_active, created_at, updated_at
            FROM repositories
            WHERE domain_id = %s AND repo_name = %s;
            """,
            (domain_id, repo_name),
        ).fetchone()
        if not row:
            return None
        return RepositoryRecord(
            repo_id=UUID(str(row[0])),
            domain_id=str(row[1]),
            repo_name=str(row[2]),
            source_type=row[3],  # type: ignore[arg-type]
            remote_url=str(row[4]),
            default_branch=str(row[5]),
            head_commit_sha=str(row[6]),
            allowed_scopes=[str(s) for s in row[7]] if isinstance(row[7], list) else [],
            metadata_json=row[8] if isinstance(row[8], dict) else json.loads(str(row[8])),
            is_active=bool(row[9]),
            created_at=_as_opt_dt(row[10]),
            updated_at=_as_opt_dt(row[11]),
        )

    def list_repositories(
        self, conn: Connection[tuple[object, ...]], domain_id: str
    ) -> list[RepositoryRecord]:
        rows = conn.execute(
            """
            SELECT repo_id, domain_id, repo_name, source_type, remote_url,
                   default_branch, head_commit_sha, allowed_scopes, metadata_json,
                   is_active, created_at, updated_at
            FROM repositories
            WHERE domain_id = %s
            ORDER BY repo_name;
            """,
            (domain_id,),
        ).fetchall()
        return [
            RepositoryRecord(
                repo_id=UUID(str(row[0])),
                domain_id=str(row[1]),
                repo_name=str(row[2]),
                source_type=row[3],  # type: ignore[arg-type]
                remote_url=str(row[4]),
                default_branch=str(row[5]),
                head_commit_sha=str(row[6]),
                allowed_scopes=[str(s) for s in row[7]] if isinstance(row[7], list) else [],
                metadata_json=row[8] if isinstance(row[8], dict) else json.loads(str(row[8])),
                is_active=bool(row[9]),
                created_at=_as_opt_dt(row[10]),
                updated_at=_as_opt_dt(row[11]),
            )
            for row in rows
        ]

    def insert_repo_commit(
        self, conn: Connection[tuple[object, ...]], record: RepoCommitRecord
    ) -> None:
        conn.execute(
            """
            INSERT INTO repo_commits (
                commit_id, tenant_id, repo_id, commit_sha, parent_shas,
                author_name, author_email, commit_message,
                committed_at, indexed_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, NOW()))
            ON CONFLICT (repo_id, commit_sha) DO NOTHING;
            """,
            (
                record.commit_id,
                record.tenant_id,
                record.repo_id,
                record.commit_sha,
                record.parent_shas,
                record.author_name,
                record.author_email,
                record.commit_message,
                record.committed_at,
                record.indexed_at,
            ),
        )

    def get_repo_commits(
        self, conn: Connection[tuple[object, ...]], repo_id: UUID
    ) -> list[RepoCommitRecord]:
        rows = conn.execute(
            """
            SELECT commit_id, repo_id, commit_sha, parent_shas,
                   author_name, author_email, commit_message,
                   committed_at, indexed_at
            FROM repo_commits
            WHERE repo_id = %s
            ORDER BY committed_at DESC;
            """,
            (repo_id,),
        ).fetchall()
        return [
            RepoCommitRecord(
                commit_id=UUID(str(row[0])),
                repo_id=UUID(str(row[1])),
                commit_sha=str(row[2]),
                parent_shas=[str(s) for s in row[3]] if isinstance(row[3], list) else [],
                author_name=str(row[4]),
                author_email=str(row[5]),
                commit_message=str(row[6]),
                committed_at=_as_dt(row[7]),
                indexed_at=_as_opt_dt(row[8]),
            )
            for row in rows
        ]

    def get_code_files_by_ids(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        file_ids: list[UUID | str],
    ) -> list[CodeFileRecord]:
        """Fetches visible code files by UUID under RLS without full-table scan."""
        clean_ids: list[str] = []
        for fid in file_ids:
            try:
                clean_ids.append(str(UUID(str(fid))))
            except ValueError as exc:
                logger.debug(
                    "Ignoring malformed file_id %r (%s): %s",
                    fid,
                    type(exc).__name__,
                    exc,
                )
                continue
        if not clean_ids:
            return []
        rows = conn.execute(
            """
            SELECT file_id, domain_id, repo_name, file_path, git_commit_sha,
                   git_blob_sha, language, allowed_scopes, raw_content, line_count,
                   tenant_id, COALESCE(ranking_weight, 1.0),
                   COALESCE(is_generated, FALSE), COALESCE(is_vendored, FALSE)
            FROM code_files
            WHERE domain_id = %s AND file_id = ANY(%s::uuid[]);
            """,
            (domain_id, clean_ids),
        ).fetchall()
        return [
            CodeFileRecord(
                file_id=UUID(str(r[0])),
                domain_id=str(r[1]),
                repo_name=str(r[2]),
                file_path=str(r[3]),
                git_commit_sha=str(r[4]),
                git_blob_sha=str(r[5]),
                language=str(r[6]),
                allowed_scopes=[str(s) for s in r[7]] if isinstance(r[7], list) else [],
                raw_content=str(r[8]),
                line_count=int(str(r[9])),
                tenant_id=UUID(str(r[10])),
                ranking_weight=float(str(r[11])),
                is_generated=bool(r[12]),
                is_vendored=bool(r[13]),
            )
            for r in rows
        ]

    def get_code_files_by_paths(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        file_paths: list[str],
    ) -> list[CodeFileRecord]:
        """Fetches visible code files by file_path under RLS without full-table scan."""
        clean_paths = [p.strip() for p in file_paths if p and p.strip()]
        if not clean_paths:
            return []
        rows = conn.execute(
            """
            SELECT file_id, domain_id, repo_name, file_path, git_commit_sha,
                   git_blob_sha, language, allowed_scopes, raw_content, line_count,
                   tenant_id, COALESCE(ranking_weight, 1.0),
                   COALESCE(is_generated, FALSE), COALESCE(is_vendored, FALSE)
            FROM code_files
            WHERE domain_id = %s AND file_path = ANY(%s::text[]);
            """,
            (domain_id, clean_paths),
        ).fetchall()
        return [
            CodeFileRecord(
                file_id=UUID(str(r[0])),
                domain_id=str(r[1]),
                repo_name=str(r[2]),
                file_path=str(r[3]),
                git_commit_sha=str(r[4]),
                git_blob_sha=str(r[5]),
                language=str(r[6]),
                allowed_scopes=[str(s) for s in r[7]] if isinstance(r[7], list) else [],
                raw_content=str(r[8]),
                line_count=int(str(r[9])),
                tenant_id=UUID(str(r[10])),
                ranking_weight=float(str(r[11])),
                is_generated=bool(r[12]),
                is_vendored=bool(r[13]),
            )
            for r in rows
        ]

    def list_visible_code_file_records(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        *,
        repo_name: str | None = None,
        limit: int = 2000,
    ) -> list[CodeFileRecord]:
        """Lists full visible CodeFileRecord rows under Forced RLS."""
        safe_limit = max(1, min(limit, 10000))
        if repo_name:
            rows = conn.execute(
                """
                SELECT file_id, domain_id, repo_name, file_path, git_commit_sha,
                       git_blob_sha, language, allowed_scopes, raw_content, line_count,
                       tenant_id, COALESCE(ranking_weight, 1.0),
                       COALESCE(is_generated, FALSE), COALESCE(is_vendored, FALSE)
                FROM code_files
                WHERE domain_id = %s AND repo_name = %s
                ORDER BY file_path ASC
                LIMIT %s;
                """,
                (domain_id, repo_name, safe_limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT file_id, domain_id, repo_name, file_path, git_commit_sha,
                       git_blob_sha, language, allowed_scopes, raw_content, line_count,
                       tenant_id, COALESCE(ranking_weight, 1.0),
                       COALESCE(is_generated, FALSE), COALESCE(is_vendored, FALSE)
                FROM code_files
                WHERE domain_id = %s
                ORDER BY repo_name ASC, file_path ASC
                LIMIT %s;
                """,
                (domain_id, safe_limit),
            ).fetchall()
        return [
            CodeFileRecord(
                file_id=UUID(str(r[0])),
                domain_id=str(r[1]),
                repo_name=str(r[2]),
                file_path=str(r[3]),
                git_commit_sha=str(r[4]),
                git_blob_sha=str(r[5]),
                language=str(r[6]),
                allowed_scopes=[str(s) for s in r[7]] if isinstance(r[7], list) else [],
                raw_content=str(r[8]),
                line_count=int(str(r[9])),
                tenant_id=UUID(str(r[10])),
                ranking_weight=float(str(r[11])),
                is_generated=bool(r[12]),
                is_vendored=bool(r[13]),
            )
            for r in rows
        ]

    def list_visible_ast_symbol_records(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        *,
        limit: int = 5000,
    ) -> list[ASTSymbolRecord]:
        """Lists visible ASTSymbolRecord rows under Forced RLS."""
        safe_limit = max(1, min(limit, 20000))
        rows = conn.execute(
            """
            SELECT symbol_id, file_id, domain_id, qualified_name, symbol_kind,
                   start_line, end_line, signature_skeleton, contextual_summary,
                   allowed_scopes, tenant_id
            FROM ast_symbols
            WHERE domain_id = %s
            ORDER BY qualified_name ASC
            LIMIT %s;
            """,
            (domain_id, safe_limit),
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
                tenant_id=UUID(str(r[10])),
            )
            for r in rows
        ]

    def list_visible_ast_edge_records(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        *,
        limit: int = 10000,
    ) -> list[ASTEdgeRecord]:
        """Lists visible ASTEdgeRecord rows under Forced RLS."""
        safe_limit = max(1, min(limit, 50000))
        rows = conn.execute(
            """
            SELECT edge_id, domain_id, source_symbol_id, target_qualified_name,
                   target_symbol_id, edge_kind, allowed_scopes,
                   target_repo_qualified_id, COALESCE(is_cross_repo, FALSE), tenant_id
            FROM ast_edges
            WHERE domain_id = %s
            ORDER BY edge_kind ASC, target_qualified_name ASC
            LIMIT %s;
            """,
            (domain_id, safe_limit),
        ).fetchall()
        return [
            ASTEdgeRecord(
                edge_id=UUID(str(r[0])),
                domain_id=str(r[1]),
                source_symbol_id=UUID(str(r[2])),
                target_qualified_name=str(r[3]),
                target_symbol_id=UUID(str(r[4])) if r[4] is not None else None,
                edge_kind=str(r[5]),
                allowed_scopes=[str(s) for s in r[6]] if isinstance(r[6], list) else [],
                target_repo_qualified_id=str(r[7]) if r[7] is not None else None,
                is_cross_repo=bool(r[8]),
                tenant_id=UUID(str(r[9])),
            )
            for r in rows
        ]
