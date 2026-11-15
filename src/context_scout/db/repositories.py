"""RLS-enforced repositories and 3-way Normalized RRF search (M1 Foundation)."""

import json
import logging
import math
import re
from collections import Counter
from datetime import date, datetime
from typing import Any
from uuid import UUID

from psycopg import Connection

from context_scout.db.bm25 import BM25Okapi
from context_scout.db.schema import (
    ASTEdgeRecord as ASTEdgeRecord,
)
from context_scout.db.schema import (
    ASTNeighborhoodRecord as ASTNeighborhoodRecord,
)
from context_scout.db.schema import (
    ASTSymbolRecord as ASTSymbolRecord,
)
from context_scout.db.schema import (
    CodeFileRecord as CodeFileRecord,
)
from context_scout.db.schema import (
    ConfluencePageRecord as ConfluencePageRecord,
)
from context_scout.db.schema import (
    DocChunkRecord as DocChunkRecord,
)
from context_scout.db.schema import (
    JiraIssueRecord as JiraIssueRecord,
)
from context_scout.db.schema import (
    RankedSearchHit as RankedSearchHit,
)
from context_scout.db.schema import (
    RepoCommitRecord as RepoCommitRecord,
)
from context_scout.db.schema import (
    RepositoryRecord as RepositoryRecord,
)
from context_scout.schemas.contracts import (
    DomainPackManifest,
    ExecutionTelemetryRecord,
    MarketRegime,
    PrincipalContext,
    compute_normalized_rrf,
)

logger = logging.getLogger(__name__)


def _as_dt(val: object) -> datetime:
    if isinstance(val, datetime):
        return val
    return datetime.fromisoformat(str(val))


def _as_opt_dt(val: object) -> datetime | None:
    if val is None:
        return None
    return _as_dt(val)


def _serialize_market_scope(market_scope: MarketRegime | str | None) -> str | None:
    """Serializes MarketRegime | str | None to SQL NULL or text without 'None' strings."""
    if market_scope is None:
        return None
    if isinstance(market_scope, MarketRegime):
        return market_scope.value
    cleaned = str(market_scope).strip()
    if not cleaned or cleaned == "None":
        return None
    return cleaned


def _parse_market_scope(raw_scope: object) -> MarketRegime | str | None:
    """Parses DB market_scope into MarketRegime, preserved str label, or None."""
    if raw_scope is None:
        return None
    val = str(raw_scope).strip()
    if not val or val == "None":
        return None
    try:
        return MarketRegime(val)
    except ValueError:
        return val


def _apply_authority_score(base_rrf: float, authority_weight: float) -> float:
    """Factors authority_weight into RRF score while preserving exact 6-decimal equality at 1.0."""
    if math.isclose(authority_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
        return base_rrf
    return round(min(1.0, max(0.0, base_rrf * authority_weight)), 6)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b) or not vec_a:
        raise ValueError("Vectors must have matching non-zero dimensions")
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _scopes_intersect(row_scopes: list[str], principal_scopes: list[str]) -> bool:
    return bool(set(row_scopes).intersection(principal_scopes))


def _trigram_similarity(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    a = f"  {text_a.lower()} "
    b = f"  {text_b.lower()} "
    if len(a) < 3 or len(b) < 3:
        return 1.0 if a.strip() == b.strip() else 0.0
    trigrams_a = Counter(a[i : i + 3] for i in range(len(a) - 2))
    trigrams_b = Counter(b[i : i + 3] for i in range(len(b) - 2))
    intersection = sum((trigrams_a & trigrams_b).values())
    total = sum(trigrams_a.values()) + sum(trigrams_b.values())
    if total == 0:
        return 0.0
    return (2.0 * intersection) / total


class PostgresRLSRepository:
    """Executes parameterized SQL inside an RLS-enforced PostgreSQL 17 transaction."""

    def upsert_domain_pack(
        self, conn: Connection[tuple[object, ...]], manifest: DomainPackManifest
    ) -> None:
        conn.execute(
            """
            INSERT INTO domain_packs (
                domain_id, display_name, lora_adapter_id,
                primary_truth_source, freshness_half_life_days, config_json
            )
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (domain_id) DO UPDATE SET
                display_name = EXCLUDED.display_name,
                lora_adapter_id = EXCLUDED.lora_adapter_id,
                primary_truth_source = EXCLUDED.primary_truth_source,
                freshness_half_life_days = EXCLUDED.freshness_half_life_days,
                config_json = EXCLUDED.config_json
            """,
            (
                manifest.domain_id,
                manifest.display_name,
                manifest.lora_adapter_id,
                manifest.authority_rules.primary_truth_source,
                manifest.authority_rules.freshness_half_life_days,
                manifest.model_dump_json(),
            ),
        )

    def insert_code_file(
        self, conn: Connection[tuple[object, ...]], record: CodeFileRecord
    ) -> None:
        conn.execute(
            """
            INSERT INTO code_files (
                file_id, domain_id, repo_name, file_path,
                git_commit_sha, git_blob_sha, language,
                allowed_scopes, raw_content, line_count
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                record.file_id,
                record.domain_id,
                record.repo_name,
                record.file_path,
                record.git_commit_sha,
                record.git_blob_sha,
                record.language,
                record.allowed_scopes,
                record.raw_content,
                record.line_count,
            ),
        )

    def insert_ast_symbol(
        self, conn: Connection[tuple[object, ...]], record: ASTSymbolRecord
    ) -> None:
        vec_literal = "[" + ",".join(f"{v:.6f}" for v in record.embedding) + "]"
        conn.execute(
            """
            INSERT INTO ast_symbols (
                symbol_id, file_id, domain_id, qualified_name,
                symbol_kind, start_line, end_line,
                signature_skeleton, contextual_summary,
                allowed_scopes, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
            """,
            (
                record.symbol_id,
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
            ),
        )

    def insert_ast_edge(self, conn: Connection[tuple[object, ...]], record: ASTEdgeRecord) -> None:
        conn.execute(
            """
            INSERT INTO ast_edges (
                edge_id, domain_id, source_symbol_id,
                target_qualified_name, target_symbol_id,
                edge_kind, allowed_scopes
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                record.edge_id,
                record.domain_id,
                record.source_symbol_id,
                record.target_qualified_name,
                record.target_symbol_id,
                record.edge_kind,
                record.allowed_scopes,
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
        return {str(r[0]): (UUID(str(r[1])), str(r[2])) for r in rows}

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
                repo_id, domain_id, repo_name, source_type, remote_url,
                default_branch, head_commit_sha, allowed_scopes, metadata_json,
                is_active, created_at, updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s,
                COALESCE(%s, NOW()), COALESCE(%s, NOW())
            )
            ON CONFLICT (domain_id, repo_name) DO UPDATE SET
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
                commit_id, repo_id, commit_sha, parent_shas,
                author_name, author_email, commit_message,
                committed_at, indexed_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, NOW()))
            ON CONFLICT (repo_id, commit_sha) DO NOTHING;
            """,
            (
                record.commit_id,
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

    def insert_confluence_page(
        self, conn: Connection[tuple[object, ...]], record: ConfluencePageRecord
    ) -> None:
        conn.execute(
            """
            INSERT INTO confluence_pages (
                page_id, domain_id, confluence_page_id, space_key,
                title, version_number, author_username, source_url,
                allowed_scopes, last_modified_at, indexed_at,
                status, is_deleted
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, NOW()), 'current', FALSE)
            ON CONFLICT (domain_id, confluence_page_id) DO UPDATE SET
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
                issue_id, domain_id, issue_key, project_key,
                summary, issue_type, status, resolution,
                assignee_username, source_url, allowed_scopes,
                updated_at, indexed_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, NOW()))
            ON CONFLICT (domain_id, issue_key) DO UPDATE SET
                project_key = EXCLUDED.project_key,
                summary = EXCLUDED.summary,
                issue_type = EXCLUDED.issue_type,
                status = EXCLUDED.status,
                resolution = EXCLUDED.resolution,
                assignee_username = EXCLUDED.assignee_username,
                source_url = EXCLUDED.source_url,
                allowed_scopes = EXCLUDED.allowed_scopes,
                updated_at = EXCLUDED.updated_at,
                indexed_at = NOW();
            """,
            (
                record.issue_id,
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
            ),
        )

    def get_jira_issue(
        self, conn: Connection[tuple[object, ...]], domain_id: str, issue_key: str
    ) -> JiraIssueRecord | None:
        row = conn.execute(
            """
            SELECT issue_id, domain_id, issue_key, project_key,
                   summary, issue_type, status, resolution,
                   assignee_username, source_url, allowed_scopes,
                   updated_at, indexed_at
            FROM jira_issues
            WHERE domain_id = %s AND issue_key = %s;
            """,
            (domain_id, issue_key),
        ).fetchone()
        if not row:
            return None
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
        )

    def list_jira_issues(
        self, conn: Connection[tuple[object, ...]], domain_id: str, project_key: str | None = None
    ) -> list[JiraIssueRecord]:
        if project_key:
            rows = conn.execute(
                """
                SELECT issue_id, domain_id, issue_key, project_key,
                       summary, issue_type, status, resolution,
                       assignee_username, source_url, allowed_scopes,
                       updated_at, indexed_at
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
                       updated_at, indexed_at
                FROM jira_issues
                WHERE domain_id = %s
                ORDER BY issue_key;
                """,
                (domain_id,),
            ).fetchall()
        return [
            JiraIssueRecord(
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
            )
            for row in rows
        ]

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
        cte_sql = """
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
                  e.edge_kind IN ('USES_TYPE', 'IMPORTS', 'INHERITS')
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
              AND e.edge_kind = 'CALLS'
              AND e.source_symbol_id NOT IN (SELECT symbol_id FROM focus_family)
            UNION
            SELECT e.edge_id, e.edge_kind, e.source_symbol_id
            FROM ast_edges e
            JOIN focus_family ff ON e.target_qualified_name = ff.qualified_name
            WHERE e.domain_id = %s
              AND e.edge_kind = 'CALLS'
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
        vec_literal = "[" + ",".join(f"{v:.6f}" for v in record.embedding) + "]"
        conn.execute(
            """
            INSERT INTO doc_chunks (
                chunk_id, domain_id, doc_id, doc_title, source_uri,
                market_scope, published_date, authority_weight,
                section_heading, contextual_header, verbatim_content,
                metadata_json, allowed_scopes, embedding,
                source_type, version_str, is_stale, stale_reason
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s::jsonb, %s, %s::vector,
                %s, %s, %s, %s
            )
            """,
            (
                record.chunk_id,
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
                chunk_id, domain_id, doc_id, doc_title, source_uri,
                market_scope, published_date, authority_weight,
                section_heading, contextual_header, verbatim_content,
                metadata_json, allowed_scopes, embedding,
                source_type, version_str, is_stale, stale_reason
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s::jsonb, %s, %s::vector,
                %s, %s, %s, %s
            )
        """
        params_seq = [
            (
                record.chunk_id,
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
                "[" + ",".join(f"{v:.6f}" for v in record.embedding) + "]",
                record.source_type,
                record.version_str,
                record.is_stale,
                record.stale_reason.replace("\x00", "")
                if record.stale_reason is not None
                else None,
            )
            for record in records
        ]
        with conn.cursor() as cur:
            cur.executemany(sql, params_seq)
        return len(records)

    def list_visible_doc_chunks(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        market_scope: MarketRegime | str | None = None,
        include_stale: bool = False,
    ) -> list[DocChunkRecord]:
        query = """
            SELECT chunk_id, domain_id, doc_id, doc_title, source_uri,
                   market_scope, published_date, authority_weight,
                   section_heading, contextual_header, verbatim_content,
                   metadata_json, allowed_scopes, embedding,
                   source_type, version_str, is_stale, stale_reason
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
                    source_type=str(r[14]) if r[14] is not None else "quant_paper",
                    version_str=str(r[15]) if r[15] is not None else "1",
                    is_stale=bool(r[16]),
                    stale_reason=str(r[17]) if r[17] is not None else None,
                )
            )
        return results

    def search_doc_chunks_hybrid_rrf(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        query_text: str,
        query_embedding: list[float],
        market_scope: MarketRegime | str | None = None,
        top_k: int = 5,
        include_stale: bool = False,
    ) -> list[RankedSearchHit]:
        vec_literal = "[" + ",".join(f"{v:.6f}" for v in query_embedding) + "]"
        market_str = _serialize_market_scope(market_scope)
        candidate_pool = max(100, top_k * 10)
        penalty_rank = candidate_pool + 1
        filler_words = {
            "what",
            "does",
            "stand",
            "stands",
            "mean",
            "means",
            "compare",
            "list",
            "top",
            "explain",
            "describe",
            "inspect",
            "give",
            "how",
            "when",
            "where",
            "why",
            "which",
            "mitigate",
            "empirical",
            "findings",
            "reasons",
            "failure",
            "modes",
        }
        raw_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", query_text)]
        content_tokens = list(
            dict.fromkeys(t for t in raw_tokens if len(t) >= 2 and t not in filler_words)
        )[:16]
        and_query = " ".join(content_tokens) if content_tokens else query_text.strip()
        or_query = " OR ".join(content_tokens) if content_tokens else query_text.strip()
        trgm_query = and_query or query_text.strip()
        query = """
            WITH domain_size AS (
                SELECT COUNT(*) AS cnt
                FROM doc_chunks
                WHERE domain_id = %s
            ),
            fts_ranked AS (
                SELECT chunk_id,
                       ROW_NUMBER() OVER (
                           ORDER BY fts_score DESC, chunk_id
                       ) AS fts_rank
                FROM (
                    SELECT chunk_id,
                           (
                               ts_rank_cd(fts_doc, plainto_tsquery('english', %s)) * 3.0
                               + ts_rank_cd(fts_doc, websearch_to_tsquery('english', %s))
                           ) * authority_weight AS fts_score
                    FROM doc_chunks
                    WHERE domain_id = %s
                      AND (%s::text IS NULL OR market_scope = %s)
                      AND authority_weight > 0.0
                      AND (%s OR NOT is_stale)
                      AND fts_doc @@ websearch_to_tsquery('english', %s)
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
                       ROW_NUMBER() OVER (
                           ORDER BY vec_dist ASC, chunk_id
                       ) AS vec_rank
                FROM (
                    SELECT chunk_id, (embedding <=> %s::vector) AS vec_dist
                    FROM doc_chunks
                    WHERE domain_id = %s
                      AND (%s::text IS NULL OR market_scope = %s)
                      AND authority_weight > 0.0
                      AND (%s OR NOT is_stale)
                      AND (
                          (SELECT cnt FROM domain_size) <= 50
                          OR (
                              LENGTH(TRIM(verbatim_content)) >= 45
                              AND LEFT(LTRIM(verbatim_content), 5) <> (CHR(37) || ' ---')
                              AND LEFT(LTRIM(verbatim_content), 11) <> '## See also'
                          )
                      )
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s
                ) vec_sub
            ),
            trgm_ranked AS (
                SELECT chunk_id,
                       ROW_NUMBER() OVER (
                           ORDER BY trgm_score DESC, chunk_id
                       ) AS trgm_rank
                FROM (
                    SELECT chunk_id,
                           (
                               word_similarity(
                                   %s,
                                   doc_title || ' ' || section_heading
                                   || ' ' || LEFT(verbatim_content, 1200)
                               ) * authority_weight
                           ) AS trgm_score
                    FROM doc_chunks
                    WHERE domain_id = %s
                      AND (%s::text IS NULL OR market_scope = %s)
                      AND authority_weight > 0.0
                      AND (%s OR NOT is_stale)
                      AND word_similarity(
                              %s,
                              doc_title || ' ' || section_heading
                              || ' ' || LEFT(verbatim_content, 1200)
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
                   d.authority_weight
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
            domain_id,
            market_str,
            market_str,
            include_stale,
        )
        rows = conn.execute(query, params).fetchall()
        scored_hits: list[tuple[float, float, float, RankedSearchHit]] = []
        for r in rows:
            chunk_id = UUID(str(r[0]))
            doc_title = str(r[1])
            section_heading = str(r[2])
            fts_rank = int(str(r[3]))
            vec_rank = int(str(r[4]))
            trgm_rank = int(str(r[5]))
            auth_weight = float(str(r[6]))
            base_rrf = compute_normalized_rrf(
                [trgm_rank, fts_rank, vec_rank], num_modalities=3, k=60
            )
            weighted_score = _apply_authority_score(base_rrf, auth_weight)
            hit = RankedSearchHit(
                item_id=chunk_id,
                qualified_name_or_title=f"{doc_title} > {section_heading}",
                normalized_rrf_score=weighted_score,
                trigram_rank=trgm_rank,
                bm25_rank=fts_rank,
                vector_rank=vec_rank,
            )
            scored_hits.append((weighted_score, base_rrf * auth_weight, auth_weight, hit))
        scored_hits.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [item[3] for item in scored_hits[:top_k]]

    def search_symbols_hybrid_rrf(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RankedSearchHit]:
        """Executes 3-way Hybrid RRF over PostgreSQL ast_symbols (GIN trigram + FTS + HNSW)."""
        vec_literal = "[" + ",".join(f"{v:.6f}" for v in query_embedding) + "]"
        candidate_pool = max(100, top_k * 10)
        penalty_rank = candidate_pool + 1
        query = """
            WITH fts_ranked AS (
                SELECT symbol_id,
                       ROW_NUMBER() OVER (
                           ORDER BY ts_rank_cd(fts_doc, plainto_tsquery('english', %s)) DESC,
                                    symbol_id
                       ) AS fts_rank
                FROM (
                    SELECT symbol_id, fts_doc
                    FROM ast_symbols
                    WHERE domain_id = %s
                    ORDER BY ts_rank_cd(fts_doc, plainto_tsquery('english', %s)) DESC, symbol_id
                    LIMIT %s
                ) fts_sub
            ),
            vec_ranked AS (
                SELECT symbol_id,
                       ROW_NUMBER() OVER (
                           ORDER BY vec_dist ASC, symbol_id
                       ) AS vec_rank
                FROM (
                    SELECT symbol_id, (embedding <=> %s::vector) AS vec_dist
                    FROM ast_symbols
                    WHERE domain_id = %s
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s
                ) vec_sub
            ),
            trgm_ranked AS (
                SELECT symbol_id,
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
                   COALESCE(t.trgm_rank, %s) AS trgm_rank
            FROM candidate_ids c
            JOIN ast_symbols s ON s.symbol_id = c.symbol_id
            LEFT JOIN fts_ranked f ON s.symbol_id = f.symbol_id
            LEFT JOIN vec_ranked v ON s.symbol_id = v.symbol_id
            LEFT JOIN trgm_ranked t ON s.symbol_id = t.symbol_id
            WHERE s.domain_id = %s;
        """
        params = (
            query_text,
            domain_id,
            query_text,
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
            domain_id,
        )
        rows = conn.execute(query, params).fetchall()
        hits: list[RankedSearchHit] = []
        for r in rows:
            sym_id = UUID(str(r[0]))
            qual_name = str(r[1])
            fts_rank = int(str(r[2]))
            vec_rank = int(str(r[3]))
            trgm_rank = int(str(r[4]))
            norm_rrf = compute_normalized_rrf(
                [trgm_rank, fts_rank, vec_rank], num_modalities=3, k=60
            )
            hits.append(
                RankedSearchHit(
                    item_id=sym_id,
                    qualified_name_or_title=qual_name,
                    normalized_rrf_score=norm_rrf,
                    trigram_rank=trgm_rank,
                    bm25_rank=fts_rank,
                    vector_rank=vec_rank,
                )
            )
        hits.sort(key=lambda h: h.normalized_rrf_score, reverse=True)
        return hits[:top_k]

    def record_execution_telemetry(
        self, conn: Connection[tuple[object, ...]], record: ExecutionTelemetryRecord
    ) -> None:
        mode_val = str(record.execution_mode)
        missing_json = json.dumps(record.missing_context_requests)
        conn.execute(
            """
            INSERT INTO execution_telemetry (
                trace_id, domain_id, principal_id, principal_role,
                execution_mode, local_tokens_read, context_pack_tokens_sent,
                paid_tokens_saved, estimated_usd_saved, preflight_passed_first_try,
                zero_touch_sufficient, missing_context_requests, latency_ms,
                created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, COALESCE(%s, NOW()))
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
                created_at
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
              AND created_at >= NOW() - (%s || ' hours')::interval;
            """,
            (principal_id, str(since_hours)),
        ).fetchone()
        return int(str(row[0])) if row and row[0] is not None else 0

    def search_visible_ast_symbols(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        query_text: str,
        limit: int = 50,
    ) -> list[ASTSymbolRecord]:
        """Searches visible AST symbols under RLS with SQL pushdown and LIMIT."""
        import re

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
                   git_blob_sha, language, allowed_scopes, raw_content, line_count
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
                   git_blob_sha, language, allowed_scopes, raw_content, line_count
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
            )
            for r in rows
        ]


class InMemoryRLSRepository:
    """Deterministic in-memory twin enforcing identical RLS scope intersection and 3-way RRF.

    Used for fast unit testing and offline verification when a Postgres 17 daemon is not active.
    """

    def __init__(self) -> None:
        self._files: dict[UUID, CodeFileRecord] = {}
        self._symbols: dict[UUID, ASTSymbolRecord] = {}
        self._edges: dict[UUID, ASTEdgeRecord] = {}
        self._docs: dict[UUID, DocChunkRecord] = {}
        self._telemetry: dict[UUID, ExecutionTelemetryRecord] = {}
        self._repos: dict[UUID, RepositoryRecord] = {}
        self._commits: dict[UUID, RepoCommitRecord] = {}
        self._confluence: dict[UUID, ConfluencePageRecord] = {}
        self._deleted_confluence: set[tuple[str, str]] = set()
        self._jira: dict[UUID, JiraIssueRecord] = {}

    def insert_repository(self, principal: PrincipalContext, record: RepositoryRecord) -> None:
        if not _scopes_intersect(record.allowed_scopes, principal.allowed_scopes):
            raise PermissionError("Principal lacks scope to insert repository row")
        # Upsert: purge existing record for (domain_id, repo_name) to mirror ON CONFLICT DO UPDATE
        for old_id, r in list(self._repos.items()):
            if r.domain_id == record.domain_id and r.repo_name == record.repo_name:
                self._repos.pop(old_id, None)
        self._repos[record.repo_id] = record

    def get_repository(
        self, principal: PrincipalContext, domain_id: str, repo_name: str
    ) -> RepositoryRecord | None:
        for r in self._repos.values():
            if (
                r.domain_id == domain_id
                and r.repo_name == repo_name
                and _scopes_intersect(r.allowed_scopes, principal.allowed_scopes)
            ):
                return r
        return None

    def list_repositories(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[RepositoryRecord]:
        return [
            r
            for r in self._repos.values()
            if r.domain_id == domain_id
            and _scopes_intersect(r.allowed_scopes, principal.allowed_scopes)
        ]

    def insert_repo_commit(self, principal: PrincipalContext, record: RepoCommitRecord) -> None:
        self._commits[record.commit_id] = record

    def get_repo_commits(
        self, principal: PrincipalContext, repo_id: UUID
    ) -> list[RepoCommitRecord]:
        return [c for c in self._commits.values() if c.repo_id == repo_id]

    def insert_confluence_page(
        self, principal: PrincipalContext, record: ConfluencePageRecord
    ) -> None:
        if not _scopes_intersect(record.allowed_scopes, principal.allowed_scopes):
            raise PermissionError("Principal lacks scope to insert confluence_page row")
        self._deleted_confluence.discard((record.domain_id, record.confluence_page_id))
        # Upsert: purge existing record for (domain_id, confluence_page_id)
        for old_id, p in list(self._confluence.items()):
            if (
                p.domain_id == record.domain_id
                and p.confluence_page_id == record.confluence_page_id
            ):
                self._confluence.pop(old_id, None)
        self._confluence[record.page_id] = record

    def get_confluence_page(
        self,
        principal: PrincipalContext,
        domain_id: str,
        confluence_page_id: str,
        include_deleted: bool = False,
    ) -> ConfluencePageRecord | None:
        if not include_deleted and (domain_id, confluence_page_id) in self._deleted_confluence:
            return None
        for p in self._confluence.values():
            if (
                p.domain_id == domain_id
                and p.confluence_page_id == confluence_page_id
                and _scopes_intersect(p.allowed_scopes, principal.allowed_scopes)
            ):
                return p
        return None

    def list_confluence_pages(
        self,
        principal: PrincipalContext,
        domain_id: str,
        space_key: str | None = None,
        include_deleted: bool = False,
    ) -> list[ConfluencePageRecord]:
        return [
            p
            for p in self._confluence.values()
            if p.domain_id == domain_id
            and (space_key is None or p.space_key == space_key)
            and (
                include_deleted
                or (p.domain_id, p.confluence_page_id) not in self._deleted_confluence
            )
            and _scopes_intersect(p.allowed_scopes, principal.allowed_scopes)
        ]

    def tombstone_confluence_pages(
        self,
        domain_id: str,
        confluence_page_ids: list[str],
        reason: str = "Page deleted or archived in Confluence",
        status: str = "deleted",
    ) -> tuple[int, int]:
        """Soft-deletes Confluence pages and marks their active doc_chunks stale in memory."""
        import dataclasses

        del status
        clean_ids = {pid for pid in confluence_page_ids if pid}
        if not clean_ids:
            return (0, 0)

        pages_tombstoned = 0
        for pid in clean_ids:
            key = (domain_id, pid)
            had_page = any(
                p.domain_id == domain_id and p.confluence_page_id == pid
                for p in self._confluence.values()
            )
            if key not in self._deleted_confluence:
                self._deleted_confluence.add(key)
                if had_page:
                    pages_tombstoned += 1

        target_doc_ids = {f"confluence:{pid}" for pid in clean_ids}
        chunks_flagged = 0
        for cid, doc in list(self._docs.items()):
            if doc.domain_id == domain_id and doc.doc_id in target_doc_ids and not doc.is_stale:
                self._docs[cid] = dataclasses.replace(
                    doc,
                    is_stale=True,
                    stale_reason=reason,
                )
                chunks_flagged += 1
        return (pages_tombstoned, chunks_flagged)

    def update_confluence_allowed_scopes(
        self,
        domain_id: str,
        confluence_page_id: str,
        allowed_scopes: list[str],
    ) -> int:
        """Synchronizes allowed_scopes across in-memory confluence_pages and active doc_chunks."""
        import dataclasses

        scopes_copy = list(allowed_scopes)
        for pid_uuid, page_rec in list(self._confluence.items()):
            if (
                page_rec.domain_id == domain_id
                and page_rec.confluence_page_id == confluence_page_id
            ):
                self._confluence[pid_uuid] = dataclasses.replace(
                    page_rec, allowed_scopes=scopes_copy
                )

        target_doc_id = f"confluence:{confluence_page_id}"
        updated_chunks = 0
        for cid, doc in list(self._docs.items()):
            if doc.domain_id == domain_id and doc.doc_id == target_doc_id and not doc.is_stale:
                self._docs[cid] = dataclasses.replace(doc, allowed_scopes=scopes_copy)
                updated_chunks += 1
        return updated_chunks

    def insert_jira_issue(self, principal: PrincipalContext, record: JiraIssueRecord) -> None:
        if not _scopes_intersect(record.allowed_scopes, principal.allowed_scopes):
            raise PermissionError("Principal lacks scope to insert jira_issue row")
        # Upsert: purge existing record for (domain_id, issue_key) to mirror ON CONFLICT DO UPDATE
        for old_id, i in list(self._jira.items()):
            if i.domain_id == record.domain_id and i.issue_key == record.issue_key:
                self._jira.pop(old_id, None)
        self._jira[record.issue_id] = record

    def get_jira_issue(
        self, principal: PrincipalContext, domain_id: str, issue_key: str
    ) -> JiraIssueRecord | None:
        for i in self._jira.values():
            if (
                i.domain_id == domain_id
                and i.issue_key == issue_key
                and _scopes_intersect(i.allowed_scopes, principal.allowed_scopes)
            ):
                return i
        return None

    def list_jira_issues(
        self, principal: PrincipalContext, domain_id: str, project_key: str | None = None
    ) -> list[JiraIssueRecord]:
        return [
            i
            for i in self._jira.values()
            if i.domain_id == domain_id
            and (project_key is None or i.project_key == project_key)
            and _scopes_intersect(i.allowed_scopes, principal.allowed_scopes)
        ]

    def insert_code_file(self, principal: PrincipalContext, record: CodeFileRecord) -> None:
        if not _scopes_intersect(record.allowed_scopes, principal.allowed_scopes):
            raise PermissionError("Principal lacks scope to insert code_file row")
        self._files[record.file_id] = record

    def insert_ast_symbol(self, principal: PrincipalContext, record: ASTSymbolRecord) -> None:
        if not _scopes_intersect(record.allowed_scopes, principal.allowed_scopes):
            raise PermissionError("Principal lacks scope to insert ast_symbol row")
        self._symbols[record.symbol_id] = record

    def insert_ast_edge(self, principal: PrincipalContext, record: ASTEdgeRecord) -> None:
        if not _scopes_intersect(record.allowed_scopes, principal.allowed_scopes):
            raise PermissionError("Principal lacks scope to insert ast_edge row")
        self._edges[record.edge_id] = record

    def insert_doc_chunk(self, principal: PrincipalContext, record: DocChunkRecord) -> None:
        if not _scopes_intersect(record.allowed_scopes, principal.allowed_scopes):
            raise PermissionError("Principal lacks scope to insert doc_chunk row")
        self._docs[record.chunk_id] = record

    def insert_doc_chunks_batch(
        self, principal: PrincipalContext, records: list[DocChunkRecord]
    ) -> int:
        for record in records:
            self.insert_doc_chunk(principal, record)
        return len(records)

    def list_visible_code_files(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[CodeFileRecord]:
        return [
            f
            for f in self._files.values()
            if f.domain_id == domain_id
            and _scopes_intersect(f.allowed_scopes, principal.allowed_scopes)
        ]

    def list_visible_ast_symbols(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[ASTSymbolRecord]:
        return [
            s
            for s in self._symbols.values()
            if s.domain_id == domain_id
            and _scopes_intersect(s.allowed_scopes, principal.allowed_scopes)
        ]

    def list_visible_ast_edges(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[ASTEdgeRecord]:
        return [
            e
            for e in self._edges.values()
            if e.domain_id == domain_id
            and _scopes_intersect(e.allowed_scopes, principal.allowed_scopes)
        ]

    def list_visible_doc_chunks(
        self,
        principal: PrincipalContext,
        domain_id: str,
        market_scope: MarketRegime | str | None = None,
        include_stale: bool = False,
    ) -> list[DocChunkRecord]:
        target_mkt = _serialize_market_scope(market_scope)
        return [
            d
            for d in self._docs.values()
            if d.domain_id == domain_id
            and (include_stale or not d.is_stale)
            and (target_mkt is None or _serialize_market_scope(d.market_scope) == target_mkt)
            and _scopes_intersect(d.allowed_scopes, principal.allowed_scopes)
        ]

    def get_file_blob_hashes(
        self, principal: PrincipalContext, domain_id: str, repo_name: str
    ) -> dict[str, tuple[UUID, str]]:
        visible = self.list_visible_code_files(principal, domain_id)
        return {
            f.file_path: (f.file_id, f.git_blob_sha) for f in visible if f.repo_name == repo_name
        }

    def delete_code_file(self, principal: PrincipalContext, file_id: UUID) -> None:
        if file_id in self._files:
            file_rec = self._files[file_id]
            if not _scopes_intersect(file_rec.allowed_scopes, principal.allowed_scopes):
                raise PermissionError("Principal lacks scope to delete code_file")
            del self._files[file_id]
            deleted_sym_ids = {
                s_id for s_id, sym in self._symbols.items() if sym.file_id == file_id
            }
            for s_id in deleted_sym_ids:
                self._symbols.pop(s_id, None)

            # Cascade delete outward edges originating from deleted symbols
            deleted_source_edge_ids = {
                e_id
                for e_id, edge in self._edges.items()
                if edge.source_symbol_id in deleted_sym_ids
            }
            for e_id in deleted_source_edge_ids:
                self._edges.pop(e_id, None)

            # Preserve inward edges with target_symbol_id = None
            # (matches PostgreSQL ON DELETE SET NULL)
            for e_id, edge in list(self._edges.items()):
                if edge.target_symbol_id in deleted_sym_ids:
                    self._edges[e_id] = ASTEdgeRecord(
                        edge_id=edge.edge_id,
                        domain_id=edge.domain_id,
                        source_symbol_id=edge.source_symbol_id,
                        target_qualified_name=edge.target_qualified_name,
                        target_symbol_id=None,
                        edge_kind=edge.edge_kind,
                        allowed_scopes=edge.allowed_scopes,
                        target_repo_qualified_id=edge.target_repo_qualified_id,
                        is_cross_repo=edge.is_cross_repo,
                    )

    def delete_code_file_and_symbols(
        self,
        domain_id: str,
        repo_name: str,
        file_path: str,
    ) -> None:
        """Deletes code file and all associated AST symbols from in-memory stores."""
        del_ids = [
            fid
            for fid, f in self._files.items()
            if f.domain_id == domain_id
            and f.repo_name in (repo_name, repo_name.replace("/", "_"))
            and f.file_path == file_path
        ]
        for fid in del_ids:
            self._files.pop(fid, None)
            sym_del_ids = [s_id for s_id, sym in self._symbols.items() if sym.file_id == fid]
            for s_id in sym_del_ids:
                self._symbols.pop(s_id, None)
            sym_del_set = set(sym_del_ids)
            # Cascade delete outward edges originating from deleted symbols
            del_edge_ids = [
                e_id for e_id, edge in self._edges.items() if edge.source_symbol_id in sym_del_set
            ]
            for e_id in del_edge_ids:
                self._edges.pop(e_id, None)
            # Preserve inward edges with target_symbol_id = None
            # (matches PostgreSQL ON DELETE SET NULL)
            for e_id, edge in list(self._edges.items()):
                if edge.target_symbol_id in sym_del_set:
                    self._edges[e_id] = ASTEdgeRecord(
                        edge_id=edge.edge_id,
                        domain_id=edge.domain_id,
                        source_symbol_id=edge.source_symbol_id,
                        target_qualified_name=edge.target_qualified_name,
                        target_symbol_id=None,
                        edge_kind=edge.edge_kind,
                        allowed_scopes=edge.allowed_scopes,
                        target_repo_qualified_id=edge.target_repo_qualified_id,
                        is_cross_repo=edge.is_cross_repo,
                    )

    def traverse_1hop_neighborhood(
        self,
        principal_or_conn: Any,
        domain_id: str,
        focus_symbol_id: UUID | str,
        max_direct_callers: int = 3,
    ) -> list[ASTNeighborhoodRecord]:
        """In-memory twin executing identical 1-hop AST neighborhood semantics."""
        focus_uuid = UUID(str(focus_symbol_id))
        focus_sym = self._symbols.get(focus_uuid)
        if not focus_sym or focus_sym.domain_id != domain_id:
            return []

        principal_scopes = (
            principal_or_conn.allowed_scopes
            if isinstance(principal_or_conn, PrincipalContext)
            else focus_sym.allowed_scopes
        )
        if not _scopes_intersect(focus_sym.allowed_scopes, principal_scopes):
            return []

        focus_file = self._files.get(focus_sym.file_id)
        focus_repo = focus_file.repo_name if focus_file else None

        records: list[ASTNeighborhoodRecord] = []
        seen_syms: set[UUID] = {focus_sym.symbol_id}

        # 1. Hop 1 Outward edges (USES_TYPE, IMPORTS, INHERITS, same-file CALLS)
        hop1_type_sym_ids: list[UUID] = []
        for e in self._edges.values():
            if e.domain_id == domain_id and e.source_symbol_id == focus_sym.symbol_id:
                if not _scopes_intersect(e.allowed_scopes, principal_scopes):
                    continue
                if e.target_symbol_id and e.target_symbol_id in self._symbols:
                    target_sym = self._symbols[e.target_symbol_id]
                    target_file = self._files.get(target_sym.file_id)
                    if not target_file or not _scopes_intersect(
                        target_file.allowed_scopes, principal_scopes
                    ):
                        continue
                    if e.edge_kind == "CALLS" and target_sym.file_id != focus_sym.file_id:
                        continue
                    if target_sym.symbol_id not in seen_syms:
                        seen_syms.add(target_sym.symbol_id)
                        records.append(
                            ASTNeighborhoodRecord(
                                intent_role="imported_type",
                                edge_kind=e.edge_kind,
                                depth=1,
                                symbol=target_sym,
                                file=target_file,
                            )
                        )
                        if e.edge_kind in ("USES_TYPE", "INHERITS"):
                            hop1_type_sym_ids.append(target_sym.symbol_id)

        # Hop 2 Outward: Transitive schema/type/inheritance closure within same repo
        for h1_id in hop1_type_sym_ids:
            for e2 in self._edges.values():
                if (
                    e2.domain_id == domain_id
                    and e2.source_symbol_id == h1_id
                    and e2.edge_kind in ("USES_TYPE", "INHERITS")
                ):
                    if not _scopes_intersect(e2.allowed_scopes, principal_scopes):
                        continue
                    if e2.target_symbol_id and e2.target_symbol_id in self._symbols:
                        h2_sym = self._symbols[e2.target_symbol_id]
                        h2_file = self._files.get(h2_sym.file_id)
                        if not h2_file or not _scopes_intersect(
                            h2_file.allowed_scopes, principal_scopes
                        ):
                            continue
                        if focus_repo and h2_file.repo_name != focus_repo:
                            continue
                        if h2_sym.symbol_id not in seen_syms:
                            seen_syms.add(h2_sym.symbol_id)
                            records.append(
                                ASTNeighborhoodRecord(
                                    intent_role="imported_type",
                                    edge_kind=e2.edge_kind,
                                    depth=2,
                                    symbol=h2_sym,
                                    file=h2_file,
                                )
                            )

        # 2. Inward callers (CALLS) with diversity across caller files
        callers_by_file: dict[UUID, tuple[ASTSymbolRecord, CodeFileRecord]] = {}
        for e in self._edges.values():
            if e.domain_id == domain_id and e.edge_kind == "CALLS":
                matches_target = (
                    e.target_symbol_id == focus_sym.symbol_id
                    or e.target_qualified_name == focus_sym.qualified_name
                )
                if matches_target and _scopes_intersect(e.allowed_scopes, principal_scopes):
                    src_sym = self._symbols.get(e.source_symbol_id)
                    if not src_sym or src_sym.symbol_id in seen_syms:
                        continue
                    src_file = self._files.get(src_sym.file_id)
                    if (
                        src_file
                        and _scopes_intersect(src_file.allowed_scopes, principal_scopes)
                        and (
                            src_file.file_id not in callers_by_file
                            or src_sym.start_line < callers_by_file[src_file.file_id][0].start_line
                        )
                    ):
                        callers_by_file[src_file.file_id] = (src_sym, src_file)

        # Limit to max_direct_callers
        caller_items = list(callers_by_file.values())
        caller_items.sort(
            key=lambda item: (
                0 if item[1].file_id == focus_sym.file_id else 1,
                str(item[1].file_id),
                item[0].start_line,
            )
        )
        for c_sym, c_file in caller_items[:max_direct_callers]:
            seen_syms.add(c_sym.symbol_id)
            records.append(
                ASTNeighborhoodRecord(
                    intent_role="direct_caller",
                    edge_kind="CALLS",
                    depth=1,
                    symbol=c_sym,
                    file=c_file,
                )
            )

        # 3. Test fixtures (TESTED_BY)
        test_count = 0
        for e in self._edges.values():
            if e.domain_id == domain_id and e.edge_kind == "TESTED_BY":
                test_sym_id = (
                    e.target_symbol_id
                    if e.source_symbol_id == focus_sym.symbol_id
                    else e.source_symbol_id
                )
                if not test_sym_id and (e.target_qualified_name == focus_sym.qualified_name):
                    test_sym_id = e.source_symbol_id
                if test_sym_id and test_sym_id in self._symbols and test_sym_id not in seen_syms:
                    t_sym = self._symbols[test_sym_id]
                    t_file = self._files.get(t_sym.file_id)
                    if t_file and _scopes_intersect(t_file.allowed_scopes, principal_scopes):
                        seen_syms.add(t_sym.symbol_id)
                        records.append(
                            ASTNeighborhoodRecord(
                                intent_role="test_fixture",
                                edge_kind="TESTED_BY",
                                depth=1,
                                symbol=t_sym,
                                file=t_file,
                            )
                        )
                        test_count += 1
                        if test_count >= 2:
                            break

        return records

    def search_symbols_hybrid_rrf(
        self,
        principal: PrincipalContext,
        domain_id: str,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RankedSearchHit]:
        """Executes RLS-filtered 3-way Normalized RRF (trigram, BM25, cosine similarity)."""
        visible = self.list_visible_ast_symbols(principal, domain_id)
        if not visible:
            return []

        q_lower = query_text.lower()
        q_tokens = set(q_lower.split())

        # 1. Trigram / Substring similarity rank
        trgm_scored = sorted(
            visible,
            key=lambda s: (
                1.0 if q_lower in s.qualified_name.lower() else 0.0,
                sum(1 for t in q_tokens if t in s.qualified_name.lower()),
            ),
            reverse=True,
        )
        trgm_ranks = {s.symbol_id: idx + 1 for idx, s in enumerate(trgm_scored)}

        # 2. BM25 / Lexical token overlap rank
        bm25_scored = sorted(
            visible,
            key=lambda s: sum(
                1
                for t in q_tokens
                if t
                in (f"{s.qualified_name} {s.signature_skeleton} {s.contextual_summary}").lower()
            ),
            reverse=True,
        )
        bm25_ranks = {s.symbol_id: idx + 1 for idx, s in enumerate(bm25_scored)}

        # 3. Dense Vector cosine similarity rank
        vec_scored = sorted(
            visible,
            key=lambda s: _cosine_similarity(s.embedding, query_embedding),
            reverse=True,
        )
        vec_ranks = {s.symbol_id: idx + 1 for idx, s in enumerate(vec_scored)}

        hits: list[RankedSearchHit] = []
        for sym in visible:
            r_trgm = trgm_ranks[sym.symbol_id]
            r_bm25 = bm25_ranks[sym.symbol_id]
            r_vec = vec_ranks[sym.symbol_id]
            norm_rrf = compute_normalized_rrf([r_trgm, r_bm25, r_vec], num_modalities=3, k=60)
            hits.append(
                RankedSearchHit(
                    item_id=sym.symbol_id,
                    qualified_name_or_title=sym.qualified_name,
                    normalized_rrf_score=norm_rrf,
                    trigram_rank=r_trgm,
                    bm25_rank=r_bm25,
                    vector_rank=r_vec,
                )
            )

        hits.sort(key=lambda h: h.normalized_rrf_score, reverse=True)
        return hits[:top_k]

    def search_doc_chunks_hybrid_rrf(
        self,
        principal: PrincipalContext,
        domain_id: str,
        query_text: str,
        query_embedding: list[float],
        market_scope: MarketRegime | str | None = None,
        top_k: int = 5,
        include_stale: bool = False,
    ) -> list[RankedSearchHit]:
        """Executes RLS-filtered 3-way Normalized RRF over doc_chunks.

        Combines BM25 lexical rank, vector cosine similarity, and trigram/authority.
        """
        visible = [
            d
            for d in self.list_visible_doc_chunks(
                principal,
                domain_id,
                market_scope=market_scope,
                include_stale=include_stale,
            )
            if d.authority_weight > 0.0
        ]
        if not visible:
            return []

        # 1. Lexical BM25 rank over (header + heading + content)
        corpus_texts = [
            f"{d.contextual_header} {d.section_heading} {d.verbatim_content}" for d in visible
        ]
        bm25_model = BM25Okapi(corpus_texts)
        bm25_scores = bm25_model.get_scores(query_text)
        bm25_order = sorted(range(len(visible)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_ranks = {visible[idx].chunk_id: rank + 1 for rank, idx in enumerate(bm25_order)}

        # 2. Dense vector cosine similarity rank
        vec_scores = [_cosine_similarity(d.embedding, query_embedding) for d in visible]
        vec_order = sorted(range(len(visible)), key=lambda i: vec_scores[i], reverse=True)
        vec_ranks = {visible[idx].chunk_id: rank + 1 for rank, idx in enumerate(vec_order)}

        # 3. Trigram / Substring & Authority similarity rank
        trgm_scores = [
            _trigram_similarity(d.verbatim_content, query_text) * d.authority_weight
            for d in visible
        ]
        trgm_order = sorted(range(len(visible)), key=lambda i: trgm_scores[i], reverse=True)
        trgm_ranks = {visible[idx].chunk_id: rank + 1 for rank, idx in enumerate(trgm_order)}

        scored_hits: list[tuple[float, float, float, RankedSearchHit]] = []
        for d in visible:
            r_trgm = trgm_ranks[d.chunk_id]
            r_bm25 = bm25_ranks[d.chunk_id]
            r_vec = vec_ranks[d.chunk_id]
            base_rrf = compute_normalized_rrf([r_trgm, r_bm25, r_vec], num_modalities=3, k=60)
            weighted_score = _apply_authority_score(base_rrf, d.authority_weight)
            hit = RankedSearchHit(
                item_id=d.chunk_id,
                qualified_name_or_title=f"{d.doc_title} > {d.section_heading}",
                normalized_rrf_score=weighted_score,
                trigram_rank=r_trgm,
                bm25_rank=r_bm25,
                vector_rank=r_vec,
            )
            scored_hits.append(
                (weighted_score, base_rrf * d.authority_weight, d.authority_weight, hit)
            )

        scored_hits.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [item[3] for item in scored_hits[:top_k]]

    def record_execution_telemetry(
        self,
        conn_or_record: Any,
        maybe_record: ExecutionTelemetryRecord | None = None,
    ) -> None:
        if isinstance(conn_or_record, ExecutionTelemetryRecord):
            record = conn_or_record
        elif maybe_record is not None:
            record = maybe_record
        else:
            raise ValueError("record_execution_telemetry requires an ExecutionTelemetryRecord")
        self._telemetry[record.trace_id] = record

    def get_execution_telemetry(
        self,
        trace_id_or_conn: Any,
        maybe_trace_id: UUID | None = None,
    ) -> ExecutionTelemetryRecord | None:
        if isinstance(trace_id_or_conn, UUID):
            return self._telemetry.get(trace_id_or_conn)
        if maybe_trace_id is not None:
            return self._telemetry.get(maybe_trace_id)
        if isinstance(trace_id_or_conn, str):
            return self._telemetry.get(UUID(trace_id_or_conn))
        return None

    def get_principal_paid_tokens_consumed(
        self,
        principal_or_conn: Any,
        maybe_principal_id: str | None = None,
        since_hours: int = 24,
    ) -> int:
        if isinstance(principal_or_conn, str):
            principal_id = principal_or_conn
        elif maybe_principal_id is not None:
            principal_id = maybe_principal_id
        else:
            return 0
        return sum(
            rec.context_pack_tokens_sent
            for rec in self._telemetry.values()
            if rec.principal_id == principal_id
        )

    def search_visible_ast_symbols(
        self,
        principal: PrincipalContext,
        domain_id: str,
        query_text: str,
        limit: int = 50,
    ) -> list[ASTSymbolRecord]:
        """Searches visible AST symbols ranked by lexical overlap under RLS."""
        import re

        visible = self.list_visible_ast_symbols(principal, domain_id)
        words = [
            w.lower()
            for w in re.findall(r"[A-Za-z0-9_]{3,}", query_text)
            if w.lower() not in {"the", "and", "for", "with", "how", "what", "where", "from"}
        ]
        if not words:
            return visible[:limit]

        scored: list[tuple[int, ASTSymbolRecord]] = []
        for sym in visible:
            qual_l = sym.qualified_name.lower()
            sig_l = sym.signature_skeleton.lower()
            sum_l = sym.contextual_summary.lower()
            score = 0
            for w in words[:8]:
                if w in qual_l:
                    score += 6
                if w in sig_l:
                    score += 3
                if w in sum_l:
                    score += 1
            if score > 0:
                scored.append((score, sym))
        scored.sort(key=lambda item: (-item[0], item[1].qualified_name))
        return [s for _, s in scored[:limit]]

    def get_code_files_by_ids(
        self,
        principal: PrincipalContext,
        domain_id: str,
        file_ids: list[UUID | str],
    ) -> list[CodeFileRecord]:
        """Fetches visible code files by UUID under RLS without full-table scan."""
        target_ids: set[UUID] = set()
        for fid in file_ids:
            try:
                target_ids.add(fid if isinstance(fid, UUID) else UUID(str(fid)))
            except ValueError as exc:
                logger.debug(
                    "Ignoring malformed file_id %r (%s): %s",
                    fid,
                    type(exc).__name__,
                    exc,
                )
                continue
        if not target_ids:
            return []
        return [
            f for f in self.list_visible_code_files(principal, domain_id) if f.file_id in target_ids
        ]
