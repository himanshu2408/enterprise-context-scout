"""Database schema definitions, records, and domain models for Enterprise Context Scout."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

from context_scout.schemas.contracts import MarketRegime

SymbolKind = Literal[
    "function",
    "class",
    "method",
    "interface",
    "type_alias",
    "hook",
    "slice",
    "selector",
    "reducer",
    "story",
    "tf_resource",
    "tf_module",
    "gcp_resource",
]

EdgeKind = Literal[
    "CALLS",
    "IMPORTS",
    "USES_TYPE",
    "INHERITS",
    "TESTED_BY",
    "PROVISIONED_BY",
    "DISPATCHES",
    "SELECTS",
    "DOCUMENTS_COMPONENT",
]


@dataclass(frozen=True)
class RepositoryRecord:
    repo_id: UUID
    domain_id: str
    repo_name: str
    source_type: Literal["gitlab", "github", "local_git"]
    remote_url: str
    default_branch: str
    allowed_scopes: list[str]
    head_commit_sha: str = ""
    metadata_json: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class RepoCommitRecord:
    commit_id: UUID
    repo_id: UUID
    commit_sha: str
    parent_shas: list[str]
    author_name: str
    author_email: str
    commit_message: str
    committed_at: datetime
    indexed_at: datetime | None = None


@dataclass(frozen=True)
class ConfluencePageRecord:
    page_id: UUID
    domain_id: str
    confluence_page_id: str
    space_key: str
    title: str
    version_number: int
    author_username: str
    source_url: str
    allowed_scopes: list[str]
    last_modified_at: datetime
    indexed_at: datetime | None = None


@dataclass(frozen=True)
class JiraIssueRecord:
    issue_id: UUID
    domain_id: str
    issue_key: str
    project_key: str
    summary: str
    issue_type: str
    status: str
    source_url: str
    allowed_scopes: list[str]
    updated_at: datetime
    resolution: str | None = None
    assignee_username: str | None = None
    indexed_at: datetime | None = None


@dataclass(frozen=True)
class CodeFileRecord:
    file_id: UUID
    domain_id: str
    repo_name: str
    file_path: str
    git_commit_sha: str
    git_blob_sha: str
    language: str
    allowed_scopes: list[str]
    raw_content: str
    line_count: int
    repo_id: UUID | None = None


@dataclass(frozen=True)
class ASTSymbolRecord:
    symbol_id: UUID
    file_id: UUID
    domain_id: str
    qualified_name: str
    symbol_kind: str
    start_line: int
    end_line: int
    signature_skeleton: str
    contextual_summary: str
    allowed_scopes: list[str]
    embedding: list[float] = field(default_factory=list)
    repo_id: UUID | None = None
    repo_qualified_id: str | None = None
    bare_name: str | None = None


@dataclass(frozen=True)
class ASTEdgeRecord:
    edge_id: UUID
    domain_id: str
    source_symbol_id: UUID
    target_qualified_name: str
    target_symbol_id: UUID | None
    edge_kind: str
    allowed_scopes: list[str]
    target_repo_qualified_id: str | None = None
    is_cross_repo: bool = False


@dataclass(frozen=True)
class DocChunkRecord:
    chunk_id: UUID
    domain_id: str
    doc_id: str
    doc_title: str
    source_uri: str
    market_scope: MarketRegime | str | None
    published_date: date | None
    authority_weight: float
    section_heading: str
    contextual_header: str
    verbatim_content: str
    metadata_json: dict[str, object]
    allowed_scopes: list[str]
    embedding: list[float]
    source_type: str = "quant_paper"
    version_str: str = "1"
    is_stale: bool = False
    stale_reason: str | None = None


@dataclass(frozen=True)
class ASTNeighborhoodRecord:
    """Resolved 1-hop AST neighbor record returned by recursive CTE or memory traversal."""

    intent_role: Literal["imported_type", "direct_caller", "test_fixture"]
    edge_kind: str
    depth: int
    symbol: ASTSymbolRecord
    file: CodeFileRecord


@dataclass(frozen=True)
class RankedSearchHit:
    item_id: UUID
    qualified_name_or_title: str
    normalized_rrf_score: float
    trigram_rank: int | None
    bm25_rank: int | None
    vector_rank: int | None
