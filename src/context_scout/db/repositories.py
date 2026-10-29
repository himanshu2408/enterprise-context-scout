"""RLS-enforced repositories and 3-way Normalized RRF search (M1 Foundation)."""

import math
from dataclasses import dataclass
from datetime import date
from typing import Literal
from uuid import UUID

from psycopg import Connection

from context_scout.schemas.contracts import (
    DomainPackManifest,
    MarketRegime,
    PrincipalContext,
    compute_normalized_rrf,
)


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


@dataclass(frozen=True)
class ASTSymbolRecord:
    symbol_id: UUID
    file_id: UUID
    domain_id: str
    qualified_name: str
    symbol_kind: Literal["function", "class", "method", "interface", "type_alias"]
    start_line: int
    end_line: int
    signature_skeleton: str
    contextual_summary: str
    allowed_scopes: list[str]
    embedding: list[float]


@dataclass(frozen=True)
class ASTEdgeRecord:
    edge_id: UUID
    domain_id: str
    source_symbol_id: UUID
    target_qualified_name: str
    target_symbol_id: UUID | None
    edge_kind: Literal["CALLS", "IMPORTS", "USES_TYPE", "INHERITS", "TESTED_BY"]
    allowed_scopes: list[str]


@dataclass(frozen=True)
class DocChunkRecord:
    chunk_id: UUID
    domain_id: str
    doc_id: str
    doc_title: str
    source_uri: str
    market_scope: MarketRegime
    published_date: date | None
    authority_weight: float
    section_heading: str
    contextual_header: str
    verbatim_content: str
    metadata_json: dict[str, object]
    allowed_scopes: list[str]
    embedding: list[float]


@dataclass(frozen=True)
class RankedSearchHit:
    item_id: UUID
    qualified_name_or_title: str
    normalized_rrf_score: float
    trigram_rank: int | None
    bm25_rank: int | None
    vector_rank: int | None


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

    def insert_ast_edge(
        self, conn: Connection[tuple[object, ...]], record: ASTEdgeRecord
    ) -> None:
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


class InMemoryRLSRepository:
    """Deterministic in-memory twin enforcing identical RLS scope intersection and 3-way RRF.

    Used for fast unit testing and offline verification when a Postgres 17 daemon is not active.
    """

    def __init__(self) -> None:
        self._files: dict[UUID, CodeFileRecord] = {}
        self._symbols: dict[UUID, ASTSymbolRecord] = {}
        self._edges: dict[UUID, ASTEdgeRecord] = {}
        self._docs: dict[UUID, DocChunkRecord] = {}

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
        self, principal: PrincipalContext, domain_id: str
    ) -> list[DocChunkRecord]:
        return [
            d
            for d in self._docs.values()
            if d.domain_id == domain_id
            and _scopes_intersect(d.allowed_scopes, principal.allowed_scopes)
        ]

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
                in (
                    f"{s.qualified_name} {s.signature_skeleton} {s.contextual_summary}"
                ).lower()
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
