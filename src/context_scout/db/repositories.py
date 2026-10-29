"""RLS-enforced repositories and 3-way Normalized RRF search (M1 Foundation)."""

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Literal
from uuid import UUID

from psycopg import Connection

from context_scout.db.bm25 import BM25Okapi
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

    def insert_doc_chunk(
        self, conn: Connection[tuple[object, ...]], record: DocChunkRecord
    ) -> None:
        market_val = (
            record.market_scope.value
            if hasattr(record.market_scope, "value")
            else str(record.market_scope)
        )
        vec_literal = "[" + ",".join(f"{v:.6f}" for v in record.embedding) + "]"
        conn.execute(
            """
            INSERT INTO doc_chunks (
                chunk_id, domain_id, doc_id, doc_title, source_uri,
                market_scope, published_date, authority_weight,
                section_heading, contextual_header, verbatim_content,
                metadata_json, allowed_scopes, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::vector)
            """,
            (
                record.chunk_id,
                record.domain_id,
                record.doc_id,
                record.doc_title,
                record.source_uri,
                market_val,
                record.published_date,
                record.authority_weight,
                record.section_heading,
                record.contextual_header,
                record.verbatim_content,
                json.dumps(record.metadata_json),
                record.allowed_scopes,
                vec_literal,
            ),
        )


    def insert_doc_chunks_batch(
        self, conn: Connection[tuple[object, ...]], records: list[DocChunkRecord]
    ) -> int:
        for record in records:
            self.insert_doc_chunk(conn, record)
        return len(records)

    def list_visible_doc_chunks(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        market_scope: MarketRegime | None = None,
    ) -> list[DocChunkRecord]:
        query = """
            SELECT chunk_id, domain_id, doc_id, doc_title, source_uri,
                   market_scope, published_date, authority_weight,
                   section_heading, contextual_header, verbatim_content,
                   metadata_json, allowed_scopes, embedding
            FROM doc_chunks
            WHERE domain_id = %s
        """
        params: list[object] = [domain_id]
        if market_scope is not None:
            query += " AND market_scope = %s"
            params.append(
                market_scope.value if hasattr(market_scope, "value") else str(market_scope)
            )
        query += " ORDER BY doc_id, chunk_id;"

        rows = conn.execute(query, tuple(params)).fetchall()
        results: list[DocChunkRecord] = []
        for r in rows:
            raw_emb = r[13]
            if isinstance(raw_emb, str):
                emb_floats = [
                    float(x.strip()) for x in raw_emb.strip("[]").split(",") if x.strip()
                ]
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
                    market_scope=MarketRegime(str(r[5])),
                    published_date=pub_date,
                    authority_weight=float(str(r[7])),
                    section_heading=str(r[8]),
                    contextual_header=str(r[9]),
                    verbatim_content=str(r[10]),
                    metadata_json=meta_dict,
                    allowed_scopes=scopes_list,
                    embedding=emb_floats,
                )
            )
        return results

    def search_doc_chunks_hybrid_rrf(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        query_text: str,
        query_embedding: list[float],
        market_scope: MarketRegime | None = None,
        top_k: int = 5,
    ) -> list[RankedSearchHit]:
        vec_literal = "[" + ",".join(f"{v:.6f}" for v in query_embedding) + "]"
        market_str = (
            (market_scope.value if hasattr(market_scope, "value") else str(market_scope))
            if market_scope is not None
            else None
        )
        query = """
            WITH fts_ranked AS (
                SELECT chunk_id,
                       ROW_NUMBER() OVER (
                           ORDER BY ts_rank_cd(fts_doc, plainto_tsquery('english', %s)) DESC,
                                    chunk_id
                       ) AS fts_rank
                FROM doc_chunks
                WHERE domain_id = %s
                  AND (%s::text IS NULL OR market_scope = %s)
            ),
            vec_ranked AS (
                SELECT chunk_id,
                       ROW_NUMBER() OVER (
                           ORDER BY (embedding <=> %s::vector) ASC, chunk_id
                       ) AS vec_rank
                FROM doc_chunks
                WHERE domain_id = %s
                  AND (%s::text IS NULL OR market_scope = %s)
            ),
            trgm_ranked AS (
                SELECT chunk_id,
                       ROW_NUMBER() OVER (
                           ORDER BY (similarity(verbatim_content, %s) * authority_weight) DESC,
                                    chunk_id
                       ) AS trgm_rank
                FROM doc_chunks
                WHERE domain_id = %s
                  AND (%s::text IS NULL OR market_scope = %s)
            )
            SELECT d.chunk_id, d.doc_title, d.section_heading,
                   f.fts_rank, v.vec_rank, t.trgm_rank
            FROM doc_chunks d
            JOIN fts_ranked f ON d.chunk_id = f.chunk_id
            JOIN vec_ranked v ON d.chunk_id = v.chunk_id
            JOIN trgm_ranked t ON d.chunk_id = t.chunk_id
            WHERE d.domain_id = %s
              AND (%s::text IS NULL OR d.market_scope = %s);
        """
        params = (
            query_text, domain_id, market_str, market_str,
            vec_literal, domain_id, market_str, market_str,
            query_text, domain_id, market_str, market_str,
            domain_id, market_str, market_str,
        )
        rows = conn.execute(query, params).fetchall()
        hits: list[RankedSearchHit] = []
        for r in rows:
            chunk_id = UUID(str(r[0]))
            doc_title = str(r[1])
            section_heading = str(r[2])
            fts_rank = int(str(r[3]))
            vec_rank = int(str(r[4]))
            trgm_rank = int(str(r[5]))
            norm_rrf = compute_normalized_rrf(
                [trgm_rank, fts_rank, vec_rank], num_modalities=3, k=60
            )
            hits.append(
                RankedSearchHit(
                    item_id=chunk_id,
                    qualified_name_or_title=f"{doc_title} > {section_heading}",
                    normalized_rrf_score=norm_rrf,
                    trigram_rank=trgm_rank,
                    bm25_rank=fts_rank,
                    vector_rank=vec_rank,
                )
            )
        hits.sort(key=lambda h: h.normalized_rrf_score, reverse=True)
        return hits[:top_k]



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
        market_scope: MarketRegime | None = None,
    ) -> list[DocChunkRecord]:
        return [
            d
            for d in self._docs.values()
            if d.domain_id == domain_id
            and (market_scope is None or d.market_scope == market_scope)
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
            deleted_edge_ids = {
                e_id
                for e_id, edge in self._edges.items()
                if edge.source_symbol_id in deleted_sym_ids
                or edge.target_symbol_id in deleted_sym_ids
            }
            for e_id in deleted_edge_ids:
                self._edges.pop(e_id, None)

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
        market_scope: MarketRegime | None = None,
        top_k: int = 5,
    ) -> list[RankedSearchHit]:
        """Executes RLS-filtered 3-way Normalized RRF over doc_chunks.

        Combines BM25 lexical rank, vector cosine similarity, and trigram/authority.
        """
        visible = self.list_visible_doc_chunks(principal, domain_id, market_scope=market_scope)
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

        hits: list[RankedSearchHit] = []
        for d in visible:
            r_trgm = trgm_ranks[d.chunk_id]
            r_bm25 = bm25_ranks[d.chunk_id]
            r_vec = vec_ranks[d.chunk_id]
            norm_rrf = compute_normalized_rrf([r_trgm, r_bm25, r_vec], num_modalities=3, k=60)
            hits.append(
                RankedSearchHit(
                    item_id=d.chunk_id,
                    qualified_name_or_title=f"{d.doc_title} > {d.section_heading}",
                    normalized_rrf_score=norm_rrf,
                    trigram_rank=r_trgm,
                    bm25_rank=r_bm25,
                    vector_rank=r_vec,
                )
            )

        hits.sort(key=lambda h: h.normalized_rrf_score, reverse=True)
        return hits[:top_k]

