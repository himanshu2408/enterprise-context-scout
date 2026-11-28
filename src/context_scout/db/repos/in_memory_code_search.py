"""In-memory RLS repository mixin: repos, code files, AST symbols, 1-hop & hybrid search."""

from __future__ import annotations

import logging
import math
import re
from typing import Any
from uuid import UUID

from context_scout.db.bm25 import BM25Okapi
from context_scout.db.repos.in_memory_base import _InMemoryStateBase
from context_scout.db.repos.types_and_helpers import (
    CalibratedRankedSearchHit,
    _apply_authority_score,
    _compute_calibrated_relevance,
    _connector_ingest_or_scopes_intersect,
    _cosine_similarity,
    _decompose_identifier_words,
    _extract_substantive_tokens,
    _scopes_intersect,
    _token_matches_text,
    _trigram_similarity,
)
from context_scout.db.schema import (
    ASTEdgeRecord,
    ASTNeighborhoodRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    RankedSearchHit,
    RepoCommitRecord,
    RepositoryRecord,
)
from context_scout.schemas.contracts import (
    PrincipalContext,
    compute_normalized_rrf,
)

logger = logging.getLogger(__name__)


class _InMemoryCodeAndSearchMixin(_InMemoryStateBase):
    def insert_repository(self, principal: PrincipalContext, record: RepositoryRecord) -> None:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on insert_repository")
        if not _connector_ingest_or_scopes_intersect(
            record.allowed_scopes, principal.allowed_scopes
        ):
            raise PermissionError("Principal lacks scope to insert repository row")
        # Upsert: purge existing record for (tenant_id, domain_id, repo_name) to mirror ON CONFLICT
        for old_id, r in list(self._repos.items()):
            if (
                r.tenant_id == record.tenant_id
                and r.domain_id == record.domain_id
                and r.repo_name == record.repo_name
            ):
                self._repos.pop(old_id, None)
        self._repos[record.repo_id] = record

    def upsert_repository(self, principal: PrincipalContext, record: RepositoryRecord) -> None:
        self.insert_repository(principal, record)

    def get_repository(
        self, principal: PrincipalContext, domain_id: str, repo_name: str
    ) -> RepositoryRecord | None:
        for r in self._repos.values():
            if (
                r.tenant_id == principal.tenant_id
                and r.domain_id == domain_id
                and r.repo_name == repo_name
                and _connector_ingest_or_scopes_intersect(
                    r.allowed_scopes, principal.allowed_scopes
                )
            ):
                return r
        return None

    def list_repositories(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[RepositoryRecord]:
        return [
            r
            for r in self._repos.values()
            if r.tenant_id == principal.tenant_id
            and r.domain_id == domain_id
            and _connector_ingest_or_scopes_intersect(r.allowed_scopes, principal.allowed_scopes)
        ]

    def insert_repo_commit(self, principal: PrincipalContext, record: RepoCommitRecord) -> None:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on insert_repo_commit")
        self._commits[record.commit_id] = record

    def get_repo_commits(
        self, principal: PrincipalContext, repo_id: UUID
    ) -> list[RepoCommitRecord]:
        visible_repo_ids = {
            r.repo_id
            for r in self._repos.values()
            if r.tenant_id == principal.tenant_id
            and r.repo_id == repo_id
            and _connector_ingest_or_scopes_intersect(r.allowed_scopes, principal.allowed_scopes)
        }
        return [
            c
            for c in self._commits.values()
            if c.tenant_id == principal.tenant_id
            and c.repo_id == repo_id
            and (not self._repos or c.repo_id in visible_repo_ids)
        ]

    def insert_code_file(self, principal: PrincipalContext, record: CodeFileRecord) -> None:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on insert_code_file")
        if not _connector_ingest_or_scopes_intersect(
            record.allowed_scopes, principal.allowed_scopes
        ):
            raise PermissionError("Principal lacks scope to insert code_file row")
        self._files[record.file_id] = record

    def insert_ast_symbol(self, principal: PrincipalContext, record: ASTSymbolRecord) -> None:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on insert_ast_symbol")
        if not _connector_ingest_or_scopes_intersect(
            record.allowed_scopes, principal.allowed_scopes
        ):
            raise PermissionError("Principal lacks scope to insert ast_symbol row")
        self._symbols[record.symbol_id] = record

    def insert_ast_edge(self, principal: PrincipalContext, record: ASTEdgeRecord) -> None:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on insert_ast_edge")
        if not _connector_ingest_or_scopes_intersect(
            record.allowed_scopes, principal.allowed_scopes
        ):
            raise PermissionError("Principal lacks scope to insert ast_edge row")
        self._edges[record.edge_id] = record

    def list_visible_code_files(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[CodeFileRecord]:
        return [
            f
            for f in self._files.values()
            if f.tenant_id == principal.tenant_id
            and f.domain_id == domain_id
            and _connector_ingest_or_scopes_intersect(f.allowed_scopes, principal.allowed_scopes)
        ]

    def list_visible_ast_symbols(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[ASTSymbolRecord]:
        return [
            s
            for s in self._symbols.values()
            if s.tenant_id == principal.tenant_id
            and s.domain_id == domain_id
            and _connector_ingest_or_scopes_intersect(s.allowed_scopes, principal.allowed_scopes)
        ]

    def list_visible_ast_edges(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[ASTEdgeRecord]:
        return [
            e
            for e in self._edges.values()
            if e.tenant_id == principal.tenant_id
            and e.domain_id == domain_id
            and _connector_ingest_or_scopes_intersect(e.allowed_scopes, principal.allowed_scopes)
        ]

    def get_file_blob_hashes(
        self, principal: PrincipalContext, domain_id: str, repo_name: str
    ) -> dict[str, tuple[UUID, str]]:
        return {
            f.file_path: (f.file_id, f.git_blob_sha)
            for f in self._files.values()
            if f.tenant_id == principal.tenant_id
            and f.domain_id == domain_id
            and f.repo_name == repo_name
            and _connector_ingest_or_scopes_intersect(f.allowed_scopes, principal.allowed_scopes)
        }

    def delete_code_file(self, principal: PrincipalContext, file_id: UUID) -> None:
        if file_id in self._files:
            file_rec = self._files[file_id]
            if file_rec.tenant_id != principal.tenant_id:
                raise PermissionError("Principal tenant_id mismatch on delete_code_file")
            if not _scopes_intersect(file_rec.allowed_scopes, principal.allowed_scopes):
                raise PermissionError("Principal lacks scope to delete code_file")
            del self._files[file_id]
            for se_id in [
                se_id
                for se_id, se in self._side_effects.items()
                if se.file_id == file_id
                or (
                    se.tenant_id == file_rec.tenant_id
                    and se.domain_id == file_rec.domain_id
                    and se.repo_name == file_rec.repo_name
                    and se.file_path == file_rec.file_path
                )
            ]:
                self._side_effects.pop(se_id, None)
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
            removed_file = self._files.pop(fid, None)
            for se_id in [
                se_id
                for se_id, se in self._side_effects.items()
                if se.file_id == fid
                or (
                    removed_file is not None
                    and se.tenant_id == removed_file.tenant_id
                    and se.domain_id == domain_id
                    and se.repo_name in (repo_name, repo_name.replace("/", "_"))
                    and se.file_path == file_path
                )
            ]:
                self._side_effects.pop(se_id, None)
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
        if isinstance(principal_or_conn, PrincipalContext) and (
            focus_sym.tenant_id != principal_or_conn.tenant_id
        ):
            return []

        principal_scopes = (
            principal_or_conn.allowed_scopes
            if isinstance(principal_or_conn, PrincipalContext)
            else focus_sym.allowed_scopes
        )
        target_tenant_id = (
            principal_or_conn.tenant_id
            if isinstance(principal_or_conn, PrincipalContext)
            else focus_sym.tenant_id
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
            if (
                e.tenant_id == target_tenant_id
                and e.domain_id == domain_id
                and e.source_symbol_id == focus_sym.symbol_id
            ):
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
                        if e.edge_kind in (
                            "USES_TYPE",
                            "INHERITS",
                            "IMPORTS",
                            "PROVISIONED_BY",
                            "DISPATCHES",
                            "SELECTS",
                            "DOCUMENTS_COMPONENT",
                            "OWNED_BY_TEAM",
                            "PART_OF_SYSTEM",
                            "PUBLISHES_EVENT",
                            "SUBSCRIBES_EVENT",
                            "READS_TABLE",
                            "WRITES_TABLE",
                            "RENDERS_TEMPLATE",
                            "IMPLEMENTS_SPEC",
                            "IMPLEMENTS_ISSUE",
                            "BINDS_SERVICE_ACCOUNT",
                            "ACCESSES_SECRET",
                        ):
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

        # 2. Inward callers & cross-repo contract consumers with diversity across caller files
        _inward_contract_kinds = (
            "USES_TYPE",
            "PROVISIONED_BY",
            "DOCUMENTS_COMPONENT",
            "PUBLISHES_EVENT",
            "SUBSCRIBES_EVENT",
            "READS_TABLE",
            "WRITES_TABLE",
            "RENDERS_TEMPLATE",
            "IMPLEMENTS_SPEC",
            "IMPLEMENTS_ISSUE",
            "BINDS_SERVICE_ACCOUNT",
            "ACCESSES_SECRET",
        )
        callers_by_file: dict[UUID, tuple[ASTSymbolRecord, CodeFileRecord]] = {}
        for e in self._edges.values():
            if e.domain_id == domain_id and (
                e.edge_kind == "CALLS"
                or (e.is_cross_repo and e.edge_kind in _inward_contract_kinds)
            ):
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
        min_calibrated_score: float = 0.0,
    ) -> list[RankedSearchHit]:
        """Executes RLS-filtered 3-way Normalized RRF (trigram, BM25, cosine similarity)."""
        visible = self.list_visible_ast_symbols(principal, domain_id)
        if not visible:
            return []

        q_lower = query_text.lower()
        q_tokens = _extract_substantive_tokens(query_text) or list(set(q_lower.split()))

        # 1. Trigram / Substring similarity rank
        trgm_raw = {
            s.symbol_id: max(
                _trigram_similarity(s.qualified_name, query_text),
                _trigram_similarity(s.signature_skeleton, query_text),
            )
            for s in visible
        }
        trgm_scored = sorted(
            visible,
            key=lambda s: (
                1.0 if q_lower in s.qualified_name.lower() else 0.0,
                sum(
                    1
                    for t in q_tokens
                    if _token_matches_text(
                        t,
                        f"{s.qualified_name.lower()} "
                        f"{_decompose_identifier_words(s.qualified_name)}",
                    )
                ),
                trgm_raw[s.symbol_id],
            ),
            reverse=True,
        )
        trgm_ranks = {s.symbol_id: idx + 1 for idx, s in enumerate(trgm_scored)}

        # 2. BM25 / Lexical token overlap rank (with CamelCase/snake_case decomposition)
        decomposed_cache = {
            s.symbol_id: (
                f"{s.qualified_name} {_decompose_identifier_words(s.qualified_name)} "
                f"{s.signature_skeleton} {_decompose_identifier_words(s.signature_skeleton)} "
                f"{s.contextual_summary}"
            )
            for s in visible
        }
        corpus_texts = [decomposed_cache[s.symbol_id] for s in visible]
        bm25_model = BM25Okapi(corpus_texts)
        bm25_raw_list = bm25_model.get_scores(query_text)
        bm25_raw = {visible[i].symbol_id: float(bm25_raw_list[i]) for i in range(len(visible))}
        bm25_scored = sorted(
            visible,
            key=lambda s: (
                sum(
                    1
                    for t in q_tokens
                    if _token_matches_text(t, decomposed_cache[s.symbol_id].lower())
                ),
                bm25_raw[s.symbol_id],
            ),
            reverse=True,
        )
        bm25_ranks = {s.symbol_id: idx + 1 for idx, s in enumerate(bm25_scored)}

        # 3. Dense Vector cosine similarity rank
        vec_raw = {s.symbol_id: _cosine_similarity(s.embedding, query_embedding) for s in visible}
        vec_scored = sorted(
            visible,
            key=lambda s: vec_raw[s.symbol_id],
            reverse=True,
        )
        vec_ranks = {s.symbol_id: idx + 1 for idx, s in enumerate(vec_scored)}

        hits: list[CalibratedRankedSearchHit] = []
        for sym in visible:
            r_trgm = trgm_ranks[sym.symbol_id]
            r_bm25 = bm25_ranks[sym.symbol_id]
            r_vec = vec_ranks[sym.symbol_id]
            norm_rrf = compute_normalized_rrf([r_trgm, r_bm25, r_vec], num_modalities=3, k=60)
            calibrated_score, lexical_cov, _ = _compute_calibrated_relevance(
                query_text=query_text,
                title=sym.qualified_name,
                section=sym.signature_skeleton,
                content=sym.contextual_summary or "",
                cosine_similarity=vec_raw[sym.symbol_id],
                bm25_score=bm25_raw[sym.symbol_id],
                trigram_score=trgm_raw[sym.symbol_id],
                raw_rrf_score=norm_rrf,
                authority_weight=1.0,
                is_code_symbol=True,
            )
            file_rec = self._files.get(sym.file_id)
            ranking_weight = file_rec.ranking_weight if file_rec is not None else 1.0
            if not math.isclose(ranking_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
                norm_rrf = _apply_authority_score(norm_rrf, ranking_weight)
                calibrated_score = round(min(1.0, max(0.0, calibrated_score * ranking_weight)), 6)
            if min_calibrated_score > 0.0 and calibrated_score < min_calibrated_score:
                continue
            hits.append(
                CalibratedRankedSearchHit(
                    item_id=sym.symbol_id,
                    qualified_name_or_title=sym.qualified_name,
                    normalized_rrf_score=norm_rrf,
                    trigram_rank=r_trgm,
                    bm25_rank=r_bm25,
                    vector_rank=r_vec,
                    cosine_similarity=round(vec_raw[sym.symbol_id], 6),
                    bm25_score=round(bm25_raw[sym.symbol_id], 6),
                    trigram_score=round(trgm_raw[sym.symbol_id], 6),
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
        principal: PrincipalContext,
        domain_id: str,
        query_text: str,
        limit: int = 50,
    ) -> list[ASTSymbolRecord]:
        """Searches visible AST symbols ranked by lexical overlap under RLS."""

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

    def get_code_files_by_paths(
        self,
        principal: PrincipalContext,
        domain_id: str,
        file_paths: list[str],
    ) -> list[CodeFileRecord]:
        """Fetches visible code files by file_path under RLS."""
        target_paths = {p.strip() for p in file_paths if p and p.strip()}
        if not target_paths:
            return []
        return [
            f
            for f in self.list_visible_code_files(principal, domain_id)
            if f.file_path in target_paths
        ]

    def list_visible_code_file_records(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        repo_name: str | None = None,
        limit: int = 2000,
    ) -> list[CodeFileRecord]:
        files = self.list_visible_code_files(principal, domain_id)
        if repo_name:
            files = [f for f in files if f.repo_name == repo_name]
        return files[: max(1, limit)]

    def list_visible_ast_symbol_records(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        limit: int = 5000,
    ) -> list[ASTSymbolRecord]:
        return self.list_visible_ast_symbols(principal, domain_id)[: max(1, limit)]

    def list_visible_ast_edge_records(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        limit: int = 10000,
    ) -> list[ASTEdgeRecord]:
        return self.list_visible_ast_edges(principal, domain_id)[: max(1, limit)]
