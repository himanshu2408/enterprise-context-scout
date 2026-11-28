"""In-memory RLS repository mixin: manifests, embedding spaces, Confluence, Jira, docs."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
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
    _market_scope_aliases,
    _parse_market_scope,
    _scopes_intersect,
    _serialize_market_scope,
    _trigram_similarity,
)
from context_scout.db.schema import (
    ConfluencePageRecord,
    ConnectorWatermarkRecord,
    DocChunkRecord,
    DomainSyncManifestRecord,
    EmbeddingSpaceRecord,
    JiraIssueRecord,
    RankedSearchHit,
)
from context_scout.retrieval.embedder import validate_non_zero_embedding
from context_scout.schemas.contracts import (
    DEFAULT_TENANT_ID,
    DocumentScope,
    PrincipalContext,
    compute_normalized_rrf,
)

logger = logging.getLogger(__name__)


class _InMemoryDocsAndConnectorsMixin(_InMemoryStateBase):
    def upsert_domain_sync_manifest(
        self,
        principal: PrincipalContext,
        record: DomainSyncManifestRecord,
    ) -> None:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on upsert_domain_sync_manifest")
        with self._lock:
            key = (principal.tenant_id, record.domain_id)
            prev = self._sync_manifests.get(key)
            self._sync_manifests[key] = DomainSyncManifestRecord(
                tenant_id=principal.tenant_id,
                domain_id=record.domain_id,
                kind=record.kind,
                source_dir=record.source_dir,
                repo_commit_shas=dict(record.repo_commit_shas),
                empty_doc_shas=dict(record.empty_doc_shas),
                code_files_indexed=record.code_files_indexed,
                ast_symbols_indexed=record.ast_symbols_indexed,
                ast_edges_indexed=record.ast_edges_indexed,
                doc_chunks_indexed=record.doc_chunks_indexed,
                s3_manifest_uri=record.s3_manifest_uri
                if record.s3_manifest_uri is not None
                else (prev.s3_manifest_uri if prev is not None else None),
                manifest_sha256=record.manifest_sha256
                if record.manifest_sha256 is not None
                else (prev.manifest_sha256 if prev is not None else None),
                synced_at=record.synced_at or datetime.now(UTC),
            )

    def get_domain_sync_manifest(
        self,
        principal: PrincipalContext,
        domain_id: str,
    ) -> DomainSyncManifestRecord | None:
        with self._lock:
            return self._sync_manifests.get((principal.tenant_id, domain_id))

    def register_embedding_space(
        self,
        principal: PrincipalContext,
        record: EmbeddingSpaceRecord,
    ) -> EmbeddingSpaceRecord:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on register_embedding_space")
        if record.dimensions < 128 or record.dimensions > 4096:
            raise ValueError(
                f"Embedding space dimensions must be in [128, 4096], got {record.dimensions}"
            )
        with self._lock:
            if record.is_active:
                for key, existing in list(self._embedding_spaces.items()):
                    if (
                        key[0] == principal.tenant_id
                        and key[1] == record.domain_id
                        and key[2] != record.space_id
                        and existing.is_active
                    ):
                        self._embedding_spaces[key] = dataclasses.replace(
                            existing,
                            is_active=False,
                            status="retired" if existing.status == "active" else existing.status,
                        )
            effective_status: Any = "active" if record.is_active else record.status
            stored = dataclasses.replace(record, status=effective_status)
            self._embedding_spaces[(principal.tenant_id, record.domain_id, record.space_id)] = (
                stored
            )
            return stored

    def get_active_embedding_space(
        self,
        principal: PrincipalContext,
        domain_id: str,
    ) -> EmbeddingSpaceRecord | None:
        with self._lock:
            for (tid, did, _), rec in self._embedding_spaces.items():
                if tid == principal.tenant_id and did == domain_id and rec.is_active:
                    return rec
            return None

    def list_embedding_spaces(
        self,
        principal: PrincipalContext,
        domain_id: str,
    ) -> list[EmbeddingSpaceRecord]:
        with self._lock:
            items = [
                rec
                for (tid, did, _), rec in self._embedding_spaces.items()
                if tid == principal.tenant_id and did == domain_id
            ]
        items.sort(key=lambda r: (not r.is_active, r.space_id))
        return items

    def stage_shadow_embeddings(
        self,
        principal: PrincipalContext,
        domain_id: str,
        shadow_space_id: str,
        *,
        symbol_embeddings: Mapping[UUID, Sequence[float]] | None = None,
        chunk_embeddings: Mapping[UUID, Sequence[float]] | None = None,
    ) -> dict[str, int]:
        with self._lock:
            space = self._embedding_spaces.get((principal.tenant_id, domain_id, shadow_space_id))
            if space is None:
                raise KeyError(
                    f"Shadow embedding_space '{shadow_space_id}' not registered for '{domain_id}'"
                )
            syms_staged = 0
            if symbol_embeddings:
                for sym_id, vec in symbol_embeddings.items():
                    checked = validate_non_zero_embedding(vec, expected_dim=space.dimensions)
                    sym = self._symbols.get(sym_id)
                    if (
                        sym is not None
                        and sym.tenant_id == principal.tenant_id
                        and sym.domain_id == domain_id
                    ):
                        self._shadow_symbol_vecs[(principal.tenant_id, sym_id, shadow_space_id)] = (
                            checked
                        )
                        syms_staged += 1
            chks_staged = 0
            if chunk_embeddings:
                for chk_id, vec in chunk_embeddings.items():
                    checked = validate_non_zero_embedding(vec, expected_dim=space.dimensions)
                    chk = self._docs.get(chk_id)
                    if (
                        chk is not None
                        and chk.tenant_id == principal.tenant_id
                        and chk.domain_id == domain_id
                    ):
                        self._shadow_doc_vecs[(principal.tenant_id, chk_id, shadow_space_id)] = (
                            checked
                        )
                        chks_staged += 1
            return {"symbols_staged": syms_staged, "chunks_staged": chks_staged}

    def reindex_domain_shadow_space(
        self,
        principal: PrincipalContext,
        domain_id: str,
        shadow_space: EmbeddingSpaceRecord,
        embedder: Any,
    ) -> dict[str, int]:
        staged_rec = dataclasses.replace(shadow_space, is_active=False, status="shadow")
        self.register_embedding_space(principal, staged_rec)
        syms = self.list_visible_ast_symbols(principal, domain_id)
        sym_map: dict[UUID, Sequence[float]] = {}
        if syms:
            sym_vecs = embedder.embed_batch(
                [
                    f"{s.qualified_name} {s.signature_skeleton} {s.contextual_summary}".strip()
                    for s in syms
                ],
                is_query=False,
            )
            for s, v in zip(syms, sym_vecs, strict=True):
                sym_map[s.symbol_id] = v
        chks = self.list_visible_doc_chunks(principal, domain_id)
        chk_map: dict[UUID, Sequence[float]] = {}
        if chks:
            chk_vecs = embedder.embed_batch(
                [
                    (
                        f"{c.doc_title} {c.section_heading} "
                        f"{c.contextual_header} {c.verbatim_content}"
                    ).strip()
                    for c in chks
                ],
                is_query=False,
            )
            for c, v in zip(chks, chk_vecs, strict=True):
                chk_map[c.chunk_id] = v
        return self.stage_shadow_embeddings(
            principal,
            domain_id,
            shadow_space.space_id,
            symbol_embeddings=sym_map,
            chunk_embeddings=chk_map,
        )

    def activate_embedding_space(
        self,
        principal: PrincipalContext,
        domain_id: str,
        target_space_id: str,
        *,
        require_complete_shadow: bool = True,
    ) -> EmbeddingSpaceRecord:
        space_key = (principal.tenant_id, domain_id, target_space_id)
        with self._lock:
            if space_key not in self._embedding_spaces:
                raise KeyError(
                    f"Target embedding_space '{target_space_id}' not registered for '{domain_id}'"
                )
            if require_complete_shadow:
                for sym in self._symbols.values():
                    if (
                        sym.tenant_id == principal.tenant_id
                        and sym.domain_id == domain_id
                        and _scopes_intersect(sym.allowed_scopes, principal.allowed_scopes)
                        and sym.embedding_space_id != target_space_id
                        and (principal.tenant_id, sym.symbol_id, target_space_id)
                        not in self._shadow_symbol_vecs
                    ):
                        raise RuntimeError(
                            f"Symbol {sym.symbol_id} missing staged shadow "
                            f"embedding for '{target_space_id}'"
                        )
                for chk in self._docs.values():
                    if (
                        chk.tenant_id == principal.tenant_id
                        and chk.domain_id == domain_id
                        and _scopes_intersect(chk.allowed_scopes, principal.allowed_scopes)
                        and not chk.is_stale
                        and chk.embedding_space_id != target_space_id
                        and (principal.tenant_id, chk.chunk_id, target_space_id)
                        not in self._shadow_doc_vecs
                    ):
                        raise RuntimeError(
                            f"Chunk {chk.chunk_id} missing staged shadow "
                            f"embedding for '{target_space_id}'"
                        )
            for sid, sym in list(self._symbols.items()):
                sh_vec = self._shadow_symbol_vecs.pop(
                    (principal.tenant_id, sid, target_space_id), None
                )
                if sh_vec is not None:
                    self._symbols[sid] = dataclasses.replace(
                        sym,
                        embedding=sh_vec,
                        embedding_space_id=target_space_id,
                    )
            for cid, chk in list(self._docs.items()):
                sh_vec = self._shadow_doc_vecs.pop(
                    (principal.tenant_id, cid, target_space_id), None
                )
                if sh_vec is not None:
                    self._docs[cid] = dataclasses.replace(
                        chk,
                        embedding=sh_vec,
                        embedding_space_id=target_space_id,
                    )
            for key, existing in list(self._embedding_spaces.items()):
                if key[0] == principal.tenant_id and key[1] == domain_id:
                    if key[2] == target_space_id:
                        self._embedding_spaces[key] = dataclasses.replace(
                            existing,
                            is_active=True,
                            status="active",
                        )
                    elif existing.is_active:
                        self._embedding_spaces[key] = dataclasses.replace(
                            existing,
                            is_active=False,
                            status="retired",
                        )
            return self._embedding_spaces[space_key]

    def insert_confluence_page(
        self, principal: PrincipalContext, record: ConfluencePageRecord
    ) -> None:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on insert_confluence_page")
        if not _connector_ingest_or_scopes_intersect(
            record.allowed_scopes, principal.allowed_scopes
        ):
            raise PermissionError("Principal lacks scope to insert confluence_page row")
        self._deleted_confluence.discard(
            (record.tenant_id, record.domain_id, record.confluence_page_id)
        )
        # Upsert: purge existing record for (tenant_id, domain_id, confluence_page_id)
        for old_id, p in list(self._confluence.items()):
            if (
                p.tenant_id == record.tenant_id
                and p.domain_id == record.domain_id
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
        if (
            not include_deleted
            and (principal.tenant_id, domain_id, confluence_page_id) in self._deleted_confluence
        ):
            return None
        for p in self._confluence.values():
            if (
                p.tenant_id == principal.tenant_id
                and p.domain_id == domain_id
                and p.confluence_page_id == confluence_page_id
                and _connector_ingest_or_scopes_intersect(
                    p.allowed_scopes, principal.allowed_scopes
                )
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
            if p.tenant_id == principal.tenant_id
            and p.domain_id == domain_id
            and (space_key is None or p.space_key == space_key)
            and (
                include_deleted
                or (p.tenant_id, p.domain_id, p.confluence_page_id) not in self._deleted_confluence
            )
            and _connector_ingest_or_scopes_intersect(p.allowed_scopes, principal.allowed_scopes)
        ]

    def tombstone_confluence_pages(
        self,
        domain_id: str,
        confluence_page_ids: list[str],
        reason: str = "Page deleted or archived in Confluence",
        status: str = "deleted",
        *,
        tenant_id: UUID = DEFAULT_TENANT_ID,
    ) -> tuple[int, int]:
        """Soft-deletes Confluence pages and marks their active doc_chunks stale in memory."""
        import dataclasses

        del status
        clean_ids = {pid for pid in confluence_page_ids if pid}
        if not clean_ids:
            return (0, 0)

        pages_tombstoned = 0
        for pid in clean_ids:
            key = (tenant_id, domain_id, pid)
            had_page = any(
                p.tenant_id == tenant_id
                and p.domain_id == domain_id
                and p.confluence_page_id == pid
                for p in self._confluence.values()
            )
            if key not in self._deleted_confluence:
                self._deleted_confluence.add(key)
                if had_page:
                    pages_tombstoned += 1

        target_doc_ids = {f"confluence:{pid}" for pid in clean_ids}
        chunks_flagged = 0
        for cid, doc in list(self._docs.items()):
            if (
                doc.tenant_id == tenant_id
                and doc.domain_id == domain_id
                and doc.doc_id in target_doc_ids
                and not doc.is_stale
            ):
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
        *,
        tenant_id: UUID = DEFAULT_TENANT_ID,
    ) -> int:
        """Synchronizes allowed_scopes across in-memory confluence_pages and active doc_chunks."""
        import dataclasses

        scopes_copy = list(allowed_scopes)
        for pid_uuid, page_rec in list(self._confluence.items()):
            if (
                page_rec.tenant_id == tenant_id
                and page_rec.domain_id == domain_id
                and page_rec.confluence_page_id == confluence_page_id
            ):
                self._confluence[pid_uuid] = dataclasses.replace(
                    page_rec, allowed_scopes=scopes_copy
                )

        target_doc_id = f"confluence:{confluence_page_id}"
        updated_chunks = 0
        for cid, doc in list(self._docs.items()):
            if (
                doc.tenant_id == tenant_id
                and doc.domain_id == domain_id
                and doc.doc_id == target_doc_id
                and not doc.is_stale
            ):
                self._docs[cid] = dataclasses.replace(doc, allowed_scopes=scopes_copy)
                updated_chunks += 1
        return updated_chunks

    def insert_jira_issue(self, principal: PrincipalContext, record: JiraIssueRecord) -> None:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on insert_jira_issue")
        if not _connector_ingest_or_scopes_intersect(
            record.allowed_scopes, principal.allowed_scopes
        ):
            raise PermissionError("Principal lacks scope to insert jira_issue row")
        # Upsert: purge existing record for (tenant_id, domain_id, issue_key) to mirror ON CONFLICT
        for old_id, i in list(self._jira.items()):
            if (
                i.tenant_id == record.tenant_id
                and i.domain_id == record.domain_id
                and i.issue_key == record.issue_key
            ):
                self._jira.pop(old_id, None)
        self._jira[record.issue_id] = record

    def get_jira_issue(
        self, principal: PrincipalContext, domain_id: str, issue_key: str
    ) -> JiraIssueRecord | None:
        for i in self._jira.values():
            if (
                i.tenant_id == principal.tenant_id
                and i.domain_id == domain_id
                and i.issue_key == issue_key
                and _connector_ingest_or_scopes_intersect(
                    i.allowed_scopes, principal.allowed_scopes
                )
            ):
                return i
        return None

    def list_jira_issues(
        self, principal: PrincipalContext, domain_id: str, project_key: str | None = None
    ) -> list[JiraIssueRecord]:
        return [
            i
            for i in self._jira.values()
            if i.tenant_id == principal.tenant_id
            and i.domain_id == domain_id
            and (project_key is None or i.project_key == project_key)
            and _connector_ingest_or_scopes_intersect(i.allowed_scopes, principal.allowed_scopes)
        ]

    def insert_doc_chunk(self, principal: PrincipalContext, record: DocChunkRecord) -> None:
        import dataclasses

        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on insert_doc_chunk")
        if not _connector_ingest_or_scopes_intersect(
            record.allowed_scopes, principal.allowed_scopes
        ):
            raise PermissionError("Principal lacks scope to insert doc_chunk row")
        normalized = dataclasses.replace(
            record,
            market_scope=_parse_market_scope(_serialize_market_scope(record.market_scope)),
        )
        self._docs[normalized.chunk_id] = normalized

    def insert_doc_chunks_batch(
        self, principal: PrincipalContext, records: list[DocChunkRecord]
    ) -> int:
        for record in records:
            self.insert_doc_chunk(principal, record)
        return len(records)

    def list_visible_doc_chunks(
        self,
        principal: PrincipalContext,
        domain_id: str,
        market_scope: DocumentScope | str | None = None,
        include_stale: bool = False,
    ) -> list[DocChunkRecord]:
        target_mkt = _serialize_market_scope(market_scope)
        return [
            d
            for d in self._docs.values()
            if d.tenant_id == principal.tenant_id
            and d.domain_id == domain_id
            and (include_stale or not d.is_stale)
            and (target_mkt is None or _serialize_market_scope(d.market_scope) == target_mkt)
            and _connector_ingest_or_scopes_intersect(d.allowed_scopes, principal.allowed_scopes)
        ]

    def search_doc_chunks_hybrid_rrf(
        self,
        principal: PrincipalContext,
        domain_id: str,
        query_text: str,
        query_embedding: list[float],
        market_scope: DocumentScope | str | None = None,
        top_k: int = 5,
        include_stale: bool = False,
        min_calibrated_score: float = 0.0,
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

        # 1. Lexical BM25 rank over (title + market aliases + header + heading + content)
        corpus_texts = [
            (
                f"{d.doc_title} {_market_scope_aliases(d.market_scope, domain_id=domain_id)} "
                f"{d.contextual_header} {d.section_heading} {d.verbatim_content}"
            )
            for d in visible
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
            _trigram_similarity(
                f"{d.doc_title} {_market_scope_aliases(d.market_scope, domain_id=domain_id)} "
                f"{d.section_heading} {d.verbatim_content}",
                query_text,
            )
            * d.authority_weight
            for d in visible
        ]
        trgm_order = sorted(range(len(visible)), key=lambda i: trgm_scores[i], reverse=True)
        trgm_ranks = {visible[idx].chunk_id: rank + 1 for rank, idx in enumerate(trgm_order)}

        domain_cnt = len(visible)
        scored_hits: list[tuple[float, float, float, CalibratedRankedSearchHit]] = []
        for idx, d in enumerate(visible):
            r_trgm = trgm_ranks[d.chunk_id]
            r_bm25 = bm25_ranks[d.chunk_id]
            r_vec = vec_ranks[d.chunk_id]
            base_rrf = compute_normalized_rrf([r_trgm, r_bm25, r_vec], num_modalities=3, k=60)
            weighted_score = _apply_authority_score(base_rrf, d.authority_weight)
            calibrated_score, lexical_cov, is_boiler = _compute_calibrated_relevance(
                query_text=query_text,
                title=d.doc_title,
                section=d.section_heading,
                content=d.verbatim_content,
                cosine_similarity=float(vec_scores[idx]),
                bm25_score=float(bm25_scores[idx]),
                trigram_score=float(trgm_scores[idx]),
                raw_rrf_score=base_rrf,
                authority_weight=d.authority_weight,
                market_scope=d.market_scope,
                domain_id=domain_id,
            )
            if is_boiler and domain_cnt > 50:
                continue
            if min_calibrated_score > 0.0 and calibrated_score < min_calibrated_score:
                continue
            hit = CalibratedRankedSearchHit(
                item_id=d.chunk_id,
                qualified_name_or_title=f"{d.doc_title} > {d.section_heading}",
                normalized_rrf_score=weighted_score,
                trigram_rank=r_trgm,
                bm25_rank=r_bm25,
                vector_rank=r_vec,
                cosine_similarity=round(float(vec_scores[idx]), 6),
                bm25_score=round(float(bm25_scores[idx]), 6),
                trigram_score=round(float(trgm_scores[idx]), 6),
                raw_rrf_score=round(base_rrf, 6),
                calibrated_score=calibrated_score,
                domain_id=domain_id,
                lexical_coverage=lexical_cov,
                is_boilerplate=is_boiler,
            )
            rank_key = calibrated_score if domain_cnt > 50 else weighted_score
            scored_hits.append((rank_key, weighted_score, d.authority_weight, hit))

        scored_hits.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [item[3] for item in scored_hits[:top_k]]

    def upsert_connector_watermark(
        self,
        principal: PrincipalContext,
        record: ConnectorWatermarkRecord,
    ) -> None:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on upsert_connector_watermark")
        with self._lock:
            key = (record.tenant_id, record.domain_id, record.source_id, record.container_key)
            prev = self._watermarks.get(key)
            now = datetime.now(UTC)
            self._watermarks[key] = ConnectorWatermarkRecord(
                tenant_id=record.tenant_id,
                domain_id=record.domain_id,
                source_id=record.source_id,
                container_key=record.container_key,
                watermark_cursor=record.watermark_cursor
                or (prev.watermark_cursor if prev is not None else ""),
                sync_tier=record.sync_tier,
                items_synced=record.items_synced,
                last_webhook_at=record.last_webhook_at
                or (prev.last_webhook_at if prev is not None else None),
                last_watermark_sync_at=record.last_watermark_sync_at
                or (prev.last_watermark_sync_at if prev is not None else None),
                last_reconciliation_at=record.last_reconciliation_at
                or (prev.last_reconciliation_at if prev is not None else None),
                metadata_json=dict(record.metadata_json),
                updated_at=record.updated_at or now,
            )

    def reset_connector_watermark(
        self,
        principal: PrincipalContext,
        record: ConnectorWatermarkRecord,
    ) -> None:
        """Unconditionally rewinds watermark_cursor (including to empty) after PITR."""
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on reset_connector_watermark")
        with self._lock:
            key = (record.tenant_id, record.domain_id, record.source_id, record.container_key)
            prev = self._watermarks.get(key)
            now = datetime.now(UTC)
            self._watermarks[key] = ConnectorWatermarkRecord(
                tenant_id=record.tenant_id,
                domain_id=record.domain_id,
                source_id=record.source_id,
                container_key=record.container_key,
                watermark_cursor=record.watermark_cursor,
                sync_tier=record.sync_tier,
                items_synced=record.items_synced,
                last_webhook_at=record.last_webhook_at
                or (prev.last_webhook_at if prev is not None else None),
                last_watermark_sync_at=record.last_watermark_sync_at
                or (prev.last_watermark_sync_at if prev is not None else None),
                last_reconciliation_at=record.last_reconciliation_at or now,
                metadata_json=dict(record.metadata_json),
                updated_at=now,
            )

    def get_connector_watermark(
        self,
        principal: PrincipalContext,
        domain_id: str,
        source_id: str,
        container_key: str,
    ) -> ConnectorWatermarkRecord | None:
        with self._lock:
            return self._watermarks.get((principal.tenant_id, domain_id, source_id, container_key))

    def list_connector_watermarks(
        self,
        principal: PrincipalContext,
        domain_id: str,
        source_id: str | None = None,
    ) -> list[ConnectorWatermarkRecord]:
        with self._lock:
            items = [
                wm
                for wm in self._watermarks.values()
                if wm.tenant_id == principal.tenant_id
                and wm.domain_id == domain_id
                and (source_id is None or wm.source_id == source_id)
            ]
            items.sort(key=lambda w: (w.source_id, w.container_key))
            return items
