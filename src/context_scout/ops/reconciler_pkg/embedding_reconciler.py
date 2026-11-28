"""Batched embedding watermark and halfvec reconciliation for PG17 and in-memory repos."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from context_scout.db.repositories import (
    InMemoryRLSRepository,
    PostgresRLSRepository,
    _format_db_vector_and_halfvec,
)
from context_scout.db.schema import (
    EmbeddingSpaceRecord,
)
from context_scout.ops.reconciler_pkg.base import _StateReconcilerBase
from context_scout.ops.reconciler_pkg.models import (
    EmbeddingWatermarkDrift,
)
from context_scout.retrieval.embedder import (
    DEFAULT_EMBEDDING_SPACE_ID,
    DEFAULT_GOOGLE_EMBEDDING_MODEL,
    EmbeddingProviderSPI,
    get_embedding_provider,
    validate_non_zero_embedding,
)
from context_scout.schemas.contracts import PrincipalContext

logger = logging.getLogger(__name__)


class _EmbeddingReconcilerMixin(_StateReconcilerBase):
    """Batched embedding watermark and halfvec reconciliation for PG17 and in-memory repos."""

    def _resolve_embedder_for_space(self, space: EmbeddingSpaceRecord) -> EmbeddingProviderSPI:
        if self.embedder is not None:
            return self.embedder
        return get_embedding_provider(
            settings=self._settings,
            provider_backend=space.provider,
            model_id=space.model_id,
            dimensions=space.dimensions,
        )

    def _reconcile_embeddings_pg(
        self,
        conn: Any,
        *,
        principal: PrincipalContext,
        domain_id: str,
        warnings: list[str],
    ) -> EmbeddingWatermarkDrift:
        pg_repo = self.repository
        assert isinstance(pg_repo, PostgresRLSRepository)

        spaces = pg_repo.list_embedding_spaces(conn, domain_id)
        registered_ids = [s.space_id for s in spaces]
        active_space = pg_repo.get_active_embedding_space(conn, domain_id)

        # 1. Clear orphaned shadow vectors pointing to embedding_spaces absent after PITR
        orphaned_cleared = 0
        for tbl in ("ast_symbols", "doc_chunks"):
            if registered_ids:
                res = conn.execute(
                    f"""
                    UPDATE {tbl}
                    SET shadow_embedding_space_id = NULL,
                        shadow_embedding_halfvec = NULL
                    WHERE domain_id = %s
                      AND shadow_embedding_space_id IS NOT NULL
                      AND NOT (shadow_embedding_space_id = ANY(%s::text[]));
                    """,
                    (domain_id, registered_ids),
                )
            else:
                res = conn.execute(
                    f"""
                    UPDATE {tbl}
                    SET shadow_embedding_space_id = NULL,
                        shadow_embedding_halfvec = NULL
                    WHERE domain_id = %s
                      AND shadow_embedding_space_id IS NOT NULL;
                    """,
                    (domain_id,),
                )
            orphaned_cleared += max(int(res.rowcount or 0), 0)

        # Ensure an active space exists if vector rows exist
        if active_space is None:
            sym_cnt_row = conn.execute(
                "SELECT COUNT(*) FROM ast_symbols WHERE domain_id = %s;", (domain_id,)
            ).fetchone()
            chk_cnt_row = conn.execute(
                "SELECT COUNT(*) FROM doc_chunks WHERE domain_id = %s AND NOT is_stale;",
                (domain_id,),
            ).fetchone()
            has_rows = int(str((sym_cnt_row or (0,))[0])) + int(str((chk_cnt_row or (0,))[0])) > 0
            if not has_rows:
                return EmbeddingWatermarkDrift(
                    active_space_id=None,
                    orphaned_shadow_rows_cleared=orphaned_cleared,
                )
            if spaces:
                active_space = pg_repo.register_embedding_space(
                    conn,
                    EmbeddingSpaceRecord(
                        tenant_id=principal.tenant_id,
                        domain_id=domain_id,
                        space_id=spaces[0].space_id,
                        model_id=spaces[0].model_id,
                        provider=spaces[0].provider,
                        dimensions=spaces[0].dimensions,
                        quantization=spaces[0].quantization,
                        status="active",
                        is_active=True,
                        index_operator_class=spaces[0].index_operator_class,
                    ),
                )
            else:
                default_dim = (
                    self.embedder.dimensions
                    if self.embedder is not None
                    else self._settings.embedding_dimensions
                )
                default_model = (
                    self.embedder.model_id
                    if self.embedder is not None
                    else DEFAULT_GOOGLE_EMBEDDING_MODEL
                )
                active_space = pg_repo.register_embedding_space(
                    conn,
                    EmbeddingSpaceRecord(
                        tenant_id=principal.tenant_id,
                        domain_id=domain_id,
                        space_id=DEFAULT_EMBEDDING_SPACE_ID,
                        model_id=default_model,
                        dimensions=default_dim,
                        status="active",
                        is_active=True,
                    ),
                )

        # 2. Detect drifted rows (embedding_space_id mismatch, NULL halfvec, or wrong dims)
        sym_rows = conn.execute(
            """
            SELECT symbol_id, qualified_name, signature_skeleton, contextual_summary
            FROM ast_symbols
            WHERE domain_id = %s
              AND (
                  COALESCE(embedding_space_id, '') <> %s
                  OR embedding_halfvec IS NULL
                  OR vector_dims(embedding_halfvec) <> %s
              )
            ORDER BY symbol_id;
            """,
            (domain_id, active_space.space_id, active_space.dimensions),
        ).fetchall()

        chk_rows = conn.execute(
            """
            SELECT chunk_id, doc_title, section_heading, contextual_header, verbatim_content
            FROM doc_chunks
            WHERE domain_id = %s
              AND NOT is_stale
              AND (
                  COALESCE(embedding_space_id, '') <> %s
                  OR embedding_halfvec IS NULL
                  OR vector_dims(embedding_halfvec) <> %s
              )
            ORDER BY chunk_id;
            """,
            (domain_id, active_space.space_id, active_space.dimensions),
        ).fetchall()

        if not sym_rows and not chk_rows:
            return EmbeddingWatermarkDrift(
                active_space_id=active_space.space_id,
                orphaned_shadow_rows_cleared=orphaned_cleared,
            )

        drifted_syms = len(sym_rows)
        drifted_chks = len(chk_rows)
        sym_reembedded = 0
        chk_reembedded = 0
        unhealed = 0

        try:
            embedder = self._resolve_embedder_for_space(active_space)
            bs = self.reconcile_batch_size
            if sym_rows:
                for start in range(0, len(sym_rows), bs):
                    batch_rows = sym_rows[start : start + bs]
                    sym_texts = [f"{r[1]} {r[2]} {r[3]}".strip() for r in batch_rows]
                    sym_vecs = embedder.embed_batch(sym_texts, is_query=False)
                    if len(sym_vecs) != len(batch_rows):
                        raise RuntimeError(
                            f"Embedder returned {len(sym_vecs)} vectors for "
                            f"{len(batch_rows)} symbols."
                        )
                    sym_updates: list[tuple[str, str, str, str, UUID]] = []
                    for row, vec in zip(batch_rows, sym_vecs, strict=True):
                        checked = validate_non_zero_embedding(
                            vec, expected_dim=active_space.dimensions
                        )
                        vec768_lit, hv_lit = _format_db_vector_and_halfvec(checked)
                        sym_updates.append(
                            (
                                vec768_lit,
                                active_space.space_id,
                                hv_lit,
                                domain_id,
                                UUID(str(row[0])),
                            )
                        )
                    with conn.cursor() as cur:
                        cur.executemany(
                            """
                            UPDATE ast_symbols
                            SET embedding = %s::vector,
                                embedding_space_id = %s,
                                embedding_halfvec = %s::halfvec
                            WHERE domain_id = %s AND symbol_id = %s;
                            """,
                            sym_updates,
                        )
                        sym_reembedded += max(int(cur.rowcount or 0), len(sym_updates))

            if chk_rows:
                for start in range(0, len(chk_rows), bs):
                    batch_rows = chk_rows[start : start + bs]
                    chk_texts = [f"{r[1]} {r[2]} {r[3]} {r[4]}".strip() for r in batch_rows]
                    chk_vecs = embedder.embed_batch(chk_texts, is_query=False)
                    if len(chk_vecs) != len(batch_rows):
                        raise RuntimeError(
                            f"Embedder returned {len(chk_vecs)} vectors for "
                            f"{len(batch_rows)} chunks."
                        )
                    chk_updates: list[tuple[str, str, str, str, UUID]] = []
                    for row, vec in zip(batch_rows, chk_vecs, strict=True):
                        checked = validate_non_zero_embedding(
                            vec, expected_dim=active_space.dimensions
                        )
                        vec768_lit, hv_lit = _format_db_vector_and_halfvec(checked)
                        chk_updates.append(
                            (
                                vec768_lit,
                                active_space.space_id,
                                hv_lit,
                                domain_id,
                                UUID(str(row[0])),
                            )
                        )
                    with conn.cursor() as cur:
                        cur.executemany(
                            """
                            UPDATE doc_chunks
                            SET embedding = %s::vector,
                                embedding_space_id = %s,
                                embedding_halfvec = %s::halfvec
                            WHERE domain_id = %s AND chunk_id = %s;
                            """,
                            chk_updates,
                        )
                        chk_reembedded += max(int(cur.rowcount or 0), len(chk_updates))
        except Exception as exc:
            unhealed = (drifted_syms - sym_reembedded) + (drifted_chks - chk_reembedded)
            warnings.append(
                f"Fail-closed embedding reconciliation deferred ({type(exc).__name__}): {exc}"
            )
            if self.task_queue is not None:
                self.task_queue.enqueue_task(
                    domain_id=domain_id,
                    event_type="reconcile_sweep",
                    payload={
                        "reason": "pitr_reembed_domain",
                        "domain_id": domain_id,
                        "space_id": active_space.space_id,
                        "unhealed_rows": unhealed,
                    },
                    priority=20,
                    tenant_id=principal.tenant_id,
                )

        return EmbeddingWatermarkDrift(
            active_space_id=active_space.space_id,
            orphaned_shadow_rows_cleared=orphaned_cleared,
            drifted_symbol_rows_found=drifted_syms,
            drifted_chunk_rows_found=drifted_chks,
            symbol_rows_reembedded=sym_reembedded,
            chunk_rows_reembedded=chk_reembedded,
            unhealed_rows=unhealed,
        )

    def _reconcile_embeddings_inmemory(
        self,
        *,
        principal: PrincipalContext,
        domain_id: str,
        warnings: list[str],
    ) -> EmbeddingWatermarkDrift:
        import dataclasses

        mem_repo = self.repository
        assert isinstance(mem_repo, InMemoryRLSRepository)

        spaces = mem_repo.list_embedding_spaces(principal, domain_id)
        registered_ids = {s.space_id for s in spaces}
        active_space = mem_repo.get_active_embedding_space(principal, domain_id)

        orphaned_cleared = 0
        with mem_repo._lock:
            domain_sym_ids = {
                s.symbol_id
                for s in mem_repo._symbols.values()
                if s.tenant_id == principal.tenant_id and s.domain_id == domain_id
            }
            domain_doc_ids = {
                d.chunk_id
                for d in mem_repo._docs.values()
                if d.tenant_id == principal.tenant_id and d.domain_id == domain_id
            }
            for key in list(mem_repo._shadow_symbol_vecs):
                tid, sid, sp_id = key
                if (
                    tid == principal.tenant_id
                    and sid in domain_sym_ids
                    and sp_id not in registered_ids
                ):
                    mem_repo._shadow_symbol_vecs.pop(key, None)
                    orphaned_cleared += 1
            for key in list(mem_repo._shadow_doc_vecs):
                tid, cid, sp_id = key
                if (
                    tid == principal.tenant_id
                    and cid in domain_doc_ids
                    and sp_id not in registered_ids
                ):
                    mem_repo._shadow_doc_vecs.pop(key, None)
                    orphaned_cleared += 1

        syms = mem_repo.list_visible_ast_symbols(principal, domain_id)
        chks = [c for c in mem_repo.list_visible_doc_chunks(principal, domain_id) if not c.is_stale]
        if active_space is None:
            if not syms and not chks:
                return EmbeddingWatermarkDrift(
                    active_space_id=None,
                    orphaned_shadow_rows_cleared=orphaned_cleared,
                )
            if spaces:
                active_space = mem_repo.register_embedding_space(
                    principal,
                    dataclasses.replace(spaces[0], status="active", is_active=True),
                )
            else:
                active_space = mem_repo.register_embedding_space(
                    principal,
                    EmbeddingSpaceRecord(
                        tenant_id=principal.tenant_id,
                        domain_id=domain_id,
                        space_id=DEFAULT_EMBEDDING_SPACE_ID,
                        model_id=(
                            self.embedder.model_id
                            if self.embedder is not None
                            else DEFAULT_GOOGLE_EMBEDDING_MODEL
                        ),
                        dimensions=(
                            self.embedder.dimensions
                            if self.embedder is not None
                            else self._settings.embedding_dimensions
                        ),
                        status="active",
                        is_active=True,
                    ),
                )

        def _is_drifted_vec(vec: Sequence[float], space: EmbeddingSpaceRecord) -> bool:
            return len(vec) != space.dimensions or sum(float(x) * float(x) for x in vec) <= 1e-9

        drifted_sym_list = [
            s
            for s in syms
            if s.embedding_space_id != active_space.space_id
            or _is_drifted_vec(s.embedding, active_space)
        ]
        drifted_chk_list = [
            c
            for c in chks
            if c.embedding_space_id != active_space.space_id
            or _is_drifted_vec(c.embedding, active_space)
        ]
        if not drifted_sym_list and not drifted_chk_list:
            return EmbeddingWatermarkDrift(
                active_space_id=active_space.space_id,
                orphaned_shadow_rows_cleared=orphaned_cleared,
            )

        sym_reembedded = 0
        chk_reembedded = 0
        unhealed = 0
        try:
            embedder = self._resolve_embedder_for_space(active_space)
            if drifted_sym_list:
                sym_vecs = embedder.embed_batch(
                    [
                        f"{s.qualified_name} {s.signature_skeleton} {s.contextual_summary}".strip()
                        for s in drifted_sym_list
                    ],
                    is_query=False,
                )
                with mem_repo._lock:
                    for sym, vec in zip(drifted_sym_list, sym_vecs, strict=True):
                        checked = validate_non_zero_embedding(
                            vec, expected_dim=active_space.dimensions
                        )
                        mem_repo._symbols[sym.symbol_id] = dataclasses.replace(
                            sym,
                            embedding=checked,
                            embedding_space_id=active_space.space_id,
                        )
                        sym_reembedded += 1
            if drifted_chk_list:
                chk_vecs = embedder.embed_batch(
                    [
                        (
                            f"{c.doc_title} {c.section_heading} "
                            f"{c.contextual_header} {c.verbatim_content}"
                        ).strip()
                        for c in drifted_chk_list
                    ],
                    is_query=False,
                )
                with mem_repo._lock:
                    for chk, vec in zip(drifted_chk_list, chk_vecs, strict=True):
                        checked = validate_non_zero_embedding(
                            vec, expected_dim=active_space.dimensions
                        )
                        mem_repo._docs[chk.chunk_id] = dataclasses.replace(
                            chk,
                            embedding=checked,
                            embedding_space_id=active_space.space_id,
                        )
                        chk_reembedded += 1
        except Exception as exc:
            unhealed = (len(drifted_sym_list) - sym_reembedded) + (
                len(drifted_chk_list) - chk_reembedded
            )
            warnings.append(
                f"Fail-closed embedding reconciliation deferred ({type(exc).__name__}): {exc}"
            )
            if self.task_queue is not None:
                self.task_queue.enqueue_task(
                    domain_id=domain_id,
                    event_type="reconcile_sweep",
                    payload={
                        "reason": "pitr_reembed_domain",
                        "domain_id": domain_id,
                        "space_id": active_space.space_id,
                        "unhealed_rows": unhealed,
                    },
                    priority=20,
                    tenant_id=principal.tenant_id,
                )

        return EmbeddingWatermarkDrift(
            active_space_id=active_space.space_id,
            orphaned_shadow_rows_cleared=orphaned_cleared,
            drifted_symbol_rows_found=len(drifted_sym_list),
            drifted_chunk_rows_found=len(drifted_chk_list),
            symbol_rows_reembedded=sym_reembedded,
            chunk_rows_reembedded=chk_reembedded,
            unhealed_rows=unhealed,
        )
