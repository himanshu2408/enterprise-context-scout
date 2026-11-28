"""In-memory RLS repository mixin: ref snapshots, overlay deltas, ownership, side effects."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from context_scout.db.repos.in_memory_base import _InMemoryStateBase
from context_scout.db.repos.types_and_helpers import (
    OwnershipEdgeRecord,
    _scopes_intersect,
)
from context_scout.db.schema import (
    ASTSymbolRecord,
    CodeFileRecord,
    OverlayFileDeltaRecord,
    RefSnapshotRecord,
    ResolvedOverlayFileRecord,
    SideEffectRecord,
)
from context_scout.schemas.contracts import (
    PrincipalContext,
)

logger = logging.getLogger(__name__)


class _InMemoryOverlaysAndEffectsMixin(_InMemoryStateBase):
    def upsert_ref_snapshot(
        self,
        principal: PrincipalContext,
        record: RefSnapshotRecord,
    ) -> RefSnapshotRecord:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on upsert_ref_snapshot")
        if not _scopes_intersect(record.allowed_scopes, principal.allowed_scopes):
            raise PermissionError("Principal lacks scope to upsert ref_snapshot")
        with self._lock:
            existing_id: UUID | None = None
            for sid, existing in self._ref_snapshots.items():
                if (
                    existing.tenant_id == record.tenant_id
                    and existing.domain_id == record.domain_id
                    and existing.repo_name == record.repo_name
                    and existing.ref_name == record.ref_name
                ):
                    existing_id = sid
                    break
            target_id = existing_id if existing_id is not None else record.snapshot_id
            stored = dataclasses.replace(
                record,
                snapshot_id=target_id,
                updated_at=record.updated_at or datetime.now(),
            )
            self._ref_snapshots[target_id] = stored
            return stored

    def get_ref_snapshot(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        snapshot_id: UUID | None = None,
        repo_name: str | None = None,
        ref_name: str | None = None,
        mr_iid: str | None = None,
    ) -> RefSnapshotRecord | None:
        with self._lock:
            candidates = [
                snap
                for snap in self._ref_snapshots.values()
                if snap.tenant_id == principal.tenant_id
                and snap.domain_id == domain_id
                and _scopes_intersect(snap.allowed_scopes, principal.allowed_scopes)
            ]
            if snapshot_id is not None:
                for snap in candidates:
                    if snap.snapshot_id == snapshot_id:
                        return snap
                return None
            if repo_name is not None and ref_name is not None:
                for snap in candidates:
                    if snap.repo_name == repo_name and snap.ref_name == ref_name:
                        return snap
                return None
            if repo_name is not None and mr_iid is not None:
                matching = [
                    snap
                    for snap in candidates
                    if snap.repo_name == repo_name and snap.mr_iid == mr_iid
                ]
                return matching[-1] if matching else None
            raise ValueError("Specify snapshot_id, (repo_name, ref_name), or (repo_name, mr_iid)")

    def list_ref_snapshots(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        repo_name: str | None = None,
        state: str | None = None,
    ) -> list[RefSnapshotRecord]:
        with self._lock:
            items = [
                snap
                for snap in self._ref_snapshots.values()
                if snap.tenant_id == principal.tenant_id
                and snap.domain_id == domain_id
                and (repo_name is None or snap.repo_name == repo_name)
                and (state is None or snap.state == state)
                and _scopes_intersect(snap.allowed_scopes, principal.allowed_scopes)
            ]
            items.sort(key=lambda s: s.ref_name)
            return items

    def delete_ref_snapshot(self, principal: PrincipalContext, snapshot_id: UUID) -> int:
        with self._lock:
            snap = self._ref_snapshots.get(snapshot_id)
            if snap is None:
                return 0
            if snap.tenant_id != principal.tenant_id:
                raise PermissionError("Principal tenant_id mismatch on delete_ref_snapshot")
            if not _scopes_intersect(snap.allowed_scopes, principal.allowed_scopes):
                raise PermissionError("Principal lacks scope to delete ref_snapshot")
            self._ref_snapshots.pop(snapshot_id, None)
            del_keys = [
                k
                for k in self._overlay_deltas
                if k[0] == principal.tenant_id and k[1] == snapshot_id
            ]
            for k in del_keys:
                self._overlay_deltas.pop(k, None)
            return 1

    def upsert_overlay_file_delta(
        self,
        principal: PrincipalContext,
        record: OverlayFileDeltaRecord,
    ) -> OverlayFileDeltaRecord:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on upsert_overlay_file_delta")
        if not _scopes_intersect(record.allowed_scopes, principal.allowed_scopes):
            raise PermissionError("Principal lacks scope to upsert overlay_file_delta")
        with self._lock:
            key = (record.tenant_id, record.snapshot_id, record.file_path)
            existing = self._overlay_deltas.get(key)
            target_id = existing.delta_id if existing is not None else record.delta_id
            stored = dataclasses.replace(record, delta_id=target_id)
            self._overlay_deltas[key] = stored
            return stored

    def list_overlay_file_deltas(
        self,
        principal: PrincipalContext,
        domain_id: str,
        snapshot_id: UUID,
    ) -> list[OverlayFileDeltaRecord]:
        with self._lock:
            items = [
                d
                for d in self._overlay_deltas.values()
                if d.tenant_id == principal.tenant_id
                and d.domain_id == domain_id
                and d.snapshot_id == snapshot_id
                and _scopes_intersect(d.allowed_scopes, principal.allowed_scopes)
            ]
            items.sort(key=lambda d: d.file_path)
            return items

    def _collect_snapshot_chain(
        self,
        principal: PrincipalContext,
        domain_id: str,
        snapshot_id: UUID,
    ) -> list[RefSnapshotRecord]:
        chain: list[RefSnapshotRecord] = []
        seen: set[UUID] = set()
        curr_id: UUID | None = snapshot_id
        while curr_id is not None and curr_id not in seen and len(chain) < 16:
            seen.add(curr_id)
            snap = self.get_ref_snapshot(principal, domain_id, snapshot_id=curr_id)
            if snap is None:
                break
            chain.append(snap)
            curr_id = snap.base_snapshot_id
        return chain

    def _effective_overlay_deltas_inmemory(
        self,
        principal: PrincipalContext,
        domain_id: str,
        snapshot_id: UUID,
    ) -> tuple[RefSnapshotRecord | None, dict[str, OverlayFileDeltaRecord], set[str]]:
        chain = self._collect_snapshot_chain(principal, domain_id, snapshot_id)
        if not chain:
            return None, {}, set()
        target_snap = chain[0]
        effective: dict[str, OverlayFileDeltaRecord] = {}
        renamed_old_paths: set[str] = set()
        for snap in chain:
            for d in self.list_overlay_file_deltas(principal, domain_id, snap.snapshot_id):
                if d.previous_file_path and d.previous_file_path != d.file_path:
                    renamed_old_paths.add(d.previous_file_path)
                if d.file_path not in effective:
                    effective[d.file_path] = d
        return target_snap, effective, renamed_old_paths

    def resolve_overlay_files(
        self,
        principal: PrincipalContext,
        domain_id: str,
        snapshot_id: UUID,
        *,
        file_path: str | None = None,
    ) -> list[ResolvedOverlayFileRecord]:
        with self._lock:
            target_snap, effective_deltas, renamed_old_paths = (
                self._effective_overlay_deltas_inmemory(principal, domain_id, snapshot_id)
            )
            if target_snap is None:
                return []

            base_by_path: dict[str, CodeFileRecord] = {}
            for cf in self.list_visible_code_files(principal, domain_id):
                if cf.repo_name != target_snap.repo_name:
                    continue
                if cf.file_path in renamed_old_paths:
                    continue
                if file_path is not None and cf.file_path != file_path:
                    continue
                base_by_path[cf.file_path] = cf

            if file_path is not None:
                delta_keys = {file_path} if file_path in effective_deltas else set()
            else:
                delta_keys = set(effective_deltas.keys())
            all_paths = set(base_by_path.keys()) | delta_keys
            resolved: list[ResolvedOverlayFileRecord] = []
            for path in sorted(all_paths):
                delta = effective_deltas.get(path)
                base = base_by_path.get(path)
                if delta is not None:
                    if delta.change_type == "deleted":
                        continue
                    resolved.append(
                        ResolvedOverlayFileRecord(
                            domain_id=delta.domain_id,
                            repo_name=delta.repo_name,
                            file_path=delta.file_path,
                            git_commit_sha=target_snap.head_commit_sha,
                            git_blob_sha=delta.overlay_blob_sha or "",
                            language=delta.language,
                            allowed_scopes=list(delta.allowed_scopes),
                            raw_content=delta.raw_content or "",
                            line_count=delta.line_count,
                            source_layer="overlay",
                            snapshot_id=target_snap.snapshot_id,
                            base_file_id=base.file_id if base is not None else None,
                            delta_id=delta.delta_id,
                            change_type=delta.change_type,
                            ranking_weight=delta.ranking_weight,
                            is_generated=delta.is_generated,
                            is_vendored=delta.is_vendored,
                            tenant_id=delta.tenant_id,
                        )
                    )
                elif base is not None:
                    resolved.append(
                        ResolvedOverlayFileRecord(
                            domain_id=base.domain_id,
                            repo_name=base.repo_name,
                            file_path=base.file_path,
                            git_commit_sha=base.git_commit_sha,
                            git_blob_sha=base.git_blob_sha,
                            language=base.language,
                            allowed_scopes=list(base.allowed_scopes),
                            raw_content=base.raw_content,
                            line_count=base.line_count,
                            source_layer="base_snapshot",
                            snapshot_id=target_snap.snapshot_id,
                            base_file_id=base.file_id,
                            delta_id=None,
                            change_type=None,
                            ranking_weight=base.ranking_weight,
                            is_generated=base.is_generated,
                            is_vendored=base.is_vendored,
                            tenant_id=base.tenant_id,
                        )
                    )
            return resolved

    def get_overlay_file(
        self,
        principal: PrincipalContext,
        domain_id: str,
        snapshot_id: UUID,
        file_path: str,
    ) -> ResolvedOverlayFileRecord | None:
        items = self.resolve_overlay_files(principal, domain_id, snapshot_id, file_path=file_path)
        return items[0] if items else None

    def resolve_overlay_symbols(
        self,
        principal: PrincipalContext,
        domain_id: str,
        snapshot_id: UUID,
    ) -> list[ASTSymbolRecord]:
        with self._lock:
            target_snap, effective_deltas, renamed_old_paths = (
                self._effective_overlay_deltas_inmemory(principal, domain_id, snapshot_id)
            )
            if target_snap is None:
                return []

            suppressed_paths = set(renamed_old_paths) | set(effective_deltas.keys())
            for d in effective_deltas.values():
                if d.previous_file_path:
                    suppressed_paths.add(d.previous_file_path)

            visible_files_by_id = {
                cf.file_id: cf
                for cf in self.list_visible_code_files(principal, domain_id)
                if cf.repo_name == target_snap.repo_name
            }
            base_symbols = [
                sym
                for sym in self.list_visible_ast_symbols(principal, domain_id)
                if sym.file_id in visible_files_by_id
                and visible_files_by_id[sym.file_id].file_path not in suppressed_paths
            ]

            overlay_symbols: list[ASTSymbolRecord] = []
            for delta in effective_deltas.values():
                if delta.change_type == "deleted":
                    continue
                for sym_dict in delta.symbols_json:
                    if not isinstance(sym_dict, dict):
                        continue
                    overlay_symbols.append(
                        ASTSymbolRecord(
                            symbol_id=UUID(str(sym_dict["symbol_id"])),
                            file_id=delta.delta_id,
                            domain_id=delta.domain_id,
                            qualified_name=str(sym_dict["qualified_name"]),
                            symbol_kind=str(sym_dict.get("symbol_kind", "function")),
                            start_line=int(sym_dict.get("start_line", 1)),
                            end_line=int(sym_dict.get("end_line", 1)),
                            signature_skeleton=str(sym_dict.get("signature_skeleton", "")),
                            contextual_summary=str(sym_dict.get("contextual_summary", "")),
                            allowed_scopes=list(delta.allowed_scopes),
                            embedding=[],
                            repo_qualified_id=sym_dict.get("repo_qualified_id"),
                            bare_name=sym_dict.get("bare_name"),
                            tenant_id=delta.tenant_id,
                        )
                    )

            combined = base_symbols + overlay_symbols
            combined.sort(key=lambda s: s.qualified_name)
            return combined

    def list_ownership_edges(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        edge_kinds: Sequence[str] = ("OWNED_BY_TEAM", "PART_OF_SYSTEM"),
    ) -> list[OwnershipEdgeRecord]:
        with self._lock:
            allowed_kinds = set(edge_kinds)
            visible_syms = {
                s.symbol_id: s for s in self.list_visible_ast_symbols(principal, domain_id)
            }
            out: list[OwnershipEdgeRecord] = []
            for edge in self.list_visible_ast_edges(principal, domain_id):
                if edge.edge_kind not in allowed_kinds:
                    continue
                src = visible_syms.get(edge.source_symbol_id)
                if src is None:
                    continue
                tgt = (
                    visible_syms.get(edge.target_symbol_id)
                    if edge.target_symbol_id is not None
                    else None
                )
                out.append(
                    OwnershipEdgeRecord(
                        edge_id=edge.edge_id,
                        edge_kind=edge.edge_kind,
                        source_qualified_name=src.qualified_name,
                        source_symbol_kind=src.symbol_kind,
                        target_qualified_name=edge.target_qualified_name,
                        target_symbol_kind=tgt.symbol_kind if tgt is not None else "",
                    )
                )
            out.sort(key=lambda e: (e.source_qualified_name, e.edge_kind, e.target_qualified_name))
            return out

    def upsert_side_effects_batch(
        self,
        principal: PrincipalContext,
        records: Sequence[SideEffectRecord],
    ) -> int:
        with self._lock:
            inserted = 0
            for rec in records:
                if rec.tenant_id != principal.tenant_id:
                    raise PermissionError("Principal tenant_id mismatch on upsert_side_effects")
                if not _scopes_intersect(rec.allowed_scopes, principal.allowed_scopes):
                    raise PermissionError("Principal lacks scope to insert side_effect row")
                nat_key = (
                    rec.tenant_id,
                    rec.domain_id,
                    rec.repo_name,
                    rec.file_path,
                    rec.qualified_symbol,
                    rec.effect_category,
                    rec.target_identifier,
                    rec.start_line,
                )
                for existing_id, existing in list(self._side_effects.items()):
                    if (
                        existing.tenant_id,
                        existing.domain_id,
                        existing.repo_name,
                        existing.file_path,
                        existing.qualified_symbol,
                        existing.effect_category,
                        existing.target_identifier,
                        existing.start_line,
                    ) == nat_key:
                        self._side_effects.pop(existing_id, None)
                self._side_effects[rec.side_effect_id] = rec
                inserted += 1
            return inserted

    def list_visible_side_effects(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        effect_category: str | None = None,
        repo_name: str | None = None,
        file_path: str | None = None,
    ) -> list[SideEffectRecord]:
        with self._lock:
            out = [
                se
                for se in self._side_effects.values()
                if se.tenant_id == principal.tenant_id
                and se.domain_id == domain_id
                and (effect_category is None or se.effect_category == effect_category)
                and (repo_name is None or se.repo_name == repo_name)
                and (file_path is None or se.file_path == file_path)
                and _scopes_intersect(se.allowed_scopes, principal.allowed_scopes)
            ]
            out.sort(key=lambda s: (s.repo_name, s.file_path, s.start_line, s.effect_category))
            return out

    def delete_side_effects_for_file(
        self,
        principal: PrincipalContext,
        domain_id: str,
        repo_name: str,
        file_path: str,
    ) -> int:
        with self._lock:
            del_ids = [
                se_id
                for se_id, se in self._side_effects.items()
                if se.tenant_id == principal.tenant_id
                and se.domain_id == domain_id
                and se.repo_name == repo_name
                and se.file_path == file_path
                and _scopes_intersect(se.allowed_scopes, principal.allowed_scopes)
            ]
            for se_id in del_ids:
                self._side_effects.pop(se_id, None)
            return len(del_ids)

    def delete_side_effects_for_repo(
        self,
        principal: PrincipalContext,
        domain_id: str,
        repo_name: str,
    ) -> int:
        with self._lock:
            del_ids = [
                se_id
                for se_id, se in self._side_effects.items()
                if se.tenant_id == principal.tenant_id
                and se.domain_id == domain_id
                and se.repo_name == repo_name
                and _scopes_intersect(se.allowed_scopes, principal.allowed_scopes)
            ]
            for se_id in del_ids:
                self._side_effects.pop(se_id, None)
            return len(del_ids)
