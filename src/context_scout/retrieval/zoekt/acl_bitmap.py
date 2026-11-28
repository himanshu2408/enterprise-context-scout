"""Multi-tenant Repo-ACL bitmap pre-filtering registry."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from context_scout.schemas.contracts import PrincipalContext


@dataclass(frozen=True)
class RepoACLResult:
    """Resolved repository authorization bitmap and IDs for a PrincipalContext."""

    tenant_id: UUID
    bitmap: int
    repo_ids: tuple[int, ...]
    repo_names: frozenset[str]
    popcount: int


@dataclass
class _RegisteredRepoEntry:
    repo_id: int
    tenant_id: UUID
    domain_id: str
    repo_name: str
    allowed_scopes: frozenset[str]


class RepoACLBitmapRegistry:
    """Thread-safe repository ACL bitmap registry for fail-closed Zoekt pre-filtering."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._next_repo_id = 1
        self._entries_by_key: dict[tuple[UUID, str, str], _RegisteredRepoEntry] = {}
        self._entries_by_id: dict[int, _RegisteredRepoEntry] = {}
        self._tenant_masks: dict[UUID, int] = {}
        self._domain_masks: dict[tuple[UUID, str], int] = {}
        self._scope_masks: dict[tuple[UUID, str], int] = {}

    def _rebuild_masks(self) -> None:
        self._tenant_masks.clear()
        self._domain_masks.clear()
        self._scope_masks.clear()
        for entry in self._entries_by_id.values():
            bit = 1 << entry.repo_id
            self._tenant_masks[entry.tenant_id] = self._tenant_masks.get(entry.tenant_id, 0) | bit
            dom_key = (entry.tenant_id, entry.domain_id)
            self._domain_masks[dom_key] = self._domain_masks.get(dom_key, 0) | bit
            for scope in entry.allowed_scopes:
                s_key = (entry.tenant_id, scope)
                self._scope_masks[s_key] = self._scope_masks.get(s_key, 0) | bit

    def register_repo(
        self,
        tenant_id: UUID,
        domain_id: str,
        repo_name: str,
        allowed_scopes: Sequence[str],
        *,
        repo_id: int | None = None,
    ) -> int:
        clean_scopes = frozenset(s.strip() for s in allowed_scopes if s and s.strip())
        key = (tenant_id, domain_id, repo_name)
        with self._lock:
            existing = self._entries_by_key.get(key)
            if existing is not None:
                if repo_id is not None and repo_id != existing.repo_id:
                    raise ValueError(
                        f"Repository {key!r} already registered with repo_id={existing.repo_id}"
                    )
                existing.allowed_scopes = clean_scopes
                self._rebuild_masks()
                return existing.repo_id

            if repo_id is not None:
                if repo_id <= 0:
                    raise ValueError(f"Zoekt repo_id must be positive, got {repo_id}")
                if repo_id in self._entries_by_id:
                    raise ValueError(f"Duplicate repo_id={repo_id} in RepoACLBitmapRegistry")
                assigned_id = repo_id
                if assigned_id >= self._next_repo_id:
                    self._next_repo_id = assigned_id + 1
            else:
                assigned_id = self._next_repo_id
                self._next_repo_id += 1

            entry = _RegisteredRepoEntry(
                repo_id=assigned_id,
                tenant_id=tenant_id,
                domain_id=domain_id,
                repo_name=repo_name,
                allowed_scopes=clean_scopes,
            )
            self._entries_by_key[key] = entry
            self._entries_by_id[assigned_id] = entry
            self._rebuild_masks()
            return assigned_id

    def update_repo_scopes(
        self,
        tenant_id: UUID,
        domain_id: str,
        repo_name: str,
        allowed_scopes: Sequence[str],
    ) -> int:
        return self.register_repo(tenant_id, domain_id, repo_name, allowed_scopes)

    def remove_repo(self, tenant_id: UUID, domain_id: str, repo_name: str) -> None:
        key = (tenant_id, domain_id, repo_name)
        with self._lock:
            entry = self._entries_by_key.pop(key, None)
            if entry is not None:
                self._entries_by_id.pop(entry.repo_id, None)
                self._rebuild_masks()

    def unregister_repo(self, tenant_id: UUID, domain_id: str, repo_name: str) -> bool:
        """Removes repository from bitmap ACL index and returns True if previously present."""
        key = (tenant_id, domain_id, repo_name)
        with self._lock:
            entry = self._entries_by_key.pop(key, None)
            if entry is not None:
                self._entries_by_id.pop(entry.repo_id, None)
                self._rebuild_masks()
                return True
            return False

    def get_repo_id(self, tenant_id: UUID, domain_id: str, repo_name: str) -> int | None:
        with self._lock:
            entry = self._entries_by_key.get((tenant_id, domain_id, repo_name))
            return entry.repo_id if entry is not None else None

    def compute_authorized_bitmap(
        self,
        principal: PrincipalContext,
        *,
        domain_id: str | None = None,
        repo_names: Sequence[str] | None = None,
    ) -> RepoACLResult:
        clean_principal_scopes = [s.strip() for s in principal.allowed_scopes if s and s.strip()]
        if not clean_principal_scopes:
            return RepoACLResult(
                tenant_id=principal.tenant_id,
                bitmap=0,
                repo_ids=(),
                repo_names=frozenset(),
                popcount=0,
            )

        with self._lock:
            tenant_mask = self._tenant_masks.get(principal.tenant_id, 0)
            if tenant_mask == 0:
                return RepoACLResult(
                    tenant_id=principal.tenant_id,
                    bitmap=0,
                    repo_ids=(),
                    repo_names=frozenset(),
                    popcount=0,
                )

            scope_mask = 0
            for scope in clean_principal_scopes:
                scope_mask |= self._scope_masks.get((principal.tenant_id, scope), 0)

            bitmap = tenant_mask & scope_mask
            if domain_id is not None:
                bitmap &= self._domain_masks.get((principal.tenant_id, domain_id), 0)

            if repo_names is not None:
                requested = {r.strip() for r in repo_names if r and r.strip()}
                filter_mask = 0
                for entry in self._entries_by_id.values():
                    if entry.tenant_id == principal.tenant_id and entry.repo_name in requested:
                        filter_mask |= 1 << entry.repo_id
                bitmap &= filter_mask

            if bitmap == 0:
                return RepoACLResult(
                    tenant_id=principal.tenant_id,
                    bitmap=0,
                    repo_ids=(),
                    repo_names=frozenset(),
                    popcount=0,
                )

            repo_ids: list[int] = []
            names: set[str] = set()
            m = bitmap
            while m:
                lsb = m & -m
                rid = lsb.bit_length() - 1
                m &= m - 1
                entry_opt = self._entries_by_id.get(rid)
                if entry_opt is not None:
                    repo_ids.append(rid)
                    names.add(entry_opt.repo_name)

            return RepoACLResult(
                tenant_id=principal.tenant_id,
                bitmap=bitmap,
                repo_ids=tuple(sorted(repo_ids)),
                repo_names=frozenset(names),
                popcount=len(repo_ids),
            )
