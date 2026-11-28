"""In-process Zoekt inverted trigram shard engine with MR CoW overlay merging."""

from __future__ import annotations

import base64
import heapq
import logging
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from context_scout.db.schema import OverlayFileDeltaRecord
from context_scout.retrieval.zoekt.acl_bitmap import RepoACLBitmapRegistry
from context_scout.retrieval.zoekt.query_builder import (
    UNCAPPED_ZOEK_DISPLAY_COUNT,
    RE2Pattern,
    ZoektFileMatch,
    ZoektLineMatch,
    ZoektSearchResponse,
    ZoektSearchStats,
    compile_re2,
    extract_required_trigrams,
)
from context_scout.schemas.contracts import PrincipalContext

logger = logging.getLogger(__name__)


def _decode_zoekt_text(raw: Any) -> str:
    """Decodes Zoekt base64-encoded bytes fields or UTF-8 strings."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        return raw
    return str(raw)


def _decode_zoekt_b64_or_text(raw: Any) -> str:
    """Decodes base64-encoded Zoekt payload fields (`FileName`, `Line`, `Before`, `After`)."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    text = str(raw)
    if not text:
        return ""
    try:
        decoded = base64.b64decode(text, validate=True)
        return decoded.decode("utf-8", errors="strict")
    except (ValueError, UnicodeDecodeError):
        return text


def _scan_content_lines(
    content: str,
    compiled_re: RE2Pattern,
    *,
    context_lines: int,
) -> list[ZoektLineMatch]:
    """Scans file content line-by-line with O(n) RE2 pattern and collects line matches."""
    if not compiled_re.contains(content):
        return []
    lines = content.splitlines()
    matches: list[ZoektLineMatch] = []
    ctx = max(0, min(context_lines, 5))
    for idx, line_text in enumerate(lines):
        frags: list[tuple[int, int]] = []
        for m in compiled_re.finditer(line_text):
            frags.append((m.start(), m.end()))
        if not frags:
            continue
        before = tuple(lines[max(0, idx - ctx) : idx]) if ctx > 0 else ()
        after = tuple(lines[idx + 1 : min(len(lines), idx + 1 + ctx)]) if ctx > 0 else ()
        matches.append(
            ZoektLineMatch(
                line_number=idx + 1,
                line_text=line_text,
                before_lines=before,
                after_lines=after,
                fragments=tuple(frags),
            )
        )
    return matches


def _build_overlay_matches(
    *,
    overlay_deltas: Sequence[OverlayFileDeltaRecord],
    compiled_query: RE2Pattern,
    compiled_file_re: RE2Pattern | None,
    authorized_repos: frozenset[str],
    acl_registry: RepoACLBitmapRegistry,
    principal: PrincipalContext,
    branch_label: str,
    context_lines: int,
) -> tuple[list[ZoektFileMatch], set[tuple[str, str]]]:
    """Scans CoW overlay deltas and returns (overlay_matches, suppressed_base_keys)."""
    suppressed: set[tuple[str, str]] = set()
    matches: list[ZoektFileMatch] = []
    principal_scopes = set(principal.allowed_scopes)

    for delta in overlay_deltas:
        if delta.tenant_id != principal.tenant_id:
            continue
        if delta.repo_name not in authorized_repos:
            continue
        if not set(delta.allowed_scopes).intersection(principal_scopes):
            continue

        suppressed.add((delta.repo_name, delta.file_path))
        if delta.previous_file_path:
            suppressed.add((delta.repo_name, delta.previous_file_path))

        if delta.change_type == "deleted" or delta.raw_content is None:
            continue
        if compiled_file_re is not None and not compiled_file_re.contains(delta.file_path):
            continue

        line_matches = _scan_content_lines(
            delta.raw_content, compiled_query, context_lines=context_lines
        )
        if not line_matches:
            continue

        base_score = 1.0 + 0.1 * min(len(line_matches), 20)
        weighted_score = round(base_score * delta.ranking_weight, 6)
        rid = acl_registry.get_repo_id(delta.tenant_id, delta.domain_id, delta.repo_name)
        matches.append(
            ZoektFileMatch(
                repo_name=delta.repo_name,
                repo_id=rid,
                file_path=delta.file_path,
                branch=branch_label,
                language=delta.language,
                score=weighted_score,
                ranking_weight=delta.ranking_weight,
                is_generated=delta.is_generated,
                is_vendored=delta.is_vendored,
                source_layer="overlay",
                line_matches=tuple(line_matches),
                checksum=delta.overlay_blob_sha,
            )
        )
    return matches, suppressed


@dataclass
class _IndexedShardDoc:
    doc_id: int
    tenant_id: UUID
    domain_id: str
    repo_id: int
    repo_name: str
    file_path: str
    branch: str
    language: str
    content: str
    allowed_scopes: frozenset[str]
    ranking_weight: float
    is_generated: bool
    is_vendored: bool
    checksum: str | None
    trigrams: frozenset[str]


class LocalTrigramZoektIndex:
    """In-process Zoekt-compatible inverted trigram shard engine with CoW overlay support."""

    def __init__(self, acl_registry: RepoACLBitmapRegistry | None = None) -> None:
        self.acl_registry = acl_registry or RepoACLBitmapRegistry()
        self._lock = threading.RLock()
        self._next_doc_id = 0
        self._docs: dict[int, _IndexedShardDoc] = {}
        self._doc_key_to_id: dict[tuple[UUID, str, str, str, str], int] = {}
        self._trigram_postings: dict[str, int] = {}
        self._repo_doc_masks: dict[int, int] = {}

    def register_repository(
        self,
        *,
        tenant_id: UUID,
        domain_id: str,
        repo_name: str,
        allowed_scopes: Sequence[str],
        repo_id: int | None = None,
    ) -> int:
        return self.acl_registry.register_repo(
            tenant_id=tenant_id,
            domain_id=domain_id,
            repo_name=repo_name,
            allowed_scopes=allowed_scopes,
            repo_id=repo_id,
        )

    def _remove_doc_locked(self, doc_id: int) -> None:
        doc = self._docs.pop(doc_id, None)
        if doc is None:
            return
        self._doc_key_to_id.pop(
            (doc.tenant_id, doc.domain_id, doc.repo_name, doc.branch, doc.file_path), None
        )
        clear_mask = ~(1 << doc_id)
        if doc.repo_id in self._repo_doc_masks:
            updated_repo_mask = self._repo_doc_masks[doc.repo_id] & clear_mask
            if updated_repo_mask:
                self._repo_doc_masks[doc.repo_id] = updated_repo_mask
            else:
                self._repo_doc_masks.pop(doc.repo_id, None)
        for trig in doc.trigrams:
            posting = self._trigram_postings.get(trig, 0) & clear_mask
            if posting:
                self._trigram_postings[trig] = posting
            else:
                self._trigram_postings.pop(trig, None)

    def index_file(
        self,
        *,
        tenant_id: UUID,
        domain_id: str,
        repo_name: str,
        file_path: str,
        content: str,
        allowed_scopes: Sequence[str],
        language: str = "text",
        branch: str = "HEAD",
        ranking_weight: float = 1.0,
        is_generated: bool = False,
        is_vendored: bool = False,
        checksum: str | None = None,
    ) -> int:
        clean_scopes = frozenset(s.strip() for s in allowed_scopes if s and s.strip())
        doc_key = (tenant_id, domain_id, repo_name, branch, file_path)
        lowered = content.lower()
        trigrams = frozenset(lowered[i : i + 3] for i in range(max(0, len(lowered) - 2)))

        with self._lock:
            existing_rid = self.acl_registry.get_repo_id(tenant_id, domain_id, repo_name)
            if existing_rid is None:
                repo_id = self.acl_registry.register_repo(
                    tenant_id, domain_id, repo_name, list(clean_scopes)
                )
            else:
                repo_id = existing_rid
                if clean_scopes:
                    existing_entry = self.acl_registry._entries_by_id[repo_id]
                    merged_scopes = existing_entry.allowed_scopes | clean_scopes
                    if merged_scopes != existing_entry.allowed_scopes:
                        self.acl_registry.update_repo_scopes(
                            tenant_id, domain_id, repo_name, sorted(merged_scopes)
                        )

            old_doc_id = self._doc_key_to_id.get(doc_key)
            if old_doc_id is not None:
                self._remove_doc_locked(old_doc_id)

            doc_id = self._next_doc_id
            self._next_doc_id += 1
            bit = 1 << doc_id
            doc = _IndexedShardDoc(
                doc_id=doc_id,
                tenant_id=tenant_id,
                domain_id=domain_id,
                repo_id=repo_id,
                repo_name=repo_name,
                file_path=file_path,
                branch=branch,
                language=language,
                content=content,
                allowed_scopes=clean_scopes,
                ranking_weight=ranking_weight,
                is_generated=is_generated,
                is_vendored=is_vendored,
                checksum=checksum,
                trigrams=trigrams,
            )
            self._docs[doc_id] = doc
            self._doc_key_to_id[doc_key] = doc_id
            self._repo_doc_masks[repo_id] = self._repo_doc_masks.get(repo_id, 0) | bit
            for trig in trigrams:
                self._trigram_postings[trig] = self._trigram_postings.get(trig, 0) | bit
            return doc_id

    def delete_file(
        self,
        *,
        tenant_id: UUID,
        domain_id: str,
        repo_name: str,
        file_path: str,
        branch: str = "HEAD",
    ) -> bool:
        doc_key = (tenant_id, domain_id, repo_name, branch, file_path)
        with self._lock:
            doc_id = self._doc_key_to_id.get(doc_key)
            if doc_id is None:
                return False
            self._remove_doc_locked(doc_id)
            return True

    def delete_repo_files(self, *, tenant_id: UUID, domain_id: str, repo_name: str) -> int:
        with self._lock:
            to_delete = [
                doc_id
                for doc_id, doc in self._docs.items()
                if doc.tenant_id == tenant_id
                and doc.domain_id == domain_id
                and doc.repo_name == repo_name
            ]
            for doc_id in to_delete:
                self._remove_doc_locked(doc_id)
            return len(to_delete)

    def delete_file_all_branches(
        self,
        *,
        tenant_id: UUID,
        domain_id: str,
        repo_name: str,
        file_path: str,
    ) -> int:
        """Deletes all branch shards matching (tenant_id, domain_id, repo_name, file_path)."""
        with self._lock:
            to_delete = [
                doc_id
                for doc_id, doc in self._docs.items()
                if doc.tenant_id == tenant_id
                and doc.domain_id == domain_id
                and doc.repo_name == repo_name
                and doc.file_path == file_path
            ]
            for doc_id in to_delete:
                self._remove_doc_locked(doc_id)
            return len(to_delete)

    def purge_repo(self, *, tenant_id: UUID, domain_id: str, repo_name: str) -> int:
        """Hard-deletes all repository shards and revokes bitmap ACL registration."""
        with self._lock:
            deleted = self.delete_repo_files(
                tenant_id=tenant_id,
                domain_id=domain_id,
                repo_name=repo_name,
            )
            self.acl_registry.unregister_repo(tenant_id, domain_id, repo_name)
            return deleted

    def purge_by_checksum(self, *, tenant_id: UUID, domain_id: str, checksum: str) -> int:
        """Hard-deletes all indexed shards matching a specific blob SHA/checksum."""
        clean_sum = checksum.strip()
        if not clean_sum:
            return 0
        with self._lock:
            to_delete = [
                doc_id
                for doc_id, doc in self._docs.items()
                if doc.tenant_id == tenant_id
                and doc.domain_id == domain_id
                and doc.checksum == clean_sum
            ]
            for doc_id in to_delete:
                self._remove_doc_locked(doc_id)
            return len(to_delete)

    def search(
        self,
        query: str,
        principal: PrincipalContext,
        *,
        domain_id: str | None = None,
        file_pattern: str | None = None,
        repo_names: Sequence[str] | None = None,
        branch: str | None = None,
        limit: int = 50,
        uncapped: bool = False,
        case_sensitive: bool = False,
        context_lines: int = 1,
        overlay_deltas: Sequence[OverlayFileDeltaRecord] | None = None,
    ) -> ZoektSearchResponse:
        t0 = time.perf_counter()
        compiled_query = compile_re2(query, case_insensitive=not case_sensitive)
        compiled_file_re = (
            compile_re2(file_pattern, case_insensitive=False)
            if file_pattern and file_pattern.strip()
            else None
        )
        plan = extract_required_trigrams(query, is_regex=True)

        acl = self.acl_registry.compute_authorized_bitmap(
            principal,
            domain_id=domain_id,
            repo_names=repo_names,
        )
        if acl.popcount == 0:
            duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return ZoektSearchResponse(
                query=query,
                file_matches=(),
                stats=ZoektSearchStats(
                    duration_ms=duration_ms,
                    files_considered=0,
                    files_matched=0,
                    matches_returned=0,
                    truncated=False,
                    short_circuited_acl=True,
                    authorized_repo_count=0,
                    authorized_repo_bitmap_popcount=0,
                    trigram_candidates=0,
                ),
            )

        overlay_matches: list[ZoektFileMatch] = []
        suppressed_base: set[tuple[str, str]] = set()
        if overlay_deltas:
            overlay_matches, suppressed_base = _build_overlay_matches(
                overlay_deltas=overlay_deltas,
                compiled_query=compiled_query,
                compiled_file_re=compiled_file_re,
                authorized_repos=acl.repo_names,
                acl_registry=self.acl_registry,
                principal=principal,
                branch_label=branch or "overlay",
                context_lines=context_lines,
            )

        principal_scopes = frozenset(principal.allowed_scopes)
        with self._lock:
            auth_doc_mask = 0
            for rid in acl.repo_ids:
                auth_doc_mask |= self._repo_doc_masks.get(rid, 0)

            candidate_mask = auth_doc_mask
            for group in plan.required_groups:
                group_mask = 0
                for trig in group:
                    group_mask |= self._trigram_postings.get(trig, 0)
                candidate_mask &= group_mask
                if candidate_mask == 0:
                    break

            candidates: list[_IndexedShardDoc] = []
            m = candidate_mask
            while m:
                lsb = m & -m
                doc_id = lsb.bit_length() - 1
                m &= m - 1
                doc = self._docs.get(doc_id)
                if doc is not None:
                    candidates.append(doc)

        trigram_candidates = len(candidates)
        all_matches: list[ZoektFileMatch] = list(overlay_matches)
        considered = len(overlay_deltas) if overlay_deltas else 0

        for doc in candidates:
            if doc.tenant_id != principal.tenant_id:
                continue
            if not doc.allowed_scopes.intersection(principal_scopes):
                continue
            if branch is not None and doc.branch != branch:
                continue
            if (doc.repo_name, doc.file_path) in suppressed_base:
                continue
            if compiled_file_re is not None and not compiled_file_re.contains(doc.file_path):
                continue
            considered += 1
            line_matches = _scan_content_lines(
                doc.content, compiled_query, context_lines=context_lines
            )
            if not line_matches:
                continue

            base_score = 1.0 + 0.1 * min(len(line_matches), 20)
            weighted_score = round(base_score * doc.ranking_weight, 6)
            all_matches.append(
                ZoektFileMatch(
                    repo_name=doc.repo_name,
                    repo_id=doc.repo_id,
                    file_path=doc.file_path,
                    branch=doc.branch,
                    language=doc.language,
                    score=weighted_score,
                    ranking_weight=doc.ranking_weight,
                    is_generated=doc.is_generated,
                    is_vendored=doc.is_vendored,
                    source_layer="base_snapshot",
                    line_matches=tuple(line_matches),
                    checksum=doc.checksum,
                )
            )

        total_matched = len(all_matches)
        if uncapped or limit <= 0:
            selected = sorted(
                all_matches,
                key=lambda item: (-item.score, item.repo_name, item.file_path),
            )
            truncated = False
        else:
            selected = heapq.nsmallest(
                limit,
                all_matches,
                key=lambda item: (-item.score, item.repo_name, item.file_path),
            )
            truncated = total_matched > len(selected)

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return ZoektSearchResponse(
            query=query,
            file_matches=tuple(selected),
            stats=ZoektSearchStats(
                duration_ms=duration_ms,
                files_considered=considered,
                files_matched=total_matched,
                matches_returned=len(selected),
                truncated=truncated,
                short_circuited_acl=False,
                authorized_repo_count=len(acl.repo_ids),
                authorized_repo_bitmap_popcount=acl.popcount,
                trigram_candidates=trigram_candidates,
            ),
        )

    def enumerate_file_matches(
        self,
        query: str,
        principal: PrincipalContext,
        *,
        domain_id: str | None = None,
        file_pattern: str | None = None,
        repo_names: Sequence[str] | None = None,
        branch: str | None = None,
        case_sensitive: bool = False,
        context_lines: int = 1,
        overlay_deltas: Sequence[OverlayFileDeltaRecord] | None = None,
    ) -> Iterator[ZoektFileMatch]:
        response = self.search(
            query,
            principal,
            domain_id=domain_id,
            file_pattern=file_pattern,
            repo_names=repo_names,
            branch=branch,
            limit=UNCAPPED_ZOEK_DISPLAY_COUNT,
            uncapped=True,
            case_sensitive=case_sensitive,
            context_lines=context_lines,
            overlay_deltas=overlay_deltas,
        )
        yield from response.file_matches
