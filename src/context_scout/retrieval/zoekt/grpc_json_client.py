"""Connect gRPC-JSON & streaming HTTP client for remote Zoekt webserver shards."""

from __future__ import annotations

import json
import logging
import re
import struct
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Literal
from uuid import UUID

import httpx

from context_scout.db.schema import OverlayFileDeltaRecord
from context_scout.retrieval.zoekt.acl_bitmap import RepoACLBitmapRegistry, RepoACLResult
from context_scout.retrieval.zoekt.local_trigram_index import (
    _build_overlay_matches,
    _decode_zoekt_b64_or_text,
    _decode_zoekt_text,
)
from context_scout.retrieval.zoekt.query_builder import (
    _CONNECT_RPC_SERVICE,
    UNCAPPED_ZOEK_DISPLAY_COUNT,
    ZoektFileMatch,
    ZoektLineMatch,
    ZoektSearchResponse,
    ZoektSearchStats,
    ZoektUnavailableError,
    compile_re2,
    format_zoekt_shard_repo_name,
    strip_zoekt_shard_repo_name,
)
from context_scout.schemas.contracts import PrincipalContext

logger = logging.getLogger(__name__)


def _parse_connect_stream_envelopes(raw: bytes) -> list[dict[str, Any]]:
    """Parses Connect streaming binary frames, NDJSON lines, or single JSON payloads."""
    if not raw:
        return []

    # Connect streaming binary frames begin with flags byte 0x00 or 0x02 + 4-byte uint32 BE len.
    if len(raw) >= 5 and raw[0] in {0, 2}:
        frames: list[dict[str, Any]] = []
        offset = 0
        parsed_valid_frames = False
        while offset + 5 <= len(raw):
            flags = raw[offset]
            frame_len = struct.unpack(">I", raw[offset + 1 : offset + 5])[0]
            offset += 5
            if flags not in {0, 1, 2, 3} or offset + frame_len > len(raw):
                parsed_valid_frames = False
                break
            chunk = raw[offset : offset + frame_len]
            offset += frame_len
            parsed_valid_frames = True
            obj = json.loads(chunk.decode("utf-8"))
            if isinstance(obj, dict):
                if flags & 0x02:
                    if obj.get("error"):
                        raise ZoektUnavailableError(
                            f"Zoekt StreamSearch error frame: {obj['error']}"
                        )
                    continue
                frames.append(obj)
        if parsed_valid_frames and offset == len(raw):
            return frames

    text = raw.decode("utf-8").strip()
    if not text:
        return []
    try:
        single = json.loads(text)
        if isinstance(single, dict):
            return [single]
        if isinstance(single, list):
            return [item for item in single if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    ndjson_frames: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            ndjson_frames.append(obj)
    return ndjson_frames


class ZoektGrpcJsonClient:
    """Zoekt client speaking Connect gRPC-JSON (`/Search`, `/StreamSearch`) and `/api/search`."""

    def __init__(
        self,
        endpoint_url: str,
        *,
        acl_registry: RepoACLBitmapRegistry | None = None,
        protocol: Literal["grpc_json", "json_api"] = "grpc_json",
        timeout_seconds: float = 10.0,
        transport: httpx.BaseTransport | None = None,
        http_client: httpx.Client | None = None,
        weight_resolver: Callable[[str, str, str], tuple[float, bool, bool]] | None = None,
        repository: Any | None = None,
        connection_manager: Any | None = None,
    ) -> None:
        cleaned = endpoint_url.strip().rstrip("/")
        if not cleaned:
            raise ValueError("Zoekt endpoint_url cannot be empty.")
        self.endpoint_url = cleaned
        self.acl_registry = acl_registry or RepoACLBitmapRegistry()
        self.protocol = protocol
        self.timeout_seconds = timeout_seconds
        self._weight_resolver = weight_resolver
        self._repository = repository
        self._connection_manager = connection_manager
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=timeout_seconds,
            transport=transport,
        )

    def _hydrate_acl_registry_from_db(
        self,
        principal: PrincipalContext,
        *,
        domain_id: str | None = None,
    ) -> int:
        """Hydrates RepoACLBitmapRegistry from PG17 `repositories` (or InMemory) under RLS."""
        hydrated = 0
        if self._repository is not None and hasattr(self._repository, "_repos"):
            repos_map = getattr(self._repository, "_repos", {})
            for rec in repos_map.values():
                if rec.tenant_id != principal.tenant_id:
                    continue
                if domain_id is not None and rec.domain_id != domain_id:
                    continue
                if not getattr(rec, "is_active", True):
                    continue
                if principal.allowed_scopes and not (
                    set(rec.allowed_scopes) & set(principal.allowed_scopes)
                ):
                    continue
                self.acl_registry.register_repo(
                    rec.tenant_id,
                    rec.domain_id,
                    rec.repo_name,
                    rec.allowed_scopes,
                )
                hydrated += 1
            return hydrated

        if self._connection_manager is None:
            return 0

        from context_scout.security.rbac import scoped_db_session

        with scoped_db_session(self._connection_manager, principal) as conn:
            if domain_id is not None:
                rows = conn.execute(
                    """
                    SELECT tenant_id, domain_id, repo_name, allowed_scopes
                    FROM repositories
                    WHERE is_active = TRUE
                      AND domain_id = %s;
                    """,
                    (domain_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT tenant_id, domain_id, repo_name, allowed_scopes
                    FROM repositories
                    WHERE is_active = TRUE;
                    """
                ).fetchall()
            for r in rows:
                t_id = r[0] if isinstance(r[0], UUID) else UUID(str(r[0]))
                d_id = str(r[1])
                r_name = str(r[2])
                scopes_raw = r[3]
                scopes = (
                    [str(s) for s in scopes_raw] if isinstance(scopes_raw, (list, tuple)) else []
                )
                self.acl_registry.register_repo(t_id, d_id, r_name, scopes)
                hydrated += 1
        return hydrated

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def purge_repo(self, *, tenant_id: UUID, domain_id: str, repo_name: str) -> int:
        """Revoke ACL bitmap entry immediately and request remote Zoekt shard purge."""
        bare_repo, matched = strip_zoekt_shard_repo_name(tenant_id, repo_name)
        logical_repo = bare_repo if matched else repo_name
        shard_repo = format_zoekt_shard_repo_name(tenant_id, logical_repo)
        removed = 1 if self.acl_registry.unregister_repo(tenant_id, domain_id, logical_repo) else 0
        try:
            resp = self._client.post(
                f"{self.endpoint_url}/api/admin/purge",
                json={
                    "tenant_id": str(tenant_id),
                    "domain_id": domain_id,
                    "repo_name": shard_repo,
                    "logical_repo_name": logical_repo,
                },
            )
            if 200 <= resp.status_code < 300:
                removed = max(removed, 1)
        except httpx.HTTPError as exc:
            logger.warning(
                "Remote Zoekt purge endpoint failed for repo '%s' (%s); "
                "ACL bitmap entry was revoked fail-closed.",
                shard_repo,
                type(exc).__name__,
            )
        return removed

    def _build_zoekt_query(
        self,
        *,
        tenant_id: UUID,
        query: str,
        authorized_repos: frozenset[str],
        file_pattern: str | None,
        branch: str | None,
        case_sensitive: bool,
    ) -> str:
        repo_alt = "|".join(
            re.escape(format_zoekt_shard_repo_name(tenant_id, r)).replace(r"\-", "-")
            for r in sorted(authorized_repos)
        )
        clauses = [f"r:^({repo_alt})$"]
        if branch and branch.strip():
            clauses.append(f"b:^{re.escape(branch.strip())}$")
        if file_pattern and file_pattern.strip():
            clauses.append(f"f:{file_pattern.strip()}")
        clauses.append("case:yes" if case_sensitive else "case:no")
        clauses.append(f"({query})")
        return " ".join(clauses)

    def _parse_zoekt_payloads(
        self,
        payloads: Sequence[dict[str, Any]],
        *,
        acl: RepoACLResult,
        domain_id: str,
        suppressed_base: set[tuple[str, str]],
        branch_fallback: str,
    ) -> tuple[list[ZoektFileMatch], int]:
        matches: list[ZoektFileMatch] = []
        considered = 0
        authorized_ids = set(acl.repo_ids)

        for payload in payloads:
            raw_res = payload.get("Result")
            result_obj = raw_res if isinstance(raw_res, dict) else payload
            if not isinstance(result_obj, dict):
                continue
            stats_obj = result_obj.get("Stats")
            if isinstance(stats_obj, dict):
                considered += int(stats_obj.get("FilesConsidered", 0) or 0)
            raw_files = result_obj.get("Files") or result_obj.get("FileMatches") or []
            if not isinstance(raw_files, list):
                continue

            for raw_file in raw_files:
                if not isinstance(raw_file, dict):
                    continue
                raw_repo = _decode_zoekt_text(
                    raw_file.get("Repository") or raw_file.get("repo_name")
                )
                repo_name, matched_tenant = strip_zoekt_shard_repo_name(acl.tenant_id, raw_repo)
                raw_rid = raw_file.get("RepositoryID") or raw_file.get("RepoID")
                repo_id = int(raw_rid) if raw_rid is not None else None
                # Fail-closed defense-in-depth post-filter against stale/spoofed server shards.
                if not matched_tenant or repo_name not in acl.repo_names:
                    logger.warning(
                        "Zoekt post-filter dropped unauthorized repo '%s' for tenant %s",
                        raw_repo,
                        acl.tenant_id,
                    )
                    continue
                if repo_id is not None and authorized_ids and repo_id not in authorized_ids:
                    logger.warning(
                        "Zoekt post-filter dropped unauthorized repo_id=%s for tenant %s",
                        repo_id,
                        acl.tenant_id,
                    )
                    continue

                file_path = _decode_zoekt_b64_or_text(
                    raw_file.get("FileName") or raw_file.get("file_path")
                )
                if (repo_name, file_path) in suppressed_base:
                    continue

                branches = raw_file.get("Branches") or []
                file_branch = (
                    str(branches[0]) if isinstance(branches, list) and branches else branch_fallback
                )
                language = str(raw_file.get("Language") or "text")
                checksum = (
                    str(raw_file["Checksum"]) if raw_file.get("Checksum") is not None else None
                )
                raw_score = float(raw_file.get("Score") or 1.0)

                weight = 1.0
                is_gen = False
                is_vend = False
                if self._weight_resolver is not None:
                    weight, is_gen, is_vend = self._weight_resolver(domain_id, repo_name, file_path)

                line_matches: list[ZoektLineMatch] = []
                raw_lm_list = raw_file.get("LineMatches") or raw_file.get("ChunkMatches") or []
                if isinstance(raw_lm_list, list):
                    for lm in raw_lm_list:
                        if not isinstance(lm, dict):
                            continue
                        content_start = lm.get("ContentStart")
                        fallback_ln = (
                            content_start.get("LineNumber")
                            if isinstance(content_start, dict)
                            else None
                        )
                        line_num = int(lm.get("LineNumber") or fallback_ln or 1)
                        line_txt = _decode_zoekt_b64_or_text(
                            lm.get("Line") or lm.get("Content")
                        ).rstrip("\r\n")
                        before_raw = lm.get("Before")
                        after_raw = lm.get("After")
                        before_lines = (
                            tuple(_decode_zoekt_b64_or_text(before_raw).splitlines())
                            if before_raw
                            else ()
                        )
                        after_lines = (
                            tuple(_decode_zoekt_b64_or_text(after_raw).splitlines())
                            if after_raw
                            else ()
                        )
                        frags: list[tuple[int, int]] = []
                        for frag in lm.get("LineFragments") or []:
                            if isinstance(frag, dict):
                                off = int(frag.get("LineOffset") or 0)
                                length = int(frag.get("MatchLength") or 0)
                                frags.append((off, off + length))
                        line_matches.append(
                            ZoektLineMatch(
                                line_number=line_num,
                                line_text=line_txt,
                                before_lines=before_lines,
                                after_lines=after_lines,
                                fragments=tuple(frags),
                            )
                        )

                matches.append(
                    ZoektFileMatch(
                        repo_name=repo_name,
                        repo_id=repo_id,
                        file_path=file_path,
                        branch=file_branch,
                        language=language,
                        score=round(raw_score * weight, 6),
                        ranking_weight=weight,
                        is_generated=is_gen,
                        is_vendored=is_vend,
                        source_layer="base_snapshot",
                        line_matches=tuple(line_matches),
                        checksum=checksum,
                    )
                )
        return matches, considered

    def _post_rpc(
        self,
        *,
        effective_q: str,
        repo_ids: tuple[int, ...],
        max_docs: int,
        max_matches: int,
        context_lines: int,
        stream: bool,
    ) -> list[dict[str, Any]]:
        opts = {
            "MaxDocDisplayCount": max_docs,
            "TotalMaxMatchCount": max_matches,
            "NumContextLines": max(0, min(context_lines, 5)),
        }
        body = {
            "Q": effective_q,
            "RepoIds": list(repo_ids),
            "Opts": opts,
        }
        try:
            if self.protocol == "grpc_json":
                method = "StreamSearch" if stream else "Search"
                url = f"{self.endpoint_url}/{_CONNECT_RPC_SERVICE}/{method}"
                headers = {
                    "Content-Type": "application/json",
                    "Connect-Protocol-Version": "1",
                    "Accept": (
                        "application/connect+json, application/x-ndjson, application/json"
                        if stream
                        else "application/json"
                    ),
                }
                resp = self._client.post(url, json=body, headers=headers)
                resp.raise_for_status()
                if stream:
                    return _parse_connect_stream_envelopes(resp.content)
                data = resp.json()
                return [data] if isinstance(data, dict) else []

            url = f"{self.endpoint_url}/api/search"
            resp = self._client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
            return [data] if isinstance(data, dict) else []
        except (httpx.HTTPError, ValueError, struct.error) as exc:
            raise ZoektUnavailableError(
                f"Zoekt RPC request failed against {self.endpoint_url}: {exc}"
            ) from exc

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
        acl = self.acl_registry.compute_authorized_bitmap(
            principal, domain_id=domain_id, repo_names=repo_names
        )
        missing_requested = bool(
            repo_names and any(r not in acl.repo_names for r in repo_names if r)
        )
        if (acl.popcount == 0 or missing_requested) and self._hydrate_acl_registry_from_db(
            principal, domain_id=domain_id
        ) > 0:
            acl = self.acl_registry.compute_authorized_bitmap(
                principal, domain_id=domain_id, repo_names=repo_names
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

        effective_q = self._build_zoekt_query(
            tenant_id=principal.tenant_id,
            query=query,
            authorized_repos=acl.repo_names,
            file_pattern=file_pattern,
            branch=branch,
            case_sensitive=case_sensitive,
        )
        max_docs = UNCAPPED_ZOEK_DISPLAY_COUNT if uncapped or limit <= 0 else limit
        max_matches = (
            UNCAPPED_ZOEK_DISPLAY_COUNT if uncapped or limit <= 0 else max(limit * 20, 200)
        )
        payloads = self._post_rpc(
            effective_q=effective_q,
            repo_ids=acl.repo_ids,
            max_docs=max_docs,
            max_matches=max_matches,
            context_lines=context_lines,
            stream=uncapped,
        )
        base_matches, considered = self._parse_zoekt_payloads(
            payloads,
            acl=acl,
            domain_id=domain_id or "",
            suppressed_base=suppressed_base,
            branch_fallback=branch or "HEAD",
        )
        all_matches = overlay_matches + base_matches
        total_matched = len(all_matches)
        if uncapped or limit <= 0:
            selected = sorted(
                all_matches,
                key=lambda item: (-item.score, item.repo_name, item.file_path),
            )
            truncated = False
        else:
            selected = sorted(
                all_matches,
                key=lambda item: (-item.score, item.repo_name, item.file_path),
            )[:limit]
            truncated = total_matched > len(selected)

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return ZoektSearchResponse(
            query=query,
            file_matches=tuple(selected),
            stats=ZoektSearchStats(
                duration_ms=duration_ms,
                files_considered=considered + (len(overlay_deltas) if overlay_deltas else 0),
                files_matched=total_matched,
                matches_returned=len(selected),
                truncated=truncated,
                short_circuited_acl=False,
                authorized_repo_count=len(acl.repo_ids),
                authorized_repo_bitmap_popcount=acl.popcount,
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
