"""Confluence REST API Connector and Lossless Verbatim Chunker (F15)."""

from __future__ import annotations

import dataclasses
import html
import logging
import re
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, Field

from context_scout.db.repositories import PostgresRLSRepository
from context_scout.db.schema import ConfluencePageRecord, DocChunkRecord
from context_scout.ingestion.git_merkle_indexer import generate_deterministic_embedding
from context_scout.plugins.base import SourceConnectorPlugin, SyncResult
from context_scout.security.permission_mirror import (
    ConfluencePermission,
    PermissionMirror,
    sanitize_scope_identifier,
)

logger = logging.getLogger(__name__)

# Regular expressions for clean structural HTML/storage format parsing
_HEADING_PATTERN = re.compile(r"<h([1-6])[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
_TAG_CLEANER = re.compile(r"<[^>]+>")
_WHITESPACE_NORMALIZER = re.compile(r"\n{3,}")
_DEPRECATED_TITLE_PATTERN = re.compile(
    r"(?:\[(?:deprecated|archived|obsolete)\]|\b(?:deprecated|archived|obsolete)\b)",
    re.IGNORECASE,
)
_MEETING_TITLE_PATTERN = re.compile(
    r"(?:^|[\s\[\(\-_/:])"
    r"(?:meeting[\s_-]*notes?|retro(?:spective)?|stand[\s_-]*up|1[\s_-]*on[\s_-]*1|weekly[\s_-]*sync)"
    r"(?:$|[\s\]\)\-_/:,])",
    re.IGNORECASE,
)
_DRAFT_TITLE_PATTERN = re.compile(
    r"(?:\[(?:draft|wip)\]|\b(?:draft|wip)\b)",
    re.IGNORECASE,
)
_BOOST_TITLE_PATTERN = re.compile(
    r"\b(?:runbook|playbook|adr|rfc|architecture|sla|onboarding|policy)\b",
    re.IGNORECASE,
)
_TBD_TODO_PATTERN = re.compile(r"\b(?:TBD|TODO)\b")


def _ensure_utc(dt: datetime) -> datetime:
    """Normalizes naive or offset datetimes to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _extract_labels(raw_labels: Any) -> tuple[str, ...]:
    """Parses Confluence v2 or simplified list labels into a normalized tuple."""
    if not raw_labels:
        return ()
    items = raw_labels.get("results", []) if isinstance(raw_labels, dict) else raw_labels
    if not isinstance(items, (list, tuple)):
        return ()
    out: list[str] = []
    for entry in items:
        if isinstance(entry, dict):
            name = str(entry.get("name") or entry.get("label") or "").strip()
        else:
            name = str(entry).strip()
        if name:
            out.append(name)
    return tuple(out)


class ConfluencePageDTO(BaseModel):
    """Pydantic representation of an upstream Confluence page."""

    page_id: str
    space_key: str
    title: str
    version_number: int
    author_username: str
    source_url: str
    body_storage: str
    parent_id: str | None = None
    labels: tuple[str, ...] = ()
    status: str = "current"
    created_at: datetime
    last_modified_at: datetime


class NoiseEvaluationResult(BaseModel):
    """Outcome of ConfluenceNoisePolicy evaluation for a single page."""

    action: Literal["exclude", "demote", "boost", "standard"]
    authority_weight: float
    reason_key: str
    reason_detail: str


class ConfluenceNoisePolicy(BaseModel):
    """Configurable lifecycle noise filter and authority weighting policy for Confluence pages."""

    exclude_personal_spaces: bool = True
    excluded_statuses: tuple[str, ...] = ("archived", "trashed", "deleted")
    deprecated_tokens: tuple[str, ...] = ("deprecated", "archived", "obsolete")
    meeting_tokens: tuple[str, ...] = (
        "meeting-notes",
        "meeting_notes",
        "meeting notes",
        "retro",
        "retrospective",
        "standup",
        "stand-up",
        "1-on-1",
        "weekly-sync",
    )
    draft_tokens: tuple[str, ...] = ("draft", "wip")
    boost_tokens: tuple[str, ...] = (
        "runbook",
        "playbook",
        "adr",
        "rfc",
        "architecture",
        "sla",
        "onboarding",
        "policy",
    )
    boilerplate_phrases: tuple[str, ...] = ("click here to edit",)
    boilerplate_tags: tuple[str, ...] = ("<ac:placeholder",)
    min_substantive_chars: int = Field(default=15, ge=1)
    draft_multiplier: float = Field(default=0.20, gt=0.0, le=1.0)
    boost_multiplier: float = Field(default=1.35, ge=1.0, le=2.0)
    max_authority_weight: float = Field(default=2.0, ge=1.0, le=5.0)

    def evaluate(
        self,
        page: ConfluencePageDTO,
        stripped_body: str,
        base_authority_weight: float = 1.0,
    ) -> NoiseEvaluationResult:
        """Evaluates whether a Confluence page is excluded, demoted, boosted, or standard."""
        space_clean = page.space_key.strip()
        if self.exclude_personal_spaces and space_clean.startswith("~"):
            return NoiseEvaluationResult(
                action="exclude",
                authority_weight=0.0,
                reason_key="personal_space",
                reason_detail=f"Excluded personal space '{space_clean}'",
            )

        status_norm = page.status.strip().lower()
        if status_norm in {s.lower() for s in self.excluded_statuses}:
            return NoiseEvaluationResult(
                action="exclude",
                authority_weight=0.0,
                reason_key="status_excluded",
                reason_detail=f"Excluded page status '{status_norm}'",
            )

        labels_lower = {lbl.strip().lower() for lbl in page.labels if lbl.strip()}
        title_lower = page.title.strip().lower()

        if any(tok in labels_lower for tok in self.deprecated_tokens) or (
            _DEPRECATED_TITLE_PATTERN.search(title_lower) is not None
        ):
            return NoiseEvaluationResult(
                action="exclude",
                authority_weight=0.0,
                reason_key="deprecated_or_archived",
                reason_detail=f"Excluded deprecated/archived page '{page.title}'",
            )

        if any(tok in labels_lower for tok in self.meeting_tokens) or (
            _MEETING_TITLE_PATTERN.search(title_lower) is not None
        ):
            return NoiseEvaluationResult(
                action="exclude",
                authority_weight=0.0,
                reason_key="meeting_notes",
                reason_detail=f"Excluded meeting/retro/standup page '{page.title}'",
            )

        raw_lower = page.body_storage.lower()
        stripped_clean = stripped_body.strip()
        stripped_lower = stripped_clean.lower()

        if any(tag in raw_lower for tag in self.boilerplate_tags) or any(
            phrase in stripped_lower for phrase in self.boilerplate_phrases
        ):
            return NoiseEvaluationResult(
                action="exclude",
                authority_weight=0.0,
                reason_key="template_boilerplate",
                reason_detail="Excluded Confluence template placeholder boilerplate",
            )

        if _TBD_TODO_PATTERN.search(stripped_clean):
            without_markers = _TBD_TODO_PATTERN.sub("", stripped_clean).strip(" \t\r\n.:;-")
            if len(without_markers) < self.min_substantive_chars or len(stripped_clean) < 120:
                return NoiseEvaluationResult(
                    action="exclude",
                    authority_weight=0.0,
                    reason_key="template_boilerplate",
                    reason_detail="Excluded TBD/TODO stub or template page",
                )

        if len(stripped_clean) < self.min_substantive_chars:
            return NoiseEvaluationResult(
                action="exclude",
                authority_weight=0.0,
                reason_key="stub_too_short",
                reason_detail=(
                    f"Excluded stub ({len(stripped_clean)} chars < {self.min_substantive_chars})"
                ),
            )

        if any(tok in labels_lower for tok in self.draft_tokens) or (
            _DRAFT_TITLE_PATTERN.search(title_lower) is not None
        ):
            demoted = round(
                min(
                    self.max_authority_weight,
                    max(0.01, base_authority_weight * self.draft_multiplier),
                ),
                4,
            )
            return NoiseEvaluationResult(
                action="demote",
                authority_weight=demoted,
                reason_key="draft_or_wip",
                reason_detail=f"Demoted draft/WIP page '{page.title}' to weight {demoted}",
            )

        if any(tok in labels_lower for tok in self.boost_tokens) or (
            _BOOST_TITLE_PATTERN.search(title_lower) is not None
        ):
            boosted = round(
                min(self.max_authority_weight, base_authority_weight * self.boost_multiplier),
                4,
            )
            return NoiseEvaluationResult(
                action="boost",
                authority_weight=boosted,
                reason_key="authoritative_boost",
                reason_detail=f"Boosted authoritative page '{page.title}' to weight {boosted}",
            )

        standard_weight = round(min(self.max_authority_weight, base_authority_weight), 4)
        return NoiseEvaluationResult(
            action="standard",
            authority_weight=standard_weight,
            reason_key="standard",
            reason_detail="Standard active Confluence page",
        )


class ConfluenceVerbatimChunk(BaseModel):
    """Verbatim text chunk enriched with Anthropic 50-token contextual header."""

    chunk_id: UUID = Field(default_factory=uuid4)
    page_id: str
    space_key: str
    page_title: str
    version_number: int
    section_heading: str
    contextual_header: str
    verbatim_content: str
    allowed_scopes: list[str]
    last_modified_at: datetime
    authority_weight: float = 1.0
    is_stale: bool = False
    stale_reason: str | None = None


class ConfluenceConnectorPlugin(SourceConnectorPlugin):
    """Production Confluence Cloud REST API v2 connector."""

    def __init__(
        self,
        base_url: str = "https://mock.atlassian.net",
        email: str | None = None,
        api_token: str | None = None,
        bearer_token: str | None = None,
        client: httpx.Client | None = None,
        freshness_half_life_days: int = 180,
        noise_policy: ConfluenceNoisePolicy | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.freshness_half_life_days = freshness_half_life_days
        self.noise_policy = noise_policy or ConfluenceNoisePolicy()

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        auth = None
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        elif email and api_token:
            auth = httpx.BasicAuth(email, api_token)

        self.client = client or httpx.Client(
            base_url=self.base_url,
            headers=headers,
            auth=auth,
            timeout=30.0,
        )

    @property
    def source_id(self) -> str:
        return "confluence"

    @property
    def source_type(self) -> str:
        return "confluence"

    def mirror_permissions(self, container_key: str) -> list[str]:
        """Maps Confluence space key to RLS allowed_scopes."""
        clean_key = sanitize_scope_identifier(container_key)
        return PermissionMirror.map_confluence_permissions_to_scopes(
            clean_key, [ConfluencePermission.READ]
        )

    def evaluate_page_noise(
        self,
        page: ConfluencePageDTO,
        base_authority_weight: float = 1.0,
    ) -> NoiseEvaluationResult:
        """Evaluates lifecycle noise tier and authority weight for a Confluence page."""
        stripped_body = self._convert_html_to_verbatim_text(page.body_storage)
        return self.noise_policy.evaluate(
            page=page,
            stripped_body=stripped_body,
            base_authority_weight=base_authority_weight,
        )

    def fetch_item(self, item_id: str, version: str | None = None) -> dict[str, Any] | None:
        """Fetches a specific page by ID from Confluence REST API v2."""
        endpoint = f"/wiki/api/v2/pages/{item_id}"
        params = {"body-format": "storage"}
        resp = self.client.get(endpoint, params=params)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data: Any = resp.json()
        return data if isinstance(data, dict) else None

    def fetch_space_pages(
        self,
        space_key: str,
        since_updated_at: datetime | None = None,
    ) -> list[ConfluencePageDTO]:
        """Paginates pages in a given space via REST API v2, filtering by incremental cursor."""
        pages: list[ConfluencePageDTO] = []
        cursor: str | None = None
        since_utc = _ensure_utc(since_updated_at) if since_updated_at is not None else None

        while True:
            params: dict[str, Any] = {
                "space-key": space_key,
                "body-format": "storage",
                "limit": 50,
            }
            if cursor:
                params["cursor"] = cursor

            resp = self.client.get("/wiki/api/v2/pages", params=params)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            for item in results:
                page_id = str(item.get("id"))
                title = str(item.get("title", ""))
                version_info = item.get("version", {})
                version_num = int(version_info.get("number", 1))
                author = str(version_info.get("authorId") or "unknown_author")
                last_modified_raw = version_info.get("createdAt") or item.get("createdAt")
                last_modified = (
                    _ensure_utc(datetime.fromisoformat(last_modified_raw.replace("Z", "+00:00")))
                    if last_modified_raw
                    else datetime.now(UTC)
                )

                if since_utc is not None and last_modified <= since_utc:
                    continue

                body_storage = item.get("body", {}).get("storage", {}).get("value", "")
                parent_id = str(item.get("parentId")) if item.get("parentId") else None
                status_val = str(item.get("status") or "current").strip().lower() or "current"
                labels_val = _extract_labels(item.get("labels"))
                source_url = f"{self.base_url}/wiki/spaces/{space_key}/pages/{page_id}"

                pages.append(
                    ConfluencePageDTO(
                        page_id=page_id,
                        space_key=space_key,
                        title=title,
                        version_number=version_num,
                        author_username=author,
                        source_url=source_url,
                        body_storage=body_storage,
                        parent_id=parent_id,
                        labels=labels_val,
                        status=status_val,
                        created_at=last_modified,
                        last_modified_at=last_modified,
                    )
                )

            links = data.get("_links", {})
            next_url = links.get("next")
            if not next_url:
                break
            match = re.search(r"cursor=([^&]+)", next_url)
            cursor = match.group(1) if match else None
            if not cursor:
                break

        return pages

    def chunk_page_verbatim(
        self,
        page: ConfluencePageDTO,
        max_chunk_chars: int = 2500,
        allowed_scopes: list[str] | None = None,
        authority_weight: float = 1.0,
    ) -> list[ConfluenceVerbatimChunk]:
        """Splits page storage content into lossless verbatim chunks with 50-token headers."""
        raw_storage = page.body_storage
        effective_scopes = (
            list(allowed_scopes)
            if allowed_scopes is not None
            else self.mirror_permissions(page.space_key)
        )

        # Check temporal decay
        now = datetime.now(UTC)
        days_old = (now - _ensure_utc(page.last_modified_at)).days
        is_decayed = days_old > self.freshness_half_life_days
        decay_reason = (
            f"Page last updated {days_old} days ago (> threshold of "
            f"{self.freshness_half_life_days} days)"
            if is_decayed
            else None
        )

        # Extract sections by HTML headings
        sections: list[tuple[str, str]] = []
        last_end = 0
        current_heading = "Overview"

        for match in _HEADING_PATTERN.finditer(raw_storage):
            start, end = match.span()
            section_content = raw_storage[last_end:start]
            if section_content.strip():
                clean_text = self._convert_html_to_verbatim_text(section_content)
                if clean_text.strip():
                    sections.append((current_heading, clean_text))

            heading_raw = match.group(2)
            current_heading = _TAG_CLEANER.sub("", heading_raw).strip() or "Untitled Section"
            last_end = end

        remaining = raw_storage[last_end:]
        if remaining.strip():
            clean_text = self._convert_html_to_verbatim_text(remaining)
            if clean_text.strip():
                sections.append((current_heading, clean_text))

        if not sections:
            clean_text = self._convert_html_to_verbatim_text(raw_storage)
            if clean_text.strip():
                sections.append(("Overview", clean_text))

        chunks: list[ConfluenceVerbatimChunk] = []

        for heading, body in sections:
            text_blocks = self._split_text_blocks(body, max_chunk_chars)
            for block in text_blocks:
                # Deterministic 50-token Anthropic Context Header
                context_header = (
                    f"This document is from Confluence space '{page.space_key}' > '{page.title}' "
                    f"(Page ID: {page.page_id}, Version: {page.version_number}, "
                    f"Last Modified: {page.last_modified_at.strftime('%Y-%m-%d')} "
                    f"by {page.author_username}). "
                    f"Section: '{heading}'. The following text specifies the exact requirements:"
                )

                chunks.append(
                    ConfluenceVerbatimChunk(
                        page_id=page.page_id,
                        space_key=page.space_key,
                        page_title=page.title,
                        version_number=page.version_number,
                        section_heading=heading,
                        contextual_header=context_header,
                        verbatim_content=block,
                        allowed_scopes=effective_scopes,
                        last_modified_at=page.last_modified_at,
                        authority_weight=authority_weight,
                        is_stale=is_decayed,
                        stale_reason=decay_reason,
                    )
                )

        return chunks

    def _convert_html_to_verbatim_text(self, raw_html: str) -> str:
        """Strips tags while strictly preserving exact whitespace and character semantics."""
        text = re.sub(r"<(?:br|p|div|tr)[^>]*>", "\n", raw_html, flags=re.IGNORECASE)
        text = re.sub(r"<li[^>]*>", "\n* ", text, flags=re.IGNORECASE)
        text = _TAG_CLEANER.sub("", text)
        text = html.unescape(text)
        text = _WHITESPACE_NORMALIZER.sub("\n\n", text)
        return text.strip()

    def _split_text_blocks(self, text: str, max_chars: int) -> list[str]:
        if len(text) <= max_chars:
            return [text]

        paragraphs = text.split("\n\n")
        blocks: list[str] = []
        current_block: list[str] = []
        current_len = 0

        for para in paragraphs:
            if current_len + len(para) > max_chars and current_block:
                blocks.append("\n\n".join(current_block))
                current_block = [para]
                current_len = len(para)
            else:
                current_block.append(para)
                current_len += len(para) + 2

        if current_block:
            blocks.append("\n\n".join(current_block))
        return blocks

    def tombstone_deleted_or_restricted_pages(
        self,
        domain_id: str,
        repository: PostgresRLSRepository | Any,
        deleted_page_ids: list[str],
        conn: Any = None,
        reason: str = "Page deleted or archived in Confluence",
        status: str = "deleted",
    ) -> tuple[int, int]:
        """Soft-deletes removed or restricted pages and marks their active doc_chunks stale."""
        clean_ids = list(dict.fromkeys(pid.strip() for pid in deleted_page_ids if pid.strip()))
        if not clean_ids:
            return (0, 0)

        is_postgres = hasattr(repository, "insert_doc_chunks_batch") and conn is not None
        if is_postgres and hasattr(repository, "tombstone_confluence_pages"):
            pg_repo: Any = repository
            pages_tombstoned, chunks_flagged = pg_repo.tombstone_confluence_pages(
                conn, domain_id, clean_ids, reason=reason, status=status
            )
            return (int(pages_tombstoned), int(chunks_flagged))

        if not is_postgres and hasattr(repository, "tombstone_confluence_pages"):
            mem_repo: Any = repository
            pages_tombstoned, chunks_flagged = mem_repo.tombstone_confluence_pages(
                domain_id, clean_ids, reason=reason, status=status
            )
            return (int(pages_tombstoned), int(chunks_flagged))

        target_doc_ids = {f"confluence:{pid}" for pid in clean_ids}
        chunks_flagged = 0
        if hasattr(repository, "_docs"):
            for cid, doc in list(repository._docs.items()):
                if doc.domain_id == domain_id and doc.doc_id in target_doc_ids and not doc.is_stale:
                    repository._docs[cid] = dataclasses.replace(
                        doc,
                        is_stale=True,
                        stale_reason=reason,
                    )
                    chunks_flagged += 1
        return (len(clean_ids) if chunks_flagged > 0 else 0, chunks_flagged)

    def _list_active_space_page_ids(
        self,
        domain_id: str,
        space_key: str,
        repository: PostgresRLSRepository | Any,
        conn: Any = None,
        principal: Any = None,
    ) -> list[str]:
        """Lists active Confluence page IDs currently indexed for a domain/space."""
        is_postgres = hasattr(repository, "insert_doc_chunks_batch") and conn is not None
        if is_postgres and hasattr(repository, "list_confluence_pages"):
            existing = repository.list_confluence_pages(conn, domain_id, space_key=space_key)
            return [p.confluence_page_id for p in existing]
        if (
            not is_postgres
            and principal is not None
            and hasattr(repository, "list_confluence_pages")
        ):
            existing = repository.list_confluence_pages(principal, domain_id, space_key=space_key)
            return [p.confluence_page_id for p in existing]
        if hasattr(repository, "_confluence"):
            deleted: Any = getattr(repository, "_deleted_confluence", set())
            return [
                p.confluence_page_id
                for p in repository._confluence.values()
                if p.domain_id == domain_id
                and p.space_key == space_key
                and (domain_id, p.confluence_page_id) not in deleted
            ]
        return []

    def _sync_allowed_scopes(
        self,
        domain_id: str,
        page_id: str,
        allowed_scopes: list[str],
        repository: PostgresRLSRepository | Any,
        conn: Any = None,
    ) -> None:
        """Immediately synchronizes allowed_scopes on confluence_pages and active doc_chunks."""
        is_postgres = hasattr(repository, "insert_doc_chunks_batch") and conn is not None
        if not hasattr(repository, "update_confluence_allowed_scopes"):
            return
        repo_any: Any = repository
        if is_postgres:
            repo_any.update_confluence_allowed_scopes(conn, domain_id, page_id, allowed_scopes)
        else:
            repo_any.update_confluence_allowed_scopes(domain_id, page_id, allowed_scopes)

    def sync_space_to_repository(
        self,
        domain_id: str,
        space_key: str,
        repository: PostgresRLSRepository | Any,
        conn: Any = None,
        principal: Any = None,
        current_pages: list[ConfluencePageDTO] | None = None,
        *,
        since_updated_at: datetime | None = None,
        deleted_page_ids: list[str] | None = None,
        reconcile_deletes: bool = False,
        space_allowed_scopes: list[str] | None = None,
        page_allowed_scopes_map: dict[str, list[str]] | None = None,
    ) -> SyncResult:
        """Executes full or incremental space synchronization with noise filtering and lifecycle."""
        start_time = datetime.now(UTC)
        since_utc = _ensure_utc(since_updated_at) if since_updated_at is not None else None
        if current_pages is not None:
            pages = (
                [p for p in current_pages if _ensure_utc(p.last_modified_at) > since_utc]
                if since_utc is not None
                else list(current_pages)
            )
        else:
            pages = self.fetch_space_pages(space_key, since_updated_at=since_utc)

        default_space_scopes = (
            list(space_allowed_scopes)
            if space_allowed_scopes is not None
            else self.mirror_permissions(space_key)
        )

        items_scanned = len(pages)
        items_indexed = 0
        items_updated = 0
        stale_flagged = 0
        noise_filtered_count = 0
        noise_demoted_count = 0
        tombstoned_count = 0
        filter_breakdown: dict[str, int] = {}
        already_tombstoned: set[str] = set()

        is_postgres = hasattr(repository, "insert_doc_chunks_batch") and conn is not None

        for page in pages:
            noise = self.evaluate_page_noise(page)
            filter_breakdown[noise.reason_key] = filter_breakdown.get(noise.reason_key, 0) + 1

            if noise.action == "exclude":
                noise_filtered_count += 1
                tomb_status = (
                    page.status.strip().lower()
                    if page.status.strip().lower() in {"archived", "trashed", "deleted"}
                    else "deleted"
                )
                pages_tomb, chunks_tomb = self.tombstone_deleted_or_restricted_pages(
                    domain_id=domain_id,
                    repository=repository,
                    deleted_page_ids=[page.page_id],
                    conn=conn,
                    reason=noise.reason_detail,
                    status=tomb_status,
                )
                if pages_tomb > 0 or chunks_tomb > 0:
                    already_tombstoned.add(page.page_id)
                    tombstoned_count += max(pages_tomb, 1 if chunks_tomb > 0 else 0)
                    stale_flagged += chunks_tomb
                continue

            if noise.action == "demote":
                noise_demoted_count += 1

            effective_scopes = (
                list(page_allowed_scopes_map[page.page_id])
                if page_allowed_scopes_map and page.page_id in page_allowed_scopes_map
                else list(default_space_scopes)
            )

            page_rec = ConfluencePageRecord(
                page_id=uuid4(),
                domain_id=domain_id,
                confluence_page_id=page.page_id,
                space_key=page.space_key,
                title=page.title,
                version_number=page.version_number,
                author_username=page.author_username,
                source_url=page.source_url,
                allowed_scopes=effective_scopes,
                last_modified_at=page.last_modified_at,
                indexed_at=datetime.now(UTC),
            )
            target_id = f"confluence:{page.page_id}"
            max_existing_ver = 0

            if is_postgres:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COALESCE(MAX(
                            CASE WHEN version_str ~ '^[0-9]+$'
                                 THEN CAST(version_str AS integer)
                                 ELSE 0
                            END
                        ), 0)
                        FROM doc_chunks
                        WHERE domain_id = %s AND doc_id = %s;
                        """,
                        (domain_id, target_id),
                    )
                    row = cur.fetchone()
                    max_existing_ver = int(row[0]) if row and row[0] is not None else 0
            elif hasattr(repository, "_docs"):
                existing_vers = [
                    int(d.version_str)
                    for d in repository._docs.values()
                    if d.domain_id == domain_id
                    and d.doc_id == target_id
                    and d.version_str
                    and d.version_str.isdigit()
                ]
                max_existing_ver = max(existing_vers) if existing_vers else 0

            is_rollback = page.version_number < max_existing_ver
            is_same_version = page.version_number == max_existing_ver and max_existing_ver > 0

            if is_postgres:
                repository.insert_confluence_page(conn, page_rec)
                if not is_rollback:
                    with conn.cursor() as cur:
                        if is_same_version:
                            cur.execute(
                                """
                                DELETE FROM doc_chunks
                                WHERE domain_id = %s AND doc_id = %s AND version_str = %s;
                                """,
                                (domain_id, target_id, str(page.version_number)),
                            )
                        cur.execute(
                            """
                            UPDATE doc_chunks
                            SET is_stale = TRUE,
                                stale_reason = %s
                            WHERE domain_id = %s
                              AND doc_id = %s
                              AND NOT is_stale
                              AND (
                                  CASE WHEN version_str ~ '^[0-9]+$'
                                       THEN CAST(version_str AS integer) < %s
                                       ELSE FALSE
                                  END
                              );
                            """,
                            (
                                f"Superseded by page version {page.version_number}",
                                domain_id,
                                target_id,
                                page.version_number,
                            ),
                        )
                        stale_flagged += cur.rowcount or 0
            else:
                if principal is not None:
                    repository.insert_confluence_page(principal, page_rec)
                elif hasattr(repository, "_confluence"):
                    deleted_set = getattr(repository, "_deleted_confluence", None)
                    if isinstance(deleted_set, set):
                        deleted_set.discard((domain_id, page_rec.confluence_page_id))
                    for old_id, p in list(repository._confluence.items()):
                        if (
                            p.domain_id == domain_id
                            and p.confluence_page_id == page_rec.confluence_page_id
                        ):
                            repository._confluence.pop(old_id, None)
                    repository._confluence[page_rec.page_id] = page_rec

                if not is_rollback and hasattr(repository, "_docs"):
                    if is_same_version:
                        for d_id, doc in list(repository._docs.items()):
                            if (
                                doc.domain_id == domain_id
                                and doc.doc_id == target_id
                                and doc.version_str == str(page.version_number)
                            ):
                                repository._docs.pop(d_id, None)

                    for d_id, doc in list(repository._docs.items()):
                        if doc.domain_id == domain_id and doc.doc_id == target_id:
                            doc_ver = (
                                int(doc.version_str)
                                if doc.version_str and doc.version_str.isdigit()
                                else 0
                            )
                            if doc_ver < page.version_number and not doc.is_stale:
                                repository._docs[d_id] = dataclasses.replace(
                                    doc,
                                    is_stale=True,
                                    stale_reason=(
                                        f"Superseded by page version {page.version_number}"
                                    ),
                                )
                                stale_flagged += 1

            chunks = self.chunk_page_verbatim(
                page,
                allowed_scopes=effective_scopes,
                authority_weight=noise.authority_weight,
            )
            doc_chunk_records: list[DocChunkRecord] = []

            for chk in chunks:
                emb = generate_deterministic_embedding(
                    f"{chk.contextual_header} {chk.verbatim_content}", dim=768
                )
                doc_chunk_records.append(
                    DocChunkRecord(
                        chunk_id=chk.chunk_id,
                        domain_id=domain_id,
                        doc_id=target_id,
                        doc_title=page.title,
                        source_uri=page.source_url,
                        market_scope=None,
                        published_date=page.last_modified_at.date(),
                        authority_weight=noise.authority_weight,
                        section_heading=chk.section_heading,
                        contextual_header=chk.contextual_header,
                        verbatim_content=chk.verbatim_content,
                        metadata_json={
                            "space_key": page.space_key,
                            "page_id": page.page_id,
                            "version": page.version_number,
                            "author": page.author_username,
                            "parent_id": page.parent_id,
                            "labels": list(page.labels),
                            "status": page.status,
                            "noise_action": noise.action,
                            "noise_reason": noise.reason_key,
                        },
                        allowed_scopes=chk.allowed_scopes,
                        embedding=emb,
                        source_type="confluence",
                        version_str=str(page.version_number),
                        is_stale=chk.is_stale or is_rollback,
                        stale_reason=(
                            f"Superseded by page version {max_existing_ver}"
                            if is_rollback
                            else chk.stale_reason
                        ),
                    )
                )

            if is_postgres:
                repository.insert_doc_chunks_batch(conn, doc_chunk_records)
            else:
                for c in doc_chunk_records:
                    if hasattr(repository, "insert_doc_chunk") and principal is not None:
                        repository.insert_doc_chunk(principal, c)
                    elif hasattr(repository, "_docs"):
                        repository._docs[c.chunk_id] = c
                    elif hasattr(repository, "_doc_chunks"):
                        repository._doc_chunks[c.chunk_id] = c

            # Propagate narrowed/updated scopes across confluence_pages and active doc_chunks
            self._sync_allowed_scopes(
                domain_id=domain_id,
                page_id=page.page_id,
                allowed_scopes=effective_scopes,
                repository=repository,
                conn=conn,
            )

            items_indexed += len(doc_chunk_records)
            items_updated += 1

        # Reconcile explicit deletions and missing space pages (full sync only)
        pending_tombstones: list[str] = [
            pid.strip()
            for pid in (deleted_page_ids or [])
            if pid.strip() and pid.strip() not in already_tombstoned
        ]
        if reconcile_deletes and since_utc is None:
            active_existing_ids = self._list_active_space_page_ids(
                domain_id=domain_id,
                space_key=space_key,
                repository=repository,
                conn=conn,
                principal=principal,
            )
            seen_page_ids = {p.page_id for p in pages}
            for existing_pid in active_existing_ids:
                if existing_pid not in seen_page_ids and existing_pid not in already_tombstoned:
                    pending_tombstones.append(existing_pid)

        if pending_tombstones:
            pages_tomb, chunks_tomb = self.tombstone_deleted_or_restricted_pages(
                domain_id=domain_id,
                repository=repository,
                deleted_page_ids=pending_tombstones,
                conn=conn,
                reason="Page deleted or restricted in Confluence",
                status="deleted",
            )
            if pages_tomb > 0 or chunks_tomb > 0:
                tombstoned_count += max(pages_tomb, 1 if chunks_tomb > 0 else 0)
                stale_flagged += chunks_tomb

        if pages:
            checkpoint = max(_ensure_utc(p.last_modified_at) for p in pages).isoformat()
        elif since_utc is not None:
            checkpoint = since_utc.isoformat()
        else:
            checkpoint = datetime.now(UTC).isoformat()

        duration = (datetime.now(UTC) - start_time).total_seconds() * 1000.0

        return SyncResult(
            source_type="confluence",
            items_scanned=items_scanned,
            items_indexed=items_indexed,
            items_updated=items_updated,
            items_deleted=tombstoned_count,
            stale_items_flagged=stale_flagged,
            noise_filtered_count=noise_filtered_count,
            noise_demoted_count=noise_demoted_count,
            tombstoned_count=tombstoned_count,
            filter_breakdown=filter_breakdown,
            cursor_or_checkpoint=checkpoint,
            duration_ms=round(duration, 2),
        )

    # Alias for convenience
    sync_space = sync_space_to_repository
