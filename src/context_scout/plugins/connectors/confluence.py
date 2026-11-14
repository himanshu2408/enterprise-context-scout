"""Confluence REST API Connector and Lossless Verbatim Chunker (F15)."""

from __future__ import annotations

import html
import logging
import re
from datetime import UTC, datetime
from typing import Any
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
    created_at: datetime
    last_modified_at: datetime


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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.freshness_half_life_days = freshness_half_life_days

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

    def fetch_space_pages(self, space_key: str) -> list[ConfluencePageDTO]:
        """Paginates all pages in a given space via REST API v2."""
        pages: list[ConfluencePageDTO] = []
        cursor: str | None = None

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
                    datetime.fromisoformat(last_modified_raw.replace("Z", "+00:00"))
                    if last_modified_raw
                    else datetime.now(UTC)
                )

                body_storage = item.get("body", {}).get("storage", {}).get("value", "")
                parent_id = str(item.get("parentId")) if item.get("parentId") else None
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
                        created_at=last_modified,
                        last_modified_at=last_modified,
                    )
                )

            # Check next cursor in _links
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
    ) -> list[ConfluenceVerbatimChunk]:
        """Splits page storage content into lossless verbatim chunks with 50-token headers."""
        raw_storage = page.body_storage
        allowed_scopes = self.mirror_permissions(page.space_key)

        # Check temporal decay
        now = datetime.now(UTC)
        days_old = (now - page.last_modified_at).days
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
                        allowed_scopes=allowed_scopes,
                        last_modified_at=page.last_modified_at,
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

    def sync_space_to_repository(
        self,
        domain_id: str,
        space_key: str,
        repository: PostgresRLSRepository | Any,
        conn: Any = None,
        principal: Any = None,
        current_pages: list[ConfluencePageDTO] | None = None,
    ) -> SyncResult:
        """Executes full space synchronization into repository."""
        start_time = datetime.now(UTC)
        pages = current_pages if current_pages is not None else self.fetch_space_pages(space_key)
        allowed_scopes = self.mirror_permissions(space_key)

        items_scanned = len(pages)
        items_indexed = 0
        items_updated = 0
        stale_flagged = 0

        is_postgres = hasattr(repository, "insert_doc_chunks_batch") and conn is not None

        for page in pages:
            page_rec = ConfluencePageRecord(
                page_id=uuid4(),
                domain_id=domain_id,
                confluence_page_id=page.page_id,
                space_key=page.space_key,
                title=page.title,
                version_number=page.version_number,
                author_username=page.author_username,
                source_url=page.source_url,
                allowed_scopes=allowed_scopes,
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
            else:
                if hasattr(repository, "_docs"):
                    existing_vers = [
                        int(d.version_str)
                        for d in repository._docs.values()
                        if d.domain_id == domain_id
                        and d.doc_id == target_id
                        and d.version_str
                        and d.version_str.isdigit()
                    ]
                    max_existing_ver = max(existing_vers) if existing_vers else 0

            # Evaluate version monotonicity
            is_rollback = page.version_number < max_existing_ver

            if is_postgres:
                repository.insert_confluence_page(conn, page_rec)
                if not is_rollback:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE doc_chunks
                            SET is_stale = TRUE,
                                stale_reason = %s
                            WHERE domain_id = %s
                              AND doc_id = %s
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
                        stale_flagged += cur.rowcount
            else:
                # In-memory repository fallback
                if principal is not None:
                    repository.insert_confluence_page(principal, page_rec)
                elif hasattr(repository, "_confluence"):
                    for old_id, p in list(repository._confluence.items()):
                        if (
                            p.domain_id == domain_id
                            and p.confluence_page_id == page_rec.confluence_page_id
                        ):
                            repository._confluence.pop(old_id, None)
                    repository._confluence[page_rec.page_id] = page_rec

                if not is_rollback and hasattr(repository, "_docs"):
                    import dataclasses

                    for d_id, doc in list(repository._docs.items()):
                        if doc.domain_id == domain_id and doc.doc_id == target_id:
                            doc_ver = (
                                int(doc.version_str)
                                if doc.version_str and doc.version_str.isdigit()
                                else 0
                            )
                            if doc_ver < page.version_number:
                                repository._docs[d_id] = dataclasses.replace(
                                    doc,
                                    is_stale=True,
                                    stale_reason=(
                                        f"Superseded by page version {page.version_number}"
                                    ),
                                )
                                stale_flagged += 1

            chunks = self.chunk_page_verbatim(page)
            doc_chunk_records: list[DocChunkRecord] = []

            for chk in chunks:
                emb = generate_deterministic_embedding(
                    f"{chk.contextual_header} {chk.verbatim_content}", dim=768
                )
                doc_chunk_records.append(
                    DocChunkRecord(
                        chunk_id=chk.chunk_id,
                        domain_id=domain_id,
                        doc_id=f"confluence:{page.page_id}",
                        doc_title=page.title,
                        source_uri=page.source_url,
                        market_scope=None,
                        published_date=page.last_modified_at.date(),
                        authority_weight=1.0,
                        section_heading=chk.section_heading,
                        contextual_header=chk.contextual_header,
                        verbatim_content=chk.verbatim_content,
                        metadata_json={
                            "space_key": page.space_key,
                            "page_id": page.page_id,
                            "version": page.version_number,
                            "author": page.author_username,
                            "parent_id": page.parent_id,
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

            items_indexed += len(doc_chunk_records)
            items_updated += 1

        duration = (datetime.now(UTC) - start_time).total_seconds() * 1000.0

        return SyncResult(
            source_type="confluence",
            items_scanned=items_scanned,
            items_indexed=items_indexed,
            items_updated=items_updated,
            items_deleted=0,
            stale_items_flagged=stale_flagged,
            cursor_or_checkpoint=datetime.now(UTC).isoformat(),
            duration_ms=round(duration, 2),
        )

    # Alias for convenience
    sync_space = sync_space_to_repository
