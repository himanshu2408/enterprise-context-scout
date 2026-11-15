"""Jira Cloud REST API Connector and Citation Contract Specification (F16)."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field

from context_scout.db.repositories import PostgresRLSRepository
from context_scout.db.schema import DocChunkRecord, JiraIssueRecord
from context_scout.ingestion.git_merkle_indexer import generate_deterministic_embedding
from context_scout.plugins.base import SourceConnectorPlugin, SyncResult
from context_scout.security.permission_mirror import (
    JiraPermission,
    PermissionMirror,
    sanitize_scope_identifier,
)

logger = logging.getLogger(__name__)

STALE_STATUSES = frozenset(
    {
        "closed",
        "archived",
        "obsolete",
        "resolved (obsolete)",
        "resolved(obsolete)",
        "cancelled",
        "rejected",
        "wontfix",
        "won't fix",
        "wont fix",
        "won't do",
        "wont do",
        "wontdo",
        "abandoned",
    }
)
STALE_RESOLUTIONS = frozenset(
    {
        "won't fix",
        "wont fix",
        "wontfix",
        "won't do",
        "wont do",
        "wontdo",
        "abandoned",
        "duplicate",
        "incomplete",
        "cannot reproduce",
        "declined",
        "rejected",
        "cancelled",
        "obsolete",
    }
)


class JiraIssueLink(BaseModel):
    """Directed dependency between two Jira issues."""

    target_key: str
    link_type: str  # e.g. "blocks", "is_blocked_by", "relates_to", "parent", "subtask"
    target_status: str | None = None


class JiraIssueDTO(BaseModel):
    """Pydantic representation of an upstream Jira issue."""

    issue_id: str
    issue_key: str = Field(pattern=r"^[A-Z][A-Z0-9]+-[0-9]+$")
    project_key: str
    summary: str
    issue_type: str
    status: str
    resolution: str | None = None
    assignee: str | None = None
    reporter: str | None = None
    source_url: str
    description_text: str
    acceptance_criteria: str | None = None
    linked_issues: list[JiraIssueLink] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class JiraConnectorPlugin(SourceConnectorPlugin):
    """Production Jira Cloud REST API v3 connector."""

    def __init__(
        self,
        base_url: str = "https://mock.atlassian.net",
        email: str | None = None,
        api_token: str | None = None,
        bearer_token: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")

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
        return "jira"

    @property
    def source_type(self) -> str:
        return "jira"

    def mirror_permissions(self, container_key: str) -> list[str]:
        """Maps Jira project key to RLS allowed_scopes."""
        clean_key = sanitize_scope_identifier(container_key)
        return PermissionMirror.map_jira_permissions_to_scopes(
            clean_key, [JiraPermission.BROWSE]
        )

    def fetch_item(self, item_id: str, version: str | None = None) -> dict[str, Any] | None:
        """Fetches a specific issue by key from Jira REST API v3."""
        endpoint = f"/rest/api/3/issue/{item_id}"
        params = {"expand": "renderedFields,names,changelog"}
        resp = self.client.get(endpoint, params=params)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def search_issues_incremental(
        self,
        project_key: str,
        since_datetime: datetime | None = None,
        batch_size: int = 50,
    ) -> list[JiraIssueDTO]:
        """Paginates issues using JQL."""
        clean_project = sanitize_scope_identifier(project_key)
        jql = f"project = '{clean_project}'"
        if since_datetime:
            jql_time = since_datetime.strftime("%Y-%m-%d %H:%M")
            jql += f" AND updated >= '{jql_time}'"
        jql += " ORDER BY updated ASC"

        issues: list[JiraIssueDTO] = []
        start_at = 0

        while True:
            payload = {
                "jql": jql,
                "startAt": start_at,
                "maxResults": batch_size,
                "fields": [
                    "summary",
                    "issuetype",
                    "status",
                    "resolution",
                    "assignee",
                    "reporter",
                    "created",
                    "updated",
                    "description",
                    "issuelinks",
                    "parent",
                    "subtasks",
                    "customfield_10014",  # Common acceptance criteria field
                ],
            }
            resp = self.client.post("/rest/api/3/search", json=payload)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("issues", [])
            if not results:
                break

            for item in results:
                parsed = self._parse_jira_issue(item, clean_project)
                issues.append(parsed)

            start_at += len(results)
            total = data.get("total", 0)
            if start_at >= total:
                break

        return issues

    def _parse_jira_issue(self, raw: dict[str, Any], fallback_project: str) -> JiraIssueDTO:
        fields = raw.get("fields", {})
        issue_key = raw.get("key", "")
        project_key = (
            fields.get("project", {}).get("key") or issue_key.split("-")[0] or fallback_project
        )

        status_name = fields.get("status", {}).get("name", "Unknown")
        res_obj = fields.get("resolution")
        resolution_name = res_obj.get("name") if res_obj else None

        assignee_obj = fields.get("assignee")
        assignee_name = (
            assignee_obj.get("displayName") or assignee_obj.get("accountId")
            if assignee_obj
            else None
        )
        reporter_obj = fields.get("reporter")
        reporter_name = (
            reporter_obj.get("displayName") or reporter_obj.get("accountId")
            if reporter_obj
            else None
        )

        created_raw = fields.get("created", "")
        updated_raw = fields.get("updated", "")
        created_at = (
            datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
            if created_raw
            else datetime.now(UTC)
        )
        updated_at = (
            datetime.fromisoformat(updated_raw.replace("Z", "+00:00"))
            if updated_raw
            else datetime.now(UTC)
        )

        desc_raw = fields.get("description")
        desc_text = self._extract_text_from_adf_or_string(desc_raw)

        ac_raw = fields.get("customfield_10014")
        ac_text = self._extract_text_from_adf_or_string(ac_raw) if ac_raw else None
        if not ac_text:
            ac_match = re.search(
                r"(?i)(?:acceptance criteria|conditions of satisfaction)[:\n]+(.*?)(?=\n#{1,3}|\Z)",
                desc_text,
                re.DOTALL,
            )
            if ac_match:
                ac_text = ac_match.group(1).strip()

        links: list[JiraIssueLink] = []
        for lk in fields.get("issuelinks", []):
            if "outwardIssue" in lk:
                target = lk["outwardIssue"]["key"]
                rel = lk.get("type", {}).get("outward", "relates_to").lower().replace(" ", "_")
                st = lk["outwardIssue"].get("fields", {}).get("status", {}).get("name")
                links.append(
                    JiraIssueLink(target_key=target, link_type=rel, target_status=st)
                )
            elif "inwardIssue" in lk:
                target = lk["inwardIssue"]["key"]
                rel = lk.get("type", {}).get("inward", "relates_to").lower().replace(" ", "_")
                st = lk["inwardIssue"].get("fields", {}).get("status", {}).get("name")
                links.append(
                    JiraIssueLink(target_key=target, link_type=rel, target_status=st)
                )

        if "parent" in fields:
            parent_key = fields["parent"].get("key")
            if parent_key:
                links.append(JiraIssueLink(target_key=parent_key, link_type="parent"))
        for st in fields.get("subtasks", []):
            st_key = st.get("key")
            if st_key:
                links.append(JiraIssueLink(target_key=st_key, link_type="subtask"))

        source_url = f"{self.base_url}/browse/{issue_key}"

        return JiraIssueDTO(
            issue_id=str(raw.get("id", "")),
            issue_key=issue_key,
            project_key=project_key,
            summary=str(fields.get("summary", "")),
            issue_type=str(fields.get("issuetype", {}).get("name", "Task")),
            status=status_name,
            resolution=resolution_name,
            assignee=assignee_name,
            reporter=reporter_name,
            source_url=source_url,
            description_text=desc_text,
            acceptance_criteria=ac_text,
            linked_issues=links,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _extract_text_from_adf_or_string(self, content: Any) -> str:
        """Recursively extracts plain text from Atlassian Document Format (ADF) or strings."""
        if not content:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, dict):
            text_parts: list[str] = []
            if content.get("type") == "text":
                return str(content.get("text", ""))
            for child in content.get("content", []):
                child_text = self._extract_text_from_adf_or_string(child)
                if child_text:
                    text_parts.append(child_text)
            sep = (
                "\n"
                if content.get("type") in {
                    "paragraph",
                    "bulletList",
                    "orderedList",
                    "listItem",
                    "heading",
                    "codeBlock",
                }
                else " "
            )
            return sep.join(text_parts)
        if isinstance(content, list):
            return "\n".join(self._extract_text_from_adf_or_string(item) for item in content)
        return str(content)

    def is_ticket_stale(self, issue: JiraIssueDTO) -> tuple[bool, str | None]:
        """Evaluates whether ticket is closed, archived, or resolved won't-fix/obsolete.

        Completed tickets (e.g. status='Done' or 'Resolved' with resolution='Done'/'Fixed'
        or unresolved) are fresh and valid for citation unless explicitly obsolete.
        """
        status_clean = issue.status.strip().lower()
        status_norm = status_clean.replace(" ", "").replace("'", "").replace("-", "")
        if status_clean in STALE_STATUSES or status_norm in STALE_STATUSES:
            res_desc = f" with resolution '{issue.resolution}'" if issue.resolution else ""
            return True, f"Ticket status is '{issue.status}'{res_desc}"

        if issue.resolution:
            res_clean = issue.resolution.strip().lower()
            res_norm = res_clean.replace(" ", "").replace("'", "").replace("-", "")
            if res_clean in STALE_RESOLUTIONS or res_norm in STALE_RESOLUTIONS:
                return True, f"Ticket resolved as '{issue.resolution}'"

        return False, None

    def chunk_issue_verbatim(
        self,
        issue: JiraIssueDTO,
        domain_id: str = "default",
    ) -> list[DocChunkRecord]:
        """Splits Jira issue into verbatim chunks with 50-token contextual headers."""
        is_stale, stale_reason = self.is_ticket_stale(issue)
        allowed_scopes = self.mirror_permissions(issue.project_key)
        doc_chunks: list[DocChunkRecord] = []

        ctx_base = (
            f"This document is Jira ticket {issue.issue_key} in project '{issue.project_key}' "
            f"(Type: {issue.issue_type}, Status: {issue.status}, "
            f"Resolution: {issue.resolution or 'Unresolved'}, "
            f"Updated: {issue.updated_at.strftime('%Y-%m-%d')} "
            f"by {issue.assignee or 'Unassigned'}). Summary: {issue.summary}."
        )

        # Chunk A: Description
        if issue.description_text:
            header_desc = f"{ctx_base} Section: 'Description'."
            emb_desc = generate_deterministic_embedding(
                f"{header_desc} {issue.description_text}", dim=768
            )
            doc_chunks.append(
                DocChunkRecord(
                    chunk_id=uuid4(),
                    domain_id=domain_id,
                    doc_id=f"jira:{issue.issue_key}",
                    doc_title=f"{issue.issue_key}: {issue.summary}",
                    source_uri=issue.source_url,
                    market_scope=None,
                    published_date=issue.updated_at.date(),
                    authority_weight=1.0,
                    section_heading="Description",
                    contextual_header=header_desc,
                    verbatim_content=issue.description_text,
                    metadata_json={
                        "project_key": issue.project_key,
                        "issue_key": issue.issue_key,
                        "issue_type": issue.issue_type,
                        "status": issue.status,
                        "resolution": issue.resolution,
                        "linked_issues": [
                            link_item.model_dump() for link_item in issue.linked_issues
                        ],
                    },
                    allowed_scopes=allowed_scopes,
                    embedding=emb_desc,
                    source_type="jira",
                    version_str=str(int(issue.updated_at.timestamp())),
                    is_stale=is_stale,
                    stale_reason=stale_reason,
                )
            )

        # Chunk B: Acceptance Criteria
        if issue.acceptance_criteria:
            header_ac = f"{ctx_base} Section: 'Acceptance Criteria'."
            emb_ac = generate_deterministic_embedding(
                f"{header_ac} {issue.acceptance_criteria}", dim=768
            )
            doc_chunks.append(
                DocChunkRecord(
                    chunk_id=uuid4(),
                    domain_id=domain_id,
                    doc_id=f"jira:{issue.issue_key}",
                    doc_title=f"{issue.issue_key}: {issue.summary}",
                    source_uri=issue.source_url,
                    market_scope=None,
                    published_date=issue.updated_at.date(),
                    authority_weight=1.0,
                    section_heading="Acceptance Criteria",
                    contextual_header=header_ac,
                    verbatim_content=issue.acceptance_criteria,
                    metadata_json={
                        "project_key": issue.project_key,
                        "issue_key": issue.issue_key,
                        "section": "acceptance_criteria",
                        "status": issue.status,
                    },
                    allowed_scopes=allowed_scopes,
                    embedding=emb_ac,
                    source_type="jira",
                    version_str=str(int(issue.updated_at.timestamp())),
                    is_stale=is_stale,
                    stale_reason=stale_reason,
                )
            )

        return doc_chunks

    def sync_project_to_repository(
        self,
        domain_id: str,
        project_key: str,
        repository: PostgresRLSRepository | Any,
        conn: Any = None,
        principal: Any = None,
        since_datetime: datetime | None = None,
    ) -> SyncResult:
        """Syncs Jira project issues into repository."""
        start_time = datetime.now(UTC)
        issues = self.search_issues_incremental(project_key, since_datetime=since_datetime)
        allowed_scopes = self.mirror_permissions(project_key)

        items_scanned = len(issues)
        items_indexed = 0
        items_updated = 0
        stale_flagged = 0

        is_postgres = hasattr(repository, "insert_doc_chunks_batch") and conn is not None

        for issue in issues:
            is_stale, _ = self.is_ticket_stale(issue)
            if is_stale:
                stale_flagged += 1

            issue_rec = JiraIssueRecord(
                issue_id=uuid4(),
                domain_id=domain_id,
                issue_key=issue.issue_key,
                project_key=issue.project_key,
                summary=issue.summary,
                issue_type=issue.issue_type,
                status=issue.status,
                resolution=issue.resolution,
                assignee_username=issue.assignee,
                source_url=issue.source_url,
                allowed_scopes=allowed_scopes,
                updated_at=issue.updated_at,
                indexed_at=datetime.now(UTC),
            )
            if is_postgres:
                repository.insert_jira_issue(conn, issue_rec)
            else:
                if principal is not None:
                    repository.insert_jira_issue(principal, issue_rec)
                elif hasattr(repository, "_jira"):
                    for old_id, i in list(repository._jira.items()):
                        if i.domain_id == domain_id and i.issue_key == issue_rec.issue_key:
                            repository._jira.pop(old_id, None)
                    repository._jira[issue_rec.issue_id] = issue_rec

            doc_chunks_to_insert = self.chunk_issue_verbatim(issue, domain_id)

            if doc_chunks_to_insert:
                if is_postgres:
                    repository.insert_doc_chunks_batch(conn, doc_chunks_to_insert)
                else:
                    for c in doc_chunks_to_insert:
                        if hasattr(repository, "insert_doc_chunk") and principal is not None:
                            repository.insert_doc_chunk(principal, c)
                        elif hasattr(repository, "_docs"):
                            # Purge old chunks for this ticket to ensure freshness
                            target_doc_id = f"jira:{issue.issue_key}"
                            for d_id, doc in list(repository._docs.items()):
                                if doc.domain_id == domain_id and doc.doc_id == target_doc_id:
                                    repository._docs.pop(d_id, None)
                            repository._docs[c.chunk_id] = c
                        elif hasattr(repository, "_doc_chunks"):
                            repository._doc_chunks[c.chunk_id] = c

                items_indexed += len(doc_chunks_to_insert)

            items_updated += 1

        duration = (datetime.now(UTC) - start_time).total_seconds() * 1000.0

        return SyncResult(
            source_type="jira",
            items_scanned=items_scanned,
            items_indexed=items_indexed,
            items_updated=items_updated,
            items_deleted=0,
            stale_items_flagged=stale_flagged,
            cursor_or_checkpoint=datetime.now(UTC).isoformat(),
            duration_ms=round(duration, 2),
        )
