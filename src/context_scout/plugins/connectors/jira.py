"""Jira Cloud REST API Connector and Citation Contract Specification (F16)."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, Field

from context_scout.db.repositories import PostgresRLSRepository
from context_scout.db.schema import DocChunkRecord, JiraIssueRecord
from context_scout.plugins.base import SourceConnectorPlugin, SyncResult
from context_scout.retrieval.embedder import get_embedding_provider
from context_scout.schemas.contracts import DEFAULT_TENANT_ID
from context_scout.security.permission_mirror import (
    JiraPermission,
    PermissionMirror,
    sanitize_scope_identifier,
)
from context_scout.security.pre_index_scrubber import PreIndexScrubber

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


class JiraCommentDTO(BaseModel):
    """Pydantic representation of a Jira issue comment with optional visibility ACL."""

    comment_id: str
    author: str = "unknown"
    body_text: str
    created_at: datetime
    updated_at: datetime
    visibility_type: str | None = None  # "role" or "group"
    visibility_value: str | None = None
    is_visibility_malformed: bool = False


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
    epic_key: str | None = None
    parent_issue_key: str | None = None
    linked_mrs: list[str] = Field(default_factory=list)
    security_level_id: str | None = None
    security_level_name: str | None = None
    security_level_principals: tuple[str, ...] | None = None
    browse_principals: tuple[str, ...] | None = None
    comments: list[JiraCommentDTO] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


_JIRA_KEY_EXACT_RE = re.compile(r"^[A-Z][A-Z0-9]+-[0-9]+$")
_GL_MR_URL_RE = re.compile(r"https?://[^\s)\]'\"<>]+/-/merge_requests/\d+")
_GH_PR_URL_RE = re.compile(r"https?://[^\s)\]'\"<>]+/pull/\d+")
_GL_MR_SHORT_RE = re.compile(
    r"(?:^|[\s(\[,])((?:[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)?![0-9]+)(?=$|[\s)\],.;:])"
)
_GH_PR_SHORT_RE = re.compile(r"\b([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#[0-9]+)\b")


def extract_linked_mrs(
    *,
    raw: dict[str, Any] | None = None,
    fields: dict[str, Any] | None = None,
    texts: Sequence[str | None] = (),
) -> list[str]:
    """Extract deduplicated GitLab MR / GitHub PR references from Jira payloads & prose."""
    found: list[str] = []
    seen: set[str] = set()

    def _add(ref: str) -> None:
        cleaned = ref.strip().rstrip(".,;:)")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            found.append(cleaned)

    for container in (raw, fields):
        if not isinstance(container, dict):
            continue
        dev_mrs = container.get("customfield_dev_mrs")
        if isinstance(dev_mrs, list):
            for entry in dev_mrs:
                if isinstance(entry, str):
                    _add(entry)
                elif isinstance(entry, dict) and entry.get("url"):
                    _add(str(entry["url"]))
        remotelinks = container.get("remotelinks")
        if isinstance(remotelinks, list):
            for rl in remotelinks:
                if isinstance(rl, str):
                    _add(rl)
                elif isinstance(rl, dict):
                    obj = rl.get("object") if isinstance(rl.get("object"), dict) else rl
                    url_val = str(obj.get("url", "")) if isinstance(obj, dict) else ""
                    if "/merge_requests/" in url_val or "/pull/" in url_val:
                        _add(url_val)

    for text in texts:
        if not text:
            continue
        for pat in (_GL_MR_URL_RE, _GH_PR_URL_RE, _GL_MR_SHORT_RE, _GH_PR_SHORT_RE):
            for m in pat.finditer(text):
                val = m.group(1) if m.lastindex else m.group(0)
                _add(val)

    return found


class JiraConnectorPlugin(SourceConnectorPlugin):
    """Production Jira Cloud REST API v3 connector."""

    def __init__(
        self,
        base_url: str = "https://mock.atlassian.net",
        email: str | None = None,
        api_token: str | None = None,
        bearer_token: str | None = None,
        client: httpx.Client | None = None,
        scrubber: PreIndexScrubber | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.scrubber = scrubber or PreIndexScrubber()

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
        return PermissionMirror.map_jira_permissions_to_scopes(clean_key, [JiraPermission.BROWSE])

    def fetch_item(self, item_id: str, version: str | None = None) -> dict[str, Any] | None:
        """Fetches a specific issue by key from Jira REST API v3."""
        endpoint = f"/rest/api/3/issue/{item_id}"
        params = {"expand": "renderedFields,names,changelog"}
        resp = self.client.get(endpoint, params=params)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data: Any = resp.json()
        return data if isinstance(data, dict) else None

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
                    "security",
                    "comment",
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
                links.append(JiraIssueLink(target_key=target, link_type=rel, target_status=st))
            elif "inwardIssue" in lk:
                target = lk["inwardIssue"]["key"]
                rel = lk.get("type", {}).get("inward", "relates_to").lower().replace(" ", "_")
                st = lk["inwardIssue"].get("fields", {}).get("status", {}).get("name")
                links.append(JiraIssueLink(target_key=target, link_type=rel, target_status=st))

        parent_issue_key: str | None = None
        epic_key: str | None = None
        parent_obj = fields.get("parent")
        if isinstance(parent_obj, dict) and parent_obj.get("key"):
            parent_key = str(parent_obj["key"]).strip()
            parent_issue_key = parent_key
            links.append(JiraIssueLink(target_key=parent_key, link_type="parent"))
            parent_type = (
                parent_obj.get("fields", {}).get("issuetype", {}).get("name", "")
                if isinstance(parent_obj.get("fields"), dict)
                else ""
            )
            if str(parent_type).strip().lower() == "epic":
                epic_key = parent_key

        if epic_key is None:
            for epic_cand in (
                fields.get("epic"),
                fields.get("customfield_10014"),
                fields.get("customfield_10008"),
                fields.get("epic_link"),
            ):
                if isinstance(epic_cand, dict) and epic_cand.get("key"):
                    cand_str = str(epic_cand["key"]).strip()
                    if _JIRA_KEY_EXACT_RE.fullmatch(cand_str):
                        epic_key = cand_str
                        break
                elif isinstance(epic_cand, str):
                    cand_str = epic_cand.strip()
                    if _JIRA_KEY_EXACT_RE.fullmatch(cand_str):
                        epic_key = cand_str
                        break

        for st in fields.get("subtasks", []):
            st_key = st.get("key")
            if st_key:
                links.append(JiraIssueLink(target_key=st_key, link_type="subtask"))

        issue_type_name = str(fields.get("issuetype", {}).get("name", "Task"))
        if epic_key is None and issue_type_name.strip().lower() != "epic":
            for lk in links:
                if "epic" in lk.link_type:
                    epic_key = lk.target_key
                    break

        source_url = f"{self.base_url}/browse/{issue_key}"

        security_level_id, security_level_name, security_level_principals = (
            self._parse_security_field(fields.get("security"))
        )
        comments = self._parse_comments_field(
            fields.get("comment"), default_created=created_at, default_updated=updated_at
        )
        summary_str = str(fields.get("summary", ""))
        linked_mrs = extract_linked_mrs(
            raw=raw,
            fields=fields,
            texts=[summary_str, desc_text, ac_text, *(c.body_text for c in comments)],
        )

        return JiraIssueDTO(
            issue_id=str(raw.get("id", "")),
            issue_key=issue_key,
            project_key=project_key,
            summary=summary_str,
            issue_type=issue_type_name,
            status=status_name,
            resolution=resolution_name,
            assignee=assignee_name,
            reporter=reporter_name,
            source_url=source_url,
            description_text=desc_text,
            acceptance_criteria=ac_text,
            linked_issues=links,
            epic_key=epic_key,
            parent_issue_key=parent_issue_key,
            linked_mrs=linked_mrs,
            security_level_id=security_level_id,
            security_level_name=security_level_name,
            security_level_principals=security_level_principals,
            comments=comments,
            created_at=created_at,
            updated_at=updated_at,
        )

    @staticmethod
    def _parse_security_field(
        sec_raw: Any,
    ) -> tuple[str | None, str | None, tuple[str, ...] | None]:
        """Parses Jira issue security level field, failing closed on malformed payloads."""
        if sec_raw is None:
            return None, None, None
        if not isinstance(sec_raw, dict):
            return "__malformed__", None, ()

        raw_id = sec_raw.get("id")
        raw_name = sec_raw.get("name")
        sec_id = str(raw_id).strip() if raw_id is not None and str(raw_id).strip() else None
        sec_name = str(raw_name).strip() if raw_name is not None and str(raw_name).strip() else None
        if sec_id is None and sec_name is None:
            return "__malformed__", None, ()

        raw_principals = (
            sec_raw.get("principals")
            if "principals" in sec_raw
            else sec_raw.get("allowedPrincipals")
        )
        sec_principals: tuple[str, ...] | None = None
        if raw_principals is not None:
            if isinstance(raw_principals, (list, tuple, set)):
                sec_principals = tuple(str(p).strip() for p in raw_principals if str(p).strip())
            else:
                sec_principals = ()

        return sec_id, sec_name, sec_principals

    def _parse_comments_field(
        self,
        comment_block: Any,
        *,
        default_created: datetime,
        default_updated: datetime,
    ) -> list[JiraCommentDTO]:
        """Parses Jira REST v3 comment list and fails closed on malformed visibility."""
        if not isinstance(comment_block, dict):
            return []
        raw_comments = comment_block.get("comments", [])
        if not isinstance(raw_comments, list):
            return []

        parsed_comments: list[JiraCommentDTO] = []
        for idx, c in enumerate(raw_comments):
            if not isinstance(c, dict):
                continue
            comment_id = str(c.get("id") or f"comment-{idx}")
            author_obj = c.get("author")
            author = "unknown"
            if isinstance(author_obj, dict):
                author = str(
                    author_obj.get("displayName") or author_obj.get("accountId") or "unknown"
                )
            body_text = self._extract_text_from_adf_or_string(c.get("body"))

            c_created_raw = c.get("created")
            c_updated_raw = c.get("updated")
            c_created = (
                datetime.fromisoformat(str(c_created_raw).replace("Z", "+00:00"))
                if isinstance(c_created_raw, str) and c_created_raw
                else default_created
            )
            c_updated = (
                datetime.fromisoformat(str(c_updated_raw).replace("Z", "+00:00"))
                if isinstance(c_updated_raw, str) and c_updated_raw
                else c_created or default_updated
            )

            vis_raw = c.get("visibility")
            vis_type: str | None = None
            vis_val: str | None = None
            is_malformed = False
            if vis_raw is not None:
                if not isinstance(vis_raw, dict):
                    is_malformed = True
                else:
                    v_type_raw = vis_raw.get("type")
                    v_type_str = (
                        str(v_type_raw).strip().lower() if isinstance(v_type_raw, str) else ""
                    )
                    v_val_raw = (
                        vis_raw.get("value")
                        if vis_raw.get("value") is not None
                        else vis_raw.get("identifier")
                    )
                    v_val_str = str(v_val_raw).strip() if v_val_raw is not None else ""
                    if v_type_str not in {"role", "group"} or not v_val_str:
                        is_malformed = True
                        vis_type = v_type_str or None
                        vis_val = v_val_str or None
                    else:
                        try:
                            sanitize_scope_identifier(v_val_str.replace(" ", "_"))
                            vis_type = v_type_str
                            vis_val = v_val_str
                        except ValueError:
                            is_malformed = True
                            vis_type = v_type_str
                            vis_val = v_val_str

            parsed_comments.append(
                JiraCommentDTO(
                    comment_id=comment_id,
                    author=author,
                    body_text=body_text,
                    created_at=c_created,
                    updated_at=c_updated,
                    visibility_type=vis_type,
                    visibility_value=vis_val,
                    is_visibility_malformed=is_malformed,
                )
            )
        return parsed_comments

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
                if content.get("type")
                in {
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

    @staticmethod
    def _sanitize_principals(principals: tuple[str, ...]) -> set[str]:
        """Sanitizes a tuple of principal identifiers; raises ValueError on invalid chars."""
        return {sanitize_scope_identifier(p.strip()) for p in principals if p.strip()}

    @staticmethod
    def _security_level_slug(issue: JiraIssueDTO) -> str | None:
        """Builds a sanitized security-level slug when an Issue Security Level is set."""
        if issue.security_level_id is None and issue.security_level_name is None:
            return None
        if issue.security_level_id == "__malformed__":
            raise ValueError("Malformed security level payload")
        raw_sec = (
            issue.security_level_id
            if issue.security_level_id is not None
            else (issue.security_level_name or "")
        )
        return sanitize_scope_identifier(raw_sec.strip().replace(" ", "_"))

    def resolve_issue_allowed_scopes(self, issue: JiraIssueDTO) -> list[str]:
        """Computes BROWSE ∩ Issue Security Level RLS allowed_scopes (fail-closed)."""
        clean_proj = sanitize_scope_identifier(issue.project_key)
        fail_closed = [f"jira:project:{clean_proj}:restricted:none"]

        has_security_level = (
            issue.security_level_id is not None or issue.security_level_name is not None
        )
        if not has_security_level:
            if issue.browse_principals is None:
                return self.mirror_permissions(clean_proj)
            if not issue.browse_principals:
                return fail_closed
            try:
                browse_set = self._sanitize_principals(issue.browse_principals)
            except ValueError:
                return fail_closed
            if not browse_set:
                return fail_closed
            return sorted(f"jira:project:{clean_proj}:principal:{p}:read" for p in browse_set)

        try:
            sec_slug = self._security_level_slug(issue)
        except ValueError:
            return fail_closed
        if not sec_slug:
            return fail_closed

        if issue.security_level_principals is not None:
            try:
                sec_set = self._sanitize_principals(issue.security_level_principals)
                if issue.browse_principals is not None:
                    browse_set = self._sanitize_principals(issue.browse_principals)
                    effective = sec_set & browse_set
                else:
                    effective = sec_set
            except ValueError:
                return fail_closed
            if not effective:
                return fail_closed
            return sorted(
                f"jira:project:{clean_proj}:security_level:{sec_slug}:principal:{p}:read"
                for p in effective
            )

        if issue.browse_principals is not None:
            if not issue.browse_principals:
                return fail_closed
            try:
                browse_set = self._sanitize_principals(issue.browse_principals)
            except ValueError:
                return fail_closed
            if not browse_set:
                return fail_closed
            return sorted(
                f"jira:project:{clean_proj}:security_level:{sec_slug}:principal:{p}:read"
                for p in browse_set
            )

        return [f"jira:project:{clean_proj}:security_level:{sec_slug}:read"]

    def resolve_comment_allowed_scopes(
        self, issue: JiraIssueDTO, comment: JiraCommentDTO
    ) -> list[str]:
        """Computes BROWSE ∩ Issue Security Level ∩ Comment Visibility scopes (fail-closed)."""
        clean_proj = sanitize_scope_identifier(issue.project_key)
        fail_closed = [f"jira:project:{clean_proj}:restricted:none"]

        if comment.is_visibility_malformed:
            return fail_closed

        issue_scopes = self.resolve_issue_allowed_scopes(issue)
        if issue_scopes == fail_closed:
            return fail_closed

        if comment.visibility_type is None and comment.visibility_value is None:
            return issue_scopes
        if comment.visibility_type is None or comment.visibility_value is None:
            return fail_closed

        vtype = comment.visibility_type.strip().lower()
        raw_val = comment.visibility_value.strip()
        if vtype not in {"role", "group"} or not raw_val:
            return fail_closed

        try:
            clean_val = sanitize_scope_identifier(raw_val.replace(" ", "_"))
        except ValueError:
            return fail_closed

        # If explicit principal lists use role:/group: prefixes, enforce intersection.
        effective_principals: set[str] | None = None
        if issue.security_level_principals is not None and issue.browse_principals is not None:
            effective_principals = self._sanitize_principals(
                issue.security_level_principals
            ) & self._sanitize_principals(issue.browse_principals)
        elif issue.security_level_principals is not None:
            effective_principals = self._sanitize_principals(issue.security_level_principals)
        elif issue.browse_principals is not None:
            effective_principals = self._sanitize_principals(issue.browse_principals)

        if (
            effective_principals is not None
            and any(p.startswith(("role:", "group:")) for p in effective_principals)
            and f"{vtype}:{clean_val}" not in effective_principals
        ):
            return fail_closed

        sec_slug = self._security_level_slug(issue)
        if sec_slug:
            return [f"jira:project:{clean_proj}:security_level:{sec_slug}:{vtype}:{clean_val}:read"]
        return [f"jira:project:{clean_proj}:{vtype}:{clean_val}:read"]

    def chunk_issue_verbatim(
        self,
        issue: JiraIssueDTO,
        domain_id: str = "default",
        *,
        tenant_id: UUID = DEFAULT_TENANT_ID,
    ) -> list[DocChunkRecord]:
        """Splits Jira issue into scrubbed verbatim chunks with 50-token headers."""
        is_stale, stale_reason = self.is_ticket_stale(issue)
        allowed_scopes = self.resolve_issue_allowed_scopes(issue)
        doc_chunks: list[DocChunkRecord] = []

        scrubbed_summary = self.scrubber.scrub_text(
            issue.summary, source_name=f"jira:{issue.issue_key}:summary"
        ).scrubbed_text
        scrubbed_desc = (
            self.scrubber.scrub_text(
                issue.description_text, source_name=f"jira:{issue.issue_key}:description"
            ).scrubbed_text
            if issue.description_text
            else ""
        )
        scrubbed_ac = (
            self.scrubber.scrub_text(
                issue.acceptance_criteria, source_name=f"jira:{issue.issue_key}:ac"
            ).scrubbed_text
            if issue.acceptance_criteria
            else None
        )
        resolved_mrs = issue.linked_mrs or extract_linked_mrs(
            texts=[
                issue.summary,
                issue.description_text,
                issue.acceptance_criteria,
                *(c.body_text for c in issue.comments),
            ]
        )

        ctx_base = (
            f"This document is Jira ticket {issue.issue_key} in project '{issue.project_key}' "
            f"(Type: {issue.issue_type}, Status: {issue.status}, "
            f"Resolution: {issue.resolution or 'Unresolved'}, "
            f"Updated: {issue.updated_at.strftime('%Y-%m-%d')} "
            f"by {issue.assignee or 'Unassigned'}). Summary: {scrubbed_summary}."
        )

        embedder = get_embedding_provider()

        # Chunk A: Description
        if scrubbed_desc:
            header_desc = f"{ctx_base} Section: 'Description'."
            emb_desc = embedder.embed_text(f"{header_desc} {scrubbed_desc}", is_query=False)
            doc_chunks.append(
                DocChunkRecord(
                    chunk_id=uuid4(),
                    domain_id=domain_id,
                    doc_id=f"jira:{issue.issue_key}",
                    doc_title=f"{issue.issue_key}: {scrubbed_summary}",
                    source_uri=issue.source_url,
                    market_scope=None,
                    published_date=issue.updated_at.date(),
                    authority_weight=1.0,
                    section_heading="Description",
                    contextual_header=header_desc,
                    verbatim_content=scrubbed_desc,
                    metadata_json={
                        "project_key": issue.project_key,
                        "issue_key": issue.issue_key,
                        "issue_type": issue.issue_type,
                        "status": issue.status,
                        "resolution": issue.resolution,
                        "epic_key": issue.epic_key,
                        "parent_issue_key": issue.parent_issue_key,
                        "linked_mrs": list(resolved_mrs),
                        "security_level_id": issue.security_level_id,
                        "security_level_name": issue.security_level_name,
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
                    tenant_id=tenant_id,
                )
            )

        # Chunk B: Acceptance Criteria
        if scrubbed_ac:
            header_ac = f"{ctx_base} Section: 'Acceptance Criteria'."
            emb_ac = embedder.embed_text(f"{header_ac} {scrubbed_ac}", is_query=False)
            doc_chunks.append(
                DocChunkRecord(
                    chunk_id=uuid4(),
                    domain_id=domain_id,
                    doc_id=f"jira:{issue.issue_key}",
                    doc_title=f"{issue.issue_key}: {scrubbed_summary}",
                    source_uri=issue.source_url,
                    market_scope=None,
                    published_date=issue.updated_at.date(),
                    authority_weight=1.0,
                    section_heading="Acceptance Criteria",
                    contextual_header=header_ac,
                    verbatim_content=scrubbed_ac,
                    metadata_json={
                        "project_key": issue.project_key,
                        "issue_key": issue.issue_key,
                        "section": "acceptance_criteria",
                        "status": issue.status,
                        "epic_key": issue.epic_key,
                        "parent_issue_key": issue.parent_issue_key,
                        "linked_mrs": list(resolved_mrs),
                        "security_level_id": issue.security_level_id,
                        "security_level_name": issue.security_level_name,
                    },
                    allowed_scopes=allowed_scopes,
                    embedding=emb_ac,
                    source_type="jira",
                    version_str=str(int(issue.updated_at.timestamp())),
                    is_stale=is_stale,
                    stale_reason=stale_reason,
                    tenant_id=tenant_id,
                )
            )

        # Chunk C+: Comments with individual role/group visibility ACLs
        for comment in issue.comments:
            if not comment.body_text.strip():
                continue
            scrubbed_comment = self.scrubber.scrub_text(
                comment.body_text,
                source_name=f"jira:{issue.issue_key}:comment:{comment.comment_id}",
            ).scrubbed_text
            comment_scopes = self.resolve_comment_allowed_scopes(issue, comment)
            header_comment = (
                f"{ctx_base} Section: 'Comment:{comment.comment_id}' by {comment.author}."
            )
            emb_comment = embedder.embed_text(
                f"{header_comment} {scrubbed_comment}", is_query=False
            )
            doc_chunks.append(
                DocChunkRecord(
                    chunk_id=uuid4(),
                    domain_id=domain_id,
                    doc_id=f"jira:{issue.issue_key}",
                    doc_title=f"{issue.issue_key}: {scrubbed_summary}",
                    source_uri=issue.source_url,
                    market_scope=None,
                    published_date=comment.updated_at.date(),
                    authority_weight=1.0,
                    section_heading=f"Comment:{comment.comment_id}",
                    contextual_header=header_comment,
                    verbatim_content=scrubbed_comment,
                    metadata_json={
                        "project_key": issue.project_key,
                        "issue_key": issue.issue_key,
                        "comment_id": comment.comment_id,
                        "author": comment.author,
                        "epic_key": issue.epic_key,
                        "parent_issue_key": issue.parent_issue_key,
                        "linked_mrs": list(resolved_mrs),
                        "visibility_type": comment.visibility_type,
                        "visibility_value": comment.visibility_value,
                        "is_visibility_malformed": comment.is_visibility_malformed,
                        "security_level_id": issue.security_level_id,
                        "security_level_name": issue.security_level_name,
                    },
                    allowed_scopes=comment_scopes,
                    embedding=emb_comment,
                    source_type="jira",
                    version_str=str(int(comment.updated_at.timestamp())),
                    is_stale=is_stale,
                    stale_reason=stale_reason,
                    tenant_id=tenant_id,
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

        items_scanned = len(issues)
        items_indexed = 0
        items_updated = 0
        stale_flagged = 0

        effective_tenant_id: UUID = (
            getattr(principal, "tenant_id", DEFAULT_TENANT_ID)
            if principal is not None
            else DEFAULT_TENANT_ID
        )
        is_postgres = hasattr(repository, "insert_doc_chunks_batch") and conn is not None

        for issue in issues:
            is_stale, _ = self.is_ticket_stale(issue)
            if is_stale:
                stale_flagged += 1

            issue_allowed_scopes = self.resolve_issue_allowed_scopes(issue)
            scrubbed_summary = self.scrubber.scrub_text(
                issue.summary, source_name=f"jira:{issue.issue_key}:summary"
            ).scrubbed_text
            resolved_mrs = issue.linked_mrs or extract_linked_mrs(
                texts=[
                    issue.summary,
                    issue.description_text,
                    issue.acceptance_criteria,
                    *(c.body_text for c in issue.comments),
                ]
            )
            issue_rec = JiraIssueRecord(
                issue_id=uuid4(),
                domain_id=domain_id,
                issue_key=issue.issue_key,
                project_key=issue.project_key,
                summary=scrubbed_summary,
                issue_type=issue.issue_type,
                status=issue.status,
                resolution=issue.resolution,
                assignee_username=issue.assignee,
                source_url=issue.source_url,
                allowed_scopes=issue_allowed_scopes,
                updated_at=issue.updated_at,
                indexed_at=datetime.now(UTC),
                tenant_id=effective_tenant_id,
                epic_key=issue.epic_key,
                parent_issue_key=issue.parent_issue_key,
                linked_issues=[lk.model_dump() for lk in issue.linked_issues],
                linked_mrs=list(resolved_mrs),
            )
            if is_postgres:
                repository.insert_jira_issue(conn, issue_rec)
            else:
                if principal is not None:
                    repository.insert_jira_issue(principal, issue_rec)
                elif hasattr(repository, "_jira"):
                    for old_id, i in list(repository._jira.items()):
                        if (
                            i.tenant_id == effective_tenant_id
                            and i.domain_id == domain_id
                            and i.issue_key == issue_rec.issue_key
                        ):
                            repository._jira.pop(old_id, None)
                    repository._jira[issue_rec.issue_id] = issue_rec

            doc_chunks_to_insert = self.chunk_issue_verbatim(
                issue, domain_id, tenant_id=effective_tenant_id
            )

            if doc_chunks_to_insert:
                if is_postgres:
                    repository.insert_doc_chunks_batch(conn, doc_chunks_to_insert)
                else:
                    if principal is None and hasattr(repository, "_docs"):
                        # Purge old chunks once per ticket before inserting all new chunks
                        target_doc_id = f"jira:{issue.issue_key}"
                        for d_id, doc in list(repository._docs.items()):
                            if (
                                doc.tenant_id == effective_tenant_id
                                and doc.domain_id == domain_id
                                and doc.doc_id == target_doc_id
                            ):
                                repository._docs.pop(d_id, None)

                    for c in doc_chunks_to_insert:
                        if hasattr(repository, "insert_doc_chunk") and principal is not None:
                            repository.insert_doc_chunk(principal, c)
                        elif hasattr(repository, "_docs"):
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

    def build_epic_story_mr_hierarchy(
        self,
        issues: Sequence[JiraIssueDTO | JiraIssueRecord],
        *,
        allowed_scopes: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Build a strict ACL-bounded Epic -> Story -> Subtask -> MR hierarchy."""
        return build_epic_story_mr_hierarchy(
            issues,
            allowed_scopes=allowed_scopes,
            scope_resolver=self.resolve_issue_allowed_scopes,
        )


def build_epic_story_mr_hierarchy(
    issues: Sequence[JiraIssueDTO | JiraIssueRecord],
    *,
    allowed_scopes: Sequence[str] | None = None,
    scope_resolver: Any | None = None,
) -> dict[str, Any]:
    """Assemble Epic -> Story -> Subtask -> MR hierarchy strictly bounded by allowed_scopes."""
    principal_scopes = (
        {s.strip() for s in allowed_scopes if s and s.strip()}
        if allowed_scopes is not None
        else None
    )

    normalized: list[dict[str, Any]] = []
    for item in issues:
        if isinstance(item, JiraIssueDTO):
            scopes = (
                scope_resolver(item)
                if scope_resolver is not None
                else list(item.browse_principals or ())
            )
            linked_mrs = list(
                item.linked_mrs
                or extract_linked_mrs(
                    texts=[
                        item.summary,
                        item.description_text,
                        item.acceptance_criteria,
                        *(c.body_text for c in item.comments),
                    ]
                )
            )
            linked_list = [lk.model_dump() for lk in item.linked_issues]
            entry: dict[str, Any] = {
                "issue_key": item.issue_key,
                "summary": item.summary,
                "issue_type": item.issue_type,
                "status": item.status,
                "epic_key": item.epic_key,
                "parent_issue_key": item.parent_issue_key,
                "linked_mrs": linked_mrs,
                "linked_issues": linked_list,
                "allowed_scopes": list(scopes),
            }
        else:
            scopes = list(item.allowed_scopes)
            entry = {
                "issue_key": item.issue_key,
                "summary": item.summary,
                "issue_type": item.issue_type,
                "status": item.status,
                "epic_key": item.epic_key,
                "parent_issue_key": item.parent_issue_key,
                "linked_mrs": list(item.linked_mrs),
                "linked_issues": list(item.linked_issues),
                "allowed_scopes": scopes,
            }

        if principal_scopes is not None:
            item_scopes = {s.strip() for s in entry["allowed_scopes"] if s and s.strip()}
            if not item_scopes.intersection(principal_scopes):
                continue
        normalized.append(entry)

    by_key = {e["issue_key"]: e for e in normalized}
    epics_map: dict[str, dict[str, Any]] = {}
    for e in normalized:
        if str(e["issue_type"]).strip().lower() == "epic":
            epics_map[e["issue_key"]] = {
                "epic_key": e["issue_key"],
                "summary": e["summary"],
                "status": e["status"],
                "linked_mrs": list(e["linked_mrs"]),
                "stories": [],
                "all_linked_mrs": set(e["linked_mrs"]),
            }

    subtasks_by_parent: dict[str, list[dict[str, Any]]] = {}
    stories: list[dict[str, Any]] = []
    for e in normalized:
        if e["issue_key"] in epics_map:
            continue
        it_low = str(e["issue_type"]).strip().lower()
        p_key = e.get("parent_issue_key")
        if (
            it_low in {"sub-task", "subtask"}
            and p_key
            and p_key in by_key
            and p_key not in epics_map
        ):
            subtasks_by_parent.setdefault(p_key, []).append(
                {
                    "issue_key": e["issue_key"],
                    "summary": e["summary"],
                    "issue_type": e["issue_type"],
                    "status": e["status"],
                    "linked_mrs": sorted(set(e["linked_mrs"])),
                }
            )
        else:
            stories.append(e)

    orphans: list[dict[str, Any]] = []
    for st in stories:
        target_epic = st.get("epic_key")
        if not target_epic and st.get("parent_issue_key") in epics_map:
            target_epic = st["parent_issue_key"]

        subs = sorted(subtasks_by_parent.get(st["issue_key"], []), key=lambda x: x["issue_key"])
        combined_story_mrs = sorted(
            set(st["linked_mrs"]).union(*(set(s["linked_mrs"]) for s in subs))
        )
        story_node = {
            "issue_key": st["issue_key"],
            "summary": st["summary"],
            "issue_type": st["issue_type"],
            "status": st["status"],
            "linked_mrs": combined_story_mrs,
            "subtasks": subs,
        }
        if target_epic and target_epic in epics_map:
            epic_bucket = epics_map[target_epic]
            epic_bucket["stories"].append(story_node)
            epic_bucket["all_linked_mrs"].update(combined_story_mrs)
        else:
            # Never leak details of an invisible/restricted Epic in orphan output.
            orphans.append(story_node)

    epics_out: list[dict[str, Any]] = []
    for epic_key in sorted(epics_map):
        bucket = epics_map[epic_key]
        bucket["stories"].sort(key=lambda s: s["issue_key"])
        epics_out.append(
            {
                "epic_key": bucket["epic_key"],
                "summary": bucket["summary"],
                "status": bucket["status"],
                "linked_mrs": sorted(set(bucket["linked_mrs"])),
                "stories": bucket["stories"],
                "all_linked_mrs": sorted(bucket["all_linked_mrs"]),
            }
        )

    orphans.sort(key=lambda o: o["issue_key"])
    return {"epics": epics_out, "orphan_issues": orphans}
