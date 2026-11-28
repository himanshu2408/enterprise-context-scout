"""Stale-Content Benchmark Evaluation Dataset (EVAL-02, N >= 100).

Statistically justified sample size N=100 guaranteeing that observing 0 false executions
proves true stale error rate <= 3.0% at 95% confidence under Rule of Three.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

MIN_STALE_SAMPLE_SIZE: Final[int] = 100


class StaleContentType(StrEnum):
    SUPERSEDED_CONFLUENCE_PAGE = "SUPERSEDED_CONFLUENCE_PAGE"
    CLOSED_ARCHIVED_JIRA_TICKET = "CLOSED_ARCHIVED_JIRA_TICKET"
    OUTDATED_GIT_COMMIT_SHA = "OUTDATED_GIT_COMMIT_SHA"
    DOC_CODE_SPEC_CONFLICT = "DOC_CODE_SPEC_CONFLICT"


class StaleItem(BaseModel):
    """Single stale item test record."""

    model_config = ConfigDict(frozen=True)

    item_id: str = Field(min_length=1)
    content_type: StaleContentType
    source_identifier: str = Field(min_length=1)
    cited_version_or_status: str
    current_head_version_or_status: str
    is_stale: bool = True
    abstention_required: bool = True
    sha256_hash: str = ""

    def compute_hash(self) -> str:
        payload = (
            f"{self.item_id}|{self.content_type}|{self.source_identifier}|"
            f"{self.cited_version_or_status}|{self.current_head_version_or_status}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_stale_content_set(sample_size: int = MIN_STALE_SAMPLE_SIZE) -> list[StaleItem]:
    """Generates N >= 100 deterministic stale-content evaluation records."""
    if sample_size < MIN_STALE_SAMPLE_SIZE:
        msg = f"Sample size {sample_size} violates threshold (N >= {MIN_STALE_SAMPLE_SIZE})"
        raise ValueError(msg)

    items: list[StaleItem] = []

    # 1. Superseded Confluence Pages (30 items)
    for i in range(30):
        cited_v = 1
        head_v = cited_v + 1 + (i % 4)
        page_id = f"CONF-PAGE-{i:03d}"
        item = StaleItem(
            item_id=f"STALE-CONF-{i:03d}",
            content_type=StaleContentType.SUPERSEDED_CONFLUENCE_PAGE,
            source_identifier=f"spaces/ENG/pages/{page_id}",
            cited_version_or_status=f"v{cited_v}",
            current_head_version_or_status=f"v{head_v}",
            is_stale=True,
        )
        items.append(item.model_copy(update={"sha256_hash": item.compute_hash()}))

    # 2. Closed / Archived Jira Tickets (30 items)
    statuses = ["Closed", "Done", "Cancelled", "Archived", "Rejected"]
    for i in range(30):
        status = statuses[i % len(statuses)]
        key = f"PROJ-{1000 + i}"
        item = StaleItem(
            item_id=f"STALE-JIRA-{i:03d}",
            content_type=StaleContentType.CLOSED_ARCHIVED_JIRA_TICKET,
            source_identifier=f"jira/issues/{key}",
            cited_version_or_status=status,
            current_head_version_or_status=status,
            is_stale=True,
        )
        items.append(item.model_copy(update={"sha256_hash": item.compute_hash()}))

    # 3. Outdated Git Commit SHAs (25 items)
    for i in range(25):
        cited_sha = hashlib.sha1(f"old_commit_{i}".encode()).hexdigest()[:12]
        head_sha = hashlib.sha1(f"head_commit_{i}".encode()).hexdigest()[:12]
        repo_name = f"enterprise/service-mesh-{i:02d}"
        item = StaleItem(
            item_id=f"STALE-GIT-{i:03d}",
            content_type=StaleContentType.OUTDATED_GIT_COMMIT_SHA,
            source_identifier=f"{repo_name}/src/worker.py",
            cited_version_or_status=cited_sha,
            current_head_version_or_status=head_sha,
            is_stale=True,
        )
        items.append(item.model_copy(update={"sha256_hash": item.compute_hash()}))

    # 4. Doc-Code Spec Conflicts (15 items)
    for i in range(15):
        doc_ref = f"doc_auth_rfc_{i:02d}"
        item = StaleItem(
            item_id=f"STALE-DOC-CODE-{i:03d}",
            content_type=StaleContentType.DOC_CODE_SPEC_CONFLICT,
            source_identifier=f"urn:enterprise:rfc:{doc_ref}",
            cited_version_or_status="DeprecatedAuthAPIv1",
            current_head_version_or_status="OAuth2PKCEAuthAPIv2",
            is_stale=True,
        )
        items.append(item.model_copy(update={"sha256_hash": item.compute_hash()}))

    # Extra items if sample_size > 100
    for i in range(len(items), sample_size):
        item = StaleItem(
            item_id=f"STALE-EXTRA-{i:03d}",
            content_type=StaleContentType.SUPERSEDED_CONFLUENCE_PAGE,
            source_identifier=f"spaces/ARCH/pages/PAGE-EXTRA-{i}",
            cited_version_or_status="v1",
            current_head_version_or_status="v3",
            is_stale=True,
        )
        items.append(item.model_copy(update={"sha256_hash": item.compute_hash()}))

    return items
