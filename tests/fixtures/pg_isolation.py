"""Ephemeral per-session PostgreSQL tenant domain helpers.

Provides strict ``eval_*`` domain creation and child-to-parent teardown so live
PostgreSQL tests never mutate seeded benchmark corpora (``trading_code_scout``,
``quant_research_scout``, ``polyglot_unseen_v2``).
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Final

import psycopg

EPHEMERAL_PG_DOMAIN_PREFIX: Final[str] = "eval_"

_TENANT_TABLES_CHILD_TO_PARENT: Final[tuple[str, ...]] = (
    "ast_edges",
    "ast_symbols",
    "code_files",
    "repositories",
    "confluence_pages",
    "jira_issues",
    "doc_chunks",
    "execution_telemetry",
    "indexing_queue",
    "indexing_dead_letter",
    "domain_packs",
)


def _sanitize_suite_slug(suite: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", suite.lower()).strip("_")
    return slug[:32] or "suite"


def make_ephemeral_pg_domain_id(suite: str = "suite") -> str:
    """Return a unique ephemeral PostgreSQL domain identifier with ``eval_`` prefix."""
    slug = _sanitize_suite_slug(suite)
    return f"{EPHEMERAL_PG_DOMAIN_PREFIX}{slug}_{uuid.uuid4().hex[:8]}"


def validate_ephemeral_pg_domain_id(domain_id: str) -> str:
    """Refuse any domain_id that is not a well-formed ``eval_*`` session domain."""
    if (
        not isinstance(domain_id, str)
        or not domain_id.startswith(EPHEMERAL_PG_DOMAIN_PREFIX)
        or len(domain_id) <= len(EPHEMERAL_PG_DOMAIN_PREFIX)
    ):
        raise ValueError(
            f"Refusing PostgreSQL isolation operation for non-ephemeral domain_id={domain_id!r}. "
            f"Only '{EPHEMERAL_PG_DOMAIN_PREFIX}<suite>_<uuid>' domains are permitted."
        )
    return domain_id


def create_ephemeral_pg_domain(
    conn: psycopg.Connection[Any],
    suite: str = "suite",
    *,
    domain_id: str | None = None,
    display_name: str | None = None,
    lora_adapter_id: str = "eval-none",
    primary_truth_source: str = "code",
    freshness_half_life_days: int = 180,
    config: dict[str, Any] | None = None,
) -> str:
    """Register a unique ``eval_*`` domain in ``domain_packs`` and commit."""
    resolved_id = (
        validate_ephemeral_pg_domain_id(domain_id)
        if domain_id is not None
        else make_ephemeral_pg_domain_id(suite)
    )
    resolved_name = display_name or f"Eval Domain ({resolved_id})"
    resolved_config = json.dumps(config or {})
    conn.execute(
        """
        INSERT INTO domain_packs
            (domain_id, display_name, lora_adapter_id, primary_truth_source,
             freshness_half_life_days, config_json)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (domain_id) DO NOTHING
        """,
        (
            resolved_id,
            resolved_name,
            lora_adapter_id,
            primary_truth_source,
            freshness_half_life_days,
            resolved_config,
        ),
    )
    conn.commit()
    return resolved_id


def cleanup_ephemeral_pg_domain(
    conn: psycopg.Connection[Any],
    domain_id: str,
) -> dict[str, int]:
    """Delete all rows belonging strictly to ``domain_id`` in child-to-parent order."""
    validated_id = validate_ephemeral_pg_domain_id(domain_id)
    deleted: dict[str, int] = {}
    repo_commits_cur = conn.execute(
        """
        DELETE FROM repo_commits
        WHERE repo_id IN (SELECT repo_id FROM repositories WHERE domain_id = %s)
        """,
        (validated_id,),
    )
    deleted["repo_commits"] = max(repo_commits_cur.rowcount or 0, 0)
    for table in _TENANT_TABLES_CHILD_TO_PARENT:
        cur = conn.execute(f"DELETE FROM {table} WHERE domain_id = %s", (validated_id,))
        deleted[table] = max(cur.rowcount or 0, 0)
    conn.commit()
    return deleted
