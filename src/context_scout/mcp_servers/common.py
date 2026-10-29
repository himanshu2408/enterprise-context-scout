"""Shared MCP Utilities: Out-of-band context, payload caps, and Postgres graph adapter."""

from __future__ import annotations

import copy
import json
from typing import Any, Final
from uuid import UUID

from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.schemas.contracts import PrincipalContext
from context_scout.security.rbac import (
    get_current_principal,
    scoped_db_session,
    validate_tool_call_arguments,
)

MAX_PAYLOAD_ITEMS: Final[int] = 10
MAX_PAYLOAD_BYTES: Final[int] = 32768  # 32 KB hard serialized JSON cap
TRUNCATION_INDICATOR: Final[str] = "[PAYLOAD_TRUNCATED_AT_32KB]"


def bound_payload_dict(
    payload: dict[str, Any],
    max_items: int = MAX_PAYLOAD_ITEMS,
    max_bytes: int = MAX_PAYLOAD_BYTES,
) -> dict[str, Any]:
    """Clamps item counts to <= max_items and enforces <= max_bytes serialized JSON limit."""
    data = copy.deepcopy(payload)

    # 1. Clamp any collection fields to max_items
    collection_keys = (
        "items",
        "matches",
        "symbols",
        "hits",
        "issues",
        "equations",
        "tables",
        "comparison",
    )
    for key in collection_keys:
        if key in data and isinstance(data[key], list) and len(data[key]) > max_items:
            data[key] = data[key][:max_items]

    # 2. Check byte size
    raw_json = json.dumps(data, default=str)
    raw_bytes = len(raw_json.encode("utf-8"))
    if raw_bytes <= max_bytes:
        return data

    # 3. Truncate long text fields within collections
    text_fields = (
        "verbatim_content",
        "raw_content",
        "snippet",
        "excerpt",
        "contextual_summary",
        "signature_skeleton",
        "content",
    )
    for key in collection_keys:
        if key in data and isinstance(data[key], list):
            items_list = data[key]
            for item in items_list:
                if isinstance(item, dict):
                    for tf in text_fields:
                        if tf in item and isinstance(item[tf], str) and len(item[tf]) > 300:
                            item[tf] = item[tf][:300] + "... [TRUNCATED]"

    data["payload_truncated"] = True
    data["truncation_notice"] = TRUNCATION_INDICATOR

    raw_json = json.dumps(data, default=str)
    raw_bytes = len(raw_json.encode("utf-8"))
    if raw_bytes <= max_bytes:
        return data

    # 4. Progressively shed items from list until payload satisfies limit
    for key in collection_keys:
        if key in data and isinstance(data[key], list) and len(data[key]) > 1:
            while len(data[key]) > 1:
                data[key].pop()
                if len(json.dumps(data, default=str).encode("utf-8")) <= max_bytes:
                    return data

    # 5. Fallback: if single item exceeds limit, aggressively truncate its text fields
    for key in collection_keys:
        if key in data and isinstance(data[key], list) and data[key]:
            single_item = data[key][0]
            if isinstance(single_item, dict):
                for tf in text_fields:
                    if tf in single_item and isinstance(single_item[tf], str):
                        single_item[tf] = single_item[tf][:150] + "... " + TRUNCATION_INDICATOR

    # Ensure final JSON is strictly within max_bytes
    final_json = json.dumps(data, default=str)
    if len(final_json.encode("utf-8")) > max_bytes:
        # Ultimate guarantee: minimal truncated payload
        return {
            "payload_truncated": True,
            "truncation_notice": TRUNCATION_INDICATOR,
            "error": "Payload exceeded 32 KB threshold and was truncated to minimal representation",
            "retained_metadata": {
                k: v for k, v in data.items() if not isinstance(v, list) and len(str(v)) < 100
            },
        }

    return data


class PostgresScoutGraphAdapter:
    """Live PostgreSQL 17 graph repository adapter conforming to ScoutGraphRepository (SEC-03)."""

    def __init__(
        self,
        connection_manager: RLSConnectionManager,
        settings: ScoutSettings | None = None,
    ) -> None:
        self.connection_manager = connection_manager
        self.settings = settings or ScoutSettings()

    def list_visible_code_files(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[CodeFileRecord]:
        """Returns all code files visible to the principal under Forced RLS."""
        with scoped_db_session(self.connection_manager, principal) as conn:
            rows = conn.execute(
                """
                SELECT file_id, domain_id, repo_name, file_path, git_commit_sha,
                       git_blob_sha, language, allowed_scopes, raw_content, line_count
                FROM code_files
                WHERE domain_id = %s
                ORDER BY file_path;
                """,
                (domain_id,),
            ).fetchall()
            results: list[CodeFileRecord] = []
            for r in rows:
                scopes_raw = r[7]
                scopes_list = (
                    [str(s) for s in scopes_raw] if isinstance(scopes_raw, (list, tuple)) else []
                )
                results.append(
                    CodeFileRecord(
                        file_id=UUID(str(r[0])),
                        domain_id=str(r[1]),
                        repo_name=str(r[2]),
                        file_path=str(r[3]),
                        git_commit_sha=str(r[4]),
                        git_blob_sha=str(r[5]),
                        language=str(r[6]),
                        allowed_scopes=scopes_list,
                        raw_content=str(r[8]),
                        line_count=int(str(r[9])),
                    )
                )
            return results

    def list_visible_ast_symbols(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[ASTSymbolRecord]:
        """Returns all AST symbols visible to the principal under Forced RLS."""
        with scoped_db_session(self.connection_manager, principal) as conn:
            rows = conn.execute(
                """
                SELECT symbol_id, file_id, domain_id, qualified_name, symbol_kind,
                       start_line, end_line, signature_skeleton, contextual_summary,
                       allowed_scopes, embedding
                FROM ast_symbols
                WHERE domain_id = %s
                ORDER BY qualified_name;
                """,
                (domain_id,),
            ).fetchall()
            results: list[ASTSymbolRecord] = []
            for r in rows:
                raw_emb = r[10]
                if isinstance(raw_emb, str):
                    emb_floats = [
                        float(x.strip()) for x in raw_emb.strip("[]").split(",") if x.strip()
                    ]
                elif hasattr(raw_emb, "tolist"):
                    emb_floats = [float(x) for x in raw_emb.tolist()]
                elif isinstance(raw_emb, (list, tuple)):
                    emb_floats = [float(x) for x in raw_emb]
                else:
                    emb_floats = []

                scopes_raw = r[9]
                scopes_list = (
                    [str(s) for s in scopes_raw] if isinstance(scopes_raw, (list, tuple)) else []
                )
                kind_str = str(r[4])
                results.append(
                    ASTSymbolRecord(
                        symbol_id=UUID(str(r[0])),
                        file_id=UUID(str(r[1])),
                        domain_id=str(r[2]),
                        qualified_name=str(r[3]),
                        symbol_kind=kind_str,  # type: ignore[arg-type]
                        start_line=int(str(r[5])),
                        end_line=int(str(r[6])),
                        signature_skeleton=str(r[7]),
                        contextual_summary=str(r[8]),
                        allowed_scopes=scopes_list,
                        embedding=emb_floats,
                    )
                )
            return results

    def list_visible_ast_edges(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[ASTEdgeRecord]:
        """Returns all directed AST call/import edges visible to the principal under Forced RLS."""
        with scoped_db_session(self.connection_manager, principal) as conn:
            rows = conn.execute(
                """
                SELECT edge_id, domain_id, source_symbol_id, target_qualified_name,
                       target_symbol_id, edge_kind, allowed_scopes
                FROM ast_edges
                WHERE domain_id = %s
                ORDER BY target_qualified_name;
                """,
                (domain_id,),
            ).fetchall()
            results: list[ASTEdgeRecord] = []
            for r in rows:
                t_sym_id = UUID(str(r[4])) if r[4] is not None else None
                scopes_raw = r[6]
                scopes_list = (
                    [str(s) for s in scopes_raw] if isinstance(scopes_raw, (list, tuple)) else []
                )
                results.append(
                    ASTEdgeRecord(
                        edge_id=UUID(str(r[0])),
                        domain_id=str(r[1]),
                        source_symbol_id=UUID(str(r[2])),
                        target_qualified_name=str(r[3]),
                        target_symbol_id=t_sym_id,
                        edge_kind=str(r[5]),  # type: ignore[arg-type]
                        allowed_scopes=scopes_list,
                    )
                )
            return results


__all__ = [
    "MAX_PAYLOAD_BYTES",
    "MAX_PAYLOAD_ITEMS",
    "TRUNCATION_INDICATOR",
    "PostgresScoutGraphAdapter",
    "bound_payload_dict",
    "get_current_principal",
    "scoped_db_session",
    "validate_tool_call_arguments",
]
