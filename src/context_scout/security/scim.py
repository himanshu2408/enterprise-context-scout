"""SCIM 2.0 Directory Sync Service with PG17 FORCE RLS & Transitive Group Resolution (M5-01)."""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Final
from uuid import UUID

from context_scout.config import ScoutSettings, get_storage_backend
from context_scout.db.session import RLSConnectionManager, get_query_pool
from context_scout.schemas.contracts import PrincipalContext, RoleType
from context_scout.security.permission_mirror import (
    CachedPermissionMirrorService,
    sanitize_scope_identifier,
)
from context_scout.security.scim_models import (
    SCIMGroupRecord as SCIMGroupRecord,
)
from context_scout.security.scim_models import (
    SCIMPrincipalResolution as SCIMPrincipalResolution,
)
from context_scout.security.scim_models import (
    SCIMUserRecord as SCIMUserRecord,
)
from context_scout.security.scim_models import (
    _assert_no_nul_in_obj as _assert_no_nul_in_obj,
)
from context_scout.security.scim_models import (
    _build_members_json as _build_members_json,
)
from context_scout.security.scim_models import (
    _extract_member_lists as _extract_member_lists,
)
from context_scout.security.scim_models import (
    _extract_scim_email as _extract_scim_email,
)
from context_scout.security.scim_models import (
    _extract_scim_emails_json as _extract_scim_emails_json,
)
from context_scout.security.scim_models import (
    _extract_scim_role as _extract_scim_role,
)
from context_scout.security.scim_models import (
    _has_nul_or_surrogate as _has_nul_or_surrogate,
)
from context_scout.security.scim_models import (
    _parse_role as _parse_role,
)
from context_scout.security.scim_models import (
    _reject_nul_text as _reject_nul_text,
)
from context_scout.security.scim_models import (
    _reject_reserved_system_scopes as _reject_reserved_system_scopes,
)
from context_scout.security.scim_models import (
    _sanitize_group_scope_slug as _sanitize_group_scope_slug,
)

_MEMBER_FILTER_RE: Final[re.Pattern[str]] = re.compile(
    r'^members\[\s*value\s+eq\s+["\']([^"\']+)["\']\s*\]$',
    re.IGNORECASE,
)

_ROLE_PRECEDENCE: Final[dict[RoleType, int]] = {
    RoleType.COMPLIANCE_OFFICER: 4,
    RoleType.ENGINEER: 3,
    RoleType.PRODUCT_MANAGER: 2,
    RoleType.DOMAIN_RESEARCHER: 1,
}

logger = logging.getLogger(__name__)


class SCIMDirectoryService:
    """Thread-safe SCIM 2.0 user/group store backed by PG17 FORCE RLS or in-memory state."""

    def __init__(
        self,
        *,
        connection_manager: RLSConnectionManager | None = None,
        permission_cache: CachedPermissionMirrorService | None = None,
    ) -> None:
        self._conn_mgr = connection_manager
        self._permission_cache = permission_cache
        self._lock = threading.RLock()
        self._users: dict[tuple[UUID, str], SCIMUserRecord] = {}
        self._groups: dict[tuple[UUID, str], SCIMGroupRecord] = {}

    def _service_principal(self, tenant_id: UUID) -> PrincipalContext:
        return PrincipalContext(
            user_id="scim_directory_sync",
            role=RoleType.COMPLIANCE_OFFICER,
            allowed_scopes=["audit:compliance", "scim:sync"],
            tenant_id=tenant_id,
        )

    def _invalidate_user_cache(self, user: SCIMUserRecord) -> None:
        if self._permission_cache is None:
            return
        for candidate in (user.scim_user_id, user.external_id, user.user_name, user.email):
            if candidate and candidate.strip():
                self._permission_cache.invalidate_user(candidate.strip(), tenant_id=user.tenant_id)

    def _invalidate_group_members_cache(
        self, group: SCIMGroupRecord, extra_user_ids: Sequence[str] = ()
    ) -> None:
        if self._permission_cache is None:
            return
        for uid in [*group.member_user_ids, *extra_user_ids]:
            if uid and uid.strip():
                self._permission_cache.invalidate_user(uid.strip(), tenant_id=group.tenant_id)
        self._permission_cache.invalidate_scope_prefix(
            f"group:{_sanitize_group_scope_slug(group.display_name)}"
        )

    # ------------------------------------------------------------------
    # User CRUD + RFC 7644 PatchOp
    # ------------------------------------------------------------------

    def upsert_user(
        self,
        tenant_id: UUID,
        payload: Mapping[str, Any],
        *,
        user_id: str | None = None,
        domain_id: str = "default",
    ) -> SCIMUserRecord:
        """Creates or updates a SCIM 2.0 user under strict tenant isolation."""
        _assert_no_nul_in_obj(payload, "user")
        _reject_nul_text(domain_id, "domain_id")
        if user_id is not None:
            _reject_nul_text(user_id, "user_id")
        raw_uname = payload.get("userName")
        if raw_uname is not None and not isinstance(raw_uname, str):
            raise ValueError("SCIM User 'userName' must be a string.")
        user_name = str(raw_uname or "").strip()
        if not user_name:
            raise ValueError("SCIM User requires a non-empty 'userName'.")
        raw_id = user_id if user_id is not None else payload.get("id")
        if raw_id is not None and not isinstance(raw_id, str):
            raise ValueError("SCIM User 'id' must be a string.")
        scim_id = str(raw_id or uuid.uuid4()).strip()
        if not scim_id:
            raise ValueError("SCIM User 'id' must not be blank.")
        _reject_nul_text(scim_id, "id")
        external_id = str(payload.get("externalId") or "").strip()
        email = _extract_scim_email(payload) or (user_name if "@" in user_name else "")
        display_name = str(payload.get("displayName") or user_name).strip()
        active = bool(payload.get("active", True))
        role_override = _extract_scim_role(payload)
        raw_scopes = payload.get("customScopes")
        if raw_scopes is None:
            raw_scopes_list: list[Any] = []
        elif isinstance(raw_scopes, list):
            raw_scopes_list = raw_scopes
        else:
            raise ValueError("SCIM User 'customScopes' must be an array of scope strings.")
        custom_scopes = [
            sanitize_scope_identifier(str(s)) for s in raw_scopes_list if str(s).strip()
        ]
        _reject_reserved_system_scopes(custom_scopes, "customScopes")
        emails_json = _extract_scim_emails_json(payload, email)
        now = datetime.now(UTC)

        rec = SCIMUserRecord(
            tenant_id=tenant_id,
            scim_user_id=scim_id,
            domain_id=domain_id,
            external_id=external_id,
            user_name=user_name,
            email=email,
            display_name=display_name,
            active=active,
            role_override=role_override,
            custom_scopes=custom_scopes,
            metadata_json=dict(payload.get("metadata") or {})
            if isinstance(payload.get("metadata"), Mapping)
            else {},
            created_at=now,
            updated_at=now,
        )

        if self._conn_mgr is not None:
            principal = self._service_principal(tenant_id)
            with self._conn_mgr.scoped_transaction(
                principal, enable_hnsw_iterative_scan=False
            ) as conn:
                row = conn.execute(
                    """
                    INSERT INTO scim_users (
                        tenant_id, scim_user_id, domain_id, external_id, user_name,
                        email, emails_json, display_name, active, role_override, custom_scopes,
                        metadata_json, created_at, updated_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s::jsonb, %s, %s, %s, %s,
                        %s::jsonb, NOW(), NOW()
                    )
                    ON CONFLICT (tenant_id, scim_user_id) DO UPDATE SET
                        domain_id = EXCLUDED.domain_id,
                        external_id = EXCLUDED.external_id,
                        user_name = EXCLUDED.user_name,
                        email = EXCLUDED.email,
                        emails_json = EXCLUDED.emails_json,
                        display_name = EXCLUDED.display_name,
                        active = EXCLUDED.active,
                        role_override = EXCLUDED.role_override,
                        custom_scopes = EXCLUDED.custom_scopes,
                        metadata_json = EXCLUDED.metadata_json,
                        updated_at = NOW()
                    RETURNING tenant_id, scim_user_id, domain_id, external_id, user_name,
                              email, display_name, active, role_override, custom_scopes,
                              metadata_json, created_at, updated_at;
                    """,
                    (
                        tenant_id,
                        rec.scim_user_id,
                        rec.domain_id,
                        rec.external_id,
                        rec.user_name,
                        rec.email,
                        json.dumps(emails_json),
                        rec.display_name,
                        rec.active,
                        rec.role_override.value if rec.role_override else None,
                        rec.custom_scopes,
                        json.dumps(rec.metadata_json),
                    ),
                ).fetchone()
                assert row is not None
                rec = self._row_to_user(row)
        else:
            with self._lock:
                existing = self._users.get((tenant_id, scim_id))
                if existing is not None:
                    rec = rec.model_copy(update={"created_at": existing.created_at})
                self._users[(tenant_id, scim_id)] = rec

        if not rec.active:
            self._invalidate_user_cache(rec)
        return rec

    def get_user(self, tenant_id: UUID, scim_user_id: str) -> SCIMUserRecord | None:
        _reject_nul_text(scim_user_id, "user_id")
        if not scim_user_id.strip():
            raise ValueError("SCIM user_id must not be blank.")
        if self._conn_mgr is not None:
            principal = self._service_principal(tenant_id)
            with self._conn_mgr.scoped_transaction(
                principal, enable_hnsw_iterative_scan=False
            ) as conn:
                row = conn.execute(
                    """
                    SELECT tenant_id, scim_user_id, domain_id, external_id, user_name,
                           email, display_name, active, role_override, custom_scopes,
                           metadata_json, created_at, updated_at
                    FROM scim_users
                    WHERE scim_user_id = %s;
                    """,
                    (scim_user_id,),
                ).fetchone()
                return self._row_to_user(row) if row is not None else None
        with self._lock:
            return self._users.get((tenant_id, scim_user_id))

    def list_users(self, tenant_id: UUID) -> list[SCIMUserRecord]:
        if self._conn_mgr is not None:
            principal = self._service_principal(tenant_id)
            with self._conn_mgr.scoped_transaction(
                principal, enable_hnsw_iterative_scan=False
            ) as conn:
                rows = conn.execute(
                    """
                    SELECT tenant_id, scim_user_id, domain_id, external_id, user_name,
                           email, display_name, active, role_override, custom_scopes,
                           metadata_json, created_at, updated_at
                    FROM scim_users
                    ORDER BY user_name ASC;
                    """
                ).fetchall()
                return [self._row_to_user(r) for r in rows]
        with self._lock:
            return sorted(
                [u for (tid, _), u in self._users.items() if tid == tenant_id],
                key=lambda x: x.user_name.lower(),
            )

    def patch_user(
        self,
        tenant_id: UUID,
        scim_user_id: str,
        operations: Sequence[Mapping[str, Any]],
    ) -> SCIMUserRecord | None:
        """Applies RFC 7644 PatchOp operations to a SCIM 2.0 User."""
        user = self.get_user(tenant_id, scim_user_id)
        if user is None:
            return None

        active = user.active
        user_name = user.user_name
        display_name = user.display_name
        external_id = user.external_id
        email = user.email
        role_override = user.role_override
        custom_scopes = list(user.custom_scopes)

        for op_entry in operations:
            if not isinstance(op_entry, Mapping):
                raise ValueError("Each SCIM PatchOp operation must be a JSON object.")
            _assert_no_nul_in_obj(op_entry, "Operations[]")
            op = str(op_entry.get("op") or "replace").strip().lower()
            path = str(op_entry.get("path") or "").strip()
            value = op_entry.get("value")
            if not path and isinstance(value, Mapping):
                if "active" in value:
                    active = bool(value["active"])
                if "userName" in value and str(value["userName"]).strip():
                    user_name = str(value["userName"]).strip()
                if "displayName" in value:
                    display_name = str(value["displayName"]).strip()
                if "externalId" in value:
                    external_id = str(value["externalId"]).strip()
                if "emails" in value:
                    email = _extract_scim_email(value)
                if "roles" in value or "roleOverride" in value:
                    role_override = _extract_scim_role(value)
                if "customScopes" in value and isinstance(value["customScopes"], list):
                    custom_scopes = [
                        sanitize_scope_identifier(str(s))
                        for s in value["customScopes"]
                        if str(s).strip()
                    ]
                continue

            path_l = path.lower()
            if path_l == "active":
                if op == "remove":
                    active = False
                else:
                    active = bool(value) if not isinstance(value, str) else value.lower() == "true"
            elif path_l == "username" and value is not None:
                user_name = str(value).strip()
            elif path_l == "displayname" and value is not None:
                display_name = str(value).strip()
            elif path_l == "externalid" and value is not None:
                external_id = str(value).strip()
            elif path_l.startswith("emails"):
                if op == "remove":
                    email = ""
                elif isinstance(value, list):
                    email = _extract_scim_email({"emails": value})
                elif isinstance(value, str):
                    email = value.strip()
            elif path_l in {"roles", "roleoverride"}:
                if op == "remove":
                    role_override = None
                elif isinstance(value, list):
                    role_override = _extract_scim_role({"roles": value})
                else:
                    role_override = _parse_role(value)
            elif path_l == "customscopes":
                vals = value if isinstance(value, list) else ([value] if value is not None else [])
                cleaned = [sanitize_scope_identifier(str(s)) for s in vals if str(s).strip()]
                if op == "replace":
                    custom_scopes = cleaned
                elif op == "add":
                    for s in cleaned:
                        if s not in custom_scopes:
                            custom_scopes.append(s)
                elif op == "remove":
                    custom_scopes = [s for s in custom_scopes if s not in set(cleaned)]

        updated = self.upsert_user(
            tenant_id,
            {
                "id": scim_user_id,
                "externalId": external_id,
                "userName": user_name,
                "email": email,
                "displayName": display_name,
                "active": active,
                "roleOverride": role_override.value if role_override else None,
                "customScopes": custom_scopes,
                "metadata": user.metadata_json,
            },
            user_id=scim_user_id,
            domain_id=user.domain_id,
        )
        self._invalidate_user_cache(updated)
        return updated

    def delete_user(self, tenant_id: UUID, scim_user_id: str) -> bool:
        existing = self.get_user(tenant_id, scim_user_id)
        if existing is None:
            return False
        if self._conn_mgr is not None:
            principal = self._service_principal(tenant_id)
            with self._conn_mgr.scoped_transaction(
                principal, enable_hnsw_iterative_scan=False
            ) as conn:
                cur = conn.execute(
                    "DELETE FROM scim_users WHERE scim_user_id = %s;",
                    (scim_user_id,),
                )
                deleted = (cur.rowcount or 0) > 0
        else:
            with self._lock:
                deleted = self._users.pop((tenant_id, scim_user_id), None) is not None
        if deleted:
            self._invalidate_user_cache(existing)
        return deleted

    # ------------------------------------------------------------------
    # Group CRUD + RFC 7644 PatchOp
    # ------------------------------------------------------------------

    def upsert_group(
        self,
        tenant_id: UUID,
        payload: Mapping[str, Any],
        *,
        group_id: str | None = None,
        domain_id: str = "default",
    ) -> SCIMGroupRecord:
        """Creates or updates a SCIM 2.0 group under strict tenant isolation."""
        _assert_no_nul_in_obj(payload, "group")
        _reject_nul_text(domain_id, "domain_id")
        if group_id is not None:
            _reject_nul_text(group_id, "group_id")
        raw_dname = payload.get("displayName")
        if raw_dname is not None and not isinstance(raw_dname, str):
            raise ValueError("SCIM Group 'displayName' must be a string.")
        display_name = str(raw_dname or "").strip()
        if not display_name:
            raise ValueError("SCIM Group requires a non-empty 'displayName'.")
        raw_gid = group_id if group_id is not None else payload.get("id")
        if raw_gid is not None and not isinstance(raw_gid, str):
            raise ValueError("SCIM Group 'id' must be a string.")
        scim_id = str(raw_gid or uuid.uuid4()).strip()
        if not scim_id:
            raise ValueError("SCIM Group 'id' must not be blank.")
        _reject_nul_text(scim_id, "id")
        external_id = str(payload.get("externalId") or "").strip()
        member_users, member_groups = _extract_member_lists(payload.get("members"))
        raw_muids = payload.get("memberUserIds")
        if raw_muids is not None:
            if not isinstance(raw_muids, list):
                raise ValueError("SCIM Group 'memberUserIds' must be an array.")
            for u in raw_muids:
                su = str(u).strip()
                if su and su not in member_users:
                    member_users.append(su)
        raw_mgids = payload.get("memberGroupIds")
        if raw_mgids is not None:
            if not isinstance(raw_mgids, list):
                raise ValueError("SCIM Group 'memberGroupIds' must be an array.")
            for g in raw_mgids:
                sg = str(g).strip()
                if sg and sg not in member_groups:
                    member_groups.append(sg)

        raw_scopes = payload.get("mappedScopes")
        if raw_scopes is None:
            raw_scopes_list: list[Any] = []
        elif isinstance(raw_scopes, list):
            raw_scopes_list = raw_scopes
        else:
            raise ValueError("SCIM Group 'mappedScopes' must be an array of scope strings.")
        mapped_scopes = [
            sanitize_scope_identifier(str(s)) for s in raw_scopes_list if str(s).strip()
        ]
        _reject_reserved_system_scopes(mapped_scopes, "mappedScopes")
        members_json = _build_members_json(member_users, member_groups)
        mapped_role = _extract_scim_role(payload)
        now = datetime.now(UTC)

        rec = SCIMGroupRecord(
            tenant_id=tenant_id,
            scim_group_id=scim_id,
            domain_id=domain_id,
            external_id=external_id,
            display_name=display_name,
            member_user_ids=member_users,
            member_group_ids=member_groups,
            mapped_scopes=mapped_scopes,
            mapped_role=mapped_role,
            metadata_json=dict(payload.get("metadata") or {})
            if isinstance(payload.get("metadata"), Mapping)
            else {},
            created_at=now,
            updated_at=now,
        )

        if self._conn_mgr is not None:
            principal = self._service_principal(tenant_id)
            with self._conn_mgr.scoped_transaction(
                principal, enable_hnsw_iterative_scan=False
            ) as conn:
                row = conn.execute(
                    """
                    INSERT INTO scim_groups (
                        tenant_id, scim_group_id, domain_id, external_id, display_name,
                        member_user_ids, member_group_ids, members_json, mapped_scopes, mapped_role,
                        metadata_json, created_at, updated_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s::jsonb, %s, %s,
                        %s::jsonb, NOW(), NOW()
                    )
                    ON CONFLICT (tenant_id, scim_group_id) DO UPDATE SET
                        domain_id = EXCLUDED.domain_id,
                        external_id = EXCLUDED.external_id,
                        display_name = EXCLUDED.display_name,
                        member_user_ids = EXCLUDED.member_user_ids,
                        member_group_ids = EXCLUDED.member_group_ids,
                        members_json = EXCLUDED.members_json,
                        mapped_scopes = EXCLUDED.mapped_scopes,
                        mapped_role = EXCLUDED.mapped_role,
                        metadata_json = EXCLUDED.metadata_json,
                        updated_at = NOW()
                    RETURNING tenant_id, scim_group_id, domain_id, external_id, display_name,
                              member_user_ids, member_group_ids, mapped_scopes, mapped_role,
                              metadata_json, created_at, updated_at;
                    """,
                    (
                        tenant_id,
                        rec.scim_group_id,
                        rec.domain_id,
                        rec.external_id,
                        rec.display_name,
                        rec.member_user_ids,
                        rec.member_group_ids,
                        json.dumps(members_json),
                        rec.mapped_scopes,
                        rec.mapped_role.value if rec.mapped_role else None,
                        json.dumps(rec.metadata_json),
                    ),
                ).fetchone()
                assert row is not None
                rec = self._row_to_group(row)
        else:
            with self._lock:
                existing = self._groups.get((tenant_id, scim_id))
                if existing is not None:
                    rec = rec.model_copy(update={"created_at": existing.created_at})
                self._groups[(tenant_id, scim_id)] = rec

        self._invalidate_group_members_cache(rec)
        return rec

    def get_group(self, tenant_id: UUID, scim_group_id: str) -> SCIMGroupRecord | None:
        _reject_nul_text(scim_group_id, "group_id")
        if not scim_group_id.strip():
            raise ValueError("SCIM group_id must not be blank.")
        if self._conn_mgr is not None:
            principal = self._service_principal(tenant_id)
            with self._conn_mgr.scoped_transaction(
                principal, enable_hnsw_iterative_scan=False
            ) as conn:
                row = conn.execute(
                    """
                    SELECT tenant_id, scim_group_id, domain_id, external_id, display_name,
                           member_user_ids, member_group_ids, mapped_scopes, mapped_role,
                           metadata_json, created_at, updated_at
                    FROM scim_groups
                    WHERE scim_group_id = %s;
                    """,
                    (scim_group_id,),
                ).fetchone()
                return self._row_to_group(row) if row is not None else None
        with self._lock:
            return self._groups.get((tenant_id, scim_group_id))

    def list_groups(self, tenant_id: UUID) -> list[SCIMGroupRecord]:
        if self._conn_mgr is not None:
            principal = self._service_principal(tenant_id)
            with self._conn_mgr.scoped_transaction(
                principal, enable_hnsw_iterative_scan=False
            ) as conn:
                rows = conn.execute(
                    """
                    SELECT tenant_id, scim_group_id, domain_id, external_id, display_name,
                           member_user_ids, member_group_ids, mapped_scopes, mapped_role,
                           metadata_json, created_at, updated_at
                    FROM scim_groups
                    ORDER BY display_name ASC;
                    """
                ).fetchall()
                return [self._row_to_group(r) for r in rows]
        with self._lock:
            return sorted(
                [g for (tid, _), g in self._groups.items() if tid == tenant_id],
                key=lambda x: x.display_name.lower(),
            )

    def patch_group(
        self,
        tenant_id: UUID,
        scim_group_id: str,
        operations: Sequence[Mapping[str, Any]],
    ) -> SCIMGroupRecord | None:
        """Applies RFC 7644 PatchOp (including members[value eq "..."]) to a SCIM Group."""
        group = self.get_group(tenant_id, scim_group_id)
        if group is None:
            return None

        display_name = group.display_name
        external_id = group.external_id
        member_users = list(group.member_user_ids)
        member_groups = list(group.member_group_ids)
        mapped_scopes = list(group.mapped_scopes)
        mapped_role = group.mapped_role
        removed_users: list[str] = []

        for op_entry in operations:
            if not isinstance(op_entry, Mapping):
                raise ValueError("Each SCIM PatchOp operation must be a JSON object.")
            _assert_no_nul_in_obj(op_entry, "Operations[]")
            op = str(op_entry.get("op") or "replace").strip().lower()
            path = str(op_entry.get("path") or "").strip()
            value = op_entry.get("value")

            if not path and isinstance(value, Mapping):
                if "displayName" in value and str(value["displayName"]).strip():
                    display_name = str(value["displayName"]).strip()
                if "externalId" in value:
                    external_id = str(value["externalId"]).strip()
                if "mappedScopes" in value and isinstance(value["mappedScopes"], list):
                    mapped_scopes = [
                        sanitize_scope_identifier(str(s))
                        for s in value["mappedScopes"]
                        if str(s).strip()
                    ]
                if "mappedRole" in value or "roles" in value:
                    mapped_role = _extract_scim_role(value)
                if "members" in value:
                    u_list, g_list = _extract_member_lists(value["members"])
                    if op == "remove":
                        removed_users.extend(u_list)
                        member_users = [u for u in member_users if u not in set(u_list)]
                        member_groups = [g for g in member_groups if g not in set(g_list)]
                    elif op == "add":
                        for u in u_list:
                            if u not in member_users:
                                member_users.append(u)
                        for g in g_list:
                            if g not in member_groups:
                                member_groups.append(g)
                    else:
                        removed_users.extend([u for u in member_users if u not in set(u_list)])
                        member_users = u_list
                        member_groups = g_list
                continue

            filter_match = _MEMBER_FILTER_RE.match(path)
            if filter_match:
                target_member = filter_match.group(1).strip()
                if op == "remove":
                    if target_member in member_users:
                        removed_users.append(target_member)
                    member_users = [u for u in member_users if u != target_member]
                    member_groups = [g for g in member_groups if g != target_member]
                elif op in {"add", "replace"} and target_member:
                    if target_member not in member_users:
                        member_users.append(target_member)
                continue

            path_l = path.lower()
            if path_l == "members":
                u_list, g_list = _extract_member_lists(
                    value if isinstance(value, list) else ([value] if value is not None else [])
                )
                if op == "remove":
                    if not u_list and not g_list and value is None:
                        removed_users.extend(member_users)
                        member_users = []
                        member_groups = []
                    else:
                        removed_users.extend(u_list)
                        member_users = [u for u in member_users if u not in set(u_list)]
                        member_groups = [g for g in member_groups if g not in set(g_list)]
                elif op == "add":
                    for u in u_list:
                        if u not in member_users:
                            member_users.append(u)
                    for g in g_list:
                        if g not in member_groups:
                            member_groups.append(g)
                elif op == "replace":
                    removed_users.extend([u for u in member_users if u not in set(u_list)])
                    member_users = u_list
                    member_groups = g_list
            elif path_l == "displayname" and value is not None:
                display_name = str(value).strip()
            elif path_l == "externalid" and value is not None:
                external_id = str(value).strip()
            elif path_l == "mappedscopes":
                vals = value if isinstance(value, list) else ([value] if value is not None else [])
                cleaned = [sanitize_scope_identifier(str(s)) for s in vals if str(s).strip()]
                if op == "replace":
                    mapped_scopes = cleaned
                elif op == "add":
                    for s in cleaned:
                        if s not in mapped_scopes:
                            mapped_scopes.append(s)
                elif op == "remove":
                    mapped_scopes = [s for s in mapped_scopes if s not in set(cleaned)]
            elif path_l == "mappedrole":
                mapped_role = None if op == "remove" else _parse_role(value)

        updated = self.upsert_group(
            tenant_id,
            {
                "id": scim_group_id,
                "externalId": external_id,
                "displayName": display_name,
                "memberUserIds": member_users,
                "memberGroupIds": member_groups,
                "mappedScopes": mapped_scopes,
                "mappedRole": mapped_role.value if mapped_role else None,
                "metadata": group.metadata_json,
            },
            group_id=scim_group_id,
            domain_id=group.domain_id,
        )
        self._invalidate_group_members_cache(updated, extra_user_ids=removed_users)
        return updated

    def delete_group(self, tenant_id: UUID, scim_group_id: str) -> bool:
        existing = self.get_group(tenant_id, scim_group_id)
        if existing is None:
            return False
        if self._conn_mgr is not None:
            principal = self._service_principal(tenant_id)
            with self._conn_mgr.scoped_transaction(
                principal, enable_hnsw_iterative_scan=False
            ) as conn:
                cur = conn.execute(
                    "DELETE FROM scim_groups WHERE scim_group_id = %s;",
                    (scim_group_id,),
                )
                deleted = (cur.rowcount or 0) > 0
        else:
            with self._lock:
                deleted = self._groups.pop((tenant_id, scim_group_id), None) is not None
        if deleted:
            self._invalidate_group_members_cache(existing)
        return deleted

    # ------------------------------------------------------------------
    # Principal Enrichment & Transitive Nested Group Expansion
    # ------------------------------------------------------------------

    def _lookup_matched_user_and_groups_pg(
        self,
        tenant_id: UUID,
        raw_identifiers: Sequence[str],
        norm_ids: set[str],
        norm_group_refs: set[str],
    ) -> tuple[SCIMUserRecord | None, list[SCIMGroupRecord]]:
        assert self._conn_mgr is not None
        principal = self._service_principal(tenant_id)
        exact_ids = list({i.strip() for i in raw_identifiers if i and i.strip()})
        lower_ids = sorted(norm_ids)
        email_probes = [json.dumps([{"value": lid}]) for lid in lower_ids if "@" in lid or lid]

        with self._conn_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
            matched_user: SCIMUserRecord | None = None
            if exact_ids or lower_ids:
                u_row = conn.execute(
                    """
                    SELECT tenant_id, scim_user_id, domain_id, external_id, user_name,
                           email, display_name, active, role_override, custom_scopes,
                           metadata_json, created_at, updated_at
                    FROM scim_users
                    WHERE scim_user_id = ANY(%s)
                       OR lower(user_name) = ANY(%s)
                       OR external_id = ANY(%s)
                       OR lower(external_id) = ANY(%s)
                       OR lower(email) = ANY(%s)
                       OR emails_json @> ANY(%s::jsonb[])
                    ORDER BY user_name ASC
                    LIMIT 1;
                    """,
                    (exact_ids, lower_ids, exact_ids, lower_ids, lower_ids, email_probes),
                ).fetchone()
                if u_row is not None:
                    matched_user = self._row_to_user(u_row)

            if matched_user is not None and not matched_user.active:
                return matched_user, []

            eff_raw_u = set(exact_ids)
            eff_low_u = set(lower_ids)
            if matched_user is not None:
                for val in (
                    matched_user.scim_user_id,
                    matched_user.user_name,
                    matched_user.external_id,
                    matched_user.email,
                ):
                    if val and val.strip():
                        eff_raw_u.add(val.strip())
                        eff_low_u.add(val.strip().lower())
            eff_user_tokens = sorted(eff_raw_u | eff_low_u)
            eff_user_lower = sorted(eff_low_u)
            eff_group_lower = sorted(norm_group_refs)
            member_user_json_probes = [
                json.dumps([{"value": tok, "type": "User"}]) for tok in eff_user_tokens
            ]

            matched_by_id: dict[str, SCIMGroupRecord] = {}
            frontier_tokens: set[str] = set()
            seed_rows = conn.execute(
                """
                SELECT tenant_id, scim_group_id, domain_id, external_id, display_name,
                       member_user_ids, member_group_ids, mapped_scopes, mapped_role,
                       metadata_json, created_at, updated_at
                FROM scim_groups
                WHERE member_user_ids && %s
                   OR members_json @> ANY(%s::jsonb[])
                   OR EXISTS (
                       SELECT 1 FROM unnest(member_user_ids) AS mu WHERE lower(mu) = ANY(%s)
                   )
                   OR lower(scim_group_id) = ANY(%s)
                   OR lower(external_id) = ANY(%s)
                   OR lower(display_name) = ANY(%s)
                ORDER BY display_name ASC;
                """,
                (
                    eff_user_tokens,
                    member_user_json_probes,
                    eff_user_lower,
                    eff_group_lower,
                    eff_group_lower,
                    eff_group_lower,
                ),
            ).fetchall()
            for r in seed_rows:
                grp = self._row_to_group(r)
                matched_by_id[grp.scim_group_id] = grp
                for tok in (grp.scim_group_id, grp.external_id, grp.display_name):
                    if tok and tok.strip():
                        frontier_tokens.add(tok.strip())
                        frontier_tokens.add(tok.strip().lower())

            # Cycle-safe bounded parent group expansion via indexed GIN array/jsonb lookup
            while frontier_tokens:
                tok_list = sorted(frontier_tokens)
                tok_lower = sorted({t.lower() for t in frontier_tokens})
                seen_ids = list(matched_by_id.keys())
                grp_json_probes = [
                    json.dumps([{"value": tok, "type": "Group"}]) for tok in tok_list
                ]
                parent_rows = conn.execute(
                    """
                    SELECT tenant_id, scim_group_id, domain_id, external_id, display_name,
                           member_user_ids, member_group_ids, mapped_scopes, mapped_role,
                           metadata_json, created_at, updated_at
                    FROM scim_groups
                    WHERE NOT (scim_group_id = ANY(%s))
                      AND (
                          member_group_ids && %s
                          OR members_json @> ANY(%s::jsonb[])
                          OR EXISTS (
                              SELECT 1 FROM unnest(member_group_ids) AS mg
                              WHERE lower(mg) = ANY(%s)
                          )
                      )
                    ORDER BY display_name ASC;
                    """,
                    (seen_ids, tok_list, grp_json_probes, tok_lower),
                ).fetchall()
                next_frontier: set[str] = set()
                for r in parent_rows:
                    grp = self._row_to_group(r)
                    if grp.scim_group_id in matched_by_id:
                        continue
                    matched_by_id[grp.scim_group_id] = grp
                    for tok in (grp.scim_group_id, grp.external_id, grp.display_name):
                        if tok and tok.strip():
                            next_frontier.add(tok.strip())
                            next_frontier.add(tok.strip().lower())
                frontier_tokens = next_frontier

            sorted_groups = sorted(matched_by_id.values(), key=lambda g: g.display_name.lower())
            return matched_user, sorted_groups

    def resolve_principal_enrichment(
        self,
        tenant_id: UUID,
        identifiers: Sequence[str],
        idp_group_refs: Sequence[str] = (),
    ) -> SCIMPrincipalResolution:
        """Resolves SCIM active status, transitive nested groups, scopes, and role."""
        norm_ids = {i.strip().lower() for i in identifiers if i and i.strip()}
        norm_group_refs = {g.strip().lower() for g in idp_group_refs if g and g.strip()}

        if self._conn_mgr is not None:
            matched_user, matched_groups = self._lookup_matched_user_and_groups_pg(
                tenant_id, identifiers, norm_ids, norm_group_refs
            )
            if matched_user is not None and not matched_user.active:
                return SCIMPrincipalResolution(
                    found_user=True,
                    deprovisioned=True,
                    scim_user=matched_user,
                )
        else:
            users = self.list_users(tenant_id)
            matched_user = None
            for user in users:
                user_keys = {
                    user.scim_user_id.lower(),
                    user.user_name.lower(),
                    user.external_id.lower(),
                    user.email.lower(),
                }
                user_keys.discard("")
                if norm_ids & user_keys:
                    matched_user = user
                    break

            if matched_user is not None and not matched_user.active:
                return SCIMPrincipalResolution(
                    found_user=True,
                    deprovisioned=True,
                    scim_user=matched_user,
                )

            effective_user_keys = set(norm_ids)
            if matched_user is not None:
                effective_user_keys.update(
                    k
                    for k in (
                        matched_user.scim_user_id.lower(),
                        matched_user.user_name.lower(),
                        matched_user.external_id.lower(),
                        matched_user.email.lower(),
                    )
                    if k
                )

            all_groups = self.list_groups(tenant_id)
            matched_group_ids: set[str] = set()
            matched_group_tokens: set[str] = set(norm_group_refs)

            for grp in all_groups:
                grp_keys = {
                    grp.scim_group_id.lower(),
                    grp.external_id.lower(),
                    grp.display_name.lower(),
                }
                grp_keys.discard("")
                member_u_keys = {m.strip().lower() for m in grp.member_user_ids if m.strip()}
                if (effective_user_keys & member_u_keys) or (norm_group_refs & grp_keys):
                    matched_group_ids.add(grp.scim_group_id)
                    matched_group_tokens.update(grp_keys)

            # Cycle-safe transitive nested group expansion:
            # If child group G is matched and parent group P contains G in member_group_ids.
            changed = True
            while changed:
                changed = False
                for grp in all_groups:
                    if grp.scim_group_id in matched_group_ids:
                        continue
                    child_refs = {m.strip().lower() for m in grp.member_group_ids if m.strip()}
                    if child_refs & matched_group_tokens:
                        matched_group_ids.add(grp.scim_group_id)
                        matched_group_tokens.update(
                            k
                            for k in (
                                grp.scim_group_id.lower(),
                                grp.external_id.lower(),
                                grp.display_name.lower(),
                            )
                            if k
                        )
                        changed = True

            matched_groups = [g for g in all_groups if g.scim_group_id in matched_group_ids]
        scopes: list[str] = []
        best_role: RoleType | None = matched_user.role_override if matched_user else None

        if matched_user is not None:
            for sc in matched_user.custom_scopes:
                clean_sc = sanitize_scope_identifier(sc)
                if clean_sc not in scopes:
                    scopes.append(clean_sc)

        group_names: list[str] = []
        for grp in matched_groups:
            group_names.append(grp.display_name)
            grp_scope = f"group:{_sanitize_group_scope_slug(grp.display_name)}"
            if grp_scope not in scopes:
                scopes.append(grp_scope)
            for sc in grp.mapped_scopes:
                clean_sc = sanitize_scope_identifier(sc)
                if clean_sc not in scopes:
                    scopes.append(clean_sc)
            if (
                (matched_user is None or matched_user.role_override is None)
                and grp.mapped_role is not None
                and (
                    best_role is None
                    or _ROLE_PRECEDENCE.get(grp.mapped_role, 0) > _ROLE_PRECEDENCE.get(best_role, 0)
                )
            ):
                best_role = grp.mapped_role

        return SCIMPrincipalResolution(
            found_user=matched_user is not None,
            deprovisioned=False,
            scim_user=matched_user,
            group_names=group_names,
            scopes=scopes,
            role=best_role,
        )

    @staticmethod
    def _row_to_user(row: tuple[object, ...]) -> SCIMUserRecord:
        raw_meta = row[10]
        meta = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta or {})  # type: ignore[call-overload]
        raw_scopes = row[9]
        scopes = [str(s) for s in raw_scopes] if isinstance(raw_scopes, (list, tuple)) else []
        return SCIMUserRecord(
            tenant_id=row[0] if isinstance(row[0], UUID) else UUID(str(row[0])),
            scim_user_id=str(row[1]),
            domain_id=str(row[2]),
            external_id=str(row[3] or ""),
            user_name=str(row[4]),
            email=str(row[5] or ""),
            display_name=str(row[6] or ""),
            active=bool(row[7]),
            role_override=_parse_role(row[8]),
            custom_scopes=scopes,
            metadata_json=meta,
            created_at=row[11] if isinstance(row[11], datetime) else datetime.now(UTC),
            updated_at=row[12] if isinstance(row[12], datetime) else datetime.now(UTC),
        )

    @staticmethod
    def _row_to_group(row: tuple[object, ...]) -> SCIMGroupRecord:
        raw_meta = row[9]
        meta = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta or {})  # type: ignore[call-overload]
        raw_users = row[5]
        raw_groups = row[6]
        raw_scopes = row[7]
        return SCIMGroupRecord(
            tenant_id=row[0] if isinstance(row[0], UUID) else UUID(str(row[0])),
            scim_group_id=str(row[1]),
            domain_id=str(row[2]),
            external_id=str(row[3] or ""),
            display_name=str(row[4]),
            member_user_ids=[str(u) for u in raw_users]
            if isinstance(raw_users, (list, tuple))
            else [],
            member_group_ids=[str(g) for g in raw_groups]
            if isinstance(raw_groups, (list, tuple))
            else [],
            mapped_scopes=[str(s) for s in raw_scopes]
            if isinstance(raw_scopes, (list, tuple))
            else [],
            mapped_role=_parse_role(row[8]),
            metadata_json=meta,
            created_at=row[10] if isinstance(row[10], datetime) else datetime.now(UTC),
            updated_at=row[11] if isinstance(row[11], datetime) else datetime.now(UTC),
        )


_SCIM_LOCK = threading.Lock()
_GLOBAL_SCIM_SERVICE: SCIMDirectoryService | None = None
_GLOBAL_SCIM_EXPLICIT: bool = False
_GLOBAL_SCIM_KEY: tuple[str, str] | None = None


def get_scim_directory_service() -> SCIMDirectoryService:
    """Returns shared SCIMDirectoryService for storage_backend without silent fallback."""
    global _GLOBAL_SCIM_SERVICE, _GLOBAL_SCIM_EXPLICIT, _GLOBAL_SCIM_KEY
    if _GLOBAL_SCIM_EXPLICIT and _GLOBAL_SCIM_SERVICE is not None:
        return _GLOBAL_SCIM_SERVICE
    backend = get_storage_backend()
    settings = ScoutSettings()
    desired_key = (backend, settings.db_app_url if backend == "postgres" else "memory")
    if _GLOBAL_SCIM_SERVICE is not None and desired_key == _GLOBAL_SCIM_KEY:
        return _GLOBAL_SCIM_SERVICE
    with _SCIM_LOCK:
        if _GLOBAL_SCIM_EXPLICIT and _GLOBAL_SCIM_SERVICE is not None:
            return _GLOBAL_SCIM_SERVICE
        if _GLOBAL_SCIM_SERVICE is not None and desired_key == _GLOBAL_SCIM_KEY:
            return _GLOBAL_SCIM_SERVICE
        if backend == "memory":
            _GLOBAL_SCIM_SERVICE = SCIMDirectoryService()
        else:
            _GLOBAL_SCIM_SERVICE = SCIMDirectoryService(connection_manager=get_query_pool(settings))
        _GLOBAL_SCIM_KEY = desired_key
        return _GLOBAL_SCIM_SERVICE


def set_scim_directory_service(service: SCIMDirectoryService | None) -> None:
    """Overrides or resets the global SCIMDirectoryService instance (used in tests)."""
    global _GLOBAL_SCIM_SERVICE, _GLOBAL_SCIM_EXPLICIT, _GLOBAL_SCIM_KEY
    with _SCIM_LOCK:
        _GLOBAL_SCIM_SERVICE = service
        _GLOBAL_SCIM_EXPLICIT = service is not None
        _GLOBAL_SCIM_KEY = None


__all__ = [
    "SCIMDirectoryService",
    "SCIMGroupRecord",
    "SCIMPrincipalResolution",
    "SCIMUserRecord",
    "get_scim_directory_service",
    "set_scim_directory_service",
]
