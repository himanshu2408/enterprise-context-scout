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

from pydantic import BaseModel, Field

from context_scout.config import ScoutSettings
from context_scout.db.session import RLSConnectionManager, get_query_pool
from context_scout.schemas.contracts import DEFAULT_TENANT_ID, PrincipalContext, RoleType
from context_scout.security.permission_mirror import (
    CachedPermissionMirrorService,
    sanitize_scope_identifier,
)

logger = logging.getLogger(__name__)

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


def _parse_role(raw_role: object) -> RoleType | None:
    if raw_role is None:
        return None
    if isinstance(raw_role, RoleType):
        return raw_role
    cleaned = str(raw_role).strip().lower()
    if not cleaned:
        return None
    try:
        return RoleType(cleaned)
    except ValueError:
        return None


def _sanitize_group_scope_slug(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\./:]+", "_", raw.strip()).strip("_")
    if not cleaned:
        return "group"
    return sanitize_scope_identifier(cleaned)


def _has_nul_or_surrogate(value: str) -> bool:
    return "\x00" in value or any(0xD800 <= ord(ch) <= 0xDFFF for ch in value)


def _reject_nul_text(value: str, field_name: str) -> str:
    if _has_nul_or_surrogate(value):
        raise ValueError(
            f"SCIM field '{field_name}' must not contain NUL bytes or lone UTF-16 surrogates."
        )
    return value


def _assert_no_nul_in_obj(obj: Any, path: str = "payload", *, depth: int = 0) -> None:
    if depth > 32:
        raise ValueError(f"SCIM {path} exceeds maximum nesting depth.")
    if isinstance(obj, str):
        if _has_nul_or_surrogate(obj):
            raise ValueError(
                f"SCIM field '{path}' must not contain NUL bytes or lone UTF-16 surrogates."
            )
        return
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if isinstance(k, str) and _has_nul_or_surrogate(k):
                raise ValueError(
                    f"SCIM field key '{path}' must not contain NUL bytes or lone UTF-16 surrogates."
                )
            _assert_no_nul_in_obj(v, f"{path}.{k}", depth=depth + 1)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for idx, item in enumerate(obj):
            _assert_no_nul_in_obj(item, f"{path}[{idx}]", depth=depth + 1)


class SCIMUserRecord(BaseModel):
    """Canonical SCIM 2.0 User record scoped to tenant_id."""

    tenant_id: UUID = Field(default=DEFAULT_TENANT_ID)
    scim_user_id: str = Field(min_length=1)
    domain_id: str = Field(default="default")
    external_id: str = Field(default="")
    user_name: str = Field(min_length=1)
    email: str = Field(default="")
    display_name: str = Field(default="")
    active: bool = Field(default=True)
    role_override: RoleType | None = Field(default=None)
    custom_scopes: list[str] = Field(default_factory=list)
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_scim_resource(self, groups: Sequence[SCIMGroupRecord] = ()) -> dict[str, Any]:
        """Serializes record into RFC 7643 SCIM 2.0 User JSON representation."""
        emails_payload = [{"value": self.email, "primary": True}] if self.email else []
        roles_payload = [{"value": self.role_override.value}] if self.role_override else []
        return {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            "id": self.scim_user_id,
            "externalId": self.external_id,
            "userName": self.user_name,
            "displayName": self.display_name or self.user_name,
            "emails": emails_payload,
            "active": self.active,
            "roles": roles_payload,
            "customScopes": list(self.custom_scopes),
            "groups": [{"value": g.scim_group_id, "display": g.display_name} for g in groups],
            "meta": {
                "resourceType": "User",
                "created": self.created_at.isoformat(),
                "lastModified": self.updated_at.isoformat(),
            },
        }


class SCIMGroupRecord(BaseModel):
    """Canonical SCIM 2.0 Group record scoped to tenant_id."""

    tenant_id: UUID = Field(default=DEFAULT_TENANT_ID)
    scim_group_id: str = Field(min_length=1)
    domain_id: str = Field(default="default")
    external_id: str = Field(default="")
    display_name: str = Field(min_length=1)
    member_user_ids: list[str] = Field(default_factory=list)
    member_group_ids: list[str] = Field(default_factory=list)
    mapped_scopes: list[str] = Field(default_factory=list)
    mapped_role: RoleType | None = Field(default=None)
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_scim_resource(self) -> dict[str, Any]:
        """Serializes record into RFC 7643 SCIM 2.0 Group JSON representation."""
        members: list[dict[str, str]] = [
            {"value": uid, "type": "User"} for uid in self.member_user_ids
        ]
        members.extend({"value": gid, "type": "Group"} for gid in self.member_group_ids)
        return {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
            "id": self.scim_group_id,
            "externalId": self.external_id,
            "displayName": self.display_name,
            "members": members,
            "mappedScopes": list(self.mapped_scopes),
            "mappedRole": self.mapped_role.value if self.mapped_role else None,
            "meta": {
                "resourceType": "Group",
                "created": self.created_at.isoformat(),
                "lastModified": self.updated_at.isoformat(),
            },
        }


class SCIMPrincipalResolution(BaseModel):
    """Resolved SCIM status, transitive groups, scopes, and role enrichment."""

    found_user: bool = False
    deprovisioned: bool = False
    scim_user: SCIMUserRecord | None = None
    group_names: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    role: RoleType | None = None


def _extract_scim_email(payload: Mapping[str, Any]) -> str:
    emails = payload.get("emails")
    if isinstance(emails, list):
        for entry in emails:
            if isinstance(entry, Mapping) and entry.get("primary") and entry.get("value"):
                return str(entry["value"]).strip()
        for entry in emails:
            if isinstance(entry, Mapping) and entry.get("value"):
                return str(entry["value"]).strip()
            if isinstance(entry, str) and entry.strip():
                return entry.strip()
    email = payload.get("email")
    return str(email).strip() if isinstance(email, str) else ""


def _extract_scim_role(payload: Mapping[str, Any]) -> RoleType | None:
    if "roleOverride" in payload:
        parsed = _parse_role(payload.get("roleOverride"))
        if parsed is not None:
            return parsed
    if "mappedRole" in payload:
        parsed = _parse_role(payload.get("mappedRole"))
        if parsed is not None:
            return parsed
    roles = payload.get("roles")
    if isinstance(roles, list):
        for item in roles:
            raw_val = item.get("value") if isinstance(item, Mapping) else item
            parsed = _parse_role(raw_val)
            if parsed is not None:
                return parsed
    return None


def _extract_member_lists(raw_members: Any) -> tuple[list[str], list[str]]:
    users: list[str] = []
    groups: list[str] = []
    if not isinstance(raw_members, list):
        return (users, groups)
    for item in raw_members:
        if isinstance(item, Mapping):
            val = str(item.get("value") or "").strip()
            if not val:
                continue
            m_type = str(item.get("type") or "User").strip().lower()
            if m_type == "group":
                if val not in groups:
                    groups.append(val)
            elif val not in users:
                users.append(val)
        elif isinstance(item, str) and item.strip():
            val = item.strip()
            if val not in users:
                users.append(val)
    return (users, groups)


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
                self._permission_cache.invalidate_user(candidate.strip())

    def _invalidate_group_members_cache(
        self, group: SCIMGroupRecord, extra_user_ids: Sequence[str] = ()
    ) -> None:
        if self._permission_cache is None:
            return
        for uid in [*group.member_user_ids, *extra_user_ids]:
            if uid and uid.strip():
                self._permission_cache.invalidate_user(uid.strip())
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
                        email, display_name, active, role_override, custom_scopes,
                        metadata_json, created_at, updated_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s::jsonb, NOW(), NOW()
                    )
                    ON CONFLICT (tenant_id, scim_user_id) DO UPDATE SET
                        domain_id = EXCLUDED.domain_id,
                        external_id = EXCLUDED.external_id,
                        user_name = EXCLUDED.user_name,
                        email = EXCLUDED.email,
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
                        member_user_ids, member_group_ids, mapped_scopes, mapped_role,
                        metadata_json, created_at, updated_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s::jsonb, NOW(), NOW()
                    )
                    ON CONFLICT (tenant_id, scim_group_id) DO UPDATE SET
                        domain_id = EXCLUDED.domain_id,
                        external_id = EXCLUDED.external_id,
                        display_name = EXCLUDED.display_name,
                        member_user_ids = EXCLUDED.member_user_ids,
                        member_group_ids = EXCLUDED.member_group_ids,
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

    def resolve_principal_enrichment(
        self,
        tenant_id: UUID,
        identifiers: Sequence[str],
        idp_group_refs: Sequence[str] = (),
    ) -> SCIMPrincipalResolution:
        """Resolves SCIM active status, transitive nested groups, scopes, and role."""
        norm_ids = {i.strip().lower() for i in identifiers if i and i.strip()}
        norm_group_refs = {g.strip().lower() for g in idp_group_refs if g and g.strip()}

        users = self.list_users(tenant_id)
        matched_user: SCIMUserRecord | None = None
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
        # If child group G is matched and parent group P contains G in member_group_ids, P matches.
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


def get_scim_directory_service() -> SCIMDirectoryService:
    """Returns shared SCIMDirectoryService backed by PG17 RLS when reachable, else memory."""
    global _GLOBAL_SCIM_SERVICE
    if _GLOBAL_SCIM_SERVICE is not None:
        return _GLOBAL_SCIM_SERVICE
    with _SCIM_LOCK:
        if _GLOBAL_SCIM_SERVICE is not None:
            return _GLOBAL_SCIM_SERVICE
        settings = ScoutSettings()
        try:
            conn_mgr = get_query_pool(settings)
            probe = PrincipalContext(
                user_id="scim_probe",
                role=RoleType.COMPLIANCE_OFFICER,
                allowed_scopes=["audit:compliance"],
            )
            with conn_mgr.scoped_transaction(probe, enable_hnsw_iterative_scan=False) as conn:
                conn.execute("SELECT 1 FROM scim_users LIMIT 1;")
            _GLOBAL_SCIM_SERVICE = SCIMDirectoryService(connection_manager=conn_mgr)
        except Exception as exc:
            logger.debug(
                "PG17 scim_users unavailable (%s); using in-memory SCIMDirectoryService.",
                exc,
            )
            _GLOBAL_SCIM_SERVICE = SCIMDirectoryService()
        return _GLOBAL_SCIM_SERVICE


def set_scim_directory_service(service: SCIMDirectoryService | None) -> None:
    """Overrides or resets the global SCIMDirectoryService instance (used in tests)."""
    global _GLOBAL_SCIM_SERVICE
    with _SCIM_LOCK:
        _GLOBAL_SCIM_SERVICE = service


__all__ = [
    "SCIMDirectoryService",
    "SCIMGroupRecord",
    "SCIMPrincipalResolution",
    "SCIMUserRecord",
    "get_scim_directory_service",
    "set_scim_directory_service",
]
