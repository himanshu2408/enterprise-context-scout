"""SCIM 2.0 DTOs, NUL/scope validators, and payload extraction helpers (ECS-170)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from context_scout.schemas.contracts import DEFAULT_TENANT_ID, RoleType
from context_scout.security.permission_mirror import sanitize_scope_identifier


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


def _reject_reserved_system_scopes(scopes: Sequence[str], field_name: str) -> None:
    for scope in scopes:
        if scope.strip().lower().startswith("system:"):
            raise ValueError(
                f"Reserved internal scope '{scope}' cannot be provisioned via SCIM '{field_name}'."
            )


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


def _extract_scim_emails_json(
    payload: Mapping[str, Any], primary_email: str
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw_emails = payload.get("emails")
    if isinstance(raw_emails, list):
        for entry in raw_emails:
            if isinstance(entry, Mapping) and entry.get("value"):
                val = str(entry["value"]).strip().lower()
                if val and val not in seen:
                    seen.add(val)
                    obj: dict[str, Any] = {"value": val}
                    if entry.get("primary"):
                        obj["primary"] = True
                    entries.append(obj)
            elif isinstance(entry, str) and entry.strip():
                val = entry.strip().lower()
                if val not in seen:
                    seen.add(val)
                    entries.append({"value": val})
    if primary_email:
        norm = primary_email.strip().lower()
        if norm and norm not in seen:
            entries.insert(0, {"value": norm, "primary": True})
    return entries


def _build_members_json(
    member_users: Sequence[str], member_groups: Sequence[str]
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for uid in member_users:
        if uid.strip():
            out.append({"value": uid.strip(), "type": "User"})
    for gid in member_groups:
        if gid.strip():
            out.append({"value": gid.strip(), "type": "Group"})
    return out
