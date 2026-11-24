"""EU Works-Council (BetrVG § 87 Abs. 1 Nr. 6) HMAC User-ID Pseudonymization & Privacy Guard."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from collections.abc import Mapping
from typing import Any, Final
from uuid import UUID

from context_scout.schemas.contracts import DEFAULT_TENANT_ID, PrincipalContext

_PSEUDONYM_RE: Final[re.Pattern[str]] = re.compile(r"^betrvg_[a-zA-Z0-9]+_[0-9a-f]{24}$")
_EPOCH_RE: Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z0-9]{1,16}$")

_INSECURE_BETRVG_SALTS: Final[frozenset[str]] = frozenset(
    {
        "scout-dev-hmac-secret-override-in-prod-sec04",
        "scout-dev-secret-key-change-in-production",
        "changeme",
        "secret",
        "dev-secret",
        "insecure-default-secret",
    }
)

_PROMPT_SNIPPET_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "scout.retrieval.query",
        "scout.retrieval.query_snippet",
        "scout.gating.prompt_snippet",
        "query",
    }
)

_PRINCIPAL_ID_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "scout.principal_id",
        "context_scout.principal_id",
        "principal_id",
        "actor_id",
    }
)


def is_betrvg_enabled() -> bool:
    """Returns True when EU Works-Council (BetrVG) privacy mode is active."""
    raw = os.environ.get("SCOUT_BETRVG_PRIVACY_MODE", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def resolve_betrvg_epoch(explicit_epoch: str | None = None) -> str:
    """Resolves and validates key rotation epoch slug for BetrVG pseudonyms."""
    raw = (
        explicit_epoch
        if explicit_epoch is not None
        else os.environ.get("SCOUT_BETRVG_KEY_EPOCH", "v1")
    ).strip()
    if not _EPOCH_RE.fullmatch(raw):
        raise ValueError(f"Invalid BetrVG key epoch {raw!r}; expected 1-16 alphanumeric chars.")
    return raw


def resolve_betrvg_salt(explicit_salt: str | None = None) -> bytes:
    """Resolves BetrVG HMAC salt fail-closed (minimum 32 characters, never weak defaults)."""
    raw = (
        explicit_salt if explicit_salt is not None else os.environ.get("SCOUT_BETRVG_HMAC_SALT", "")
    ).strip()
    if not raw or len(raw) < 32 or raw in _INSECURE_BETRVG_SALTS:
        raise RuntimeError(
            "Fail-closed BetrVG privacy mode requires a >=32-character "
            "SCOUT_BETRVG_HMAC_SALT (never an insecure default)."
        )
    return raw.encode("utf-8")


class BetrVGPrivacyGuard:
    """Enforces HMAC-SHA256 user-ID pseudonymization and telemetry prompt redaction."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        salt: str | None = None,
        epoch: str | None = None,
    ) -> None:
        self._enabled = enabled
        self._salt = salt
        self._epoch = epoch

    @property
    def enabled(self) -> bool:
        return is_betrvg_enabled() if self._enabled is None else self._enabled

    def pseudonymize_user_id(
        self,
        user_id: str,
        tenant_id: UUID,
        *,
        salt: bytes | None = None,
        epoch: str | None = None,
    ) -> str:
        """Deterministically pseudonymizes a user_id per tenant via keyed HMAC-SHA256."""
        cleaned = user_id.strip()
        if not cleaned:
            raise ValueError("user_id cannot be blank for BetrVG pseudonymization.")
        active_epoch = resolve_betrvg_epoch(epoch if epoch is not None else self._epoch)
        if cleaned.startswith(f"betrvg_{active_epoch}_") and _PSEUDONYM_RE.fullmatch(cleaned):
            return cleaned
        key = salt if salt is not None else resolve_betrvg_salt(self._salt)
        msg = f"{tenant_id}:{cleaned.lower()}".encode()
        digest = hmac.new(key, msg, hashlib.sha256).hexdigest()[:24]
        return f"betrvg_{active_epoch}_{digest}"

    def apply_to_principal(self, principal: PrincipalContext) -> PrincipalContext:
        """Returns a copy of principal with pseudonymized user_id when BetrVG mode is active."""
        if not self.enabled:
            return principal
        pseudo_id = self.pseudonymize_user_id(principal.user_id, principal.tenant_id)
        return principal.model_copy(update={"user_id": pseudo_id})

    def sanitize_telemetry_attributes(
        self,
        attrs: Mapping[str, Any],
        *,
        tenant_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Redacts raw prompt snippets and pseudonymizes principal identifiers in BetrVG mode."""
        out = dict(attrs)
        if not self.enabled:
            return out
        resolved_tid = tenant_id or DEFAULT_TENANT_ID
        for key in list(out.keys()):
            if key in _PROMPT_SNIPPET_ATTRS:
                out[key] = "[REDACTED_BETRVG]"
            elif key in _PRINCIPAL_ID_ATTRS:
                val = out[key]
                if isinstance(val, str) and val.strip() and val.strip().lower() != "anonymous":
                    out[key] = self.pseudonymize_user_id(val, resolved_tid)
        return out


def apply_betrvg_to_principal(principal: PrincipalContext) -> PrincipalContext:
    """Applies active BetrVG pseudonymization policy to a PrincipalContext."""
    return BetrVGPrivacyGuard().apply_to_principal(principal)


def pseudonymize_actor_if_enabled(actor_id: str, tenant_id: UUID) -> str:
    """Pseudonymizes actor_id for storage/audit logs when BetrVG mode is active."""
    guard = BetrVGPrivacyGuard()
    if not guard.enabled:
        return actor_id
    clean = actor_id.strip()
    if not clean or clean.lower() in {"anonymous", "system"}:
        return clean or "anonymous"
    return guard.pseudonymize_user_id(clean, tenant_id)


__all__ = [
    "BetrVGPrivacyGuard",
    "apply_betrvg_to_principal",
    "is_betrvg_enabled",
    "pseudonymize_actor_if_enabled",
    "resolve_betrvg_epoch",
    "resolve_betrvg_salt",
]
