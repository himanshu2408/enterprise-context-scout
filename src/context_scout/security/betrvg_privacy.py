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
        "user_query",
        "prompt",
        "prompt_snippet",
        "query_snippet",
        "input_text",
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

_EMAIL_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9_%+-]{1,64}(?:\.[A-Za-z0-9_%+-]{1,64}){0,4}"
    r"@(?:[A-Za-z0-9-]{1,63}\.){1,6}[A-Za-z]{2,24}\b"
)
_DIAGNOSTIC_QUERY_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(\b(?:query|prompt|user_task|input)\b\s*[:=]\s*)(?:'[^'\n]{0,512}'|\"[^\"\n]{0,512}\"|[^\n;]{1,512})"
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

    def scrub_text_if_enabled(
        self,
        text: str,
        *,
        tenant_id: UUID | None = None,
    ) -> str:
        """Redacts emails and embedded query/prompt snippets from diagnostic/error strings."""
        if not self.enabled or not text:
            return text
        resolved_tid = tenant_id or DEFAULT_TENANT_ID

        def _replace_email(match: re.Match[str]) -> str:
            return self.pseudonymize_user_id(match.group(0), resolved_tid)

        scrubbed = _EMAIL_RE.sub(_replace_email, text) if "@" in text else text
        scrubbed = _DIAGNOSTIC_QUERY_RE.sub(r"\1[REDACTED_BETRVG]", scrubbed)
        return scrubbed

    def scrub_dict_if_enabled(
        self,
        payload: Mapping[str, Any],
        *,
        tenant_id: UUID | None = None,
        _depth: int = 0,
    ) -> dict[str, Any]:
        """Recursively redacts prompts, emails, and principal IDs from nested dictionaries."""
        out: dict[str, Any] = {}
        if not self.enabled:
            return dict(payload)
        if _depth > 16:
            return {"_truncated": "[REDACTED_BETRVG_DEPTH]"}
        resolved_tid = tenant_id or DEFAULT_TENANT_ID
        for key, val in payload.items():
            key_str = str(key)
            key_l = key_str.lower()
            if key_l in _PROMPT_SNIPPET_ATTRS or key_str in _PROMPT_SNIPPET_ATTRS:
                out[key_str] = "[REDACTED_BETRVG]"
            elif key_l in _PRINCIPAL_ID_ATTRS or key_str in _PRINCIPAL_ID_ATTRS:
                if (
                    isinstance(val, str)
                    and val.strip()
                    and val.strip().lower() not in {"anonymous", "system"}
                ):
                    out[key_str] = self.pseudonymize_user_id(val, resolved_tid)
                else:
                    out[key_str] = val
            elif isinstance(val, Mapping):
                out[key_str] = self.scrub_dict_if_enabled(
                    val, tenant_id=resolved_tid, _depth=_depth + 1
                )
            elif isinstance(val, list):
                cleaned_list: list[Any] = []
                for item in val:
                    if isinstance(item, Mapping):
                        cleaned_list.append(
                            self.scrub_dict_if_enabled(
                                item, tenant_id=resolved_tid, _depth=_depth + 1
                            )
                        )
                    elif isinstance(item, str):
                        cleaned_list.append(
                            self.scrub_text_if_enabled(item, tenant_id=resolved_tid)
                        )
                    else:
                        cleaned_list.append(item)
                out[key_str] = cleaned_list
            elif isinstance(val, str):
                out[key_str] = self.scrub_text_if_enabled(val, tenant_id=resolved_tid)
            else:
                out[key_str] = val
        return out

    def sanitize_telemetry_attributes(
        self,
        attrs: Mapping[str, Any],
        *,
        tenant_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Redacts raw prompt snippets and pseudonymizes principal identifiers in BetrVG mode."""
        if not self.enabled:
            return dict(attrs)
        return self.scrub_dict_if_enabled(attrs, tenant_id=tenant_id)


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
