"""Enterprise Out-of-Band RBAC and Session Principal Binding (SEC-04)."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, ClassVar, Final

from psycopg import Connection

from context_scout.db.session import RLSConnectionManager
from context_scout.schemas.contracts import PrincipalContext

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from context_scout.agents.domain_registry import DomainRegistry


class SecurityError(Exception):
    """Base exception for all security violations."""


class AuthenticationRequiredError(SecurityError):
    """Raised when an operation requires an authenticated PrincipalContext but none is bound."""


class DomainToolAccessDeniedError(SecurityError):
    """Raised when an MCP tool invocation violates the DomainPack allow-list."""


class ScopeAccessDeniedError(SecurityError):
    """Raised when a principal attempts an action without holding the required scope."""


_CURRENT_PRINCIPAL: Final[ContextVar[PrincipalContext | None]] = ContextVar(
    "current_session_principal", default=None
)

FORBIDDEN_TOOL_ARGUMENTS: Final[frozenset[str]] = frozenset(
    {
        "principal",
        "principal_context",
        "allowed_scopes",
        "user_id",
        "role",
        "roles",
        "scopes",
        "principal_scopes",
        "app.principal_scopes",
        "tenant_id",
        "app.tenant_id",
        "app.current_tenant_id",
        "team_id",
        "max_session_paid_tokens",
        "session_budget_tokens",
    }
)


@contextmanager
def bind_session_principal(principal: PrincipalContext) -> Iterator[None]:
    """Binds PrincipalContext to the active async task or thread out-of-band (SEC-04)."""
    token = _CURRENT_PRINCIPAL.set(principal)
    try:
        yield
    finally:
        _CURRENT_PRINCIPAL.reset(token)


def get_current_principal() -> PrincipalContext:
    """Retrieves the active out-of-band PrincipalContext.

    Raises:
        AuthenticationRequiredError: If no PrincipalContext is bound to the current context.
    """
    principal = _CURRENT_PRINCIPAL.get()
    if principal is None:
        raise AuthenticationRequiredError(
            "Out-of-band PrincipalContext is required. "
            "Tools cannot be executed without session binding."
        )
    return principal


def has_bound_principal() -> bool:
    """Returns True if a PrincipalContext is currently bound out-of-band."""
    return _CURRENT_PRINCIPAL.get() is not None


@contextmanager
def scoped_db_session(
    connection_manager: RLSConnectionManager,
    principal: PrincipalContext | None = None,
    enable_hnsw_iterative_scan: bool = True,
) -> Iterator[Connection[tuple[object, ...]]]:
    """Opens a PostgreSQL 17 transaction with app.principal_scopes bound out-of-band."""
    resolved_principal = principal or get_current_principal()
    with connection_manager.scoped_transaction(
        resolved_principal, enable_hnsw_iterative_scan=enable_hnsw_iterative_scan
    ) as conn:
        yield conn


_MAX_TOOL_ARG_DEPTH: Final[int] = 16


def _scan_tool_value_for_injection(
    value: object,
    *,
    depth: int,
    forbidden_detected: list[str],
) -> None:
    if depth > _MAX_TOOL_ARG_DEPTH:
        raise ValueError(
            f"Tool arguments exceed maximum allowed nesting depth ({_MAX_TOOL_ARG_DEPTH})."
        )
    if isinstance(value, str):
        if "\x00" in value:
            raise ValueError("NUL byte detected in nested tool argument")
        return
    if isinstance(value, Mapping):
        for k, v in value.items():
            key_str = str(k)
            if "\x00" in key_str:
                raise ValueError(f"NUL byte detected in argument key '{key_str}'")
            if key_str.lower() in FORBIDDEN_TOOL_ARGUMENTS:
                forbidden_detected.append(key_str)
            _scan_tool_value_for_injection(
                v, depth=depth + 1, forbidden_detected=forbidden_detected
            )
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _scan_tool_value_for_injection(
                item, depth=depth + 1, forbidden_detected=forbidden_detected
            )


def validate_tool_call_arguments(arguments: Mapping[str, object]) -> None:
    """Ensures PrincipalContext, tenant/role keys, and NUL bytes are absent in tool arguments."""
    forbidden_detected: list[str] = []
    _scan_tool_value_for_injection(arguments, depth=0, forbidden_detected=forbidden_detected)
    if forbidden_detected:
        raise SecurityError(
            f"Security parameter injection detected: {sorted(set(forbidden_detected))}. "
            "PrincipalContext and authorization scopes cannot be supplied via tool parameters "
            "(SEC-04)."
        )


def require_scope(required_scope: str, principal: PrincipalContext | None = None) -> None:
    """Verifies that the principal possesses the required scope; raises ScopeAccessDeniedError."""
    p = principal or get_current_principal()
    if required_scope not in p.allowed_scopes:
        raise ScopeAccessDeniedError(
            f"Principal '{p.user_id}' does not possess required scope '{required_scope}'. "
            f"Granted scopes: {p.allowed_scopes}"
        )


class DomainAuthorizationGate:
    """Enforces per-DomainPack tool allow-lists (allowed_mcp_tools from domain.yaml)."""

    TOOL_ALIASES: ClassVar[dict[str, dict[str, str]]] = {}

    def __init__(self, registry: DomainRegistry) -> None:
        self._registry = registry

    @classmethod
    def register_tool_aliases(cls, domain_id: str, aliases: dict[str, str]) -> None:
        """Registers domain-specific tool aliases dynamically from domain plugins."""
        cls.TOOL_ALIASES.setdefault(domain_id, {}).update(aliases)

    def canonicalize_tool_name(self, domain_id: str, tool_name: str) -> str:
        """Resolves tool aliases and qualified prefixes to canonical manifest tool names."""
        domain_aliases = self.TOOL_ALIASES.get(domain_id, {})
        if tool_name in domain_aliases:
            return domain_aliases[tool_name]

        if ":" not in tool_name:
            try:
                manifest = self._registry.get(domain_id)
                for declared in manifest.mcp_read_tools_allowed:
                    if declared.split(":")[-1] == tool_name:
                        return declared
                # Derive prefix dynamically from manifest tools or default server
                if hasattr(manifest, "default_mcp_server") and manifest.default_mcp_server:
                    return f"{manifest.default_mcp_server}:{tool_name}"
                elif manifest.mcp_read_tools_allowed:
                    prefix = manifest.mcp_read_tools_allowed[0].split(":")[0]
                    return f"{prefix}:{tool_name}"
            except KeyError as exc:
                logger.debug(
                    "Domain %r not found in DomainRegistry during tool canonicalization (%s)",
                    domain_id,
                    type(exc).__name__,
                )

        return tool_name

    def is_tool_allowed(self, domain_id: str, tool_name: str) -> bool:
        """Checks if a tool is permitted for the given domain."""
        try:
            manifest = self._registry.get(domain_id)
        except KeyError as exc:
            logger.debug(
                "Denying tool %r: domain %r not registered (%s)",
                tool_name,
                domain_id,
                type(exc).__name__,
            )
            return False

        canonical = self.canonicalize_tool_name(domain_id, tool_name)
        if canonical in manifest.mcp_read_tools_allowed:
            return True
        if tool_name in manifest.mcp_read_tools_allowed:
            return True

        if ":" not in tool_name:
            return any(
                declared.split(":")[-1] == tool_name for declared in manifest.mcp_read_tools_allowed
            )
        return False

    def authorize_tool_call(
        self,
        domain_id: str,
        tool_name: str,
        principal: PrincipalContext | None = None,
        enforce_authentication: bool = True,
    ) -> PrincipalContext | None:
        """Validates tool authorization against domain allow-list and principal context."""
        resolved_principal: PrincipalContext | None = None
        if enforce_authentication:
            resolved_principal = principal or get_current_principal()
        else:
            resolved_principal = principal

        if not self.is_tool_allowed(domain_id, tool_name):
            try:
                allowed = self._registry.get(domain_id).mcp_read_tools_allowed
            except KeyError as exc:
                logger.debug(
                    "No tool allow-list registered for domain %r (%s)",
                    domain_id,
                    type(exc).__name__,
                )
                allowed = []
            raise DomainToolAccessDeniedError(
                f"Tool '{tool_name}' is not authorized for domain '{domain_id}'. "
                f"Allowed tools: {allowed}"
            )

        return resolved_principal
