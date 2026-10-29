"""Enterprise Out-of-Band RBAC and Session Principal Binding (SEC-04)."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Final

from psycopg import Connection

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.db.session import RLSConnectionManager
from context_scout.schemas.contracts import PrincipalContext


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
        "scopes",
        "principal_scopes",
        "app.principal_scopes",
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


def validate_tool_call_arguments(arguments: Mapping[str, object]) -> None:
    """Ensures PrincipalContext and scopes are not passed as LLM tool arguments (SEC-04)."""
    forbidden_detected = [k for k in arguments if k.lower() in FORBIDDEN_TOOL_ARGUMENTS]
    if forbidden_detected:
        raise SecurityError(
            f"Security parameter injection detected: {sorted(forbidden_detected)}. "
            "PrincipalContext and authorization scopes cannot be supplied via tool parameters "
            "(SEC-04)."
        )
    for k, v in arguments.items():
        if "\x00" in k:
            raise ValueError(f"NUL byte detected in argument key '{k}'")
        if isinstance(v, str) and "\x00" in v:
            raise ValueError(f"NUL byte detected in argument '{k}'")


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

    TOOL_ALIASES: Final[dict[str, dict[str, str]]] = {
        "trading_code_scout": {
            "search_code_trigram": "code_repo_mcp:grep_repo",
            "lookup_ast_symbol": "code_repo_mcp:search_ast_symbols",
            "expand_ast_1hop_closure": "code_repo_mcp:get_symbol_callers",
            "slice_file_lines": "code_repo_mcp:get_symbol_definition",
            "search_github_issues": "code_repo_mcp:list_github_issues",
        },
        "quant_research_scout": {
            "search_quant_papers_hybrid": "quant_corpus_mcp:search_papers_hybrid",
            "extract_paper_equations_and_tables": "quant_corpus_mcp:get_paper_section",
            "verify_market_friction_assumptions": "quant_corpus_mcp:compare_market_frictions",
            "compare_cross_market_regimes": "quant_corpus_mcp:compare_market_frictions",
        },
    }

    def __init__(self, registry: DomainRegistry) -> None:
        self._registry = registry

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
            except KeyError:
                pass

            if domain_id == "trading_code_scout":
                return f"code_repo_mcp:{tool_name}"
            if domain_id == "quant_research_scout":
                return f"quant_corpus_mcp:{tool_name}"

        return tool_name

    def is_tool_allowed(self, domain_id: str, tool_name: str) -> bool:
        """Checks if a tool is permitted for the given domain."""
        try:
            manifest = self._registry.get(domain_id)
        except KeyError:
            return False

        canonical = self.canonicalize_tool_name(domain_id, tool_name)
        if canonical in manifest.mcp_read_tools_allowed:
            return True
        if tool_name in manifest.mcp_read_tools_allowed:
            return True

        bare = tool_name.split(":")[-1]
        return any(declared.split(":")[-1] == bare for declared in manifest.mcp_read_tools_allowed)

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
            except KeyError:
                allowed = []
            raise DomainToolAccessDeniedError(
                f"Tool '{tool_name}' is not authorized for domain '{domain_id}'. "
                f"Allowed tools: {allowed}"
            )

        return resolved_principal
