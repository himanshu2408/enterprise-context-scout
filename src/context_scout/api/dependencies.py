"""FastAPI dependencies: Out-of-band principal extraction (SEC-04), registries, and stores."""

from __future__ import annotations

import ast
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final

from fastapi import Header, HTTPException

if TYPE_CHECKING:
    from context_scout.agents.scout_workflow import LocalGemmaClient

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    ASTSymbolRecord,
    CodeFileRecord,
    InMemoryRLSRepository,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.plugins.registry import PluginRegistry
from context_scout.schemas.contracts import (
    ImplementationContextPack,
    PrincipalContext,
    RoleType,
)
from context_scout.security.rbac import _CURRENT_PRINCIPAL
from context_scout.telemetry import ScoutTelemetryTracer

logger = logging.getLogger(__name__)

DEFAULT_ROLE_SCOPES: Final[dict[RoleType, list[str]]] = {
    RoleType.ENGINEER: ["public_eng", "repo:trading_strategies", "public_quant"],
    RoleType.QUANT_RESEARCHER: ["public_quant", "quant_restricted", "corpus:quant_papers"],
    RoleType.PRODUCT_MANAGER: ["public_eng", "public_quant"],
    RoleType.COMPLIANCE_OFFICER: [
        "public_eng",
        "public_quant",
        "quant_restricted",
        "audit:compliance",
    ],
}


DEFAULT_AUTH_HMAC_SECRET: Final[str] = "scout-dev-hmac-secret-override-in-prod-sec04"

INSECURE_DEFAULT_SECRETS: Final[frozenset[str]] = frozenset(
    {
        "scout-dev-hmac-secret-override-in-prod-sec04",
        "scout-dev-secret-key-change-in-production",
        "changeme",
        "secret",
        "dev-secret",
        "insecure-default-secret",
    }
)


def is_production_environment() -> bool:
    """Detects whether context-scout is operating in a production or staging environment."""
    env = (
        (
            os.environ.get("SCOUT_ENV")
            or os.environ.get("ENVIRONMENT")
            or os.environ.get("ENV")
            or ""
        )
        .strip()
        .lower()
    )
    return env in {"production", "prod", "staging"}


def _resolve_auth_secret(secret_key: str | None = None) -> bytes:
    """Resolves HMAC secret key with absolute fail-closed safety enforcement.

    Implements SEC-VULN-01 / Critical B fail-closed authentication.

    Fails closed by default:
    - In production (is_production_environment() is True), SCOUT_ALLOW_INSECURE_DEV_SECRET
      is strictly IGNORED. Any unset secret, secret in INSECURE_DEFAULT_SECRETS, or
      secret shorter than 32 characters (whether from SCOUT_AUTH_HMAC_SECRET or an
      explicit secret_key) unconditionally raises a fatal RuntimeError.
    - In non-production environments:
      - If an explicit non-empty secret_key argument is passed (e.g., for testing or
        crafting adversarial test tokens), it is accepted directly and logs a warning
        when matching INSECURE_DEFAULT_SECRETS.
      - Otherwise, SCOUT_AUTH_HMAC_SECRET must be >=32 characters and not in
        INSECURE_DEFAULT_SECRETS; if unset, empty, weak (<32 characters), or in
        INSECURE_DEFAULT_SECRETS, it raises RuntimeError unless
        SCOUT_ALLOW_INSECURE_DEV_SECRET is explicitly set to 'true' ('1', 'true', 'yes').
    """
    is_prod = is_production_environment()
    allow_insecure = not is_prod and os.environ.get(
        "SCOUT_ALLOW_INSECURE_DEV_SECRET", "false"
    ).strip().lower() in {"1", "true", "yes"}

    # 1. Caller passed an explicit secret_key (e.g. testing or programmatic injection)
    if secret_key is not None and len(secret_key) > 0:
        raw_key = secret_key
        if is_prod and (raw_key in INSECURE_DEFAULT_SECRETS or len(raw_key) < 32):
            raise RuntimeError(
                "Insecure default or weak HMAC secret detected in production environment. "
                "A cryptographically secure SCOUT_AUTH_HMAC_SECRET (minimum 32 characters) "
                "must be configured in production (SEC-VULN-01)."
            )
        if raw_key in INSECURE_DEFAULT_SECRETS:
            logger.warning(
                "Using known insecure development HMAC secret key. "
                "This is insecure and strictly prohibited in production environments."
            )
        return raw_key.encode("utf-8")

    # 2. Check environment variable SCOUT_AUTH_HMAC_SECRET
    env_secret = os.environ.get("SCOUT_AUTH_HMAC_SECRET")
    if env_secret is None or not env_secret.strip():
        # In production, dev override is banned: unconditionally fail closed
        if is_prod:
            raise RuntimeError(
                "Insecure default or weak HMAC secret detected in production environment. "
                "A cryptographically secure SCOUT_AUTH_HMAC_SECRET (minimum 32 characters) "
                "must be configured in production (SEC-VULN-01)."
            )
        # In non-production, allow insecure dev secret only if explicitly opted in
        if allow_insecure:
            logger.warning(
                "Using default development HMAC secret for SCOUT_AUTH_HMAC_SECRET because "
                "SCOUT_ALLOW_INSECURE_DEV_SECRET is explicitly set. "
                "This is insecure and strictly prohibited in production environments."
            )
            return DEFAULT_AUTH_HMAC_SECRET.encode("utf-8")

        raise RuntimeError(
            "SCOUT_AUTH_HMAC_SECRET is unset or empty. Context Scout enforces fail-closed "
            "authentication: you must configure a cryptographically secure secret "
            "(minimum 32 characters) via SCOUT_AUTH_HMAC_SECRET, or explicitly set "
            "SCOUT_ALLOW_INSECURE_DEV_SECRET='true' for local development."
        )

    # 3. Environment secret is present: validate length and insecurity
    raw_env = env_secret.strip()
    if is_prod and (raw_env in INSECURE_DEFAULT_SECRETS or len(raw_env) < 32):
        raise RuntimeError(
            "Insecure default or weak HMAC secret detected in production environment. "
            "A cryptographically secure SCOUT_AUTH_HMAC_SECRET (minimum 32 characters) "
            "must be configured in production (SEC-VULN-01)."
        )

    if (
        not is_prod
        and not allow_insecure
        and (raw_env in INSECURE_DEFAULT_SECRETS or len(raw_env) < 32)
    ):
        raise RuntimeError(
            "Insecure or weak HMAC secret detected in SCOUT_AUTH_HMAC_SECRET "
            "(minimum 32 characters). Context Scout enforces fail-closed "
            "authentication unless SCOUT_ALLOW_INSECURE_DEV_SECRET='true' "
            "is explicitly configured."
        )

    if raw_env in INSECURE_DEFAULT_SECRETS:
        logger.warning(
            "Using default development HMAC secret for SCOUT_AUTH_HMAC_SECRET. "
            "This is insecure and strictly prohibited in production environments."
        )

    return raw_env.encode("utf-8")


def validate_auth_secret() -> bytes:
    """Validates application HMAC authentication configuration fail-closed on startup."""
    return _resolve_auth_secret()


def create_signed_principal_token(
    user_id: str,
    role: str | RoleType,
    allowed_scopes: list[str],
    secret_key: str | None = None,
    ttl_seconds: int = 3600,
    max_session_paid_tokens: int = 50_000,
) -> str:
    """Create an HMAC-SHA256 signed principal attestation token (SEC-04)."""
    role_val = role.value if isinstance(role, RoleType) else str(role).strip().lower()
    now_ts = int(time.time())
    payload = {
        "sub": user_id.strip(),
        "role": role_val,
        "scopes": list(allowed_scopes),
        "max_tokens": max_session_paid_tokens,
        "iat": now_ts,
        "exp": now_ts + ttl_seconds,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    b64_payload = base64.urlsafe_b64encode(payload_bytes).decode("ascii").rstrip("=")
    sig = hmac.new(
        _resolve_auth_secret(secret_key),
        b64_payload.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return f"{b64_payload}.{sig}"


def verify_signed_principal_token(
    token: str,
    secret_key: str | None = None,
) -> PrincipalContext:
    """Verify an HMAC-SHA256 signed principal token and return PrincipalContext (SEC-04)."""
    parts = token.strip().split(".")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise HTTPException(
            status_code=401,
            detail="Malformed signed principal token (expected '<payload>.<hmac_sha256>').",
        )
    b64_payload, provided_sig = parts
    expected_sig = hmac.new(
        _resolve_auth_secret(secret_key),
        b64_payload.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(provided_sig, expected_sig):
        raise HTTPException(
            status_code=401,
            detail="Invalid HMAC-SHA256 signature on principal token (SEC-04).",
        )
    pad = "=" * (-len(b64_payload) % 4)
    try:
        raw_json = base64.urlsafe_b64decode((b64_payload + pad).encode("ascii"))
        data = json.loads(raw_json.decode("utf-8"))
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid base64/JSON payload in signed principal token.",
        ) from None

    exp = int(data.get("exp", 0))
    if time.time() > exp:
        raise HTTPException(
            status_code=401,
            detail="Signed principal token has expired (SEC-04).",
        )
    try:
        role = RoleType(str(data.get("role", "")).lower())
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Invalid role inside signed principal token.",
        ) from None

    scopes_raw = data.get("scopes")
    scopes = (
        [str(s).strip() for s in scopes_raw if str(s).strip()]
        if isinstance(scopes_raw, list) and scopes_raw
        else list(DEFAULT_ROLE_SCOPES[role])
    )
    return PrincipalContext(
        user_id=str(data.get("sub") or "authenticated_user").strip(),
        role=role,
        allowed_scopes=scopes,
        max_session_paid_tokens=int(data.get("max_tokens", 50_000)),
    )


def get_principal_context(
    x_principal_user_id: Annotated[str | None, Header()] = "demo_engineer",
    x_principal_role: Annotated[str | None, Header()] = "engineer",
    x_principal_scopes: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> PrincipalContext:
    """Extracts immutable PrincipalContext out-of-band from signed token or headers (SEC-04)."""
    raw_require = os.environ.get("SCOUT_REQUIRE_SIGNED_AUTH")
    if raw_require is None or not raw_require.strip():
        require_signed = True
    else:
        require_signed = raw_require.strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
            "disable",
            "disabled",
        }

    if authorization and authorization.strip():
        auth_val = authorization.strip()
        if not auth_val.lower().startswith("bearer "):
            raise HTTPException(
                status_code=401,
                detail="Authorization header must use 'Bearer <signed_token>' scheme.",
            )
        raw_token = auth_val[7:].strip()
        principal = verify_signed_principal_token(raw_token)
        _CURRENT_PRINCIPAL.set(principal)
        return principal

    if require_signed:
        raise HTTPException(
            status_code=401,
            detail="Signed Authorization Bearer token is required.",
        )

    user_id = (x_principal_user_id or "demo_engineer").strip()
    raw_role = (x_principal_role or "engineer").strip().lower()

    try:
        role = RoleType(raw_role)
    except ValueError:
        valid_roles = [r.value for r in RoleType]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid principal role '{raw_role}'. Must be one of: {valid_roles}",
        ) from None

    if x_principal_scopes and x_principal_scopes.strip():
        scopes = [s.strip() for s in x_principal_scopes.split(",") if s.strip()]
    else:
        scopes = list(DEFAULT_ROLE_SCOPES[role])

    principal = PrincipalContext(
        user_id=user_id,
        role=role,
        allowed_scopes=scopes,
        max_session_paid_tokens=50_000,
    )

    # Bind out-of-band contextvar
    _CURRENT_PRINCIPAL.set(principal)
    return principal


def reset_current_principal(token: Any = None) -> None:
    """Explicitly resets the session principal ContextVar (SEC-04)."""
    if token is not None:
        _CURRENT_PRINCIPAL.reset(token)
    else:
        _CURRENT_PRINCIPAL.set(None)


@dataclass
class TokenRecord:
    """Preview confirmation token record."""

    token: str
    pack: ImplementationContextPack
    patch_diff: str
    target_files: list[str]
    model_id: str
    principal: PrincipalContext
    created_at: float
    ttl_seconds: int = 3600


class ConfirmationTokenStore:
    """Thread-safe in-memory store for handoff confirmation tokens."""

    def __init__(self, default_ttl_seconds: int = 3600) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, TokenRecord] = {}
        self._default_ttl_seconds = default_ttl_seconds

    def save(
        self,
        token: str,
        pack: ImplementationContextPack,
        patch_diff: str,
        target_files: list[str],
        model_id: str,
        principal: PrincipalContext,
        ttl_seconds: int | None = None,
    ) -> TokenRecord:
        with self._lock:
            now = time.time()
            expired_keys = [
                k for k, v in self._records.items() if now - v.created_at > v.ttl_seconds
            ]
            for k in expired_keys:
                del self._records[k]
            rec = TokenRecord(
                token=token,
                pack=pack,
                patch_diff=patch_diff,
                target_files=target_files,
                model_id=model_id,
                principal=principal,
                created_at=now,
                ttl_seconds=ttl_seconds or self._default_ttl_seconds,
            )
            self._records[token] = rec
            return rec

    def get(self, token: str) -> TokenRecord | None:
        with self._lock:
            rec = self._records.get(token)
            if rec is None:
                return None
            if time.time() - rec.created_at > rec.ttl_seconds:
                del self._records[token]
                return None
            return rec

    def invalidate(self, token: str) -> None:
        with self._lock:
            self._records.pop(token, None)


_token_store = ConfirmationTokenStore()


def get_confirmation_token_store() -> ConfirmationTokenStore:
    """Provides singleton ConfirmationTokenStore dependency."""
    return _token_store


def get_repo_root() -> Path:
    """Resolves enterprise-context-scout root directory."""
    # From src/context_scout/api/dependencies.py up 4 levels to root
    return Path(__file__).resolve().parents[3]


_domain_registry: DomainRegistry | None = None
_domain_registry_lock = threading.Lock()


def get_domain_registry() -> DomainRegistry:
    """Provides pre-loaded DomainRegistry singleton."""
    global _domain_registry
    if _domain_registry is None:
        with _domain_registry_lock:
            if _domain_registry is None:
                domains_dir = get_repo_root() / "domains"
                registry = DomainRegistry(domains_dir)
                registry.load_all()
                _domain_registry = registry
    return _domain_registry


_shared_repo: Any = None
_shared_repo_lock = threading.Lock()


def _populate_in_memory_repository(repo_root: Path) -> InMemoryRLSRepository:
    """Populates an in-memory repository with real codebase files, AST symbols, and doc chunks."""
    repo = InMemoryRLSRepository()
    admin_principal = PrincipalContext(
        user_id="api_bootstrapper",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "public_quant", "quant_restricted", "audit:compliance"],
    )
    primary_plugin = PluginRegistry.get_primary_domain_plugin()
    code_domain_id = (
        primary_plugin.get_role_domain_ids().get("code", "default")
        if primary_plugin is not None
        else "default"
    )

    src_dir = repo_root / "src" / "context_scout"
    if src_dir.exists():
        for py_file in sorted(src_dir.rglob("*.py")):
            try:
                content = py_file.read_text(encoding="utf-8")
            except Exception as exc:
                logger.debug(
                    "Skipping unreadable file %s during in-memory bootstrap (%s): %s",
                    py_file,
                    type(exc).__name__,
                    exc,
                )
                continue
            rel_path = str(py_file.relative_to(repo_root))
            lines = content.splitlines()
            f_id = uuid.uuid4()
            rec = CodeFileRecord(
                file_id=f_id,
                domain_id=code_domain_id,
                repo_name="enterprise-context-scout",
                file_path=rel_path,
                git_commit_sha="c0ffee123456",
                git_blob_sha="b10b123456",
                language="python",
                allowed_scopes=["public_eng"],
                raw_content=content,
                line_count=len(lines),
            )
            repo.insert_code_file(admin_principal, rec)

            module_parts = py_file.relative_to(repo_root / "src").with_suffix("").parts
            mod_dotted = ".".join(module_parts)

            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        cls_name = node.name
                        q_name = f"{mod_dotted}.{cls_name}"
                        repo.insert_ast_symbol(
                            admin_principal,
                            ASTSymbolRecord(
                                symbol_id=uuid.uuid4(),
                                file_id=f_id,
                                domain_id=code_domain_id,
                                qualified_name=q_name,
                                symbol_kind="class",
                                start_line=node.lineno,
                                end_line=node.end_lineno or min(node.lineno + 50, len(lines)),
                                signature_skeleton=f"class {cls_name}:",
                                contextual_summary=f"Class definition for {cls_name}",
                                allowed_scopes=["public_eng"],
                                embedding=[0.05] * 768,
                            ),
                        )
            except Exception as exc:
                logger.debug(
                    "AST parse failed for %s, falling back to regex class scan (%s): %s",
                    rel_path,
                    type(exc).__name__,
                    exc,
                )
                for idx, line in enumerate(lines, start=1):
                    cls_match = re.match(r"^class\s+([A-Za-z0-9_]+)", line)
                    if cls_match:
                        cls_name = cls_match.group(1)
                        q_name = f"{mod_dotted}.{cls_name}"
                        repo.insert_ast_symbol(
                            admin_principal,
                            ASTSymbolRecord(
                                symbol_id=uuid.uuid4(),
                                file_id=f_id,
                                domain_id=code_domain_id,
                                qualified_name=q_name,
                                symbol_kind="class",
                                start_line=idx,
                                end_line=min(idx + 50, len(lines)),
                                signature_skeleton=f"class {cls_name}:",
                                contextual_summary=f"Class definition for {cls_name}",
                                allowed_scopes=["public_eng"],
                                embedding=[0.05] * 768,
                            ),
                        )

    for domain_plugin in PluginRegistry.list_domain_plugins():
        seed_payload = domain_plugin.get_demo_seed_payloads()
        for doc_rec in seed_payload.get("doc_chunks", []):
            repo.insert_doc_chunk(admin_principal, doc_rec)
    return repo


def get_repository() -> Any:
    """Provides shared repository (Postgres if connected, else populated in-memory)."""
    global _shared_repo
    if _shared_repo is None:
        with _shared_repo_lock:
            if _shared_repo is None:
                repo_root = get_repo_root()
                settings = ScoutSettings()
                try:
                    conn_mgr = RLSConnectionManager(settings.db_app_url)
                    probe = PrincipalContext(
                        user_id="db_probe",
                        role=RoleType.ENGINEER,
                        allowed_scopes=["public_eng"],
                    )
                    with conn_mgr.scoped_transaction(probe) as conn:
                        conn.execute("SELECT 1;")
                    # Live PostgreSQL available
                    from context_scout.mcp_servers.common import PostgresScoutGraphAdapter

                    _shared_repo = PostgresScoutGraphAdapter(conn_mgr, settings)
                    logger.info("API initialized with live PostgreSQL 17 repository.")
                except Exception as exc:
                    logger.info(
                        "Live PostgreSQL 17 unavailable (%s); using InMemoryRLSRepository.",
                        exc,
                    )
                    _shared_repo = _populate_in_memory_repository(repo_root)
    return _shared_repo


def get_telemetry_tracer() -> ScoutTelemetryTracer:
    """Provides ScoutTelemetryTracer singleton."""
    return ScoutTelemetryTracer.get_instance()


_local_gemma_client: LocalGemmaClient | None = None
_local_gemma_lock = threading.Lock()


def get_local_gemma_client() -> LocalGemmaClient:
    """Provides a shared LocalGemmaClient so loaded MLX weights stay warm across requests."""
    global _local_gemma_client
    if _local_gemma_client is None:
        with _local_gemma_lock:
            if _local_gemma_client is None:
                from context_scout.agents.scout_workflow import LocalGemmaClient

                registry = get_domain_registry()
                _local_gemma_client = LocalGemmaClient(adapter_pool=registry.adapter_pool)
    return _local_gemma_client
