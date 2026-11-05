"""FastAPI dependencies: Out-of-band principal extraction (SEC-04), registries, and stores."""

from __future__ import annotations

import base64
import datetime
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
from typing import Annotated, Any, Final

from fastapi import Header, HTTPException

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
    InMemoryRLSRepository,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.schemas.contracts import (
    ImplementationContextPack,
    MarketRegime,
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


def _resolve_auth_secret(secret_key: str | None = None) -> bytes:
    raw = secret_key or os.environ.get("SCOUT_AUTH_HMAC_SECRET") or DEFAULT_AUTH_HMAC_SECRET
    return raw.encode("utf-8")


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
    require_signed = os.environ.get("SCOUT_REQUIRE_SIGNED_AUTH", "false").strip().lower() in {
        "1",
        "true",
        "yes",
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
            detail=(
                "Signed Authorization Bearer token is required when "
                "SCOUT_REQUIRE_SIGNED_AUTH=true (SEC-04)."
            ),
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

    src_dir = repo_root / "src" / "context_scout"
    if src_dir.exists():
        for py_file in sorted(src_dir.rglob("*.py")):
            try:
                content = py_file.read_text(encoding="utf-8")
            except Exception:
                continue
            rel_path = str(py_file.relative_to(repo_root))
            lines = content.splitlines()
            f_id = uuid.uuid4()
            rec = CodeFileRecord(
                file_id=f_id,
                domain_id="trading_code_scout",
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
                            domain_id="trading_code_scout",
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

    # Seed research doc chunks for quant_research_scout
    repo.insert_doc_chunk(
        admin_principal,
        DocChunkRecord(
            chunk_id=uuid.uuid4(),
            domain_id="quant_research_scout",
            doc_id="arxiv:2606.31387",
            doc_title="Signature-Based Optimal Execution for Statistical Arbitrage",
            source_uri="urn:arxiv:2606.31387.pdf",
            market_scope=MarketRegime.US_NYSE_NASDAQ,
            published_date=datetime.date(2026, 6, 15),
            authority_weight=1.0,
            section_heading="Optimal Liquidation Trajectory",
            contextual_header="Quantitative optimal execution analysis.",
            verbatim_content=(
                "Optimal execution requires balancing market impact and inventory risk. "
                "Order flow imbalance exhibits mean reversion over sub-second horizons."
            ),
            metadata_json={},
            allowed_scopes=["public_eng", "public_quant", "quant_restricted"],
            embedding=[0.05] * 768,
        ),
    )
    repo.insert_doc_chunk(
        admin_principal,
        DocChunkRecord(
            chunk_id=uuid.uuid4(),
            domain_id="quant_research_scout",
            doc_id="arxiv:2607.12944",
            doc_title="Microstructure Alpha in European Equities and XETRA Order Books",
            source_uri="urn:arxiv:2607.12944.pdf",
            market_scope=MarketRegime.GERMANY_XETRA_DAX,
            published_date=datetime.date(2026, 7, 20),
            authority_weight=1.0,
            section_heading="Tick-Level Dynamics",
            contextual_header="European microstructure alpha modeling.",
            verbatim_content=(
                "XETRA continuous trading auctions enforce a minimum round-trip friction floor. "
                "Statistical arbitrage strategies must clear at least 10 bps to remain profitable."
            ),
            metadata_json={},
            allowed_scopes=["public_eng", "public_quant", "quant_restricted"],
            embedding=[0.05] * 768,
        ),
    )
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
