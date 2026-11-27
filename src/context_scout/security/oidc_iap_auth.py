"""Google Cloud IAP (ES256 JWKS) & Enterprise OIDC JWKS Verifiers with SCIM & BetrVG (M5-01)."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal
from urllib.parse import urlparse
from uuid import UUID

import httpx
import jwt
from jwt.exceptions import InvalidTokenError

from context_scout.schemas.contracts import DEFAULT_TENANT_ID, PrincipalContext, RoleType
from context_scout.security.betrvg_privacy import BetrVGPrivacyGuard
from context_scout.security.permission_mirror import sanitize_scope_identifier
from context_scout.security.scim import SCIMDirectoryService, get_scim_directory_service

logger = logging.getLogger(__name__)

OIDCProviderKind = Literal["google_workspace", "keycloak", "entra_id", "okta", "generic"]

CLOUD_IAP_ISSUER: Final[str] = "https://cloud.google.com/iap"
CLOUD_IAP_JWKS_URL: Final[str] = "https://www.gstatic.com/iap/verify/public_key-jwk"
GOOGLE_WORKSPACE_ISSUER: Final[str] = "https://accounts.google.com"
GOOGLE_WORKSPACE_JWKS_URL: Final[str] = "https://www.googleapis.com/oauth2/v3/certs"

ALLOWED_OIDC_ALGORITHMS: Final[frozenset[str]] = frozenset(
    {"RS256", "RS384", "RS512", "ES256", "ES384", "PS256"}
)
ALLOWED_IAP_ALGORITHMS: Final[frozenset[str]] = frozenset({"ES256"})

_LOCAL_TEST_HOSTS: Final[frozenset[str]] = frozenset(
    {"127.0.0.1", "localhost", "::1", "testserver"}
)

_ROLE_PRECEDENCE: Final[dict[RoleType, int]] = {
    RoleType.COMPLIANCE_OFFICER: 4,
    RoleType.ENGINEER: 3,
    RoleType.PRODUCT_MANAGER: 2,
    RoleType.DOMAIN_RESEARCHER: 1,
}


class AuthVerificationError(Exception):
    """Fail-closed authentication or authorization verification failure."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class _CachedJWKS:
    fetched_at: float
    keys_by_kid: dict[str, jwt.PyJWK]
    ordered_keys: list[jwt.PyJWK]
    negative_kid_cooldowns: dict[str, float] = field(default_factory=dict)
    refresh_tokens: float = 3.0
    last_refill_at: float = 0.0


class JWKSCache:
    """Thread-safe JWKS cache with per-(jwks_url, kid) cooldown & refresh token bucket."""

    _MAX_NEGATIVE_KIDS: Final[int] = 1024
    _REFRESH_BUCKET_CAPACITY: Final[float] = 3.0

    def __init__(
        self,
        *,
        cache_ttl_seconds: float = 300.0,
        min_refresh_interval_seconds: float = 10.0,
        refresh_bucket_capacity: float = 3.0,
        http_client: httpx.Client | None = None,
        jwks_loader: Callable[[str], Mapping[str, Any]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = max(1.0, float(cache_ttl_seconds))
        self._min_refresh = max(0.0, float(min_refresh_interval_seconds))
        self._bucket_capacity = max(1.0, float(refresh_bucket_capacity))
        self._http_client = http_client
        self._jwks_loader = jwks_loader
        self._clock = clock
        self._lock = threading.RLock()
        self._cache: dict[str, _CachedJWKS] = {}

    def _validate_url(self, jwks_url: str) -> str:
        cleaned = jwks_url.strip()
        if not cleaned:
            raise AuthVerificationError(503, "JWKS URL is not configured.")
        parsed = urlparse(cleaned)
        if parsed.scheme == "https":
            return cleaned
        host = (parsed.hostname or "").lower()
        if parsed.scheme == "http" and (
            host in _LOCAL_TEST_HOSTS
            or self._http_client is not None
            or self._jwks_loader is not None
        ):
            return cleaned
        raise AuthVerificationError(
            503, f"Insecure JWKS URL scheme rejected (https required): {jwks_url!r}"
        )

    def _fetch_jwks(self, jwks_url: str) -> _CachedJWKS:
        validated_url = self._validate_url(jwks_url)
        now = self._clock()
        try:
            if self._jwks_loader is not None:
                payload = self._jwks_loader(validated_url)
            elif self._http_client is not None:
                resp = self._http_client.get(validated_url, timeout=5.0)
                resp.raise_for_status()
                payload = resp.json()
            else:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(validated_url)
                    resp.raise_for_status()
                    payload = resp.json()
        except AuthVerificationError:
            raise
        except Exception as exc:
            raise AuthVerificationError(
                503, f"Failed fetching JWKS from {validated_url}: {type(exc).__name__}"
            ) from exc

        raw_keys = payload.get("keys") if isinstance(payload, Mapping) else None
        if not isinstance(raw_keys, list) or not raw_keys:
            raise AuthVerificationError(503, f"Empty or invalid JWKS payload from {validated_url}.")

        by_kid: dict[str, jwt.PyJWK] = {}
        ordered: list[jwt.PyJWK] = []
        for entry in raw_keys:
            if not isinstance(entry, Mapping):
                continue
            use = entry.get("use")
            if use is not None and str(use).lower() != "sig":
                continue
            try:
                jwk_obj = jwt.PyJWK.from_dict(dict(entry))
            except Exception as exc:
                logger.debug("Skipping malformed JWK entry (%s): %s", type(exc).__name__, exc)
                continue
            ordered.append(jwk_obj)
            kid_val = entry.get("kid")
            if isinstance(kid_val, str) and kid_val.strip():
                by_kid[kid_val.strip()] = jwk_obj

        if not ordered:
            raise AuthVerificationError(
                503, f"No usable signing keys in JWKS from {validated_url}."
            )
        existing = self._cache.get(validated_url)
        neg_cooldowns = dict(existing.negative_kid_cooldowns) if existing is not None else {}
        tokens = existing.refresh_tokens if existing is not None else self._bucket_capacity
        last_refill = existing.last_refill_at if existing is not None else now
        for known_kid in by_kid:
            neg_cooldowns.pop(known_kid, None)
        cached = _CachedJWKS(
            fetched_at=now,
            keys_by_kid=by_kid,
            ordered_keys=ordered,
            negative_kid_cooldowns=neg_cooldowns,
            refresh_tokens=tokens,
            last_refill_at=last_refill,
        )
        self._cache[validated_url] = cached
        return cached

    def _consume_refresh_token(self, entry: _CachedJWKS, now: float) -> bool:
        if self._min_refresh <= 0.0:
            return True
        elapsed = max(0.0, now - entry.last_refill_at)
        refill_rate = 1.0 / self._min_refresh
        entry.refresh_tokens = min(
            self._bucket_capacity,
            entry.refresh_tokens + elapsed * refill_rate,
        )
        entry.last_refill_at = now
        if entry.refresh_tokens >= 1.0:
            entry.refresh_tokens -= 1.0
            return True
        return False

    def _record_negative_kid(self, entry: _CachedJWKS, kid: str, now: float) -> None:
        if kid not in entry.negative_kid_cooldowns and (
            len(entry.negative_kid_cooldowns) >= self._MAX_NEGATIVE_KIDS
        ):
            oldest_kid = next(iter(entry.negative_kid_cooldowns))
            entry.negative_kid_cooldowns.pop(oldest_kid, None)
        entry.negative_kid_cooldowns[kid] = now

    def get_signing_key(
        self,
        jwks_url: str,
        kid: str | None,
        *,
        require_kid: bool = False,
    ) -> jwt.PyJWK:
        """Resolves a verification JWK by kid with per-kid negative cooldown & token bucket."""
        if require_kid and (not kid or not kid.strip()):
            raise AuthVerificationError(401, "JWT header is missing required 'kid'.")
        clean_kid = kid.strip() if isinstance(kid, str) and kid.strip() else None

        with self._lock:
            now = self._clock()
            entry = self._cache.get(jwks_url.strip())
            if entry is None or (now - entry.fetched_at) >= self._ttl:
                entry = self._fetch_jwks(jwks_url)
            elif clean_kid is not None and clean_kid not in entry.keys_by_kid:
                last_miss = entry.negative_kid_cooldowns.get(clean_kid)
                kid_cooled_down = last_miss is not None and (now - last_miss) < self._min_refresh
                if not kid_cooled_down and self._consume_refresh_token(entry, now):
                    entry = self._fetch_jwks(jwks_url)
                    if clean_kid not in entry.keys_by_kid:
                        self._record_negative_kid(entry, clean_kid, now)

            if clean_kid is not None:
                found = entry.keys_by_kid.get(clean_kid)
                if found is None:
                    raise AuthVerificationError(
                        401, f"Signing key kid={clean_kid!r} not found in JWKS."
                    )
                return found

            if len(entry.ordered_keys) == 1:
                return entry.ordered_keys[0]
            raise AuthVerificationError(
                401, "JWT header omits 'kid' and JWKS contains multiple signing keys."
            )

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


def _parse_role_candidate(raw: object) -> RoleType | None:
    if isinstance(raw, RoleType):
        return raw
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip().lower()
    if not cleaned:
        return None
    try:
        return RoleType(cleaned)
    except ValueError:
        return None


def _sanitize_group_slug(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\./:]+", "_", raw.strip().lstrip("/")).strip("_")
    return sanitize_scope_identifier(cleaned or "group")


def _extract_email_domain(email: str | None) -> str:
    if not email or "@" not in email:
        return ""
    return email.rsplit("@", 1)[-1].strip().lower()


def _resolve_tenant_uuid(
    claims: Mapping[str, Any],
    *,
    default_tenant_id: UUID = DEFAULT_TENANT_ID,
    override_tenant_id: UUID | None = None,
) -> UUID:
    for key in ("scout_tenant_id", "tenant_id"):
        raw = claims.get(key)
        if raw:
            try:
                claim_tid = UUID(str(raw).strip())
            except ValueError as exc:
                raise AuthVerificationError(401, f"Invalid UUID in token claim '{key}'.") from exc
            if override_tenant_id is not None and override_tenant_id != claim_tid:
                raise AuthVerificationError(
                    403,
                    f"Tenant override {override_tenant_id} conflicts with signed token claim "
                    f"'{key}'={claim_tid} (ASI03).",
                )
            return claim_tid
    if override_tenant_id is not None:
        return override_tenant_id
    return default_tenant_id


def _build_principal_from_claims(
    *,
    user_id: str,
    email: str,
    tenant_id: UUID,
    idp_groups: Sequence[str],
    idp_roles: Sequence[str],
    direct_scopes: Sequence[str],
    scim_service: SCIMDirectoryService,
    group_scope_map: Mapping[str, Sequence[str]],
    role_map: Mapping[str, str],
    betrvg_guard: BetrVGPrivacyGuard,
    default_role: RoleType = RoleType.DOMAIN_RESEARCHER,
) -> PrincipalContext:
    identifiers = [x for x in (user_id, email) if x and x.strip()]
    scim_res = scim_service.resolve_principal_enrichment(
        tenant_id=tenant_id,
        identifiers=identifiers,
        idp_group_refs=idp_groups,
    )
    if scim_res.deprovisioned:
        raise AuthVerificationError(
            403, f"Principal '{user_id}' has been deprovisioned via SCIM directory."
        )

    best_role: RoleType | None = scim_res.role
    if best_role is None:
        mapped_candidates: list[RoleType] = []
        for token in [*idp_roles, *idp_groups]:
            if token in role_map:
                parsed = _parse_role_candidate(role_map[token])
                if parsed is not None:
                    mapped_candidates.append(parsed)
            direct_parsed = _parse_role_candidate(token)
            if direct_parsed is not None:
                mapped_candidates.append(direct_parsed)
        for cand in mapped_candidates:
            if best_role is None or _ROLE_PRECEDENCE.get(cand, 0) > _ROLE_PRECEDENCE.get(
                best_role, 0
            ):
                best_role = cand

    resolved_role = best_role or default_role

    scopes: list[str] = []
    for s in scim_res.scopes:
        if s not in scopes:
            scopes.append(s)

    all_groups = list(dict.fromkeys([*idp_groups, *scim_res.group_names]))
    for grp in all_groups:
        norm_grp = grp.strip().lstrip("/")
        if not norm_grp:
            continue
        grp_scope = f"group:{_sanitize_group_slug(norm_grp)}"
        if grp_scope not in scopes:
            scopes.append(grp_scope)
        for mapped in group_scope_map.get(norm_grp, ()):
            clean_mapped = sanitize_scope_identifier(mapped)
            if clean_mapped not in scopes:
                scopes.append(clean_mapped)
        for mapped in group_scope_map.get(grp, ()):
            clean_mapped = sanitize_scope_identifier(mapped)
            if clean_mapped not in scopes:
                scopes.append(clean_mapped)

    for ds in direct_scopes:
        if ds and ds.strip():
            clean_ds = sanitize_scope_identifier(ds)
            if clean_ds not in scopes:
                scopes.append(clean_ds)

    for s in [*scim_res.scopes, *scopes]:
        if s.lower().startswith("system:"):
            raise AuthVerificationError(
                403,
                f"Reserved internal scope {s!r} is forbidden on external identities (ASI03).",
            )

    if not scopes:
        scopes = [f"unprivileged:user:{_sanitize_group_slug(user_id)}"]

    principal = PrincipalContext(
        user_id=user_id,
        role=resolved_role,
        allowed_scopes=scopes,
        tenant_id=tenant_id,
        max_session_paid_tokens=50_000,
    )
    return betrvg_guard.apply_to_principal(principal)


class CloudIAPVerifier:
    """Verifies Google Cloud Identity-Aware Proxy (x-goog-iap-jwt-assertion) ES256 JWTs."""

    def __init__(
        self,
        *,
        expected_audience: str | None,
        issuer: str = CLOUD_IAP_ISSUER,
        jwks_url: str = CLOUD_IAP_JWKS_URL,
        allowed_hosted_domains: Sequence[str] = (),
        clock_skew_seconds: int = 30,
        jwks_cache: JWKSCache | None = None,
        scim_service: SCIMDirectoryService | None = None,
        group_scope_map: Mapping[str, Sequence[str]] | None = None,
        role_map: Mapping[str, str] | None = None,
        betrvg_guard: BetrVGPrivacyGuard | None = None,
    ) -> None:
        self.expected_audience = (expected_audience or "").strip()
        self.issuer = issuer.strip() or CLOUD_IAP_ISSUER
        self.jwks_url = jwks_url.strip() or CLOUD_IAP_JWKS_URL
        self.allowed_hosted_domains = {
            d.strip().lower() for d in allowed_hosted_domains if d.strip()
        }
        self.clock_skew_seconds = max(0, int(clock_skew_seconds))
        self.jwks_cache = jwks_cache or JWKSCache()
        self.scim_service = scim_service or get_scim_directory_service()
        self.group_scope_map = dict(group_scope_map or {})
        self.role_map = dict(role_map or {})
        self.betrvg_guard = betrvg_guard or BetrVGPrivacyGuard()

    def verify(
        self,
        assertion: str,
        *,
        tenant_id: UUID | None = None,
        now_ts: float | None = None,
    ) -> PrincipalContext:
        """Verifies ES256 Cloud IAP JWT assertion and returns PrincipalContext."""
        if not self.expected_audience:
            raise AuthVerificationError(
                503,
                "Google Cloud IAP expected_audience is not configured "
                "(SCOUT_IAP_EXPECTED_AUDIENCE).",
            )
        token = assertion.strip()
        if not token or token.count(".") != 2:
            raise AuthVerificationError(401, "Malformed x-goog-iap-jwt-assertion JWT.")

        try:
            header = jwt.get_unverified_header(token)
        except InvalidTokenError as exc:
            raise AuthVerificationError(401, f"Invalid Cloud IAP JWT header: {exc}") from exc

        alg = str(header.get("alg") or "")
        if alg not in ALLOWED_IAP_ALGORITHMS:
            raise AuthVerificationError(
                401, f"Cloud IAP JWT must use ES256 algorithm; rejected alg={alg!r}."
            )

        kid = header.get("kid")
        jwk_obj = self.jwks_cache.get_signing_key(
            self.jwks_url,
            str(kid) if kid is not None else None,
            require_kid=True,
        )
        key_type = str(getattr(jwk_obj, "key_type", "") or "").upper()
        if key_type and key_type != "EC":
            raise AuthVerificationError(401, f"Cloud IAP requires EC JWKS key, got {key_type!r}.")

        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                key=jwk_obj.key,
                algorithms=["ES256"],
                audience=self.expected_audience,
                issuer=self.issuer,
                leeway=self.clock_skew_seconds,
                options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            )
        except InvalidTokenError as exc:
            raise AuthVerificationError(401, f"Cloud IAP JWT verification failed: {exc}") from exc

        current_time = now_ts if now_ts is not None else time.time()
        iat_val = float(claims["iat"])
        if iat_val > current_time + self.clock_skew_seconds:
            raise AuthVerificationError(401, "Cloud IAP JWT 'iat' is in the future.")

        sub = str(claims.get("sub") or "").strip()
        if not sub:
            raise AuthVerificationError(401, "Cloud IAP JWT missing 'sub' claim.")
        email = str(claims.get("email") or "").strip()

        if self.allowed_hosted_domains:
            hd = str(claims.get("hd") or "").strip().lower() or _extract_email_domain(email)
            if not hd or hd not in self.allowed_hosted_domains:
                raise AuthVerificationError(
                    403,
                    f"Cloud IAP hosted domain '{hd or 'unknown'}' is not authorized.",
                )

        raw_gcip = claims.get("gcip")
        gcip_claims: Mapping[str, Any] = raw_gcip if isinstance(raw_gcip, Mapping) else {}
        raw_groups = claims.get("groups") or gcip_claims.get("groups") or []
        idp_groups = (
            [str(g).strip() for g in raw_groups if str(g).strip()]
            if isinstance(raw_groups, list)
            else []
        )
        raw_roles = claims.get("roles") or []
        idp_roles = (
            [str(r).strip() for r in raw_roles if str(r).strip()]
            if isinstance(raw_roles, list)
            else []
        )
        if isinstance(claims.get("role"), str):
            idp_roles.append(str(claims["role"]))

        resolved_tid = _resolve_tenant_uuid(claims, override_tenant_id=tenant_id)
        user_identifier = email or sub

        return _build_principal_from_claims(
            user_id=user_identifier,
            email=email,
            tenant_id=resolved_tid,
            idp_groups=idp_groups,
            idp_roles=idp_roles,
            direct_scopes=[],
            scim_service=self.scim_service,
            group_scope_map=self.group_scope_map,
            role_map=self.role_map,
            betrvg_guard=self.betrvg_guard,
        )


def resolve_provider_jwks_url(
    provider: OIDCProviderKind,
    issuer: str | None,
    explicit_jwks_url: str | None = None,
) -> str:
    """Derives provider JWKS endpoint URL from OIDC provider preset and issuer."""
    if explicit_jwks_url and explicit_jwks_url.strip():
        return explicit_jwks_url.strip()
    if provider == "google_workspace":
        return GOOGLE_WORKSPACE_JWKS_URL
    clean_iss = (issuer or "").strip().rstrip("/")
    if not clean_iss:
        return ""
    if provider == "keycloak":
        return f"{clean_iss}/protocol/openid-connect/certs"
    if provider == "entra_id":
        base = clean_iss[:-5] if clean_iss.endswith("/v2.0") else clean_iss
        return f"{base}/discovery/v2.0/keys"
    if provider == "okta":
        return f"{clean_iss}/v1/keys"
    return f"{clean_iss}/.well-known/jwks.json"


class OIDCJWKSVerifier:
    """Verifies OIDC Bearer JWTs across Google Workspace, Keycloak, Entra ID, and Okta."""

    def __init__(
        self,
        *,
        provider: OIDCProviderKind = "generic",
        issuer: str | None = None,
        audience: str | None = None,
        jwks_url: str | None = None,
        allowed_hosted_domains: Sequence[str] = (),
        allowed_entra_tenants: Sequence[str] = (),
        clock_skew_seconds: int = 30,
        jwks_cache: JWKSCache | None = None,
        scim_service: SCIMDirectoryService | None = None,
        group_scope_map: Mapping[str, Sequence[str]] | None = None,
        role_map: Mapping[str, str] | None = None,
        betrvg_guard: BetrVGPrivacyGuard | None = None,
    ) -> None:
        self.provider: OIDCProviderKind = provider
        default_iss = GOOGLE_WORKSPACE_ISSUER if provider == "google_workspace" else ""
        self.issuer = (issuer or default_iss).strip()
        self.audience = (audience or "").strip()
        self.jwks_url = resolve_provider_jwks_url(provider, self.issuer, jwks_url)
        self.allowed_hosted_domains = {
            d.strip().lower() for d in allowed_hosted_domains if d.strip()
        }
        self.allowed_entra_tenants = {t.strip().lower() for t in allowed_entra_tenants if t.strip()}
        self.clock_skew_seconds = max(0, int(clock_skew_seconds))
        self.jwks_cache = jwks_cache or JWKSCache()
        self.scim_service = scim_service or get_scim_directory_service()
        self.group_scope_map = dict(group_scope_map or {})
        self.role_map = dict(role_map or {})
        self.betrvg_guard = betrvg_guard or BetrVGPrivacyGuard()

    def _extract_provider_claims(
        self, claims: Mapping[str, Any]
    ) -> tuple[str, str, list[str], list[str], list[str]]:
        sub = str(claims.get("sub") or "").strip()
        email = str(
            claims.get("email") or claims.get("preferred_username") or claims.get("upn") or ""
        ).strip()
        groups: list[str] = []
        roles: list[str] = []
        direct_scopes: list[str] = []

        raw_groups = claims.get("groups")
        if isinstance(raw_groups, list):
            groups.extend(str(g).strip().lstrip("/") for g in raw_groups if str(g).strip())

        raw_roles = claims.get("roles")
        if isinstance(raw_roles, list):
            roles.extend(str(r).strip() for r in raw_roles if str(r).strip())
        if isinstance(claims.get("role"), str):
            roles.append(str(claims["role"]).strip())

        if self.provider == "google_workspace":
            if email and claims.get("email_verified") is not True:
                raise AuthVerificationError(
                    401, "Google Workspace OIDC token email is not verified."
                )
            if self.allowed_hosted_domains:
                hd = str(claims.get("hd") or "").strip().lower() or _extract_email_domain(email)
                if not hd or hd not in self.allowed_hosted_domains:
                    raise AuthVerificationError(
                        403, f"Google Workspace domain '{hd or 'unknown'}' is not authorized."
                    )

        elif self.provider == "keycloak":
            realm_access = claims.get("realm_access")
            if isinstance(realm_access, Mapping) and isinstance(realm_access.get("roles"), list):
                roles.extend(str(r).strip() for r in realm_access["roles"] if str(r).strip())
            res_access = claims.get("resource_access")
            if isinstance(res_access, Mapping):
                for client_id, entry in res_access.items():
                    if client_id == self.audience and isinstance(entry, Mapping):
                        client_roles = entry.get("roles")
                        if isinstance(client_roles, list):
                            roles.extend(str(r).strip() for r in client_roles if str(r).strip())

        elif self.provider == "entra_id":
            tid_claim = str(claims.get("tid") or "").strip().lower()
            if self.allowed_entra_tenants and tid_claim not in self.allowed_entra_tenants:
                raise AuthVerificationError(
                    403, f"Entra ID tenant '{tid_claim or 'unknown'}' is not authorized."
                )
            # Detect Entra ID group overage (_claim_names.groups / hasgroups) and rely on SCIM sync
            claim_names = claims.get("_claim_names")
            if (isinstance(claim_names, Mapping) and "groups" in claim_names) or claims.get(
                "hasgroups"
            ) is True:
                logger.info(
                    "Entra ID group overage detected for sub=%s; resolving groups via SCIM.",
                    sub,
                )

        elif self.provider == "okta":
            if isinstance(claims.get("uid"), str) and not email:
                email = str(claims["uid"]).strip()

        raw_scp = claims.get("scp")
        if isinstance(raw_scp, str):
            for tok in raw_scp.split():
                if tok.strip() and tok.strip() not in {
                    "openid",
                    "profile",
                    "email",
                    "offline_access",
                }:
                    direct_scopes.append(tok.strip())
        elif isinstance(raw_scp, list):
            for tok in raw_scp:
                st = str(tok).strip()
                if st and st not in {"openid", "profile", "email", "offline_access"}:
                    direct_scopes.append(st)

        user_id = email or str(claims.get("preferred_username") or sub).strip()
        return (
            user_id,
            email,
            list(dict.fromkeys(groups)),
            list(dict.fromkeys(roles)),
            direct_scopes,
        )

    def verify(
        self,
        token: str,
        *,
        tenant_id: UUID | None = None,
        now_ts: float | None = None,
    ) -> PrincipalContext:
        """Verifies asymmetric OIDC JWT via JWKS and resolves PrincipalContext + SCIM groups."""
        if not self.issuer or not self.audience or not self.jwks_url:
            raise AuthVerificationError(
                503,
                "OIDC JWKS verification is not configured "
                "(issuer, audience, and jwks_url required).",
            )
        raw_jwt = token.strip()
        if not raw_jwt or raw_jwt.count(".") != 2:
            raise AuthVerificationError(401, "Malformed OIDC Bearer JWT.")

        try:
            header = jwt.get_unverified_header(raw_jwt)
        except InvalidTokenError as exc:
            raise AuthVerificationError(401, f"Invalid OIDC JWT header: {exc}") from exc

        alg = str(header.get("alg") or "")
        if alg not in ALLOWED_OIDC_ALGORITHMS:
            raise AuthVerificationError(
                401,
                f"Rejected unsafe or unsupported OIDC JWT algorithm {alg!r} "
                f"(allowed: {sorted(ALLOWED_OIDC_ALGORITHMS)}).",
            )

        kid = header.get("kid")
        jwk_obj = self.jwks_cache.get_signing_key(
            self.jwks_url,
            str(kid) if kid is not None else None,
            require_kid=False,
        )

        try:
            claims: dict[str, Any] = jwt.decode(
                raw_jwt,
                key=jwk_obj.key,
                algorithms=sorted(ALLOWED_OIDC_ALGORITHMS),
                audience=self.audience,
                issuer=self.issuer,
                leeway=self.clock_skew_seconds,
                options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            )
        except InvalidTokenError as exc:
            raise AuthVerificationError(401, f"OIDC JWT verification failed: {exc}") from exc

        current_time = now_ts if now_ts is not None else time.time()
        if float(claims["iat"]) > current_time + self.clock_skew_seconds:
            raise AuthVerificationError(401, "OIDC JWT 'iat' is in the future.")

        user_id, email, groups, roles, direct_scopes = self._extract_provider_claims(claims)
        resolved_tid = _resolve_tenant_uuid(claims, override_tenant_id=tenant_id)

        return _build_principal_from_claims(
            user_id=user_id,
            email=email,
            tenant_id=resolved_tid,
            idp_groups=groups,
            idp_roles=roles,
            direct_scopes=direct_scopes,
            scim_service=self.scim_service,
            group_scope_map=self.group_scope_map,
            role_map=self.role_map,
            betrvg_guard=self.betrvg_guard,
        )


_GLOBAL_JWKS_CACHE = JWKSCache()


def get_global_jwks_cache() -> JWKSCache:
    """Returns shared process-wide JWKSCache instance."""
    return _GLOBAL_JWKS_CACHE


def build_iap_verifier_from_env(
    *,
    jwks_cache: JWKSCache | None = None,
    scim_service: SCIMDirectoryService | None = None,
) -> CloudIAPVerifier:
    """Constructs CloudIAPVerifier from current SCOUT_IAP_* environment settings."""
    raw_domains = os.environ.get("SCOUT_IAP_ALLOWED_HOSTED_DOMAINS", "")
    domains = [d.strip() for d in raw_domains.split(",") if d.strip()]
    return CloudIAPVerifier(
        expected_audience=os.environ.get("SCOUT_IAP_EXPECTED_AUDIENCE"),
        issuer=os.environ.get("SCOUT_IAP_ISSUER", CLOUD_IAP_ISSUER),
        jwks_url=os.environ.get("SCOUT_IAP_JWKS_URL", CLOUD_IAP_JWKS_URL),
        allowed_hosted_domains=domains,
        clock_skew_seconds=int(os.environ.get("SCOUT_OIDC_CLOCK_SKEW_SECONDS", "30")),
        jwks_cache=jwks_cache or _GLOBAL_JWKS_CACHE,
        scim_service=scim_service or get_scim_directory_service(),
    )


def build_oidc_verifier_from_env(
    *,
    jwks_cache: JWKSCache | None = None,
    scim_service: SCIMDirectoryService | None = None,
) -> OIDCJWKSVerifier | None:
    """Constructs OIDCJWKSVerifier when SCOUT_OIDC_ISSUER or SCOUT_OIDC_PROVIDER is set."""
    raw_provider = (os.environ.get("SCOUT_OIDC_PROVIDER") or "").strip().lower()
    raw_issuer = (os.environ.get("SCOUT_OIDC_ISSUER") or "").strip()
    raw_aud = (os.environ.get("SCOUT_OIDC_AUDIENCE") or "").strip()
    raw_jwks = (os.environ.get("SCOUT_OIDC_JWKS_URL") or "").strip()
    if not (raw_provider or raw_issuer or raw_jwks):
        return None
    provider: OIDCProviderKind = (
        raw_provider  # type: ignore[assignment]
        if raw_provider in {"google_workspace", "keycloak", "entra_id", "okta", "generic"}
        else "generic"
    )
    raw_domains = os.environ.get("SCOUT_OIDC_ALLOWED_HOSTED_DOMAINS", "")
    domains = [d.strip() for d in raw_domains.split(",") if d.strip()]
    raw_tids = os.environ.get("SCOUT_OIDC_ALLOWED_ENTRA_TENANTS", "")
    tids = [t.strip() for t in raw_tids.split(",") if t.strip()]
    return OIDCJWKSVerifier(
        provider=provider,
        issuer=raw_issuer or None,
        audience=raw_aud or None,
        jwks_url=raw_jwks or None,
        allowed_hosted_domains=domains,
        allowed_entra_tenants=tids,
        clock_skew_seconds=int(os.environ.get("SCOUT_OIDC_CLOCK_SKEW_SECONDS", "30")),
        jwks_cache=jwks_cache or _GLOBAL_JWKS_CACHE,
        scim_service=scim_service or get_scim_directory_service(),
    )


__all__ = [
    "ALLOWED_IAP_ALGORITHMS",
    "ALLOWED_OIDC_ALGORITHMS",
    "CLOUD_IAP_ISSUER",
    "CLOUD_IAP_JWKS_URL",
    "GOOGLE_WORKSPACE_ISSUER",
    "GOOGLE_WORKSPACE_JWKS_URL",
    "AuthVerificationError",
    "CloudIAPVerifier",
    "JWKSCache",
    "OIDCJWKSVerifier",
    "OIDCProviderKind",
    "build_iap_verifier_from_env",
    "build_oidc_verifier_from_env",
    "get_global_jwks_cache",
    "resolve_provider_jwks_url",
]
