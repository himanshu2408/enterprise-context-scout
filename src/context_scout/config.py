"""Typed configuration settings, model governance policy, and hardware auto-detection."""

import fnmatch
import functools
import json
import logging
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Final, Literal
from urllib.parse import parse_qs, unquote, urlparse

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

logger = logging.getLogger(__name__)

DEFAULT_MODEL_REGISTRY_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2] / "config" / "model_registry.yaml"
)

DEFAULT_ALLOWED_SPDX_LICENSES: Final[frozenset[str]] = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "BSD-3-Clause",
        "BSD-2-Clause",
        "CC-BY-4.0",
        "ISC",
        "gemma",
        "llama4",
        "qwen",
        "deepseek",
    }
)

DEFAULT_LOCAL_MODEL_PATTERNS: Final[tuple[str, ...]] = (
    "google/gemma-4-*",
    "mlx-community/gemma-4-*",
    "google/embeddinggemma-2*",
    "google/shieldgemma-2*",
    "gemini-embedding-*",
)

DEFAULT_CLOUD_HANDOFF_PATTERNS: Final[tuple[str, ...]] = (
    "gemini-3.5-*",
    "gemini-3.8-*",
    "gemini-4-*",
)

UNSAFE_WEIGHT_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".bin", ".pt", ".pth", ".pkl", ".pickle", ".h5", ".ckpt"}
)

INSECURE_DB_PASSWORDS: Final[frozenset[str]] = frozenset(
    {
        "scout_local_dev_pw",
        "postgres_local_admin_pw",
        "postgres",
        "password",
        "changeme",
        "secret",
    }
)


_DEV_ENV_ALIASES: Final[frozenset[str]] = frozenset(
    {"", "development", "dev", "local", "test", "testing"}
)


def normalize_environment_name(raw: str | None) -> str:
    """Normalizes runtime environment strings across SCOUT_ENVIRONMENT / SCOUT_ENV / ENVIRONMENT."""
    cleaned = (raw or "").strip().lower()
    if cleaned in _DEV_ENV_ALIASES:
        return "development"
    if cleaned in {"prod", "production"}:
        return "production"
    if cleaned in {"staging", "qa"}:
        return cleaned
    return cleaned


def get_normalized_environment() -> str:
    """Returns canonical environment from SCOUT_ENVIRONMENT > SCOUT_ENV > ENVIRONMENT > ENV."""
    raw = (
        os.environ.get("SCOUT_ENVIRONMENT")
        or os.environ.get("SCOUT_ENV")
        or os.environ.get("ENVIRONMENT")
        or os.environ.get("ENV")
        or "development"
    )
    return normalize_environment_name(raw)


def _is_production_env() -> bool:
    return get_normalized_environment() != "development"


def validate_db_app_url(db_app_url: str, *, require_env_in_prod: bool = True) -> None:
    """Fails closed in production if SCOUT_DB_APP_URL is unset, superuser, or weak."""
    if not _is_production_env():
        return
    if require_env_in_prod and not os.environ.get("SCOUT_DB_APP_URL", "").strip():
        raise RuntimeError(
            "SCOUT_DB_APP_URL must be explicitly configured in production environments."
        )
    parsed = urlparse(db_app_url.strip())
    username = unquote(parsed.username or "").strip().lower()
    if username in {"postgres", "supabase_admin", "rdsadmin", "cloudsqlsuperuser"}:
        raise RuntimeError(
            f"SCOUT_DB_APP_URL cannot connect as privileged role '{username}' in production."
        )
    password = unquote(parsed.password or "")
    if password in INSECURE_DB_PASSWORDS:
        raise RuntimeError(
            "Insecure default database password detected in SCOUT_DB_APP_URL in production."
        )
    qs = parse_qs(parsed.query)
    host_param = (qs.get("host") or [""])[0]
    is_unix_socket = host_param.startswith("/") or (parsed.hostname or "").startswith("/")
    uses_iam_or_cert = bool(
        qs.get("iam")
        or qs.get("aws_iam")
        or qs.get("gcp_iam")
        or qs.get("sslcert")
        or qs.get("passfile")
    )
    if not password and not is_unix_socket and not uses_iam_or_cert:
        raise RuntimeError(
            "Passwordless SCOUT_DB_APP_URL is prohibited over TCP in production unless "
            "IAM or client-certificate authentication is configured."
        )


PROHIBITED_STALE_MODEL_REGEX: Final[re.Pattern[str]] = re.compile(
    r"(?:^|/)(?:text-embedding-004|gemini-ultra|gemini-2\.5-|gemma-[123](?:-|$)|codegemma)",
    re.IGNORECASE,
)

ALLOWED_EU_REGIONS: Final[frozenset[str]] = frozenset(
    {
        "europe-west1",  # Belgium
        "europe-west2",  # London (UK)
        "europe-west3",  # Frankfurt (Germany) - Primary Default
        "europe-west4",  # Eemshaven (Netherlands)
        "europe-west6",  # Zurich (Switzerland)
        "europe-west8",  # Milan (Italy)
        "europe-west9",  # Paris (France)
        "europe-west10",  # Berlin (Germany)
        "europe-west12",  # Turin (Italy)
        "europe-north1",  # Hamina (Finland)
        "europe-central2",  # Warsaw (Poland)
        "europe-southwest1",  # Madrid (Spain)
    }
)


def validate_vertex_location(location: str, allow_non_eu: bool = False) -> str:
    """Enforces EU data residency for Vertex AI execution unless explicitly overridden."""
    cleaned = location.strip().lower()
    if not allow_non_eu and not (cleaned.startswith("europe-") or cleaned in ALLOWED_EU_REGIONS):
        raise ValueError(
            f"Vertex AI region '{location}' violates EU data residency policy. "
            "Only EU regions (e.g. 'europe-west3') are permitted to comply with GDPR data "
            "residency requirements, unless allow_non_eu_regions=True "
            "(or SCOUT_ALLOW_NON_EU_REGIONS=true) is explicitly configured."
        )
    return cleaned


def _parse_registry_yaml(registry_path: Path) -> dict[str, Any]:
    """Parses model governance registry YAML into ModelGovernancePolicy kwargs."""
    if not registry_path.is_file():
        raise FileNotFoundError(f"Model registry YAML not found at: {registry_path}")
    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid model registry YAML structure at: {registry_path}")

    gov = raw.get("governance") or {}
    local_defaults = raw.get("local_defaults") or {}
    local_ids: set[str] = set()
    for section in local_defaults.values():
        if isinstance(section, list):
            for entry in section:
                if isinstance(entry, dict) and entry.get("id"):
                    local_ids.add(str(entry["id"]).strip())

    byo_section = raw.get("byo_open_weights") or {}
    byo_patterns: list[str] = []
    for group in byo_section.values():
        if isinstance(group, list):
            for family_entry in group:
                if isinstance(family_entry, dict):
                    for pat in family_entry.get("patterns") or []:
                        pat_str = str(pat).strip()
                        if pat_str:
                            byo_patterns.append(pat_str)

    cloud_section = raw.get("cloud_handoff") or {}
    cloud_ids: set[str] = set()
    for entry in cloud_section.get("models") or []:
        if isinstance(entry, dict) and entry.get("id"):
            cloud_ids.add(str(entry["id"]).strip())

    blocked_section = raw.get("blocked_models") or {}
    blocked_patterns = tuple(
        str(pat).strip() for pat in (blocked_section.get("patterns") or []) if str(pat).strip()
    )

    raw_spdx = gov.get("allowed_spdx_licenses")
    spdx_licenses = (
        frozenset(str(lic).strip() for lic in raw_spdx if str(lic).strip())
        if isinstance(raw_spdx, list) and raw_spdx
        else DEFAULT_ALLOWED_SPDX_LICENSES
    )

    return {
        "allow_external_cloud_egress": bool(gov.get("allow_external_cloud_egress", False)),
        "require_safetensors_only": bool(gov.get("require_safetensors_only", True)),
        "trust_remote_code": bool(gov.get("trust_remote_code", False)),
        "allowed_spdx_licenses": spdx_licenses,
        "local_default_ids": frozenset(local_ids),
        "local_patterns": DEFAULT_LOCAL_MODEL_PATTERNS,
        "byo_patterns": tuple(byo_patterns),
        "cloud_handoff_ids": frozenset(cloud_ids),
        "cloud_handoff_patterns": DEFAULT_CLOUD_HANDOFF_PATTERNS,
        "blocked_patterns": blocked_patterns,
    }


@functools.lru_cache(maxsize=1)
def _default_registry_spec() -> dict[str, Any]:
    return _parse_registry_yaml(DEFAULT_MODEL_REGISTRY_PATH)


class ModelGovernancePolicy(BaseModel):
    """Registry-backed BYO open-weights & supply-chain model governance policy."""

    model_config = ConfigDict(frozen=True)

    allow_external_cloud_egress: bool = Field(
        default_factory=lambda: bool(_default_registry_spec()["allow_external_cloud_egress"])
    )
    require_safetensors_only: bool = Field(
        default_factory=lambda: bool(_default_registry_spec()["require_safetensors_only"])
    )
    trust_remote_code: bool = Field(
        default_factory=lambda: bool(_default_registry_spec()["trust_remote_code"])
    )
    allowed_spdx_licenses: frozenset[str] = Field(
        default_factory=lambda: frozenset(_default_registry_spec()["allowed_spdx_licenses"])
    )
    local_default_ids: frozenset[str] = Field(
        default_factory=lambda: frozenset(_default_registry_spec()["local_default_ids"])
    )
    local_patterns: tuple[str, ...] = Field(
        default_factory=lambda: tuple(_default_registry_spec()["local_patterns"])
    )
    byo_patterns: tuple[str, ...] = Field(
        default_factory=lambda: tuple(_default_registry_spec()["byo_patterns"])
    )
    cloud_handoff_ids: frozenset[str] = Field(
        default_factory=lambda: frozenset(_default_registry_spec()["cloud_handoff_ids"])
    )
    cloud_handoff_patterns: tuple[str, ...] = Field(
        default_factory=lambda: tuple(_default_registry_spec()["cloud_handoff_patterns"])
    )
    blocked_patterns: tuple[str, ...] = Field(
        default_factory=lambda: tuple(_default_registry_spec()["blocked_patterns"])
    )

    @classmethod
    def from_registry_file(
        cls,
        registry_path: Path | str | None = None,
        *,
        allow_external_cloud_egress: bool | None = None,
    ) -> "ModelGovernancePolicy":
        """Loads ModelGovernancePolicy from a YAML registry file."""
        target = (
            Path(registry_path).resolve()
            if registry_path is not None
            else DEFAULT_MODEL_REGISTRY_PATH
        )
        spec = dict(_parse_registry_yaml(target))
        if allow_external_cloud_egress is not None:
            spec["allow_external_cloud_egress"] = allow_external_cloud_egress
        return cls(**spec)

    @classmethod
    @functools.lru_cache(maxsize=1)
    def default(cls) -> "ModelGovernancePolicy":
        """Returns the cached default policy backed by config/model_registry.yaml."""
        return cls.from_registry_file()

    def is_blocked_model(self, model_id: str) -> bool:
        """Returns True if model_id matches any retired or prohibited pattern."""
        cleaned = model_id.strip()
        if not cleaned:
            return False
        if PROHIBITED_STALE_MODEL_REGEX.search(cleaned):
            return True
        cleaned_lower = cleaned.lower()
        basename_lower = cleaned_lower.split("/")[-1]
        return any(
            fnmatch.fnmatchcase(cleaned_lower, pat.lower())
            or fnmatch.fnmatchcase(basename_lower, pat.lower())
            for pat in self.blocked_patterns
        )

    def is_local_or_byo_model(self, model_id: str) -> bool:
        """Returns True if model_id is a local Gemma default or approved BYO open-weights model."""
        cleaned = model_id.strip()
        if not cleaned or self.is_blocked_model(cleaned):
            return False
        cleaned_lower = cleaned.lower()
        if cleaned_lower in {m.lower() for m in self.local_default_ids}:
            return True
        if any(fnmatch.fnmatchcase(cleaned_lower, pat.lower()) for pat in self.local_patterns):
            return True
        return any(fnmatch.fnmatchcase(cleaned_lower, pat.lower()) for pat in self.byo_patterns)

    def is_cloud_handoff_model(self, model_id: str) -> bool:
        """Returns True if model_id is a registered external cloud handoff model."""
        cleaned = model_id.strip()
        if not cleaned or self.is_blocked_model(cleaned):
            return False
        cleaned_lower = cleaned.lower()
        if cleaned_lower in {m.lower() for m in self.cloud_handoff_ids}:
            return True
        return any(
            fnmatch.fnmatchcase(cleaned_lower, pat.lower()) for pat in self.cloud_handoff_patterns
        )

    def validate_model_id(
        self,
        model_id: str,
        *,
        allow_cloud_handoff: bool = True,
        enforce_egress_gate: bool = False,
    ) -> str:
        """Validates model_id against config/model_registry.yaml and governance constraints."""
        cleaned = model_id.strip()
        if not cleaned:
            raise ValueError("Model ID cannot be empty.")
        if self.is_blocked_model(cleaned):
            raise ValueError(f"Prohibited or retired model ID '{model_id}'.")
        if self.is_local_or_byo_model(cleaned):
            return cleaned
        if self.is_cloud_handoff_model(cleaned):
            if not allow_cloud_handoff:
                raise ValueError(
                    f"Model '{model_id}' is an external cloud handoff model and is not "
                    "permitted for local self-hosted execution under ModelGovernancePolicy."
                )
            if enforce_egress_gate and not self.allow_external_cloud_egress:
                raise ValueError(
                    f"Model '{model_id}' requires external cloud egress, which is blocked "
                    "by ModelGovernancePolicy (allow_external_cloud_egress=False)."
                )
            return cleaned
        raise ValueError(
            f"Model '{model_id}' violates ModelGovernancePolicy and is not approved "
            "in config/model_registry.yaml."
        )

    def validate_local_model_id(self, model_id: str) -> str:
        """Validates that model_id is an approved local default or BYO open-weights model."""
        return self.validate_model_id(model_id, allow_cloud_handoff=False)

    def validate_cloud_egress(self, model_id: str) -> str:
        """Validates model_id for outbound cloud egress (fails closed unless egress is enabled)."""
        return self.validate_model_id(
            model_id,
            allow_cloud_handoff=True,
            enforce_egress_gate=True,
        )

    def validate_spdx_license(self, spdx_id: str) -> str:
        """Validates that a model license identifier is on the approved SPDX allowlist."""
        cleaned = spdx_id.strip()
        if not cleaned:
            raise ValueError("Model SPDX license identifier is required by ModelGovernancePolicy.")
        allowed_lower = {lic.lower(): lic for lic in self.allowed_spdx_licenses}
        if cleaned.lower() not in allowed_lower:
            raise ValueError(
                f"Model license '{spdx_id}' is not in ModelGovernancePolicy "
                f"allowed_spdx_licenses: {sorted(self.allowed_spdx_licenses)}"
            )
        return cleaned

    def validate_weights_safety(
        self,
        weight_files: Sequence[str | Path],
        *,
        trust_remote_code: bool = False,
        config_payload: Mapping[str, Any] | None = None,
    ) -> list[str]:
        """Enforces .safetensors-only weight loading and trust_remote_code=False."""
        if not self.trust_remote_code:
            has_auto_map = bool(config_payload and "auto_map" in config_payload)
            cfg_trust = bool(config_payload and config_payload.get("trust_remote_code") is True)
            if trust_remote_code or has_auto_map or cfg_trust:
                raise ValueError(
                    "ModelGovernancePolicy prohibits trust_remote_code=True or "
                    "auto_map remote code execution."
                )

        file_names = [Path(wf).name for wf in weight_files]
        unsafe = [
            name for name in file_names if Path(name).suffix.lower() in UNSAFE_WEIGHT_EXTENSIONS
        ]
        if self.require_safetensors_only and unsafe:
            raise ValueError(
                "ModelGovernancePolicy requires .safetensors weights only; "
                f"unsafe binary/pickle files rejected: {unsafe}"
            )

        safetensors = [name for name in file_names if name.lower().endswith(".safetensors")]
        if self.require_safetensors_only and not safetensors:
            raise ValueError(
                "ModelGovernancePolicy requires at least one .safetensors weight shard."
            )
        return safetensors

    def format_chat_prompt(
        self,
        messages: Sequence[Mapping[str, str]],
        tokenizer: Any | None = None,
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        """Formats chat messages via tokenizer.apply_chat_template when available."""
        msg_list = [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in messages
        ]
        apply_fn = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_fn):
            try:
                rendered = apply_fn(
                    msg_list,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
                if isinstance(rendered, str) and rendered:
                    return rendered
            except Exception as exc:
                logger.debug(
                    "tokenizer.apply_chat_template failed (%s): %s; using neutral fallback",
                    type(exc).__name__,
                    exc,
                )
        parts = [f"<|{m['role']}|>\n{m['content']}\n" for m in msg_list]
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "".join(parts)


def validate_model_id(
    model_id: str,
    *,
    allow_cloud_handoff: bool = True,
    enforce_egress_gate: bool = False,
) -> str:
    """Validates model_id using the default registry-backed ModelGovernancePolicy."""
    return ModelGovernancePolicy.default().validate_model_id(
        model_id,
        allow_cloud_handoff=allow_cloud_handoff,
        enforce_egress_gate=enforce_egress_gate,
    )


class ClaimVerificationPolicy(StrEnum):
    """Configurable policy governing claim verification in compiled gates (M4 Requirement 6)."""

    STRICT_ABSTAIN_ON_ANY_UNVERIFIED = "strict_abstain"
    PRUNE_AND_MARK_LOW_TRUST = "prune_and_warn"
    MIN_VERIFIED_COUNT = "min_verified_count"


class ClaimVerifierConfig(BaseModel):
    """Configuration for ClaimVerifier policy and thresholds."""

    policy: ClaimVerificationPolicy = ClaimVerificationPolicy.STRICT_ABSTAIN_ON_ANY_UNVERIFIED
    min_verified_claims: int = Field(default=1, ge=1)
    record_pruned_audit_trail: bool = True
    reject_homoglyphs: bool = True


class DomainConfig(BaseModel):
    """Multi-tenant domain configuration with neutral defaults and zero hardcoded workflows."""

    domain_id: str = "default"
    min_retrieval_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    cross_encoder_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    jira_terminal_statuses: list[str] = Field(
        default_factory=lambda: ["Closed", "Resolved", "Archived"]
    )
    jira_negative_resolutions: list[str] = Field(
        default_factory=lambda: [
            "Won't Fix",
            "Duplicate",
            "Cannot Reproduce",
            "Abandoned",
            "Won't Do",
            "Rejected",
            "Cancelled",
        ]
    )
    confluence_strict_freshness: bool = True
    claim_policy: ClaimVerificationPolicy = ClaimVerificationPolicy.STRICT_ABSTAIN_ON_ANY_UNVERIFIED
    claim_config: ClaimVerifierConfig = Field(default_factory=ClaimVerifierConfig)


class ScoutConfig(BaseModel):
    """Global scout configuration integrating multi-tenant domain overrides."""

    default_domain_config: DomainConfig = Field(default_factory=DomainConfig)
    domain_configs: dict[str, DomainConfig] = Field(default_factory=dict)

    def get_domain_config(self, domain_id: str) -> DomainConfig:
        return self.domain_configs.get(domain_id, self.default_domain_config)


class ScoutSettings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_prefix="SCOUT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_admin_url: str = Field(default="postgresql://postgres@127.0.0.1:5432/context_scout")
    db_app_url: str = Field(default="postgresql://scout_app_user@127.0.0.1:5432/context_scout")

    query_pool_min_size: int = Field(default=1, ge=1, le=64)
    query_pool_max_size: int = Field(default=16, ge=1, le=128)
    query_statement_timeout_ms: int = Field(default=5_000, ge=100, le=60_000)
    query_lock_timeout_ms: int = Field(default=2_000, ge=100, le=30_000)
    query_idle_tx_timeout_ms: int = Field(default=10_000, ge=500, le=120_000)

    ingestion_pool_min_size: int = Field(default=1, ge=1, le=32)
    ingestion_pool_max_size: int = Field(default=8, ge=1, le=64)
    ingestion_statement_timeout_ms: int = Field(default=300_000, ge=1_000, le=3_600_000)
    ingestion_lock_timeout_ms: int = Field(default=10_000, ge=100, le=120_000)
    ingestion_idle_tx_timeout_ms: int = Field(default=300_000, ge=1_000, le=3_600_000)

    s3_endpoint_url: str | None = Field(default=None)
    s3_bucket: str | None = Field(default=None)
    s3_region: str = Field(default="eu-central-1")
    s3_access_key_id: str | None = Field(default=None)
    s3_secret_access_key: str | None = Field(default=None)
    s3_force_path_style: bool = Field(default=True)

    gcs_bucket: str | None = Field(default=None)
    gcs_project: str | None = Field(default=None)
    gcs_emulator_host: str | None = Field(default=None)

    primary_model_id: str = Field(default="google/gemma-4-26B-A4B-it")
    contextualizer_model_id: str = Field(default="google/gemma-4-E4B-it")
    teacher_model_id: str = Field(default="google/gemma-4-31B-it")
    embedding_model_id: str = Field(default="google/embeddinggemma-2")
    embedding_provider: str = Field(default="sentence-transformers")
    embedding_endpoint_url: str | None = Field(default=None)
    embedding_api_key: str | None = Field(default=None)
    reranker_model_id: str = Field(default="BAAI/bge-reranker-v2-m3")
    reranker_provider: Literal["local", "vllm", "tei", "sentence-transformers"] = Field(
        default="local"
    )
    reranker_endpoint_url: str | None = Field(default=None)
    reranker_api_key: str | None = Field(default=None)
    reranker_api_mode: Literal["auto", "rerank", "score"] = Field(default="auto")
    reranker_timeout_seconds: float = Field(default=10.0, ge=0.5, le=120.0)
    shield_model_id: str = Field(default="google/shieldgemma-2-2b-it")
    guardrail_provider: Literal["local", "vllm", "transformers", "model_armor"] = Field(
        default="local"
    )
    guardrail_endpoint_url: str | None = Field(default=None)
    guardrail_timeout_ms: float = Field(default=25.0, ge=1.0, le=500.0)
    enable_gcp_model_armor: bool = Field(default=False)
    model_armor_template_id: str | None = Field(default=None)
    model_armor_vpc_psc_endpoint: str | None = Field(default=None)
    frontier_model_id: str = Field(default="gemini-3.8-flash")
    allow_external_cloud_egress: bool = Field(
        default=False,
        description="Explicitly permits outbound cloud handoff calls when set to True.",
    )

    min_retrieval_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    embedding_dimensions: int = Field(default=768, ge=128, le=4096)
    local_max_kv_tokens: int = Field(default=32_768, ge=4_096, le=131_072)
    local_mlx_endpoint: str = Field(default="http://127.0.0.1:8000/v1")
    zoekt_endpoint_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "zoekt_endpoint_url",
            "zoekt_endpoint",
            "SCOUT_ZOEKT_ENDPOINT_URL",
            "SCOUT_ZOEKT_ENDPOINT",
        ),
        description="Optional Zoekt webserver HTTP/Connect gRPC-JSON endpoint URL.",
    )
    zoekt_protocol: Literal["grpc_json", "json_api"] = Field(
        default="grpc_json",
        description="Zoekt RPC wire protocol ('grpc_json' Connect or 'json_api').",
    )
    zoekt_timeout_seconds: float = Field(default=10.0, ge=0.5, le=120.0)
    mcp_allowed_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "127.0.0.1",
            "127.0.0.1:*",
            "localhost",
            "localhost:*",
            "[::1]",
            "[::1]:*",
            "testserver",
            "testserver:*",
            "context-scout-api",
            "context-scout-api:*",
            "*.svc.cluster.local",
            "*.svc.cluster.local:*",
        ],
        description="Allowed Host header values/patterns for stateless HTTP MCP endpoints.",
    )
    mcp_allowed_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "http://127.0.0.1",
            "http://127.0.0.1:*",
            "http://localhost",
            "http://localhost:*",
            "http://[::1]",
            "http://[::1]:*",
            "http://testserver",
            "http://context-scout-api",
            "http://context-scout-api:*",
        ],
        description="Allowed Origin header values/patterns for stateless HTTP MCP endpoints.",
    )
    mcp_expected_resource: str | None = Field(
        default=None,
        description="Optional RFC 8707 expected resource/audience identifier for HTTP MCP tokens.",
    )
    mcp_public_tool_schemas: bool = Field(
        default=False,
        description="When True, /.well-known/mcp includes full tool inputSchema unauthenticated.",
    )

    @property
    def zoekt_endpoint(self) -> str | None:
        return self.zoekt_endpoint_url

    @field_validator("mcp_allowed_hosts", "mcp_allowed_origins", mode="before")
    @classmethod
    def _parse_mcp_csv_or_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            return [part.strip() for part in stripped.split(",") if part.strip()]
        if isinstance(v, Sequence):
            return [str(item).strip() for item in v if str(item).strip()]
        return [str(v).strip()] if str(v).strip() else []

    docling_endpoint_url: str | None = Field(
        default=None,
        description="Optional self-hosted IBM docling-serve HTTP endpoint URL.",
    )
    docling_api_key: str | None = Field(
        default=None,
        description="Optional API key for self-hosted IBM docling-serve.",
    )
    docling_timeout_seconds: float = Field(default=60.0, ge=1.0, le=600.0)
    redis_url: str | None = Field(
        default=None,
        description=(
            "Optional Redis / Memorystore URL for ACL cache and 429 rate-limit coordination."
        ),
    )
    hardware_device: str | None = Field(default=None)
    hardware_memory_gb: float | None = Field(default=None, ge=4.0, le=1024.0)
    hardware_quantization: str | None = Field(default=None)
    lora_slots_capacity: int = Field(default=4, ge=1, le=32)
    gpu_max_global_inflight: int = Field(default=16, ge=1, le=512)
    gpu_max_tenant_tokens_per_min: int = Field(default=500_000, ge=1_000)
    gpu_max_team_tokens_per_min: int = Field(default=250_000, ge=1_000)
    gpu_max_user_tokens_per_min: int = Field(default=100_000, ge=1_000)
    domains_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "domains"
    )
    default_domain_id: str = Field(
        default="default",
        description="Neutral default domain identifier for multi-tenant context scout operations.",
    )

    auth_hmac_secret: str | None = Field(
        default=None,
        description=(
            "HMAC-SHA256 secret key for signing principal attestation tokens "
            "(minimum 32 characters)."
        ),
    )
    allow_insecure_dev_secret: bool = Field(
        default=False,
        description="Explicitly allow insecure default HMAC secret for local development.",
    )
    environment: str = Field(
        default="development",
        validation_alias=AliasChoices("SCOUT_ENVIRONMENT", "SCOUT_ENV", "ENVIRONMENT"),
        description=(
            "Deployment runtime environment ('development', 'staging', 'qa', 'production')."
        ),
    )

    @property
    def normalized_environment(self) -> str:
        """Canonical normalized environment name ('development', 'staging', 'qa', 'production')."""
        return normalize_environment_name(self.environment)

    # Google Cloud IAP & Multi-Provider OIDC / SCIM Zero-Trust Auth (M5-01)
    iap_enabled: bool = Field(
        default=False,
        description="Enables Google Cloud IAP JWT verification on x-goog-iap-jwt-assertion.",
    )
    iap_expected_audience: str | None = Field(
        default=None,
        description=(
            "Expected Cloud IAP JWT audience (/projects/<NUM>/global/backendServices/<ID>)."
        ),
    )
    iap_issuer: str = Field(
        default="https://cloud.google.com/iap",
        description="Expected Cloud IAP JWT issuer.",
    )
    iap_jwks_url: str = Field(
        default="https://www.gstatic.com/iap/verify/public_key-jwk",
        description="Google Cloud IAP ES256 JWKS endpoint URL.",
    )
    iap_allowed_hosted_domains: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="Optional Workspace/Cloud Identity hosted domains allowlist for Cloud IAP.",
    )

    oidc_provider: Literal["google_workspace", "keycloak", "entra_id", "okta", "generic"] | None = (
        Field(
            default=None,
            description="Enterprise OIDC IdP preset for Bearer JWT validation.",
        )
    )
    oidc_issuer: str | None = Field(default=None)
    oidc_audience: str | None = Field(default=None)
    oidc_jwks_url: str | None = Field(default=None)
    oidc_allowed_hosted_domains: Annotated[list[str], NoDecode] = Field(default_factory=list)
    oidc_allowed_entra_tenants: Annotated[list[str], NoDecode] = Field(default_factory=list)
    oidc_jwks_cache_ttl_seconds: int = Field(default=300, ge=10, le=86400)
    oidc_clock_skew_seconds: int = Field(default=30, ge=0, le=300)
    oidc_group_scope_map: dict[str, list[str]] = Field(default_factory=dict)
    oidc_role_map: dict[str, str] = Field(default_factory=dict)

    scim_bearer_token: str | None = Field(
        default=None,
        description="Fail-closed Bearer token authorizing SCIM 2.0 provisioning endpoints.",
    )

    # Hash-Chained Audit Export, Cloud Logging & BetrVG Privacy (M5-01)
    audit_gcp_cloud_logging_enabled: bool = Field(default=True)
    siem_webhook_url: str | None = Field(default=None)
    siem_webhook_secret: str | None = Field(default=None)
    siem_ndjson_path: Path | None = Field(default=None)

    betrvg_privacy_mode: bool = Field(
        default=False,
        description="Enables EU Works-Council (BetrVG) HMAC-SHA256 user-ID pseudonymization.",
    )
    betrvg_hmac_salt: str | None = Field(
        default=None,
        description="Fail-closed >=32-char HMAC salt for BetrVG user-ID pseudonymization.",
    )
    betrvg_key_epoch: str = Field(default="v1")

    # Webhooks & Prometheus / GMP Retrieval-Drift Observability (M5-01)
    allow_unsigned_dev_webhooks: bool = Field(default=False)
    metrics_bearer_token: str | None = Field(default=None)
    drift_ewma_alpha: float = Field(default=0.2, gt=0.0, le=1.0)
    drift_alert_threshold: float = Field(default=0.15, gt=0.0, le=1.0)

    # GCP Vertex AI Data Residency & Secret Hygiene Settings
    gcp_project: str | None = Field(
        default=None,
        description="Google Cloud Project ID for Vertex AI execution.",
    )
    gcp_location: str = Field(
        default="europe-west3",
        description="Google Cloud Region for Vertex AI execution (default: europe-west3).",
    )
    use_vertex_ai: bool = Field(
        default=True,
        description="Enforces Google Cloud Vertex AI mode instead of Developer API.",
    )
    service_token: str | None = Field(
        default=None,
        description="Read-only service bearer token for Vertex AI authentication.",
    )
    allow_non_eu_regions: bool = Field(
        default=False,
        description="Allows non-EU Vertex AI regions if explicitly enabled.",
    )

    @field_validator(
        "iap_allowed_hosted_domains",
        "oidc_allowed_hosted_domains",
        "oidc_allowed_entra_tenants",
        mode="before",
    )
    @classmethod
    def _parse_csv_or_json_str_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw.startswith("["):
                parsed = json.loads(raw)
                return [str(x).strip() for x in parsed if str(x).strip()]
            return [part.strip() for part in raw.split(",") if part.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(x).strip() for x in value if str(x).strip()]
        return [str(value).strip()]

    @field_validator(
        "primary_model_id",
        "contextualizer_model_id",
        "embedding_model_id",
        "reranker_model_id",
        "shield_model_id",
    )
    @classmethod
    def _enforce_local_models(cls, value: str) -> str:
        return ModelGovernancePolicy.default().validate_local_model_id(value)

    @field_validator("teacher_model_id", "frontier_model_id")
    @classmethod
    def _enforce_governed_models(cls, value: str) -> str:
        return ModelGovernancePolicy.default().validate_model_id(value)

    @model_validator(mode="after")
    def _enforce_residency_rules(self) -> "ScoutSettings":
        validate_db_app_url(self.db_app_url)
        if self.use_vertex_ai and self.gcp_location:
            self.gcp_location = validate_vertex_location(
                self.gcp_location, allow_non_eu=self.allow_non_eu_regions
            )
        if (
            self.embedding_model_id != "google/embeddinggemma-2"
            and "embedding_dimensions" not in self.model_fields_set
            and "SCOUT_EMBEDDING_DIMENSIONS" not in os.environ
        ):
            from context_scout.retrieval.embedder import resolve_embedding_model_spec

            _, native_dim = resolve_embedding_model_spec(
                self.embedding_model_id, requested_dimensions=None
            )
            self.embedding_dimensions = native_dim
        return self


def detect_hardware_profile(
    settings: ScoutSettings | None = None,
    lora_slots_allocated: int = 2,
) -> dict[str, Any]:
    """Auto-detects host CPU/GPU device, physical memory, quantization backend, and safe KV cap.

    Respects optional SCOUT_HARDWARE_DEVICE, SCOUT_HARDWARE_MEMORY_GB, and
    SCOUT_LOCAL_MAX_KV_TOKENS environment overrides so any developer cloning the
    repository gets accurate hardware settings out of the box.
    """
    cfg = settings or ScoutSettings()

    # 1. Detect physical memory (GB)
    mem_gb = cfg.hardware_memory_gb
    if mem_gb is None:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            mem_gb = round((page_size * phys_pages) / (1024**3), 1)
        except Exception as exc:
            logger.debug(
                "Physical memory auto-detection via sysconf failed (%s): %s; defaulting to 16.0 GB",
                type(exc).__name__,
                exc,
            )
            mem_gb = 16.0

    # 2. Detect chip / accelerator
    device_name = cfg.hardware_device
    if not device_name:
        if sys.platform == "darwin":
            try:
                raw_chip = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=1.0,
                    check=False,
                ).stdout.strip()
                if raw_chip.startswith("Apple "):
                    device_name = raw_chip.replace("Apple ", "Apple Silicon ", 1)
                else:
                    device_name = raw_chip or "Apple Silicon"
            except Exception as exc:
                logger.debug(
                    "macOS chip brand detection failed (%s): %s; defaulting to Apple Silicon",
                    type(exc).__name__,
                    exc,
                )
                device_name = "Apple Silicon"
        else:
            try:
                import torch

                if torch.cuda.is_available():
                    device_name = str(torch.cuda.get_device_name(0))
                else:
                    device_name = platform.processor() or platform.machine() or "Linux CPU"
            except Exception as exc:
                logger.debug(
                    "Accelerator detection via torch failed (%s): %s; using platform identifier",
                    type(exc).__name__,
                    exc,
                )
                device_name = platform.processor() or platform.machine() or "Generic Host"

    # 3. Detect quantization & compute backend
    quantization = cfg.hardware_quantization
    if not quantization:
        if sys.platform == "darwin" and platform.machine() == "arm64":
            quantization = "4-bit (MLX)"
        else:
            try:
                import torch

                quantization = "4-bit (CUDA)" if torch.cuda.is_available() else "4-bit (PyTorch)"
            except Exception as exc:
                logger.debug(
                    "Quantization backend probe failed (%s): %s; defaulting to 4-bit (PyTorch)",
                    type(exc).__name__,
                    exc,
                )
                quantization = "4-bit (PyTorch)"

    # 4. Compute safe KV token budget scaled to available memory unless explicitly overridden
    if "SCOUT_LOCAL_MAX_KV_TOKENS" in os.environ:
        safe_kv = cfg.local_max_kv_tokens
    elif mem_gb < 24.0:
        safe_kv = 8_192
    elif mem_gb < 36.0:
        safe_kv = 16_384
    else:
        safe_kv = min(cfg.local_max_kv_tokens, 32_768)

    return {
        "device": device_name,
        "unified_memory": f"{mem_gb:.1f} GB",
        "memory_gb": mem_gb,
        "max_kv_tokens": safe_kv,
        "quantization": quantization,
        "lora_slots_allocated": lora_slots_allocated,
        "lora_slots_capacity": cfg.lora_slots_capacity,
    }
