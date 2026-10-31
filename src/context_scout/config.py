"""Typed configuration settings enforcing the 100% Google-only model policy."""

import re
from pathlib import Path
from typing import Final

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ALLOWED_GOOGLE_MODEL_PREFIXES: Final[tuple[str, ...]] = (
    "google/gemma-4-",
    "mlx-community/gemma-4-",
    "google/embeddinggemma-2",
    "google/shieldgemma-2",
    "gemini-2.5-",
    "gemini-3.5-",
    "gemini-3.8-",
    "gemini-4-",
    "gemini-4-argon",
    "gemini-embedding-",
)

PROHIBITED_STALE_MODEL_REGEX: Final[re.Pattern[str]] = re.compile(
    r"(?:^|/)(?:text-embedding-004|gemini-ultra|gemma-[123](?:-|$)|codegemma)",
    re.IGNORECASE,
)


def validate_google_model_id(model_id: str) -> str:
    """Enforces that only current October 2026 Google model IDs are configured."""
    cleaned = model_id.strip()
    if PROHIBITED_STALE_MODEL_REGEX.search(cleaned):
        raise ValueError(f"Prohibited or retired model ID '{model_id}'.")
    if not any(cleaned.startswith(prefix) for prefix in ALLOWED_GOOGLE_MODEL_PREFIXES):
        raise ValueError(
            f"Model '{model_id}' violates the 100% Google-only model policy. "
            f"Allowed prefixes: {ALLOWED_GOOGLE_MODEL_PREFIXES}"
        )
    return cleaned


class ScoutSettings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_prefix="SCOUT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_admin_url: str = Field(
        default="postgresql://postgres:postgres_local_admin_pw@127.0.0.1:54329/context_scout"
    )
    db_app_url: str = Field(
        default="postgresql://scout_app_user:scout_local_dev_pw@127.0.0.1:54329/context_scout"
    )

    primary_model_id: str = Field(default="google/gemma-4-26B-A4B-it")
    contextualizer_model_id: str = Field(default="google/gemma-4-E4B-it")
    teacher_model_id: str = Field(default="google/gemma-4-31B-it")
    embedding_model_id: str = Field(default="google/embeddinggemma-2")
    shield_model_id: str = Field(default="google/shieldgemma-2-2b-it")
    frontier_model_id: str = Field(default="gemini-3.8-flash")

    embedding_dimensions: int = Field(default=768, ge=128, le=3072)
    local_max_kv_tokens: int = Field(default=32_768, ge=4_096, le=32_768)
    local_mlx_endpoint: str = Field(default="http://127.0.0.1:8000/v1")
    domains_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "domains"
    )

    @field_validator(
        "primary_model_id",
        "contextualizer_model_id",
        "teacher_model_id",
        "embedding_model_id",
        "shield_model_id",
        "frontier_model_id",
    )
    @classmethod
    def _enforce_google_models(cls, value: str) -> str:
        return validate_google_model_id(value)
