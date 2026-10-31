"""Typed configuration settings and runtime hardware auto-detection."""

import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

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
    local_max_kv_tokens: int = Field(default=32_768, ge=4_096, le=131_072)
    local_mlx_endpoint: str = Field(default="http://127.0.0.1:8000/v1")
    hardware_device: str | None = Field(default=None)
    hardware_memory_gb: float | None = Field(default=None, ge=4.0, le=1024.0)
    hardware_quantization: str | None = Field(default=None)
    lora_slots_capacity: int = Field(default=4, ge=1, le=32)
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
        except Exception:
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
            except Exception:
                device_name = "Apple Silicon"
        else:
            try:
                import torch

                if torch.cuda.is_available():
                    device_name = str(torch.cuda.get_device_name(0))
                else:
                    device_name = platform.processor() or platform.machine() or "Linux CPU"
            except Exception:
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
            except Exception:
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
