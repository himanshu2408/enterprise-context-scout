"""Guardrail types, nonce validation, homoglyph folding, and model catalog."""

from __future__ import annotations

import fnmatch
import re
import secrets
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from context_scout.config import ModelGovernancePolicy
from context_scout.schemas.contracts import PrincipalContext

NONCE_HEX_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-fA-F]{32}$")
DEFAULT_GUARDRAIL_MODEL: Final[str] = "google/shieldgemma-2-2b-it"
_MODEL_ARMOR_SCOPE: Final[str] = "https://www.googleapis.com/auth/cloud-platform"
_DEFAULT_TAG_ATTR_SANITIZE_RE: Final[re.Pattern[str]] = re.compile(r"[\"'<>\r\n\x00-\x1f\x7f]")
_TAG_NAME_SANITIZE_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_.-]")

_HOMOGLYPH_FOLD_MAP: Final[dict[int, str]] = str.maketrans(
    {
        # Cyrillic lowercase lookalikes
        "\u0430": "a",
        "\u0432": "b",
        "\u0435": "e",
        "\u0451": "e",
        "\u0454": "e",
        "\u0456": "i",
        "\u0457": "i",
        "\u0458": "j",
        "\u043a": "k",
        "\u043c": "m",
        "\u043d": "h",
        "\u043e": "o",
        "\u0440": "p",
        "\u0441": "c",
        "\u0442": "t",
        "\u0443": "y",
        "\u04af": "y",
        "\u0445": "x",
        "\u0455": "s",
        "\u04bb": "h",
        "\u04cf": "i",
        "\u0501": "d",
        "\u051b": "q",
        "\u051d": "w",
        # Cyrillic uppercase lookalikes
        "\u0410": "A",
        "\u0412": "B",
        "\u0421": "C",
        "\u0415": "E",
        "\u041d": "H",
        "\u0406": "I",
        "\u0408": "J",
        "\u041a": "K",
        "\u041c": "M",
        "\u041e": "O",
        "\u0420": "P",
        "\u0405": "S",
        "\u0422": "T",
        "\u0425": "X",
        "\u04ae": "Y",
        "\u04c0": "I",
        # Greek uppercase lookalikes
        "\u0391": "A",
        "\u0392": "B",
        "\u0395": "E",
        "\u0396": "Z",
        "\u0397": "H",
        "\u0399": "I",
        "\u039a": "K",
        "\u039c": "M",
        "\u039d": "N",
        "\u039f": "O",
        "\u03a1": "P",
        "\u03a4": "T",
        "\u03a5": "Y",
        "\u03a7": "X",
        "\u03f9": "C",
        # Greek lowercase lookalikes
        "\u03b1": "a",
        "\u03b5": "e",
        "\u03b9": "i",
        "\u03ba": "k",
        "\u03bd": "v",
        "\u03bf": "o",
        "\u03c1": "p",
        "\u03c4": "t",
        "\u03c5": "u",
        "\u03c7": "x",
        "\u03f2": "c",
        "\u03f3": "j",
    }
)


def _fold_homoglyphs(text: str) -> str:
    """Normalizes NFKC fullwidth forms and folds Cyrillic/Greek lookalikes for detection."""
    if text.isascii():
        return text
    return unicodedata.normalize("NFKC", text).translate(_HOMOGLYPH_FOLD_MAP)


def generate_untrusted_nonce() -> str:
    """Generates a cryptographically random 128-bit (32-hex-char) XML boundary nonce."""
    return secrets.token_hex(16)


def validate_untrusted_nonce(nonce: str) -> str:
    """Validates a caller-supplied 128-bit hex nonce and normalizes it to lowercase."""
    cleaned = nonce.strip()
    if not NONCE_HEX_PATTERN.match(cleaned):
        raise ValueError("XML boundary nonce must be a 128-bit (32-character hexadecimal) token.")
    return cleaned.lower()


class SafetyCategory(StrEnum):
    """Safety violation categories aligned with Google ShieldGemma 2 taxonomy."""

    PROMPT_INJECTION = "prompt_injection"
    SECRET_LEAKAGE = "secret_leakage"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MALICIOUS_INPUT = "malicious_input"
    POLICY_VIOLATION = "policy_violation"


class ShieldGemmaDecision(BaseModel):
    """Structured guardrail evaluation decision across deterministic, neural, and Model Armor."""

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    violated_categories: list[SafetyCategory] = Field(default_factory=list)
    rationale: str
    sanitized_content: str | None = None
    redacted_tokens: list[str] = Field(default_factory=list)
    used_fallback: bool = False
    provider_name: str = "deterministic"
    model_id: str = DEFAULT_GUARDRAIL_MODEL
    deterministic_latency_ms: float = 0.0
    neural_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    stage_evaluated: str = "deterministic"
    model_armor_invoked: bool = False


@dataclass(frozen=True)
class GuardrailModelSpec:
    """Metadata describing an approved local neural guardrail model family."""

    pattern: str
    family: str
    prompt_template_style: str
    max_length: int
    latency_budget_ms: float = 30.0


GUARDRAIL_MODEL_CATALOG: Final[tuple[GuardrailModelSpec, ...]] = (
    GuardrailModelSpec(
        pattern="google/shieldgemma-2*",
        family="shieldgemma-2",
        prompt_template_style="shieldgemma_policy",
        max_length=8192,
        latency_budget_ms=30.0,
    ),
    GuardrailModelSpec(
        pattern="meta-llama/Llama-Prompt-Guard-2*",
        family="Prompt-Guard-2",
        prompt_template_style="prompt_guard_2_binary",
        max_length=512,
        latency_budget_ms=30.0,
    ),
    GuardrailModelSpec(
        pattern="Prompt-Guard-2*",
        family="Prompt-Guard-2",
        prompt_template_style="prompt_guard_2_binary",
        max_length=512,
        latency_budget_ms=30.0,
    ),
    GuardrailModelSpec(
        pattern="Qwen/Qwen3Guard*",
        family="Qwen3Guard",
        prompt_template_style="qwen3_guard_safety",
        max_length=8192,
        latency_budget_ms=30.0,
    ),
    GuardrailModelSpec(
        pattern="Qwen3Guard*",
        family="Qwen3Guard",
        prompt_template_style="qwen3_guard_safety",
        max_length=8192,
        latency_budget_ms=30.0,
    ),
    GuardrailModelSpec(
        pattern="ibm-granite/granite-guardian*",
        family="Granite-Guardian",
        prompt_template_style="granite_guardian_risk",
        max_length=8192,
        latency_budget_ms=30.0,
    ),
    GuardrailModelSpec(
        pattern="IBM/Granite-Guardian*",
        family="Granite-Guardian",
        prompt_template_style="granite_guardian_risk",
        max_length=8192,
        latency_budget_ms=30.0,
    ),
    GuardrailModelSpec(
        pattern="Granite-Guardian*",
        family="Granite-Guardian",
        prompt_template_style="granite_guardian_risk",
        max_length=8192,
        latency_budget_ms=30.0,
    ),
)


def resolve_guardrail_model_spec(model_id: str) -> GuardrailModelSpec:
    """Validates model_id via ModelGovernancePolicy and resolves guardrail metadata."""
    cleaned = ModelGovernancePolicy.default().validate_local_model_id(model_id)
    cleaned_lower = cleaned.lower()
    for spec in GUARDRAIL_MODEL_CATALOG:
        if fnmatch.fnmatchcase(cleaned_lower, spec.pattern.lower()):
            return spec
    return GuardrailModelSpec(
        pattern=cleaned,
        family="custom_local_guardrail",
        prompt_template_style="shieldgemma_policy",
        max_length=4096,
        latency_budget_ms=30.0,
    )


@runtime_checkable
class GuardrailProviderSPI(Protocol):
    """Pluggable safety guardrail provider interface (<2ms rules -> <30ms neural / Model Armor)."""

    @property
    def model_id(self) -> str: ...

    @property
    def provider_name(self) -> str: ...

    def evaluate(
        self, text: str, principal: PrincipalContext | None = None
    ) -> ShieldGemmaDecision: ...
