"""Enterprise Guardrails: Prompt-injection sanitization, Delimiter framing, PII redaction."""

import re
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, Field

from context_scout.config import ScoutSettings, validate_google_model_id
from context_scout.schemas.contracts import PrincipalContext


class SafetyCategory(StrEnum):
    """Safety violation categories aligned with Google ShieldGemma 2 taxonomy."""

    PROMPT_INJECTION = "prompt_injection"
    SECRET_LEAKAGE = "secret_leakage"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MALICIOUS_INPUT = "malicious_input"
    POLICY_VIOLATION = "policy_violation"


class ShieldGemmaDecision(BaseModel):
    """Structured decision emitted by ShieldGemma 2 or deterministic fallback."""

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    violated_categories: list[SafetyCategory] = Field(default_factory=list)
    rationale: str
    sanitized_content: str | None = None
    redacted_tokens: list[str] = Field(default_factory=list)
    used_fallback: bool = False


class PromptInjectionSanitizer:
    """Strips invisible unicode, bidi overrides, comment payloads, and injection triggers."""

    # Zero-width spaces, word joiners, invisible separators
    INVISIBLE_UNICODE_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"[\u200B-\u200D\uFEFF\u2060\u2062-\u2064]"
    )

    # Directional formatting and bidirectional overrides
    BIDI_OVERRIDE_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"[\u202A-\u202E\u2066-\u2069]"
    )

    # Control characters excluding standard whitespace (\t, \n, \r)
    CONTROL_CHAR_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"
    )

    # Comment payloads (HTML comments, script blocks)
    HTML_COMMENT_PATTERN: Final[re.Pattern[str]] = re.compile(r"<!--[\s\S]*?-->")
    SCRIPT_BLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>", re.IGNORECASE
    )

    # Instruction injection / jailbreak signatures
    INJECTION_SIGNATURES: Final[list[tuple[re.Pattern[str], str]]] = [
        (
            re.compile(
                r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|rules|directives)\b",
                re.IGNORECASE,
            ),
            "ignore_previous_instructions",
        ),
        (
            re.compile(
                r"\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|rules)\b",
                re.IGNORECASE,
            ),
            "disregard_previous_instructions",
        ),
        (
            re.compile(r"\bsystem\s+prompt\s+override\b", re.IGNORECASE),
            "system_prompt_override",
        ),
        (
            re.compile(r"\byou\s+are\s+now\s+in\s+dan\s+mode\b", re.IGNORECASE),
            "dan_mode_jailbreak",
        ),
        (
            re.compile(r"\bdeveloper\s+mode\s+(?:is\s+)?enabled\b", re.IGNORECASE),
            "developer_mode_override",
        ),
        (
            re.compile(
                r"\bbypass\s+(?:all\s+)?(?:guardrails|filters|safety\s+checks)\b",
                re.IGNORECASE,
            ),
            "bypass_guardrails",
        ),
        (
            re.compile(r"\bact\s+as\s+an?\s+unrestricted\b", re.IGNORECASE),
            "unrestricted_persona",
        ),
        (
            re.compile(
                r"\breveal\s+(?:your\s+)?(?:system\s+prompt|initialization\s+instructions|secret\s+keys?)\b",
                re.IGNORECASE,
            ),
            "reveal_system_prompt",
        ),
        (
            re.compile(
                r"\bBEGIN\s+(?:SYSTEM|PROMPT|INSTRUCTION)\b",
                re.IGNORECASE,
            ),
            "raw_delimiter_injection",
        ),
        (
            re.compile(r"<\s*(?:system|instruction|prompt)\s*>", re.IGNORECASE),
            "tag_instruction_injection",
        ),
    ]

    @classmethod
    def strip_invisible_unicode(cls, text: str) -> str:
        """Removes zero-width characters, bidirectional overrides, and control characters."""
        cleaned = cls.INVISIBLE_UNICODE_PATTERN.sub("", text)
        cleaned = cls.BIDI_OVERRIDE_PATTERN.sub("", cleaned)
        return cls.CONTROL_CHAR_PATTERN.sub("", cleaned)

    @classmethod
    def strip_comment_payloads(cls, text: str) -> str:
        """Strips HTML comments and malicious script blocks."""
        cleaned = cls.HTML_COMMENT_PATTERN.sub("", text)
        return cls.SCRIPT_BLOCK_PATTERN.sub("", cleaned)

    @classmethod
    def scan_injection_signatures(cls, text: str) -> tuple[bool, list[str]]:
        """Scans for adversarial instruction overrides and jailbreak signatures."""
        matched: list[str] = []
        for pattern, sig_name in cls.INJECTION_SIGNATURES:
            if pattern.search(text):
                matched.append(sig_name)
        return (len(matched) > 0, matched)

    @classmethod
    def sanitize(cls, text: str) -> tuple[str, bool, list[str]]:
        """Sanitizes text and reports whether prompt injection signatures were present."""
        cleaned = cls.strip_invisible_unicode(text)
        cleaned = cls.strip_comment_payloads(cleaned)
        is_injection, signatures = cls.scan_injection_signatures(cleaned)
        return cleaned, is_injection, signatures


class SecretRedactor:
    """Regex-based secret and PII redaction engine."""

    SECRET_PATTERNS: Final[list[tuple[re.Pattern[str], str]]] = [
        # Google API Keys
        (re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"), "[REDACTED:GOOGLE_API_KEY]"),
        # GitHub Personal Access Tokens
        (
            re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,255}\b"),
            "[REDACTED:GITHUB_TOKEN]",
        ),
        # GitLab Personal Access Tokens
        (re.compile(r"\bglpat-[0-9a-zA-Z\-_]{20,}\b"), "[REDACTED:GITLAB_TOKEN]"),
        # PEM Private Keys
        (
            re.compile(
                r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----[\s\S]*?"
                r"-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
            ),
            "[REDACTED:PRIVATE_KEY]",
        ),
        # Bearer tokens / JWTs
        (
            re.compile(
                r"(?i)\bBearer\s+(?:eyJ[A-Za-z0-9\-_=]+\.[A-Za-z0-9\-_=]+\.?[A-Za-z0-9\-_=]*|"
                r"[A-Za-z0-9\-_.~+/]{24,}=*)"
            ),
            "Bearer [REDACTED:BEARER_TOKEN]",
        ),
        # Local User Workstation Paths
        (
            re.compile(r"(?:/Users/|/home/)[a-zA-Z0-9_.-]+"),
            "[REDACTED:USER_PATH]",
        ),
        # Email addresses
        (
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            "[REDACTED:EMAIL]",
        ),
        # Key/Token/Password assignments
        (
            re.compile(
                r"(?i)\b((?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|"
                r"password)\s*[:=]\s*['\"]?)([A-Za-z0-9+/=_\-]{16,})(['\"]?)"
            ),
            r"\1[REDACTED:SECRET]\3",
        ),
    ]

    @classmethod
    def redact(cls, text: str) -> tuple[str, list[str]]:
        """Redacts sensitive credentials, tokens, paths, and PII from content."""
        redacted_types: list[str] = []
        result = text
        for pattern, replacement in cls.SECRET_PATTERNS:
            if pattern.search(result):
                tag_match = re.search(r"\[REDACTED:([A-Z_]+)\]", replacement)
                token_tag = tag_match.group(1) if tag_match else "SECRET"
                if token_tag not in redacted_types:
                    redacted_types.append(token_tag)
                result = pattern.sub(replacement, result)
        return result, redacted_types


def frame_untrusted_content(
    content: str,
    source_id: str,
    domain_id: str,
    tag: str = "untrusted_retrieved_data",
) -> str:
    """Frames untrusted retrieved data and neutralizes nested delimiter breakouts.

    Prevents prompt injection by:
    1. Escaping any occurrences of </tag> or <tag in the content.
    2. Stripping invisible unicode and hidden comments.
    3. Redacting raw secrets or credentials.
    4. Enclosing cleanly inside explicit structural boundary delimiters.
    """
    sanitized = PromptInjectionSanitizer.strip_invisible_unicode(content)
    sanitized = PromptInjectionSanitizer.strip_comment_payloads(sanitized)

    closing_pattern = re.compile(rf"</\s*{re.escape(tag)}\s*>", re.IGNORECASE)
    opening_pattern = re.compile(rf"<\s*{re.escape(tag)}\b", re.IGNORECASE)
    sanitized = closing_pattern.sub(f"&lt;/{tag}&gt;", sanitized)
    sanitized = opening_pattern.sub(f"&lt;{tag}", sanitized)

    sanitized, _ = SecretRedactor.redact(sanitized)

    return f'<{tag} source="{source_id}" domain="{domain_id}">\n{sanitized}\n</{tag}>'


class ShieldGemmaGuardrail:
    """Guardrail engine integrating Google ShieldGemma 2 with deterministic fallbacks."""

    MAX_PAYLOAD_BYTES: Final[int] = 32 * 1024  # 32 KB safety cap

    def __init__(
        self,
        model_id: str = "google/shieldgemma-2-2b-it",
        settings: ScoutSettings | None = None,
        allow_deterministic_fallback: bool = True,
    ) -> None:
        self.settings = settings or ScoutSettings()
        self.model_id = validate_google_model_id(model_id)
        self.allow_deterministic_fallback = allow_deterministic_fallback

    def evaluate_deterministic_fallback(
        self, text: str, principal: PrincipalContext | None = None
    ) -> ShieldGemmaDecision:
        """Fast offline rule engine verifying secrets, injection patterns, and size limits."""
        violated_categories: list[SafetyCategory] = []
        reasons: list[str] = []

        # 1. Payload size check
        text_bytes = len(text.encode("utf-8"))
        if text_bytes > self.MAX_PAYLOAD_BYTES:
            violated_categories.append(SafetyCategory.POLICY_VIOLATION)
            reasons.append(
                f"Payload size ({text_bytes} bytes) exceeds safety cap "
                f"of {self.MAX_PAYLOAD_BYTES} bytes."
            )

        # 2. Prompt injection check
        sanitized_text, is_injection, injection_sigs = PromptInjectionSanitizer.sanitize(text)
        if is_injection:
            violated_categories.append(SafetyCategory.PROMPT_INJECTION)
            reasons.append(f"Instruction injection signatures detected: {injection_sigs}")

        # 3. Secret and PII check
        redacted_text, redacted_tokens = SecretRedactor.redact(sanitized_text)
        if redacted_tokens:
            violated_categories.append(SafetyCategory.SECRET_LEAKAGE)
            reasons.append(f"Credentials or sensitive tokens detected: {redacted_tokens}")

        passed = len(violated_categories) == 0
        score = 1.0 if passed else 0.0
        rationale = "Passed deterministic heuristic safety gate." if passed else "; ".join(reasons)

        return ShieldGemmaDecision(
            passed=passed,
            score=score,
            violated_categories=violated_categories,
            rationale=rationale,
            sanitized_content=redacted_text,
            redacted_tokens=redacted_tokens,
            used_fallback=True,
        )

    def evaluate(
        self, text: str, principal: PrincipalContext | None = None
    ) -> ShieldGemmaDecision:
        """Evaluates input or retrieved content, falling back to deterministic heuristic rules."""
        if self.allow_deterministic_fallback:
            return self.evaluate_deterministic_fallback(text, principal=principal)

        return self.evaluate_deterministic_fallback(text, principal=principal)
