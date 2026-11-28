"""Deterministic SecretRedactor and linear PEM private-key redaction."""

from __future__ import annotations

import re
from typing import Final

_PEM_BEGIN_RE: Final[re.Pattern[str]] = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED |PGP )?PRIVATE KEY(?: BLOCK)?-----"
)
_PEM_END_RE: Final[re.Pattern[str]] = re.compile(
    r"-----END (?:RSA |EC |OPENSSH |DSA |ENCRYPTED |PGP )?PRIVATE KEY(?: BLOCK)?-----"
)


def _redact_pem_private_keys_linear(text: str) -> tuple[str, int]:
    """Redacts PEM private key blocks in strictly O(n) time without quadratic backtracking."""
    if "-----BEGIN " not in text or "-----END " not in text:
        return text, 0
    chunks: list[str] = []
    pos = 0
    n = len(text)
    count = 0
    while pos < n:
        m_begin = _PEM_BEGIN_RE.search(text, pos)
        if m_begin is None:
            chunks.append(text[pos:])
            break
        m_end = _PEM_END_RE.search(text, m_begin.end())
        if m_end is None:
            chunks.append(text[pos:])
            break
        if m_begin.start() > pos:
            chunks.append(text[pos : m_begin.start()])
        chunks.append("[REDACTED:PRIVATE_KEY]")
        count += 1
        pos = m_end.end()
    if count == 0:
        return text, 0
    return "".join(chunks), count


def _extract_redacted_tag(replacement: str) -> str:
    match = re.search(r"\[REDACTED:([A-Z_]+)\]", replacement)
    return match.group(1) if match else "SECRET"


class SecretRedactor:
    """Regex-based secret and PII redaction engine."""

    SECRET_PATTERNS: Final[list[tuple[re.Pattern[str], str]]] = [
        # Google API Keys
        (re.compile(r"\bAIza[0-9A-Za-z\-_]{35,40}\b"), "[REDACTED:GOOGLE_API_KEY]"),
        # GitHub Personal Access Tokens (classic & fine-grained)
        (
            re.compile(
                r"\b(?:(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,255}|github_pat_[A-Za-z0-9_]{22,255})\b"
            ),
            "[REDACTED:GITHUB_TOKEN]",
        ),
        # GitLab Personal Access Tokens
        (re.compile(r"\bglpat-[0-9a-zA-Z\-_]{20,}\b"), "[REDACTED:GITLAB_TOKEN]"),
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
        # Email addresses (negative lookbehind prevents O(n^2) restarts on dotted/delimited runs)
        (
            re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
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

    _SECRET_PATTERN_SPECS: Final[tuple[tuple[str, re.Pattern[str], str], ...]] = tuple(
        (_extract_redacted_tag(repl), pat, repl) for pat, repl in SECRET_PATTERNS
    )

    @classmethod
    def redact(
        cls,
        text: str,
        *,
        skip_tags: frozenset[str] = frozenset(),
    ) -> tuple[str, list[str]]:
        """Redacts sensitive credentials, tokens, paths, and PII from content."""
        redacted_types: list[str] = []
        result = text
        if "PRIVATE_KEY" not in skip_tags:
            result, pem_count = _redact_pem_private_keys_linear(result)
            if pem_count > 0:
                redacted_types.append("PRIVATE_KEY")
        for token_tag, pattern, replacement in cls._SECRET_PATTERN_SPECS:
            if token_tag in skip_tags:
                continue
            result, n_subs = pattern.subn(replacement, result)
            if n_subs > 0 and token_tag not in redacted_types:
                redacted_types.append(token_tag)
        return result, redacted_types
