"""PromptInjectionSanitizer, nonce XML framing, and <2ms DeterministicRuleGuardrailProvider."""

from __future__ import annotations

import html
import re
import time
from typing import Final

from context_scout.schemas.contracts import PrincipalContext
from context_scout.security.guardrails_pkg.linear_scanners import (
    _has_adversarial_html_comment_linear,
    _has_inline_html_event_handler_linear,
    _strip_tokens_linear,
)
from context_scout.security.guardrails_pkg.models_and_types import (
    _DEFAULT_TAG_ATTR_SANITIZE_RE,
    _TAG_NAME_SANITIZE_RE,
    DEFAULT_GUARDRAIL_MODEL,
    SafetyCategory,
    ShieldGemmaDecision,
    _fold_homoglyphs,
    generate_untrusted_nonce,
    validate_untrusted_nonce,
)
from context_scout.security.guardrails_pkg.secret_redactor import SecretRedactor


class PromptInjectionSanitizer:
    """Strips invisible unicode, bidi overrides, comment payloads, and injection triggers."""

    # Zero-width spaces, word joiners, soft hyphen, CGJ, Cf format chars, Variation Selectors, Tags
    INVISIBLE_UNICODE_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"[\u00AD\u034F\u0600-\u0605\u061C\u06DD\u070F\u0890-\u0891\u08E2"
        r"\u17B4-\u17B5\u180B-\u180F\u200B-\u200F\u202A-\u202E\u2060-\u2064\u2066-\u206F"
        r"\u3164\uFE00-\uFE0F\uFEFF\uFFF9-\uFFFB"
        r"\U000110BD\U000110CD\U00013430-\U0001343F\U0001BCA0-\U0001BCA3"
        r"\U0001D173-\U0001D17A\U000E0001-\U000E007F\U000E0100-\U000E01EF]"
    )

    # Directional formatting and bidirectional overrides
    BIDI_OVERRIDE_PATTERN: Final[re.Pattern[str]] = re.compile(r"[\u202A-\u202E\u2066-\u2069]")

    # Control characters excluding standard whitespace (\t, \n, \r)
    CONTROL_CHAR_PATTERN: Final[re.Pattern[str]] = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

    # Malicious active script / URI payloads (OWASP ASI01 WHATWG HTML5 & O(n) anti-ReDoS)
    MALICIOUS_INPUT_PATTERNS: Final[list[tuple[re.Pattern[str], str]]] = [
        (re.compile(r"<\s*script\b", re.IGNORECASE), "script_tag_payload"),
        (
            re.compile(
                r"\bj[\t\n\r]*a[\t\n\r]*v[\t\n\r]*a[\t\n\r]*s[\t\n\r]*"
                r"c[\t\n\r]*r[\t\n\r]*i[\t\n\r]*p[\t\n\r]*t\s*:",
                re.IGNORECASE,
            ),
            "javascript_uri_scheme",
        ),
        (
            re.compile(
                r"\bv[\t\n\r]*b[\t\n\r]*s[\t\n\r]*c[\t\n\r]*r[\t\n\r]*"
                r"i[\t\n\r]*p[\t\n\r]*t\s*:",
                re.IGNORECASE,
            ),
            "vbscript_uri_scheme",
        ),
        (
            re.compile(r"\bdata\s*:\s*(?:text/html|image/svg\+xml)\b", re.IGNORECASE),
            "data_html_uri",
        ),
        (
            re.compile(
                r"""<\s*[a-zA-Z](?:"[^"]*"|'[^']*'|[^<>"'])*\bon(?:error|load|click|mouseover|focus|submit|input)\s*=""",
                re.IGNORECASE,
            ),
            "inline_html_event_handler",
        ),
    ]

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
            re.compile(r"\bsystem\s+(?:prompt\s+)?override\b", re.IGNORECASE),
            "system_prompt_override",
        ),
        (
            re.compile(
                r"\b(?:dump|exfiltrate|leak|print)\s+(?:all\s+)?"
                r"(?:private\s+keys?|secret\s+keys?|secret\s+credentials?|database_url)\b",
                re.IGNORECASE,
            ),
            "secret_exfiltration_attempt",
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
        (
            re.compile(
                r"</\s*untrusted_(?:data|retrieved_data)(?:_[0-9a-zA-Z_-]+)?\s*>",
                re.IGNORECASE,
            ),
            "xml_boundary_breakout",
        ),
    ]

    @classmethod
    def strip_invisible_unicode(cls, text: str) -> str:
        """Removes zero-width characters, bidirectional overrides, and control characters."""
        if text.isascii():
            return cls.CONTROL_CHAR_PATTERN.sub("", text)
        cleaned = cls.INVISIBLE_UNICODE_PATTERN.sub("", text)
        cleaned = cls.BIDI_OVERRIDE_PATTERN.sub("", cleaned)
        return cls.CONTROL_CHAR_PATTERN.sub("", cleaned)

    @classmethod
    def strip_html_comments(cls, text: str) -> str:
        """Strips HTML comments (--> and --!>) to a fixed point in linear O(n) time."""
        cleaned, _ = _strip_tokens_linear(text, strip_comments=True, strip_scripts=False)
        return cleaned

    @classmethod
    def strip_comment_payloads(cls, text: str) -> str:
        """Strips HTML comments and malicious script blocks in a single O(n) pass."""
        cleaned, _ = _strip_tokens_linear(text, strip_comments=True, strip_scripts=True)
        return cleaned

    @classmethod
    def _folded_scan_candidates(cls, text: str) -> list[str]:
        candidates = [_fold_homoglyphs(cls.strip_invisible_unicode(text))]
        if "\x0c" in text or "\x0b" in text:
            ws_text = text.replace("\x0c", " ").replace("\x0b", " ")
            ws_folded = _fold_homoglyphs(cls.strip_invisible_unicode(ws_text))
            if ws_folded != candidates[0]:
                candidates.append(ws_folded)
        return candidates

    @classmethod
    def scan_injection_signatures(cls, text: str) -> tuple[bool, list[str]]:
        """Scans for adversarial instruction overrides and jailbreak signatures."""
        matched: list[str] = []
        for folded in cls._folded_scan_candidates(text):
            for pattern, sig_name in cls.INJECTION_SIGNATURES:
                if pattern.search(folded):
                    matched.append(sig_name)
            if _has_adversarial_html_comment_linear(folded):
                matched.append("html_comment_directive_injection")
        unique_matched = list(dict.fromkeys(matched))
        return (len(unique_matched) > 0, unique_matched)

    @classmethod
    def scan_malicious_input(cls, text: str) -> tuple[bool, list[str]]:
        """Scans for active script blocks, dangerous URI schemes, and event-handler XSS."""
        matched: list[str] = []
        for folded in cls._folded_scan_candidates(text):
            for pattern, sig_name in cls.MALICIOUS_INPUT_PATTERNS[:4]:
                if pattern.search(folded):
                    matched.append(sig_name)
            if "&" in folded:
                unescaped_uri = _fold_homoglyphs(cls.strip_invisible_unicode(html.unescape(folded)))
                for pattern, sig_name in cls.MALICIOUS_INPUT_PATTERNS[1:4]:
                    if pattern.search(unescaped_uri):
                        matched.append(sig_name)
            if _has_inline_html_event_handler_linear(folded):
                matched.append("inline_html_event_handler")
        unique_matched = list(dict.fromkeys(matched))
        return (len(unique_matched) > 0, unique_matched)

    @classmethod
    def sanitize(cls, text: str) -> tuple[str, bool, list[str]]:
        """Sanitizes text and reports whether prompt injection signatures were present."""
        candidates = cls._folded_scan_candidates(text)
        matched: list[str] = []
        for folded in candidates:
            for pattern, sig_name in cls.INJECTION_SIGNATURES:
                if pattern.search(folded):
                    matched.append(sig_name)

        no_invis = cls.strip_invisible_unicode(text)
        cleaned, has_adv = _strip_tokens_linear(
            no_invis,
            strip_comments=True,
            strip_scripts=True,
            check_adversarial_comments=True,
        )
        if has_adv or (len(candidates) > 1 and _has_adversarial_html_comment_linear(candidates[1])):
            matched.append("html_comment_directive_injection")

        if cleaned != no_invis:
            post_folded = _fold_homoglyphs(cleaned)
            for pattern, sig_name in cls.INJECTION_SIGNATURES:
                if pattern.search(post_folded):
                    matched.append(sig_name)
        signatures = list(dict.fromkeys(matched))
        return cleaned, len(signatures) > 0, signatures


def _sanitize_xml_attr(value: str) -> str:
    cleaned = PromptInjectionSanitizer.strip_invisible_unicode(value)
    return _DEFAULT_TAG_ATTR_SANITIZE_RE.sub("", cleaned).strip()


def frame_untrusted_content(
    content: str,
    source_id: str,
    domain_id: str,
    tag: str = "untrusted_data",
    *,
    nonce: str | None = None,
) -> str:
    """Frames untrusted retrieved data inside a per-request 128-bit random XML boundary nonce.

    Prevents prompt-injection boundary breakouts by:
    1. Generating (or validating) a 128-bit CSPRNG hex nonce (`<untrusted_data_{nonce}>`).
    2. Stripping invisible unicode, bidi overrides, and HTML/script comment payloads.
    3. Stripping any matching closing or opening untrusted boundary tags inside the body
       until a fixed point so nested delimiter reconstruction cannot recreate a boundary tag.
    4. Redacting raw secrets, credentials, and PII.
    """
    active_nonce = (
        validate_untrusted_nonce(nonce) if nonce is not None else generate_untrusted_nonce()
    )
    cleaned_base = _TAG_NAME_SANITIZE_RE.sub("", tag.strip()) or "untrusted_data"
    if cleaned_base in {"untrusted_data", "untrusted_retrieved_data"}:
        effective_tag = f"untrusted_data_{active_nonce}"
    elif cleaned_base.endswith(f"_{active_nonce}"):
        effective_tag = cleaned_base
    else:
        effective_tag = f"{cleaned_base}_{active_nonce}"

    sanitized = PromptInjectionSanitizer.strip_invisible_unicode(content)

    tag_family = (
        f"(?:untrusted_data|untrusted_retrieved_data|{re.escape(cleaned_base)})"
        r"(?:_[0-9a-zA-Z_-]+)?"
    )
    untrusted_tag_prefix_re = re.compile(
        rf"</?\s*(?:{re.escape(effective_tag)}|{tag_family})\b",
        re.IGNORECASE,
    )
    # Single-pass O(n) stripping of comments, scripts, untrusted boundary tags, and chat delimiters.
    sanitized, _ = _strip_tokens_linear(
        sanitized,
        strip_comments=True,
        strip_scripts=True,
        strip_chat_delimiters=True,
        untrusted_tag_prefix_re=untrusted_tag_prefix_re,
    )

    sanitized, _ = SecretRedactor.redact(sanitized)
    safe_source = _sanitize_xml_attr(source_id)
    safe_domain = _sanitize_xml_attr(domain_id)

    return (
        f'<{effective_tag} source="{safe_source}" domain="{safe_domain}">\n'
        f"{sanitized}\n"
        f"</{effective_tag}>"
    )


_INLINE_CODE_SPAN_RE: Final[re.Pattern[str]] = re.compile(r"`([^`\r\n]{1,64})`")
_BENIGN_TECH_QUERY_FRAMING_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:\?|\b(?:how|why|what|where|when|does|do|explain|describe|handle|handles|sanitize|"
    r"sanitizer|strip|strips|escape|escapes|parse|parser|block|blocks|prevent|detect|test|"
    r"audit|fix|review)\b)",
    re.IGNORECASE,
)
_UNQUOTED_IMPERATIVE_RISK_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:<!--|</\s*untrusted_(?:data|retrieved_data)|"
    r"\b(?:ignore|disregard|override|forget|bypass|reveal|dump|exfiltrate|leak|"
    r"jailbreak|unrestricted|pretend|roleplay|dan\s+mode|developer\s+mode|sudo\s+mode|"
    r"god\s+mode|do\s+not\s+mention|never\s+tell\s+the\s+user)\b)",
    re.IGNORECASE,
)
_ACTIVE_QUOTED_SPAN_DISALLOWED_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:[();]|//|https?:|javascript:|vbscript:|data:|<!--|</\s*untrusted_|"
    r"\b(?:ignore|disregard|override|bypass|reveal|dump|exfiltrate|secret|password|"
    r"token|cookie|document\.|window\.|eval|alert|fetch|src=|href=)\b)",
    re.IGNORECASE,
)


def _mask_benign_inline_code_quotes(text: str) -> str:
    """Masks passive syntax tokens inside inline backticks for O(n) benign dev queries (ECS-006)."""
    if "`" not in text or len(text) > 4096:
        return text
    if not _BENIGN_TECH_QUERY_FRAMING_RE.search(text):
        return text
    spans = list(_INLINE_CODE_SPAN_RE.finditer(text))
    if not spans or len(spans) > 8:
        return text

    # Build unquoted outside prose first and abort if prose carries injection cues
    outside_parts: list[str] = []
    cursor = 0
    for m in spans:
        outside_parts.append(text[cursor : m.start()])
        outside_parts.append(" CODE_TOKEN ")
        cursor = m.end()
    outside_parts.append(text[cursor:])
    outside_prose = "".join(outside_parts)
    if _UNQUOTED_IMPERATIVE_RISK_RE.search(outside_prose):
        return text

    # Verify each individual quoted token is passive syntax (not an active script/exfil/override)
    out_parts: list[str] = []
    pos = 0
    changed = False
    for m in spans:
        inner = m.group(1)
        out_parts.append(text[pos : m.start()])
        if _ACTIVE_QUOTED_SPAN_DISALLOWED_RE.search(inner):
            out_parts.append(m.group(0))
        else:
            out_parts.append(" CODE_TOKEN ")
            changed = True
        pos = m.end()
    if not changed:
        return text
    out_parts.append(text[pos:])
    return "".join(out_parts)


class DeterministicRuleGuardrailProvider:
    """Stage A (<2ms) deterministic rule gate verifying size, injections, XSS, and secrets."""

    MAX_PAYLOAD_BYTES: Final[int] = 32 * 1024  # 32 KB safety cap

    def __init__(self, model_id: str = DEFAULT_GUARDRAIL_MODEL) -> None:
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider_name(self) -> str:
        return "deterministic"

    def evaluate(self, text: str, principal: PrincipalContext | None = None) -> ShieldGemmaDecision:
        """Evaluates deterministic safety rules in <2ms without model or network calls."""
        del principal
        t0 = time.perf_counter()
        violated_categories: list[SafetyCategory] = []
        reasons: list[str] = []

        text_bytes = len(text.encode("utf-8"))
        scan_text = text
        if text_bytes > self.MAX_PAYLOAD_BYTES:
            violated_categories.append(SafetyCategory.POLICY_VIOLATION)
            reasons.append(
                f"Payload size ({text_bytes} bytes) exceeds safety cap "
                f"of {self.MAX_PAYLOAD_BYTES} bytes."
            )
            scan_text = text[: self.MAX_PAYLOAD_BYTES]

        raw_no_invis = PromptInjectionSanitizer.strip_invisible_unicode(scan_text)
        eval_text = _mask_benign_inline_code_quotes(scan_text)
        no_invis = (
            raw_no_invis
            if eval_text == scan_text
            else PromptInjectionSanitizer.strip_invisible_unicode(eval_text)
        )
        post_comment_text = PromptInjectionSanitizer.strip_html_comments(no_invis)
        _, pre_malicious_sigs = PromptInjectionSanitizer.scan_malicious_input(eval_text)
        post_comment_malicious_sigs: list[str] = []
        if post_comment_text != no_invis:
            _, post_comment_malicious_sigs = PromptInjectionSanitizer.scan_malicious_input(
                post_comment_text
            )
        sanitized_text, is_injection, injection_sigs = PromptInjectionSanitizer.sanitize(eval_text)
        post_sanitize_malicious_sigs: list[str] = []
        if sanitized_text not in {eval_text, no_invis, post_comment_text}:
            _, post_sanitize_malicious_sigs = PromptInjectionSanitizer.scan_malicious_input(
                sanitized_text
            )
        malicious_sigs = list(
            dict.fromkeys(
                [*pre_malicious_sigs, *post_comment_malicious_sigs, *post_sanitize_malicious_sigs]
            )
        )
        is_malicious = len(malicious_sigs) > 0

        if is_injection:
            violated_categories.append(SafetyCategory.PROMPT_INJECTION)
            reasons.append(f"Instruction injection signatures detected: {injection_sigs}")

        if is_malicious:
            violated_categories.append(SafetyCategory.MALICIOUS_INPUT)
            reasons.append(f"Malicious active input patterns detected: {malicious_sigs}")

        redacted_input = (
            sanitized_text
            if eval_text == scan_text
            else PromptInjectionSanitizer.strip_comment_payloads(raw_no_invis)
        )
        redacted_text, redacted_tokens = SecretRedactor.redact(redacted_input)
        if redacted_tokens:
            violated_categories.append(SafetyCategory.SECRET_LEAKAGE)
            reasons.append(f"Credentials or sensitive tokens detected: {redacted_tokens}")

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
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
            provider_name=self.provider_name,
            model_id=self._model_id,
            deterministic_latency_ms=elapsed_ms,
            neural_latency_ms=0.0,
            total_latency_ms=elapsed_ms,
            stage_evaluated="deterministic",
            model_armor_invoked=False,
        )
