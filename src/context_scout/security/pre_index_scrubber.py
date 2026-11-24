"""Pre-index secret detection and PII scrubbing before persistence or embedding.

Combines an optional local ``gitleaks detect --pipe`` subprocess check (when
the ``gitleaks`` binary is installed in the private environment) with
deterministic O(n) built-in secret and PII rules (Luhn card check, ISO 13616
MOD-97 IBAN check, SSN, email, phone, and DSN/token/PEM patterns), plus
defense-in-depth ``SecretRedactor`` verification.

No raw secret or PII material is ever stored in findings or logs; each finding
records only a truncated SHA-256 fingerprint, rule id, and character offsets.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Final, Literal

from context_scout.security.guardrails import SecretRedactor

logger = logging.getLogger(__name__)

_SAFE_PLACEHOLDER_PREFIXES: Final[tuple[str, ...]] = (
    "<",
    "${",
    "{{",
    "[REDACTED",
    "REDACTED",
    "EXAMPLE",
    "PLACEHOLDER",
    "CHANGEME",
    "YOUR_",
    "DUMMY",
    "TEST_",
)
_SAFE_PLACEHOLDER_EXACT: Final[frozenset[str]] = frozenset(
    {
        "none",
        "null",
        "true",
        "false",
        "undefined",
        "changeme",
        "placeholder",
        "example",
        "redacted",
        "password",
        "secret",
        "token",
    }
)


def _fingerprint(value: str) -> str:
    """Return a non-reversible SHA-256 prefix for audit correlation."""
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _luhn_valid(digits: str) -> bool:
    """Validate payment card candidate digits using the Luhn checksum."""
    if not (13 <= len(digits) <= 19) or not digits.isdigit():
        return False
    if len(set(digits)) <= 1:
        return False
    total = 0
    reverse_digits = digits[::-1]
    for idx, ch in enumerate(reverse_digits):
        n = ord(ch) - 48
        if idx % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _iban_valid(candidate: str) -> bool:
    """Validate candidate IBAN strings using ISO 13616 MOD 97-10."""
    compact = re.sub(r"[\s-]", "", candidate).upper()
    if not (15 <= len(compact) <= 34):
        return False
    if not (compact[:2].isalpha() and compact[2:4].isdigit() and compact[4:].isalnum()):
        return False
    rearranged = compact[4:] + compact[:4]
    remainder = 0
    for ch in rearranged:
        if "0" <= ch <= "9":
            remainder = (remainder * 10 + (ord(ch) - 48)) % 97
        else:
            val = ord(ch) - 55  # A=10 .. Z=35
            remainder = (remainder * 100 + val) % 97
    return remainder == 1


@dataclass(frozen=True)
class ScrubFinding:
    """Metadata for a single redacted secret or PII occurrence."""

    category: Literal["secret", "pii"]
    rule_id: str
    start: int
    end: int
    fingerprint: str


@dataclass(frozen=True)
class ScrubResult:
    """Output of running ``PreIndexScrubber`` on text."""

    scrubbed_text: str
    findings: tuple[ScrubFinding, ...]
    secret_count: int
    pii_count: int
    used_gitleaks_cli: bool
    findings_by_rule: dict[str, int]


# Strictly linear O(n) patterns without nested/ambiguous quantifiers.
_PEM_PRIVATE_KEY_RE: Final[re.Pattern[str]] = re.compile(
    r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"
    r"[\s\S]{1,16384}?"
    r"-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----",
    re.MULTILINE,
)
_AWS_ACCESS_KEY_RE: Final[re.Pattern[str]] = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
_GITHUB_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,255}|github_pat_[A-Za-z0-9_]{20,255})\b"
)
_GITLAB_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:glpat|glptt|gldt|gloas)-[A-Za-z0-9_-]{20,128}\b"
)
_GOOGLE_API_KEY_RE: Final[re.Pattern[str]] = re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")
_SLACK_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,128}\b")
_SLACK_WEBHOOK_RE: Final[re.Pattern[str]] = re.compile(
    r"https://hooks\.slack\.com/services/T[A-Z0-9]{8,}/B[A-Z0-9]{8,}/[A-Za-z0-9]{20,}"
)
_JWT_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
)
_DSN_CREDENTIALS_RE: Final[re.Pattern[str]] = re.compile(
    r"\b((?:postgresql|postgres|mysql|mariadb|mongodb(?:\+srv)?|redis|rediss|amqps?)"
    r"://[^:\s/@'\"]{1,64}:)([^@\s/'\"]{4,128})(@[^/\s'\"]{1,128})",
    re.IGNORECASE,
)
_QUOTED_SECRET_ASSIGN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(aws_secret_access_key|api[_-]?key|apikey|access[_-]?token|auth[_-]?token|"
    r"client[_-]?secret|webhook[_-]?secret|signing[_-]?secret|private[_-]?key|"
    r"secret[_-]?key|db[_-]?password|password|passwd|bearer)\b"
    r"(\s*[:=]\s*)"
    r"([\"'])([^\"'\r\n]{8,256})\3"
)
_BEARER_HEADER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(Authorization\s*[:=]\s*[\"']?(?:Bearer|Basic)\s+)([A-Za-z0-9._+/=-]{12,512})"
)

_SSN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<!\d)(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)"
)
_IBAN_CANDIDATE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b[A-Z]{2}\d{2}(?:[ -]?[A-Z0-9]){11,30}\b"
)
_CARD_CANDIDATE_RE: Final[re.Pattern[str]] = re.compile(r"(?<!\d)(?:\d{4}[ -]?){3}\d{1,7}(?!\d)")
_EMAIL_RE: Final[re.Pattern[str]] = re.compile(
    r"\b[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,128}\.[A-Za-z]{2,24}\b"
)
_PHONE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w\-./])(?:\+\d{1,3}[-.\s])?(?:\(\d{2,4}\)|\d{2,4})[-.\s]\d{3,4}[-.\s]\d{4}(?![\w\-./])"
)
_DOTTED_QUAD_RE: Final[re.Pattern[str]] = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


class PreIndexScrubber:
    """Deterministic pre-index secret & PII sanitizer for code and doc connectors."""

    def __init__(
        self,
        *,
        use_gitleaks_cli: bool = True,
        scrub_secrets: bool = True,
        scrub_pii: bool = True,
        gitleaks_timeout_seconds: float = 5.0,
        allowlisted_email_domains: tuple[str, ...] = (),
        gitleaks_binary: str | None = None,
    ) -> None:
        self._use_gitleaks_cli = use_gitleaks_cli
        self._scrub_secrets = scrub_secrets
        self._scrub_pii = scrub_pii
        self._gitleaks_timeout_seconds = max(0.5, gitleaks_timeout_seconds)
        self._allowlisted_email_domains = frozenset(
            d.strip().lower() for d in allowlisted_email_domains if d.strip()
        )
        self._gitleaks_binary = (
            gitleaks_binary if gitleaks_binary is not None else shutil.which("gitleaks")
        )

    def scrub_text(self, text: str, *, source_name: str = "") -> ScrubResult:
        """Scrub secrets and PII from ``text`` before persistence or embedding."""
        if not text:
            return ScrubResult(
                scrubbed_text="",
                findings=(),
                secret_count=0,
                pii_count=0,
                used_gitleaks_cli=False,
                findings_by_rule={},
            )

        current = text
        findings: list[ScrubFinding] = []
        used_gitleaks_cli = False

        if self._scrub_secrets and self._use_gitleaks_cli and self._gitleaks_binary:
            current, gl_findings, used_gitleaks_cli = self._run_gitleaks_pipe(
                current, source_name=source_name
            )
            findings.extend(gl_findings)

        if self._scrub_secrets:
            current, sec_findings = self._apply_secret_rules(current)
            findings.extend(sec_findings)

        if self._scrub_pii:
            current, pii_findings = self._apply_pii_rules(current)
            findings.extend(pii_findings)

        if self._scrub_secrets:
            # Defense-in-depth pass via shared SecretRedactor after explicit PII rules;
            # skip EMAIL here so _apply_pii_rules owns email PII & allowlisted domains.
            redacted, guard_labels = SecretRedactor.redact(
                current,
                skip_tags=frozenset({"EMAIL"}),
            )
            if guard_labels:
                current = redacted
                for label in guard_labels:
                    findings.append(
                        ScrubFinding(
                            category="secret",
                            rule_id=f"guardrail:{label}",
                            start=0,
                            end=0,
                            fingerprint=_fingerprint(label),
                        )
                    )

        counts: dict[str, int] = {}
        secret_count = 0
        pii_count = 0
        for item in findings:
            counts[item.rule_id] = counts.get(item.rule_id, 0) + 1
            if item.category == "secret":
                secret_count += 1
            else:
                pii_count += 1

        return ScrubResult(
            scrubbed_text=current,
            findings=tuple(findings),
            secret_count=secret_count,
            pii_count=pii_count,
            used_gitleaks_cli=used_gitleaks_cli,
            findings_by_rule=counts,
        )

    def _run_gitleaks_pipe(
        self, text: str, *, source_name: str
    ) -> tuple[str, list[ScrubFinding], bool]:
        if not self._gitleaks_binary:
            return text, [], False
        try:
            proc = subprocess.run(
                [
                    self._gitleaks_binary,
                    "detect",
                    "--pipe",
                    "--no-banner",
                    "--report-format",
                    "json",
                    "--report-path",
                    "-",
                ],
                input=text,
                capture_output=True,
                text=True,
                timeout=self._gitleaks_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning(
                "gitleaks pipe invocation failed for %s (%s); continuing with built-in rules",
                source_name or "buffer",
                type(exc).__name__,
            )
            return text, [], False

        # gitleaks exits 0 (clean) or 1 (leaks found).
        if proc.returncode not in (0, 1):
            logger.warning(
                "gitleaks pipe exited %s for %s; continuing with built-in rules",
                proc.returncode,
                source_name or "buffer",
            )
            return text, [], False

        if proc.returncode == 0 or not proc.stdout.strip():
            return text, [], True

        try:
            parsed = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            logger.warning("gitleaks pipe emitted non-JSON output (%s)", type(exc).__name__)
            return text, [], True

        if not isinstance(parsed, list):
            return text, [], True

        updated = text
        findings: list[ScrubFinding] = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            secret_val = str(entry.get("Secret") or entry.get("Match") or "")
            if not secret_val or "[REDACTED" in secret_val:
                continue
            rule_id = str(entry.get("RuleID") or "gitleaks")
            pos = updated.find(secret_val)
            while pos != -1:
                replacement = f"[REDACTED:secret:{rule_id}]"
                findings.append(
                    ScrubFinding(
                        category="secret",
                        rule_id=f"gitleaks:{rule_id}",
                        start=pos,
                        end=pos + len(secret_val),
                        fingerprint=_fingerprint(secret_val),
                    )
                )
                updated = updated[:pos] + replacement + updated[pos + len(secret_val) :]
                pos = updated.find(secret_val, pos + len(replacement))

        return updated, findings, True

    def _apply_secret_rules(self, text: str) -> tuple[str, list[ScrubFinding]]:
        findings: list[ScrubFinding] = []
        current = text

        direct_rules: tuple[tuple[str, re.Pattern[str]], ...] = (
            ("private-key-pem", _PEM_PRIVATE_KEY_RE),
            ("aws-access-key", _AWS_ACCESS_KEY_RE),
            ("github-token", _GITHUB_TOKEN_RE),
            ("gitlab-token", _GITLAB_TOKEN_RE),
            ("google-api-key", _GOOGLE_API_KEY_RE),
            ("slack-token", _SLACK_TOKEN_RE),
            ("slack-webhook", _SLACK_WEBHOOK_RE),
            ("jwt-token", _JWT_TOKEN_RE),
        )
        for rule_id, pattern in direct_rules:

            def _replace_direct(m: re.Match[str], _rule: str = rule_id) -> str:
                matched = m.group(0)
                if "[REDACTED" in matched:
                    return matched
                findings.append(
                    ScrubFinding(
                        category="secret",
                        rule_id=_rule,
                        start=m.start(),
                        end=m.end(),
                        fingerprint=_fingerprint(matched),
                    )
                )
                return f"[REDACTED:secret:{_rule}]"

            current = pattern.sub(_replace_direct, current)

        def _replace_dsn(m: re.Match[str]) -> str:
            secret_part = m.group(2)
            if "[REDACTED" in secret_part:
                return m.group(0)
            findings.append(
                ScrubFinding(
                    category="secret",
                    rule_id="dsn-credentials",
                    start=m.start(2),
                    end=m.end(2),
                    fingerprint=_fingerprint(secret_part),
                )
            )
            return f"{m.group(1)}[REDACTED:secret:dsn-credentials]{m.group(3)}"

        current = _DSN_CREDENTIALS_RE.sub(_replace_dsn, current)

        def _replace_bearer(m: re.Match[str]) -> str:
            token_val = m.group(2)
            if "[REDACTED" in token_val or token_val.startswith(("<", "${", "{{")):
                return m.group(0)
            findings.append(
                ScrubFinding(
                    category="secret",
                    rule_id="http-auth-token",
                    start=m.start(2),
                    end=m.end(2),
                    fingerprint=_fingerprint(token_val),
                )
            )
            return f"{m.group(1)}[REDACTED:secret:http-auth-token]"

        current = _BEARER_HEADER_RE.sub(_replace_bearer, current)

        def _replace_assigned(m: re.Match[str]) -> str:
            key_name = m.group(1)
            sep = m.group(2)
            quote = m.group(3)
            secret_val = m.group(4)
            lowered = secret_val.strip().lower()
            uppered = secret_val.strip().upper()
            if (
                lowered in _SAFE_PLACEHOLDER_EXACT
                or any(uppered.startswith(prefix) for prefix in _SAFE_PLACEHOLDER_PREFIXES)
                or len(set(secret_val.strip())) <= 2
            ):
                return m.group(0)
            rule_name = f"assigned-{key_name.lower().replace('_', '-')}"
            findings.append(
                ScrubFinding(
                    category="secret",
                    rule_id=rule_name,
                    start=m.start(4),
                    end=m.end(4),
                    fingerprint=_fingerprint(secret_val),
                )
            )
            return f"{key_name}{sep}{quote}[REDACTED:secret:{rule_name}]{quote}"

        current = _QUOTED_SECRET_ASSIGN_RE.sub(_replace_assigned, current)
        return current, findings

    def _apply_pii_rules(self, text: str) -> tuple[str, list[ScrubFinding]]:
        findings: list[ScrubFinding] = []
        current = text

        def _replace_iban(m: re.Match[str]) -> str:
            matched = m.group(0)
            if not _iban_valid(matched):
                return matched
            findings.append(
                ScrubFinding(
                    category="pii",
                    rule_id="pii-iban",
                    start=m.start(),
                    end=m.end(),
                    fingerprint=_fingerprint(matched),
                )
            )
            return "[REDACTED:pii:iban]"

        current = _IBAN_CANDIDATE_RE.sub(_replace_iban, current)

        def _replace_card(m: re.Match[str]) -> str:
            matched = m.group(0)
            digits = "".join(ch for ch in matched if "0" <= ch <= "9")
            if not _luhn_valid(digits):
                return matched
            findings.append(
                ScrubFinding(
                    category="pii",
                    rule_id="pii-payment-card",
                    start=m.start(),
                    end=m.end(),
                    fingerprint=_fingerprint(digits),
                )
            )
            return "[REDACTED:pii:payment-card]"

        current = _CARD_CANDIDATE_RE.sub(_replace_card, current)

        def _replace_ssn(m: re.Match[str]) -> str:
            matched = m.group(0)
            findings.append(
                ScrubFinding(
                    category="pii",
                    rule_id="pii-ssn",
                    start=m.start(),
                    end=m.end(),
                    fingerprint=_fingerprint(matched),
                )
            )
            return "[REDACTED:pii:ssn]"

        current = _SSN_RE.sub(_replace_ssn, current)

        def _replace_email(m: re.Match[str]) -> str:
            matched = m.group(0)
            domain = matched.rsplit("@", 1)[-1].lower()
            if domain in self._allowlisted_email_domains:
                return matched
            findings.append(
                ScrubFinding(
                    category="pii",
                    rule_id="pii-email",
                    start=m.start(),
                    end=m.end(),
                    fingerprint=_fingerprint(matched),
                )
            )
            return "[REDACTED:pii:email]"

        current = _EMAIL_RE.sub(_replace_email, current)

        def _replace_phone(m: re.Match[str]) -> str:
            matched = m.group(0)
            if _DOTTED_QUAD_RE.match(matched.strip()):
                return matched
            digits = [ch for ch in matched if "0" <= ch <= "9"]
            if not (10 <= len(digits) <= 15):
                return matched
            # Skip date-like YYYY.MM.DD / version strings when '.' is separator
            if "." in matched and not matched.strip().startswith("+"):
                dot_parts = matched.split(".")
                if len(dot_parts) == 3 and len(dot_parts[0]) == 4:
                    return matched
            findings.append(
                ScrubFinding(
                    category="pii",
                    rule_id="pii-phone",
                    start=m.start(),
                    end=m.end(),
                    fingerprint=_fingerprint(matched),
                )
            )
            return "[REDACTED:pii:phone]"

        current = _PHONE_RE.sub(_replace_phone, current)
        return current, findings
