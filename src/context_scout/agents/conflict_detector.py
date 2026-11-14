"""Domain-neutral conflict detector for Confluence pages and live HEAD code vs docs (Step 2.2)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date

from context_scout.db.schema import DocChunkRecord
from context_scout.schemas.contracts import VerbatimCodeSlice

# Concepts commonly specified in both runbooks/ADRs and runtime code configuration.
_TOPIC_CONCEPTS: tuple[str, ...] = (
    "timeout",
    "retries",
    "retry",
    "sla",
    "uptime",
    "availability",
    "latency",
    "backoff",
    "ttl",
    "cache",
    "pool",
    "connections",
    "concurrency",
    "workers",
    "threads",
    "port",
    "rate_limit",
    "batch_size",
    "replicas",
    "retention",
    "interval",
    "window",
    "threshold",
)

_CONCEPT_ALIASES: dict[str, str] = {
    "retry": "retries",
    "max_retry": "retries",
    "max_retries": "retries",
    "retry_count": "retries",
    "retries": "retries",
    "request_timeout": "timeout",
    "connect_timeout": "timeout",
    "read_timeout": "timeout",
    "timeout": "timeout",
    "cache_ttl": "ttl",
    "ttl": "ttl",
    "pool_size": "pool",
    "max_pool_size": "pool",
    "db_pool": "pool",
    "pool": "pool",
    "max_connections": "connections",
    "connections": "connections",
    "rate_limit": "rate_limit",
    "ratelimit": "rate_limit",
    "batch_size": "batch_size",
    "uptime": "sla",
    "availability": "sla",
    "sla": "sla",
}

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "over",
        "under",
        "use",
        "uses",
        "using",
        "default",
        "config",
        "configuration",
        "service",
        "guide",
        "overview",
        "runbook",
        "architecture",
        "policy",
        "spec",
        "page",
        "section",
    }
)

_DURATION_CONCEPTS: frozenset[str] = frozenset(
    {"timeout", "ttl", "latency", "backoff", "interval", "window"}
)

_KV_NUMERIC_PATTERN = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z0-9_.-]{1,48})\s*(?:=|:|\bis\b|\bof\b|\bset\s+to\b|\bdefaults?\s+to\b)\s*"
    r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>ms|msec|milliseconds?|s|sec|secs|seconds?|m|min|mins|minutes?|h|hr|hrs|hours?|days?|%|req/s|rps)?\b",
    re.IGNORECASE,
)
_REVERSE_CONCEPT_PATTERN = re.compile(
    r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>ms|msec|milliseconds?|s|sec|secs|seconds?|m|min|mins|minutes?|h|hr|hrs|hours?|days?|%|req/s|rps)\s+"
    r"(?P<key>timeout|retries|retry|sla|uptime|availability|latency|ttl|pool|port|retention|backoff)\b",
    re.IGNORECASE,
)
_ENDPOINT_PATTERN = re.compile(r"(?<![A-Za-z0-9_])/(?:api|v[0-9]+|internal|health)[A-Za-z0-9_./-]*")
_ENV_CODE_PATTERN = re.compile(
    r"(?:os\.getenv|os\.environ\.get|environ\.get|process\.env\.)\s*\(?\s*['\"]?"
    r"(?P<var>[A-Z][A-Z0-9_]{2,})['\"]?(?:\s*,\s*['\"](?P<default>[^'\"]+)['\"])?",
)
_ENV_ASSIGN_PATTERN = re.compile(
    r"\b(?P<var>[A-Z][A-Z0-9_]{3,})\s*(?:=|:|\bdefaults?\s+to\b)\s*['\"]?(?P<default>[A-Za-z0-9_./:-]+)['\"]?"
)
_CODE_LINE_PREFIX = re.compile(r"^\s*\d+\s*:\s?")
_SPACE_HEADER_RE = re.compile(r"space\s+'([^']+)'", re.IGNORECASE)
_SPACE_URL_RE = re.compile(r"/spaces/([^/]+)/")
_DATE_HEADER_RE = re.compile(r"Last Modified:\s*(\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class _NumericAssertion:
    canonical_key: str
    normalized_value: float
    unit_kind: str
    display_value: str


def _normalize_key(raw_key: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", raw_key.strip().lower()).strip("_")
    for suffix in ("_ms", "_msec", "_sec", "_secs", "_seconds", "_s", "_min", "_mins"):
        if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    if cleaned in _CONCEPT_ALIASES:
        return _CONCEPT_ALIASES[cleaned]
    for alias_key, canonical in _CONCEPT_ALIASES.items():
        if cleaned.endswith(f"_{alias_key}") or cleaned.startswith(f"{alias_key}_"):
            return canonical
    return cleaned


def _unit_from_key_hint(raw_key: str, raw_unit: str | None) -> str:
    if raw_unit:
        return raw_unit.lower()
    lower = raw_key.lower()
    if lower.endswith(("_ms", "_msec")):
        return "ms"
    if lower.endswith(("_sec", "_secs", "_seconds")):
        return "s"
    if lower.endswith(("_min", "_mins", "_minutes")):
        return "m"
    return ""


def _normalize_numeric(
    canonical_key: str,
    raw_value: str,
    raw_key: str,
    raw_unit: str | None,
) -> _NumericAssertion:
    value = float(raw_value)
    unit = _unit_from_key_hint(raw_key, raw_unit)
    display_unit = raw_unit or unit

    if canonical_key in _DURATION_CONCEPTS and unit in {
        "ms",
        "msec",
        "millisecond",
        "milliseconds",
        "s",
        "sec",
        "secs",
        "second",
        "seconds",
        "m",
        "min",
        "mins",
        "minute",
        "minutes",
        "h",
        "hr",
        "hrs",
        "hour",
        "hours",
    }:
        if unit in {"ms", "msec", "millisecond", "milliseconds"}:
            seconds = value / 1000.0
        elif unit in {"m", "min", "mins", "minute", "minutes"}:
            seconds = value * 60.0
        elif unit in {"h", "hr", "hrs", "hour", "hours"}:
            seconds = value * 3600.0
        else:
            seconds = value
        disp = f"{raw_value}{display_unit}" if display_unit else raw_value
        return _NumericAssertion(
            canonical_key=canonical_key,
            normalized_value=seconds,
            unit_kind="seconds",
            display_value=disp,
        )

    disp = f"{raw_value}{display_unit}" if display_unit else raw_value
    return _NumericAssertion(
        canonical_key=canonical_key,
        normalized_value=value,
        unit_kind=unit or "scalar",
        display_value=disp,
    )


def _extract_numeric_assertions(text: str) -> dict[str, _NumericAssertion]:
    assertions: dict[str, _NumericAssertion] = {}
    for match in _KV_NUMERIC_PATTERN.finditer(text):
        raw_key = match.group("key")
        canonical = _normalize_key(raw_key)
        if not canonical or canonical in _STOPWORDS:
            continue
        if canonical not in _TOPIC_CONCEPTS and not any(
            tok in canonical for tok in _TOPIC_CONCEPTS
        ):
            continue
        assertion = _normalize_numeric(
            canonical_key=canonical,
            raw_value=match.group("val"),
            raw_key=raw_key,
            raw_unit=match.group("unit"),
        )
        assertions.setdefault(canonical, assertion)

    for match in _REVERSE_CONCEPT_PATTERN.finditer(text):
        canonical = _normalize_key(match.group("key"))
        if canonical in assertions:
            continue
        assertions[canonical] = _normalize_numeric(
            canonical_key=canonical,
            raw_value=match.group("val"),
            raw_key=match.group("key"),
            raw_unit=match.group("unit"),
        )
    return assertions


def _topic_tokens(title: str, section: str, body: str) -> set[str]:
    title_words = {
        tok
        for tok in re.findall(r"[a-z0-9]{3,}", f"{title} {section}".lower())
        if tok not in _STOPWORDS
    }
    body_concepts = {tok for tok in _TOPIC_CONCEPTS if re.search(rf"\b{tok}\b", body.lower())}
    return title_words | body_concepts


def _chunk_date(chunk: DocChunkRecord) -> date | None:
    if chunk.published_date is not None:
        return chunk.published_date
    meta_mod = chunk.metadata_json.get("last_modified_at") if chunk.metadata_json else None
    if isinstance(meta_mod, str) and len(meta_mod) >= 10:
        try:
            return date.fromisoformat(meta_mod[:10])
        except ValueError:
            pass
    hdr_match = _DATE_HEADER_RE.search(chunk.contextual_header)
    if hdr_match:
        try:
            return date.fromisoformat(hdr_match.group(1))
        except ValueError:
            pass
    return None


def _chunk_space_key(chunk: DocChunkRecord) -> str:
    if chunk.metadata_json:
        raw = chunk.metadata_json.get("space_key")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    hdr_match = _SPACE_HEADER_RE.search(chunk.contextual_header)
    if hdr_match:
        return hdr_match.group(1)
    url_match = _SPACE_URL_RE.search(chunk.source_uri)
    if url_match:
        return url_match.group(1)
    return "UNKNOWN"


def _format_doc_citation(chunk: DocChunkRecord) -> str:
    title = chunk.doc_title or chunk.doc_id
    section = chunk.section_heading or "Overview"
    space_key = _chunk_space_key(chunk)
    dt = _chunk_date(chunk)
    date_str = dt.isoformat() if dt is not None else "unknown"
    return f"[{title} > {section} | Space: {space_key} | Last Updated: {date_str}]"


def _authority_rank_key(chunk: DocChunkRecord) -> tuple[float, int, int]:
    dt = _chunk_date(chunk)
    dt_ord = dt.toordinal() if dt is not None else 0
    ver = int(chunk.version_str) if chunk.version_str and chunk.version_str.isdigit() else 0
    return (float(chunk.authority_weight), dt_ord, ver)


def _strip_code_line_numbers(code: str) -> str:
    return "\n".join(_CODE_LINE_PREFIX.sub("", line) for line in code.splitlines())


def _endpoint_stem(path: str) -> str:
    segments = [seg for seg in path.lower().split("/") if seg]
    non_version = [seg for seg in segments if not re.fullmatch(r"v\d+", seg)]
    return non_version[-1] if non_version else path.lower()


def _is_technical_doc(chunk: DocChunkRecord) -> bool:
    source_ok = chunk.source_type in {
        "confluence",
        "enterprise_doc",
        "internal_adr",
        "adr",
        "rfc",
        "markdown",
    }
    if source_ok:
        return True
    title_lower = chunk.doc_title.lower()
    return any(tok in title_lower for tok in ("adr", "rfc", "runbook", "architecture", "spec"))


class ConflictDetector:
    """Detects cross-page Confluence conflicts and live HEAD code vs documentation drift."""

    def detect_confluence_page_conflicts(
        self,
        chunks: list[DocChunkRecord],
        max_age_divergence_days: int = 365,
    ) -> list[str]:
        """Detects numeric/SLA/config conflicts or >=365-day age divergence across Confluence pages.

        Prefers higher-authority and more recently updated pages, emitting clear warning
        strings formatted with ``[Page Title > Section | Space: <KEY> | Last Updated: YYYY-MM-DD]``.
        """
        active_chunks = [
            c
            for c in chunks
            if not c.is_stale
            and (c.source_type == "confluence" or c.doc_id.startswith("confluence:"))
        ]
        if len(active_chunks) < 2:
            return []

        warnings: list[str] = []
        seen_pairs: set[tuple[str, str, str]] = set()

        extracted: list[
            tuple[DocChunkRecord, set[str], dict[str, _NumericAssertion], date | None]
        ] = [
            (
                chunk,
                _topic_tokens(chunk.doc_title, chunk.section_heading, chunk.verbatim_content),
                _extract_numeric_assertions(chunk.verbatim_content),
                _chunk_date(chunk),
            )
            for chunk in active_chunks
        ]

        for idx in range(len(extracted)):
            c1, toks1, nums1, dt1 = extracted[idx]
            for jdx in range(idx + 1, len(extracted)):
                c2, toks2, nums2, dt2 = extracted[jdx]
                if c1.doc_id == c2.doc_id:
                    continue

                shared_topics = toks1 & toks2
                # Require meaningful shared topic context (title/section or domain concepts)
                title_overlap = {
                    t
                    for t in re.findall(r"[a-z0-9]{3,}", c1.doc_title.lower())
                    if t not in _STOPWORDS
                } & {
                    t
                    for t in re.findall(r"[a-z0-9]{3,}", c2.doc_title.lower())
                    if t not in _STOPWORDS
                }

                # Sort so preferred/authoritative page comes first
                if _authority_rank_key(c1) >= _authority_rank_key(c2):
                    preferred, other = c1, c2
                else:
                    preferred, other = c2, c1

                pref_cite = _format_doc_citation(preferred)
                other_cite = _format_doc_citation(other)

                # 1. Numeric / SLA / Config conflict check on shared keys
                shared_numeric_keys = set(nums1) & set(nums2)
                for key in sorted(shared_numeric_keys):
                    a1 = nums1[key]
                    a2 = nums2[key]
                    if a1.unit_kind == a2.unit_kind and not math.isclose(
                        a1.normalized_value, a2.normalized_value, rel_tol=1e-6, abs_tol=1e-9
                    ):
                        id_a, id_b = sorted((c1.doc_id, c2.doc_id))
                        pair_key = (id_a, id_b, f"num:{key}")
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)
                        pref_val = a1.display_value if preferred is c1 else a2.display_value
                        other_val = a2.display_value if preferred is c1 else a1.display_value
                        warnings.append(
                            f"CONFLUENCE_CONFIG_CONFLICT ({key}): Prefer {pref_cite} "
                            f"({pref_val}) over conflicting {other_cite} ({other_val})."
                        )

                # 2. Age divergence check on matching topic pages (>= max_age_divergence_days)
                if dt1 is not None and dt2 is not None:
                    age_gap_days = abs((dt1 - dt2).days)
                    has_matching_topic = bool(title_overlap) or len(shared_topics) >= 3
                    if age_gap_days >= max_age_divergence_days and has_matching_topic:
                        id_a, id_b = sorted((c1.doc_id, c2.doc_id))
                        pair_age_key = (id_a, id_b, "age")
                        if pair_age_key not in seen_pairs:
                            seen_pairs.add(pair_age_key)
                            warnings.append(
                                f"CONFLUENCE_AGE_DIVERGENCE ({age_gap_days} days >= "
                                f"{max_age_divergence_days} days): Prefer newer/authoritative "
                                f"{pref_cite} over potentially outdated {other_cite}."
                            )

        return warnings

    def detect_code_vs_doc_conflicts(
        self,
        code_slices: list[VerbatimCodeSlice],
        doc_chunks: list[DocChunkRecord],
    ) -> list[str]:
        """Compares live HEAD code slices against technical docs/ADRs/RFCs.

        Live HEAD implementation wins the actual runtime state; conflicting doc values emit
        warnings for ``ImplementationContextPack.doc_vs_code_conflicts_detected``.
        """
        active_docs = [c for c in doc_chunks if not c.is_stale and _is_technical_doc(c)]
        if not code_slices or not active_docs:
            return []

        warnings: list[str] = []
        seen_keys: set[tuple[str, str, str]] = set()

        for slc in code_slices:
            clean_code = _strip_code_line_numbers(slc.verbatim_code_with_lines)
            code_nums = _extract_numeric_assertions(clean_code)
            code_endpoints = set(_ENDPOINT_PATTERN.findall(clean_code))
            code_envs: dict[str, str | None] = {}
            for m in _ENV_CODE_PATTERN.finditer(clean_code):
                code_envs[m.group("var")] = m.group("default")
            for m in _ENV_ASSIGN_PATTERN.finditer(clean_code):
                code_envs.setdefault(m.group("var"), m.group("default"))

            code_ref = (
                f"{slc.file_path}:{slc.start_line}-{slc.end_line} "
                f"({slc.qualified_symbol}@{slc.git_commit_sha[:8]})"
            )

            for doc in active_docs:
                doc_cite = _format_doc_citation(doc)
                doc_text = doc.verbatim_content

                # 1. Numeric / timeout / retry / pool / port constants
                doc_nums = _extract_numeric_assertions(doc_text)
                for key in sorted(set(code_nums) & set(doc_nums)):
                    c_val = code_nums[key]
                    d_val = doc_nums[key]
                    if c_val.unit_kind == d_val.unit_kind and not math.isclose(
                        c_val.normalized_value, d_val.normalized_value, rel_tol=1e-6, abs_tol=1e-9
                    ):
                        sig = (code_ref, doc.doc_id, f"num:{key}")
                        if sig not in seen_keys:
                            seen_keys.add(sig)
                            warnings.append(
                                f"CODE_VS_DOC_CONFLICT ({key}): Live HEAD code at {code_ref} "
                                f"defines {c_val.display_value}, overriding documented "
                                f"{d_val.display_value} in {doc_cite}. "
                                "HEAD code wins implementation state."
                            )

                # 2. Endpoint path drift (same stem, different path/version)
                doc_endpoints = set(_ENDPOINT_PATTERN.findall(doc_text))
                if code_endpoints and doc_endpoints:
                    doc_by_stem = {_endpoint_stem(ep): ep for ep in doc_endpoints}
                    for c_ep in sorted(code_endpoints):
                        stem = _endpoint_stem(c_ep)
                        d_ep = doc_by_stem.get(stem)
                        if d_ep and d_ep != c_ep:
                            sig = (code_ref, doc.doc_id, f"endpoint:{stem}")
                            if sig not in seen_keys:
                                seen_keys.add(sig)
                                warnings.append(
                                    f"CODE_VS_DOC_ENDPOINT_DRIFT ({stem}): Live HEAD code at "
                                    f"{code_ref} exposes '{c_ep}', whereas {doc_cite} "
                                    f"documents '{d_ep}'. HEAD code wins implementation state."
                                )

                # 3. Environment variable default / name divergence
                doc_envs: dict[str, str | None] = {
                    m.group("var"): m.group("default")
                    for m in _ENV_ASSIGN_PATTERN.finditer(doc_text)
                }
                doc_env_mentions = set(re.findall(r"\b[A-Z][A-Z0-9_]{3,}\b", doc_text))
                for env_var, c_default in sorted(code_envs.items()):
                    if env_var in doc_envs and c_default is not None:
                        d_default = doc_envs[env_var]
                        if d_default is not None and d_default != c_default:
                            sig = (code_ref, doc.doc_id, f"env:{env_var}")
                            if sig not in seen_keys:
                                seen_keys.add(sig)
                                warnings.append(
                                    f"CODE_VS_DOC_ENV_CONFLICT ({env_var}): Live HEAD code at "
                                    f"{code_ref} uses default '{c_default}', whereas {doc_cite} "
                                    f"states '{d_default}'. HEAD code wins implementation state."
                                )
                    elif c_default is None and env_var not in doc_env_mentions:
                        # Check for renamed env var sharing the trailing or leading prefix token
                        code_parts = set(env_var.split("_"))
                        for d_var in sorted(doc_env_mentions):
                            d_parts = set(d_var.split("_"))
                            if (
                                len(code_parts & d_parts) >= 2
                                and env_var != d_var
                                and len(d_var) >= 6
                            ):
                                sig = (code_ref, doc.doc_id, f"env_rename:{env_var}:{d_var}")
                                if sig not in seen_keys:
                                    seen_keys.add(sig)
                                    warnings.append(
                                        f"CODE_VS_DOC_ENV_DRIFT: Live HEAD code at {code_ref} "
                                        f"reads env '{env_var}', whereas {doc_cite} references "
                                        f"'{d_var}'. HEAD code wins implementation state."
                                    )

        return warnings
