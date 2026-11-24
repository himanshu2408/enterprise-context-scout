"""Pluggable RerankerProviderSPI and domain glossary synonym expansion (M3-01).

Supports:
1. Self-hosted `vLLM` `/v1/rerank`, `/rerank`, `/v1/score`, `/score` and Hugging Face `TEI`
   `/rerank` for `Qwen/Qwen3-Reranker-0.6B/4B` and `BAAI/bge-reranker-v2-m3`.
2. Local `sentence-transformers` `CrossEncoder` execution (`trust_remote_code=False`).
3. Zero-network `LocalDeterministicCrossReranker` combining lexical cross-features,
   `glossary.yaml` synonym expansion, authority weighting, and stale-wiki disambiguation.
"""

from __future__ import annotations

import fnmatch
import logging
import math
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Protocol, runtime_checkable

import httpx
import yaml

from context_scout.config import ModelGovernancePolicy, ScoutSettings

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL: Final[str] = "BAAI/bge-reranker-v2-m3"
DEFAULT_QWEN3_RERANKER_INSTRUCTION: Final[str] = (
    "Given a technical engineering or enterprise documentation search query, "
    "retrieve the most relevant and authoritative code or document passages."
)

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_]{2,}")


def _split_camel_case(segment: str) -> list[str]:
    """Splits a CamelCase / acronym identifier segment into subwords in O(n)."""
    parts: list[str] = []
    n = len(segment)
    i = 0
    while i < n:
        ch = segment[i]
        if not ch.isalnum():
            i += 1
            continue
        start = i
        if ch.isupper():
            i += 1
            if i < n and segment[i].isupper():
                while i < n and segment[i].isupper():
                    i += 1
                if i < n and (segment[i].islower() or segment[i].isdigit()):
                    i -= 1
            else:
                while i < n and (segment[i].islower() or segment[i].isdigit()):
                    i += 1
        else:
            while i < n and (segment[i].islower() or segment[i].isdigit()):
                i += 1
        parts.append(segment[start:i])
    return parts


_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "what",
        "when",
        "where",
        "which",
        "how",
        "does",
        "are",
        "was",
        "were",
        "been",
        "being",
        "have",
        "has",
        "had",
        "will",
        "would",
        "should",
        "could",
        "about",
        "under",
        "over",
        "between",
    }
)

_CROSS_ENCODER_LOCK: Final[threading.Lock] = threading.Lock()
_CROSS_ENCODER_CACHE: dict[tuple[str, int], Any] = {}


@dataclass(frozen=True)
class RerankerModelSpec:
    """Metadata describing an approved cross-encoder reranker model."""

    pattern: str
    family: str
    max_length: int
    supports_instruction: bool
    default_instruction: str | None = None


RERANKER_MODEL_CATALOG: Final[tuple[RerankerModelSpec, ...]] = (
    RerankerModelSpec(
        pattern="Qwen/Qwen3-Reranker-0.6B*",
        family="Qwen3-Reranker",
        max_length=8192,
        supports_instruction=True,
        default_instruction=DEFAULT_QWEN3_RERANKER_INSTRUCTION,
    ),
    RerankerModelSpec(
        pattern="Qwen/Qwen3-Reranker-4B*",
        family="Qwen3-Reranker",
        max_length=8192,
        supports_instruction=True,
        default_instruction=DEFAULT_QWEN3_RERANKER_INSTRUCTION,
    ),
    RerankerModelSpec(
        pattern="Qwen/Qwen3-Reranker-*",
        family="Qwen3-Reranker",
        max_length=8192,
        supports_instruction=True,
        default_instruction=DEFAULT_QWEN3_RERANKER_INSTRUCTION,
    ),
    RerankerModelSpec(
        pattern="BAAI/bge-reranker-v2-m3*",
        family="BAAI/bge-reranker-v2-m3",
        max_length=8192,
        supports_instruction=False,
        default_instruction=None,
    ),
)


def resolve_reranker_model_spec(model_id: str) -> RerankerModelSpec:
    """Validates model_id via ModelGovernancePolicy and resolves reranker metadata."""
    cleaned = ModelGovernancePolicy.default().validate_local_model_id(model_id)
    cleaned_lower = cleaned.lower()
    for spec in RERANKER_MODEL_CATALOG:
        if fnmatch.fnmatchcase(cleaned_lower, spec.pattern.lower()):
            return spec
    return RerankerModelSpec(
        pattern=cleaned,
        family="custom",
        max_length=4096,
        supports_instruction=False,
        default_instruction=None,
    )


def _normalize_score(raw_score: float) -> float:
    """Validates finite score and maps raw logits onto [0.0, 1.0]."""
    val = float(raw_score)
    if not math.isfinite(val):
        raise ValueError(f"Non-finite reranker score received: {raw_score!r}")
    if 0.0 <= val <= 1.0:
        return round(val, 6)
    clamped = max(-40.0, min(40.0, val))
    return round(1.0 / (1.0 + math.exp(-clamped)), 6)


def _tokenize_terms(text: str) -> list[str]:
    """Extracts normalized lexical and subword identifier tokens in O(n)."""
    tokens: list[str] = []
    for raw_tok in _TOKEN_RE.findall(text):
        low = raw_tok.lower()
        if low not in _STOPWORDS:
            tokens.append(low)
        if "_" in raw_tok or any(ch.isupper() for ch in raw_tok[1:]):
            for seg in raw_tok.split("_"):
                if len(seg) >= 2:
                    for sub in _split_camel_case(seg):
                        sub_low = sub.lower()
                        if len(sub_low) >= 2 and sub_low not in _STOPWORDS and sub_low != low:
                            tokens.append(sub_low)
    return tokens


@dataclass(frozen=True)
class GlossaryTermEntry:
    """Single domain glossary term with aliases and canonical expansion tokens."""

    term: str
    definition: str
    aliases: tuple[str, ...] = ()
    expansion_tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class GlossaryExpansionResult:
    """Result of expanding a query using a domain glossary.yaml file."""

    original_query: str
    expanded_query: str
    matched_terms: dict[str, str] = field(default_factory=dict)
    added_synonyms: tuple[str, ...] = ()


class DomainGlossary:
    """Domain-specific jargon & acronym glossary for query expansion and rerank boosting."""

    def __init__(
        self,
        *,
        domain_id: str,
        entries: Sequence[GlossaryTermEntry],
    ) -> None:
        self.domain_id = domain_id
        self._entries: tuple[GlossaryTermEntry, ...] = tuple(entries)
        self._compiled_matchers: tuple[tuple[GlossaryTermEntry, re.Pattern[str]], ...] = tuple(
            (entry, self._compile_entry_matcher(entry)) for entry in self._entries
        )

    @staticmethod
    def _compile_entry_matcher(entry: GlossaryTermEntry) -> re.Pattern[str]:
        variants: list[str] = []
        for candidate in (entry.term, *entry.aliases):
            cleaned = candidate.strip()
            if not cleaned:
                continue
            variants.append(re.escape(cleaned))
            if "_" in cleaned:
                variants.append(re.escape(cleaned.replace("_", " ")))
                variants.append(re.escape(cleaned.replace("_", "-")))
        deduped = sorted(set(variants), key=len, reverse=True)
        joined = "|".join(deduped) if deduped else re.escape(entry.term)
        return re.compile(rf"(?<![A-Za-z0-9_])(?:{joined})(?![A-Za-z0-9_])", re.IGNORECASE)

    @property
    def entries(self) -> tuple[GlossaryTermEntry, ...]:
        return self._entries

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, domain_id: str | None = None) -> DomainGlossary:
        resolved_domain = str(raw.get("domain_id") or domain_id or "default").strip()
        raw_terms = raw.get("terms") or {}
        raw_synonyms = raw.get("synonyms") or {}

        entries: list[GlossaryTermEntry] = []
        if isinstance(raw_terms, Mapping):
            for raw_term, val in raw_terms.items():
                term_str = str(raw_term).strip()
                if not term_str:
                    continue
                aliases: list[str] = []
                definition = ""
                if isinstance(val, str):
                    definition = val.strip()
                elif isinstance(val, list):
                    str_items = [str(item).strip() for item in val if str(item).strip()]
                    aliases.extend(str_items)
                    definition = "; ".join(str_items)
                elif isinstance(val, Mapping):
                    definition = str(val.get("definition") or val.get("summary") or "").strip()
                    for alias_key in ("aliases", "synonyms"):
                        alias_vals = val.get(alias_key) or []
                        if isinstance(alias_vals, list):
                            aliases.extend(str(a).strip() for a in alias_vals if str(a).strip())
                extra_syns = raw_synonyms.get(term_str) if isinstance(raw_synonyms, Mapping) else []
                if isinstance(extra_syns, list):
                    aliases.extend(str(a).strip() for a in extra_syns if str(a).strip())
                expansion_tokens = _extract_expansion_tokens(term_str, definition, aliases)
                entries.append(
                    GlossaryTermEntry(
                        term=term_str,
                        definition=definition,
                        aliases=tuple(dict.fromkeys(aliases)),
                        expansion_tokens=expansion_tokens,
                    )
                )

        if isinstance(raw_synonyms, Mapping):
            known_terms = {e.term.lower() for e in entries}
            for raw_term, syn_list in raw_synonyms.items():
                term_str = str(raw_term).strip()
                if (
                    not term_str
                    or term_str.lower() in known_terms
                    or not isinstance(syn_list, list)
                ):
                    continue
                aliases = [str(s).strip() for s in syn_list if str(s).strip()]
                definition = ", ".join(aliases)
                expansion_tokens = _extract_expansion_tokens(term_str, definition, aliases)
                entries.append(
                    GlossaryTermEntry(
                        term=term_str,
                        definition=definition,
                        aliases=tuple(aliases),
                        expansion_tokens=expansion_tokens,
                    )
                )

        return cls(domain_id=resolved_domain, entries=entries)

    @classmethod
    def from_yaml(cls, path: Path, *, domain_id: str | None = None) -> DomainGlossary:
        if not path.is_file():
            raise FileNotFoundError(f"Domain glossary YAML not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"Invalid glossary YAML structure at: {path}")
        return cls.from_dict(raw, domain_id=domain_id)

    def match_query_entries(self, query: str) -> list[GlossaryTermEntry]:
        if not query or not self._compiled_matchers:
            return []
        matched: list[GlossaryTermEntry] = []
        for entry, pattern in self._compiled_matchers:
            if pattern.search(query):
                matched.append(entry)
        return matched

    def expand_query(self, query: str, *, max_added_terms: int = 8) -> GlossaryExpansionResult:
        """Expands domain jargon/acronyms in query using glossary.yaml terms."""
        cleaned = query.strip()
        if not cleaned:
            return GlossaryExpansionResult(original_query=query, expanded_query=query)

        matched_entries = self.match_query_entries(cleaned)
        if not matched_entries:
            return GlossaryExpansionResult(original_query=query, expanded_query=cleaned)

        existing_lower = set(_tokenize_terms(cleaned))
        matched_map: dict[str, str] = {}
        added: list[str] = []

        for entry in matched_entries:
            matched_map[entry.term] = entry.definition
            if entry.term.lower() not in existing_lower and entry.term not in added:
                added.append(entry.term)
                existing_lower.add(entry.term.lower())
            for alias in entry.aliases:
                if alias.lower() not in existing_lower and alias not in added:
                    added.append(alias)
                    existing_lower.add(alias.lower())
            for tok in entry.expansion_tokens:
                if tok not in existing_lower and tok not in added:
                    added.append(tok)
                    existing_lower.add(tok)
                if len(added) >= max_added_terms:
                    break
            if len(added) >= max_added_terms:
                break

        added_tuple = tuple(added[:max_added_terms])
        if not added_tuple:
            return GlossaryExpansionResult(
                original_query=query,
                expanded_query=cleaned,
                matched_terms=matched_map,
                added_synonyms=(),
            )
        expanded = f"{cleaned} ({', '.join(added_tuple)})"
        return GlossaryExpansionResult(
            original_query=query,
            expanded_query=expanded,
            matched_terms=matched_map,
            added_synonyms=added_tuple,
        )

    def score_candidate_boost(self, query: str, candidate_text: str) -> float:
        """Computes a bounded [0.0, 0.20] lexical affinity boost from matched glossary entries."""
        matched_entries = self.match_query_entries(query)
        if not matched_entries or not candidate_text.strip():
            return 0.0
        cand_lower = candidate_text.lower()
        cand_tokens = set(_tokenize_terms(candidate_text))
        total_boost = 0.0
        for entry in matched_entries:
            if entry.term.lower() in cand_lower:
                total_boost += 0.08
            if any(alias.lower() in cand_lower for alias in entry.aliases):
                total_boost += 0.07
            if entry.expansion_tokens:
                hits = sum(1 for tok in entry.expansion_tokens if tok in cand_tokens)
                total_boost += 0.08 * (hits / len(entry.expansion_tokens))
        return round(min(0.20, total_boost), 6)


def _extract_expansion_tokens(
    term: str,
    definition: str,
    aliases: Sequence[str],
) -> tuple[str, ...]:
    term_low = term.lower()
    collected: list[str] = []
    for source in [*aliases, definition]:
        for tok in _tokenize_terms(source):
            if tok != term_low and len(tok) >= 3 and tok not in collected:
                collected.append(tok)
            if len(collected) >= 8:
                return tuple(collected)
    return tuple(collected)


def load_domain_glossary(
    domain_id: str,
    *,
    domains_dir: Path | None = None,
    glossary_path: str | Path | None = None,
) -> DomainGlossary | None:
    """Loads a domain's glossary.yaml if present (consulting specialization.yaml first)."""
    repo_root = Path(__file__).resolve().parents[3]
    resolved_domains_dir = domains_dir or (repo_root / "domains")

    if glossary_path is not None:
        explicit = Path(glossary_path)
        if not explicit.is_absolute():
            explicit = repo_root / explicit
        if explicit.is_file():
            return DomainGlossary.from_yaml(explicit, domain_id=domain_id)

    domain_dir = resolved_domains_dir / domain_id
    spec_path = domain_dir / "specialization.yaml"
    if spec_path.is_file():
        try:
            spec_raw = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
            if isinstance(spec_raw, Mapping) and spec_raw.get("glossary_path"):
                g_rel = Path(str(spec_raw["glossary_path"]).strip())
                g_cand = g_rel if g_rel.is_absolute() else (repo_root / g_rel)
                if g_cand.is_file():
                    return DomainGlossary.from_yaml(g_cand, domain_id=domain_id)
        except Exception as exc:
            logger.debug(
                "Skipping glossary_path lookup from %s (%s): %s",
                spec_path,
                type(exc).__name__,
                exc,
            )

    default_path = domain_dir / "glossary.yaml"
    if default_path.is_file():
        return DomainGlossary.from_yaml(default_path, domain_id=domain_id)
    return None


@dataclass(frozen=True)
class RerankCandidate:
    """Candidate chunk, symbol, or file passed into RerankerProviderSPI."""

    candidate_id: str
    text: str
    title: str = ""
    prior_score: float = 0.0
    authority_weight: float = 1.0
    is_stale: bool = False
    published_ordinal: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankResult:
    """Cross-encoder reranked candidate with calibrated score and rank delta."""

    candidate_id: str
    rerank_score: float
    calibrated_score: float
    original_rank: int
    new_rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RerankerProviderSPI(Protocol):
    """Protocol for local, self-hosted vLLM/TEI, and deterministic cross-encoder rerankers."""

    @property
    def model_id(self) -> str: ...

    @property
    def provider_name(self) -> str: ...

    def score_pairs(self, query: str, documents: Sequence[str]) -> list[float]: ...

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int | None = None,
        glossary: DomainGlossary | None = None,
    ) -> list[RerankResult]: ...


def _finalize_rerank_results(
    query: str,
    candidates: Sequence[RerankCandidate],
    raw_scores: Sequence[float],
    *,
    top_k: int | None = None,
    glossary: DomainGlossary | None = None,
) -> list[RerankResult]:
    if len(candidates) != len(raw_scores):
        raise ValueError(
            f"Reranker score count mismatch: expected {len(candidates)}, got {len(raw_scores)}"
        )

    dated_ordinals = [int(c.published_ordinal) for c in candidates if int(c.published_ordinal) > 0]
    min_ord = min(dated_ordinals) if len(dated_ordinals) >= 2 else 0
    max_ord = max(dated_ordinals) if len(dated_ordinals) >= 2 else 0
    ord_span = max(365, max_ord - min_ord) if max_ord > min_ord else 0

    scored_rows: list[tuple[int, RerankCandidate, float, float]] = []
    for idx, (cand, raw_s) in enumerate(zip(candidates, raw_scores, strict=True), start=1):
        norm_rerank = _normalize_score(raw_s)
        prior_clamped = max(0.0, min(1.0, float(cand.prior_score)))
        auth_clamped = max(0.2, min(1.5, float(cand.authority_weight)))
        auth_bonus = min(0.08, max(-0.04, (auth_clamped - 1.0) * 0.16))
        stale_penalty = 0.25 if cand.is_stale else 0.0
        recency_bonus = 0.0
        if ord_span > 0 and int(cand.published_ordinal) > 0:
            recency_ratio = 1.0 - min(
                1.0, (max_ord - int(cand.published_ordinal)) / float(ord_span)
            )
            recency_bonus = 0.08 * recency_ratio
        gloss_boost = (
            glossary.score_candidate_boost(query, f"{cand.title}\n{cand.text}")
            if glossary is not None
            else 0.0
        )
        calibrated = (
            0.70 * norm_rerank
            + 0.30 * prior_clamped
            + gloss_boost
            + auth_bonus
            + recency_bonus
            - stale_penalty
        )
        calibrated = round(max(0.0, min(1.0, calibrated)), 6)
        scored_rows.append((idx, cand, norm_rerank, calibrated))

    scored_rows.sort(
        key=lambda item: (
            -item[3],
            -item[2],
            1 if item[1].is_stale else 0,
            -item[1].published_ordinal,
            item[0],
        )
    )

    limited = scored_rows[:top_k] if (top_k is not None and top_k >= 0) else scored_rows
    results: list[RerankResult] = []
    for new_rank, (orig_rank, cand, norm_rerank, calibrated) in enumerate(limited, start=1):
        results.append(
            RerankResult(
                candidate_id=cand.candidate_id,
                rerank_score=norm_rerank,
                calibrated_score=calibrated,
                original_rank=orig_rank,
                new_rank=new_rank,
                metadata=dict(cand.metadata),
            )
        )
    return results


def _extract_scores_from_response(
    payload: Any,
    *,
    expected_count: int,
) -> list[float]:
    """Parses vLLM /rerank, vLLM /score, or TEI /rerank JSON payloads fail-closed."""
    if expected_count == 0:
        return []

    raw_items: list[Any] | None = None
    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, Mapping):
        for key in ("results", "data"):
            val = payload.get(key)
            if isinstance(val, list):
                raw_items = val
                break
        if raw_items is None and isinstance(payload.get("scores"), list):
            scores_list = [float(s) for s in payload["scores"]]
            if len(scores_list) != expected_count:
                raise ValueError(
                    f"Reranker response length mismatch: expected {expected_count}, "
                    f"got {len(scores_list)}"
                )
            for sc in scores_list:
                if not math.isfinite(sc):
                    raise ValueError(f"Non-finite reranker score: {sc!r}")
            return scores_list

    if raw_items is None:
        raise ValueError(f"Unrecognized reranker response payload: {type(payload).__name__}")

    if len(raw_items) != expected_count:
        raise ValueError(
            f"Reranker response item count mismatch: expected {expected_count}, "
            f"got {len(raw_items)}"
        )

    ordered_scores: list[float | None] = [None] * expected_count
    for pos, item in enumerate(raw_items):
        if isinstance(item, (int, float)):
            sc_val = float(item)
            if not math.isfinite(sc_val):
                raise ValueError(f"Non-finite reranker score: {sc_val!r}")
            ordered_scores[pos] = sc_val
            continue
        if not isinstance(item, Mapping):
            raise ValueError(f"Invalid reranker result item type: {type(item).__name__}")
        raw_idx = item.get("index", pos)
        idx = int(raw_idx)
        if idx < 0 or idx >= expected_count:
            raise ValueError(f"Reranker result index {idx} out of range [0, {expected_count})")
        raw_score = item.get("relevance_score", item.get("score"))
        if raw_score is None:
            raise ValueError(f"Reranker result item missing score field: {dict(item)}")
        sc_val = float(raw_score)
        if not math.isfinite(sc_val):
            raise ValueError(f"Non-finite reranker score: {sc_val!r}")
        ordered_scores[idx] = sc_val

    if any(s is None for s in ordered_scores):
        raise ValueError("Reranker response omitted one or more candidate indices.")
    return [float(s) for s in ordered_scores if s is not None]


class VLLMRerankerProvider:
    """Self-hosted vLLM (/v1/rerank, /rerank, /v1/score, /score) and TEI (/rerank) provider."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_RERANKER_MODEL,
        endpoint_url: str,
        api_key: str | None = None,
        api_mode: Literal["auto", "rerank", "score"] = "auto",
        provider_mode: Literal["vllm", "tei"] = "vllm",
        timeout_seconds: float = 10.0,
        instruction: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._spec = resolve_reranker_model_spec(model_id)
        self._model_id = model_id.strip()
        cleaned_url = endpoint_url.strip().rstrip("/")
        if not cleaned_url:
            raise ValueError("endpoint_url must be non-empty for VLLMRerankerProvider.")
        self.endpoint_url = cleaned_url
        self.api_key = api_key
        self.api_mode = api_mode
        self.provider_mode = provider_mode
        self.timeout_seconds = timeout_seconds
        self.instruction = instruction or self._spec.default_instruction
        self._http_client = http_client

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider_name(self) -> str:
        return self.provider_mode

    def _format_query(self, query: str) -> str:
        if self._spec.supports_instruction and self.instruction:
            return f"<Instruct>: {self.instruction}\n<Query>: {query}"
        return query

    def _build_candidate_urls(self, kind: Literal["rerank", "score"]) -> list[str]:
        base = self.endpoint_url
        if base.endswith(f"/{kind}"):
            return [base]
        if base.endswith("/rerank") or base.endswith("/score"):
            root = base.rsplit("/", 1)[0]
            return [f"{root}/{kind}"]
        if base.endswith("/v1"):
            root_no_v1 = base[:-3].rstrip("/")
            return [f"{base}/{kind}", f"{root_no_v1}/{kind}"]
        return [f"{base}/v1/{kind}", f"{base}/{kind}"]

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ContextScout-Reranker",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_json(self, url: str, payload: dict[str, Any]) -> httpx.Response:
        if self._http_client is not None:
            return self._http_client.post(
                url, json=payload, headers=self._headers(), timeout=self.timeout_seconds
            )
        with httpx.Client(timeout=self.timeout_seconds) as client:
            return client.post(url, json=payload, headers=self._headers())

    def _request_rerank(self, formatted_query: str, documents: list[str]) -> list[float]:
        urls = self._build_candidate_urls("rerank")
        if self.provider_mode == "tei":
            payload: dict[str, Any] = {"query": formatted_query, "texts": documents}
        else:
            payload = {
                "model": self._model_id,
                "query": formatted_query,
                "documents": documents,
                "top_n": len(documents),
            }
        last_exc: Exception | None = None
        for idx, url in enumerate(urls):
            resp = self._post_json(url, payload)
            if resp.status_code in {404, 405} and idx + 1 < len(urls):
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code} from {url}",
                    request=resp.request,
                    response=resp,
                )
                continue
            resp.raise_for_status()
            return _extract_scores_from_response(resp.json(), expected_count=len(documents))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No rerank endpoint URL available.")

    def _request_score(self, formatted_query: str, documents: list[str]) -> list[float]:
        urls = self._build_candidate_urls("score")
        payload: dict[str, Any] = {
            "model": self._model_id,
            "text_1": formatted_query,
            "text_2": documents,
        }
        last_exc: Exception | None = None
        for idx, url in enumerate(urls):
            resp = self._post_json(url, payload)
            if resp.status_code in {404, 405} and idx + 1 < len(urls):
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code} from {url}",
                    request=resp.request,
                    response=resp,
                )
                continue
            resp.raise_for_status()
            return _extract_scores_from_response(resp.json(), expected_count=len(documents))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No score endpoint URL available.")

    def score_pairs(self, query: str, documents: Sequence[str]) -> list[float]:
        docs = [str(d) for d in documents]
        if not docs:
            return []
        formatted = self._format_query(query)
        if self.api_mode == "score":
            raw_scores = self._request_score(formatted, docs)
        elif self.api_mode == "rerank" or self.provider_mode == "tei":
            raw_scores = self._request_rerank(formatted, docs)
        else:
            try:
                raw_scores = self._request_rerank(formatted, docs)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in {404, 405}:
                    raise
                raw_scores = self._request_score(formatted, docs)
        return [_normalize_score(s) for s in raw_scores]

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int | None = None,
        glossary: DomainGlossary | None = None,
    ) -> list[RerankResult]:
        if not candidates:
            return []
        expanded_query = glossary.expand_query(query).expanded_query if glossary else query
        docs = [f"{c.title}\n{c.text}".strip() if c.title else c.text for c in candidates]
        raw_scores = self.score_pairs(expanded_query, docs)
        return _finalize_rerank_results(
            query,
            candidates,
            raw_scores,
            top_k=top_k,
            glossary=glossary,
        )


class SentenceTransformersRerankerProvider:
    """Local PyTorch/MPS/CUDA SentenceTransformers CrossEncoder reranker provider."""

    def __init__(self, *, model_id: str = DEFAULT_RERANKER_MODEL) -> None:
        self._spec = resolve_reranker_model_spec(model_id)
        self._model_id = model_id.strip()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider_name(self) -> str:
        return "sentence-transformers"

    def _get_model(self) -> Any:
        cache_key = (self._model_id, self._spec.max_length)
        with _CROSS_ENCODER_LOCK:
            cached = _CROSS_ENCODER_CACHE.get(cache_key)
            if cached is not None:
                return cached
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(
                self._model_id,
                max_length=self._spec.max_length,
                trust_remote_code=False,
            )
            _CROSS_ENCODER_CACHE[cache_key] = model
            return model

    def score_pairs(self, query: str, documents: Sequence[str]) -> list[float]:
        docs = [str(d) for d in documents]
        if not docs:
            return []
        formatted = (
            f"<Instruct>: {self._spec.default_instruction}\n<Query>: {query}"
            if self._spec.supports_instruction and self._spec.default_instruction
            else query
        )
        model = self._get_model()
        pairs = [(formatted, doc) for doc in docs]
        raw_preds = model.predict(pairs)
        scores = [float(val) for val in list(raw_preds)]
        if len(scores) != len(docs):
            raise ValueError(
                f"CrossEncoder score mismatch: expected {len(docs)}, got {len(scores)}"
            )
        return [_normalize_score(s) for s in scores]

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int | None = None,
        glossary: DomainGlossary | None = None,
    ) -> list[RerankResult]:
        if not candidates:
            return []
        expanded_query = glossary.expand_query(query).expanded_query if glossary else query
        docs = [f"{c.title}\n{c.text}".strip() if c.title else c.text for c in candidates]
        raw_scores = self.score_pairs(expanded_query, docs)
        return _finalize_rerank_results(
            query,
            candidates,
            raw_scores,
            top_k=top_k,
            glossary=glossary,
        )


class LocalDeterministicCrossReranker:
    """Zero-network deterministic cross-attention feature reranker with glossary awareness."""

    def __init__(self, *, model_id: str = DEFAULT_RERANKER_MODEL) -> None:
        self._spec = resolve_reranker_model_spec(model_id)
        self._model_id = model_id.strip()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider_name(self) -> str:
        return "local-cross-feature"

    def _score_pair_raw(self, query: str, doc_text: str) -> float:
        q_tokens = _tokenize_terms(query)
        d_tokens = _tokenize_terms(doc_text)
        if not q_tokens or not d_tokens:
            return 0.0

        q_set = set(q_tokens)
        d_set = set(d_tokens)
        overlap = q_set & d_set
        recall = len(overlap) / len(q_set)
        precision = len(overlap) / min(len(d_set), max(len(q_set) * 3, 1))
        f1 = (2.0 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0

        # Bigram proximity captures multi-word phrase alignment better than bag-of-words RRF.
        q_bigrams = {(q_tokens[i], q_tokens[i + 1]) for i in range(len(q_tokens) - 1)}
        d_bigrams = {(d_tokens[i], d_tokens[i + 1]) for i in range(len(d_tokens) - 1)}
        bigram_overlap = (len(q_bigrams & d_bigrams) / len(q_bigrams)) if q_bigrams else recall

        # Exact camelCase / snake_case identifier hits in raw text.
        raw_idents = [
            tok
            for tok in _TOKEN_RE.findall(query)
            if "_" in tok or any(ch.isupper() for ch in tok[1:]) or tok.isupper()
        ]
        doc_lower = doc_text.lower()
        ident_bonus = (
            sum(1 for ident in raw_idents if ident.lower() in doc_lower) / len(raw_idents)
            if raw_idents
            else recall
        )

        # Penalize explicit deprecation / superseded / archival disclaimers inside candidate body.
        deprecation_penalty = 0.0
        query_lower = query.lower()
        for marker in (
            "deprecated",
            "superseded",
            "legacy policy",
            "legacy architecture",
            "configuration archive",
            "historical notes",
            "do not use",
            "obsolete",
            "archived",
        ):
            if marker in doc_lower and marker not in query_lower:
                deprecation_penalty = 0.22
                break

        raw = 0.45 * recall + 0.20 * f1 + 0.20 * bigram_overlap + 0.15 * ident_bonus
        return round(max(0.0, min(1.0, raw - deprecation_penalty)), 6)

    def score_pairs(self, query: str, documents: Sequence[str]) -> list[float]:
        return [self._score_pair_raw(query, str(doc)) for doc in documents]

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int | None = None,
        glossary: DomainGlossary | None = None,
    ) -> list[RerankResult]:
        if not candidates:
            return []
        expanded_query = glossary.expand_query(query).expanded_query if glossary else query
        docs = [f"{c.title}\n{c.text}".strip() if c.title else c.text for c in candidates]
        raw_scores = self.score_pairs(expanded_query, docs)
        return _finalize_rerank_results(
            query,
            candidates,
            raw_scores,
            top_k=top_k,
            glossary=glossary,
        )


def get_reranker_provider(
    settings: ScoutSettings | None = None,
    *,
    model_id: str | None = None,
    provider_mode: Literal["local", "vllm", "tei", "sentence-transformers"] | None = None,
    endpoint_url: str | None = None,
    api_key: str | None = None,
    api_mode: Literal["auto", "rerank", "score"] | None = None,
    http_client: httpx.Client | None = None,
) -> RerankerProviderSPI:
    """Factory resolving configured RerankerProviderSPI under ModelGovernancePolicy."""
    cfg = settings or ScoutSettings()
    resolved_model = model_id or cfg.reranker_model_id
    resolved_mode = provider_mode or cfg.reranker_provider
    resolved_endpoint = endpoint_url if endpoint_url is not None else cfg.reranker_endpoint_url
    resolved_api_key = api_key if api_key is not None else cfg.reranker_api_key
    resolved_api_mode = api_mode or cfg.reranker_api_mode

    if resolved_mode in {"vllm", "tei"} or (resolved_endpoint and resolved_endpoint.strip()):
        if not resolved_endpoint or not resolved_endpoint.strip():
            raise ValueError(
                "reranker_endpoint_url must be configured when "
                f"reranker_provider='{resolved_mode}'."
            )
        http_provider_mode: Literal["vllm", "tei"] = "tei" if resolved_mode == "tei" else "vllm"
        return VLLMRerankerProvider(
            model_id=resolved_model,
            endpoint_url=resolved_endpoint,
            api_key=resolved_api_key,
            api_mode=resolved_api_mode,
            provider_mode=http_provider_mode,
            timeout_seconds=cfg.reranker_timeout_seconds,
            http_client=http_client,
        )

    if resolved_mode == "sentence-transformers":
        return SentenceTransformersRerankerProvider(model_id=resolved_model)

    return LocalDeterministicCrossReranker(model_id=resolved_model)


__all__ = [
    "DEFAULT_QWEN3_RERANKER_INSTRUCTION",
    "DEFAULT_RERANKER_MODEL",
    "RERANKER_MODEL_CATALOG",
    "DomainGlossary",
    "GlossaryExpansionResult",
    "GlossaryTermEntry",
    "LocalDeterministicCrossReranker",
    "RerankCandidate",
    "RerankResult",
    "RerankerModelSpec",
    "RerankerProviderSPI",
    "SentenceTransformersRerankerProvider",
    "VLLMRerankerProvider",
    "get_reranker_provider",
    "load_domain_glossary",
    "resolve_reranker_model_spec",
]
