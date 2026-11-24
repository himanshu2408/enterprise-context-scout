"""Data Engineering Concepts Domain Plugin & Realism Verification Gate.

Provides domain-specific heuristics, multi-chunk conceptual synthesis,
teacher-student distillation templates, and the 6-check deterministic
`DataEngineeringRealismGate` composed with `DocumentCitationGate`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from context_scout.agents.citation_gate import (
    DocumentCitationGate,
    DocumentCitationGateResult,
)
from context_scout.db.repositories import DocChunkRecord
from context_scout.ingestion.doc_parser import DocumentChunk
from context_scout.schemas.contracts import (
    DocCitationSpan,
    DocumentScope,
    DomainKnowledgeBrief,
    build_domain_knowledge_brief_prompt,
    parse_domain_knowledge_brief_json,
)

logger = logging.getLogger(__name__)

DOMAIN_ID_DATA_ENG: Final[str] = "data_engineering_scout"

DATA_ENG_REALISM_CHECK_NAMES: Final[tuple[str, ...]] = (
    "EOS_WITHOUT_SINK_IDEMPOTENCY",
    "UNPARTITIONED_SCAN_RISK",
    "ICEBERG_EQUALITY_DELETE_UNCOMPACTED",
    "SCD2_NON_ATOMIC_OR_OVERLAP",
    "CDC_SLOT_WAL_UNBOUNDED",
    "UNVERIFIED_THROUGHPUT_CLAIM",
)

_BENCHMARK_CLAIM_RE: Final[re.Pattern[str]] = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:x\s+faster|gb/s|mb/s|rows/s|events/s|ms\s+p99|million\s+rows)\b",
    re.IGNORECASE,
)

_FAILURE_TERMS: Final[tuple[str, ...]] = (
    "failure",
    "spill",
    "oom",
    "amplification",
    "skew",
    "wal",
    "bloat",
    "duplicate",
    "overlap",
    "lag",
    "backpressure",
    "stall",
    "corrupt",
    "stale",
    "out-of-order",
    "tombstone",
)

_CAVEAT_TERMS: Final[tuple[str, ...]] = (
    "require",
    "must",
    "trade-off",
    "tradeoff",
    "caveat",
    "bound",
    "idempoten",
    "compact",
    "partition",
    "watermark",
    "checkpoint",
    "retention",
    "pruning",
    "atomic",
    "isolation",
)


@dataclass(frozen=True)
class DataEngineeringGateResult:
    """Outcome of the 6-check deterministic DataEngineeringRealismGate."""

    passed: bool
    checks_evaluated: int
    failed_checks: list[str]
    violations: list[str]
    citations_valid: bool
    citations_verified_count: int
    failed_citations: list[tuple[DocCitationSpan, str]] = field(default_factory=list)
    citation_gate_result: DocumentCitationGateResult | None = None
    reason: str = ""


class DataEngineeringRealismGate:
    """Deterministic 6-check verification gate for Data Engineering architectural briefs.

    Composes domain-specific invariants with `DocumentCitationGate` verbatim verification.
    Never early-exits as passing on empty or ungrounded inputs.
    """

    def __init__(
        self,
        *,
        domain_id: str = DOMAIN_ID_DATA_ENG,
        citation_gate: DocumentCitationGate | None = None,
    ) -> None:
        self.domain_id = domain_id
        self.citation_gate = citation_gate or DocumentCitationGate(
            min_overlap_ratio=0.98,
            require_citations_default=True,
            strict_freshness=False,
        )

    def evaluate_domain_rules(self, brief: DomainKnowledgeBrief) -> DataEngineeringGateResult:
        """Evaluates all 6 Data Engineering architectural & operational invariant checks."""
        if (
            not brief.brief_title.strip()
            or not brief.executive_summary.strip()
            or not brief.core_findings_and_rules
            or not brief.verified_citations
        ):
            return DataEngineeringGateResult(
                passed=False,
                checks_evaluated=len(DATA_ENG_REALISM_CHECK_NAMES),
                failed_checks=["EMPTY_OR_UNGROUNDED_BRIEF"],
                violations=[
                    "Brief must contain non-empty title, summary, findings, and verified_citations."
                ],
                citations_valid=False,
                citations_verified_count=0,
                reason="EMPTY_OR_UNGROUNDED_BRIEF: missing required brief synthesis or citations.",
            )

        corpus_text = "\n".join(
            [
                brief.brief_title,
                brief.executive_summary,
                brief.formal_expression_or_rule or "",
                *brief.core_findings_and_rules,
                *brief.domain_constraints_and_caveats,
                *brief.known_risks_and_failure_modes,
            ]
        ).lower()
        quote_text = " ".join(c.verbatim_quote.lower() for c in brief.verified_citations)
        combined = f"{corpus_text}\n{quote_text}"

        failed_checks: list[str] = []
        violations: list[str] = []

        def _has_guard(text: str, guards: tuple[str, ...]) -> bool:
            # Word-boundary left edge prevents "unpartitioned"/"non-idempotent" substring bypasses
            return any(re.search(rf"\b{re.escape(g)}", text) for g in guards)

        # 1. EOS_WITHOUT_SINK_IDEMPOTENCY
        eos_triggers = (
            "exactly-once",
            "exactly once",
            "end-to-end guarantee",
            "two-phase commit",
        )
        eos_guards = (
            "idempoten",
            "dedup",
            "upsert",
            "transactional sink",
            "outbox",
            "atomic commit",
            "compensat",
            "unique key",
        )
        if any(t in corpus_text for t in eos_triggers) and not _has_guard(combined, eos_guards):
            failed_checks.append("EOS_WITHOUT_SINK_IDEMPOTENCY")
            violations.append(
                "Streaming exactly-once claims require sink idempotency, deduplication, "
                "or transactional commit caveats."
            )

        # 2. UNPARTITIONED_SCAN_RISK
        scan_triggers = (
            "full scan",
            "table scan",
            "unpartitioned",
            "wide scan",
            "warehouse scan",
            "shuffle join",
        )
        scan_guards = (
            "partition",
            "prun",
            "z-order",
            "liquid cluster",
            "predicate pushdown",
            "zone map",
            "sort key",
            "broadcast",
            "cluster key",
        )
        if any(t in corpus_text for t in scan_triggers) and not _has_guard(combined, scan_guards):
            failed_checks.append("UNPARTITIONED_SCAN_RISK")
            violations.append(
                "Large analytical scan/join patterns must specify partition pruning, "
                "clustering, or predicate pushdown guards."
            )

        # 3. ICEBERG_EQUALITY_DELETE_UNCOMPACTED
        eq_delete_triggers = (
            "equality delete",
            "equality-delete",
            "merge-on-read",
            "deletion vector",
        )
        eq_delete_guards = (
            "compact",
            "rewrite_data_files",
            "read amplification",
            "copy-on-write",
            "position delete",
            "expire_snapshots",
            "vacuum",
        )
        if any(t in corpus_text for t in eq_delete_triggers) and not _has_guard(
            combined, eq_delete_guards
        ):
            failed_checks.append("ICEBERG_EQUALITY_DELETE_UNCOMPACTED")
            violations.append(
                "Merge-on-read / equality delete designs must address compaction "
                "(`rewrite_data_files`) and read amplification."
            )

        # 4. SCD2_NON_ATOMIC_OR_OVERLAP
        scd2_triggers = (
            "scd2",
            "scd type 2",
            "slowly changing dimension",
            "dbt snapshot",
        )
        scd2_guards = (
            "atomic",
            "merge",
            "transaction",
            "non-overlapping",
            "valid_to",
            "effective_to",
            "is_current",
            "idempotent",
            "check_cols",
        )
        if any(t in corpus_text for t in scd2_triggers) and not _has_guard(combined, scd2_guards):
            failed_checks.append("SCD2_NON_ATOMIC_OR_OVERLAP")
            violations.append(
                "SCD Type 2 designs must enforce atomic MERGE/transaction boundaries "
                "and non-overlapping validity windows."
            )

        # 5. CDC_SLOT_WAL_UNBOUNDED
        cdc_triggers = (
            "replication slot",
            "debezium",
            "pgoutput",
            "logical replication",
            "change data capture",
        )
        cdc_guards = (
            "max_slot_wal_keep_size",
            "wal_keep_size",
            "wal retention",
            "wal bound",
            "wal limit",
            "heartbeat",
            "slot_timeout",
            "retention",
            "backpressure",
            "lag bound",
            "checkpoint",
        )
        if any(t in corpus_text for t in cdc_triggers) and not _has_guard(combined, cdc_guards):
            failed_checks.append("CDC_SLOT_WAL_UNBOUNDED")
            violations.append(
                "CDC / logical replication slot architectures must bound WAL retention "
                "(`max_slot_wal_keep_size`) and slot lag."
            )

        # 6. UNVERIFIED_THROUGHPUT_CLAIM
        perf_guards = (
            "workload",
            "hardware",
            "cluster",
            "concurrency",
            "codec",
            "skew",
            "batch",
            "benchmark",
            "cold-cache",
            "memory",
        )
        if (
            _BENCHMARK_CLAIM_RE.search(corpus_text) or "sub-second latency" in corpus_text
        ) and not _has_guard(combined, perf_guards):
            failed_checks.append("UNVERIFIED_THROUGHPUT_CLAIM")
            violations.append(
                "Numeric throughput/latency claims must state workload boundaries "
                "(hardware, codec, concurrency, or benchmark scope)."
            )

        passed = len(failed_checks) == 0
        return DataEngineeringGateResult(
            passed=passed,
            checks_evaluated=len(DATA_ENG_REALISM_CHECK_NAMES),
            failed_checks=failed_checks,
            violations=violations,
            citations_valid=True,
            citations_verified_count=len(brief.verified_citations),
            reason=(
                "All 6 Data Engineering realism invariants satisfied."
                if passed
                else " | ".join(violations)
            ),
        )

    def evaluate_brief(
        self,
        brief: DomainKnowledgeBrief,
        corpus_chunks: list[DocChunkRecord] | list[DocumentChunk] | dict[str, str],
    ) -> DataEngineeringGateResult:
        """Composes the 6 Data Engineering invariant checks with DocumentCitationGate."""
        domain_res = self.evaluate_domain_rules(brief)
        citation_res = self.citation_gate.verify_citations(
            citations=brief.verified_citations,
            corpus_chunks=corpus_chunks,
            domain_id=self.domain_id,
            require_citations=True,
        )

        failed_checks = list(domain_res.failed_checks)
        violations = list(domain_res.violations)
        if not citation_res.passed:
            failed_checks.append("CITATION_VERIFICATION_FAILED")
            violations.append(
                f"{citation_res.failed_count} of {citation_res.total_citations} "
                "citations failed verbatim verification."
            )

        passed = domain_res.passed and citation_res.passed
        return DataEngineeringGateResult(
            passed=passed,
            checks_evaluated=domain_res.checks_evaluated,
            failed_checks=failed_checks,
            violations=violations,
            citations_valid=citation_res.passed,
            citations_verified_count=citation_res.verified_count,
            failed_citations=citation_res.failed_citations,
            citation_gate_result=citation_res,
            reason=" | ".join(violations) if violations else domain_res.reason,
        )


_BOILERPLATE_SUBSTRINGS: Final[tuple[str, ...]] = (
    "apache license",
    "licensed to the apache software foundation",
    "licensed under the apache",
    "spdx-license-identifier",
    "unless required by applicable law",
    'distributed on an "as is" basis',
    "without warranties or conditions",
    "contributor license agreements",
    "see the notice file",
    "permission is hereby granted",
    "copyright (c)",
    "all rights reserved",
)

_MD_LINK_ONLY_RE: Final[re.Pattern[str]] = re.compile(r"^(?:[-*•]\s*)?\[[^\]]+\]\([^)]+\)\s*$")


_SLOP_SUBSTRINGS: Final[tuple[str, ...]] = (
    "the rest of this paper",
    "see also",
    "declaration on generative ai",
    "declaration of generative ai",
    "{prodname}",
    '[id="',
    "ifdef::",
    "endif::",
    "xref:",
    "link:https://",
    "\\bibitem",
    "// category:",
    "\\label{",
    "\\sys",
    "<td",
    "<tr",
    "<span",
    "<table",
    "<div",
    "</",
    "author ",
    "### author",
)

_LATEX_CMD_RE: Final[re.Pattern[str]] = re.compile(
    r"\\(?:citet|citep|citeauthor|citeyear|cite|ref|eqref|label|sys|textbf|textit|emph|section|subsection|paragraph|textsc|texttt|bibitem|footnotemark)\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?"
)
_HTML_TAG_RE: Final[re.Pattern[str]] = re.compile(r"</?[a-zA-Z][^>]*>")


_YAML_FRONTMATTER_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:---|title:\s*|layout:\s*|permalink:\s*|-\s+/docs/|category:\s*)",
    re.IGNORECASE,
)


def is_slop_text(text: str) -> bool:
    """Returns True for boilerplate, headings, HTML/LaTeX/AsciiDoc slop, or meta-outlines."""
    stripped = text.strip()
    if not stripped or is_boilerplate_text(stripped):
        return True
    if stripped.startswith(("#", "%", "```", "|", "<<", "//", "---")):
        return True
    if _YAML_FRONTMATTER_RE.match(stripped):
        return True
    if stripped.startswith("$") and stripped.endswith("$"):
        return True
    low = stripped.lower()
    return any(marker in low for marker in _SLOP_SUBSTRINGS)


def clean_display_text(text: str) -> str:
    """Normalizes LaTeX/HTML/newline noise for human-readable synthesis fields."""
    cleaned = text.replace("{prodname}", "Debezium")
    cleaned = _LATEX_CMD_RE.sub("", cleaned)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("~", " ")
    # Unwrap single newlines while preserving sentence spacing
    cleaned = " ".join(cleaned.splitlines())
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def clean_brief_title(raw_title: str, default: str = "Data Engineering Systems Synthesis") -> str:
    """Sanitizes a document title into a clean brief title."""
    cleaned = re.sub(r"^#+\s*", "", raw_title.strip())
    cleaned = clean_display_text(cleaned)
    if not cleaned or is_slop_text(cleaned) or len(cleaned) < 4:
        return default
    return cleaned


def is_boilerplate_text(text: str) -> bool:
    """Returns True when text is license/copyright boilerplate or a pure link stub."""
    stripped = text.strip()
    if not stripped:
        return True
    low = stripped.lower()
    if any(marker in low for marker in _BOILERPLATE_SUBSTRINGS):
        return True
    if _MD_LINK_ONLY_RE.match(stripped):
        return True
    return stripped.startswith("%")


def _has_complete_sentence(text: str) -> bool:
    """Returns True if text contains at least one complete sentence with >=5 words."""
    stripped = text.strip()
    if len(re.findall(r"[A-Za-z]{3,}", stripped)) < 5:
        return False
    return bool(re.search(r"[.!?](?:\s|$)", stripped))


def extract_paragraph_blocks(text: str) -> list[str]:
    """Splits text on double-newline paragraph blocks and filters slop/frontmatter."""
    blocks: list[str] = []
    for raw_blk in re.split(r"\n\s*\n", text):
        blk = raw_blk.strip()
        if len(blk) < 40 or blk.endswith(":"):
            continue
        first_ln = blk.splitlines()[0].strip() if blk else ""
        if is_slop_text(blk) or is_slop_text(first_ln):
            continue
        if not _has_complete_sentence(blk):
            continue
        blocks.append(blk)
    return blocks


def extract_clean_quote(text: str, max_chars: int = 260) -> str:
    """Selects an intact, non-boilerplate verbatim paragraph or line from chunk text."""
    for blk in extract_paragraph_blocks(text):
        if len(blk) <= max_chars:
            return blk
        cut = blk[:max_chars].rsplit(" ", 1)[0].strip()
        if len(cut) >= 40:
            return cut
        return blk[:max_chars].strip()
    for raw_ln in text.splitlines():
        ln = raw_ln.strip()
        if len(ln) < 35 or ln.endswith(":"):
            continue
        if is_slop_text(ln):
            continue
        if len(re.findall(r"[A-Za-z]{3,}", ln)) < 4:
            continue
        if len(ln) <= max_chars:
            return ln
        # Trim strictly at a word boundary so the quote remains an exact substring prefix
        cut = ln[:max_chars].rsplit(" ", 1)[0].strip()
        if len(cut) >= 35:
            return cut
        return ln[:max_chars].strip()
    fallback = " ".join(text.split())
    if len(fallback) <= max_chars:
        return fallback
    return fallback[:max_chars].rsplit(" ", 1)[0].strip() or fallback[:max_chars]


def _extract_candidate_sentences(
    chunks: list[Any], *, limit: int = 12, max_failures: int = 8, max_caveats: int = 8
) -> tuple[list[str], list[str], list[str]]:
    """Extracts complete paragraph-unwrapped findings, caveats, and failure modes."""
    findings: list[str] = []
    caveats: list[str] = []
    failures: list[str] = []

    for chk in chunks:
        content = str(getattr(chk, "verbatim_content", "") or "")
        blocks = extract_paragraph_blocks(content)
        candidates = blocks if blocks else [ln.strip() for ln in content.splitlines()]
        for raw_cand in candidates:
            s_cand = raw_cand.strip().lstrip("-*•0123456789. ")
            if len(s_cand) < 35 or s_cand.endswith(":"):
                continue
            if is_slop_text(s_cand):
                continue
            cleaned_ln = clean_display_text(s_cand)
            if len(cleaned_ln) < 35 or not _has_complete_sentence(cleaned_ln):
                continue
            low = cleaned_ln.lower()
            if any(term in low for term in _FAILURE_TERMS):
                if cleaned_ln not in failures and len(failures) < max_failures:
                    failures.append(cleaned_ln)
            elif any(term in low for term in _CAVEAT_TERMS):
                if cleaned_ln not in caveats and len(caveats) < max_caveats:
                    caveats.append(cleaned_ln)
            elif cleaned_ln not in findings and len(findings) < limit:
                findings.append(cleaned_ln)

    return findings, caveats, failures


DATA_ENGINEERING_ACRONYMS: dict[str, list[tuple[str, str, str]]] = {
    "ACID": [
        ("A", "Atomicity", "ACID Atomicity transaction commit rollback"),
        ("C", "Consistency", "ACID Consistency database constraints invariants"),
        ("I", "Isolation", "ACID Isolation concurrent transactions MVCC"),
        ("D", "Durability", "ACID Durability write-ahead log WAL crash recovery"),
    ],
    "CAP": [
        ("C", "Consistency", "CAP Consistency distributed linearizability"),
        ("A", "Availability", "CAP Availability distributed non-failing response"),
        ("P", "Partition Tolerance", "CAP Partition Tolerance network split"),
    ],
    "CDC": [
        ("C", "Change", "CDC Change Data Capture log"),
        ("D", "Data", "CDC Data replication stream"),
        ("C", "Capture", "CDC Capture WAL debezium"),
    ],
    "OLAP": [
        ("O", "Online", "OLAP Online analytical query"),
        ("A", "Analytical", "OLAP Analytical columnar aggregation"),
        ("P", "Processing", "OLAP Processing warehouse"),
    ],
    "OLTP": [
        ("O", "Online", "OLTP Online transactional query"),
        ("T", "Transactional", "OLTP Transactional row-store ACID"),
        ("P", "Processing", "OLTP Processing database"),
    ],
}


class DataEngineeringDomainPlugin:
    """Data Engineering domain plugin (`data_engineering_scout`)."""

    domain_id: str = DOMAIN_ID_DATA_ENG

    def __init__(self, citation_gate: DocumentCitationGate | None = None) -> None:
        self.citation_gate = citation_gate or DocumentCitationGate(
            min_overlap_ratio=0.98,
            require_citations_default=True,
            strict_freshness=False,
        )
        self.realism_gate = DataEngineeringRealismGate(
            domain_id=self.domain_id,
            citation_gate=self.citation_gate,
        )

    def supported_domain_ids(self) -> tuple[str, ...]:
        return (DOMAIN_ID_DATA_ENG,)

    def get_role_domain_ids(self) -> dict[str, str]:
        return {
            "research": DOMAIN_ID_DATA_ENG,
            "knowledge": DOMAIN_ID_DATA_ENG,
        }

    def get_default_scopes(self) -> list[str]:
        return [
            "public",
            "public_eng",
            f"domain:{DOMAIN_ID_DATA_ENG}",
        ]

    def get_adapter_specs(self) -> list[dict[str, str]]:
        return [
            {
                "domain_id": DOMAIN_ID_DATA_ENG,
                "train_jsonl": f"domains/{DOMAIN_ID_DATA_ENG}/tuning/train.jsonl",
                "valid_jsonl": f"domains/{DOMAIN_ID_DATA_ENG}/tuning/valid.jsonl",
                "adapter_rel_path": (
                    f"domains/{DOMAIN_ID_DATA_ENG}/adapters/gemma4_26b_data_engineering"
                ),
                "label": "Data Engineering Concepts & Systems Specialist",
                "role": "knowledge",
            }
        ]

    def is_federated_domain(self, domain_id: str) -> bool:
        return False

    def is_research_domain(self, domain_id: str) -> bool:
        return domain_id.lower() == DOMAIN_ID_DATA_ENG

    def should_auto_route_research(
        self,
        prompt: str,
        domain_id: str,
        allowed_scopes: list[str] | None = None,
    ) -> bool:
        _ = prompt, allowed_scopes
        return domain_id.lower() == DOMAIN_ID_DATA_ENG

    def infer_friction_label(self, market_scope: str) -> str:
        scope_map = {
            "STORAGE_LAKEHOUSE": "Lakehouse Storage & Table Format Spec",
            "BATCH_MPP_ENGINES": "Distributed Compute & Query Optimizer Spec",
            "STREAMING_CDC": "Streaming, Watermarks & CDC Semantics Spec",
            "MODELING_QUALITY_GOV": "Data Modeling, Quality & Governance Spec",
        }
        return scope_map.get(market_scope, "Data Engineering Systems Architecture")

    def infer_market_scopes(self, prompt: str) -> list[DocumentScope]:
        _ = prompt
        return [DocumentScope.GLOBAL_MULTI]

    def expand_query_plan(self, plan: Any) -> Any:
        """Expands recognized Data Engineering acronyms (e.g. ACID) into component facets."""
        from context_scout.agents.intent_router import FacetSpec, QueryPlan

        if not isinstance(plan, QueryPlan) or not plan.is_acronym_query or not plan.acronym:
            return plan
        components = DATA_ENGINEERING_ACRONYMS.get(plan.acronym.upper())
        if not components:
            return plan
        expanded_facets = [
            FacetSpec(
                facet_id=f"facet_{idx}",
                facet_query=f"{plan.acronym.upper()} {term}",
                key_terms=[term.lower()],
                expected_source_kind="docs",
                is_acronym_component=True,
            )
            for idx, (_letter, term, _enriched_query) in enumerate(components, start=1)
        ]
        return QueryPlan(
            original_query=plan.original_query,
            requested_cardinality=plan.requested_cardinality or len(expanded_facets),
            is_acronym_query=True,
            acronym=plan.acronym.upper(),
            facets=expanded_facets,
        )

    def synthesize_research_brief(
        self,
        user_prompt: str,
        domain_id: str,
        chunks: list[Any],
        verified_citations: list[Any],
    ) -> DomainKnowledgeBrief:
        """Synthesizes a complete, un-truncated Data Engineering DomainKnowledgeBrief."""
        from context_scout.agents.intent_router import (
            decompose_query_facets,
            evaluate_facet_grounding,
        )

        plan = self.expand_query_plan(decompose_query_facets(user_prompt))
        req_card = plan.requested_cardinality or 6
        findings, caveats, failures = _extract_candidate_sentences(
            chunks,
            limit=max(10, req_card * 2),
            max_failures=max(6, req_card),
            max_caveats=6,
        )
        citations = list(verified_citations)
        facet_covs = [
            evaluate_facet_grounding(
                facet,
                doc_chunks=chunks,
                code_slices=[],
                domain_id=domain_id,
            )
            for facet in plan.facets
        ]
        missing_aspects = [msg for _, msg in facet_covs if msg]
        grounded_count = sum(1 for cov, _ in facet_covs if cov.status == "grounded")
        partial_count = sum(1 for cov, _ in facet_covs if cov.status == "partial")

        # Acronym expansion adherence (only cite lines genuinely present in chunks)
        acronym_findings: list[str] = []
        if plan.is_acronym_query and plan.acronym and plan.acronym in DATA_ENGINEERING_ACRONYMS:
            synonym_map: dict[str, tuple[str, ...]] = {
                "atomicity": (
                    "atomic",
                    "atomicity",
                    "begin transaction",
                    "rollback",
                    "all-or-nothing",
                    "all or nothing",
                    "commit",
                ),
                "consistency": (
                    "consistency",
                    "consistent",
                    "constraint",
                    "invariant",
                    "foreign key",
                    "check constraint",
                    "valid state",
                ),
                "isolation": (
                    "isolation",
                    "isolated",
                    "concurrent",
                    "mvcc",
                    "serializable",
                    "repeatable read",
                    "snapshot",
                ),
                "durability": (
                    "durability",
                    "durable",
                    "wal",
                    "write-ahead log",
                    "flushed",
                    "crash",
                    "fsync",
                    "persisted",
                ),
            }
            for letter, term, _enriched in DATA_ENGINEERING_ACRONYMS[plan.acronym]:
                term_low = term.lower()
                match_terms = synonym_map.get(term_low, (term_low,))
                matched_line = ""
                for chk in chunks:
                    content = str(getattr(chk, "verbatim_content", "") or "")
                    blocks = extract_paragraph_blocks(content)
                    candidates = blocks if blocks else [ln.strip() for ln in content.splitlines()]
                    for raw_cand in candidates:
                        s_cand = raw_cand.strip().lstrip("-*•0123456789. ")
                        if len(s_cand) < 25 or is_slop_text(s_cand):
                            continue
                        cleaned_ln = clean_display_text(s_cand)
                        if len(cleaned_ln) < 25:
                            continue
                        low_ln = cleaned_ln.lower()
                        if any(mt in low_ln for mt in match_terms):
                            matched_line = cleaned_ln
                            break
                    if matched_line:
                        break
                if matched_line:
                    acronym_findings.append(f"{term} ({letter}): {matched_line}")
                else:
                    missing_aspects.append(
                        f"Missing chunk evidence for {plan.acronym} component '{term} ({letter})'."
                    )

        for cit in reversed(citations):
            q = str(getattr(cit, "verbatim_quote", "") or "").strip()
            cleaned_q = clean_display_text(q)
            if cleaned_q and not is_slop_text(q) and cleaned_q not in findings:
                findings.insert(0, cleaned_q)

        if acronym_findings:
            merged_findings = acronym_findings + [
                f for f in findings if not any(f in af for af in acronym_findings)
            ]
            findings = merged_findings

        if not findings:
            findings = ["No verified findings extracted from retrieved chunks."]
        if not caveats:
            caveats = [
                "Evaluate partition layout, compaction cadence, and idempotency guarantees "
                "against target workload.",
            ]
        if not failures:
            failures = [
                "Unbounded state/WAL retention, uncompacted delete files, or partition skew "
                "under traffic spikes.",
            ]

        titles = list(
            dict.fromkeys(
                clean_brief_title(str(getattr(c, "doc_title", "") or ""))
                for c in citations
                if getattr(c, "doc_title", "")
            )
        )
        brief_title = (
            f"{titles[0]} — Architectural Synthesis"
            if titles
            else "Data Engineering Systems Synthesis"
        )
        lead_quote = (
            clean_display_text(str(getattr(citations[0], "verbatim_quote", "")))
            if citations and not is_slop_text(str(getattr(citations[0], "verbatim_quote", "")))
            else findings[0]
        )
        exec_summary = (
            f"Multi-source synthesis across {len(citations)} verified citation(s) "
            f"for '{user_prompt.strip()}': {lead_quote}"
        )

        sub_scopes = sorted(
            {
                str((getattr(c, "metadata_json", {}) or {}).get("sub_scope", ""))
                for c in chunks
                if isinstance(getattr(c, "metadata_json", None), dict)
                and (getattr(c, "metadata_json", {}) or {}).get("sub_scope")
            }
        )
        if not chunks or not citations or (grounded_count == 0 and partial_count == 0):
            sufficiency_verdict: Literal["sufficient", "partial", "insufficient"] = "insufficient"
            completeness_ratio = 0.0
        elif missing_aspects or partial_count > 0:
            sufficiency_verdict = "partial"
            completeness_ratio = round(
                (grounded_count + 0.5 * partial_count) / max(1, len(plan.facets)), 3
            )
        else:
            sufficiency_verdict = "sufficient"
            completeness_ratio = 1.0

        return DomainKnowledgeBrief(
            brief_title=brief_title,
            domain_id=domain_id,
            executive_summary=exec_summary,
            formal_expression_or_rule=None,
            core_findings_and_rules=findings[: max(10, req_card * 2)],
            domain_constraints_and_caveats=caveats[:6],
            known_risks_and_failure_modes=failures[: max(6, req_card)],
            structured_metadata={
                "domain_id": domain_id,
                "verified_citations_count": len(citations),
                "source_documents_count": max(1, len(titles)),
                "sub_scopes": ", ".join(sub_scopes) if sub_scopes else "data_engineering",
            },
            verified_citations=citations,
            missing_or_unverified_aspects=missing_aspects,
            completeness_ratio=completeness_ratio,
            sufficiency_verdict=sufficiency_verdict,
            missing_aspects=missing_aspects,
            synthesis_mode="deterministic_extractive_fallback",
        )

    def extract_brief_metadata(self, brief: Any) -> dict[str, Any]:
        if isinstance(brief, DomainKnowledgeBrief):
            return dict(brief.structured_metadata)
        return {}

    def validate_research_brief(
        self,
        brief: Any,
        doc_chunks: list[Any],
    ) -> tuple[bool, str]:
        if not isinstance(brief, DomainKnowledgeBrief):
            return True, ""
        res = self.realism_gate.evaluate_brief(brief, corpus_chunks=doc_chunks)
        return res.passed, res.reason

    def build_distillation_prompt_and_brief(
        self,
        domain_id: str,
        doc_id: str,
        title: str,
        sec_head: str,
        quote: str,
        chk_id: str,
        manifest: dict[str, Any],
    ) -> tuple[str, str]:
        _ = manifest
        doc_slug = doc_id.split(":")[-1]
        cit = DocCitationSpan(
            chunk_id=chk_id,
            doc_id=doc_id,
            doc_title=title,
            source_uri=f"urn:{domain_id}:{doc_slug}",
            section_heading=sec_head,
            verbatim_quote=quote,
        )
        kb_brief = DomainKnowledgeBrief(
            brief_title=f"{title} -- {sec_head}",
            domain_id=domain_id,
            executive_summary=(
                f"Grounded Data Engineering synthesis for {title} ({sec_head}): {quote}"
            ),
            formal_expression_or_rule=None,
            core_findings_and_rules=[quote],
            domain_constraints_and_caveats=[
                f"Grounded in {doc_id} ({sec_head}); verify partition pruning, "
                "compaction, and idempotency invariants."
            ],
            known_risks_and_failure_modes=[
                "Unbounded CDC WAL retention, uncompacted equality deletes, "
                "or non-idempotent sink retries."
            ],
            structured_metadata={
                "domain_id": domain_id,
                "verified_citations_count": 1,
                "section": sec_head,
            },
            verified_citations=[cit],
        )
        user_prompt = (
            f"Synthesize architectural trade-offs, operational caveats, and failure modes "
            f"for `{title}` ({sec_head}).\n"
            f"Retrieved Corpus Chunk: [chunk_id={chk_id} | doc_id={doc_id} | "
            f"title={title} | uri=urn:{domain_id}:{doc_slug} | section={sec_head} | "
            f"topic={kb_brief.brief_title} | quote={json.dumps(quote)}]"
        )
        return user_prompt, kb_brief.model_dump_json(indent=2)

    def build_multi_chunk_distillation_sample(
        self,
        domain_id: str,
        chunks: list[Any],
        quotes: list[str],
        rng: Any,
    ) -> dict[str, Any]:
        """Pairs 2-3 chunks into a cross-source conceptual trade-off distillation sample."""
        _ = rng
        selected_chunks = chunks[:3] if chunks else []
        citations: list[DocCitationSpan] = []
        chunk_lines: list[str] = []

        for idx, chk in enumerate(selected_chunks):
            raw_quote = (
                quotes[idx] if idx < len(quotes) else extract_clean_quote(str(chk.verbatim_content))
            )
            quote = (
                raw_quote
                if len(raw_quote) <= 220
                else (raw_quote[:220].rsplit(" ", 1)[0].strip() or raw_quote[:220])
            )
            cit = DocCitationSpan(
                chunk_id=str(chk.chunk_id),
                doc_id=chk.doc_id,
                doc_title=chk.doc_title,
                source_uri=chk.source_uri,
                section_heading=chk.section_heading,
                verbatim_quote=quote,
            )
            citations.append(cit)
            chunk_lines.append(
                f"Retrieved Corpus Chunk: [chunk_id={chk.chunk_id} | doc_id={chk.doc_id} | "
                f"title={chk.doc_title} | uri={chk.source_uri} | section={chk.section_heading} | "
                f"quote={json.dumps(quote)}]"
            )

        lead_chk = selected_chunks[0]
        val_user_prompt = (
            f"Compare architectural trade-offs, invariants, and operational failure modes across "
            f"{len(selected_chunks)} Data Engineering sources led by `{lead_chk.doc_title}`."
        )
        user_prompt = build_domain_knowledge_brief_prompt(
            user_prompt=val_user_prompt,
            domain_id=domain_id,
            chunk_lines=chunk_lines,
        )
        brief = self.synthesize_research_brief(
            user_prompt=val_user_prompt,
            domain_id=domain_id,
            chunks=selected_chunks,
            verified_citations=citations,
        )
        bad_citation = citations[0].model_copy(
            update={
                "verbatim_quote": "Hallucinated system benchmark quote absent from corpus text."
            }
        )
        bad_brief = brief.model_copy(update={"verified_citations": [bad_citation]})
        return {
            "brief": brief,
            "bad_brief": bad_brief,
            "brief_kwarg": {},
            "bad_brief_kwarg": {},
            "citation": citations[0],
            "bad_citation": bad_citation,
            "val_user_prompt": val_user_prompt,
            "user_prompt": user_prompt,
        }

    def build_research_distillation_sample(
        self,
        domain_id: str,
        target_chunk: Any,
        quote_text: str,
        rng: Any,
    ) -> dict[str, Any]:
        """Builds valid/invalid DomainKnowledgeBrief payloads for TeacherDistiller."""
        sibling_chunks = getattr(target_chunk, "_sibling_chunks", None)
        clean_lead_quote = extract_clean_quote(
            str(getattr(target_chunk, "verbatim_content", "") or quote_text),
            max_chars=220,
        )
        if not clean_lead_quote:
            clean_lead_quote = quote_text[:220]
        if isinstance(sibling_chunks, list) and len(sibling_chunks) >= 2:
            paired = [target_chunk, *[s for s in sibling_chunks if s is not target_chunk][:2]]
            quotes = [clean_lead_quote] + [
                extract_clean_quote(str(getattr(sc, "verbatim_content", "") or ""), max_chars=220)
                for sc in paired[1:]
            ]
            return self.build_multi_chunk_distillation_sample(
                domain_id=domain_id,
                chunks=paired,
                quotes=quotes,
                rng=rng,
            )
        return self.build_multi_chunk_distillation_sample(
            domain_id=domain_id,
            chunks=[target_chunk],
            quotes=[clean_lead_quote],
            rng=rng,
        )

    def verify_research_brief_output(
        self,
        raw_output: str,
    ) -> tuple[bool, list[str], list[str]]:
        kb_brief, strict_valid = parse_domain_knowledge_brief_json(
            raw_output,
            normalize_scalars=False,
        )
        if kb_brief is not None and strict_valid:
            is_valid = bool(
                kb_brief.brief_title
                and kb_brief.executive_summary
                and kb_brief.core_findings_and_rules
                and kb_brief.verified_citations
            )
            doc_ids = [c.doc_id for c in kb_brief.verified_citations[:5] if c.doc_id]
            titles = [c.doc_title for c in kb_brief.verified_citations[:5] if c.doc_title]
            return is_valid, doc_ids, titles
        return False, [], []

    def get_demo_seed_payloads(self) -> dict[str, Any]:
        return {
            "research_domain_id": DOMAIN_ID_DATA_ENG,
            "code_files": [],
            "ast_symbols": [],
            "doc_chunks": [],
        }

    def get_tuning_specs(self) -> dict[str, Any]:
        return {
            "research_domain_id": DOMAIN_ID_DATA_ENG,
            "system_scopes": self.get_default_scopes(),
            "adapter_specs": self.get_adapter_specs(),
        }


__all__ = [
    "DATA_ENG_REALISM_CHECK_NAMES",
    "DOMAIN_ID_DATA_ENG",
    "DataEngineeringDomainPlugin",
    "DataEngineeringGateResult",
    "DataEngineeringRealismGate",
]
