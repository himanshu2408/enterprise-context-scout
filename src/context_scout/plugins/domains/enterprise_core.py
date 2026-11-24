"""Core Enterprise Domain Plugin (code_architecture_scout, enterprise_docs_scout, federated_all).

Provides neutral domain-specific synthesis, verification, and adapter metadata
without any vertical or vendor lock-in.
"""

from __future__ import annotations

import json
import random
from datetime import date
from typing import Any, Final
from uuid import NAMESPACE_URL, uuid5

from context_scout.agents.citation_gate import DocumentCitationGate
from context_scout.db.repositories import DocChunkRecord
from context_scout.retrieval.embedder import get_embedding_provider
from context_scout.schemas.contracts import (
    DocCitationSpan,
    DocumentScope,
    DomainKnowledgeBrief,
    build_domain_knowledge_brief_prompt,
    parse_domain_knowledge_brief_json,
)

DOMAIN_ID_CODE_ARCH: Final[str] = "code_architecture_scout"
DOMAIN_ID_ENTERPRISE_DOCS: Final[str] = "enterprise_docs_scout"
DOMAIN_ID_FEDERATED_ALL: Final[str] = "federated_all"


class EnterpriseCoreDomainPlugin:
    """Default neutral DomainPlugin for core code architecture and enterprise docs."""

    @property
    def domain_id(self) -> str:
        return DOMAIN_ID_CODE_ARCH

    def supported_domain_ids(self) -> tuple[str, ...]:
        return (DOMAIN_ID_CODE_ARCH, DOMAIN_ID_ENTERPRISE_DOCS, DOMAIN_ID_FEDERATED_ALL)

    def get_role_domain_ids(self) -> dict[str, str]:
        return {
            "code": DOMAIN_ID_CODE_ARCH,
            "research": DOMAIN_ID_ENTERPRISE_DOCS,
            "federated": DOMAIN_ID_FEDERATED_ALL,
        }

    def get_default_scopes(self) -> list[str]:
        return [
            "public",
            "public_eng",
            "public_docs",
            f"repo:{DOMAIN_ID_CODE_ARCH}",
            "corpus:enterprise_docs",
            f"domain:{DOMAIN_ID_CODE_ARCH}",
            f"domain:{DOMAIN_ID_ENTERPRISE_DOCS}",
        ]

    def get_adapter_specs(self) -> list[dict[str, str]]:
        return [
            {
                "domain_id": DOMAIN_ID_CODE_ARCH,
                "adapter_rel_path": f"domains/{DOMAIN_ID_CODE_ARCH}/adapters/gemma4_26b_code_scout",
                "label": "Code Architecture Scout",
                "role": "code",
            },
            {
                "domain_id": DOMAIN_ID_ENTERPRISE_DOCS,
                "adapter_rel_path": (
                    f"domains/{DOMAIN_ID_ENTERPRISE_DOCS}/adapters/gemma4_26b_enterprise_docs"
                ),
                "label": "Enterprise Docs & ADRs Scout",
                "role": "research",
            },
        ]

    def is_federated_domain(self, domain_id: str) -> bool:
        return domain_id.strip().lower() == DOMAIN_ID_FEDERATED_ALL

    def is_research_domain(self, domain_id: str) -> bool:
        return domain_id.strip().lower() == DOMAIN_ID_ENTERPRISE_DOCS

    def should_auto_route_research(
        self,
        prompt: str,
        domain_id: str,
        allowed_scopes: list[str] | None = None,
    ) -> bool:
        if self.is_research_domain(domain_id):
            return True
        if self.is_federated_domain(domain_id):
            return True
        lower = prompt.lower()
        doc_cues = ("adr", "confluence", "jira", "policy", "runbook", "specification", "compliance")
        if any(cue in lower for cue in doc_cues):
            return True
        return bool(
            allowed_scopes
            and "public_docs" in allowed_scopes
            and "public_eng" not in allowed_scopes
        )

    def infer_friction_label(self, market_scope: str) -> str:
        scope_upper = market_scope.strip().upper()
        if scope_upper in {"INTERNAL_ADR", "CONFLUENCE", "JIRA", "ENTERPRISE_DOC"}:
            return f"Enterprise Scope: {scope_upper}"
        return f"Document Scope: {scope_upper or 'GLOBAL'}"

    def infer_market_scopes(self, prompt: str) -> list[DocumentScope]:
        lower = prompt.lower()
        scopes: list[DocumentScope] = []
        if "adr" in lower or "architecture decision" in lower:
            scopes.append(DocumentScope.INTERNAL_ADR)
        if "confluence" in lower:
            scopes.append(DocumentScope.CONFLUENCE)
        if "jira" in lower:
            scopes.append(DocumentScope.JIRA)
        if not scopes:
            scopes.append(DocumentScope.GLOBAL_MULTI)
        return scopes

    def synthesize_research_brief(
        self,
        user_prompt: str,
        domain_id: str,
        chunks: list[Any],
        verified_citations: list[Any],
    ) -> DomainKnowledgeBrief:
        citations = [
            c if isinstance(c, DocCitationSpan) else DocCitationSpan.model_validate(c)
            for c in verified_citations
        ]
        if not citations:
            raise ValueError("Cannot synthesize DomainKnowledgeBrief without verified_citations")

        first_title = citations[0].doc_title
        findings = [f"[{c.section_heading}] {c.verbatim_quote[:240]}" for c in citations[:4]]
        return DomainKnowledgeBrief(
            brief_title=f"{first_title}: {user_prompt[:80].strip()}",
            domain_id=domain_id or DOMAIN_ID_ENTERPRISE_DOCS,
            executive_summary=(
                f"Synthesized grounded enterprise knowledge brief from {len(citations)} "
                f"verified citation(s) in {domain_id or DOMAIN_ID_ENTERPRISE_DOCS}."
            ),
            formal_expression_or_rule=None,
            core_findings_and_rules=findings,
            domain_constraints_and_caveats=[
                f"Grounded strictly against {len(chunks)} retrieved document chunk(s)."
            ],
            known_risks_and_failure_modes=[
                "Verify document version and deprecation status before applying changes."
            ],
            structured_metadata={
                "source_count": len(citations),
                "domain_id": domain_id or DOMAIN_ID_ENTERPRISE_DOCS,
            },
            verified_citations=citations,
            completeness_ratio=1.0,
            sufficiency_verdict="sufficient",
            synthesis_mode="deterministic_extractive_fallback",
        )

    def extract_brief_metadata(self, brief: Any) -> dict[str, Any]:
        if isinstance(brief, DomainKnowledgeBrief):
            return {
                "brief_title": brief.brief_title,
                "domain_id": brief.domain_id,
                "citation_count": len(brief.verified_citations),
                "sufficiency_verdict": brief.sufficiency_verdict,
                **brief.structured_metadata,
            }
        return {}

    def validate_research_brief(
        self,
        brief: Any,
        doc_chunks: list[Any],
    ) -> tuple[bool, str]:
        if not isinstance(brief, DomainKnowledgeBrief):
            return False, "Expected DomainKnowledgeBrief instance"
        if not brief.verified_citations:
            return False, "DomainKnowledgeBrief has no verified_citations"
        gate = DocumentCitationGate(min_overlap_ratio=0.98, require_citations_default=True)
        res = gate.verify_citations(brief.verified_citations, doc_chunks)
        if not res.passed:
            return False, res.reason
        return True, "DomainKnowledgeBrief verified against source chunks"

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
        chunk_line = (
            f"[Chunk 1 | chunk_id={chk_id} | doc_id={doc_id} | title={title} | "
            f"uri=urn:{domain_id}:{doc_id} | section={sec_head}]\n{quote}"
        )
        prompt = build_domain_knowledge_brief_prompt(
            user_prompt=f"Synthesize a verified brief for `{title}` covering `{sec_head}`.",
            domain_id=domain_id,
            chunk_lines=[chunk_line],
        )
        brief = DomainKnowledgeBrief(
            brief_title=f"{title}: {sec_head}",
            domain_id=domain_id,
            executive_summary=f"Verified synthesis of `{sec_head}` from `{title}`.",
            formal_expression_or_rule=None,
            core_findings_and_rules=[quote[:200]],
            domain_constraints_and_caveats=[f"Evaluated against {domain_id} chunks."],
            known_risks_and_failure_modes=["Verify document version before production rollout."],
            structured_metadata={"domain_id": domain_id, "doc_id": doc_id},
            verified_citations=[
                DocCitationSpan(
                    chunk_id=chk_id,
                    doc_id=doc_id,
                    doc_title=title,
                    source_uri=f"urn:{domain_id}:{doc_id}",
                    section_heading=sec_head,
                    verbatim_quote=quote,
                )
            ],
            synthesis_mode="deterministic_extractive_fallback",
        )
        return prompt, brief.model_dump_json()

    def get_demo_seed_payloads(self) -> dict[str, Any]:
        doc_specs = [
            (
                "adr-001-rls-isolation",
                "ADR-001: PostgreSQL 17 FORCE RLS Tenant Isolation",
                "urn:enterprise_docs_scout:adr-001-rls-isolation",
                DocumentScope.INTERNAL_ADR,
                "Row-Level Security Policy",
                (
                    "Every tenant-scoped table in PostgreSQL 17 MUST enforce tenant_id UUID "
                    "NOT NULL, ENABLE ROW LEVEL SECURITY, and FORCE ROW LEVEL SECURITY under "
                    "NOBYPASSRLS. RLSConnectionManager sets app.tenant_id and app.principal_scopes "
                    "inside scoped transactions."
                ),
            ),
            (
                "spec-002-zero-spend-abstention",
                "Spec-002: Stage 4 Zero-Spend Abstention Policy",
                "urn:enterprise_docs_scout:spec-002-zero-spend-abstention",
                DocumentScope.ENTERPRISE_DOC,
                "Deterministic Abstention",
                (
                    "When Stage 4 PreFlightValidator checks fail after local bounce-back, the "
                    "workflow abstains locally with paid_tokens_spent=0 and emits an "
                    "InsufficientContextDiagnostic payload."
                ),
            ),
            (
                "conf-101-citation-provenance",
                "Confluence & Jira Citation Provenance Standard",
                "https://confluence.example.com/pages/101",
                DocumentScope.CONFLUENCE,
                "Provenance Fields",
                (
                    "All DocCitationSpan entries from Confluence MUST record page_id and "
                    "version, while Jira citations MUST record ticket_key alongside chunk_id "
                    "and verbatim_quote."
                ),
            ),
            (
                "doc-004-hybrid-rrf",
                "Normalized Hybrid RRF Retrieval Architecture",
                "urn:enterprise_docs_scout:doc-004-hybrid-rrf",
                DocumentScope.GLOBAL_MULTI,
                "Reciprocal Rank Fusion",
                (
                    "Hybrid RRF fuses pgvector cosine similarity rank and PostgreSQL full-text "
                    "search rank into a normalized [0.0, 1.0] score via compute_normalized_rrf."
                ),
            ),
            (
                "policy-eu-residency",
                "EU Air-Gapped Model Residency Policy",
                "urn:enterprise_docs_scout:policy-eu-residency",
                DocumentScope.REGIONAL_EU,
                "Data Residency",
                (
                    "All local inference and embedding pipelines operate inside private VPCs or "
                    "air-gapped Kubernetes clusters with zero external network egress."
                ),
            ),
            (
                "CORE-101",
                "Jira CORE-101: Stale Document Tombstone Workflow",
                "https://jira.example.com/browse/CORE-101",
                DocumentScope.JIRA,
                "Tombstone Lifecycle",
                (
                    "Deleted or superseded Confluence pages are marked is_stale=True and "
                    "excluded from synthesis and PreFlightValidator grounding."
                ),
            ),
        ]
        doc_chunks: list[DocChunkRecord] = []
        embedder = get_embedding_provider()
        spec_vecs = embedder.embed_batch(
            [f"{title} {heading} {content}" for _, title, _, _, heading, content in doc_specs],
            is_query=False,
        )
        for (doc_id, title, uri, scope, heading, content), emb_vec in zip(
            doc_specs, spec_vecs, strict=True
        ):
            doc_chunks.append(
                DocChunkRecord(
                    chunk_id=uuid5(NAMESPACE_URL, f"{DOMAIN_ID_ENTERPRISE_DOCS}:{doc_id}"),
                    domain_id=DOMAIN_ID_ENTERPRISE_DOCS,
                    doc_id=doc_id,
                    doc_title=title,
                    source_uri=uri,
                    market_scope=scope,
                    published_date=date(2026, 10, 1),
                    authority_weight=1.0,
                    section_heading=heading,
                    contextual_header=f"[{title} | {heading}]",
                    verbatim_content=content,
                    metadata_json={"source_type": "enterprise_doc"},
                    allowed_scopes=["public", "public_eng", "public_docs"],
                    embedding=emb_vec,
                )
            )
        return {
            "code_domain": DOMAIN_ID_CODE_ARCH,
            "research_domain": DOMAIN_ID_ENTERPRISE_DOCS,
            "research_domain_id": DOMAIN_ID_ENTERPRISE_DOCS,
            "federated_domain": DOMAIN_ID_FEDERATED_ALL,
            "fallback_symbol": "PreFlightValidator",
            "fallback_file": "src/context_scout/agents/preflight_gate.py",
            "doc_chunks": doc_chunks,
        }

    def get_tuning_specs(self) -> dict[str, Any]:
        return {
            "code_domain": DOMAIN_ID_CODE_ARCH,
            "research_domain": DOMAIN_ID_ENTERPRISE_DOCS,
            "adapter_specs": self.get_adapter_specs(),
        }

    def build_research_distillation_sample(
        self,
        domain_id: str,
        target_chunk: Any,
        quote_text: str,
        rng: random.Random,
    ) -> dict[str, Any]:
        chk_id = str(getattr(target_chunk, "chunk_id", "chunk-001"))
        doc_id = str(getattr(target_chunk, "doc_id", "doc-001"))
        title = str(getattr(target_chunk, "doc_title", "Enterprise Spec"))
        sec_head = str(getattr(target_chunk, "section_heading", "Overview"))
        source_uri = str(getattr(target_chunk, "source_uri", f"urn:{domain_id}:{doc_id}"))

        prompt = build_domain_knowledge_brief_prompt(
            user_prompt=f"Synthesize `{title}` ({sec_head}).",
            domain_id=domain_id,
            chunk_lines=[
                f"[Chunk 1 | chunk_id={chk_id} | doc_id={doc_id} | title={title} | "
                f"uri={source_uri} | section={sec_head}]\n{quote_text}"
            ],
        )
        brief = DomainKnowledgeBrief(
            brief_title=f"{title}: {sec_head}",
            domain_id=domain_id,
            executive_summary=f"Grounded summary of {sec_head}.",
            formal_expression_or_rule=None,
            core_findings_and_rules=[quote_text[:200]],
            domain_constraints_and_caveats=[f"Evaluated against {domain_id} corpus."],
            known_risks_and_failure_modes=["Check version currency."],
            structured_metadata={"domain_id": domain_id},
            verified_citations=[
                DocCitationSpan(
                    chunk_id=chk_id,
                    doc_id=doc_id,
                    doc_title=title,
                    source_uri=source_uri,
                    section_heading=sec_head,
                    verbatim_quote=quote_text,
                )
            ],
            synthesis_mode="deterministic_extractive_fallback",
        )
        return {
            "prompt": prompt,
            "valid_response": json.dumps(brief.model_dump(mode="json")),
            "invalid_response": json.dumps({"brief_title": title}),
        }

    def verify_research_brief_output(
        self,
        raw_output: str,
    ) -> tuple[bool, list[str], list[str]]:
        brief, _ = parse_domain_knowledge_brief_json(raw_output)
        if brief is None:
            return False, [], []
        doc_ids = [c.doc_id for c in brief.verified_citations]
        titles = [c.doc_title for c in brief.verified_citations]
        return True, doc_ids, titles


__all__ = [
    "DOMAIN_ID_CODE_ARCH",
    "DOMAIN_ID_ENTERPRISE_DOCS",
    "DOMAIN_ID_FEDERATED_ALL",
    "EnterpriseCoreDomainPlugin",
]
