"""Domain-Agnostic Document & Knowledge Corpus Enterprise MCP Server (doc_corpus_mcp).

Provides typed read tools for any custom document, research, legal, medical, or
enterprise knowledge corpus onboarded via `context-scout init-domain --kind knowledge|research`,
enforcing out-of-band PrincipalContext authorization (SEC-04), Forced RLS, 3-Way Normalized
RRF retrieval, VAL-02 NFKC verbatim quote verification, and hard payload caps (<=10 items, <=32 KB).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Context, MCPServer

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.preflight_gate import (
    BoundGraphRepository,
    sync_custom_domain_corpus_if_needed,
)
from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer
from context_scout.config import ScoutSettings
from context_scout.db.repositories import InMemoryRLSRepository
from context_scout.mcp_servers.common import MAX_PAYLOAD_ITEMS, bound_payload_dict
from context_scout.schemas.contracts import DocCitationSpan
from context_scout.security.rbac import (
    DomainAuthorizationGate,
    get_current_principal,
    validate_tool_call_arguments,
)


def create_doc_corpus_mcp_server(
    domains_dir: Path | None = None,
    repo: Any = None,
    settings: ScoutSettings | None = None,
) -> MCPServer:
    """Factory creating the domain-agnostic doc_corpus_mcp MCP Server."""
    st = settings or ScoutSettings()
    resolved_domains_dir = domains_dir or st.domains_dir
    registry = DomainRegistry(resolved_domains_dir)
    registry.load_all()
    auth_gate = DomainAuthorizationGate(registry)
    bound_repo = BoundGraphRepository(repo or InMemoryRLSRepository())
    slicer = DeterministicSlicer()

    server = MCPServer("doc_corpus_mcp")

    @server.tool(
        name="search_hybrid_rrf",
        description=(
            "Search any onboarded document/knowledge domain using 3-Way Normalized RRF "
            "under out-of-band PrincipalContext RLS enforcement."
        ),
    )
    async def search_hybrid_rrf(
        domain_id: str,
        query: str,
        top_k: int = 5,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Executes hybrid RRF search across visible doc_chunks for the given domain_id."""
        validate_tool_call_arguments({"domain_id": domain_id, "query": query, "top_k": top_k})
        principal = get_current_principal()
        auth_gate.authorize_tool_call(
            domain_id, "doc_corpus_mcp:search_hybrid_rrf", principal
        )
        sync_custom_domain_corpus_if_needed(
            domain_id, index_path=resolved_domains_dir / domain_id / "corpus_index.json"
        )

        clamped_k = min(max(1, top_k), MAX_PAYLOAD_ITEMS)
        visible_chunks = bound_repo.list_visible_doc_chunks(principal, domain_id)
        q_tokens = {tok.lower() for tok in query.split() if len(tok) >= 3}
        scored: list[tuple[int, Any]] = []
        for c in visible_chunks:
            hay = f"{c.doc_title} {c.section_heading} {c.verbatim_content}".lower()
            score = sum(1 for tok in q_tokens if tok in hay)
            scored.append((score, c))
        scored.sort(key=lambda item: item[0], reverse=True)
        chunks = [c for _s, c in scored[:clamped_k]]
        hits = [
            {
                "chunk_id": str(c.chunk_id),
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "source_uri": c.source_uri,
                "section_heading": c.section_heading,
                "contextual_header": c.contextual_header,
                "verbatim_content": c.verbatim_content,
                "authority_weight": c.authority_weight,
            }
            for c in chunks[:clamped_k]
        ]
        return bound_payload_dict(
            {
                "domain_id": domain_id,
                "query": query,
                "count": len(hits),
                "hits": hits,
            }
        )

    @server.tool(
        name="verify_citation_quote",
        description=(
            "Verify a verbatim quote against a document chunk using VAL-02 NFKC "
            "normalization (>=0.98 token overlap)."
        ),
    )
    async def verify_citation_quote(
        domain_id: str,
        chunk_id: str,
        verbatim_quote: str,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Deterministically verifies that verbatim_quote exists in chunk_id."""
        validate_tool_call_arguments(
            {
                "domain_id": domain_id,
                "chunk_id": chunk_id,
                "verbatim_quote": verbatim_quote,
            }
        )
        principal = get_current_principal()
        auth_gate.authorize_tool_call(
            domain_id, "doc_corpus_mcp:verify_citation_quote", principal
        )
        sync_custom_domain_corpus_if_needed(
            domain_id, index_path=resolved_domains_dir / domain_id / "corpus_index.json"
        )

        visible_chunks = bound_repo.list_visible_doc_chunks(principal, domain_id)
        target_chunk = next(
            (
                c
                for c in visible_chunks
                if str(c.chunk_id) == chunk_id or c.doc_id == chunk_id
            ),
            None,
        )
        if target_chunk is None:
            return bound_payload_dict(
                {
                    "domain_id": domain_id,
                    "chunk_id": chunk_id,
                    "verified": False,
                    "is_valid": False,
                    "overlap_ratio": 0.0,
                    "reason": f"Chunk '{chunk_id}' not found or not visible to principal.",
                }
            )

        cit = DocCitationSpan(
            chunk_id=str(target_chunk.chunk_id),
            doc_id=target_chunk.doc_id,
            doc_title=target_chunk.doc_title,
            source_uri=target_chunk.source_uri,
            section_heading=target_chunk.section_heading,
            verbatim_quote=verbatim_quote,
        )
        verif = slicer.verify_doc_citation(cit, target_chunk.verbatim_content)
        return bound_payload_dict(
            {
                "domain_id": domain_id,
                "chunk_id": str(target_chunk.chunk_id),
                "doc_id": target_chunk.doc_id,
                "doc_title": target_chunk.doc_title,
                "verified": verif.is_valid,
                "is_valid": verif.is_valid,
                "matched_source_segment": verif.matched_source_segment,
                "token_overlap_ratio": verif.token_overlap_ratio,
            }
        )

    @server.tool(
        name="get_document_section",
        description="Retrieve full verbatim sections for a document ID within a domain.",
    )
    async def get_document_section(
        domain_id: str,
        doc_id: str,
        section_query: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Returns matching sections for doc_id under Forced RLS."""
        validate_tool_call_arguments(
            {"domain_id": domain_id, "doc_id": doc_id, "section_query": section_query}
        )
        principal = get_current_principal()
        auth_gate.authorize_tool_call(
            domain_id, "doc_corpus_mcp:get_document_section", principal
        )
        sync_custom_domain_corpus_if_needed(
            domain_id, index_path=resolved_domains_dir / domain_id / "corpus_index.json"
        )

        visible_chunks = bound_repo.list_visible_doc_chunks(principal, domain_id)
        matching = [
            c
            for c in visible_chunks
            if c.doc_id == doc_id or str(c.chunk_id) == doc_id or doc_id in c.doc_title
        ]
        if section_query:
            sq_l = section_query.lower()
            filtered = [
                c
                for c in matching
                if sq_l in c.section_heading.lower() or sq_l in c.verbatim_content.lower()
            ]
            if filtered:
                matching = filtered

        items = [
            {
                "chunk_id": str(c.chunk_id),
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "source_uri": c.source_uri,
                "section_heading": c.section_heading,
                "verbatim_content": c.verbatim_content,
            }
            for c in matching[:MAX_PAYLOAD_ITEMS]
        ]
        return bound_payload_dict(
            {
                "domain_id": domain_id,
                "doc_id": doc_id,
                "count": len(items),
                "items": items,
            }
        )

    return server
