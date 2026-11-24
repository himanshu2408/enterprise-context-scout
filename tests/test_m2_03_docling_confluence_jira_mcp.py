"""M2-03 tests: Docling DocExtractorSPI, Confluence macros/diagrams, Jira, doc_corpus_mcp."""

from __future__ import annotations

import base64
import io
import zipfile
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
import yaml
from mcp.types import CallToolResult

from context_scout.db.repositories import InMemoryRLSRepository
from context_scout.db.schema import DocChunkRecord
from context_scout.ingestion.doc_parser import PdfDocParser
from context_scout.ingestion.docling_extractor import (
    CompositeDoclingExtractor,
    DoclingServeHttpExtractor,
    NativeStructuredDocExtractor,
    UnsafeDocumentError,
)
from context_scout.mcp_servers.doc_corpus_mcp import create_doc_corpus_mcp_server
from context_scout.plugins.connectors.confluence import ConfluenceConnectorPlugin, ConfluencePageDTO
from context_scout.plugins.connectors.jira import JiraConnectorPlugin
from context_scout.retrieval.embedder import get_embedding_provider
from context_scout.schemas.contracts import PrincipalContext, RoleType
from context_scout.security.rbac import bind_session_principal


def _build_minimal_docx(body_xml: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="xml" ContentType="application/xml"/>'
            "</Types>",
        )
        zf.writestr("word/document.xml", body_xml)
    return buf.getvalue()


def _build_minimal_pptx(slide_xml: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ppt/slides/slide1.xml", slide_xml)
    return buf.getvalue()


def test_docling_http_ssrf_block_and_native_office_xxe_guards() -> None:
    """Docling HTTP reconstructs tables cleanly; SSRF & OOXML XXE/ZIP-bomb fail closed."""
    with pytest.raises(UnsafeDocumentError):
        DoclingServeHttpExtractor("http://93.184.216.34/v1", allow_external_egress=False)

    def _mock_docling_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/v1/convert/file")
        return httpx.Response(
            200,
            json={
                "document": {
                    "md_content": (
                        "# Runbook\n\n| Service | Port |\n| --- | --- |\n| api | 8443 |\n"
                    ),
                    "json_content": {
                        "tables": [
                            {
                                "data": {
                                    "table_cells": [
                                        {
                                            "start_row_offset_idx": 0,
                                            "start_col_offset_idx": 0,
                                            "text": "Service",
                                        },
                                        {
                                            "start_row_offset_idx": 0,
                                            "start_col_offset_idx": 1,
                                            "text": "Port",
                                        },
                                        {
                                            "start_row_offset_idx": 1,
                                            "start_col_offset_idx": 0,
                                            "text": "gateway",
                                        },
                                        {
                                            "start_row_offset_idx": 1,
                                            "start_col_offset_idx": 1,
                                            "text": "9443",
                                        },
                                    ]
                                }
                            }
                        ]
                    },
                }
            },
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(_mock_docling_handler))
    http_ext = DoclingServeHttpExtractor(
        "http://127.0.0.1:5001",
        http_client=mock_client,
    )
    comp = CompositeDoclingExtractor(http_extractor=http_ext)
    pdf_doc = comp.extract(b"%PDF-1.4\n%fake", filename="runbook.pdf")
    assert pdf_doc.backend == "docling-serve"
    assert "| gateway | 9443 |" in pdf_doc.markdown_text

    # Native DOCX OOXML table extraction + PdfDocParser multi-format routing
    docx_bytes = _build_minimal_docx(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>Architecture Spec</w:t></w:r></w:p>"
        "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Component</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>Owner</w:t></w:r></w:p></w:tc></w:tr>"
        "<w:tr><w:tc><w:p><w:r><w:t>Ingestor</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>Platform</w:t></w:r></w:p></w:tc></w:tr>"
        "</w:tbl></w:body></w:document>"
    )
    native = NativeStructuredDocExtractor()
    docx_out = native.extract(docx_bytes, filename="arch.docx")
    assert "Architecture Spec" in docx_out.markdown_text
    assert "| Component | Owner |" in docx_out.markdown_text
    assert "| Ingestor | Platform |" in docx_out.markdown_text

    # Native PPTX extraction + PdfDocParser Docling delegation
    pptx_bytes = _build_minimal_pptx(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        "<p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Zero Egress SLA</a:t></a:r></a:p>"
        "</p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
    )
    pptx_out = native.extract(pptx_bytes, filename="deck.pptx")
    assert "Zero Egress SLA" in pptx_out.markdown_text
    pdf_txt = PdfDocParser(doc_extractor=http_ext).extract_text(b"%PDF-1.4\n" + b"0" * 64)
    assert "| gateway | 9443 |" in pdf_txt

    # XXE rejection inside DOCX XML
    xxe_docx = _build_minimal_docx(
        '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
        "<w:document><w:body><w:p>&xxe;</w:p></w:body></w:document>"
    )
    with pytest.raises(UnsafeDocumentError):
        native.extract(xxe_docx, filename="evil.docx")


def test_confluence_macros_plantuml_mermaid_drawio_tables_and_scrubbing() -> None:
    """Confluence parses macros, diagrams (raw + deflate Draw.io), rowspan tables, & scrubs PII."""
    raw_mx = (
        "<mxGraphModel><root>"
        '<mxCell id="0"/><mxCell id="1" parent="0"/>'
        '<mxCell id="2" value="Gateway" vertex="1" parent="1"/>'
        '<mxCell id="3" value="Worker" vertex="1" parent="1"/>'
        '<mxCell id="4" value="enqueue" edge="1" source="2" target="3" parent="1"/>'
        "</root></mxGraphModel>"
    )
    comp = zlib.compressobj(level=6, wbits=-15)
    deflated = comp.compress(raw_mx.encode("utf-8")) + comp.flush()
    b64_diag = base64.b64encode(deflated).decode("ascii")

    secret_pat = "ghp_" + "AbCdEf0123456789AbCdEf012345"
    body_html = f"""
<h2>Runbook Overview</h2>
<ac:structured-macro ac:name="warning"><ac:parameter ac:name="title">Rotation</ac:parameter>
<ac:rich-text-body><p>Rotate
<ac:structured-macro ac:name="jira">
<ac:parameter ac:name="key">OPS-404</ac:parameter>
</ac:structured-macro>
token {secret_pat} contact oncall@corp.local.</p></ac:rich-text-body>
</ac:structured-macro>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">python</ac:parameter>
<ac:plain-text-body><![CDATA[def parse_generic(items: list[str]) -> Result<int>:

    return len(items)
]]></ac:plain-text-body></ac:structured-macro>
<ac:structured-macro ac:name="plantuml"><ac:plain-text-body><![CDATA[
@startuml
title Dispatch Flow
Client -> Router : submitJob
Router --> Queue : enqueueTask
@enduml
]]></ac:plain-text-body></ac:structured-macro>
<ac:structured-macro ac:name="mermaid"><ac:plain-text-body><![CDATA[
flowchart LR
API[Gateway API] -->|push| Worker[Index Worker]
]]></ac:plain-text-body></ac:structured-macro>
<ac:structured-macro ac:name="drawio"><ac:parameter ac:name="diagramName">Pipeline</ac:parameter>
<ac:plain-text-body>
<![CDATA[<mxfile><diagram name="Pipeline">{b64_diag}</diagram></mxfile>]]>
</ac:plain-text-body>
</ac:structured-macro>
<table>
  <tr><th rowspan="2">Tier</th><th colspan="2">SLA</th></tr>
  <tr><th>Target</th><th>Scope</th></tr>
  <tr><td>Tier-1</td><td>&lt;60s</td><td>Webhook</td></tr>
</table>
"""
    connector = ConfluenceConnectorPlugin()
    page = ConfluencePageDTO(
        page_id="pg-777",
        space_key="ENG",
        title="Deployment Architecture OPS-500",
        version_number=3,
        author_username="architect",
        created_at=datetime.now(UTC),
        last_modified_at=datetime.now(UTC),
        source_url="https://mock.atlassian.net/wiki/spaces/ENG/pages/777",
        body_storage=body_html,
    )
    chunks = connector.chunk_page_verbatim(
        page,
        max_chunk_chars=600,
        allowed_scopes=["public_eng"],
    )
    assert chunks
    full_text = "\n".join(c.verbatim_content for c in chunks)

    # Secrets & email PII scrubbed before persistence
    assert secret_pat not in full_text
    assert "oncall@corp.local" not in full_text
    assert "[REDACTED:secret:" in full_text
    assert "[REDACTED:pii:email]" in full_text

    # Code block blank lines & generics preserved inside one fence (not split in half)
    assert "Result<int>" in full_text
    for chk in chunks:
        assert chk.verbatim_content.count("```") % 2 == 0, "Fenced block was sliced across chunks"

    # Diagrams (PlantUML, Mermaid, Draw.io) & Jira links & table captured in metadata + text
    assert "OPS-404" in chunks[0].linked_jira_keys
    assert "OPS-500" in chunks[0].linked_jira_keys
    diag_types = {d["diagram_type"] for d in chunks[0].diagrams}
    assert diag_types == {"plantuml", "mermaid", "drawio"}
    assert "Gateway -> Worker (enqueue)" in full_text
    assert "| Tier-1 | <60s | Webhook |" in full_text

    # Draw.io XXE payload fails closed
    xxe_page = ConfluencePageDTO(
        page_id="pg-xxe",
        space_key="ENG",
        title="Bad XML",
        version_number=1,
        author_username="eve",
        created_at=datetime.now(UTC),
        last_modified_at=datetime.now(UTC),
        source_url="https://mock.atlassian.net/wiki/pages/xxe",
        body_storage=(
            "<mxfile><!DOCTYPE foo [<!ENTITY x SYSTEM 'file:///etc/passwd'>]>"
            "<diagram><mxGraphModel/></diagram></mxfile>"
        ),
    )
    with pytest.raises(UnsafeDocumentError):
        connector.chunk_page_verbatim(xxe_page, allowed_scopes=["public_eng"])


def test_jira_epic_story_mr_hierarchy_acl_isolation_and_scrubbing() -> None:
    """Jira parses Epic->Story->Subtask->MR, scrubs secrets/PII, & enforces strict ACL scope."""
    connector = JiraConnectorPlugin()
    now_iso = "2026-11-18T10:00:00.000+0000"
    secret_tok = "glpat-" + "xyz0123456789abcdefghijk"

    def _raw_issue(
        key: str,
        itype: str,
        summary: str,
        desc: str,
        *,
        epic_cf: str | None = None,
        parent_key: str | None = None,
        sec_id: str | None = None,
        sec_name: str | None = None,
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "project": {"key": "ENG"},
            "summary": summary,
            "issuetype": {"name": itype},
            "status": {"name": "In Progress"},
            "description": desc,
            "created": now_iso,
            "updated": now_iso,
        }
        if epic_cf:
            fields["customfield_10014"] = epic_cf
        if parent_key:
            fields["parent"] = {"key": parent_key}
        if sec_id and sec_name:
            fields["security"] = {"id": sec_id, "name": sec_name}
        return {"id": key, "key": key, "fields": fields}

    epic_dto = connector._parse_jira_issue(
        _raw_issue("ENG-10", "Epic", "Zero Downtime Worker", "Parent architecture epic"),
        "ENG",
    )
    pub_story = connector._parse_jira_issue(
        _raw_issue(
            "ENG-11",
            "Story",
            "Implement backoff coordinator",
            f"Fixes webhook retries via acme/worker!42 token={secret_tok}",
            epic_cf="ENG-10",
        ),
        "ENG",
    )
    pub_sub = connector._parse_jira_issue(
        _raw_issue(
            "ENG-12",
            "Sub-task",
            "Add GitHub PR mirror test",
            "See https://github.com/acme/worker/pull/99",
            parent_key="ENG-11",
        ),
        "ENG",
    )
    sec_story = connector._parse_jira_issue(
        _raw_issue(
            "ENG-13",
            "Story",
            "Restricted KeyVault Purge",
            "Confidential purge tracked in acme/vault!777 for 123-45-6789",
            epic_cf="ENG-10",
            sec_id="9001",
            sec_name="sec_restricted",
        ),
        "ENG",
    )
    sec_story.security_level_principals = ("group:sec_restricted",)

    assert "acme/worker!42" in pub_story.linked_mrs
    assert "https://github.com/acme/worker/pull/99" in pub_sub.linked_mrs

    # Chunking scrubs secrets/PII before embedding/indexing
    pub_chunks = connector.chunk_issue_verbatim(pub_story, domain_id="eval_jira")
    assert secret_tok not in pub_chunks[0].verbatim_content
    assert "[REDACTED:secret:gitlab-token]" in pub_chunks[0].verbatim_content
    assert pub_chunks[0].metadata_json["epic_key"] == "ENG-10"

    base_scopes = connector.mirror_permissions("ENG")
    sec_scopes = [*base_scopes, *connector.resolve_issue_allowed_scopes(sec_story)]
    issues = [epic_dto, pub_story, pub_sub, sec_story]
    tree_pub = connector.build_epic_story_mr_hierarchy(issues, allowed_scopes=base_scopes)
    assert len(tree_pub["epics"]) == 1
    pub_epic = tree_pub["epics"][0]
    assert [s["issue_key"] for s in pub_epic["stories"]] == ["ENG-11"]
    assert "acme/vault!777" not in pub_epic["all_linked_mrs"]
    assert set(pub_epic["all_linked_mrs"]) == {
        "acme/worker!42",
        "https://github.com/acme/worker/pull/99",
    }

    tree_sec = connector.build_epic_story_mr_hierarchy(issues, allowed_scopes=sec_scopes)
    assert len(tree_sec["epics"]) == 1
    assert {s["issue_key"] for s in tree_sec["epics"][0]["stories"]} == {"ENG-11", "ENG-13"}
    assert "acme/vault!777" in tree_sec["epics"][0]["all_linked_mrs"]


@pytest.mark.asyncio
async def test_doc_corpus_mcp_hybrid_rrf_and_direct_citation_verification(
    tmp_path: Path,
) -> None:
    """Replaces word-count loop with real hybrid RRF & O(1) get_doc_chunks_by_ids quote check."""
    domains_dir = tmp_path / "domains"
    domain_id = f"eval_mcp_{uuid4().hex[:8]}"
    domain_dir = domains_dir / domain_id
    domain_dir.mkdir(parents=True, exist_ok=True)
    (domain_dir / "domain.yaml").write_text(
        yaml.safe_dump(
            {
                "domain_id": domain_id,
                "display_name": "Eval Knowledge",
                "base_model_id": "google/gemma-4-26B-A4B-it",
                "contextualizer_model_id": "google/gemma-4-E4B-it",
                "lora_adapter_id": "eval_none",
                "lora_adapter_path": str(domain_dir / "adapters"),
                "embedding_model_id": "google/embeddinggemma-2",
                "embedding_dimensions": 768,
                "mcp_read_tools_allowed": [
                    "doc_corpus_mcp:search_hybrid_rrf",
                    "doc_corpus_mcp:verify_citation_quote",
                    "doc_corpus_mcp:get_document_section",
                ],
                "authority_rules": {"primary_truth_source": "document"},
                "output_contract": "DomainKnowledgeBrief",
                "lora_enabled": False,
                "default_scopes": ["public_docs"],
            }
        ),
        encoding="utf-8",
    )

    repo = InMemoryRLSRepository()
    p_eng = PrincipalContext(
        user_id="mcp_eng",
        role=RoleType.DOMAIN_RESEARCHER,
        allowed_scopes=["public_docs"],
    )
    p_other = PrincipalContext(
        user_id="mcp_other",
        role=RoleType.DOMAIN_RESEARCHER,
        allowed_scopes=["other_scope"],
    )
    embedder = get_embedding_provider()

    txt_target = (
        "Kafka partition rebalancing executes a cooperative sticky protocol "
        "without halting consumers."
    )
    txt_noise = "Kubernetes liveness probe timeout defaults to one second on idle pods."
    chunk_target = DocChunkRecord(
        chunk_id=uuid4(),
        domain_id=domain_id,
        doc_id="doc:kafka-guide",
        doc_title="Streaming Architecture Handbook",
        source_uri="docs://kafka-guide.md",
        market_scope=None,
        published_date=datetime.now(UTC).date(),
        authority_weight=1.0,
        section_heading="Cooperative Rebalancing",
        contextual_header="Architecture handbook section on Kafka consumers.",
        verbatim_content=txt_target,
        metadata_json={"source": "handbook"},
        allowed_scopes=["public_docs"],
        embedding=embedder.embed_text(txt_target, is_query=False),
        source_type="confluence",
    )
    chunk_noise = DocChunkRecord(
        chunk_id=uuid4(),
        domain_id=domain_id,
        doc_id="doc:k8s-guide",
        doc_title="Cluster Probes",
        source_uri="docs://k8s.md",
        market_scope=None,
        published_date=datetime.now(UTC).date(),
        authority_weight=1.0,
        section_heading="Probes",
        contextual_header="Cluster health probe reference.",
        verbatim_content=txt_noise,
        metadata_json={"source": "handbook"},
        allowed_scopes=["public_docs"],
        embedding=embedder.embed_text(txt_noise, is_query=False),
        source_type="confluence",
    )
    repo.insert_doc_chunks_batch(p_eng, [chunk_target, chunk_noise])

    def _unpack(res: object) -> dict[str, Any]:
        assert isinstance(res, CallToolResult)
        assert not res.is_error
        assert isinstance(res.structured_content, dict)
        return res.structured_content

    server = create_doc_corpus_mcp_server(domains_dir=domains_dir, repo=repo)
    with bind_session_principal(p_eng):
        res_search = await server.call_tool(
            "search_hybrid_rrf",
            {
                "domain_id": domain_id,
                "query": "cooperative sticky partition rebalancing consumers",
                "top_k": 5,
            },
        )
        payload = _unpack(res_search)
        assert payload["count"] >= 1
        top_hit = payload["hits"][0]
        assert top_hit["chunk_id"] == str(chunk_target.chunk_id)
        assert float(top_hit["rrf_score"]) > 0.0
        assert float(top_hit["calibrated_score"]) > 0.0
        assert top_hit["vector_rank"] is not None

        res_ver = await server.call_tool(
            "verify_citation_quote",
            {
                "domain_id": domain_id,
                "chunk_id": str(chunk_target.chunk_id),
                "verbatim_quote": "cooperative sticky protocol without halting consumers",
            },
        )
        v_payload = _unpack(res_ver)
        assert v_payload["verified"] is True
        assert v_payload["is_valid"] is True

    # Unauthorized principal fails closed on quote verification
    with bind_session_principal(p_other):
        res_unauth = await server.call_tool(
            "verify_citation_quote",
            {
                "domain_id": domain_id,
                "chunk_id": str(chunk_target.chunk_id),
                "verbatim_quote": "cooperative sticky protocol",
            },
        )
        u_payload = _unpack(res_unauth)
        assert u_payload["verified"] is False
        assert u_payload["is_valid"] is False
