"""Verify custom plugins, domain packs, multi-tenant Jira config, and Confluence v2 sync.

Verifies without any modifications to core `src/` files:
1. Custom toy LanguagePlugin + SourceConnectorPlugin registration/unregistration in PluginRegistry,
   plus custom DomainPack loading in DomainRegistry and negative controls.
2. Multi-tenant DomainConfig inside ScoutConfig driving PreFlightValidator Jira resolution rules
   simultaneously for two tenants (allowing 'Done' and isolating custom negative resolutions).
3. ConfluenceConnectorPlugin REST API v2 cursor pagination, version monotonicity protection,
   and freshness_half_life_days temporal decay enforcement.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
import yaml

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.preflight_gate import PreFlightValidator
from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol
from context_scout.config import DomainConfig, ScoutConfig
from context_scout.db.repositories import InMemoryRLSRepository
from context_scout.db.schema import DocChunkRecord
from context_scout.plugins.base import DependencyEdge, LanguagePlugin, SourceConnectorPlugin
from context_scout.plugins.connectors.confluence import ConfluenceConnectorPlugin
from context_scout.plugins.connectors.jira import JiraConnectorPlugin, JiraIssueDTO
from context_scout.plugins.registry import PluginRegistry
from context_scout.retrieval.embedder import get_embedding_provider
from context_scout.schemas.contracts import DocCitationSpan, PrincipalContext, RoleType


class _ToyWorkflowLanguagePlugin:
    """Minimal toy LanguagePlugin for '.toydsl' workflow definitions."""

    @property
    def language_id(self) -> str:
        return "toydsl"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".toydsl",)

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("ToyLangManifest.toml",)

    def can_handle(self, file_path: str) -> bool:
        lower = file_path.lower()
        return lower.endswith(".toydsl") or lower.endswith("toylangmanifest.toml")

    def extract_ast(
        self,
        code: str,
        file_path: str,
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        offset = 0
        for line_no, raw_line in enumerate(code.splitlines(keepends=True), start=1):
            stripped = raw_line.strip()
            if stripped.startswith("node "):
                node_name = stripped.split()[1].rstrip(":")
                qname = f"{module_prefix}.{node_name}" if module_prefix else node_name
                symbols.append(
                    ParsedSymbol(
                        name=node_name,
                        qualified_name=qname,
                        symbol_kind="function",
                        start_line=line_no,
                        end_line=line_no,
                        start_byte=offset,
                        end_byte=offset + len(raw_line),
                        signature=stripped,
                        signature_skeleton=stripped,
                        repo_name=repo_name,
                    )
                )
            elif stripped.startswith("calls "):
                parts = stripped.split()
                if len(parts) >= 3:
                    src = f"{module_prefix}.{parts[1]}" if module_prefix else parts[1]
                    dst = f"{module_prefix}.{parts[2]}" if module_prefix else parts[2]
                    edges.append(
                        ParsedEdge(
                            source_qualified_name=src,
                            target_qualified_name=dst,
                            edge_kind="CALLS",
                        )
                    )
            offset += len(raw_line)
        return symbols, edges

    def extract_manifest_dependencies(self, file_path: str, content: str) -> list[DependencyEdge]:
        deps: list[DependencyEdge] = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if "=" in line and not line.startswith("#") and not line.startswith("["):
                pkg, ver = [part.strip().strip('"') for part in line.split("=", maxsplit=1)]
                deps.append(
                    DependencyEdge(
                        source_identifier="toydsl_project",
                        target_package=pkg,
                        version_constraint=ver,
                        manifest_path=file_path,
                    )
                )
        return deps


class _ToyWikiConnectorPlugin:
    """Minimal toy SourceConnectorPlugin implementing the SourceConnectorPlugin SPI."""

    def __init__(self, items: dict[str, dict[str, Any]]) -> None:
        self._items = items

    @property
    def source_id(self) -> str:
        return "toy_wiki"

    @property
    def source_type(self) -> str:
        return "toy_wiki"

    def fetch_item(self, item_id: str, version: str | None = None) -> dict[str, Any] | None:
        item = self._items.get(item_id)
        if item is None:
            return None
        if version is not None and str(item.get("version")) != version:
            return None
        return item

    def mirror_permissions(self, container_key: str) -> list[str]:
        clean = container_key.strip().lower().replace("-", "_")
        return [f"toy_wiki:space:{clean}:read"]


def _make_jira_chunk(
    connector: JiraConnectorPlugin,
    *,
    domain_id: str,
    issue_key: str,
    status: str,
    resolution: str | None,
    description: str,
) -> DocChunkRecord:
    now = datetime.now(UTC)
    issue = JiraIssueDTO(
        issue_id=f"id-{issue_key}",
        issue_key=issue_key,
        project_key="ENG",
        summary=f"Summary for {issue_key}",
        issue_type="Task",
        status=status,
        resolution=resolution,
        source_url=f"https://example.atlassian.net/browse/{issue_key}",
        description_text=description,
        created_at=now - timedelta(days=2),
        updated_at=now - timedelta(days=1),
    )
    chunks = connector.chunk_issue_verbatim(issue, domain_id=domain_id)
    assert len(chunks) == 1
    return chunks[0]


def test_custom_toy_plugin_connector_and_domain_pack_without_core_edits(tmp_path: Path) -> None:
    """Custom LanguagePlugin, SourceConnectorPlugin, and DomainPack work with 0 core edits."""
    toy_lang = _ToyWorkflowLanguagePlugin()
    toy_conn = _ToyWikiConnectorPlugin(
        items={"TW-1": {"id": "TW-1", "version": "3", "title": "Toy Architecture"}}
    )
    assert isinstance(toy_lang, LanguagePlugin)
    assert isinstance(toy_conn, SourceConnectorPlugin)

    # Negative control: prior to registration, '.toydsl' and 'toy_wiki' are absent
    PluginRegistry.reset()
    try:
        assert PluginRegistry.get_plugin_for_file("pipelines/flow.toydsl") is None
        assert PluginRegistry.get_connector("toy_wiki") is None

        PluginRegistry.register_language(toy_lang, priority=25)
        PluginRegistry.register_connector(toy_conn)

        resolved_lang = PluginRegistry.get_plugin_for_file("pipelines/flow.toydsl")
        assert resolved_lang is toy_lang
        assert "toydsl" in PluginRegistry.list_supported_languages()
        assert "toy_wiki" in PluginRegistry.list_supported_connectors()

        symbols, edges = resolved_lang.extract_ast(
            "node ingest:\nnode transform:\ncalls ingest transform\n",
            "pipelines/flow.toydsl",
            repo_name="toy-repo",
            module_prefix="pipelines.flow",
        )
        assert [s.qualified_name for s in symbols] == [
            "pipelines.flow.ingest",
            "pipelines.flow.transform",
        ]
        assert len(edges) == 1
        assert edges[0].source_qualified_name == "pipelines.flow.ingest"
        assert edges[0].target_qualified_name == "pipelines.flow.transform"
        assert edges[0].edge_kind == "CALLS"

        deps = resolved_lang.extract_manifest_dependencies(
            "ToyLangManifest.toml", '[dependencies]\ntoy_core = ">=1.2.0"\n'
        )
        assert len(deps) == 1
        assert deps[0].target_package == "toy_core"
        assert deps[0].version_constraint == ">=1.2.0"

        resolved_conn = PluginRegistry.get_connector("toy_wiki")
        assert resolved_conn is toy_conn
        assert resolved_conn.fetch_item("TW-1", version="3") == {
            "id": "TW-1",
            "version": "3",
            "title": "Toy Architecture",
        }
        assert resolved_conn.fetch_item("TW-1", version="99") is None
        assert resolved_conn.mirror_permissions("Platform-Core") == [
            "toy_wiki:space:platform_core:read"
        ]
    finally:
        PluginRegistry.unregister_language("toydsl")
        PluginRegistry.reset()

    # Negative control after teardown: toy plugin is no longer active
    assert PluginRegistry.get_plugin_for_file("pipelines/flow.toydsl") is None

    # Custom DomainPack YAML loaded purely from filesystem without core edits
    domains_dir = tmp_path / "domains"
    custom_dir = domains_dir / "toy_workflow_scout"
    custom_dir.mkdir(parents=True, exist_ok=True)
    domain_yaml = {
        "domain_id": "toy_workflow_scout",
        "display_name": "Toy Workflow DSL Scout",
        "base_model_id": "google/gemma-4-26B-A4B-it",
        "contextualizer_model_id": "google/gemma-4-E4B-it",
        "lora_adapter_id": "toy-workflow-lora-v1",
        "lora_adapter_path": str(custom_dir / "adapter"),
        "embedding_model_id": "google/embeddinggemma-2",
        "embedding_dimensions": 768,
        "mcp_read_tools_allowed": ["toy_wiki_read", "ast_slice_read"],
        "authority_rules": {
            "primary_truth_source": "code",
            "require_out_of_sample_check": False,
            "require_transaction_cost_check": False,
            "freshness_half_life_days": 90,
        },
        "output_contract": "DomainKnowledgeBrief",
    }
    (custom_dir / "domain.yaml").write_text(yaml.safe_dump(domain_yaml), encoding="utf-8")

    domain_registry = DomainRegistry(domains_dir=domains_dir)
    manifest = domain_registry.get("toy_workflow_scout")
    assert manifest.display_name == "Toy Workflow DSL Scout"
    assert manifest.authority_rules.freshness_half_life_days == 90
    assert domain_registry.is_mcp_tool_allowed("toy_workflow_scout", "toy_wiki_read") is True
    assert domain_registry.is_mcp_tool_allowed("toy_workflow_scout", "write_prod_db") is False


def test_multi_tenant_jira_status_and_resolution_config_isolation() -> None:
    """Two simultaneous tenant DomainConfigs enforce distinct Jira resolution rules."""
    scout_cfg = ScoutConfig(
        domain_configs={
            "tenant_alpha": DomainConfig(
                domain_id="tenant_alpha",
                jira_terminal_statuses=["Closed", "Resolved", "Archived"],
                jira_negative_resolutions=["Won't Fix", "Won't Do", "Duplicate", "Cancelled"],
            ),
            "tenant_beta": DomainConfig(
                domain_id="tenant_beta",
                jira_terminal_statuses=["Closed", "Archived"],
                jira_negative_resolutions=[
                    "Won't Fix",
                    "Won't Do",
                    "Duplicate",
                    "Cancelled",
                    "Superseded",
                    "Deferred",
                ],
            ),
        }
    )

    connector = JiraConnectorPlugin()
    repo = InMemoryRLSRepository()
    principal = PrincipalContext(
        user_id="tenant_tester",
        role=RoleType.ENGINEER,
        allowed_scopes=JiraConnectorPlugin().mirror_permissions("ENG"),
    )

    # 'Done' must ALWAYS remain fresh/valid across connector & both tenants
    done_issue_chunk = _make_jira_chunk(
        connector,
        domain_id="tenant_alpha",
        issue_key="ENG-101",
        status="Done",
        resolution="Done",
        description="Implemented deterministic cache key normalization for gateway requests.",
    )
    assert done_issue_chunk.is_stale is False

    # Negative-resolution tickets must be flagged stale by connector
    for idx, neg_res in enumerate(["Won't Fix", "Won't Do", "Duplicate", "Cancelled"], start=102):
        stale_chunk = _make_jira_chunk(
            connector,
            domain_id="tenant_alpha",
            issue_key=f"ENG-{idx}",
            status="Closed",
            resolution=neg_res,
            description=f"Rejected proposal with resolution {neg_res}.",
        )
        assert stale_chunk.is_stale is True
        assert stale_chunk.stale_reason is not None

    # Custom resolution 'Superseded': non-stale at connector, rejected ONLY in tenant_beta
    superseded_chunk_alpha = _make_jira_chunk(
        connector,
        domain_id="tenant_alpha",
        issue_key="ENG-201",
        status="Resolved",
        resolution="Superseded",
        description="Legacy retry policy replaced by token-bucket limiter.",
    )
    superseded_chunk_beta = _make_jira_chunk(
        connector,
        domain_id="tenant_beta",
        issue_key="ENG-201",
        status="Resolved",
        resolution="Superseded",
        description="Legacy retry policy replaced by token-bucket limiter.",
    )
    done_chunk_beta = _make_jira_chunk(
        connector,
        domain_id="tenant_beta",
        issue_key="ENG-101",
        status="Done",
        resolution="Done",
        description="Implemented deterministic cache key normalization for gateway requests.",
    )

    for chunk in (done_issue_chunk, superseded_chunk_alpha, superseded_chunk_beta, done_chunk_beta):
        repo.insert_doc_chunk(principal, chunk)

    validator_alpha = PreFlightValidator(
        repo=repo,
        domain_id="tenant_alpha",
        domain_config=scout_cfg.get_domain_config("tenant_alpha"),
    )
    validator_beta = PreFlightValidator(
        repo=repo,
        domain_id="tenant_beta",
        domain_config=scout_cfg.get_domain_config("tenant_beta"),
    )
    bound_repo = validator_alpha._resolve_graph_repo(repo)

    def _cite(chunk: DocChunkRecord) -> DocCitationSpan:
        return DocCitationSpan(
            chunk_id=str(chunk.chunk_id),
            doc_id=chunk.doc_id,
            doc_title=chunk.doc_title,
            source_uri=chunk.source_uri,
            section_heading=chunk.section_heading,
            verbatim_quote=chunk.verbatim_content,
        )

    # Both tenants allow status='Done' / resolution='Done'
    ok_done_a, errs_done_a, _ = validator_alpha.check_document_freshness(
        [_cite(done_issue_chunk)], [done_issue_chunk], bound_repo, principal, "tenant_alpha"
    )
    ok_done_b, errs_done_b, _ = validator_beta.check_document_freshness(
        [_cite(done_chunk_beta)], [done_chunk_beta], bound_repo, principal, "tenant_beta"
    )
    assert ok_done_a is True and errs_done_a == []
    assert ok_done_b is True and errs_done_b == []

    # Tenant A permits 'Superseded'; Tenant B rejects it via custom jira_negative_resolutions
    ok_sup_a, errs_sup_a, _ = validator_alpha.check_document_freshness(
        [_cite(superseded_chunk_alpha)],
        [superseded_chunk_alpha],
        bound_repo,
        principal,
        "tenant_alpha",
    )
    ok_sup_b, errs_sup_b, _ = validator_beta.check_document_freshness(
        [_cite(superseded_chunk_beta)],
        [superseded_chunk_beta],
        bound_repo,
        principal,
        "tenant_beta",
    )
    assert ok_sup_a is True and errs_sup_a == []
    assert ok_sup_b is False
    assert errs_sup_b == [f"negative_resolution:{superseded_chunk_beta.chunk_id}:superseded"]


def test_confluence_v2_pagination_version_monotonicity_and_temporal_decay() -> None:
    """ConfluenceConnectorPlugin paginates v2 cursors, guards monotonicity, and flags decay."""
    now = datetime.now(UTC)
    fresh_ts = (now - timedelta(days=5)).isoformat().replace("+00:00", "Z")
    decayed_ts = (now - timedelta(days=120)).isoformat().replace("+00:00", "Z")
    seen_cursors: list[str | None] = []

    def _transport_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/wiki/api/v2/pages":
            return httpx.Response(404, json={"error": "not found"})
        cursor = request.url.params.get("cursor")
        seen_cursors.append(cursor)
        if cursor is None:
            payload = {
                "results": [
                    {
                        "id": "1001",
                        "title": "Service Runbook",
                        "version": {"number": 2, "authorId": "alice", "createdAt": fresh_ts},
                        "body": {
                            "storage": {
                                "value": (
                                    "<h2>Health Checks</h2><p>Verify port 8000 responds 200.</p>"
                                )
                            }
                        },
                    }
                ],
                "_links": {"next": "/wiki/api/v2/pages?space-key=ENG&cursor=page2tok"},
            }
            return httpx.Response(200, content=json.dumps(payload))
        if cursor == "page2tok":
            payload = {
                "results": [
                    {
                        "id": "1002",
                        "title": "Legacy Runbook",
                        "version": {"number": 1, "authorId": "bob", "createdAt": decayed_ts},
                        "body": {
                            "storage": {
                                "value": "<h2>Deprecated Steps</h2><p>Old deployment steps.</p>"
                            }
                        },
                    }
                ],
                "_links": {},
            }
            return httpx.Response(200, content=json.dumps(payload))
        return pytest.fail(f"Unexpected Confluence pagination cursor: {cursor}")

    client = httpx.Client(
        base_url="https://example.atlassian.net",
        transport=httpx.MockTransport(_transport_handler),
    )
    connector = ConfluenceConnectorPlugin(
        base_url="https://example.atlassian.net",
        client=client,
        freshness_half_life_days=90,
    )

    pages = connector.fetch_space_pages("ENG")
    assert seen_cursors == [None, "page2tok"]
    assert [p.page_id for p in pages] == ["1001", "1002"]

    fresh_chunks = connector.chunk_page_verbatim(pages[0])
    decayed_chunks = connector.chunk_page_verbatim(pages[1])
    assert len(fresh_chunks) == 1
    assert fresh_chunks[0].is_stale is False
    assert "Health Checks" in fresh_chunks[0].contextual_header

    assert len(decayed_chunks) == 1
    assert decayed_chunks[0].is_stale is True
    assert decayed_chunks[0].stale_reason is not None
    assert "> threshold of 90 days" in decayed_chunks[0].stale_reason

    # Version monotonicity in InMemoryRLSRepository: v1 -> v2 supersedes v1; v1 rollback keeps v2
    repo = InMemoryRLSRepository()
    principal = PrincipalContext(
        user_id="conf_syncer",
        role=RoleType.ENGINEER,
        allowed_scopes=connector.mirror_permissions("ENG"),
    )
    v1_chunk = DocChunkRecord(
        chunk_id=uuid4(),
        domain_id="eval_confluence_sync",
        doc_id="confluence:1001",
        doc_title="Service Runbook",
        source_uri="https://example.atlassian.net/wiki/spaces/ENG/pages/1001",
        market_scope=None,
        published_date=now.date(),
        authority_weight=1.0,
        section_heading="Health Checks",
        contextual_header="Version 1",
        verbatim_content="Old v1 content.",
        metadata_json={"version_number": 1},
        allowed_scopes=connector.mirror_permissions("ENG"),
        embedding=get_embedding_provider(dimensions=768).embed_text(
            "Old v1 content.", is_query=False
        ),
        source_type="confluence",
        version_str="1",
        is_stale=False,
    )
    repo.insert_doc_chunk(principal, v1_chunk)

    seen_cursors.clear()
    sync_res = connector.sync_space_to_repository(
        space_key="ENG",
        repository=repo,
        domain_id="eval_confluence_sync",
        principal=principal,
    )
    assert sync_res.stale_items_flagged == 1
    assert repo._docs[v1_chunk.chunk_id].is_stale is True
    assert repo._docs[v1_chunk.chunk_id].stale_reason == "Superseded by page version 2"

    # Now simulate stale v1 returned for page 1001 (rollback); v2 chunk must stay fresh
    def _rollback_handler(request: httpx.Request) -> httpx.Response:
        payload = {
            "results": [
                {
                    "id": "1001",
                    "title": "Service Runbook",
                    "version": {"number": 1, "authorId": "alice", "createdAt": fresh_ts},
                    "body": {"storage": {"value": "<p>Stale v1 replay.</p>"}},
                }
            ],
            "_links": {},
        }
        return httpx.Response(200, content=json.dumps(payload))

    rollback_connector = ConfluenceConnectorPlugin(
        base_url="https://example.atlassian.net",
        client=httpx.Client(
            base_url="https://example.atlassian.net",
            transport=httpx.MockTransport(_rollback_handler),
        ),
        freshness_half_life_days=90,
    )
    rollback_res = rollback_connector.sync_space_to_repository(
        space_key="ENG",
        repository=repo,
        domain_id="eval_confluence_sync",
        principal=principal,
    )
    assert rollback_res.stale_items_flagged == 0
    v2_chunks = [
        d for d in repo._docs.values() if d.doc_id == "confluence:1001" and d.version_str == "2"
    ]
    assert len(v2_chunks) == 1
    assert v2_chunks[0].is_stale is False
