# Plugin, Domain Pack, and Connector Authoring Guide

This guide documents how to add a new language parser plugin, an external source connector plugin, a declarative domain pack, and tenant-specific staleness rules in `context_scout` without modifying core orchestrator, retrieval, or gate code.

## 1. Architecture Overview

Extension points are decoupled through runtime-checkable Python `Protocol` interfaces and declarative YAML manifests:

```text
domains/<domain_id>/domain.yaml  ──► DomainPackManifest ──► DomainRegistry
LanguagePlugin (Protocol)        ──► PluginRegistry.register_language(...)
SourceConnectorPlugin (Protocol) ──► PluginRegistry.register_connector(...)
DomainConfig / ScoutConfig       ──► PreFlightValidator.check_document_freshness(...)
```

All custom plugins should be tested against real `PluginRegistry`, `DomainRegistry`, `InMemoryRLSRepository` / `PostgresRLSRepository`, and `PreFlightValidator` instances. Always clean up global `PluginRegistry` state inside `try/finally` blocks or call `PluginRegistry.reset()` in test teardowns.

---

## 2. Authoring a Custom `LanguagePlugin`

Implement `context_scout.plugins.base.LanguagePlugin`:

- **`language_id -> str`**: Canonical identifier (for example `"toy_dsl"` or `"rust"`).
- **`file_extensions -> tuple[str, ...]`**: Lowercase suffixes including leading dot (for example `(".toy",)`).
- **`manifest_files -> tuple[str, ...]`**: Dependency manifest names or suffixes (for example `("ToyLangManifest.toml",)`).
- **`can_handle(file_path: str) -> bool`**: Extension or filename matcher.
- **`extract_ast(code: str, file_path: str, repo_name: str = "", module_prefix: str = "") -> tuple[list[ParsedSymbol], list[ParsedEdge]]`**:
  - Use `ParsedSymbol` and `ParsedEdge` from `context_scout.ast_engine.models`.
  - Use 1-indexed `start_line` / `end_line` and byte offsets matching the source text.
  - On malformed syntax that cannot be parsed safely, abstain cleanly instead of emitting fabricated symbols or edges.
- **`extract_manifest_dependencies(file_path: str, content: str) -> list[DependencyEdge]`**:
  - Return `DependencyEdge` instances from `context_scout.plugins.base`.

### Registration & Priority

Register your plugin via `context_scout.plugins.registry.PluginRegistry`:

```python
from context_scout.plugins.registry import PluginRegistry

plugin = MyToyLanguagePlugin()
PluginRegistry.register_language(plugin, priority=20)
matched = PluginRegistry.get_plugin_for_file("src/example.toy")
```

Note that `PluginRegistry.get_plugin_for_file(...)` triggers built-in initialization first (`python`, `typescript`, `csharp`, `react_redux`, `storybook`, `terraform`) and then returns the highest-priority plugin whose `can_handle(file_path)` returns `True`. Use `PluginRegistry.unregister_language(plugin.language_id)` in a `finally:` block if registering dynamically in tests.

---

## 3. Authoring a Custom `SourceConnectorPlugin`

Implement `context_scout.plugins.base.SourceConnectorPlugin`:

- **`source_id -> str`** and **`source_type -> str`**: Canonical connector names.
- **`fetch_item(item_id: str, version: str | None = None) -> dict[str, Any] | None`**: Fetches a single item from the upstream source; return `None` on HTTP 404.
- **`mirror_permissions(container_key: str) -> list[str]`**: Maps upstream space/project/repo ACLs into PostgreSQL RLS `allowed_scopes` strings using helpers in `context_scout.security.permission_mirror` (`sanitize_scope_identifier`, `PermissionMirror`).

Register the connector with:

```python
PluginRegistry.register_connector(my_connector)
connector = PluginRegistry.get_connector(my_connector.source_id)
```

For production connectors:

- Accept an optional `client: httpx.Client | None = None` in `__init__` so tests can inject `httpx.MockTransport` without monkeypatching sockets.
- Emit `DocChunkRecord` records with deterministic embeddings (`generate_deterministic_embedding`) and 50-token contextual headers so verbatim citation verification (`PreFlightValidator.check_citations`) stays lossless.

---

## 4. Authoring a Declarative Domain Pack (`domain.yaml`)

Each domain lives under `<domains_dir>/<domain_id>/domain.yaml` and validates against `context_scout.schemas.contracts.DomainPackManifest`:

- **`domain_id`**, **`display_name`**
- **`base_model_id`**, **`contextualizer_model_id`**, **`embedding_model_id`** (validated by `validate_google_model_id` to enforce Google Gemma / EmbeddingGemma models)
- **`lora_adapter_id`**, **`lora_adapter_path`**
- **`mcp_read_tools_allowed`**: Explicit read-only MCP tool allow-list enforced by `DomainRegistry.is_mcp_tool_allowed(domain_id, tool_name)`
- **`authority_rules`**: `AuthorityRules(primary_truth_source="code" | "document", require_out_of_sample_check=..., require_transaction_cost_check=..., freshness_half_life_days=...)`
- **`output_contract`**: `"ImplementationContextPack"`, `"QuantStrategyBrief"`, or `"DomainKnowledgeBrief"`

Load directories with `DomainRegistry(domains_dir=path)` (`context_scout.agents.domain_registry.DomainRegistry`).

---

## 5. Multi-Tenant Jira Staleness & Confluence Freshness Configuration

### Multi-Tenant Jira Rules (`ScoutConfig` + `DomainConfig`)

`context_scout.config.ScoutConfig` stores `default_domain_config: DomainConfig` and `domain_configs: dict[str, DomainConfig]`. Call `scout_cfg.get_domain_config(domain_id)` to retrieve per-tenant overrides cleanly.

Key Jira fields on `DomainConfig`:

- **`jira_terminal_statuses`** (default: `["Closed", "Resolved", "Archived"]`)
- **`jira_negative_resolutions`** (default: `["Won't Fix", "Duplicate", "Cannot Reproduce", "Abandoned", "Won't Do", "Rejected", "Cancelled"]`)

Behavior across connector and gate:

1. In `JiraConnectorPlugin.is_ticket_stale(issue)` (`context_scout.plugins.connectors.jira`):
   - Tickets with status `"Done"` (or resolved `"Done"` / `"Fixed"`) are treated as **fresh and valid** (`(False, None)`).
   - Tickets matching `STALE_STATUSES` or `STALE_RESOLUTIONS` (`"Won't Fix"`, `"Won't Do"`, `"Duplicate"`, `"Cancelled"`, `"Obsolete"`, etc.) are flagged stale at ingestion time (`is_stale=True`).
2. In `PreFlightValidator.check_document_freshness(...)` (`context_scout.agents.preflight_gate`):
   - Evaluates BOTH `chunk.is_stale` AND `self.domain_config.jira_negative_resolutions` against `chunk.metadata_json["resolution"]` (case-insensitive).
   - Passing a tenant-specific `DomainConfig` via `PreFlightValidator(..., domain_config=scout_cfg.get_domain_config(tenant_id))` allows different tenants to enforce custom negative resolutions (for example adding `"Superseded"` or `"Deferred"` in Tenant B while keeping Tenant A on standard rules) without touching any code in `src/`.

### Confluence Cloud REST API v2 Connector

`ConfluenceConnectorPlugin` (`context_scout.plugins.connectors.confluence`) implements:

- **`fetch_space_pages(space_key)`**: Paginates `/wiki/api/v2/pages` using cursor tokens extracted from `_links.next` (`cursor=...`), building `ConfluencePageDTO` objects with `version_number`, `body_storage`, and `last_modified_at`.
- **`chunk_page_verbatim(page, max_chunk_chars=2500)`**: Splits HTML by `<h1>`–`<h6>` headings, preserves verbatim text with a 50-token contextual header, and enforces temporal decay when `(now - page.last_modified_at).days > self.freshness_half_life_days`.
- **`sync_space_to_repository(space_key, repository, domain_id, ...)`**: Enforces version monotonicity:
  - When a newer `version_number` arrives, prior `DocChunkRecord` versions for `doc_id = f"confluence:{page.page_id}"` are marked `is_stale=True` with `stale_reason=f"Superseded by page version {page.version_number}"`.
  - Out-of-order older revisions (`page.version_number < max_existing_ver`) are treated as rollbacks (`is_rollback=True`) and do not supersede newer chunks.

See `tests/test_custom_plugin_and_multi_tenant_jira_config.py` for runnable verification of all four extension points.
