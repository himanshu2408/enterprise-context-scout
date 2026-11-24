"""Integration smoke and DB constraint parity tests for M2-02 12-language plugin suite."""

from __future__ import annotations

import json

import pytest

from context_scout.ast_engine.extractor import ASTExtractor
from context_scout.plugins import languages as lang_pkg
from context_scout.plugins.registry import PluginRegistry

ALLOWED_SYMBOL_KINDS: frozenset[str] = frozenset(
    {
        "function",
        "class",
        "method",
        "interface",
        "type_alias",
        "hook",
        "slice",
        "selector",
        "reducer",
        "story",
        "tf_resource",
        "tf_module",
        "gcp_resource",
        "component",
        "system",
        "team",
        "service",
        "module",
        "api",
        "domain",
    }
)

ALLOWED_EDGE_KINDS: frozenset[str] = frozenset(
    {
        "CALLS",
        "IMPORTS",
        "USES_TYPE",
        "INHERITS",
        "TESTED_BY",
        "PROVISIONED_BY",
        "DISPATCHES",
        "SELECTS",
        "DOCUMENTS_COMPONENT",
        "OWNED_BY_TEAM",
        "PART_OF_SYSTEM",
    }
)

EXPECTED_LANGUAGES: frozenset[str] = frozenset(
    {
        "api_specs",
        "csharp",
        "go",
        "java",
        "kotlin",
        "protobuf",
        "react_redux",
        "rust",
        "scip",
        "sql",
        "storybook",
        "templates_i18n",
        "terraform",
    }
)

EXPECTED_EXPORTS: frozenset[str] = frozenset(
    {
        "ApiSpecsPlugin",
        "CSharpPlugin",
        "GoPlugin",
        "JavaSpringPlugin",
        "KotlinPlugin",
        "ProtobufGrpcPlugin",
        "ReactReduxPlugin",
        "RustPlugin",
        "SCIPIndexPlugin",
        "SqlDbtPlugin",
        "StorybookPlugin",
        "TemplatesI18nPlugin",
        "TerraformPlugin",
    }
)


def test_languages_package_exports_and_registry_list() -> None:
    """All 13 language plugins are exported in __all__ and registered by default."""
    assert set(lang_pkg.__all__) == EXPECTED_EXPORTS
    for export_name in EXPECTED_EXPORTS:
        assert hasattr(lang_pkg, export_name)

    PluginRegistry.reset()
    supported = set(PluginRegistry.list_supported_languages())
    assert supported >= EXPECTED_LANGUAGES


@pytest.mark.parametrize(
    ("file_path", "expected_lang_id"),
    [
        ("components/Button.stories.tsx", "storybook"),
        ("components/Button.tsx", "react_redux"),
        ("emails/welcome.mjml", "templates_i18n"),
        ("locales/en.po", "templates_i18n"),
        ("src/locales/en.json", "templates_i18n"),
        ("specs/openapi.yaml", "api_specs"),
        ("specs/schema.graphql", "api_specs"),
        ("infra/main.tf", "terraform"),
        ("src/OrderService.cs", "csharp"),
        ("src/OrderController.java", "java"),
        ("src/OrderService.kt", "kotlin"),
        ("cmd/server/main.go", "go"),
        ("src/lib.rs", "rust"),
        ("models/orders.sql", "sql"),
        ("proto/orders.proto", "protobuf"),
        ("artifacts/index.scip.json", "scip"),
    ],
)
def test_registry_priority_dispatch_without_collisions(
    file_path: str, expected_lang_id: str
) -> None:
    """PluginRegistry dispatches each file path to the expected priority-ordered plugin."""
    PluginRegistry.reset()
    plugin = PluginRegistry.get_plugin_for_file(file_path)
    assert plugin is not None
    assert plugin.language_id == expected_lang_id


_SCIP_SYMBOL = "scip-go gomod example.com/orders v1.0.0 pkg/orders/OrderService#"

_SCIP_SAMPLE = json.dumps(
    {
        "metadata": {"tool_info": {"name": "scip-go", "version": "0.1.0"}},
        "documents": [
            {
                "relative_path": "pkg/orders/service.go",
                "language": "go",
                "occurrences": [
                    {
                        "range": [1, 0, 1, 12],
                        "symbol": _SCIP_SYMBOL,
                        "symbol_roles": 1,
                    }
                ],
                "symbols": [
                    {
                        "symbol": _SCIP_SYMBOL,
                        "kind": 7,
                        "display_name": "OrderService",
                    }
                ],
            }
        ],
    }
)

_PARITY_SAMPLES: list[tuple[str, str, str, str]] = [
    (
        "golang",
        "pkg/orders/service.go",
        "package orders\ntype OrderService struct{}\nfunc (s *OrderService) Create() {}\n",
        "OrderService",
    ),
    (
        "rs",
        "src/lib.rs",
        "pub struct OrderService;\nimpl OrderService { pub fn create(&self) {} }\n",
        "OrderService",
    ),
    (
        "kt",
        "src/OrderService.kt",
        "package com.example\nclass OrderService {\n    fun create() {}\n}\n",
        "OrderService",
    ),
    (
        "java",
        "src/OrderService.java",
        "package com.example;\npublic class OrderService { public void create() {} }\n",
        "OrderService",
    ),
    (
        "cs",
        "src/OrderService.cs",
        "namespace Example; public class OrderService { public void Create() {} }\n",
        "OrderService",
    ),
    (
        "proto",
        "proto/orders.proto",
        'syntax = "proto3"; package orders; message Order { string id = 1; }\n',
        "Order",
    ),
    (
        "graphql",
        "specs/schema.graphql",
        "type Order { id: ID! }\n",
        "Order",
    ),
    (
        "openapi",
        "specs/openapi.yaml",
        (
            "openapi: 3.1.0\ninfo:\n  title: Orders\n  version: '1.0'\n"
            "paths:\n  /orders:\n    get:\n      operationId: listOrders\n"
        ),
        "listOrders",
    ),
    (
        "asyncapi",
        "specs/asyncapi.yaml",
        (
            "asyncapi: 2.6.0\ninfo:\n  title: Events\n  version: '1.0'\n"
            "channels:\n  orders/created:\n    publish:\n      operationId: onOrderCreated\n"
        ),
        "orders/created",
    ),
    (
        "dbt",
        "models/orders.sql",
        "SELECT id FROM {{ ref('stg_orders') }};\n",
        "orders",
    ),
    (
        "mjml",
        "emails/welcome.mjml",
        (
            "<mjml><mj-body><mj-section><mj-column>"
            "<mj-text>Hi</mj-text></mj-column></mj-section></mj-body></mjml>"
        ),
        "mjml",
    ),
    (
        "handlebars",
        "templates/welcome.hbs",
        "<div>{{> header_partial}}</div>",
        "header_partial",
    ),
    (
        "jinja",
        "templates/email.j2",
        "{% macro render_btn(label) %}<button>{{ label }}</button>{% endmacro %}",
        "render_btn",
    ),
    (
        "i18n",
        "locales/en.po",
        'msgid "hello"\nmsgstr "Hello"\n',
        "hello",
    ),
    (
        "scip",
        "artifacts/index.scip.json",
        _SCIP_SAMPLE,
        "OrderService",
    ),
]


@pytest.mark.parametrize(("language", "file_path", "code", "expected_symbol"), _PARITY_SAMPLES)
def test_ast_extractor_parse_source_delegation_and_db_kind_constraints(
    language: str,
    file_path: str,
    code: str,
    expected_symbol: str,
) -> None:
    """ASTExtractor.parse_source delegates aliases cleanly and emits 0009-compliant kinds."""
    PluginRegistry.reset()
    extractor = ASTExtractor()
    symbols, edges = extractor.parse_source(
        code=code,
        language=language,
        file_path=file_path,
        repo_name="parity-repo",
    )
    assert extractor.last_parse_diagnostics is not None
    assert extractor.last_parse_diagnostics.has_parse_error is False
    assert extractor.last_parse_diagnostics.abstained is False
    assert any(expected_symbol in s.name for s in symbols), (
        f"Expected '{expected_symbol}' in {[s.name for s in symbols]} for {language}"
    )

    for sym in symbols:
        assert sym.symbol_kind in ALLOWED_SYMBOL_KINDS, (
            f"Disallowed symbol_kind={sym.symbol_kind!r} in {language} ({sym.name})"
        )
    for edge in edges:
        assert edge.edge_kind in ALLOWED_EDGE_KINDS, (
            f"Disallowed edge_kind={edge.edge_kind!r} in {language}"
        )


@pytest.mark.parametrize(
    ("alias", "code", "expected_symbol"),
    [
        ("golang", "package p\ntype Foo struct{}\n", "Foo"),
        ("rs", "pub struct Foo;\n", "Foo"),
        ("kt", "class Foo {\n    fun bar() {}\n}\n", "Foo"),
        ("proto", 'syntax = "proto3"; message Foo { string id = 1; }\n', "Foo"),
        ("graphql", "type Foo { id: ID! }\n", "Foo"),
        ("dbt", "CREATE VIEW foo AS SELECT id FROM {{ ref('stg_foo') }};\n", "foo"),
        ("scip", _SCIP_SAMPLE, "OrderService"),
    ],
)
def test_ast_extractor_pure_alias_resolution_without_file_path(
    alias: str, code: str, expected_symbol: str
) -> None:
    """ASTExtractor.parse_source resolves language aliases even when file_path is omitted."""
    PluginRegistry.reset()
    extractor = ASTExtractor()
    symbols, _edges = extractor.parse_source(code=code, language=alias, file_path="")
    assert extractor.last_parse_diagnostics is not None
    assert extractor.last_parse_diagnostics.has_parse_error is False
    assert any(expected_symbol in s.name for s in symbols)
