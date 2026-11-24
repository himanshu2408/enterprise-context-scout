"""Tests for M2.02 contracts & data plugins: SqlDbtPlugin, ProtobufGrpcPlugin, ApiSpecsPlugin."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol
from context_scout.plugins.base import LanguagePlugin
from context_scout.plugins.languages.api_specs import ApiSpecsPlugin
from context_scout.plugins.languages.protobuf_grpc import ProtobufGrpcPlugin
from context_scout.plugins.languages.sql_dbt import SqlDbtPlugin

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

REPO_ROOT = Path(__file__).resolve().parents[1]


def _assert_valid_kinds(symbols: list[ParsedSymbol], edges: list[ParsedEdge]) -> None:
    for sym in symbols:
        assert sym.symbol_kind in ALLOWED_SYMBOL_KINDS, f"Invalid symbol_kind: {sym.symbol_kind}"
    for edge in edges:
        assert edge.edge_kind in ALLOWED_EDGE_KINDS, f"Invalid edge_kind: {edge.edge_kind}"


# ---------------------------------------------------------------------------
# SqlDbtPlugin
# ---------------------------------------------------------------------------


def test_sql_dbt_protocol_and_empty_input() -> None:
    plugin = SqlDbtPlugin()
    assert isinstance(plugin, LanguagePlugin)
    assert plugin.language_id == "sql"
    assert plugin.can_handle("models/orders.sql")
    assert plugin.can_handle("dbt_project.yml")
    assert plugin.can_handle("packages.yml")
    assert plugin.can_handle("models/schema.yml")
    assert not plugin.can_handle("main.py")

    syms, edges = plugin.extract_ast("   \n\t ", "empty.sql")
    assert syms == []
    assert edges == []
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.has_parse_error is False
    assert plugin.last_parse_diagnostics.abstained is False


@pytest.mark.parametrize(
    "rel_path,expected_tables",
    [
        (
            "sql/001_init_schema.sql",
            {
                "domain_packs",
                "code_files",
                "ast_symbols",
                "ast_edges",
                "doc_chunks",
                "execution_telemetry",
            },
        ),
        (
            "sql/002_production_enterprise_schema.sql",
            {
                "repositories",
                "repo_commits",
                "confluence_pages",
                "jira_issues",
                "indexing_queue",
                "indexing_dead_letter",
            },
        ),
    ],
)
def test_sql_dbt_postgres17_repo_schemas(rel_path: str, expected_tables: set[str]) -> None:
    plugin = SqlDbtPlugin()
    sql_text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    syms, edges = plugin.extract_ast(sql_text, rel_path, repo_name="scout")

    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.has_parse_error is False
    assert plugin.last_parse_diagnostics.abstained is False
    assert plugin.last_parse_diagnostics.low_confidence_fallback is False
    _assert_valid_kinds(syms, edges)

    table_names = {s.name for s in syms if s.symbol_kind == "class"}
    assert expected_tables.issubset(table_names)
    assert any(e.edge_kind == "USES_TYPE" for e in edges)


def test_sql_dbt_bigquery_and_cte_lineage() -> None:
    plugin = SqlDbtPlugin()
    bq_sql = """
    CREATE OR REPLACE TABLE `analytics.mart.daily_revenue` (
        order_date DATE,
        tags ARRAY<STRING>,
        total_usd FLOAT64
    )
    OPTIONS(description="Daily revenue summary");

    CREATE VIEW `analytics.mart.v_recent_revenue` AS
    WITH base_orders AS (
        SELECT order_date, amount FROM `analytics.raw.orders`
    )
    SELECT order_date, SUM(amount) AS total
    FROM base_orders
    QUALIFY ROW_NUMBER() OVER (PARTITION BY order_date ORDER BY total DESC) = 1;
    """
    syms, edges = plugin.extract_ast(bq_sql, "models/mart/daily_revenue.sql")
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.has_parse_error is False
    assert plugin.last_parse_diagnostics.abstained is False
    _assert_valid_kinds(syms, edges)

    names = {s.name for s in syms}
    assert "analytics.mart.daily_revenue" in names
    assert "analytics.mart.v_recent_revenue" in names
    assert "base_orders" in names
    assert any(
        e.edge_kind == "USES_TYPE" and "analytics.raw.orders" in e.target_qualified_name
        for e in edges
    )


def test_sql_dbt_jinja_refs_sources_and_syntax_error() -> None:
    plugin = SqlDbtPlugin()
    dbt_sql = """
    {{ config(materialized='table') }}
    {% set lookback_days = 7 %}
    WITH src_events AS (
        SELECT * FROM {{ source('raw', 'events') }}
    )
    SELECT e.id, u.email
    FROM src_events e
    JOIN {{ ref('stg_users') }} u ON e.user_id = u.id
    {% if lookback_days > 0 %}
    WHERE e.created_at >= CURRENT_DATE
    {% endif %}
    """
    syms, edges = plugin.extract_ast(dbt_sql, "models/marts/fct_events.sql")
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.has_parse_error is False
    assert plugin.last_parse_diagnostics.abstained is False
    _assert_valid_kinds(syms, edges)

    targets = {(e.edge_kind, e.target_qualified_name) for e in edges}
    assert ("IMPORTS", "dbt::source::raw.events") in targets
    assert ("IMPORTS", "dbt::ref::stg_users") in targets
    assert ("USES_TYPE", "dbt::config") in targets

    # Corrupted Jinja fails closed
    bad_jinja = "SELECT * FROM {{ ref('unclosed' }}"
    b_syms, b_edges = plugin.extract_ast(bad_jinja, "models/bad.sql")
    assert b_syms == []
    assert b_edges == []
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.has_parse_error is True
    assert plugin.last_parse_diagnostics.abstained is True


def test_sql_dbt_manifest_dependencies() -> None:
    plugin = SqlDbtPlugin()
    packages_yml = """
    packages:
      - package: dbt-labs/dbt_utils
        version: 1.1.1
      - git: https://github.com/dbt-labs/dbt-codegen.git
        revision: 0.12.1
    """
    deps = plugin.extract_manifest_dependencies("packages.yml", packages_yml)
    assert len(deps) == 2
    assert {d.target_package for d in deps} == {
        "dbt-labs/dbt_utils",
        "https://github.com/dbt-labs/dbt-codegen.git",
    }

    bad_deps = plugin.extract_manifest_dependencies("packages.yml", "packages:\n  - [unclosed")
    assert bad_deps == []
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.has_parse_error is True
    assert plugin.last_parse_diagnostics.abstained is True


def test_sql_dbt_polyglot_checks(caplog: pytest.LogCaptureFixture) -> None:
    plugin = SqlDbtPlugin()

    # 2. Corrupted SQL fails closed with WARNING
    with caplog.at_level(logging.WARNING):
        syms, edges = plugin.extract_ast("CREAT TABL broken (( SELECT FROM;", "broken.sql")
    assert syms == []
    assert edges == []
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.has_parse_error is True
    assert plugin.last_parse_diagnostics.abstained is True

    # 3. Decoy tokens in comments and string literals produce zero ghost symbols
    decoy_sql = """
    -- CREATE TABLE ghost_comment_table (id INT);
    /* CREATE FUNCTION ghost_block_fn() RETURNS INT AS $$ SELECT 1; $$ LANGUAGE SQL; */
    SELECT 'CREATE TABLE ghost_string_table (id INT);' AS note;
    """
    d_syms, _ = plugin.extract_ast(decoy_sql, "decoy.sql")
    d_names = {s.name for s in d_syms}
    assert "ghost_comment_table" not in d_names
    assert "ghost_block_fn" not in d_names
    assert "ghost_string_table" not in d_names

    # 4. 100 KB linear-time bound (< 250 ms)
    stmt = "CREATE TABLE tbl_{i} (id INT PRIMARY KEY, name TEXT NOT NULL);\n"
    big_sql = "".join(stmt.format(i=i) for i in range(1800))
    assert len(big_sql.encode("utf-8")) > 100_000
    t0 = time.perf_counter()
    b_syms, _ = plugin.extract_ast(big_sql, "big.sql")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert len(b_syms) == 1800
    assert elapsed_ms < 250.0, f"SQL 100KB took {elapsed_ms:.1f} ms"


# ---------------------------------------------------------------------------
# ProtobufGrpcPlugin
# ---------------------------------------------------------------------------


def test_protobuf_grpc_extraction_and_manifest() -> None:
    plugin = ProtobufGrpcPlugin()
    assert isinstance(plugin, LanguagePlugin)
    assert plugin.language_id == "protobuf"
    assert plugin.can_handle("api/billing/v1/billing.proto")
    assert plugin.can_handle("buf.yaml")
    assert plugin.can_handle("buf.lock")
    assert not plugin.can_handle("billing.py")

    proto_code = """
    syntax = "proto3";
    package acme.billing.v1;

    import "google/protobuf/timestamp.proto";

    enum InvoiceStatus {
      INVOICE_STATUS_UNSPECIFIED = 0;
      INVOICE_STATUS_PAID = 1;
    }

    message InvoiceRequest {
      string invoice_id = 1;
      oneof target {
        string user_id = 2;
        string org_id = 3;
      }
      message LineItem {
        string sku = 1;
      }
    }

    message InvoiceResponse {
      InvoiceStatus status = 1;
    }

    service BillingService {
      rpc GetInvoice(InvoiceRequest) returns (InvoiceResponse);
      rpc StreamInvoices(stream InvoiceRequest) returns (stream InvoiceResponse);
    }
    """
    syms, edges = plugin.extract_ast(proto_code, "acme/billing/v1/billing.proto", repo_name="core")
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.has_parse_error is False
    assert plugin.last_parse_diagnostics.abstained is False
    _assert_valid_kinds(syms, edges)

    by_name = {s.name: s for s in syms}
    assert by_name["InvoiceStatus"].symbol_kind == "type_alias"
    assert by_name["InvoiceRequest"].symbol_kind == "class"
    assert "@dto InvoiceRequest" in by_name["InvoiceRequest"].contextual_summary
    assert "LineItem" in by_name
    assert "target" in by_name
    assert by_name["BillingService"].symbol_kind == "service"
    assert by_name["GetInvoice"].symbol_kind == "method"
    assert (
        "@route POST /acme.billing.v1.BillingService/GetInvoice"
        in by_name["GetInvoice"].contextual_summary
    )
    assert "streaming=unary" in by_name["GetInvoice"].contextual_summary
    assert "streaming=bidi_streaming" in by_name["StreamInvoices"].contextual_summary

    assert any(
        e.edge_kind == "IMPORTS" and e.target_qualified_name == "google/protobuf/timestamp.proto"
        for e in edges
    )
    assert any(
        e.edge_kind == "USES_TYPE" and "InvoiceRequest" in e.target_qualified_name for e in edges
    )
    assert any(
        e.edge_kind == "USES_TYPE" and "InvoiceResponse" in e.target_qualified_name for e in edges
    )

    buf_yaml = """
    version: v1
    name: buf.build/acme/billing
    deps:
      - buf.build/googleapis/googleapis
    """
    deps = plugin.extract_manifest_dependencies("buf.yaml", buf_yaml)
    assert len(deps) == 1
    assert deps[0].target_package == "buf.build/googleapis/googleapis"


def test_protobuf_grpc_polyglot_checks() -> None:
    plugin = ProtobufGrpcPlugin()

    # Corrupted proto fails closed
    syms, edges = plugin.extract_ast("message Broken { string x = ; }", "broken.proto")
    assert syms == []
    assert edges == []
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.has_parse_error is True
    assert plugin.last_parse_diagnostics.abstained is True

    # Decoy tokens in comments/strings
    decoy_proto = """
    syntax = "proto3";
    // message GhostLineComment { string id = 1; }
    /* service GhostBlockService { rpc Nope(A) returns (B); } */
    message RealMsg {
      string note = 1 [deprecated = false];
    }
    """
    d_syms, _ = plugin.extract_ast(decoy_proto, "decoy.proto")
    d_names = {s.name for s in d_syms}
    assert "RealMsg" in d_names
    assert "GhostLineComment" not in d_names
    assert "GhostBlockService" not in d_names

    # 100 KB linear-time bound (< 250 ms)
    msg_tmpl = "message Msg{i} {{ string id = 1; int64 count = 2; }}\n"
    big_proto = 'syntax = "proto3";\n' + "".join(msg_tmpl.format(i=i) for i in range(2200))
    assert len(big_proto.encode("utf-8")) > 100_000
    t0 = time.perf_counter()
    b_syms, _ = plugin.extract_ast(big_proto, "big.proto")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert len(b_syms) == 2200
    assert elapsed_ms < 250.0, f"Proto 100KB took {elapsed_ms:.1f} ms"


# ---------------------------------------------------------------------------
# ApiSpecsPlugin
# ---------------------------------------------------------------------------


def test_api_specs_can_handle_routing() -> None:
    plugin = ApiSpecsPlugin()
    assert isinstance(plugin, LanguagePlugin)
    assert plugin.can_handle("schema.graphql")
    assert plugin.can_handle("queries/user.gql")
    assert plugin.can_handle("api/openapi.yaml")
    assert plugin.can_handle("docs/swagger.json")
    assert plugin.can_handle("events/asyncapi.yml")
    assert plugin.can_handle("specs/orders.yaml")
    # Must NOT hijack unrelated YAML/JSON manifests
    assert not plugin.can_handle("domain.yaml")
    assert not plugin.can_handle("package.json")
    assert not plugin.can_handle("locales/en.json")
    assert not plugin.can_handle("infra/main.tf.json")


def test_api_specs_graphql_openapi_and_asyncapi() -> None:
    plugin = ApiSpecsPlugin()

    gql = """
    interface Node { id: ID! }
    enum Role { ADMIN MEMBER }
    type User implements Node {
      id: ID!
      role: Role!
    }
    union SearchResult = User
    input CreateUserInput { email: String! }
    type Query {
      user(id: ID!): User
    }
    type Mutation {
      createUser(input: CreateUserInput!): User!
    }
    """
    g_syms, g_edges = plugin.extract_ast(gql, "schema.graphql")
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.has_parse_error is False
    _assert_valid_kinds(g_syms, g_edges)

    g_by_name = {s.name: s for s in g_syms}
    assert g_by_name["Node"].symbol_kind == "interface"
    assert g_by_name["Role"].symbol_kind == "type_alias"
    assert g_by_name["User"].symbol_kind == "class"
    assert "@graphql query user" in g_by_name["user"].contextual_summary
    assert "@graphql mutation createUser" in g_by_name["createUser"].contextual_summary
    assert any(e.edge_kind == "INHERITS" and "Node" in e.target_qualified_name for e in g_edges)
    assert any(e.edge_kind == "USES_TYPE" and "Role" in e.target_qualified_name for e in g_edges)

    openapi_yaml = """
    openapi: "3.0.3"
    info:
      title: BillingAPI
      version: "1.0.0"
    paths:
      /invoices/{id}:
        get:
          operationId: getInvoice
          responses:
            "200":
              description: OK
              content:
                application/json:
                  schema:
                    $ref: "#/components/schemas/Invoice"
    components:
      schemas:
        Invoice:
          type: object
          properties:
            id:
              type: string
            customer:
              $ref: "#/components/schemas/Customer"
        Customer:
          type: object
          properties:
            name:
              type: string
    """
    o_syms, o_edges = plugin.extract_ast(openapi_yaml, "openapi.yaml")
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.has_parse_error is False
    _assert_valid_kinds(o_syms, o_edges)
    o_by_name = {s.name: s for s in o_syms}
    assert o_by_name["BillingAPI"].symbol_kind == "api"
    assert "@route GET /invoices/{id}" in o_by_name["getInvoice"].contextual_summary
    assert "@dto Invoice" in o_by_name["Invoice"].contextual_summary
    assert any(
        e.edge_kind == "USES_TYPE" and "Customer" in e.target_qualified_name for e in o_edges
    )

    asyncapi_json = """
    {
      "asyncapi": "2.6.0",
      "info": {"title": "OrderEvents", "version": "1.0.0"},
      "channels": {
        "orders.created": {
          "publish": {
            "message": {"$ref": "#/components/messages/OrderCreated"}
          }
        }
      },
      "components": {
        "messages": {
          "OrderCreated": {
            "payload": {"$ref": "#/components/schemas/OrderPayload"}
          }
        },
        "schemas": {
          "OrderPayload": {
            "type": "object",
            "properties": {"orderId": {"type": "string"}}
          }
        }
      }
    }
    """
    a_syms, a_edges = plugin.extract_ast(asyncapi_json, "asyncapi.json")
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.has_parse_error is False
    _assert_valid_kinds(a_syms, a_edges)
    a_names = {s.name for s in a_syms}
    assert {"OrderEvents", "orders.created", "OrderCreated", "OrderPayload"}.issubset(a_names)
    assert any(
        e.edge_kind == "USES_TYPE" and "OrderPayload" in e.target_qualified_name for e in a_edges
    )


def test_api_specs_polyglot_checks() -> None:
    plugin = ApiSpecsPlugin()

    # Corrupted GraphQL, YAML, JSON all fail closed
    for bad_code, path in [
        ("type Broken { id: ", "bad.graphql"),
        ("openapi: [unclosed", "openapi.yaml"),
        ('{"openapi": "3.0.0", "bad": ', "openapi.json"),
    ]:
        syms, edges = plugin.extract_ast(bad_code, path)
        assert syms == []
        assert edges == []
        assert plugin.last_parse_diagnostics is not None
        assert plugin.last_parse_diagnostics.has_parse_error is True
        assert plugin.last_parse_diagnostics.abstained is True

    # Decoy tokens in GraphQL comments/descriptions
    decoy_gql = '''
    # type GhostCommentType { id: ID! }
    """
    type GhostDocstringType { id: ID! }
    """
    type RealType { id: ID! }
    '''
    d_syms, _ = plugin.extract_ast(decoy_gql, "decoy.graphql")
    d_names = {s.name for s in d_syms}
    assert "RealType" in d_names
    assert "GhostCommentType" not in d_names
    assert "GhostDocstringType" not in d_names

    # 100 KB linear-time bound (< 250 ms)
    type_tmpl = "type Model{i} {{ id: ID! name: String! count: Int! }}\n"
    big_gql = "".join(type_tmpl.format(i=i) for i in range(2100))
    assert len(big_gql.encode("utf-8")) > 100_000
    t0 = time.perf_counter()
    b_syms, _ = plugin.extract_ast(big_gql, "big.graphql")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert len(b_syms) == 2100
    assert elapsed_ms < 250.0, f"GraphQL 100KB took {elapsed_ms:.1f} ms"
