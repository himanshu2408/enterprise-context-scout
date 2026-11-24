"""Verification suite for Go, Rust, and C# SCIP parity plugins (M2-02)."""

from __future__ import annotations

import logging
import time

import pytest

from context_scout.plugins.base import LanguagePlugin
from context_scout.plugins.languages.csharp import CSharpPlugin
from context_scout.plugins.languages.go import GoPlugin
from context_scout.plugins.languages.rust import RustPlugin

_ALLOWED_SYMBOL_KINDS = frozenset(
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

_ALLOWED_EDGE_KINDS = frozenset(
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


def test_go_plugin_ast_routes_goroutines_and_gomod() -> None:
    """Verifies Go AST extraction, routes, goroutines, embedding, tests, and go.mod."""
    plugin = GoPlugin()
    assert isinstance(plugin, LanguagePlugin)
    assert plugin.language_id == "go"
    assert plugin.file_extensions == (".go",)
    assert plugin.manifest_files == ("go.mod", "go.sum")
    assert plugin.can_handle("cmd/api/orders.go")

    code = """package orders

import (
\t"net/http"
\t"github.com/go-chi/chi/v5"
)

type BaseEntity struct {
\tID string `json:"id"`
}

type OrderDto struct {
\tBaseEntity
\tOrderID string `json:"order_id"`
\tAmount  int64  `json:"amount"`
}

type OrderRepo interface {
\tSave(o OrderDto) error
}

type OrderID = string

func (o *OrderDto) Validate() error {
\treturn nil
}

func RegisterRoutes(r chi.Router) {
\thttp.HandleFunc("/health", HealthCheck)
\tr.Get("/orders", ListOrders)
\tr.POST("/orders", CreateOrder)
\tgo processAsync(r)
}

func TestCreateOrder(t *testing.T) {
\tCreateOrder(nil)
}
"""
    syms, edges = plugin.extract_ast(code, "internal/orders/orders.go", repo_name="orders-svc")
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.parser_backend == "tree-sitter-go"
    assert plugin.last_parse_diagnostics.has_parse_error is False
    assert plugin.last_parse_diagnostics.abstained is False

    by_name = {s.name: s for s in syms}
    assert {
        "BaseEntity",
        "OrderDto",
        "OrderRepo",
        "OrderID",
        "Validate",
        "RegisterRoutes",
        "TestCreateOrder",
    } <= set(by_name)

    assert by_name["BaseEntity"].symbol_kind == "class"
    assert "@dto BaseEntity fields=id" in by_name["BaseEntity"].docstring
    assert by_name["OrderDto"].symbol_kind == "class"
    assert "@dto OrderDto fields=order_id,amount" in by_name["OrderDto"].docstring
    assert by_name["OrderRepo"].symbol_kind == "interface"
    assert by_name["OrderID"].symbol_kind == "type_alias"
    assert by_name["Validate"].symbol_kind == "method"
    assert by_name["Validate"].qualified_name == "orders.OrderDto.Validate"

    reg_doc = by_name["RegisterRoutes"].docstring
    assert "@route ANY /health" in reg_doc
    assert "@route GET /orders" in reg_doc
    assert "@route POST /orders" in reg_doc
    assert "[goroutine] processAsync" in reg_doc

    assert all(s.symbol_kind in _ALLOWED_SYMBOL_KINDS for s in syms)
    assert all(e.edge_kind in _ALLOWED_EDGE_KINDS for e in edges)

    edge_tuples = {(e.source_qualified_name, e.target_qualified_name, e.edge_kind) for e in edges}
    assert ("orders.OrderDto", "BaseEntity", "INHERITS") in edge_tuples
    assert ("orders.OrderDto", "net/http", "IMPORTS") in edge_tuples
    assert ("orders.RegisterRoutes", "processAsync", "CALLS") in edge_tuples
    assert ("orders.CreateOrder", "orders.TestCreateOrder", "TESTED_BY") in edge_tuples
    assert ("orders.OrderRepo", "OrderDto", "USES_TYPE") in edge_tuples

    gomod = """module github.com/acme/orders

go 1.22

require (
\tgithub.com/go-chi/chi/v5 v5.0.12
\tgithub.com/gin-gonic/gin v1.9.1 // indirect
)

require github.com/stretchr/testify v1.9.0
"""
    deps = plugin.extract_manifest_dependencies("go.mod", gomod)
    assert len(deps) == 3
    assert {d.target_package for d in deps} == {
        "github.com/go-chi/chi/v5",
        "github.com/gin-gonic/gin",
        "github.com/stretchr/testify",
    }
    assert all(d.source_identifier == "github.com/acme/orders" for d in deps)


def test_go_plugin_polyglot_checks(caplog: pytest.LogCaptureFixture) -> None:
    """Verifies Go empty input, fail-closed syntax error, decoy resistance, and 100KB <250ms."""
    plugin = GoPlugin()

    assert plugin.extract_ast("   \n\t ", "empty.go") == ([], [])
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.has_parse_error is False
    assert plugin.last_parse_diagnostics.abstained is False

    with caplog.at_level(logging.WARNING):
        syms, edges = plugin.extract_ast("package broken\nfunc ((( {", "broken.go")
    assert (syms, edges) == ([], [])
    assert plugin.last_parse_diagnostics.has_parse_error is True
    assert plugin.last_parse_diagnostics.abstained is True
    assert any("abstaining" in r.message.lower() for r in caplog.records)

    assert plugin.extract_manifest_dependencies("go.mod", "require ( broken") == []
    assert plugin.last_parse_diagnostics.abstained is True

    decoy_code = """package decoys
// func FakeCommentFunc() {}
/* type FakeBlockStruct struct {} */
const raw = `
func FakeRawStringFunc() {}
type FakeRawStruct struct {}
`
func RealFunc() {}
"""
    d_syms, _ = plugin.extract_ast(decoy_code, "decoys.go")
    d_names = {s.name for s in d_syms}
    assert d_names == {"RealFunc"}

    block = "func Fn_{i}() int {{ return {i} }}\n"
    payload = "package perf\n" + "".join(block.format(i=i) for i in range(3500))
    assert len(payload.encode("utf-8")) > 100_000
    t0 = time.perf_counter()
    p_syms, _ = plugin.extract_ast(payload, "perf.go")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert len(p_syms) == 3500
    assert elapsed_ms < 250.0


def test_rust_plugin_ast_traits_macros_and_cargo_toml() -> None:
    """Verifies Rust AST extraction, traits, impls, macros, routes, tests, and Cargo.toml."""
    plugin = RustPlugin()
    assert isinstance(plugin, LanguagePlugin)
    assert plugin.language_id == "rust"
    assert plugin.file_extensions == (".rs",)
    assert plugin.manifest_files == ("Cargo.toml",)
    assert plugin.can_handle("src/orders.rs")

    code = """use std::collections::HashMap;
use actix_web::{get, web, HttpResponse};

pub struct OrderDto {
    pub id: String,
    pub amount: i64,
}

pub enum OrderStatus {
    Pending,
    Paid,
}

pub trait OrderStore {
    fn save(&self, order: &OrderDto) -> Result<(), String>;
}

pub type OrderId = String;

impl OrderDto {
    pub fn validate(&self) -> bool {
        self.amount > 0
    }
}

impl OrderStore for InMemoryStore {
    fn save(&self, order: &OrderDto) -> Result<(), String> {
        Ok(())
    }
}

#[get("/orders/{id}")]
pub async fn get_order(path: web::Path<String>) -> HttpResponse {
    let _ = validate_internal(&path);
    HttpResponse::Ok().finish()
}

macro_rules! define_route {
    ($name:ident) => { fn $name() {} };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_order() {
        get_order(todo!());
    }
}
"""
    syms, edges = plugin.extract_ast(code, "src/orders.rs", repo_name="rust-orders")
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.parser_backend == "tree-sitter-rust"
    assert plugin.last_parse_diagnostics.has_parse_error is False
    assert plugin.last_parse_diagnostics.abstained is False

    by_name = {s.name: s for s in syms}
    assert {
        "OrderDto",
        "OrderStatus",
        "OrderStore",
        "OrderId",
        "validate",
        "save",
        "get_order",
        "define_route",
        "test_get_order",
    } <= set(by_name)

    assert by_name["OrderDto"].symbol_kind == "class"
    assert "@dto OrderDto fields=id,amount" in by_name["OrderDto"].docstring
    assert by_name["OrderStatus"].symbol_kind == "class"
    assert by_name["OrderStore"].symbol_kind == "interface"
    assert by_name["OrderId"].symbol_kind == "type_alias"
    assert by_name["validate"].symbol_kind == "method"
    assert by_name["validate"].qualified_name == "orders::OrderDto::validate"
    assert by_name["define_route"].symbol_kind == "function"
    assert "@route GET /orders/{id}" in by_name["get_order"].docstring

    assert all(s.symbol_kind in _ALLOWED_SYMBOL_KINDS for s in syms)
    assert all(e.edge_kind in _ALLOWED_EDGE_KINDS for e in edges)

    edge_tuples = {(e.source_qualified_name, e.target_qualified_name, e.edge_kind) for e in edges}
    assert ("orders::InMemoryStore", "OrderStore", "INHERITS") in edge_tuples
    assert ("orders::get_order", "validate_internal", "CALLS") in edge_tuples
    assert ("orders::tests::get_order", "orders::tests::test_get_order", "TESTED_BY") in edge_tuples
    assert ("orders::OrderDto", "actix_web::get", "IMPORTS") in edge_tuples

    cargo_toml = """[package]
name = "rust-orders"
version = "0.1.0"

[dependencies]
actix-web = "4.9"
serde = { version = "1.0", features = ["derive"] }

[dev-dependencies]
tokio = { version = "1.40", features = ["full"] }

[build-dependencies]
cc = "1.1"
"""
    deps = plugin.extract_manifest_dependencies("Cargo.toml", cargo_toml)
    assert len(deps) == 4
    by_pkg = {d.target_package: d for d in deps}
    assert by_pkg["actix-web"].scope == "runtime"
    assert by_pkg["serde"].version_constraint == "1.0"
    assert by_pkg["tokio"].scope == "dev"
    assert by_pkg["cc"].scope == "compile"


def test_rust_plugin_polyglot_checks(caplog: pytest.LogCaptureFixture) -> None:
    """Verifies Rust empty input, fail-closed syntax error, decoy resistance, and 100KB <250ms."""
    plugin = RustPlugin()

    assert plugin.extract_ast("  \n ", "empty.rs") == ([], [])
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.has_parse_error is False
    assert plugin.last_parse_diagnostics.abstained is False

    with caplog.at_level(logging.WARNING):
        syms, edges = plugin.extract_ast("fn broken( {{{", "broken.rs")
    assert (syms, edges) == ([], [])
    assert plugin.last_parse_diagnostics.has_parse_error is True
    assert plugin.last_parse_diagnostics.abstained is True
    assert any("abstaining" in r.message.lower() for r in caplog.records)

    assert plugin.extract_manifest_dependencies("Cargo.toml", "[dependencies\nbroken") == []
    assert plugin.last_parse_diagnostics.abstained is True

    decoy_code = """
// fn fake_comment_fn() {}
/* struct FakeBlockStruct {} */
const DECOY: &str = r#"
fn fake_raw_string_fn() {}
pub trait FakeRawTrait {}
"#;
pub fn real_fn() {}
"""
    d_syms, _ = plugin.extract_ast(decoy_code, "src/decoys.rs")
    assert {s.name for s in d_syms} == {"real_fn"}

    block = "pub fn fn_{i}() -> i32 {{ {i} }}\n"
    payload = "".join(block.format(i=i) for i in range(3800))
    assert len(payload.encode("utf-8")) > 100_000
    t0 = time.perf_counter()
    p_syms, _ = plugin.extract_ast(payload, "src/perf.rs")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert len(p_syms) == 3800
    assert elapsed_ms < 250.0


def test_csharp_scip_parity_and_polyglot_checks(caplog: pytest.LogCaptureFixture) -> None:
    """Verifies C# SCIP symbol/occurrence parity and /polyglot-plugin-check invariants."""
    plugin = CSharpPlugin()
    code = """
namespace EShop.Ordering.API;

public interface IOrderService {
    void PlaceOrder();
}

public class OrderService : IOrderService {
    public void PlaceOrder() {}
}
"""
    syms, _ = plugin.extract_ast(code, "Ordering/OrderService.cs", repo_name="eshop-orders")
    occs = plugin.extract_scip_occurrences(
        code,
        "Ordering/OrderService.cs",
        repo_name="eshop-orders",
        package_name="EShop.Ordering",
        package_version="2.1.0",
    )
    assert len(occs) == len(syms)
    scip_by_qn = {o["qualified_name"]: o["symbol"] for o in occs}
    assert (
        scip_by_qn["EShop.Ordering.API.IOrderService"]
        == "scip-dotnet nuget EShop.Ordering 2.1.0 EShop/Ordering/API/IOrderService#"
    )
    assert (
        scip_by_qn["EShop.Ordering.API.OrderService"]
        == "scip-dotnet nuget EShop.Ordering 2.1.0 EShop/Ordering/API/OrderService#"
    )
    assert (
        scip_by_qn["EShop.Ordering.API.OrderService.PlaceOrder"]
        == "scip-dotnet nuget EShop.Ordering 2.1.0 EShop/Ordering/API/OrderService#PlaceOrder()."
    )

    with caplog.at_level(logging.WARNING):
        bad_syms, bad_edges = plugin.extract_ast("public class Broken { {{", "Broken.cs")
    assert (bad_syms, bad_edges) == ([], [])
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.has_parse_error is True
    assert plugin.last_parse_diagnostics.abstained is True
    assert any("abstaining" in r.message.lower() for r in caplog.records)

    block = "    public int Fn{i}() => {i};\n"
    payload = (
        "namespace Perf;\npublic class Large {\n"
        + "".join(block.format(i=i) for i in range(3600))
        + "}\n"
    )
    assert len(payload.encode("utf-8")) > 100_000
    t0 = time.perf_counter()
    p_syms, _ = plugin.extract_ast(payload, "Perf.cs")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert len(p_syms) == 3601
    assert elapsed_ms < 250.0
