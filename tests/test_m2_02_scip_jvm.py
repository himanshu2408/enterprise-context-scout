"""Tests for M2-02: SCIP protobuf/JSON ingestor, Java/Spring Boot, and Kotlin plugins."""

from __future__ import annotations

import base64
import json
import logging
import time

import pytest

from context_scout.plugins.base import LanguagePlugin
from context_scout.plugins.languages.java_spring import JavaSpringPlugin
from context_scout.plugins.languages.kotlin import KotlinPlugin
from context_scout.plugins.languages.scip_ingestor import (
    SYMBOL_ROLE_DEFINITION,
    SYMBOL_ROLE_IMPORT,
    SCIPIndexPlugin,
    SCIPIngestor,
    build_scip_index_bytes,
    parse_scip_symbol,
)

ALLOWED_SYMBOL_KINDS = frozenset(
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

ALLOWED_EDGE_KINDS = frozenset(
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


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_plugins_implement_language_plugin_protocol() -> None:
    scip = SCIPIndexPlugin()
    java = JavaSpringPlugin()
    kt = KotlinPlugin()

    assert isinstance(scip, LanguagePlugin)
    assert isinstance(java, LanguagePlugin)
    assert isinstance(kt, LanguagePlugin)

    assert scip.language_id == "scip"
    assert scip.can_handle("index.scip")
    assert scip.can_handle("index.scip.json")
    assert not scip.can_handle("Main.java")

    assert java.language_id == "java"
    assert java.can_handle("OrderController.java")
    assert java.can_handle("pom.xml")
    assert java.can_handle("build.gradle")

    assert kt.language_id == "kotlin"
    assert kt.can_handle("UserController.kt")
    assert kt.can_handle("build.gradle.kts")


# ---------------------------------------------------------------------------
# SCIP Ingestor tests
# ---------------------------------------------------------------------------


def test_parse_scip_symbol_grammar() -> None:
    parsed = parse_scip_symbol(
        "scip-java maven com.acme:orders 1.0.0 com/acme/OrderService#createOrder()."
    )
    assert parsed.scheme == "scip-java"
    assert parsed.manager == "maven"
    assert parsed.package == "com.acme:orders"
    assert parsed.version == "1.0.0"
    assert parsed.display_name == "createOrder"
    assert "OrderService" in parsed.qualified_name

    local_sym = parse_scip_symbol("local 42")
    assert local_sym.is_local is True
    assert local_sym.qualified_name == "local.42"


def test_scip_protobuf_roundtrip_and_edges() -> None:
    svc_sym = "scip-java maven com.acme:orders 1.0.0 com/acme/OrderService#"
    iface_sym = "scip-java maven com.acme:orders 1.0.0 com/acme/IOrderService#"
    method_sym = "scip-java maven com.acme:orders 1.0.0 com/acme/OrderService#createOrder()."
    repo_sym = "scip-java maven com.acme:orders 1.0.0 com/acme/OrderRepo#save()."
    dto_sym = "scip-java maven com.acme:orders 1.0.0 com/acme/OrderDto#"

    raw = build_scip_index_bytes(
        documents=[
            {
                "relative_path": "src/main/java/com/acme/OrderService.java",
                "symbols": [
                    {
                        "symbol": svc_sym,
                        "kind": 7,  # Class
                        "display_name": "OrderService",
                        "documentation": ["Handles order creation."],
                        "relationships": [
                            {"symbol": iface_sym, "is_implementation": True},
                        ],
                    },
                    {
                        "symbol": method_sym,
                        "kind": 26,  # Method
                        "display_name": "createOrder",
                        "relationships": [
                            {"symbol": dto_sym, "is_type_definition": True},
                        ],
                    },
                ],
                "occurrences": [
                    {
                        "range": [4, 0, 20, 1],
                        "symbol": svc_sym,
                        "symbol_roles": SYMBOL_ROLE_DEFINITION,
                    },
                    {
                        "range": [8, 4, 15, 5],
                        "symbol": method_sym,
                        "symbol_roles": SYMBOL_ROLE_DEFINITION,
                    },
                    {"range": [10, 8, 10, 20], "symbol": repo_sym, "symbol_roles": 0},
                    {"range": [1, 0, 1, 30], "symbol": dto_sym, "symbol_roles": SYMBOL_ROLE_IMPORT},
                ],
            }
        ]
    )

    ingestor = SCIPIngestor()
    symbols, edges = ingestor.ingest_bytes(raw, file_path="index.scip")
    assert not ingestor.last_parse_diagnostics.has_parse_error
    assert not ingestor.last_parse_diagnostics.abstained

    assert {s.name for s in symbols} == {"OrderService", "createOrder"}
    assert all(s.symbol_kind in ALLOWED_SYMBOL_KINDS for s in symbols)
    assert all(e.edge_kind in ALLOWED_EDGE_KINDS for e in edges)

    edge_kinds = {
        (
            e.source_qualified_name.split(".")[-1],
            e.target_qualified_name.split(".")[-1],
            e.edge_kind,
        )
        for e in edges
    }
    assert ("OrderService", "IOrderService", "INHERITS") in edge_kinds
    assert ("createOrder", "OrderDto", "USES_TYPE") in edge_kinds
    assert ("createOrder", "save", "CALLS") in edge_kinds
    assert any(e.edge_kind == "IMPORTS" for e in edges)

    # Also verify SCIPIndexPlugin.extract_ast with base64
    plugin = SCIPIndexPlugin()
    b64_syms, b64_edges = plugin.extract_ast(
        base64.b64encode(raw).decode("ascii"), file_path="index.scip"
    )
    assert len(b64_syms) == len(symbols)
    assert len(b64_edges) == len(edges)


def test_scip_json_and_fail_closed_corruption(caplog: pytest.LogCaptureFixture) -> None:
    plugin = SCIPIndexPlugin()

    # Empty input
    syms, edges = plugin.extract_ast("   ", file_path="empty.scip.json")
    assert (syms, edges) == ([], [])
    assert not plugin.last_parse_diagnostics.has_parse_error
    assert not plugin.last_parse_diagnostics.abstained

    # Valid .scip.json
    payload = json.dumps(
        {
            "metadata": {"tool_info": {"name": "scip-kotlin", "version": "0.2"}},
            "documents": [
                {
                    "relative_path": "App.kt",
                    "symbols": [
                        {
                            "symbol": "scip-kt maven app 1.0 App#",
                            "kind": "Class",
                            "display_name": "App",
                        }
                    ],
                    "occurrences": [
                        {
                            "range": [0, 0, 5, 1],
                            "symbol": "scip-kt maven app 1.0 App#",
                            "symbol_roles": 1,
                        }
                    ],
                }
            ],
        }
    )
    syms, _ = plugin.extract_ast(payload, file_path="index.scip.json")
    assert len(syms) == 1
    assert syms[0].name == "App"

    # Truncated protobuf wire bytes -> fail closed + WARNING
    with caplog.at_level(logging.WARNING):
        bad_syms, bad_edges = plugin.ingest_bytes(b"\x12\x7f\x01\x02", file_path="bad.scip")
    assert (bad_syms, bad_edges) == ([], [])
    assert plugin.last_parse_diagnostics.has_parse_error is True
    assert plugin.last_parse_diagnostics.abstained is True
    assert any("SCIP protobuf wire decode error" in r.message for r in caplog.records)

    # Invalid varint (all continuation bits) -> fail closed
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        bad_syms2, bad_edges2 = plugin.ingest_bytes(b"\x80" * 16, file_path="badvarint.scip")
    assert (bad_syms2, bad_edges2) == ([], [])
    assert plugin.last_parse_diagnostics.has_parse_error is True
    assert plugin.last_parse_diagnostics.abstained is True

    # Malformed JSON -> fail closed
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        bad_syms3, bad_edges3 = plugin.extract_ast("{not valid json", file_path="bad.scip.json")
    assert (bad_syms3, bad_edges3) == ([], [])
    assert plugin.last_parse_diagnostics.has_parse_error is True
    assert plugin.last_parse_diagnostics.abstained is True


def test_scip_100kb_linear_performance() -> None:
    docs = [
        {
            "relative_path": f"src/File{i}.java",
            "symbols": [
                {
                    "symbol": f"scip-java maven pkg 1.0 File{i}#",
                    "kind": 7,
                    "display_name": f"File{i}",
                }
            ],
            "occurrences": [
                {
                    "range": [0, 0, 10, 1],
                    "symbol": f"scip-java maven pkg 1.0 File{i}#",
                    "symbol_roles": SYMBOL_ROLE_DEFINITION,
                }
            ],
        }
        for i in range(1500)
    ]
    raw = build_scip_index_bytes(documents=docs)
    assert len(raw) >= 100_000

    import gc

    ingestor = SCIPIngestor()
    gc.collect()
    t0 = time.perf_counter()
    syms, _ = ingestor.ingest_bytes(raw, file_path="large.scip")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert len(syms) == 1500
    assert elapsed_ms < 250.0, f"SCIP 100KB ingest took {elapsed_ms:.1f}ms"


# ---------------------------------------------------------------------------
# Java & Spring Boot tests
# ---------------------------------------------------------------------------


def test_java_spring_controller_and_edges() -> None:
    code = """
package com.acme.orders;

import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.beans.factory.annotation.Autowired;

@RestController
@RequestMapping("/api/v1/orders")
public class OrderController extends BaseController implements OrderApi {
    @Autowired
    private OrderService orderService;

    public OrderController(OrderValidator validator) {
    }

    @GetMapping("/{id}")
    @Transactional
    public OrderDto getOrder(String id) {
        return orderService.findById(id);
    }

    @Test
    void testGetOrder() {
        getOrder("1");
    }
}

public record OrderDto(String id, long amount) {}
"""
    plugin = JavaSpringPlugin()
    syms, edges = plugin.extract_ast(code, file_path="OrderController.java")
    assert not plugin.last_parse_diagnostics.has_parse_error
    assert all(s.symbol_kind in ALLOWED_SYMBOL_KINDS for s in syms)
    assert all(e.edge_kind in ALLOWED_EDGE_KINDS for e in edges)

    by_name = {s.name: s for s in syms}
    assert "OrderController" in by_name
    assert "getOrder" in by_name
    assert "OrderDto" in by_name

    assert "@route GET /api/v1/orders/{id}" in by_name["getOrder"].contextual_summary
    assert "@Transactional" in by_name["getOrder"].contextual_summary
    assert "@dto OrderDto fields=id,amount" in by_name["OrderDto"].contextual_summary

    edge_tuples = {(e.source_qualified_name, e.target_qualified_name, e.edge_kind) for e in edges}
    assert ("com.acme.orders.OrderController", "BaseController", "INHERITS") in edge_tuples
    assert ("com.acme.orders.OrderController", "OrderApi", "INHERITS") in edge_tuples
    assert ("com.acme.orders.OrderController", "OrderService", "PROVISIONED_BY") in edge_tuples
    assert ("com.acme.orders.OrderController", "OrderValidator", "PROVISIONED_BY") in edge_tuples
    assert (
        "com.acme.orders.OrderController",
        "com.acme.orders.OrderController.testGetOrder",
        "TESTED_BY",
    ) in edge_tuples
    assert (
        "com.acme.orders.OrderController.getOrder",
        "orderService.findById",
        "CALLS",
    ) in edge_tuples


def test_java_manifests_pom_and_gradle() -> None:
    plugin = JavaSpringPlugin()
    pom = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <artifactId>order-service</artifactId>
  <dependencies>
    <dependency>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-starter-web</artifactId>
      <version>3.3.0</version>
    </dependency>
    <dependency>
      <groupId>org.junit.jupiter</groupId>
      <artifactId>junit-jupiter</artifactId>
      <version>5.10.2</version>
      <scope>test</scope>
    </dependency>
  </dependencies>
</project>
"""
    deps = plugin.extract_manifest_dependencies("pom.xml", pom)
    assert len(deps) == 2
    assert deps[0].target_package == "org.springframework.boot:spring-boot-starter-web"
    assert deps[0].version_constraint == "3.3.0"
    assert deps[1].scope == "dev"

    bad_deps = plugin.extract_manifest_dependencies("pom.xml", "<project><broken>")
    assert bad_deps == []
    assert plugin.last_parse_diagnostics.has_parse_error is True
    assert plugin.last_parse_diagnostics.abstained is True


def test_java_polyglot_check_error_decoys_and_perf(caplog: pytest.LogCaptureFixture) -> None:
    plugin = JavaSpringPlugin()

    # 1. Syntax error -> fail closed + WARNING
    with caplog.at_level(logging.WARNING):
        syms, edges = plugin.extract_ast(
            "public class Broken { void f( { ", file_path="Broken.java"
        )
    assert (syms, edges) == ([], [])
    assert plugin.last_parse_diagnostics.has_parse_error is True
    assert plugin.last_parse_diagnostics.abstained is True
    assert any("parse error" in r.message.lower() for r in caplog.records)

    # 2. Decoys inside comments and Java text blocks -> zero ghost symbols
    decoy_code = '''
package com.acme.decoy;
// class GhostLineComment { void ghostMethod() {} }
/* class GhostBlockComment { void ghostBlockMethod() {} } */
public class RealClass {
    private String sql = """
        public class GhostTextBlock {
            @GetMapping("/ghost")
            public void ghostEndpoint() {}
        }
        """;
}
'''
    syms, _ = plugin.extract_ast(decoy_code, file_path="RealClass.java")
    names = {s.name for s in syms}
    assert names == {"RealClass"}

    # 3. 100 KB linear-time < 250 ms
    big_methods = "\n".join(
        f"    public int m{i}(int x) {{ return x + {i}; }}" for i in range(2200)
    )
    big_code = f"package com.acme.big;\npublic class BigService {{\n{big_methods}\n}}\n"
    assert len(big_code.encode("utf-8")) >= 100_000

    t0 = time.perf_counter()
    big_syms, _ = plugin.extract_ast(big_code, file_path="BigService.java")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert len(big_syms) == 2201
    assert elapsed_ms < 250.0, f"Java 100KB parse took {elapsed_ms:.1f}ms"


# ---------------------------------------------------------------------------
# Kotlin tests
# ---------------------------------------------------------------------------


def test_kotlin_prefix_expression_and_spring_ktor() -> None:
    code = """
package com.acme.users

import org.springframework.web.bind.annotation.RestController
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.GetMapping

@RestController
@RequestMapping("/api/v1/users")
class UserController(private val svc: UserService) : BaseController() {
    companion object {
        const val MAX_USERS = 100
    }

    @GetMapping("/{id}")
    suspend fun getUser(id: String): UserDto {
        return svc.find(id)
    }

    @Test
    fun testGetUser() {
        getUser("1")
    }
}

@Service
data class UserDto(val id: String, val name: String)

sealed interface UserEvent {
    object Created : UserEvent
}
"""
    plugin = KotlinPlugin()
    syms, edges = plugin.extract_ast(code, file_path="UserController.kt")
    assert not plugin.last_parse_diagnostics.has_parse_error
    assert all(s.symbol_kind in ALLOWED_SYMBOL_KINDS for s in syms)
    assert all(e.edge_kind in ALLOWED_EDGE_KINDS for e in edges)

    by_name = {s.name: s for s in syms}
    assert {"UserController", "Companion", "getUser", "UserDto", "UserEvent", "Created"} <= set(
        by_name
    )
    assert "@RestController" in by_name["UserController"].contextual_summary
    assert "@route ANY /api/v1/users" in by_name["UserController"].contextual_summary
    assert "@route GET /api/v1/users/{id}" in by_name["getUser"].contextual_summary
    assert "@suspend" in by_name["getUser"].contextual_summary
    assert "@dto UserDto fields=id,name" in by_name["UserDto"].contextual_summary
    assert by_name["UserEvent"].symbol_kind == "interface"

    edge_tuples = {(e.source_qualified_name, e.target_qualified_name, e.edge_kind) for e in edges}
    assert ("com.acme.users.UserController", "BaseController", "INHERITS") in edge_tuples
    assert ("com.acme.users.UserController", "UserService", "PROVISIONED_BY") in edge_tuples
    assert ("com.acme.users.UserController.getUser", "UserDto", "USES_TYPE") in edge_tuples
    assert ("com.acme.users.UserController.getUser", "svc.find", "CALLS") in edge_tuples
    assert (
        "com.acme.users.UserController",
        "com.acme.users.UserController.testGetUser",
        "TESTED_BY",
    ) in edge_tuples


def test_kotlin_gradle_kts_and_polyglot_check(caplog: pytest.LogCaptureFixture) -> None:
    plugin = KotlinPlugin()

    kts = """
dependencies {
    implementation("org.springframework.boot:spring-boot-starter-web:3.3.0")
    testImplementation("io.kotest:kotest-runner-junit5:5.9.0")
}
"""
    deps = plugin.extract_manifest_dependencies("build.gradle.kts", kts)
    assert len(deps) == 2
    assert deps[0].target_package == "org.springframework.boot:spring-boot-starter-web"
    assert deps[1].scope == "dev"

    # Syntax error -> fail closed + WARNING
    with caplog.at_level(logging.WARNING):
        syms, edges = plugin.extract_ast("class Broken { fun x( ", file_path="Broken.kt")
    assert (syms, edges) == ([], [])
    assert plugin.last_parse_diagnostics.has_parse_error is True
    assert plugin.last_parse_diagnostics.abstained is True
    assert any("parse error" in r.message.lower() for r in caplog.records)

    # Decoys inside comments & raw strings -> zero ghost symbols
    decoy_code = '''
package com.acme.decoy
// class GhostLine { fun ghost() {} }
/* object GhostBlock {} */
class RealKt {
    val template = """
        class GhostRawString {
            fun ghostRaw() {}
        }
    """
}
'''
    syms, _ = plugin.extract_ast(decoy_code, file_path="RealKt.kt")
    assert {s.name for s in syms} == {"RealKt"}

    # 100 KB linear-time < 250 ms
    big_fns = "\n".join(f"    fun fn{i}(x: Int): Int = x + {i}" for i in range(2700))
    big_code = f"package com.acme.big\nclass BigKt {{\n{big_fns}\n}}\n"
    assert len(big_code.encode("utf-8")) >= 100_000

    t0 = time.perf_counter()
    big_syms, _ = plugin.extract_ast(big_code, file_path="BigKt.kt")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert len(big_syms) == 2701
    assert elapsed_ms < 250.0, f"Kotlin 100KB parse took {elapsed_ms:.1f}ms"
