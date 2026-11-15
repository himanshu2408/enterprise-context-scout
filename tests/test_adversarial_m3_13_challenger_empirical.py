"""Empirical Challenger Deep Adversarial Suite for M3 Iteration 7 Polyglot AST Plugins.

Adversarially stresses and verifies:
1. C# 11 multiline raw strings & primary constructors (csharp.py):
   - Multiline raw strings with 3, 4, 5, and 6 delimiter quotes.
   - Raw strings containing JSON payloads, unmatched braces, escaped quotes, and SQL blocks.
   - Raw strings used as default argument values inside primary constructors.
   - Deeply nested tuples (8 levels) and complex generic types in constructors.
   - Multiline primary constructors with parameter attributes ([Required], [FromQuery]).
   - Ensuring scope stacks and subsequent classes/methods are never corrupted.
2. Terraform infrastructure declarations (terraform.py):
   - Multiple heredocs with indented (<<-EOF, <<-EOT) and unindented (<<EOF) delimiters.
   - Heredocs containing bash scripts with inner sub-heredocs and unmatched braces.
   - Interpolation resolution inside heredocs (BINDS_SERVICE_ACCOUNT, ACCESSES_SECRET).
   - Multi-tier nested dynamic blocks (up to 3 levels) inside resources.
   - Ensuring subsequent resources, data blocks, and modules are never swallowed.
3. React & Redux AST extraction (react_redux.py):
   - Chained HOC wrappers: React.memo(React.forwardRef<...>(...)) and memo(forwardRef(...)).
   - Multiline HOC declarations spanning multiple lines.
   - Named arrow functions vs function declarations with TS generics and destructured props.
   - Custom hook definitions (arrow vs function) and multi-tier hook graph propagation.
   - Verifying strict boundary isolation with zero cross-component scope leakage.
"""

from __future__ import annotations

from context_scout.plugins.languages.csharp import CSharpPlugin
from context_scout.plugins.languages.react_redux import ReactReduxPlugin
from context_scout.plugins.languages.terraform import TerraformPlugin

# ==============================================================================
# 1. C# 11 Multiline Raw Strings, Nested Tuples, and Primary Constructors
# ==============================================================================


def test_adversarial_csharp_11_variable_quote_raw_strings() -> None:
    """Stress-tests C# 11 raw string literals with 3, 4, 5, and 6 quotes across multiple types.

    Empirical Challenge: Unmatched braces, JSON objects, and SQL strings within raw string literals
    must not disrupt brace depth tracking, drop methods, or corrupt subsequent type declarations.
    """
    plugin = CSharpPlugin()
    code = '''
namespace Enterprise.Polyglot.RawStrings;

public class ConfigService
{
    // 3-quote raw string with JSON and unmatched braces
    private static readonly string JsonPayload = """
    {
        "service": "scout",
        "nested": {
            "ports": [8080, 8443],
            "unmatched_open": {
        }
    }
    """;

    // 4-quote raw string with 3 quotes inside
    private string QuadQuote = """"
    Here is a block containing """triple quotes""" without breaking the delimiter.
    { "key": "value" }
    """";

    // 5-quote raw string
    public string PentQuote()
    {
        return """""
        Nested """triple""" and """"quad"""" quotes inside 5 quotes.
        } unmatched closing brace {
        """"";
    }

    public void SyncData() {}
}

public record struct SiblingRecord(int Id, string Payload)
{
    public void Process() {}
}
'''
    symbols, _edges = plugin.extract_ast(code, "ConfigService.cs")
    sym_map = {s.name: s for s in symbols}

    assert "ConfigService" in sym_map, "ConfigService symbol not found"
    assert "PentQuote" in sym_map, "PentQuote method missing"
    assert "SyncData" in sym_map, "SyncData method missing"
    assert "SiblingRecord" in sym_map, "SiblingRecord missing (swallowed by raw string?)"
    assert "Process" in sym_map, "SiblingRecord.Process method missing"

    # Verify qualified names and scopes
    config_sym = sym_map["ConfigService"]
    sibling_sym = sym_map["SiblingRecord"]
    assert "SiblingRecord" not in config_sym.qualified_name
    assert "ConfigService" not in sibling_sym.qualified_name
    assert sym_map["SyncData"].bare_name == "ConfigService.SyncData"
    assert sym_map["Process"].bare_name == "SiblingRecord.Process"


def test_adversarial_csharp_primary_constructors_nested_tuples_and_attributes() -> None:
    """Stress-tests primary constructors with deeply nested tuples, generics, and attributes.

    Empirical Challenge: Primary constructors on class, record, and struct types containing
    8-level nested tuples, complex generics, multiline layouts, and parameter attributes
    must correctly match, extract inheritance edges, and identify methods.
    """
    plugin = CSharpPlugin()
    code = """
namespace Enterprise.Polyglot.Constructors;

// Class with 8-level nested tuple and generic task callback in primary constructor
public class DeepPrimaryClass(
    [Required]
    Dictionary<string, List<(int, (string, (bool, (double, (float, (char, (byte, short)))))))>>
        complexState,
    [FromQuery] Func<string, Task<List<(int, string)>>> queryHandler,
    int retryCount = 3
) : BaseRepository<string, int>, IAuditService, IDisposable
    where T : class
{
    public Task<int> ExecuteAsync() => Task.FromResult(42);
}

// Record with multiline primary constructor and tuples
public record DeepPrimaryRecord(
    (int Id, (string Code, (bool IsActive, double Score))) hierarchy,
    string description = "default"
) : IRecordMarker;

// Struct with primary constructor
public readonly struct DeepPrimaryStruct(
    (double X, (double Y, double Z)) coordinates
) : IStructMarker
{
    public double Magnitude() => 0.0;
}
"""
    symbols, edges = plugin.extract_ast(code, "DeepPrimaryClass.cs")
    sym_names = [s.name for s in symbols]

    assert "DeepPrimaryClass" in sym_names, f"DeepPrimaryClass missing: {sym_names}"
    assert "ExecuteAsync" in sym_names, f"ExecuteAsync method missing: {sym_names}"
    assert "DeepPrimaryRecord" in sym_names, f"DeepPrimaryRecord missing: {sym_names}"
    assert "DeepPrimaryStruct" in sym_names, f"DeepPrimaryStruct missing: {sym_names}"
    assert "Magnitude" in sym_names, f"Magnitude method missing: {sym_names}"

    # Verify inheritance edges
    edge_targets = [
        e.target_qualified_name for e in edges if e.edge_kind in ("INHERITS", "USES_TYPE")
    ]
    assert "BaseRepository" in edge_targets, f"BaseRepository missing: {edge_targets}"
    assert "IAuditService" in edge_targets, f"IAuditService missing: {edge_targets}"
    assert "IDisposable" in edge_targets, f"IDisposable missing: {edge_targets}"
    assert "IRecordMarker" in edge_targets, f"IRecordMarker missing: {edge_targets}"
    assert "IStructMarker" in edge_targets, f"IStructMarker missing: {edge_targets}"


def test_adversarial_csharp_raw_strings_in_primary_constructor_defaults() -> None:
    """Stress-tests primary constructors having multiline raw strings as default parameter values.

    Empirical Challenge: Default parameter values containing multiline raw string literals with
    unmatched braces must not corrupt the primary constructor pattern or drop the class.
    """
    plugin = CSharpPlugin()
    code = '''
namespace Enterprise.Polyglot.Constructors;

public class RawParamService(
    string defaultQuery = """
        SELECT * FROM Orders
        WHERE Details LIKE '%{"open": true}%'
    """,
    int limit = 100
) : IOrderService
{
    public void Run() {}
}

public class NextService
{
    public void Step() {}
}
'''
    symbols, edges = plugin.extract_ast(code, "RawParamService.cs")
    sym_names = [s.name for s in symbols]

    assert "RawParamService" in sym_names, f"RawParamService missing: {sym_names}"
    assert "Run" in sym_names, f"Run method missing: {sym_names}"
    assert "NextService" in sym_names, f"NextService missing: {sym_names}"
    assert "Step" in sym_names, f"Step method missing: {sym_names}"

    inherits = [e.target_qualified_name for e in edges if e.edge_kind == "INHERITS"]
    assert "IOrderService" in inherits, f"IOrderService edge missing: {inherits}"


# ==============================================================================
# 2. Terraform Multiple Heredocs, Indented Delimiters, and Dynamic Blocks
# ==============================================================================


def test_adversarial_terraform_multiple_heredocs_and_deep_dynamic_blocks() -> None:
    """Stress-tests Terraform resources with multiple heredocs and 3-level nested dynamic blocks.

    Empirical Challenge: Multiple heredocs with indented (<<-EOF, <<-EOT) and unindented (<<EOF)
    delimiters, containing inner sub-heredocs and unmatched braces, must not swallow subsequent
    resources. All interpolation bindings and nested dynamic block variables must be extracted.
    """
    plugin = TerraformPlugin()
    code = """
resource "google_service_account" "api_sa" {
  account_id   = "api-runner"
  display_name = "API Runner SA"
}

resource "google_secret_manager_secret" "api_key" {
  secret_id = "external-api-key"
  replication {
    automatic = true
  }
}

resource "google_compute_instance" "gateway" {
  name         = "api-gateway"
  machine_type = "e2-standard-4"

  # Heredoc 1: Indented <<-BASH with embedded JSON heredoc and unmatched braces
  metadata_startup_script = <<-BASH
    #!/usr/bin/env bash
    set -euo pipefail
    cat << 'JSON' > /opt/app/config.json
    {
      "environment": "production",
      "features": { "rate_limiting": true,
    }
    JSON
    export SERVICE_ACCOUNT="${google_service_account.api_sa.email}"
    export SECRET_NAME="${google_secret_manager_secret.api_key.secret_id}"
    export ENV_NAME="${var.environment_name}"
  BASH

  # Heredoc 2: Indented <<-EOT with curly braces
  custom_template = <<-EOT
    template {
      param = "val"
    }
  EOT

  # Heredoc 3: Unindented <<EOF with raw quotes
  shutdown_script = <<EOF
echo "stopping { with brace"
EOF

  # 3-level nested dynamic blocks
  dynamic "network_interface" {
    for_each = var.network_interfaces
    content {
      subnetwork = network_interface.value.subnet
      dynamic "access_config" {
        for_each = network_interface.value.has_public_ip ? [1] : []
        content {
          nat_ip = null
          dynamic "dns_record" {
            for_each = var.enable_dns ? [1] : []
            content {
              ttl = 300
            }
          }
        }
      }
    }
  }

  depends_on = [
    google_service_account.api_sa
  ]
}

data "google_compute_image" "debian" {
  family  = "debian-11"
  project = "debian-cloud"
}

resource "google_storage_bucket" "log_bucket" {
  name     = "gateway-logs-bucket"
  location = "EU"
}
"""
    symbols, edges = plugin.extract_ast(code, "infra/gateway.tf")
    sym_names = [s.name for s in symbols]

    # Verify that all resources and data blocks are preserved
    assert "api_sa" in sym_names, f"api_sa missing: {sym_names}"
    assert "api_key" in sym_names, f"api_key missing: {sym_names}"
    assert "gateway" in sym_names, f"gateway missing: {sym_names}"
    assert "debian" in sym_names, f"debian data source missing: {sym_names}"
    assert "log_bucket" in sym_names, f"log_bucket missing (swallowed?): {sym_names}"

    # Verify service account binding from heredoc interpolation
    sa_edges = [e for e in edges if e.edge_kind == "BINDS_SERVICE_ACCOUNT"]
    assert any("google_service_account.api_sa" in e.target_qualified_name for e in sa_edges), (
        f"BINDS_SERVICE_ACCOUNT missing: {sa_edges}"
    )

    # Verify secret access from heredoc interpolation
    secret_edges = [e for e in edges if e.edge_kind == "ACCESSES_SECRET"]
    assert any(
        "google_secret_manager_secret.api_key" in e.target_qualified_name for e in secret_edges
    ), f"ACCESSES_SECRET missing: {secret_edges}"

    # Verify variable references from dynamic blocks and heredocs
    var_edges = [e for e in edges if e.edge_kind == "USES_TYPE"]
    var_targets = [e.target_qualified_name for e in var_edges]
    assert any("environment_name" in t for t in var_targets), (
        f"environment_name var missing: {var_targets}"
    )
    assert any("network_interfaces" in t for t in var_targets), (
        f"network_interfaces var missing: {var_targets}"
    )
    assert any("enable_dns" in t for t in var_targets), f"enable_dns var missing: {var_targets}"

    # Verify explicit depends_on
    dep_edges = [e for e in edges if e.edge_kind == "PROVISIONED_BY"]
    assert any("google_service_account.api_sa" in e.target_qualified_name for e in dep_edges), (
        f"depends_on missing: {dep_edges}"
    )


def test_adversarial_terraform_multiple_consecutive_heredocs_in_maps() -> None:
    """Stress-tests multiple consecutive heredocs inside resource attribute maps.

    Empirical Challenge: Consecutive heredocs with identical or differing delimiters inside
    a single resource map must not bleed into each other or cause brace imbalance.
    """
    plugin = TerraformPlugin()
    code = """
resource "google_compute_instance" "multi_heredoc_node" {
  name = "multi-node"

  metadata = {
    script_one = <<-EOF
      echo "script 1"
      export SA="${google_service_account.sa_one.email}"
    EOF
    script_two = <<-EOF
      echo "script 2 with unmatched { brace"
      export SA="${google_service_account.sa_two.email}"
    EOF
    script_three = <<-EOT
      echo "script 3"
      export SECRET="${google_secret_manager_secret.db_pass.secret_id}"
    EOT
  }
}

resource "google_dns_managed_zone" "zone" {
  name     = "prod-zone"
  dns_name = "prod.example.com."
}
"""
    symbols, edges = plugin.extract_ast(code, "infra/multi_heredoc.tf")
    sym_names = [s.name for s in symbols]

    assert "multi_heredoc_node" in sym_names, f"multi_heredoc_node missing: {sym_names}"
    assert "zone" in sym_names, f"zone missing (swallowed?): {sym_names}"

    sa_edges = [
        e.target_qualified_name for e in edges if e.edge_kind == "BINDS_SERVICE_ACCOUNT"
    ]
    assert any("sa_one" in t for t in sa_edges), f"sa_one binding missing: {sa_edges}"
    assert any("sa_two" in t for t in sa_edges), f"sa_two binding missing: {sa_edges}"

    sec_edges = [e.target_qualified_name for e in edges if e.edge_kind == "ACCESSES_SECRET"]
    assert any("db_pass" in t for t in sec_edges), f"db_pass secret missing: {sec_edges}"


# ==============================================================================
# 3. React & Redux Chained HOCs, Declarations, and Multi-Tier Hook Graphs
# ==============================================================================


def test_adversarial_react_redux_chained_hocs_and_generic_declarations() -> None:
    """Stress-tests chained HOC wrappers with TypeScript generics, async functions, and arrows.

    Empirical Challenge: React.memo(React.forwardRef<...>(...)), memo(forwardRef(function ...)),
    named arrow functions, and function declarations must correctly register symbols and
    bind DISPATCHES, SELECTS, and CALLS edges without dropping components.
    """
    plugin = ReactReduxPlugin()
    code = """
import React, { memo, forwardRef } from 'react';
import { useNavigation } from '../hooks/useNavigation';
import { useAnalytics } from '../hooks/useAnalytics';
import { selectCurrentUser } from '../slices/userSlice';
import { submitOrder } from '../slices/orderSlice';

// 1. Chained HOC: React.memo with React.forwardRef and nested TypeScript generics
export const ChainedGenericCard = React.memo(
    React.forwardRef<HTMLDivElement, CardProps<Record<string, unknown>>>((props, ref) => {
        const nav = useNavigation();
        const dispatch = useDispatch();
        const user = useSelector(selectCurrentUser);

        return (
            <div ref={ref}>
                <p>{user.displayName}</p>
                <button onClick={() => dispatch(submitOrder({ amount: 100 }))}>Order</button>
            </div>
        );
    })
);

// 2. Multiline Chained HOC: memo(forwardRef(...)) spanning multiple lines
export const MultilineHocCard = memo(
    forwardRef(
        function BaseCard(props: BaseCardProps, ref: React.Ref<HTMLDivElement>) {
            const analytics = useAnalytics();
            return <div ref={ref}>{analytics.status}</div>;
        }
    )
);

// 3. Named Arrow Component with destructured props and defaults
export const ArrowButton = ({ label, disabled = false }: ButtonProps) => {
    const nav = useNavigation();
    return <button disabled={disabled}>{label}</button>;
};

// 4. Function Component with explicit TypeScript generics
export function GenericList<TData extends { id: string }>(props: ListProps<TData>) {
    const analytics = useAnalytics();
    return <ul>{props.items.map(i => <li key={i.id}>{analytics.format(i)}</li>)}</ul>;
}

// 5. Custom hook: named arrow function
export const useOrderWorkflow = (orderId: string) => {
    const dispatch = useAppDispatch();
    const nav = useNavigation();
    const handleCancel = () => dispatch(submitOrder({ orderId, cancel: true }));
    return { nav, handleCancel };
};

// 6. Custom hook: function declaration
export function useUserSession() {
    const user = useAppSelector(selectCurrentUser);
    const analytics = useAnalytics();
    return { user, analytics };
}
"""
    symbols, edges = plugin.extract_ast(code, "src/components/PolyglotAdversarial.tsx")
    sym_map = {s.name: s for s in symbols}

    expected_symbols = [
        "ChainedGenericCard",
        "MultilineHocCard",
        "ArrowButton",
        "GenericList",
        "useOrderWorkflow",
        "useUserSession",
    ]
    for sym in expected_symbols:
        assert sym in sym_map, f"Expected symbol '{sym}' missing from {list(sym_map.keys())}"

    assert sym_map["useOrderWorkflow"].symbol_kind == "hook"
    assert sym_map["useUserSession"].symbol_kind == "hook"
    assert sym_map["ChainedGenericCard"].symbol_kind == "class"

    # Verify ChainedGenericCard edges
    cg_edges = [e for e in edges if "ChainedGenericCard" in e.source_qualified_name]
    cg_targets = {e.target_qualified_name: e.edge_kind for e in cg_edges}
    assert cg_targets.get("useNavigation") == "CALLS"
    assert cg_targets.get("selectCurrentUser") == "SELECTS"
    assert cg_targets.get("submitOrder") == "DISPATCHES"

    # Verify MultilineHocCard edges
    mh_edges = [e for e in edges if "MultilineHocCard" in e.source_qualified_name]
    mh_targets = {e.target_qualified_name: e.edge_kind for e in mh_edges}
    assert mh_targets.get("useAnalytics") == "CALLS"

    # Verify ArrowButton edges
    ab_edges = [e for e in edges if "ArrowButton" in e.source_qualified_name]
    assert any(e.target_qualified_name == "useNavigation" for e in ab_edges)

    # Verify GenericList edges
    gl_edges = [e for e in edges if "GenericList" in e.source_qualified_name]
    assert any(e.target_qualified_name == "useAnalytics" for e in gl_edges)

    # Verify useOrderWorkflow edges
    ow_edges = [e for e in edges if "useOrderWorkflow" in e.source_qualified_name]
    ow_targets = {e.target_qualified_name: e.edge_kind for e in ow_edges}
    assert ow_targets.get("useNavigation") == "CALLS"
    assert ow_targets.get("submitOrder") == "DISPATCHES"

    # Verify useUserSession edges
    us_edges = [e for e in edges if "useUserSession" in e.source_qualified_name]
    us_targets = {e.target_qualified_name: e.edge_kind for e in us_edges}
    assert us_targets.get("useAnalytics") == "CALLS"
    assert us_targets.get("selectCurrentUser") == "SELECTS"


def test_adversarial_react_redux_deep_hook_graph_isolation() -> None:
    """Stress-tests a 4-tier hook dependency DAG for strict lexical boundary isolation.

    Hierarchy:
    OrderDashboard (Component)
      └─ calls useOrderPage (Hook)
           ├─ calls usePaymentGateway (Hook)
           │    ├─ selects selectPaymentConfig
           │    └─ dispatches processPayment
           └─ calls useAuditLogger (Hook)
                └─ dispatches logEvent

    Empirical Challenge: No dispatches, selects, or transitive hook calls must leak into the
    outer OrderDashboard component node or intermediate hooks.
    """
    plugin = ReactReduxPlugin()
    code = """
export function useAuditLogger() {
    const dispatch = useAppDispatch();
    return {
        log: (msg: string) => dispatch(logEvent({ msg }))
    };
}

export const usePaymentGateway = () => {
    const config = useAppSelector(selectPaymentConfig);
    const dispatch = useAppDispatch();
    const pay = () => dispatch(processPayment({ amount: 50 }));
    return { config, pay };
};

export function useOrderPage() {
    const audit = useAuditLogger();
    const payment = usePaymentGateway();
    return { audit, payment };
}

export const OrderDashboard = () => {
    const page = useOrderPage();
    return <div>Dashboard</div>;
};
"""
    symbols, edges = plugin.extract_ast(code, "src/features/OrderHierarchy.tsx")
    sym_names = [s.name for s in symbols]

    assert {"useAuditLogger", "usePaymentGateway", "useOrderPage", "OrderDashboard"} <= set(
        sym_names
    )

    edge_map: dict[str, list[tuple[str, str]]] = {}
    for e in edges:
        caller = e.source_qualified_name.split(".")[-1]
        edge_map.setdefault(caller, []).append((e.edge_kind, e.target_qualified_name))

    # 1. useAuditLogger direct edges
    audit_edges = edge_map.get("useAuditLogger", [])
    assert ("DISPATCHES", "logEvent") in audit_edges

    # 2. usePaymentGateway direct edges
    pay_edges = edge_map.get("usePaymentGateway", [])
    assert ("SELECTS", "selectPaymentConfig") in pay_edges
    assert ("DISPATCHES", "processPayment") in pay_edges

    # 3. useOrderPage direct edges
    page_edges = edge_map.get("useOrderPage", [])
    assert ("CALLS", "useAuditLogger") in page_edges
    assert ("CALLS", "usePaymentGateway") in page_edges
    # Strict isolation: useOrderPage must NOT have leaked DISPATCHES or SELECTS
    assert not any(k in ("DISPATCHES", "SELECTS") for k, _ in page_edges)

    # 4. OrderDashboard direct edges
    dashboard_edges = edge_map.get("OrderDashboard", [])
    assert ("CALLS", "useOrderPage") in dashboard_edges
    # Strict isolation: OrderDashboard must NOT contain leaked transitive edges
    assert len(dashboard_edges) == 1, (
        f"OrderDashboard has leaked edges: {dashboard_edges}"
    )
