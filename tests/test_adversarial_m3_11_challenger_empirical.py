"""Empirical Challenger Deep Adversarial Suite for M3.11 Polyglot AST Plugins.

Adversarially stresses and verifies:
1. C# 11 multiline raw strings:
   - Multiple raw string literals across classes and methods in a single file.
   - Raw strings containing unescaped quotes, braces, JSON, and nested quotes.
   - Interpolated raw strings ($"\"\", $$"\"\", $$$$""\"\").
   - Deeply nested tuples (8+ levels) and complex generic types in primary constructors.
2. Terraform infrastructure declarations:
   - Multiple heredocs per resource and file.
   - Indented (<<-EOF) and unindented (<<EOF) heredoc delimiters.
   - Heredocs containing nested braces, quotes, sub-heredocs, and bash scripts.
   - Interpolation extraction inside heredocs (BINDS_SERVICE_ACCOUNT, ACCESSES_SECRET, USES_TYPE).
   - Single and deeply nested dynamic blocks inside resources.
   - Ensuring subsequent resource blocks are never swallowed.
3. React & Redux AST extraction:
   - Chained HOC wrappers (memo(forwardRef(...)), React.memo(React.forwardRef<...>(...))).
   - Multiline HOC declarations.
   - Named arrow functions vs function declarations (with TS generics, async, destructured props).
   - Custom hook definitions and multi-tier hook edge graph propagation.
"""

from __future__ import annotations

from context_scout.plugins.languages.csharp import CSharpPlugin
from context_scout.plugins.languages.react_redux import ReactReduxPlugin
from context_scout.plugins.languages.terraform import TerraformPlugin

# ==============================================================================
# 1. C# 11 Multiline Raw Strings and Deep Primary Constructors
# ==============================================================================


def test_adversarial_csharp_multiple_complex_raw_strings_in_methods() -> None:
    """Stress-tests multiple C# 11 raw string literals with unmatched braces and quotes.

    Empirical Challenge: Multiple raw strings inside multiple methods must not corrupt
    brace tracking, drop methods, or falsely nest sibling classes.
    """
    plugin = CSharpPlugin()
    code = '''
namespace Enterprise.Data;

public class RawStringStressService
{
    private string template1 = """"
        { unmatched open brace in field raw string
        nested """quotes"""
    """";

    public void QueryDatabase()
    {
        var sql = """
        SELECT * FROM Users
        WHERE Data LIKE '%{"key": "val"}%'
        AND Status IN (
            { unclosed brace
        """;
        var x = 10;
    }

    public void ProcessPayload()
    {
        var complexRaw = $$$$""""
        Line 1: {{ "quoted": 123 }}
        Line 2: { unmatched single brace in 4-dollar raw string }
        Line 3: } unmatched close brace
        """";
    }
}

public class SubsequentService
{
    public void Finalize() {}
}
'''
    symbols, _edges = plugin.extract_ast(code, "RawStringStressService.cs")
    sym_map = {s.name: s for s in symbols}

    assert "RawStringStressService" in sym_map, "RawStringStressService missing"
    assert "QueryDatabase" in sym_map, "QueryDatabase missing"
    assert "ProcessPayload" in sym_map, "ProcessPayload missing"
    assert "SubsequentService" in sym_map, "SubsequentService missing"
    assert "Finalize" in sym_map, "Finalize missing"

    subsequent = sym_map["SubsequentService"]
    assert "RawStringStressService" not in subsequent.qualified_name, (
        f"SubsequentService falsely nested: {subsequent.qualified_name}"
    )

    finalize_sym = sym_map["Finalize"]
    assert "SubsequentService.Finalize" in finalize_sym.qualified_name, (
        f"Finalize qualified name incorrect: {finalize_sym.qualified_name}"
    )

    malformed = '''
namespace Enterprise.Data;

public class RawStringInvalidDelimiter
{
    private string bad = """
        nested """quotes""" inside 3-quote delimiter
    """;
}
'''
    bad_syms, bad_edges = plugin.extract_ast(malformed, "RawStringInvalidDelimiter.cs")
    assert bad_syms == []
    assert bad_edges == []
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.has_parse_error is True
    assert plugin.last_parse_diagnostics.abstained is True


def test_adversarial_csharp_deeply_nested_tuples_and_generics_in_primary_constructors() -> None:
    """Stress-tests primary constructors with 8-level nested tuples, generics, and default values.

    Empirical Challenge: Regex matching on primary constructor parameter lists must handle
    deeply nested parentheses without recursion errors or dropping classes/records/structs.
    """
    plugin = CSharpPlugin()
    code = """
namespace Enterprise.Complex;

public class DeepTupleService(
    Dictionary<string, List<(int, (string, (bool, (double, (float, (char, (byte, short)))))))>> p,
    Func<(int, (string, bool)), Task<int>> callback,
    int timeout = default(int)
) : BaseService<string, int>, IDeepHandler, IDisposable
    where T : class
{
    public void Run() {}
}

public record DeepRecord(
    (int, (int, (int, (int, (int, (int, (int, int))))))) payload
) : IRecordMarker;

public struct DeepStruct((double, (double, double)) coords) : IStructMarker
{
    public double Calculate() => 0.0;
}
"""
    symbols, edges = plugin.extract_ast(code, "DeepTupleService.cs")
    sym_names = [s.name for s in symbols]

    assert "DeepTupleService" in sym_names, f"DeepTupleService missing: {sym_names}"
    assert "Run" in sym_names, f"Run method missing: {sym_names}"
    assert "DeepRecord" in sym_names, f"DeepRecord missing: {sym_names}"
    assert "DeepStruct" in sym_names, f"DeepStruct missing: {sym_names}"
    assert "Calculate" in sym_names, f"Calculate missing: {sym_names}"

    edge_targets = [
        e.target_qualified_name for e in edges if e.edge_kind in ("INHERITS", "USES_TYPE")
    ]
    assert "BaseService" in edge_targets, f"BaseService missing: {edge_targets}"
    assert "IDeepHandler" in edge_targets, f"IDeepHandler missing: {edge_targets}"
    assert "IDisposable" in edge_targets, f"IDisposable missing: {edge_targets}"
    assert "IRecordMarker" in edge_targets, f"IRecordMarker missing: {edge_targets}"
    assert "IStructMarker" in edge_targets, f"IStructMarker missing: {edge_targets}"

    malformed = """
namespace Enterprise.Complex;

public record DeepRecordMissingParam((int, (int, (int, int)))) : IRecordMarker;
"""
    bad_syms, bad_edges = plugin.extract_ast(malformed, "DeepRecordMissingParam.cs")
    assert bad_syms == []
    assert bad_edges == []
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.has_parse_error is True
    assert plugin.last_parse_diagnostics.abstained is True


# ==============================================================================
# 2. Terraform Multiple Heredocs, Indented Delimiters, and Dynamic Blocks
# ==============================================================================


def test_adversarial_terraform_multiple_heredocs_and_nested_dynamic_blocks() -> None:
    """Stress-tests Terraform with multiple heredocs, indented delimiters, and dynamic blocks.

    Empirical Challenge: Unmatched braces in shell script heredocs must not corrupt brace depth
    and swallow subsequent resources. Interpolations inside heredocs must resolve bindings.
    """
    plugin = TerraformPlugin()
    code = """
resource "google_service_account" "runner" {
  account_id   = "runner-sa"
  display_name = "Runner Service Account"
}

resource "google_secret_manager_secret" "db_secret" {
  secret_id = "db-secret"
  replication {
    automatic = true
  }
}

resource "google_compute_instance" "worker" {
  name         = "worker-node"
  machine_type = "n2-standard-4"

  # Heredoc 1: indented delimiter <<-EOT with nested braces and quotes
  metadata_startup_script = <<-EOT
    #!/bin/bash
    set -euo pipefail
    cat << 'JSON' > /etc/config.json
    {
      "app": {
        "env": "production",
        "port": 8080
      }
    }
    JSON
    export SA="${google_service_account.runner.email}"
    export SECRET_NAME="${google_secret_manager_secret.db_secret.secret_id}"
    export REGION="${var.gcp_region}"
  EOT

  # Heredoc 2: unindented delimiter <<EOF
  custom_data = <<EOF
line 1 with unmatched { brace
line 2 with "quotes" and 'single'
line 3 with closing }
EOF

  dynamic "network_interface" {
    for_each = var.subnets
    content {
      subnetwork = network_interface.value
      dynamic "access_config" {
        for_each = var.enable_public_ips ? [1] : []
        content {
          nat_ip = null
        }
      }
    }
  }

  depends_on = [
    google_service_account.runner
  ]
}

resource "google_storage_bucket" "final_bucket" {
  name     = "final-bucket-check"
  location = "EU"
}
"""
    symbols, edges = plugin.extract_ast(code, "infra/main.tf")
    sym_names = [s.name for s in symbols]

    assert "runner" in sym_names, f"runner missing: {sym_names}"
    assert "db_secret" in sym_names, f"db_secret missing: {sym_names}"
    assert "worker" in sym_names, f"worker missing: {sym_names}"
    assert "final_bucket" in sym_names, f"final_bucket missing (swallowed?): {sym_names}"

    # Check service account binding from heredoc interpolation
    sa_edges = [e for e in edges if e.edge_kind == "BINDS_SERVICE_ACCOUNT"]
    assert any("google_service_account.runner" in e.target_qualified_name for e in sa_edges), (
        f"BINDS_SERVICE_ACCOUNT missing: {sa_edges}"
    )

    # Check secret access from heredoc interpolation
    sec_edges = [e for e in edges if e.edge_kind == "ACCESSES_SECRET"]
    assert any(
        "google_secret_manager_secret.db_secret" in e.target_qualified_name for e in sec_edges
    ), f"ACCESSES_SECRET missing: {sec_edges}"

    # Check var references from dynamic blocks and heredocs
    var_edges = [e for e in edges if e.edge_kind == "USES_TYPE"]
    var_targets = [e.target_qualified_name for e in var_edges]
    assert any("gcp_region" in t for t in var_targets), f"gcp_region var missing: {var_targets}"
    assert any("subnets" in t for t in var_targets), f"subnets var missing: {var_targets}"
    assert any("enable_public_ips" in t for t in var_targets), (
        f"enable_public_ips var missing: {var_targets}"
    )

    # Check explicit depends_on
    dep_edges = [e for e in edges if e.edge_kind == "PROVISIONED_BY"]
    assert any("google_service_account.runner" in e.target_qualified_name for e in dep_edges), (
        f"depends_on missing: {dep_edges}"
    )


# ==============================================================================
# 3. React & Redux Chained HOCs, Declarations, and Hook Graph Propagation
# ==============================================================================


def test_adversarial_react_redux_chained_hocs_and_declarations() -> None:
    """Stress-tests chained HOC wrappers, arrow vs function, and custom hooks.

    Empirical Challenge: Chained HOCs and varied function/arrow declarations must correctly
    extract symbols and preserve DISPATCHES, SELECTS, and CALLS edges.
    """
    plugin = ReactReduxPlugin()
    code = """
import React, { memo, forwardRef } from 'react';
import { useTheme } from '../hooks/useTheme';
import { usePermissions } from '../hooks/usePermissions';
import { selectUser } from '../slices/userSlice';
import { updateProfile } from '../slices/userSlice';

// 1. Chained HOC arrow with generics
export const ChainedArrowCard = React.memo(
    React.forwardRef<HTMLDivElement, CardProps>((props, ref) => {
        const theme = useTheme();
        const dispatch = useDispatch();
        const user = useSelector(selectUser);

        return (
            <div ref={ref} className={theme}>
                <span>{user.name}</span>
                <button onClick={() => dispatch(updateProfile({ name: 'new' }))}>Update</button>
            </div>
        );
    })
);

// 2. Chained HOC function declaration
export const ChainedFuncCard = memo(forwardRef(function InnerCard(props, ref) {
    const permissions = usePermissions();
    return <div ref={ref}>{permissions.canEdit ? 'Editable' : 'ReadOnly'}</div>;
}));

// 3. Named arrow component vs function component
export const SimpleArrow = (props: SimpleProps) => {
    const theme = useTheme();
    return <div className={theme} />;
};

export function SimpleFunction(props: SimpleProps) {
    const permissions = usePermissions();
    return <div>{permissions.role}</div>;
}

// 4. Custom hooks: named arrow vs function
export const useArrowHook = (key: string) => {
    const theme = useTheme();
    return { theme, key };
};

export function useFunctionHook(userId: string) {
    const user = useAppSelector(selectUser);
    const dispatch = useAppDispatch();
    return {
        user,
        save: () => dispatch(updateProfile({ id: userId }))
    };
}
"""
    symbols, edges = plugin.extract_ast(code, "src/components/ChainedStress.tsx")
    sym_names = [s.name for s in symbols]

    assert "ChainedArrowCard" in sym_names, f"ChainedArrowCard missing: {sym_names}"
    assert "ChainedFuncCard" in sym_names, f"ChainedFuncCard missing: {sym_names}"
    assert "SimpleArrow" in sym_names, f"SimpleArrow missing: {sym_names}"
    assert "SimpleFunction" in sym_names, f"SimpleFunction missing: {sym_names}"
    assert "useArrowHook" in sym_names, f"useArrowHook missing: {sym_names}"
    assert "useFunctionHook" in sym_names, f"useFunctionHook missing: {sym_names}"

    # Verify ChainedArrowCard edges
    c_edges = [e for e in edges if "ChainedArrowCard" in e.source_qualified_name]
    c_targets = {e.target_qualified_name: e.edge_kind for e in c_edges}
    assert "useTheme" in c_targets and c_targets["useTheme"] == "CALLS"
    assert "selectUser" in c_targets and c_targets["selectUser"] == "SELECTS"
    assert "updateProfile" in c_targets and c_targets["updateProfile"] == "DISPATCHES"

    # Verify ChainedFuncCard edges
    f_edges = [e for e in edges if "ChainedFuncCard" in e.source_qualified_name]
    f_targets = {e.target_qualified_name: e.edge_kind for e in f_edges}
    assert "usePermissions" in f_targets and f_targets["usePermissions"] == "CALLS"

    # Verify SimpleArrow and SimpleFunction edges
    sa_edges = [e for e in edges if "SimpleArrow" in e.source_qualified_name]
    assert any(e.target_qualified_name == "useTheme" for e in sa_edges)

    sf_edges = [e for e in edges if "SimpleFunction" in e.source_qualified_name]
    assert any(e.target_qualified_name == "usePermissions" for e in sf_edges)

    # Verify useArrowHook and useFunctionHook edges
    ah_edges = [e for e in edges if "useArrowHook" in e.source_qualified_name]
    assert any(e.target_qualified_name == "useTheme" for e in ah_edges)

    fh_edges = [e for e in edges if "useFunctionHook" in e.source_qualified_name]
    fh_targets = {e.target_qualified_name: e.edge_kind for e in fh_edges}
    assert "selectUser" in fh_targets and fh_targets["selectUser"] == "SELECTS"
    assert "updateProfile" in fh_targets and fh_targets["updateProfile"] == "DISPATCHES"


def test_adversarial_react_redux_hook_edge_graph_propagation() -> None:
    """Stress-tests multi-tier hook graph propagation without scope leakage.

    Hierarchy: UserProfileView -> useUser -> useSession -> useAuth -> dispatch(loginSuccess)
    Verifies that each caller node is only directly connected to its immediate dependency.
    """
    plugin = ReactReduxPlugin()
    code = """
// Hook D: innermost hook
export function useAuth() {
    const token = useAppSelector(selectAuthToken);
    const dispatch = useAppDispatch();
    const handleLogin = () => dispatch(loginSuccess());
    return { token, handleLogin };
}

// Hook C: intermediate hook
export const useSession = () => {
    const auth = useAuth();
    return { session: auth.token };
};

// Hook B: outer hook
export function useUser() {
    const session = useSession();
    return { user: session };
}

// Component A: consumer
export const UserProfileView = () => {
    const user = useUser();
    return <div>{user.user}</div>;
};
"""
    symbols, edges = plugin.extract_ast(code, "src/features/UserStack.tsx")
    sym_names = [s.name for s in symbols]

    assert {"useAuth", "useSession", "useUser", "UserProfileView"} <= set(sym_names)

    edge_by_src: dict[str, list[tuple[str, str]]] = {}
    for e in edges:
        src_short = e.source_qualified_name.split(".")[-1]
        edge_by_src.setdefault(src_short, []).append((e.edge_kind, e.target_qualified_name))

    # Verify correct direct linkages
    assert ("SELECTS", "selectAuthToken") in edge_by_src.get("useAuth", [])
    assert ("DISPATCHES", "loginSuccess") in edge_by_src.get("useAuth", [])
    assert ("CALLS", "useAuth") in edge_by_src.get("useSession", [])
    assert ("CALLS", "useSession") in edge_by_src.get("useUser", [])
    assert ("CALLS", "useUser") in edge_by_src.get("UserProfileView", [])

    # Verify absence of scope leakage: UserProfileView must not have leaked dispatches or selects
    consumer_edges = edge_by_src.get("UserProfileView", [])
    assert not any(k in ("SELECTS", "DISPATCHES") for k, _ in consumer_edges), (
        f"Leaked dispatch/select in UserProfileView: {consumer_edges}"
    )
    assert not any(target == "useAuth" for _, target in consumer_edges), (
        f"Leaked transitive call in UserProfileView: {consumer_edges}"
    )
