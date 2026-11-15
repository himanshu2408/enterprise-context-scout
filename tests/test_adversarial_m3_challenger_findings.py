"""Empirical Challenger stress tests for Milestone 3 polyglot plugins.

This suite demonstrates reproducible defects and edge case failures in:
1. C# multiline primary constructors with nested parentheses / tuples.
2. C# comment brace counting corrupting class scopes.
3. C# block comments in base clauses emitting phantom inheritance edges.
4. React/Redux multiline generic parameters dropping component symbols and hooks.
5. React/Redux multiline block comments corrupting component scope.
6. Terraform data source reference edge mismatch.
7. Storybook comments inside multiline destructured imports emitting
   'DOCUMENTS_COMPONENT -> //'.
"""

from __future__ import annotations

from context_scout.plugins.languages.csharp import CSharpPlugin
from context_scout.plugins.languages.react_redux import ReactReduxPlugin
from context_scout.plugins.languages.storybook import StorybookPlugin
from context_scout.plugins.languages.terraform import TerraformPlugin

# ==============================================================================
# 1. C# Challenger Stress Tests & Reproductions
# ==============================================================================


def test_challenger_csharp_primary_constructor_with_tuple_parameter_fails() -> None:
    """Demonstrates that C# tuple parameter in primary constructor drops the class and method."""
    plugin = CSharpPlugin()
    code = """
namespace Enterprise.Services;

public class GeoService(
    (double Lat, double Lon) coordinates,
    ILogger<GeoService> logger
) : IGeoService
{
    public void Locate() {}
}
"""
    symbols, _edges = plugin.extract_ast(code, "GeoService.cs")
    sym_names = [s.name for s in symbols]
    # DEFECT: Because of [^)]* in type_pat, GeoService is completely dropped!
    msg = f"GeoService dropped by tuple primary constructor: {sym_names}"
    assert "GeoService" in sym_names, msg
    assert "Locate" in sym_names, f"Locate method dropped: {sym_names}"


def test_challenger_csharp_primary_constructor_with_default_parentheses_fails() -> None:
    """Demonstrates that default(...) or nameof(...) in primary constructor drops the class."""
    plugin = CSharpPlugin()
    code = """
namespace Enterprise.Services;

public class OrderProcessor(
    IOrderRepository repo,
    CancellationToken ct = default(CancellationToken)
) : IOrderProcessor
{
    public void Process() {}
}
"""
    symbols, _edges = plugin.extract_ast(code, "OrderProcessor.cs")
    sym_names = [s.name for s in symbols]
    # DEFECT: default(CancellationToken) terminates the regex on first closing paren.
    assert "OrderProcessor" in sym_names, f"OrderProcessor dropped by default(ct): {sym_names}"


def test_challenger_csharp_comment_brace_corrupts_class_hierarchy() -> None:
    """Demonstrates that a comment with { corrupts brace_depth and nests sibling classes."""
    plugin = CSharpPlugin()
    code = """
namespace Enterprise.Core;

public class OuterService
{
    // Note: { opening brace inside comment
    public void Method1() {}
}

public class NextService
{
    public void Method2() {}
}
"""
    symbols, _ = plugin.extract_ast(code, "Services.cs")
    next_svc = next((s for s in symbols if s.name == "NextService"), None)
    assert next_svc is not None, "NextService not found"
    # DEFECT: line 186 counts braces in comments, causing NextService to be nested in OuterService!
    err = f"NextService falsely nested in OuterService: {next_svc.qualified_name}"
    assert "OuterService" not in next_svc.qualified_name, err


def test_challenger_csharp_base_clause_block_comment_emits_phantom_edges() -> None:
    """Demonstrates that block comments in base clause emit phantom INHERITS edges."""
    plugin = CSharpPlugin()
    code = """
namespace Enterprise.Repository;

public class Repository<T> :
    IRepository<T>, // Primary data access
    IReadOnlyRepository<T>, /* audit trail */
    IDisposable
{
    public void Dispose() {}
}
"""
    _symbols, edges = plugin.extract_ast(code, "Repository.cs")
    inherits_targets = [e.target_qualified_name for e in edges if e.edge_kind == "INHERITS"]
    # DEFECT: 'audit' and 'trail' are extracted as inherited interfaces!
    assert "audit" not in inherits_targets, f"Phantom inheritance edge 'audit': {inherits_targets}"
    assert "trail" not in inherits_targets, f"Phantom inheritance edge 'trail': {inherits_targets}"


# ==============================================================================
# 2. React / Redux Challenger Stress Tests & Reproductions
# ==============================================================================


def test_challenger_react_multiline_generic_parameters_drops_component() -> None:
    """Demonstrates that Prettier-style multiline generics drop the component and its hooks."""
    plugin = ReactReduxPlugin()
    code = """
export const ComplexTable = <
    TData extends Record<string, unknown>,
    TValue,
    TMeta
>(props: TableProps<TData, TValue>) => {
    const data = useTableData();
    return <div>Table</div>;
};
"""
    symbols, edges = plugin.extract_ast(code, "ComplexTable.tsx")
    sym_names = [s.name for s in symbols]
    edge_targets = [e.target_qualified_name for e in edges]
    # DEFECT: comp_pat is evaluated per-line; < on its own line fails to match.
    assert "ComplexTable" in sym_names, f"ComplexTable dropped by multiline generics: {sym_names}"
    assert "useTableData" in edge_targets, f"useTableData hook lost: {edge_targets}"


def test_challenger_react_multiline_block_comment_brace_corrupts_scope() -> None:
    """Demonstrates that a multiline block comment with { corrupts component stack."""
    plugin = ReactReduxPlugin()
    code = """
export const MyComp = () => {
    /*
      { unmatched brace in multiline comment
    */
    return <div>comp</div>;
};

// Outside MyComp
function helper() {
    dispatch(doSomething());
}
"""
    _symbols, edges = plugin.extract_ast(code, "src/components/MyComp.tsx")
    dispatches = [e for e in edges if e.edge_kind == "DISPATCHES"]
    assert len(dispatches) == 1, f"Expected 1 DISPATCHES edge, got {dispatches}"
    assert "doSomething" in dispatches[0].target_qualified_name
    err = (
        f"Dispatch from standalone function leaked to MyComp: {dispatches[0].source_qualified_name}"
    )
    assert not dispatches[0].source_qualified_name.endswith(".MyComp"), err


# ==============================================================================
# 3. Terraform Challenger Stress Tests & Reproductions
# ==============================================================================


def test_challenger_terraform_data_source_references_broken_edge() -> None:
    """Demonstrates that data source references emit phantom data.google_* target names."""
    plugin = TerraformPlugin()
    code = """
data "google_compute_image" "debian" {
    family  = "debian-11"
    project = "debian-cloud"
}

resource "google_compute_instance" "vm" {
    name = "my-vm"
    boot_disk {
        initialize_params {
            image = data.google_compute_image.debian.self_link
        }
    }
}
"""
    symbols, edges = plugin.extract_ast(code, "main.tf")
    sym_names = {s.bare_name for s in symbols}
    assert "google_compute_image.debian" in sym_names

    call_edges = [e for e in edges if e.edge_kind == "CALLS"]
    assert len(call_edges) == 1, f"Expected 1 CALLS edge, got {call_edges}"
    # DEFECT: Target emitted is 'main.data.google_compute_image' instead of target symbol
    edge_target = call_edges[0].target_qualified_name
    err = f"Data source reference target broken: {edge_target}"
    assert edge_target.endswith("google_compute_image.debian"), err


# ==============================================================================
# 4. Storybook Challenger Stress Tests & Reproductions
# ==============================================================================


def test_challenger_storybook_comment_in_destructured_import_emits_slash_slash() -> None:
    """Demonstrates that a comment inside multiline import emits DOCUMENTS_COMPONENT -> //."""
    plugin = StorybookPlugin()
    code = """
import {
    // Primary UI component
    Button,
    /* Props definition */
    ButtonProps
} from "./Button";

export default {
    title: "Components/Button"
};
"""
    _symbols, edges = plugin.extract_ast(code, "Button.stories.tsx")
    doc_targets = [e.target_qualified_name for e in edges if e.edge_kind == "DOCUMENTS_COMPONENT"]
    # DEFECT: '//' or '/*' is extracted as the documented component!
    assert "//" not in doc_targets, f"Extracted '//' as component target: {doc_targets}"
    assert "/*" not in doc_targets, f"Extracted '/*' as component target: {doc_targets}"
    assert "Button" in doc_targets, f"Button missing from documented targets: {doc_targets}"
