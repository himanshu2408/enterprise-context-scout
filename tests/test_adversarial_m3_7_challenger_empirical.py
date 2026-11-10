"""Empirical Challenger Stress Suite for M3.7 Polyglot AST Plugins.

Adversarially stress-tests:
1. C# 12 primary constructors with deeply nested tuple parameters (3+ levels)
   and default parameters.
2. C# string literals containing unmatched braces corrupting class hierarchy.
3. React/Redux multiline generic components with object type bounds and callback types.
4. React/Redux multiline template literals with unmatched braces corrupting component scope.
5. Terraform data source references with underscores emitting phantom CALLS edges.
6. Storybook destructured imports with comments containing braces
   dropping DOCUMENTS_COMPONENT edges.
"""

from __future__ import annotations

from context_scout.plugins.languages.csharp import CSharpPlugin
from context_scout.plugins.languages.react_redux import ReactReduxPlugin
from context_scout.plugins.languages.storybook import StorybookPlugin
from context_scout.plugins.languages.terraform import TerraformPlugin

# ==============================================================================
# 1. C# 12 Adversarial Challenges
# ==============================================================================


def test_adversarial_csharp_deeply_nested_tuples_and_defaults_in_primary_constructor() -> None:
    """Challenge: C# 12 primary constructor with 3+ levels of nested tuples and default parameters.

    Failure Mode: `type_pat` regex in csharp.py:156-161 only supports 2 levels of nested parens.
    A 3-level nested tuple drops the class, its methods, and all inheritance edges.
    """
    plugin = CSharpPlugin()
    code = """
namespace Enterprise.Services;

public class DeepTupleService(
    (int Id, (string Code, (double Rate, bool IsActive) Meta) Nested) config,
    Func<(int, string), bool> filter = default(Func<(int, string), bool>),
    string name = nameof(DeepTupleService)
) : BaseService<string>, IConfigService, IDisposable
{
    public void Execute() {}
}

public class NextService
{
    public void Run() {}
}
"""
    symbols, edges = plugin.extract_ast(code, "DeepTupleService.cs")
    sym_names = [s.name for s in symbols]
    edge_targets = [e.target_qualified_name for e in edges]

    # Empirical check:
    # Under current code, DeepTupleService is completely dropped!
    # NextService is present, but DeepTupleService is missing.
    assert "DeepTupleService" in sym_names, (
        f"DEFECT CONFIRMED: DeepTupleService dropped by 3-level nested tuple: {sym_names}"
    )
    assert "Execute" in sym_names, f"Execute method dropped: {sym_names}"
    assert "IConfigService" in edge_targets, f"IConfigService missing from edges: {edge_targets}"


def test_adversarial_csharp_unmatched_brace_in_string_literal_corrupts_hierarchy() -> None:
    """Challenge: C# string literal with unmatched '{' corrupts brace_depth and nests siblings.

    Failure Mode: `csharp.py` strips comments but preserves string literals intact.
    `line_no_comment.count('{')` counts braces inside string literals, corrupting class_stack.
    """
    plugin = CSharpPlugin()
    code = """
namespace Enterprise.Core;

public class StringClass
{
    private string msg = "Unmatched brace { in string";
    public void Method1() {}
}

public class NextClass
{
    public void Method2() {}
}
"""
    symbols, _ = plugin.extract_ast(code, "StringClass.cs")
    next_cls = next((s for s in symbols if s.name == "NextClass"), None)
    assert next_cls is not None, "NextClass not found in symbols"

    # Empirical check:
    # Under current code, NextClass.qualified_name is 'Enterprise.Core.StringClass.NextClass'
    # because StringClass was never popped from class_stack!
    assert not next_cls.qualified_name.endswith(".StringClass.NextClass"), (
        f"DEFECT CONFIRMED: NextClass falsely nested inside StringClass: {next_cls.qualified_name}"
    )


# ==============================================================================
# 2. React / Redux Adversarial Challenges
# ==============================================================================


def test_adversarial_react_multiline_generics_with_object_and_callback_bounds() -> None:
    """Challenge: React component with multiline generics containing object bounds ({ id: string })
    or callback types ((item: T) => void).

    Failure Mode: `react_redux.py:138-154` stops multiline accumulation when '{', '=>', or ';'
    is encountered. Generic bounds with inline objects or callbacks cause premature termination,
    dropping the component symbol and custom hook calls.
    """
    plugin = ReactReduxPlugin()
    code = """
export const GenericGrid = <
    TData extends { id: string },
    TValue extends Array<string>
>(
    props: GridProps<TData, TValue>
) => {
    const data = useGridData<TData>();
    return <div>Grid</div>;
};

export const Sidebar = () => {
    return <aside>Sidebar</aside>;
};
"""
    symbols, edges = plugin.extract_ast(code, "src/components/GenericGrid.tsx")
    sym_names = [s.name for s in symbols]
    edge_targets = [e.target_qualified_name for e in edges]

    # Empirical check:
    # Under current code, GenericGrid is dropped completely because of '{ id: string }'.
    assert "GenericGrid" in sym_names, (
        f"DEFECT CONFIRMED: GenericGrid dropped by object type bound: {sym_names}"
    )
    assert "useGridData" in edge_targets, (
        f"DEFECT CONFIRMED: useGridData lost because GenericGrid was dropped: {edge_targets}"
    )


def test_adversarial_react_multiline_template_string_unmatched_brace() -> None:
    """Challenge: React component containing multiline template literal with unmatched '{'.

    Failure Mode: `_clean_js_line` only removes single-line backtick strings. Intermediate
    lines of a multiline template string containing '{' corrupt brace_depth, causing subsequent
    standalone dispatches to leak into the component.
    """
    plugin = ReactReduxPlugin()
    code = """
export const Banner = () => {
    const text = `
      { unmatched brace in template literal
    `;
    return <div>{text}</div>;
};

// Outside Banner
function outsideFn() {
    dispatch(outsideAction());
}
"""
    _symbols, edges = plugin.extract_ast(code, "src/components/Banner.tsx")
    dispatches = [e for e in edges if e.edge_kind == "DISPATCHES"]
    assert len(dispatches) > 0, "No DISPATCHES edges found"

    # Empirical check:
    # Under current code, outsideAction is attributed to 'components.Banner.Banner'
    # instead of 'components.Banner.standalone'.
    assert not dispatches[0].source_qualified_name.endswith(".Banner"), (
        "DEFECT CONFIRMED: Standalone dispatch leaked into Banner: "
        f"{dispatches[0].source_qualified_name}"
    )


# ==============================================================================
# 3. Terraform Adversarial Challenges
# ==============================================================================


def test_adversarial_terraform_data_source_underscore_phantom_calls_edge() -> None:
    """Challenge: Terraform data source attribute reference where data source name has underscores.

    Failure Mode: `ref_pat` in `terraform.py:191-225` matches `debian_11.self_link`.
    Because `debian_11` has an underscore, it emits a phantom CALLS edge to
    `main.debian_11.self_link`.
    """
    plugin = TerraformPlugin()
    code = """
data "google_compute_image" "debian_11" {
  family  = "debian-11"
  project = "debian-cloud"
}

resource "google_compute_instance" "vm" {
  name = "my-vm"
  boot_disk {
    initialize_params {
      image = data.google_compute_image.debian_11.self_link
    }
  }
}
"""
    _symbols, edges = plugin.extract_ast(code, "main.tf")
    call_targets = [e.target_qualified_name for e in edges if e.edge_kind == "CALLS"]

    # Empirical check:
    # Under current code, call_targets contains 'main.debian_11.self_link' as well as
    # 'main.google_compute_image.debian_11'.
    phantom_targets = [t for t in call_targets if t.endswith(".debian_11.self_link")]
    assert len(phantom_targets) == 0, (
        f"DEFECT CONFIRMED: Phantom CALLS edge emitted for data source attribute: {phantom_targets}"
    )


# ==============================================================================
# 4. Storybook Adversarial Challenges
# ==============================================================================


def test_adversarial_storybook_comment_brace_in_destructured_import() -> None:
    r"""Challenge: Storybook destructured import with block comment containing '{}' or '}'.

    Failure Mode: `imp_pat` in `storybook.py:69-72` uses `(?:\{([^}]+)\})?`.
    A comment with a brace terminates `[^}]+` prematurely, causing the entire import to fail.
    Result: candidate imports are lost and DOCUMENTS_COMPONENT edges are dropped.
    """
    plugin = StorybookPlugin()
    code = """
import {
    // Primary UI component
    Button,
    /* Multiline comment with symbols: {} */
    Card,
} from './Components';

export default {
    title: 'UI/Button',
};

export const Default = { args: {} };
"""
    _symbols, edges = plugin.extract_ast(code, "src/components/Button.stories.tsx")
    doc_edges = [e for e in edges if e.edge_kind == "DOCUMENTS_COMPONENT"]

    # Empirical check:
    # Under current code, doc_edges is empty because imp_pat failed to match!
    assert len(doc_edges) > 0, (
        "DEFECT CONFIRMED: DOCUMENTS_COMPONENT edges dropped due to comment brace in import"
    )
