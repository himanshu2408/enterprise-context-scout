"""Adversarial stress tests for Milestone 3 polyglot language plugins and registry.

Empirically tests edge cases, malformed syntax, priority resolution, cross-language hijacking,
scope leaks, and AST extraction correctness across C#, React/Redux, Storybook, and Terraform.
"""

from __future__ import annotations

from context_scout.ast_engine.extractor import ASTExtractor
from context_scout.plugins.languages.csharp import CSharpPlugin
from context_scout.plugins.languages.react_redux import ReactReduxPlugin
from context_scout.plugins.languages.storybook import StorybookPlugin
from context_scout.plugins.languages.terraform import TerraformPlugin
from context_scout.plugins.registry import PluginRegistry

# ==============================================================================
# 1. Plugin Registry & Dispatch Hijacking Challenges
# ==============================================================================


def test_registry_does_not_hijack_non_javascript_files() -> None:
    """Verifies that ReactReduxPlugin does not hijack Python, Go, C#, or Terraform files.

    Challenge: `ReactReduxPlugin.can_handle` matches keywords like 'store', 'hook',
    'component', 'feature', 'slice', 'reducer' without verifying the file extension.
    """
    python_file = "backend/services/webhook.py"
    terraform_file = "terraform/modules/feature_store/main.tf"
    csharp_file = "src/Components/PaymentGateway.cs"
    go_file = "backend/slice_allocator.go"

    p_py = PluginRegistry.get_plugin_for_file(python_file)
    p_tf = PluginRegistry.get_plugin_for_file(terraform_file)
    p_cs = PluginRegistry.get_plugin_for_file(csharp_file)
    p_go = PluginRegistry.get_plugin_for_file(go_file)

    # None of these should resolve to ReactReduxPlugin
    assert p_py is None or p_py.language_id != "react_redux", (
        f"{python_file} was hijacked by {p_py.language_id if p_py else None}"
    )
    assert p_tf is not None and p_tf.language_id == "terraform", (
        f"{terraform_file} was hijacked by {p_tf.language_id if p_tf else None}"
    )
    assert p_cs is not None and p_cs.language_id == "csharp", (
        f"{csharp_file} was hijacked by {p_cs.language_id if p_cs else None}"
    )
    assert p_go is None or p_go.language_id != "react_redux", (
        f"{go_file} was hijacked by {p_go.language_id if p_go else None}"
    )


def test_ast_extractor_preserves_python_symbols_in_webhook_path() -> None:
    """Verifies that ASTExtractor extracts Python symbols from files with 'webhook' in path.

    If ReactReduxPlugin hijacks 'webhook.py', ASTExtractor produces 0 symbols.
    """
    extractor = ASTExtractor()
    code = """
class WebhookHandler:
    def handle_event(self, event: dict) -> bool:
        return True
"""
    symbols, _edges = extractor.parse_source(
        code=code,
        language="python",
        file_path="backend/webhook.py",
    )
    symbol_names = [s.name for s in symbols]
    assert "WebhookHandler" in symbol_names, (
        f"WebhookHandler missing from symbols: {symbol_names}"
    )
    assert "handle_event" in symbol_names, (
        f"handle_event missing from symbols: {symbol_names}"
    )


def test_registry_unregister_before_initialization() -> None:
    """Verifies that calling unregister_language before get_* does not get undone by lazy load."""
    PluginRegistry.reset()
    PluginRegistry.unregister_language("csharp")
    plugin = PluginRegistry.get_language_plugin("csharp")
    assert plugin is None, "csharp plugin re-appeared after being explicitly unregistered"


# ==============================================================================
# 2. React / Redux Plugin Adversarial Stress Tests
# ==============================================================================


def test_react_redux_generic_components_extracted() -> None:
    """Verifies that TypeScript generic components are extracted as symbols.

    Challenge: `comp_pat` does not support TypeScript generic type parameters
    like `<TData, TValue>` on arrow function or function components.
    """
    plugin = ReactReduxPlugin()
    code = """
export const DataTable = <TData, TValue>({ columns, data }: DataTableProps<TData, TValue>) => {
  return <div>Table</div>;
};

export function SelectBox<T extends Option>(props: SelectProps<T>) {
  return <div>Select</div>;
}
"""
    symbols, _ = plugin.extract_ast(code, "src/components/DataTable.tsx")
    names = [s.name for s in symbols]
    assert "DataTable" in names, f"DataTable generic component missing: {names}"
    assert "SelectBox" in names, f"SelectBox generic component missing: {names}"


def test_react_redux_no_self_referencing_calls_on_hook_definition() -> None:
    """Verifies that a custom hook definition does not emit a self-CALLS edge to itself."""
    plugin = ReactReduxPlugin()
    code = """
export function useUserData(userId: string) {
  const theme = useTheme();
  return { userId, theme };
}
"""
    _symbols, edges = plugin.extract_ast(code, "src/hooks/useUserData.ts")
    self_calls = [
        e for e in edges
        if e.edge_kind == "CALLS" and e.target_qualified_name == "useUserData"
    ]
    assert len(self_calls) == 0, f"Found self-referencing CALLS edge: {self_calls}"


def test_react_redux_current_comp_qual_does_not_leak_to_subsequent_functions() -> None:
    """Verifies that hook calls in standalone functions outside a component are not attributed."""
    plugin = ReactReduxPlugin()
    code = """
export const Button = () => {
  return <button>Click</button>;
};

// Independent standalone function outside Button
function logAction() {
  dispatch(logClick());
}
"""
    _symbols, edges = plugin.extract_ast(code, "src/components/Button.tsx")
    dispatches = [e for e in edges if e.edge_kind == "DISPATCHES"]
    for d in dispatches:
        assert not d.source_qualified_name.endswith(".Button"), (
            f"Dispatch from standalone function was attributed to Button: {d}"
        )


def test_react_redux_in_reducers_block_does_not_leak_outside_slice() -> None:
    """Verifies that helper methods defined after createSlice are not treated as reducers."""
    plugin = ReactReduxPlugin()
    code = """
export const userSlice = createSlice({
  name: 'user',
  initialState: {},
  reducers: {
    login(state) {}
  }
});

class StateValidator {
  validate(state) {}
}
"""
    symbols, _ = plugin.extract_ast(code, "src/slices/userSlice.ts")
    reducers = [s for s in symbols if s.symbol_kind == "reducer"]
    reducer_names = [r.name for r in reducers]
    assert "validate" not in reducer_names, (
        f"validate helper outside slice was extracted as a reducer: {reducer_names}"
    )


def test_react_redux_multiline_createslice_and_arrow_reducers() -> None:
    """Verifies that createSlice formatted across lines and arrow reducers are parsed."""
    plugin = ReactReduxPlugin()
    code = """
export const counterSlice = createSlice(
  {
    name: 'counter',
    initialState: 0,
    reducers: {
      increment: (state) => { state.value += 1; },
      decrement(state) { state.value -= 1; }
    }
  }
);
"""
    symbols, _ = plugin.extract_ast(code, "src/counterSlice.ts")
    names = [s.name for s in symbols]
    assert "counterSlice" in names, f"counterSlice missing: {names}"
    assert "increment" in names, f"increment arrow reducer missing: {names}"
    assert "decrement" in names, f"decrement method reducer missing: {names}"


# ==============================================================================
# 3. C# / .NET Plugin Adversarial Stress Tests
# ==============================================================================


def test_csharp_record_primary_constructor_base_clause() -> None:
    """Verifies that primary constructor records retain base types and are not methods."""
    plugin = CSharpPlugin()
    code = """
namespace Enterprise.Domain;

public record Customer(int Id, string Name) : BaseEntity, IAuditable;
"""
    symbols, edges = plugin.extract_ast(code, "Customer.cs")
    method_names = [s.name for s in symbols if s.symbol_kind == "method"]
    assert "Customer" not in method_names, (
        f"Customer record was falsely extracted as a method: {method_names}"
    )

    targets = [e.target_qualified_name for e in edges]
    assert "BaseEntity" in targets, f"BaseEntity missing from edges: {targets}"
    assert "IAuditable" in targets, f"IAuditable missing from edges: {targets}"


def test_csharp_generic_base_clause_does_not_inherit_type_parameters() -> None:
    """Verifies that generic type parameters like TEntity are not emitted as INHERITS edges."""
    plugin = CSharpPlugin()
    code = """
namespace Enterprise.Repository;

public class GenericRepo<TEntity, TKey> : IRepository<TEntity, TKey>, IDisposable
    where TEntity : class, new()
{
}
"""
    _symbols, edges = plugin.extract_ast(code, "GenericRepo.cs")
    inherits_edges = [e for e in edges if e.edge_kind == "INHERITS"]
    inherited_targets = [e.target_qualified_name for e in inherits_edges]

    assert "TEntity" not in inherited_targets, (
        f"Generic type parameter TEntity extracted as inherited base class: {inherited_targets}"
    )
    assert "TKey" not in inherited_targets, (
        f"Generic type parameter TKey extracted as inherited base class: {inherited_targets}"
    )
    assert "IRepository" in inherited_targets, (
        f"IRepository interface missing from inherits: {inherited_targets}"
    )


def test_csharp_allman_namespace_formatting() -> None:
    """Verifies that classic C# Allman formatting (brace on next line) extracts the namespace."""
    plugin = CSharpPlugin()
    code = """
namespace Enterprise.Core.Services
{
    public class WorkerService
    {
        public void Execute() {}
    }
}
"""
    symbols, _ = plugin.extract_ast(code, "WorkerService.cs")
    worker = next((s for s in symbols if s.name == "WorkerService"), None)
    assert worker is not None, "WorkerService not found in symbols"
    assert "Enterprise.Core.Services" in worker.qualified_name, (
        f"Namespace missing from qualified_name: {worker.qualified_name}"
    )


def test_csharp_nested_classes_method_attribution() -> None:
    """Verifies outer class methods following nested classes are not attributed to inner class."""
    plugin = CSharpPlugin()
    code = """
namespace Enterprise.Data;

public class OuterClass
{
    public class NestedClass
    {
        public void DoNested() {}
    }
    public void DoOuter() {}
}
"""
    symbols, _ = plugin.extract_ast(code, "OuterClass.cs")
    do_outer = next((s for s in symbols if s.name == "DoOuter"), None)
    assert do_outer is not None, "DoOuter not found"
    assert "NestedClass" not in do_outer.qualified_name, (
        f"DoOuter was falsely attributed to NestedClass: {do_outer.qualified_name}"
    )
    assert "OuterClass.DoOuter" in do_outer.qualified_name, (
        f"DoOuter qualified_name incorrect: {do_outer.qualified_name}"
    )


def test_csharp_multiple_namespaces_in_single_file() -> None:
    """Verifies that multiple namespaces in a single file do not contaminate each other."""
    plugin = CSharpPlugin()
    code = """
namespace ModuleA {
    public class ServiceA {}
}

namespace ModuleB {
    public class ServiceB {}
}
"""
    symbols, _ = plugin.extract_ast(code, "Services.cs")
    svc_a = next((s for s in symbols if s.name == "ServiceA"), None)
    svc_b = next((s for s in symbols if s.name == "ServiceB"), None)

    assert svc_a is not None and svc_b is not None
    assert "ModuleA" in svc_a.qualified_name, (
        f"ServiceA should be in ModuleA, got {svc_a.qualified_name}"
    )
    assert "ModuleB" in svc_b.qualified_name, (
        f"ServiceB should be in ModuleB, got {svc_b.qualified_name}"
    )


def test_csharp_malformed_xml_manifest_fallback() -> None:
    """Verifies that malformed .csproj XML falls back to regex and does not raise exceptions."""
    plugin = CSharpPlugin()
    bad_xml = """
<Project Sdk="Microsoft.NET.Sdk">
  <ItemGroup>
    <PackageReference Include="Serilog" Version="2.10.0" />
    <ProjectReference Include="..\\Core\\Core.csproj" />
    <!-- Unclosed XML tag below -->
    <PackageReference Include="Newtonsoft.Json" Version="13.0.1"
"""
    deps = plugin.extract_manifest_dependencies("app.csproj", bad_xml)
    targets = [d.target_package for d in deps]
    assert "Serilog" in targets, f"Serilog missing from deps: {targets}"
    assert "Newtonsoft.Json" in targets, f"Newtonsoft.Json missing from deps: {targets}"


# ==============================================================================
# 4. Terraform & Storybook Adversarial Stress Tests
# ==============================================================================


def test_terraform_unbalanced_brace_in_comment_does_not_swallow_subsequent_resources() -> None:
    """Verifies that an unmatched brace in a comment does not swallow subsequent resources."""
    plugin = TerraformPlugin()
    code = """
resource "google_compute_instance" "app_server" {
  name = "app-server"
  # Note: { unmatched opening brace in comment
  network_interface {
    network = "default"
  }
}

resource "google_storage_bucket" "backup" {
  name = "my-backup-bucket"
}
"""
    symbols, _ = plugin.extract_ast(code, "main.tf")
    names = [s.name for s in symbols]
    assert "app_server" in names, f"app_server missing: {names}"
    assert "backup" in names, (
        f"backup resource was swallowed by app_server: {names}"
    )


def test_terraform_dynamic_blocks_and_provider_versions() -> None:
    """Verifies dynamic blocks and version constraints extraction."""
    plugin = TerraformPlugin()
    code = """
resource "google_compute_security_policy" "policy" {
  name = "my-policy"
  dynamic "rule" {
    for_each = var.rules
    content {
      action   = rule.value.action
      priority = rule.value.priority
    }
  }
}
"""
    symbols, edges = plugin.extract_ast(code, "security.tf")
    assert any(s.name == "policy" for s in symbols)
    assert any(e.edge_kind == "USES_TYPE" for e in edges)

    versions_tf = """
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source = "hashicorp/google-beta"
    }
  }
}
"""
    deps = plugin.extract_manifest_dependencies("versions.tf", versions_tf)
    assert len(deps) == 2
    g_dep = next(d for d in deps if d.target_package == "hashicorp/google")
    assert g_dep.version_constraint == "~> 5.0"
    gb_dep = next(d for d in deps if d.target_package == "hashicorp/google-beta")
    assert gb_dep.version_constraint == "*"


def test_storybook_csf2_csf3_and_mdx_resilience() -> None:
    """Verifies CSF3, CSF2 bind, and MDX story variants."""
    plugin = StorybookPlugin()
    code_csf = """
import type { Meta, StoryObj } from '@storybook/react';
import { Button } from './Button';

const meta: Meta<typeof Button> = {
  title: 'Components/Button',
  component: Button,
};
export default meta;

export const Primary: Story = { args: { primary: true } };
export const Secondary = Template.bind({});
"""
    symbols, edges = plugin.extract_ast(code_csf, "Button.stories.tsx")
    names = [s.name for s in symbols]
    assert "Meta" in names
    assert "Primary" in names
    assert "Secondary" in names
    assert any(e.target_qualified_name == "Button" for e in edges)


# ==============================================================================
# 5. Milestone 3 Challenger Stress Tests (Empirical Challenge Suite M3.3)
# ==============================================================================


def test_adversarial_csharp_multiline_primary_constructor_class_and_record() -> None:
    """Verifies that multiline primary constructors on classes and records are extracted.

    Empirical Challenge: C# 12 / .NET 8 primary constructors spanning multiple lines
    fail regex line-by-line matching, dropping classes, records, methods, and inheritance edges.
    """
    plugin = CSharpPlugin()
    code = """
namespace Enterprise.Services;

public class OrderProcessor(
    IOrderRepository orderRepo,
    IPaymentGateway paymentGateway,
    ILogger<OrderProcessor> logger
) : IOrderProcessor
{
    public async Task<bool> ProcessOrderAsync(int orderId)
    {
        return await orderRepo.SaveAsync(orderId);
    }
}

public record OrderSummary(
    Guid OrderId,
    string CustomerEmail,
    decimal TotalAmount
) : IOrderSummary;
"""
    symbols, edges = plugin.extract_ast(code, "OrderProcessor.cs")
    names = [s.name for s in symbols]
    assert "OrderProcessor" in names, f"OrderProcessor class missing: {names}"
    assert "OrderSummary" in names, f"OrderSummary record missing: {names}"
    assert "ProcessOrderAsync" in names, f"ProcessOrderAsync method missing: {names}"

    inherits_targets = [e.target_qualified_name for e in edges if e.edge_kind == "INHERITS"]
    assert "IOrderProcessor" in inherits_targets, f"IOrderProcessor missing from edges: {edges}"
    assert "IOrderSummary" in inherits_targets, f"IOrderSummary missing from edges: {edges}"


def test_adversarial_csharp_nested_namespaces_qualification() -> None:
    """Verifies that nested block namespaces retain full hierarchical qualification.

    Empirical Challenge: `ns_stack` only uses `ns_stack[-1][0]` rather than joining
    the namespace hierarchy, dropping outer namespaces (e.g. 'Enterprise.Core').
    """
    plugin = CSharpPlugin()
    code = """
namespace Enterprise
{
    namespace Core
    {
        namespace Services
        {
            public class DeepService
            {
                public void Execute() {}
            }
        }
    }
}
"""
    symbols, _ = plugin.extract_ast(code, "DeepService.cs")
    svc = next((s for s in symbols if s.name == "DeepService"), None)
    assert svc is not None, "DeepService symbol missing"
    assert "Enterprise.Core.Services" in svc.qualified_name, (
        f"Outer namespaces dropped from qualified_name: {svc.qualified_name}"
    )


def test_adversarial_csharp_deeply_nested_classes_qualification() -> None:
    """Verifies that deeply nested inner classes retain their parent class hierarchy.

    Empirical Challenge: Inner classes lose their enclosing class scope, causing
    name collisions across different classes containing nested Config/State classes.
    """
    plugin = CSharpPlugin()
    code = """
namespace Enterprise.Domain;

public class Container
{
    public class NestedState
    {
        public class DeepConfig
        {
            public void Load() {}
        }
    }
}
"""
    symbols, _ = plugin.extract_ast(code, "Container.cs")
    cfg = next((s for s in symbols if s.name == "DeepConfig"), None)
    assert cfg is not None, "DeepConfig symbol missing"
    assert "Container" in cfg.qualified_name and "NestedState" in cfg.qualified_name, (
        f"Enclosing classes dropped from DeepConfig: {cfg.qualified_name}"
    )


def test_adversarial_csharp_multiline_generic_base_clause() -> None:
    """Verifies that generic classes with multiline base clauses extract inheritance edges.

    Empirical Challenge: When the base class colon ':' is followed by a newline,
    line-by-line matching produces an empty base_clause and drops all inheritance edges.
    """
    plugin = CSharpPlugin()
    code = """
namespace Enterprise.Repository;

public class SqlRepository<TEntity, TKey> :
    IRepository<TEntity, TKey>,
    IReadRepository<TEntity>,
    IDisposable
    where TEntity : class, new()
    where TKey : struct
{
    public void Dispose() {}
}
"""
    _symbols, edges = plugin.extract_ast(code, "SqlRepository.cs")
    inherits_targets = [
        e.target_qualified_name for e in edges if e.edge_kind in ("INHERITS", "USES_TYPE")
    ]
    assert "IRepository" in inherits_targets, f"IRepository missing from edges: {inherits_targets}"
    assert "IReadRepository" in inherits_targets, f"IReadRepository missing: {inherits_targets}"
    assert "IDisposable" in inherits_targets, f"IDisposable missing: {inherits_targets}"


def test_adversarial_react_redux_multi_type_complex_generic_bounds() -> None:
    """Verifies that TS generic components with complex generic bounds are extracted.

    Empirical Challenge: `<[^>]+>` terminates at the inner `>` of `Record<string, any>`,
    failing component regex matching and dropping the component symbol completely.
    """
    plugin = ReactReduxPlugin()
    code = """
export const TypedTable = <TData extends Record<string, any>, TValue, TMeta>(
    props: TableProps<TData, TValue, TMeta>
) => {
    return <div>Table</div>;
};

export function FilterSelect<TItem extends Option<string>, TContext>(
    props: SelectProps<TItem, TContext>
) {
    return <div>Select</div>;
}
"""
    symbols, _ = plugin.extract_ast(code, "src/components/TypedTable.tsx")
    names = [s.name for s in symbols]
    assert "TypedTable" in names, f"TypedTable missing: {names}"
    assert "FilterSelect" in names, f"FilterSelect missing: {names}"


def test_adversarial_react_redux_nested_components_hook_isolation() -> None:
    """Verifies that nested helper components do not destroy outer component hook attribution.

    Empirical Challenge: `current_comp_qual` is a scalar rather than a stack. Closing
    a nested helper component resets `current_comp_qual = None`, discarding all subsequent
    hooks and dispatches in the outer component.
    """
    plugin = ReactReduxPlugin()
    code = """
export const Dashboard = () => {
    const user = useCurrentUser();

    // Nested inline helper component
    const MetricBadge = () => {
        const badgeTheme = useTheme();
        return <span>{badgeTheme}</span>;
    };

    // Subsequent hooks and dispatch in outer component
    const permissions = usePermissions();
    const dispatch = useDispatch();

    return (
        <div>
            <MetricBadge />
            <button onClick={() => dispatch(refreshDashboard())}>Refresh</button>
        </div>
    );
};
"""
    _symbols, edges = plugin.extract_ast(code, "src/components/Dashboard.tsx")
    dash_edges = [e for e in edges if "Dashboard" in e.source_qualified_name]
    dash_targets = [e.target_qualified_name for e in dash_edges]

    assert "useCurrentUser" in dash_targets, f"useCurrentUser missing: {dash_targets}"
    assert "usePermissions" in dash_targets, (
        f"usePermissions lost due to nested component reset: {dash_targets}"
    )
    assert any(
        e.edge_kind == "DISPATCHES" and e.target_qualified_name == "refreshDashboard"
        for e in dash_edges
    ), f"refreshDashboard dispatch lost due to nested component reset: {dash_edges}"


def test_adversarial_terraform_block_comment_brace_imbalance() -> None:
    """Verifies that block comments with unmatched braces do not swallow subsequent resources.

    Empirical Challenge: `_strip_hcl_comment` only strips `#` and `//`. Unmatched braces
    inside block comments `/* { */` corrupt `brace_depth` and swallow subsequent resources.
    """
    plugin = TerraformPlugin()
    code = """
resource "google_compute_instance" "app_server" {
    name = "app-server"
    /* Note: { unmatched opening brace in block comment */
    network_interface {
        network = "default"
    }
}

resource "google_storage_bucket" "backup" {
    name = "my-backup-bucket"
}
"""
    symbols, _ = plugin.extract_ast(code, "main.tf")
    names = [s.name for s in symbols]
    assert "app_server" in names, f"app_server missing: {names}"
    assert "backup" in names, (
        f"backup resource was swallowed by app_server due to block comment brace: {names}"
    )


def test_adversarial_terraform_interpolated_domain_strings() -> None:
    """Verifies URLs in interpolated strings do not emit phantom CALLS edges.

    Empirical Challenge: Strings containing `${...}` are preserved in whole, and `ref_pat`
    emits phantom CALLS edges for hostnames/domains with underscores (e.g. `internal_gateway.corp`).
    """
    plugin = TerraformPlugin()
    code = """
resource "google_cloud_run_v2_service" "api" {
    name = "api"
    template {
        containers {
            image = "gcr.io/my_corp/app:${var.tag}"
            env {
                name = "URL"
                value = "https://internal_gateway.corp.internal:${var.port}"
            }
        }
    }
}
"""
    _symbols, edges = plugin.extract_ast(code, "main.tf")
    phantom_edges = [
        e
        for e in edges
        if e.edge_kind == "CALLS" and "internal_gateway" in e.target_qualified_name
    ]
    assert len(phantom_edges) == 0, (
        f"Phantom CALLS edge emitted from interpolated URL: {phantom_edges}"
    )


def test_adversarial_storybook_multiline_imports_and_component_docs() -> None:
    """Verifies that multiline destructured imports link component documentation edges.

    Empirical Challenge: `imp_pat` operates line-by-line and fails on multiline imports,
    causing Storybook stories to drop DOCUMENTS_COMPONENT edges.
    """
    plugin = StorybookPlugin()
    code = """
import {
    Button,
    ButtonProps
} from './Button';

export const Default = { args: { primary: true } };
"""
    _symbols, edges = plugin.extract_ast(code, "src/components/Button.stories.tsx")
    doc_edges = [e for e in edges if e.edge_kind == "DOCUMENTS_COMPONENT"]
    assert any(e.target_qualified_name == "Button" for e in doc_edges), (
        f"DOCUMENTS_COMPONENT edge to Button missing from multiline import: {edges}"
    )


def test_adversarial_registry_dynamic_reregistration_and_priority_override() -> None:
    """Verifies dynamic unregistration, reregistration, and priority dispatch."""
    from context_scout.plugins.base import LanguagePlugin

    class CustomPlugin(LanguagePlugin):
        @property
        def language_id(self) -> str:
            return "csharp"

        @property
        def file_extensions(self) -> tuple[str, ...]:
            return (".cs",)

        @property
        def manifest_files(self) -> tuple[str, ...]:
            return (".csproj",)

        def can_handle(self, file_path: str) -> bool:
            return file_path.endswith(".cs")

        def extract_ast(self, *args, **kwargs):
            return [], []

        def extract_manifest_dependencies(self, *args, **kwargs):
            return []

    PluginRegistry.reset()
    PluginRegistry._ensure_initialized()
    assert PluginRegistry.get_language_plugin("csharp") is not None

    # 1. Dynamic unregistration
    PluginRegistry.unregister_language("csharp")
    assert PluginRegistry.get_language_plugin("csharp") is None
    assert PluginRegistry.get_plugin_for_file("app.cs") is None

    # 2. Dynamic reregistration with higher priority
    custom = CustomPlugin()
    PluginRegistry.register_language(custom, priority=5)
    assert PluginRegistry.get_language_plugin("csharp") is custom
    assert PluginRegistry.get_plugin_for_file("app.cs") is custom

    # 3. Teardown and reset
    PluginRegistry.reset()
    PluginRegistry._ensure_initialized()
    p_orig = PluginRegistry.get_language_plugin("csharp")
    assert p_orig is not None
    assert not isinstance(p_orig, CustomPlugin)
