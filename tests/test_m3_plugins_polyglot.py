"""Automated verification suite for M3 Language Plugins (F10-F14)."""

from __future__ import annotations

from context_scout.ast_engine.extractor import ASTExtractor
from context_scout.plugins.languages.csharp import CSharpPlugin
from context_scout.plugins.languages.react_redux import ReactReduxPlugin
from context_scout.plugins.languages.storybook import StorybookPlugin
from context_scout.plugins.languages.terraform import TerraformPlugin
from context_scout.plugins.registry import PluginRegistry


def test_plugin_registry_dispatch_priority() -> None:
    """Verifies priority-ordered dispatch between Storybook, React, and TypeScript."""
    PluginRegistry.reset()
    PluginRegistry.register_language(StorybookPlugin(), priority=10)
    PluginRegistry.register_language(ReactReduxPlugin(), priority=20)

    p_sb = PluginRegistry.get_plugin_for_file("components/Button.stories.tsx")
    assert isinstance(p_sb, StorybookPlugin)

    p_react = PluginRegistry.get_plugin_for_file("components/Button.tsx")
    assert isinstance(p_react, ReactReduxPlugin)

    p_mdx = PluginRegistry.get_plugin_for_file("docs/Intro.mdx")
    assert isinstance(p_mdx, StorybookPlugin)


def test_csharp_plugin_extraction() -> None:
    """Verifies C# AST symbol and DI registration extraction."""
    code = """
    namespace EShop.Ordering.API;
    using MediatR;

    public interface IOrderService { Task<OrderDto> CreateOrder(CreateCommand cmd); }

    public class OrderService : IOrderService {
        public async Task<OrderDto> CreateOrder(CreateCommand cmd) { return new OrderDto(); }
    }

    public static class Program {
        public static void Configure(IServiceCollection services) {
            services.AddScoped<IOrderService, OrderService>();
        }
    }
    """
    plugin = CSharpPlugin()
    syms, edges = plugin.extract_ast(code, "Ordering/OrderService.cs", repo_name="eshop-orders")
    names = {s.name for s in syms}
    assert {"IOrderService", "OrderService", "CreateOrder", "Program"} <= names

    edge_kinds = {e.edge_kind for e in edges}
    assert "INHERITS" in edge_kinds
    assert "PROVISIONED_BY" in edge_kinds

    # Test manifest extraction
    csproj_content = r"""<Project Sdk="Microsoft.NET.Sdk">
      <ItemGroup>
        <PackageReference Include="MediatR" Version="12.0.1" />
        <ProjectReference Include="..\Core\Core.csproj" />
      </ItemGroup>
    </Project>"""
    deps = plugin.extract_manifest_dependencies("Ordering.API.csproj", csproj_content)
    assert len(deps) == 2
    pkg_dep = next(d for d in deps if d.target_package == "MediatR")
    assert pkg_dep.version_constraint == "12.0.1"
    assert pkg_dep.scope == "compile"
    proj_dep = next(d for d in deps if d.target_package == "Core")
    assert proj_dep.scope == "project_reference"


def test_react_redux_plugin_extraction() -> None:
    """Verifies React hook, Redux slice, reducer, selector, and DISPATCHES edge extraction."""
    code = """
    import { createSlice, PayloadAction } from '@reduxjs/toolkit';
    import { useAppDispatch, useAppSelector } from './hooks';

    export const cartSlice = createSlice({
        name: 'cart',
        initialState: { items: [] },
        reducers: {
            addItem(state, action: PayloadAction<string>) { state.items.push(action.payload); }
        }
    });

    export const selectCartItems = (state: RootState) => state.cart.items;

    export const CartView = () => {
        const dispatch = useAppDispatch();
        const items = useAppSelector(selectCartItems);
        return <button onClick={() => dispatch(cartSlice.actions.addItem('item-1'))}>Add</button>;
    };
    """
    plugin = ReactReduxPlugin()
    syms, edges = plugin.extract_ast(code, "features/cart/CartView.tsx", repo_name="frontend-app")
    kinds = {s.symbol_kind for s in syms}
    assert "slice" in kinds
    assert "reducer" in kinds
    assert "selector" in kinds

    edge_kinds = {e.edge_kind for e in edges}
    assert "DISPATCHES" in edge_kinds
    assert "SELECTS" in edge_kinds

    # Test package.json manifest dependencies
    pkg_json = (
        '{"name": "frontend", "dependencies": {"react": "^18.2.0"}, '
        '"devDependencies": {"typescript": "^5.0.0"}}'
    )
    deps = plugin.extract_manifest_dependencies("package.json", pkg_json)
    assert len(deps) == 2
    assert any(d.target_package == "react" and d.scope == "runtime" for d in deps)
    assert any(d.target_package == "typescript" and d.scope == "dev" for d in deps)


def test_storybook_plugin_extraction() -> None:
    """Verifies Storybook CSF3 meta and story variant extraction with DOCUMENTS_COMPONENT edge."""
    code = """
    import type { Meta, StoryObj } from '@storybook/react';
    import { Button } from './Button';

    const meta: Meta<typeof Button> = {
        title: 'DesignSystem/Button',
        component: Button,
    };
    export default meta;

    export const Primary: StoryObj<typeof Button> = {
        args: { label: 'Submit' }
    };
    """
    plugin = StorybookPlugin()
    syms, edges = plugin.extract_ast(
        code, "components/Button.stories.tsx", repo_name="design-system"
    )
    assert any(s.name == "Meta" for s in syms)
    assert any(s.name == "Primary" for s in syms)
    assert any(
        e.edge_kind == "DOCUMENTS_COMPONENT" and e.target_qualified_name == "Button" for e in edges
    )

    # Test MDX story extraction
    mdx_code = """
    import { Meta, Story } from '@storybook/addon-docs';
    import { Header } from './Header';

    <Meta title="Components/Header" component={Header} />

    # Header Documentation
    <Story name="DefaultHeader" />
    """
    mdx_syms, mdx_edges = plugin.extract_ast(
        mdx_code, "components/Header.stories.mdx", repo_name="design-system"
    )
    assert any(s.name == "Meta" for s in mdx_syms)
    assert any(s.name == "DefaultHeader" for s in mdx_syms)
    assert any(e.edge_kind == "DOCUMENTS_COMPONENT" for e in mdx_edges)


def test_terraform_plugin_extraction() -> None:
    """Verifies Terraform GCP resource, module, and dependency graph extraction."""
    code = """
    variable "gcp_region" { default = "europe-west3" }

    resource "google_service_account" "scout_sa" {
        account_id = "scout-sa"
    }

    resource "google_cloud_run_v2_service" "scout_api" {
        name     = "scout-api"
        location = var.gcp_region
        template {
            service_account = google_service_account.scout_sa.email
        }
        depends_on = [google_service_account.scout_sa]
    }
    """
    plugin = TerraformPlugin()
    syms, edges = plugin.extract_ast(code, "infra/cloud_run.tf", repo_name="infra-repo")
    names = {s.name for s in syms}
    assert {"gcp_region", "scout_sa", "scout_api"} <= names

    edge_kinds = {e.edge_kind for e in edges}
    assert "USES_TYPE" in edge_kinds
    assert "BINDS_SERVICE_ACCOUNT" in edge_kinds
    assert "PROVISIONED_BY" in edge_kinds

    # Test provider dependencies
    provider_code = """
    terraform {
      required_providers {
        google = {
          source  = "hashicorp/google"
          version = "~> 5.0"
        }
      }
    }
    """
    deps = plugin.extract_manifest_dependencies("versions.tf", provider_code)
    assert len(deps) == 1
    assert deps[0].target_package == "hashicorp/google"
    assert deps[0].version_constraint == "~> 5.0"
    assert deps[0].scope == "provider"


def test_extractor_delegates_to_registered_plugins() -> None:
    """Verifies ASTExtractor.parse_source automatically delegates to PluginRegistry."""
    PluginRegistry.reset()
    PluginRegistry._load_builtin_plugins()
    extractor = ASTExtractor()

    # C# delegation
    cs_code = "namespace App; public class Worker { public void Run() {} }"
    cs_syms, _ = extractor.parse_source(cs_code, "csharp", "src/App/Worker.cs", repo_name="my-app")
    assert any(s.name == "Worker" for s in cs_syms)
    assert any(s.name == "Run" for s in cs_syms)

    # Storybook delegation
    sb_code = (
        "import { Header } from './Header'; "
        "export const meta = { component: Header }; "
        "export const Default = {};"
    )
    sb_syms, sb_edges = extractor.parse_source(
        sb_code, "typescript", "components/Header.stories.tsx", repo_name="ui"
    )
    assert any(s.name == "Default" for s in sb_syms)
    assert any(e.edge_kind == "DOCUMENTS_COMPONENT" for e in sb_edges)
