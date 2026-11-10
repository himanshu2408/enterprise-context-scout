"""Empirical Challenger Deep Adversarial Suite for M3.9 Polyglot AST Plugins.

Adversarially stress-tests:
1. C# 11 multiline raw string literals (\"\"\" ... { ... \"\"\") with unmatched braces
   corrupting brace_depth and falsely nesting sibling classes.
2. Terraform heredoc blocks (<<-EOF ... { ... EOF) with unmatched braces
   swallowing subsequent resource blocks until EOF.
3. React components wrapped in memo() or forwardRef() dropping symbols and hook edges.
4. C# 12 primary constructors with 5+ levels of nested parentheses dropping the class.
"""

from __future__ import annotations

from context_scout.plugins.languages.csharp import CSharpPlugin
from context_scout.plugins.languages.react_redux import ReactReduxPlugin
from context_scout.plugins.languages.terraform import TerraformPlugin

# ==============================================================================
# 1. C# 11 Multiline Raw String Literals with Unmatched Braces
# ==============================================================================


def test_adversarial_csharp_multiline_raw_string_unmatched_brace_corrupts_class_hierarchy() -> None:
    """Challenge: C# 11 multiline raw string literal with unmatched '{' corrupts brace_depth.

    Failure Mode: `_clean_cs_line` only masks regular strings and verbatim @"" strings.
    Multiline raw string literals (\"\"\" ... \"\"\") leave intermediate lines without quotes.
    An intermediate line containing '{' increments brace_depth without a matching closing brace,
    preventing class_stack from popping and falsely nesting sibling classes.
    """
    plugin = CSharpPlugin()
    code = '''
namespace Enterprise.Data;

public class RawStringConfigService
{
    private string rawJsonTemplate = """
      Unmatched { brace inside multiline raw string
    """;

    public void Apply() {}
}

public class MetricsReporter
{
    public void Report() {}
}
'''
    symbols, _edges = plugin.extract_ast(code, "RawStringConfigService.cs")
    sym_map = {s.name: s for s in symbols}

    assert "RawStringConfigService" in sym_map, "RawStringConfigService missing"
    assert "MetricsReporter" in sym_map, "MetricsReporter missing"

    metrics_sym = sym_map["MetricsReporter"]
    # Empirical check:
    # Under current code, MetricsReporter is falsely qualified as:
    # 'Enterprise.Data.RawStringConfigService.MetricsReporter'
    assert not metrics_sym.qualified_name.endswith(
        ".RawStringConfigService.MetricsReporter"
    ), (
        "DEFECT CONFIRMED: MetricsReporter falsely nested inside RawStringConfigService: "
        f"{metrics_sym.qualified_name}"
    )


# ==============================================================================
# 2. Terraform Heredocs with Unmatched Braces Swallowing Resources
# ==============================================================================


def test_adversarial_terraform_heredoc_unmatched_brace_swallows_subsequent_resources() -> None:
    """Challenge: Terraform heredoc containing unmatched '{' corrupts brace_depth.

    Failure Mode: `terraform.py:89-97` tracks brace_depth across lines using _strip_hcl_comment.
    Because heredocs (<<-EOF ... EOF) are not stripped or masked, an unmatched '{' inside a heredoc
    (such as in shell scripts or partial configuration templates) increments brace_depth.
    The resource's closing '}' fails to reset brace_depth to 0, causing the block to swallow
    all subsequent resources until EOF.
    """
    plugin = TerraformPlugin()
    code = """
resource "google_compute_instance" "app_server" {
  name         = "app-server"
  machine_type = "e2-medium"

  metadata_startup_script = <<-EOF
    cat << 'CONF' > /etc/template.conf
    server {
        listen 80;
    CONF
  EOF

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
    }
  }
}

resource "google_storage_bucket" "assets_bucket" {
  name     = "app-assets-bucket"
  location = "EU"
}
"""
    symbols, _edges = plugin.extract_ast(code, "main.tf")
    sym_names = [s.name for s in symbols]

    # Empirical check:
    # Under current code, assets_bucket is completely missing because app_server
    # swallowed the rest of the file.
    assert "assets_bucket" in sym_names, (
        f"DEFECT CONFIRMED: assets_bucket dropped by unmasked heredoc in app_server: {sym_names}"
    )


# ==============================================================================
# 3. React Components Wrapped in memo() and forwardRef()
# ==============================================================================


def test_adversarial_react_memo_and_forward_ref_wrapped_components_dropped() -> None:
    """Challenge: React components wrapped in memo() or forwardRef().

    Failure Mode: `comp_pat` in `react_redux.py:98-103` expects `(?:const|function) <Name> = (`
    or `function <Name>(`. Higher-order component wrappers like `memo(...)`, `React.memo(...)`,
    `forwardRef(...)`, and `React.forwardRef(...)` fail to match, causing the component
    symbol and all hook calls within it to be completely dropped.
    """
    plugin = ReactReduxPlugin()
    code = """
import React, { memo, forwardRef } from 'react';
import { useTheme } from '../hooks/useTheme';
import { useUserData } from '../hooks/useUserData';

export const MemoCard = memo((props: CardProps) => {
    const theme = useTheme();
    return <div className={theme}>{props.title}</div>;
});

export const InputField = forwardRef<HTMLInputElement, InputProps>((props, ref) => {
    const user = useUserData();
    return <input ref={ref} placeholder={user.name} {...props} />;
});
"""
    symbols, edges = plugin.extract_ast(code, "src/components/WrappedComponents.tsx")
    sym_names = [s.name for s in symbols]
    edge_targets = [e.target_qualified_name for e in edges]

    # Empirical check:
    # Under current code, both MemoCard and InputField are dropped.
    assert "MemoCard" in sym_names, (
        f"DEFECT CONFIRMED: MemoCard dropped when wrapped in memo(): {sym_names}"
    )
    assert "InputField" in sym_names, (
        f"DEFECT CONFIRMED: InputField dropped when wrapped in forwardRef(): {sym_names}"
    )
    assert "useTheme" in edge_targets, (
        f"DEFECT CONFIRMED: useTheme hook edge dropped: {edge_targets}"
    )
    assert "useUserData" in edge_targets, (
        f"DEFECT CONFIRMED: useUserData hook edge dropped: {edge_targets}"
    )


# ==============================================================================
# 4. C# 12 Primary Constructor 5-Level Nested Parentheses
# ==============================================================================


def test_adversarial_csharp_primary_constructor_5_level_nested_parentheses() -> None:
    """Challenge: C# 12 primary constructor with 5 levels of nested parentheses.

    Failure Mode: `paren_rec` in `csharp.py:156` hardcodes 4 levels of nested parentheses.
    At 5 levels of nesting, `type_pat` regex fails to match, completely dropping the class,
    its methods, and all inheritance edges.
    """
    plugin = CSharpPlugin()
    code = """
namespace Enterprise.Services;

public class DeepNestingService((int, (int, (int, (int, (int, int)))))) : IDeepService
{
    public void Execute() {}
}
"""
    symbols, edges = plugin.extract_ast(code, "DeepNestingService.cs")
    sym_names = [s.name for s in symbols]
    edge_targets = [e.target_qualified_name for e in edges]

    # Empirical check:
    # Under current code, DeepNestingService is completely dropped.
    assert "DeepNestingService" in sym_names, (
        f"DEFECT CONFIRMED: DeepNestingService dropped by 5-level nested parens: {sym_names}"
    )
    assert "Execute" in sym_names, (
        f"DEFECT CONFIRMED: Execute method dropped: {sym_names}"
    )
    assert "IDeepService" in edge_targets, (
        f"DEFECT CONFIRMED: IDeepService edge dropped: {edge_targets}"
    )
