"""Critical C verification tests for tree-sitter polyglot AST parsers."""

from __future__ import annotations

import logging

import pytest

from context_scout.ast_engine import extractor as extractor_mod
from context_scout.ast_engine.extractor import ASTExtractor
from context_scout.plugins.languages import (
    csharp as csharp_mod,
)
from context_scout.plugins.languages import (
    react_redux as react_mod,
)
from context_scout.plugins.languages import (
    storybook as storybook_mod,
)
from context_scout.plugins.languages.csharp import CSharpPlugin
from context_scout.plugins.languages.react_redux import ReactReduxPlugin
from context_scout.plugins.languages.storybook import StorybookPlugin
from context_scout.plugins.languages.terraform import TerraformPlugin


def test_csharp_12_level_nesting_and_4plus_quote_raw_string_decoys() -> None:
    """C# handles 12-level nesting and ignores decoys in 4+/5-quote raw strings."""
    nested_tuple = "(int, " * 11 + "(int, int)" + ")" * 11
    nested_generic = (
        "Task<Dictionary<string, List<Tuple<int, Func<string, ValueTask<Option<Result<int>>>>>>>>"
    )
    code = (
        "namespace CriticalC.CSharp;\n\n"
        f"public class DeepService({nested_tuple} payload) : IDeepContract\n"
        "{\n"
        '    private const string DecoyFour = """"\n'
        "        public class FakeDecoy { public void FakeMethod() {} }\n"
        '        """ unmatched triple quotes and { { }\n'
        '    """";\n\n'
        "    public string BuildTemplate()\n"
        "    {\n"
        '        var sql = $$$$"""""\n'
        "        namespace FakeNs { public class FakeFiveDecoy {} }\n"
        '        {{ "unclosed": "{{" }}\n'
        '        """"";\n'
        "        return sql;\n"
        "    }\n\n"
        f"    public {nested_generic} RunAsync() => default!;\n"
        "}\n\n"
        "public class SiblingRealService\n"
        "{\n"
        "    public void Complete() {}\n"
        "}\n"
    )
    plugin = CSharpPlugin()
    symbols, edges = plugin.extract_ast(code, "src/DeepService.cs", repo_name="critical-c")

    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.parser_backend == "tree-sitter-c-sharp"
    assert plugin.last_parse_diagnostics.has_parse_error is False
    assert plugin.last_parse_diagnostics.abstained is False

    by_name = {s.name: s for s in symbols}
    assert {"DeepService", "BuildTemplate", "RunAsync", "SiblingRealService", "Complete"} <= set(
        by_name
    )
    assert "FakeDecoy" not in by_name
    assert "FakeMethod" not in by_name
    assert "FakeFiveDecoy" not in by_name

    deep_sym = by_name["DeepService"]
    assert deep_sym.symbol_kind == "class"
    assert deep_sym.qualified_name == "CriticalC.CSharp.DeepService"
    assert (
        code.encode("utf-8")[deep_sym.start_byte : deep_sym.end_byte]
        .decode("utf-8")
        .startswith("public class DeepService")
    )
    assert any(
        e.source_qualified_name == "CriticalC.CSharp.DeepService"
        and e.target_qualified_name == "IDeepContract"
        and e.edge_kind == "INHERITS"
        for e in edges
    )


def test_hcl_heredocs_nested_jsonencode_decoys_and_extractor_alignment() -> None:
    """HCL ignores decoy blocks inside heredocs/jsonencode while extracting real interpolations."""
    code = """
variable "env" {
  default = "prod"
}

resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
}

resource "aws_ecs_task_definition" "api" {
  family = "api-${var.env}"
  container_definitions = jsonencode([
    {
      name  = "app"
      image = "https://registry.example.com/app:latest"
      env = {
        DECOY = "resource \\"fake\\" \\"decoy\\" {}"
        VPC   = aws_vpc.main.id
      }
    }
  ])
  user_data = <<-EOT
    #!/bin/bash
    # resource "fake_heredoc" "decoy_block" {
    #   value = "should not exist"
    # }
    echo "vpc=${aws_vpc.main.id} env=${var.env}"
  EOT
}
"""
    plugin = TerraformPlugin()
    symbols, edges = plugin.extract_ast(code, "infra/main.tf", repo_name="infra")

    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.parser_backend == "tree-sitter-hcl"
    assert plugin.last_parse_diagnostics.has_parse_error is False
    assert plugin.last_parse_diagnostics.abstained is False

    by_name = {s.name: s for s in symbols}
    assert {"env", "main", "api"} <= set(by_name)
    assert "decoy" not in by_name
    assert "decoy_block" not in by_name

    api_edges = {
        (e.edge_kind, e.target_qualified_name)
        for e in edges
        if e.source_qualified_name == "infra::infra.main::aws_ecs_task_definition.api"
    }
    assert ("CALLS", "infra::infra.main::aws_vpc.main") in api_edges
    assert ("USES_TYPE", "infra::infra.main::env") in api_edges

    extractor = ASTExtractor()
    ext_syms, ext_edges = extractor.parse_source(
        code, "terraform", "infra/main.tf", repo_name="infra"
    )
    ext_names = {s.name for s in ext_syms}
    assert {"env", "main", "api"} <= ext_names
    assert "decoy" not in ext_names
    assert "decoy_block" not in ext_names
    ext_edge_pairs = {(e.edge_kind, e.target_qualified_name) for e in ext_edges}
    assert ("CALLS", "infra::infra.main::aws_vpc.main") in ext_edge_pairs
    assert ("USES_TYPE", "infra::infra.main::env") in ext_edge_pairs


def test_react_redux_12_level_generics_and_comment_string_decoys() -> None:
    """React/Redux handles 12-level generics/HOCs and ignores comments/string decoys."""
    deep_generic = "A<B<C<D<E<F<G<H<I<J<K<L<string>>>>>>>>>>>>"
    code = f"""
import React, {{ memo, forwardRef }} from 'react';
import {{ useSessionData }} from './hooks';

// export const FakeCommentComponent = () => <div>{{dispatch(fakeAction())}}</div>;
/*
export const fakeSlice = createSlice({{
    name: 'fake',
    reducers: {{ fakeRed() {{}} }}
}});
*/

export const RealSlice = createSlice({{
    name: 'real',
    initialState: {{}},
    reducers: {{
        setFlag: (state) => {{
            const template = "dispatch(decoyStringAction()) useDecoyHook()";
            return state;
        }}
    }}
}});

export const DeepWidget = memo(
    forwardRef<HTMLDivElement, {deep_generic}>((props, ref) => {{
        const session = useSessionData();
        const dispatch = useDispatch();
        const selected = useSelector(selectRealFlag);
        const decoyCode = `export const FakeTemplateComp = () => null;`;
        return <div ref={{ref}} onClick={{() => dispatch(triggerReal())}}>{{decoyCode}}</div>;
    }})
);
"""
    plugin = ReactReduxPlugin()
    symbols, edges = plugin.extract_ast(code, "src/DeepWidget.tsx")

    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.parser_backend == "tree-sitter-typescript"
    assert plugin.last_parse_diagnostics.has_parse_error is False

    names = {s.name for s in symbols}
    assert {"RealSlice", "setFlag", "DeepWidget"} <= names
    assert "FakeCommentComponent" not in names
    assert "fakeSlice" not in names
    assert "fakeRed" not in names
    assert "FakeTemplateComp" not in names

    edge_triples = {
        (e.source_qualified_name.split(".")[-1], e.edge_kind, e.target_qualified_name)
        for e in edges
    }
    assert ("DeepWidget", "CALLS", "useSessionData") in edge_triples
    assert ("DeepWidget", "SELECTS", "selectRealFlag") in edge_triples
    assert ("DeepWidget", "DISPATCHES", "triggerReal") in edge_triples
    assert all(
        tgt not in {"fakeAction", "decoyStringAction", "useDecoyHook"} for _, _, tgt in edge_triples
    )


def test_storybook_comment_string_decoys_and_mdx_low_confidence_flag() -> None:
    """Storybook .stories.tsx ignores comment/string decoys; .mdx flags low_confidence_fallback."""
    ts_code = """
import type { Meta, StoryObj } from '@storybook/react';
import { Card } from './Card';
// import { FakeCommentComp } from './Fake';
/* export const FakeBlockStory: StoryObj = {}; */

const meta: Meta<typeof Card> = {
    title: 'UI/Card',
    component: Card,
};
export default meta;

export const Default: StoryObj<typeof Card> = {
    args: {
        snippet: "export const FakeStringStory = {};",
    }
};
"""
    plugin = StorybookPlugin()
    syms, edges = plugin.extract_ast(ts_code, "src/Card.stories.tsx")

    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.low_confidence_fallback is False
    assert plugin.last_parse_diagnostics.has_parse_error is False

    names = {s.name for s in syms}
    assert names == {"Meta", "Default"}
    assert all(
        e.edge_kind == "DOCUMENTS_COMPONENT" and e.target_qualified_name == "Card" for e in edges
    )
    assert len(edges) == 2

    mdx_code = """
import { Meta, Story } from '@storybook/addon-docs';
import { Card } from './Card';

<Meta title="UI/Card" of={Card} />
<Story name="MdxPrimary" />
"""
    mdx_syms, mdx_edges = plugin.extract_ast(mdx_code, "src/Card.mdx")
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.parser_backend == "mdx_heuristic_fallback"
    assert plugin.last_parse_diagnostics.low_confidence_fallback is True
    assert {s.name for s in mdx_syms} == {"Meta", "MdxPrimary"}
    assert any(
        e.edge_kind == "DOCUMENTS_COMPONENT" and e.target_qualified_name == "Card"
        for e in mdx_edges
    )


@pytest.mark.parametrize(
    ("plugin_factory", "file_path", "corrupted_code", "expected_backend"),
    [
        (
            CSharpPlugin,
            "Broken.cs",
            "namespace Broken; public class Unfinished { public void M(",
            "tree-sitter-c-sharp",
        ),
        (
            TerraformPlugin,
            "broken.tf",
            'resource "aws_s3_bucket" "b" { bucket = ',
            "tree-sitter-hcl",
        ),
        (
            TerraformPlugin,
            "broken.tf.json",
            '{"resource": {"aws_s3_bucket": ',
            "json",
        ),
        (
            ReactReduxPlugin,
            "Broken.tsx",
            "export const Broken = () => { return <div><span>; }",
            "tree-sitter-typescript",
        ),
        (
            StorybookPlugin,
            "Broken.stories.tsx",
            "import { Card } from './Card'; export const Bad = { args: (",
            "tree-sitter-typescript",
        ),
    ],
)
def test_parse_errors_abstain_deterministically(
    plugin_factory: type[CSharpPlugin | TerraformPlugin | ReactReduxPlugin | StorybookPlugin],
    file_path: str,
    corrupted_code: str,
    expected_backend: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every tree-sitter / JSON parser abstains fail-closed with diagnostics on syntax errors."""
    plugin = plugin_factory()
    with caplog.at_level(logging.WARNING):
        symbols, edges = plugin.extract_ast(corrupted_code, file_path)

    assert symbols == []
    assert edges == []
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.parser_backend == expected_backend
    assert plugin.last_parse_diagnostics.has_parse_error is True
    assert plugin.last_parse_diagnostics.abstained is True
    assert any(rec.levelno == logging.WARNING for rec in caplog.records)


def test_negative_control_weakened_error_gate_fails_abstention_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control: disabling inspect_tree_errors leaks partial symbols."""
    corrupted_cs = "namespace Broken; public class PartialLeak { } public class Unclosed {"
    plugin = CSharpPlugin()

    # Baseline fail-closed gate must abstain cleanly.
    syms_safe, edges_safe = plugin.extract_ast(corrupted_cs, "PartialLeak.cs")
    assert syms_safe == []
    assert edges_safe == []
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.abstained is True

    # Negative control: weaken the error-inspection gate in all modules.
    monkeypatch.setattr(csharp_mod, "inspect_tree_errors", lambda _root: (False, 0, 0))
    monkeypatch.setattr(extractor_mod, "inspect_tree_errors", lambda _root: (False, 0, 0))
    monkeypatch.setattr(react_mod, "inspect_tree_errors", lambda _root: (False, 0, 0))
    monkeypatch.setattr(storybook_mod, "inspect_tree_errors", lambda _root: (False, 0, 0))

    syms_weak, _edges_weak = plugin.extract_ast(corrupted_cs, "PartialLeak.cs")
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.abstained is False
    assert any(s.name == "PartialLeak" for s in syms_weak)


def test_manifest_error_logging_replaces_silent_swallow(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Invalid package.json and .terraform.lock.hcl log explicit warnings instead of silent pass."""
    react_plugin = ReactReduxPlugin()
    with caplog.at_level(logging.WARNING):
        npm_deps = react_plugin.extract_manifest_dependencies("package.json", "{invalid-json")
    assert npm_deps == []
    assert react_plugin.last_parse_diagnostics is not None
    assert react_plugin.last_parse_diagnostics.has_parse_error is True
    assert react_plugin.last_parse_diagnostics.abstained is True
    assert any("package.json" in rec.message for rec in caplog.records)

    caplog.clear()
    tf_plugin = TerraformPlugin()
    with caplog.at_level(logging.WARNING):
        tf_deps = tf_plugin.extract_manifest_dependencies(
            ".terraform.lock.hcl", 'provider "aws" { version = '
        )
    assert tf_deps == []
    assert tf_plugin.last_parse_diagnostics is not None
    assert tf_plugin.last_parse_diagnostics.has_parse_error is True
    assert tf_plugin.last_parse_diagnostics.abstained is True
    assert any(rec.levelno == logging.WARNING for rec in caplog.records)
