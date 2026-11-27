"""Tests for M2.02: Templates/i18n, React-Email/i18n, Storybook, and GCP/multi-cloud Terraform."""

from __future__ import annotations

import inspect
import logging
import time

import pytest

from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol
from context_scout.plugins.base import LanguagePlugin
from context_scout.plugins.languages.react_redux import ReactReduxPlugin
from context_scout.plugins.languages.storybook import StorybookPlugin
from context_scout.plugins.languages.templates_i18n import TemplatesI18nPlugin
from context_scout.plugins.languages.terraform import TerraformPlugin

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
        "BINDS_SERVICE_ACCOUNT",
        "ACCESSES_SECRET",
    }
)


def _assert_valid_contracts(symbols: list[ParsedSymbol], edges: list[ParsedEdge]) -> None:
    for sym in symbols:
        assert sym.symbol_kind in ALLOWED_SYMBOL_KINDS, f"Invalid symbol_kind: {sym.symbol_kind}"
    for edge in edges:
        assert edge.edge_kind in ALLOWED_EDGE_KINDS, f"Invalid edge_kind: {edge.edge_kind}"


def test_templates_i18n_protocol_and_can_handle() -> None:
    plugin = TemplatesI18nPlugin()
    assert isinstance(plugin, LanguagePlugin)
    assert plugin.language_id == "templates_i18n"

    for path in (
        "emails/welcome.mjml",
        "templates/invoice.hbs",
        "templates/footer.handlebars",
        "prompts/system.j2",
        "views/index.jinja2",
        "locale/en/LC_MESSAGES/messages.po",
        "locale/messages.pot",
        "src/locales/en.json",
        "src/i18n/de.yaml",
        "translations/common.fr.yml",
        "app/common.en.json",
        "messages.es.json",
    ):
        assert plugin.can_handle(path) is True, f"Expected can_handle({path}) == True"

    for non_locale in (
        "package.json",
        "domain.yaml",
        "openapi.yaml",
        "infra/main.tf.json",
        "tsconfig.json",
        "src/App.tsx",
    ):
        assert plugin.can_handle(non_locale) is False, f"Expected can_handle({non_locale}) == False"


def test_templates_i18n_mjml_extraction_and_decoys() -> None:
    plugin = TemplatesI18nPlugin()
    code = """
    <!-- <mj-section><mj-button>GhostButton</mj-button></mj-section> -->
    <mjml>
      <mj-body>
        <mj-include path="./partials/header.mjml" />
        <mj-section>
          <mj-column>
            <mj-text>Welcome</mj-text>
            <mj-button href="https://example.com">Activate</mj-button>
          </mj-column>
        </mj-section>
      </mj-body>
    </mjml>
    """
    syms, edges = plugin.extract_ast(code, file_path="emails/welcome.mjml")
    _assert_valid_contracts(syms, edges)
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.has_parse_error is False
    assert plugin.last_parse_diagnostics.abstained is False
    assert plugin.last_parse_diagnostics.low_confidence_fallback is False

    names = {s.bare_name or s.name for s in syms}
    assert {"mjml", "mj-body", "mj-section", "mj-column", "mj-text", "mj-button"} <= names
    assert "GhostButton" not in str(syms)
    assert any(
        e.edge_kind == "IMPORTS" and e.target_qualified_name == "./partials/header.mjml"
        for e in edges
    )


def test_templates_i18n_handlebars_extraction() -> None:
    plugin = TemplatesI18nPlugin()
    code = """
    {{! {{> ghost_partial}} }}
    {{!-- {{#if ghost_helper}} {{/if}} --}}
    {{> header_nav}}
    {{#if isAdmin}}
      <span>{{userName}}</span>
    {{/if}}
    """
    syms, edges = plugin.extract_ast(code, file_path="templates/dashboard.hbs")
    _assert_valid_contracts(syms, edges)
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.abstained is False

    names = {s.bare_name or s.name for s in syms}
    assert {"header_nav", "if", "userName"} <= names
    assert "ghost_partial" not in names
    assert "ghost_helper" not in names
    assert any(e.edge_kind == "IMPORTS" and e.target_qualified_name == "header_nav" for e in edges)
    assert any(e.edge_kind == "CALLS" and e.target_qualified_name == "header_nav" for e in edges)


def test_templates_i18n_jinja2_extraction() -> None:
    plugin = TemplatesI18nPlugin()
    code = """
    {# {% macro ghost_macro() %}{% endmacro %} #}
    {% extends "base.html.j2" %}
    {% import "macros/forms.j2" as forms %}
    {% include "partials/banner.j2" %}
    {% macro render_badge(label) %}
      <span>{{ label }}</span>
    {% endmacro %}
    {% block content %}
      <p>{{ current_user.email }}</p>
    {% endblock %}
    """
    syms, edges = plugin.extract_ast(code, file_path="views/profile.j2")
    _assert_valid_contracts(syms, edges)
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.abstained is False

    names = {s.name for s in syms}
    assert {"render_badge", "content"} <= names
    assert "ghost_macro" not in names
    assert any(
        e.edge_kind == "INHERITS" and e.target_qualified_name == "base.html.j2" for e in edges
    )
    assert any(
        e.edge_kind == "IMPORTS" and e.target_qualified_name == "macros/forms.j2" for e in edges
    )
    assert any(
        e.edge_kind == "IMPORTS" and e.target_qualified_name == "partials/banner.j2" for e in edges
    )
    assert any(
        e.edge_kind == "USES_TYPE" and e.target_qualified_name == "current_user" for e in edges
    )


def test_templates_i18n_po_json_yaml_extraction() -> None:
    plugin = TemplatesI18nPlugin()

    po_code = """
# msgid "ghost.comment.key"
msgid ""
msgstr "Content-Type: text/plain; charset=UTF-8\\n"

msgid "auth.welcome"
msgstr "Hello {username}, you have %(count)d alerts"
"""
    po_syms, po_edges = plugin.extract_ast(po_code, file_path="locales/en/messages.po")
    _assert_valid_contracts(po_syms, po_edges)
    assert [s.name for s in po_syms] == ["auth.welcome"]
    assert "@i18n_key auth.welcome" in po_syms[0].docstring
    assert {e.target_qualified_name for e in po_edges} == {"username", "count"}

    json_code = '{"auth": {"login": "Sign in as {email}"}}'
    j_syms, j_edges = plugin.extract_ast(json_code, file_path="src/locales/en.json")
    _assert_valid_contracts(j_syms, j_edges)
    assert [s.name for s in j_syms] == ["auth.login"]
    assert "@i18n_key auth.login" in j_syms[0].docstring
    assert any(e.edge_kind == "USES_TYPE" and e.target_qualified_name == "email" for e in j_edges)

    yaml_code = """
# ghost:
#   key: "ignored"
billing:
  invoice_due: "Invoice {{invoice_id}} due"
"""
    y_syms, y_edges = plugin.extract_ast(yaml_code, file_path="src/i18n/en.yaml")
    _assert_valid_contracts(y_syms, y_edges)
    assert [s.name for s in y_syms] == ["billing.invoice_due"]
    assert "@i18n_key billing.invoice_due" in y_syms[0].docstring
    assert any(
        e.edge_kind == "USES_TYPE" and e.target_qualified_name == "invoice_id" for e in y_edges
    )


@pytest.mark.parametrize(
    ("file_path", "bad_code"),
    [
        ("emails/bad.mjml", "<mjml><mj-body><mj-section></mjml>"),
        ("templates/bad.hbs", "{{#if open}}<span>hi</span>{{/each}}"),
        ("views/bad.j2", "{% block broken %}{% if x %}{{ x }}{% endblock %}"),
        ("locales/bad.po", 'msgid "unterminated\nmsgstr "bad"'),
        ("src/locales/bad.json", '{"auth": {"login": }'),
        ("src/i18n/bad.yaml", "auth:\n  login: [unclosed"),
    ],
)
def test_templates_i18n_fail_closed_with_warning(
    caplog: pytest.LogCaptureFixture,
    file_path: str,
    bad_code: str,
) -> None:
    plugin = TemplatesI18nPlugin()
    with caplog.at_level(logging.WARNING):
        syms, edges = plugin.extract_ast(bad_code, file_path=file_path)
    assert syms == []
    assert edges == []
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.has_parse_error is True
    assert plugin.last_parse_diagnostics.abstained is True
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_react_redux_react_email_and_i18n() -> None:
    plugin = ReactReduxPlugin()
    code = """
import React from 'react';
import { Html, Body, Container, Section, Button, Link, Preview } from '@react-email/components';
import { useTranslation, Trans } from 'react-i18next';

// t('ghost.comment.key')
// <Button>GhostButton</Button>
export function WelcomeEmail({ url }: { url: string }) {
  const { t } = useTranslation('emails');
  const decoy = "t('ghost.string.key')";
  return (
    <Html>
      <Preview>{t('email.welcome.preview')}</Preview>
      <Body>
        <Container>
          <Section>
            <Trans i18nKey="email.welcome.heading">Welcome</Trans>
            <Button href={url}>{t('email.welcome.cta')}</Button>
            <Link href={url}>Open</Link>
          </Section>
        </Container>
      </Body>
    </Html>
  );
}

export function IntlBanner({ intl }: { intl: { formatMessage: (d: { id: string }) => string } }) {
  return <span>{intl.formatMessage({ id: 'banner.notice' })}</span>;
}
"""
    syms, edges = plugin.extract_ast(code, file_path="src/emails/WelcomeEmail.tsx")
    _assert_valid_contracts(syms, edges)
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.abstained is False

    email_comp_names = {s.bare_name for s in syms if s.symbol_kind == "component"}
    assert {"Html", "Preview", "Body", "Container", "Section", "Button", "Link"} <= email_comp_names
    assert any(
        e.edge_kind == "IMPORTS" and e.target_qualified_name == "@react-email/components"
        for e in edges
    )

    welcome_sym = next(s for s in syms if s.name == "WelcomeEmail")
    assert "@i18n_ns emails" in welcome_sym.docstring
    assert "@i18n_key email.welcome.preview" in welcome_sym.docstring
    assert "@i18n_key email.welcome.heading" in welcome_sym.docstring
    assert "@i18n_key email.welcome.cta" in welcome_sym.docstring
    assert "@i18n_key ghost.comment.key" not in welcome_sym.docstring
    assert "@i18n_key ghost.string.key" not in welcome_sym.docstring
    assert not any(
        e.target_qualified_name in {"ghost.comment.key", "ghost.string.key"} for e in edges
    )

    intl_sym = next(s for s in syms if s.name == "IntlBanner")
    assert "@i18n_key banner.notice" in intl_sym.docstring
    assert any(
        e.edge_kind == "USES_TYPE" and e.target_qualified_name == "banner.notice" for e in edges
    )


def test_storybook_csf3_and_mdx_contracts() -> None:
    plugin = StorybookPlugin()
    mdx_code = """
import { Meta, Story } from '@storybook/blocks';
import { AlertBanner } from './AlertBanner';

<!-- <Story name="GhostComment" /> -->
```tsx
<Story name="GhostCodeBlock" />
```

<Meta title="Components/AlertBanner" of={AlertBanner} />
<Story name="Critical" />
"""
    syms, edges = plugin.extract_ast(mdx_code, file_path="src/AlertBanner.stories.mdx")
    _assert_valid_contracts(syms, edges)
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.low_confidence_fallback is True
    assert plugin.last_parse_diagnostics.abstained is False
    names = {s.name for s in syms}
    assert names == {"Meta", "Critical"}
    assert any(
        e.edge_kind == "DOCUMENTS_COMPONENT" and e.target_qualified_name == "AlertBanner"
        for e in edges
    )


def test_terraform_gcp_multicloud_and_locals() -> None:
    plugin = TerraformPlugin()
    hcl = """
# resource "google_storage_bucket" "ghost_bucket" {}
variable "project_id" {
  type = string
}

locals {
  cluster_name = "scout-gke-prod"
}

module "vpc" {
  source = "terraform-google-modules/network/google"
}

resource "google_container_cluster" "primary" {
  name     = local.cluster_name
  location = "europe-west3"
  network  = "default"
}

resource "google_iam_member" "runner" {
  role   = "roles/run.invoker"
  member = "serviceAccount:scout@example.iam.gserviceaccount.com"
}

resource "aws_eks_cluster" "backup" {
  name   = "scout-eks"
  region = "eu-central-1"
}

resource "azurerm_kubernetes_cluster" "dr" {
  name     = "scout-aks"
  location = "westeurope"
}
"""
    syms, edges = plugin.extract_ast(hcl, file_path="infra/main.tf")
    _assert_valid_contracts(syms, edges)
    assert plugin.last_parse_diagnostics is not None
    assert plugin.last_parse_diagnostics.abstained is False

    names = {s.name for s in syms}
    assert "local.cluster_name" in names
    assert "ghost_bucket" not in str(syms)

    gke = next(s for s in syms if s.name == "primary")
    assert "@cloud gcp" in gke.docstring
    assert "@network default" in gke.docstring
    assert "@region europe-west3" in gke.docstring

    iam = next(s for s in syms if s.name == "runner")
    assert "@iam_role roles/run.invoker" in iam.docstring

    eks = next(s for s in syms if s.name == "backup")
    assert "@cloud aws" in eks.docstring

    aks = next(s for s in syms if s.name == "dr")
    assert "@cloud azure" in aks.docstring

    assert any(
        e.edge_kind == "IMPORTS"
        and e.target_qualified_name == "terraform-google-modules/network/google"
        for e in edges
    )
    assert any(
        e.edge_kind == "PROVISIONED_BY"
        and e.target_qualified_name == "terraform-google-modules/network/google"
        for e in edges
    )


def test_polyglot_100kb_linear_time_under_250ms() -> None:
    templates_plugin = TemplatesI18nPlugin()
    react_plugin = ReactReduxPlugin()
    storybook_plugin = StorybookPlugin()
    terraform_plugin = TerraformPlugin()

    # 100 KB+ payloads for each plugin/format
    mjml_100k = (
        "<mjml><mj-body>"
        + ("<mj-section><mj-column><mj-text>Hi</mj-text></mj-column></mj-section>\n" * 1500)
        + "</mj-body></mjml>"
    )
    hbs_100k = "{{#if ok}}" + ("<span>{{userName}}</span>\n" * 4500) + "{{/if}}"
    j2_100k = "{% block body %}\n" + ("<p>{{ item_name }}</p>\n" * 4800) + "{% endblock %}"
    po_100k = "".join(f'msgid "key.{i}"\nmsgstr "Val {i}"\n\n' for i in range(3200))
    mdx_100k = (
        '<Meta title="Perf" of={Btn} />\n'
        + ("<Meta malformed_no_close\n" * 4500)
        + '<Story name="Primary" />\n'
    )
    tsx_100k = (
        "import { Html, Button } from '@react-email/components';\n"
        "export function Email() { return (<Html>"
        + ("<Button>Go</Button>\n" * 5000)
        + "</Html>); }\n"
    )
    tf_100k = "".join(
        f'resource "google_storage_bucket" "b_{i}" {{\n'
        f'  name = "bucket-{i}"\n  location = "EU"\n}}\n'
        for i in range(1300)
    )

    cases: list[tuple[LanguagePlugin, str, str]] = [
        (templates_plugin, mjml_100k, "emails/big.mjml"),
        (templates_plugin, hbs_100k, "templates/big.hbs"),
        (templates_plugin, j2_100k, "views/big.j2"),
        (templates_plugin, po_100k, "locales/big.po"),
        (storybook_plugin, mdx_100k, "src/Big.stories.mdx"),
        (react_plugin, tsx_100k, "src/BigEmail.tsx"),
        (terraform_plugin, tf_100k, "infra/big.tf"),
    ]
    for plugin, payload, path in cases:
        assert len(payload.encode("utf-8")) >= 100_000, f"{path} payload < 100 KB"
        t0 = time.perf_counter()
        syms, edges = plugin.extract_ast(payload, file_path=path)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 250.0, f"{path} took {elapsed_ms:.1f} ms (>= 250 ms)"
        assert len(syms) > 0 or len(edges) > 0


def test_terraform_extract_ast_single_parse_and_no_model_copy_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TerraformPlugin.extract_ast parses HCL once with 0 model_copy calls."""
    from typing import Any

    import context_scout.ast_engine.extractor as extractor_mod
    import context_scout.plugins.languages.terraform as terraform_mod

    real_parser = extractor_mod._get_hcl_parser()
    parse_calls = 0
    model_copy_calls = 0

    class _CountingParser:
        def parse(self, *args: Any, **kwargs: Any) -> Any:
            nonlocal parse_calls
            parse_calls += 1
            return real_parser.parse(*args, **kwargs)

    counting_parser = _CountingParser()
    monkeypatch.setattr(extractor_mod, "_get_hcl_parser", lambda: counting_parser)
    monkeypatch.setattr(terraform_mod, "_get_hcl_parser", lambda: counting_parser)

    orig_model_copy = ParsedSymbol.model_copy

    def _counting_model_copy(self: ParsedSymbol, *args: Any, **kwargs: Any) -> ParsedSymbol:
        nonlocal model_copy_calls
        model_copy_calls += 1
        return orig_model_copy(self, *args, **kwargs)

    monkeypatch.setattr(ParsedSymbol, "model_copy", _counting_model_copy)

    rich_hcl = """
variable "region" {
  type = string
}

locals {
  svc_name = "scout-api"
}

module "network" {
  source = "terraform-google-modules/network/google"
}

resource "google_cloud_run_v2_service" "api" {
  name     = "scout-api"
  location = "europe-west3"
  network  = "default"
  depends_on = [module.network]
  env_vars = {
    SCOUT_REGION = var.region
  }
}

resource "google_iam_member" "invoker" {
  role   = "roles/run.invoker"
  member = "serviceAccount:api@example.iam.gserviceaccount.com"
}
"""
    plugin = TerraformPlugin()
    syms, edges = plugin.extract_ast(rich_hcl, file_path="infra/service.tf")
    _assert_valid_contracts(syms, edges)
    assert parse_calls == 1
    assert model_copy_calls == 0

    sym_by_name = {s.name: s for s in syms}
    assert "local.svc_name" in sym_by_name
    assert "@env TF_VAR_region" in sym_by_name["region"].docstring
    assert (
        "@module_source terraform-google-modules/network/google" in sym_by_name["network"].docstring
    )
    api_sym = sym_by_name["api"]
    for expected_tag in (
        "@cloud gcp",
        "@network default",
        "@region europe-west3",
        "@service scout-api",
        "@env SCOUT_REGION",
    ):
        assert expected_tag in api_sym.docstring
        assert expected_tag in api_sym.contextual_summary
    assert "@iam_role roles/run.invoker" in sym_by_name["invoker"].docstring
    assert any(
        e.edge_kind == "IMPORTS"
        and e.target_qualified_name == "terraform-google-modules/network/google"
        for e in edges
    )
    assert any(
        e.edge_kind == "PROVISIONED_BY"
        and e.target_qualified_name == "terraform-google-modules/network/google"
        for e in edges
    )
    assert any(
        e.edge_kind == "USES_TYPE" and e.target_qualified_name.endswith(".region") for e in edges
    )

    # Also verify 100 KB TF payload executes with 1 parse, 0 model_copy, and < 250.0 ms
    parse_calls = 0
    model_copy_calls = 0
    tf_100k = "".join(
        f'resource "google_storage_bucket" "b_{i}" {{\n'
        f'  name = "bucket-{i}"\n  location = "EU"\n}}\n'
        for i in range(1300)
    )
    t0 = time.perf_counter()
    big_syms, big_edges = plugin.extract_ast(tf_100k, file_path="infra/big.tf")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert parse_calls == 1
    assert model_copy_calls == 0
    assert len(big_syms) == 1300
    assert big_edges == []
    assert elapsed_ms < 250.0
    assert "@cloud gcp" in big_syms[0].docstring
    assert "@service bucket-0" in big_syms[0].docstring
    assert "@region EU" in big_syms[0].docstring


def test_terraform_big_hcl_under_250ms_with_gc_pressure() -> None:
    """inspect_tree_errors fast-path, .tf parity, 500k+ GC pressure <250ms & restore."""
    import gc
    from pathlib import Path

    from context_scout.ast_engine.extractor import (
        _get_hcl_parser,
        inspect_tree_errors,
        parse_hcl_ast,
    )

    plugin = TerraformPlugin()
    tf_100k = "".join(
        f'resource "google_storage_bucket" "b_{i}" {{\n'
        f'  name = "bucket-{i}"\n  location = "EU"\n}}\n'
        for i in range(1300)
    )

    # 1. inspect_tree_errors root_node.has_error fast-path parity on valid & malformed ASTs
    hcl_tree = _get_hcl_parser().parse(tf_100k.encode("utf-8"))
    t0 = time.perf_counter()
    has_err, err_cnt, miss_cnt = inspect_tree_errors(hcl_tree.root_node)
    err_ms = (time.perf_counter() - t0) * 1000.0
    assert (has_err, err_cnt, miss_cnt) == (False, 0, 0)
    assert err_ms < 10.0, f"inspect_tree_errors fast-path took {err_ms:.2f} ms"

    bad_hcl_tree = _get_hcl_parser().parse(b'resource "aws_s3_bucket" "broken" { = = ')
    bad_has_err, bad_err_cnt, bad_miss_cnt = inspect_tree_errors(bad_hcl_tree.root_node)
    assert bad_has_err is True
    assert bad_err_cnt + bad_miss_cnt > 0

    # 2. Parity vs parse_hcl_ast across all repo deploy/terraform/*/*.tf (incl. aws-eks/main.tf)
    repo_root = Path(__file__).resolve().parents[1]
    tf_files = sorted((repo_root / "deploy" / "terraform").glob("*/*.tf"))
    assert len(tf_files) >= 10
    for tf_path in tf_files:
        rel = str(tf_path.relative_to(repo_root))
        src = tf_path.read_text(encoding="utf-8")
        prefix = plugin._derive_module_prefix(rel)
        base_syms, base_edges, base_diag = parse_hcl_ast(src, file_path=rel, module_prefix=prefix)
        plug_syms, plug_edges = plugin.extract_ast(src, file_path=rel)
        assert plugin.last_parse_diagnostics == base_diag
        # Every base parse_hcl_ast symbol must appear in order before trailing local.* symbols
        assert len(plug_syms) >= len(base_syms)
        for bs, ps in zip(base_syms, plug_syms[: len(base_syms)], strict=True):
            assert ps.name == bs.name
            assert ps.qualified_name == bs.qualified_name
            assert ps.symbol_kind == bs.symbol_kind
            assert ps.start_line == bs.start_line
            assert ps.end_line == bs.end_line
            assert ps.signature_skeleton == bs.signature_skeleton
        # Base parse_hcl_ast edges must match the prefix of TerraformPlugin edges in exact order
        assert len(plug_edges) >= len(base_edges)
        assert [e.model_dump() for e in plug_edges[: len(base_edges)]] == [
            e.model_dump() for e in base_edges
        ]
        _assert_valid_contracts(plug_syms, plug_edges)

    # 3. < 250.0 ms under 500,000+ live GC-tracked container objects without caller gc.disable()
    arena: list[list[int]] = [[i] for i in range(520_000)]
    gc_phases: list[str] = []

    def _gc_spy(phase: str, _info: dict[str, int]) -> None:
        gc_phases.append(phase)
        assert gc.isenabled() is True

    try:
        assert "gc.disable" not in inspect.getsource(TerraformPlugin.extract_ast)
        assert len(arena) == 520_000
        assert gc.isenabled() is True
        gc.callbacks.append(_gc_spy)
        t_start = time.perf_counter()
        syms_big, edges_big = plugin.extract_ast(tf_100k, file_path="infra/big.tf")
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        assert elapsed_ms < 250.0, f"infra/big.tf under 520k GC arena took {elapsed_ms:.1f} ms"
        assert len(syms_big) == 1300
        assert edges_big == []
        assert gc.isenabled() is True

        # Syntax error (abstained) path must also preserve gc.isenabled() == True
        bad_syms, bad_edges = plugin.extract_ast(
            'resource "google_storage_bucket" "broken" { =', file_path="infra/bad.tf"
        )
        assert (bad_syms, bad_edges) == ([], [])
        assert plugin.last_parse_diagnostics is not None
        assert plugin.last_parse_diagnostics.abstained is True
        assert gc.isenabled() is True

        # Pre-disabled GC must remain disabled after extract_ast returns
        gc.disable()
        try:
            plugin.extract_ast(tf_100k, file_path="infra/big.tf")
            assert gc.isenabled() is False
        finally:
            gc.enable()
    finally:
        if _gc_spy in gc.callbacks:
            gc.callbacks.remove(_gc_spy)
        arena.clear()
