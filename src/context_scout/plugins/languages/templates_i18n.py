"""Templates (MJML, Handlebars, Jinja2) and i18n (.po/.pot, locale JSON/YAML) plugin."""

from __future__ import annotations

import gc
import json
import logging
import re
from pathlib import Path
from typing import Literal

import jinja2
import jinja2.meta
import jinja2.nodes
import tree_sitter_language_pack as tslp
from tree_sitter import Node, Parser

from context_scout.ast_engine.extractor import (
    ParseDiagnostics,
    _node_text,
    inspect_tree_errors,
)
from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol
from context_scout.plugins.base import DependencyEdge

logger = logging.getLogger(__name__)

_HTML_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
        "mj-include",
    }
)

_MJML_COMPONENT_TAGS = frozenset(
    {
        "mj-section",
        "mj-column",
        "mj-button",
        "mj-text",
        "mj-image",
        "mj-divider",
        "mj-spacer",
        "mj-navbar",
        "mj-hero",
        "mj-wrapper",
        "mj-group",
        "mj-table",
        "mj-social",
        "mj-accordion",
        "mj-carousel",
    }
)

_MJML_MODULE_TAGS = frozenset({"mjml", "mj-head", "mj-body", "mj-include"})

_LOCALE_DIR_MARKERS = ("/locales/", "/i18n/", "/translations/", "/lang/")
_RE_LOCALE_FILENAME = re.compile(
    r"^(?:messages(?:[._-][a-z]{2,3}(?:[_-][A-Za-z0-9]{2,4})?)?|"
    r".+\.[a-z]{2,3}(?:[_-][A-Za-z0-9]{2,4})?)\.(?:json|ya?ml)$",
    re.IGNORECASE,
)
_EXCLUDED_CONFIG_NAMES = frozenset(
    {
        "package.json",
        "package-lock.json",
        "tsconfig.json",
        "domain.yaml",
        "domain.yml",
        "openapi.yaml",
        "openapi.yml",
        "openapi.json",
        "docker-compose.yaml",
        "docker-compose.yml",
    }
)
_RE_PLACEHOLDER = re.compile(
    r"\{\{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\}\}"
    r"|\{([A-Za-z_][A-Za-z0-9_.]*)\}"
    r"|%\(([A-Za-z_][A-Za-z0-9_]*)\)[sd]"
)
_RE_HBS_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_./-]*$")
_HBS_IGNORED_HELPERS = frozenset({"else", "this", "log", "lookup"})


def _is_locale_catalog_path(file_path: str) -> bool:
    normalized = file_path.replace("\\", "/").lower()
    if normalized.endswith(".tf.json"):
        return False
    name = Path(normalized).name
    if name in _EXCLUDED_CONFIG_NAMES:
        return False
    if not normalized.endswith((".json", ".yaml", ".yml")):
        return False
    if any(marker in normalized for marker in _LOCALE_DIR_MARKERS):
        return True
    return bool(_RE_LOCALE_FILENAME.match(name))


def _unquote(text: str) -> str:
    s = text.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', "'"}:
        return s[1:-1]
    return s


def _extract_placeholders(text: str) -> list[str]:
    found: list[str] = []
    for match in _RE_PLACEHOLDER.finditer(text):
        param = match.group(1) or match.group(2) or match.group(3)
        if param and param not in found:
            found.append(param)
    return found


class TemplatesI18nPlugin:
    """Language plugin for MJML, Handlebars, Jinja2, Gettext (.po/.pot), and locale JSON/YAML."""

    def __init__(self) -> None:
        self._html_parser: Parser | None = None
        self._hbs_parser: Parser | None = None
        self._po_parser: Parser | None = None
        self._json_parser: Parser | None = None
        self._yaml_parser: Parser | None = None
        self._jinja_env = jinja2.Environment(autoescape=False)
        self.last_parse_diagnostics: ParseDiagnostics | None = None

    def _get_parser(
        self, lang: Literal["html", "embeddedtemplate", "po", "json", "yaml"]
    ) -> Parser:
        cache_attr = {
            "html": "_html_parser",
            "embeddedtemplate": "_hbs_parser",
            "po": "_po_parser",
            "json": "_json_parser",
            "yaml": "_yaml_parser",
        }[lang]
        cached: Parser | None = getattr(self, cache_attr)
        if cached is None:
            cached = tslp.get_parser(lang)
            setattr(self, cache_attr, cached)
        return cached

    @property
    def language_id(self) -> str:
        return "templates_i18n"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (
            ".mjml",
            ".hbs",
            ".handlebars",
            ".j2",
            ".jinja",
            ".jinja2",
            ".po",
            ".pot",
        )

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ()

    def can_handle(self, file_path: str) -> bool:
        lower = file_path.lower()
        if any(lower.endswith(ext) for ext in self.file_extensions):
            return True
        return _is_locale_catalog_path(file_path)

    def _derive_module_prefix(self, file_path: str) -> str:
        if not file_path:
            return "templates_i18n"
        stem = re.sub(
            r"\.(?:mjml|hbs|handlebars|j2|jinja2?|po|pot|json|ya?ml)$",
            "",
            file_path,
            flags=re.IGNORECASE,
        )
        normalized = stem.replace("\\", "/").strip("/")
        dotted = ".".join(part for part in normalized.split("/") if part)
        return dotted or "templates_i18n"

    def extract_manifest_dependencies(
        self,
        file_path: str,
        content: str,
    ) -> list[DependencyEdge]:
        return []

    def _abstain(
        self,
        backend: str,
        reason: str,
        *,
        error_count: int = 1,
        missing_count: int = 0,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        logger.warning("TemplatesI18nPlugin parse error (%s): %s; abstaining.", backend, reason)
        self.last_parse_diagnostics = ParseDiagnostics(
            parser_backend=backend,
            has_parse_error=True,
            abstained=True,
            low_confidence_fallback=False,
            error_node_count=error_count,
            missing_node_count=missing_count,
            reason=reason,
        )
        return [], []

    def extract_ast(
        self,
        code: str,
        file_path: str = "",
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        if not code or not code.strip():
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="templates_i18n")
            return [], []

        prefix = module_prefix or self._derive_module_prefix(file_path)
        lower = file_path.lower()

        was_gc = gc.isenabled()
        gc.disable()
        try:
            if lower.endswith(".mjml"):
                return self._extract_mjml(code, file_path, repo_name, prefix)
            if lower.endswith((".hbs", ".handlebars")):
                return self._extract_handlebars(code, file_path, repo_name, prefix)
            if lower.endswith((".j2", ".jinja", ".jinja2")):
                return self._extract_jinja2(code, file_path, repo_name, prefix)
            if lower.endswith((".po", ".pot")):
                return self._extract_po(code, file_path, repo_name, prefix)
            if lower.endswith(".json"):
                return self._extract_locale_json(code, file_path, repo_name, prefix)
            if lower.endswith((".yaml", ".yml")):
                return self._extract_locale_yaml(code, file_path, repo_name, prefix)

            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="templates_i18n")
            return [], []
        finally:
            if was_gc:
                gc.enable()

    def _extract_mjml(
        self,
        code: str,
        file_path: str,
        repo_name: str,
        prefix: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        parser = self._get_parser("html")
        code_bytes = code.encode("utf-8")
        tree = parser.parse(code_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            return self._abstain(
                "tree-sitter-html",
                f"Syntax error in MJML '{file_path or '<memory>'}'",
                error_count=err_cnt,
                missing_count=miss_cnt,
            )

        # Validate balanced tags iteratively (tree-sitter-html allows unclosed tags)
        stack: list[Node] = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type == "erroneous_end_tag":
                return self._abstain(
                    "tree-sitter-html",
                    f"Mismatched closing tag in MJML '{file_path or '<memory>'}'",
                )
            if node.type == "element":
                start_tag = next((c for c in node.children if c.type == "start_tag"), None)
                end_tag = next((c for c in node.children if c.type == "end_tag"), None)
                if start_tag is not None:
                    tname_nd = next((c for c in start_tag.children if c.type == "tag_name"), None)
                    tname = _node_text(tname_nd, code_bytes).lower() if tname_nd is not None else ""
                    if tname and tname not in _HTML_VOID_TAGS and end_tag is None:
                        return self._abstain(
                            "tree-sitter-html",
                            f"Unclosed <{tname}> tag in MJML '{file_path or '<memory>'}'",
                        )
            stack.extend(reversed(node.children))

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-html")
        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        seen_edges: set[tuple[str, str, str]] = set()
        tag_counts: dict[str, int] = {}

        def _get_attrs(tag_node: Node) -> dict[str, str]:
            attrs: dict[str, str] = {}
            for child in tag_node.children:
                if child.type == "attribute":
                    aname_nd = next((c for c in child.children if c.type == "attribute_name"), None)
                    aval_nd = next(
                        (
                            c
                            for c in child.children
                            if c.type in {"quoted_attribute_value", "attribute_value"}
                        ),
                        None,
                    )
                    if aname_nd is not None:
                        aname = _node_text(aname_nd, code_bytes).strip()
                        aval = (
                            _unquote(_node_text(aval_nd, code_bytes)) if aval_nd is not None else ""
                        )
                        attrs[aname] = aval
            return attrs

        walk_stack: list[Node] = [tree.root_node]
        while walk_stack:
            node = walk_stack.pop()
            if node.type == "comment":
                continue
            if node.type == "element":
                tag_node = next(
                    (c for c in node.children if c.type in {"start_tag", "self_closing_tag"}),
                    None,
                )
                if tag_node is not None:
                    tname_nd = next((c for c in tag_node.children if c.type == "tag_name"), None)
                    tname = _node_text(tname_nd, code_bytes).strip() if tname_nd is not None else ""
                    lower_tname = tname.lower()
                    if lower_tname == "mj-include":
                        attrs = _get_attrs(tag_node)
                        inc_path = attrs.get("path", "")
                        if inc_path:
                            key = (prefix, inc_path, "IMPORTS")
                            if key not in seen_edges:
                                seen_edges.add(key)
                                edges.append(
                                    ParsedEdge(
                                        source_qualified_name=prefix,
                                        target_qualified_name=inc_path,
                                        edge_kind="IMPORTS",
                                    )
                                )
                    elif (
                        lower_tname in _MJML_COMPONENT_TAGS
                        or lower_tname in _MJML_MODULE_TAGS
                        or "-" in lower_tname
                    ):
                        attrs = _get_attrs(tag_node)
                        css_class = attrs.get("css-class", "")
                        idx = tag_counts.get(tname, 0) + 1
                        tag_counts[tname] = idx
                        sym_name = f"{tname}_{idx}" if idx > 1 else tname
                        kind = "module" if lower_tname in _MJML_MODULE_TAGS else "component"
                        qn = f"{prefix}.{sym_name}"
                        sig = (
                            f"<{tname}>" if not css_class else f'<{tname} css-class="{css_class}">'
                        )
                        symbols.append(
                            ParsedSymbol(
                                name=sym_name,
                                qualified_name=qn,
                                bare_name=tname,
                                symbol_kind=kind,
                                start_line=node.start_point[0] + 1,
                                end_line=node.end_point[0] + 1,
                                start_byte=node.start_byte,
                                end_byte=node.end_byte,
                                signature=sig,
                                signature_skeleton=f"<{tname}>",
                                repo_name=repo_name,
                            )
                        )
            walk_stack.extend(reversed(node.children))

        return symbols, edges

    def _extract_handlebars(
        self,
        code: str,
        file_path: str,
        repo_name: str,
        prefix: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        parser = self._get_parser("embeddedtemplate")
        code_bytes = code.encode("utf-8")
        tree = parser.parse(code_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            return self._abstain(
                "tree-sitter-embeddedtemplate",
                f"Syntax error in Handlebars '{file_path or '<memory>'}'",
                error_count=err_cnt,
                missing_count=miss_cnt,
            )

        # Linear O(n) Handlebars mustache & block-helper scanner
        n = len(code)
        i = 0
        line_no = 1
        block_stack: list[str] = []
        partials: list[tuple[str, int, int, int]] = []
        helpers: list[tuple[str, int, int, int]] = []
        expressions: list[tuple[str, int, int, int]] = []

        while i < n:
            if code.startswith("<!--", i):
                end = code.find("-->", i + 4)
                if end == -1:
                    return self._abstain(
                        "tree-sitter-embeddedtemplate",
                        f"Unclosed HTML comment in Handlebars '{file_path or '<memory>'}'",
                    )
                line_no += code.count("\n", i, end + 3)
                i = end + 3
                continue

            if code.startswith("{{!--", i):
                end = code.find("--}}", i + 5)
                if end == -1:
                    return self._abstain(
                        "tree-sitter-embeddedtemplate",
                        f"Unclosed Handlebars block comment in '{file_path or '<memory>'}'",
                    )
                line_no += code.count("\n", i, end + 4)
                i = end + 4
                continue

            if code.startswith("{{!", i):
                end = code.find("}}", i + 3)
                if end == -1:
                    return self._abstain(
                        "tree-sitter-embeddedtemplate",
                        f"Unclosed Handlebars comment in '{file_path or '<memory>'}'",
                    )
                line_no += code.count("\n", i, end + 2)
                i = end + 2
                continue

            if code.startswith("{{{", i):
                end = code.find("}}}", i + 3)
                if end == -1:
                    return self._abstain(
                        "tree-sitter-embeddedtemplate",
                        f"Unclosed triple-stache '{{{{{{' in '{file_path or '<memory>'}'",
                    )
                expr_text = code[i + 3 : end].strip()
                start_line = line_no
                line_no += code.count("\n", i, end + 3)
                if expr_text:
                    token = expr_text.split()[0]
                    if _RE_HBS_IDENT.match(token) and token not in _HBS_IGNORED_HELPERS:
                        expressions.append((token, start_line, i, end + 3))
                i = end + 3
                continue

            if code.startswith("{{", i):
                end = code.find("}}", i + 2)
                if end == -1:
                    return self._abstain(
                        "tree-sitter-embeddedtemplate",
                        f"Unclosed mustache '{{{{' in '{file_path or '<memory>'}'",
                    )
                inner = code[i + 2 : end].strip()
                start_line = line_no
                line_no += code.count("\n", i, end + 2)
                if not inner:
                    return self._abstain(
                        "tree-sitter-embeddedtemplate",
                        f"Empty mustache '{{{{}}}}' in '{file_path or '<memory>'}'",
                    )
                if inner.startswith("#"):
                    helper_name = inner[1:].strip().split()[0] if inner[1:].strip() else ""
                    if not helper_name or not _RE_HBS_IDENT.match(helper_name):
                        return self._abstain(
                            "tree-sitter-embeddedtemplate",
                            f"Invalid block helper '{{{{{inner}}}}}' in "
                            f"'{file_path or '<memory>'}'",
                        )
                    block_stack.append(helper_name)
                    helpers.append((helper_name, start_line, i, end + 2))
                elif inner.startswith("/"):
                    close_name = inner[1:].strip().split()[0] if inner[1:].strip() else ""
                    if not block_stack or block_stack[-1] != close_name:
                        return self._abstain(
                            "tree-sitter-embeddedtemplate",
                            f"Mismatched closing block '{{{{/{close_name}}}}}' in "
                            f"'{file_path or '<memory>'}'",
                        )
                    block_stack.pop()
                elif inner.startswith(">"):
                    partial_name = inner[1:].strip().split()[0] if inner[1:].strip() else ""
                    if not partial_name:
                        return self._abstain(
                            "tree-sitter-embeddedtemplate",
                            f"Empty partial '{{{{>}}}}' in '{file_path or '<memory>'}'",
                        )
                    partials.append((partial_name, start_line, i, end + 2))
                else:
                    token = inner.split()[0]
                    if _RE_HBS_IDENT.match(token) and token not in _HBS_IGNORED_HELPERS:
                        expressions.append((token, start_line, i, end + 2))
                i = end + 2
                continue

            if code[i] == "\n":
                line_no += 1
            i += 1

        if block_stack:
            return self._abstain(
                "tree-sitter-embeddedtemplate",
                f"Unclosed Handlebars block(s) {block_stack} in '{file_path or '<memory>'}'",
            )

        self.last_parse_diagnostics = ParseDiagnostics(
            parser_backend="tree-sitter-embeddedtemplate"
        )
        symbols: list[ParsedSymbol] = [
            ParsedSymbol(
                name=Path(file_path).stem or "template",
                qualified_name=prefix,
                bare_name=Path(file_path).stem or "template",
                symbol_kind="module",
                start_line=1,
                end_line=max(1, code.count("\n") + 1),
                start_byte=0,
                end_byte=len(code_bytes),
                signature=f"hbs:{Path(file_path).name or 'template.hbs'}",
                signature_skeleton="hbs",
                repo_name=repo_name,
            )
        ]
        edges: list[ParsedEdge] = []
        seen_edges: set[tuple[str, str, str]] = set()
        seen_qns: set[str] = {prefix}

        def _add_edge(tgt: str, kind: str) -> None:
            k = (prefix, tgt, kind)
            if k not in seen_edges:
                seen_edges.add(k)
                edges.append(
                    ParsedEdge(
                        source_qualified_name=prefix,
                        target_qualified_name=tgt,
                        edge_kind=kind,
                    )
                )

        for pname, sline, sb, eb in partials:
            _add_edge(pname, "IMPORTS")
            _add_edge(pname, "CALLS")
            qn = f"{prefix}.partial.{pname}"
            if qn not in seen_qns:
                seen_qns.add(qn)
                symbols.append(
                    ParsedSymbol(
                        name=pname,
                        qualified_name=qn,
                        bare_name=pname,
                        symbol_kind="component",
                        start_line=sline,
                        end_line=sline,
                        start_byte=sb,
                        end_byte=eb,
                        signature=f"{{{{> {pname}}}}}",
                        signature_skeleton=f"{{{{> {pname}}}}}",
                        repo_name=repo_name,
                    )
                )

        for hname, sline, sb, eb in helpers:
            qn = f"{prefix}.helper.{hname}"
            if qn not in seen_qns:
                seen_qns.add(qn)
                symbols.append(
                    ParsedSymbol(
                        name=hname,
                        qualified_name=qn,
                        bare_name=hname,
                        symbol_kind="component",
                        start_line=sline,
                        end_line=sline,
                        start_byte=sb,
                        end_byte=eb,
                        signature=f"{{{{#{hname}}}}}",
                        signature_skeleton=f"{{{{#{hname}}}}}",
                        repo_name=repo_name,
                    )
                )

        for ename, sline, sb, eb in expressions:
            qn = f"{prefix}.expr.{ename}"
            if qn not in seen_qns:
                seen_qns.add(qn)
                symbols.append(
                    ParsedSymbol(
                        name=ename,
                        qualified_name=qn,
                        bare_name=ename,
                        symbol_kind="type_alias",
                        start_line=sline,
                        end_line=sline,
                        start_byte=sb,
                        end_byte=eb,
                        signature=f"{{{{{ename}}}}}",
                        signature_skeleton=f"{{{{{ename}}}}}",
                        repo_name=repo_name,
                    )
                )

        return symbols, edges

    def _extract_jinja2(
        self,
        code: str,
        file_path: str,
        repo_name: str,
        prefix: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        try:
            ast = self._jinja_env.parse(code)
        except jinja2.TemplateSyntaxError as exc:
            return self._abstain(
                "jinja2",
                f"TemplateSyntaxError at line {exc.lineno}: {exc.message}",
            )

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="jinja2")
        code_bytes = code.encode("utf-8")
        total_lines = max(1, code.count("\n") + 1)
        symbols: list[ParsedSymbol] = [
            ParsedSymbol(
                name=Path(file_path).stem or "template",
                qualified_name=prefix,
                bare_name=Path(file_path).stem or "template",
                symbol_kind="module",
                start_line=1,
                end_line=total_lines,
                start_byte=0,
                end_byte=len(code_bytes),
                signature=f"jinja2:{Path(file_path).name or 'template.j2'}",
                signature_skeleton="jinja2",
                repo_name=repo_name,
            )
        ]
        edges: list[ParsedEdge] = []
        seen_edges: set[tuple[str, str, str]] = set()
        seen_qns: set[str] = {prefix}

        def _add_edge(tgt: str, kind: str) -> None:
            if not tgt:
                return
            k = (prefix, tgt, kind)
            if k not in seen_edges:
                seen_edges.add(k)
                edges.append(
                    ParsedEdge(
                        source_qualified_name=prefix,
                        target_qualified_name=tgt,
                        edge_kind=kind,
                    )
                )

        for node in ast.find_all(
            (
                jinja2.nodes.Macro,
                jinja2.nodes.Block,
                jinja2.nodes.Extends,
                jinja2.nodes.Include,
                jinja2.nodes.Import,
                jinja2.nodes.FromImport,
            )
        ):
            lineno = getattr(node, "lineno", 1) or 1
            if isinstance(node, jinja2.nodes.Macro):
                qn = f"{prefix}.{node.name}"
                if qn not in seen_qns:
                    seen_qns.add(qn)
                    args_str = ", ".join(a.name for a in node.args if hasattr(a, "name"))
                    sig = f"macro {node.name}({args_str})"
                    symbols.append(
                        ParsedSymbol(
                            name=node.name,
                            qualified_name=qn,
                            bare_name=node.name,
                            symbol_kind="function",
                            start_line=lineno,
                            end_line=lineno,
                            start_byte=0,
                            end_byte=len(code_bytes),
                            signature=sig,
                            signature_skeleton=sig,
                            repo_name=repo_name,
                        )
                    )
            elif isinstance(node, jinja2.nodes.Block):
                qn = f"{prefix}.block.{node.name}"
                if qn not in seen_qns:
                    seen_qns.add(qn)
                    symbols.append(
                        ParsedSymbol(
                            name=node.name,
                            qualified_name=qn,
                            bare_name=node.name,
                            symbol_kind="component",
                            start_line=lineno,
                            end_line=lineno,
                            start_byte=0,
                            end_byte=len(code_bytes),
                            signature=f"{{% block {node.name} %}}",
                            signature_skeleton=f"{{% block {node.name} %}}",
                            repo_name=repo_name,
                        )
                    )
            elif isinstance(node, jinja2.nodes.Extends):
                if isinstance(node.template, jinja2.nodes.Const) and isinstance(
                    node.template.value, str
                ):
                    _add_edge(node.template.value, "INHERITS")
            elif isinstance(
                node, (jinja2.nodes.Include, jinja2.nodes.Import, jinja2.nodes.FromImport)
            ):
                if isinstance(node.template, jinja2.nodes.Const) and isinstance(
                    node.template.value, str
                ):
                    _add_edge(node.template.value, "IMPORTS")

        for var_name in sorted(jinja2.meta.find_undeclared_variables(ast)):
            _add_edge(var_name, "USES_TYPE")

        return symbols, edges

    def _extract_po(
        self,
        code: str,
        file_path: str,
        repo_name: str,
        prefix: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        parser = self._get_parser("po")
        code_bytes = code.encode("utf-8")
        tree = parser.parse(code_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            return self._abstain(
                "tree-sitter-po",
                f"Syntax error in gettext catalog '{file_path or '<memory>'}'",
                error_count=err_cnt,
                missing_count=miss_cnt,
            )

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-po")
        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        seen_qns: set[str] = set()
        seen_edges: set[tuple[str, str, str]] = set()

        def _extract_po_string(container: Node) -> str:
            parts: list[str] = []
            for child in container.children:
                if child.type == "string":
                    for frag in child.children:
                        if frag.type == "string_fragment":
                            parts.append(_node_text(frag, code_bytes))
            return "".join(parts)

        for node in tree.root_node.children:
            if node.type != "message":
                continue
            msgid_node = next((c for c in node.children if c.type == "msgid"), None)
            msgstr_node = next((c for c in node.children if c.type == "msgstr"), None)
            if msgid_node is None:
                continue
            key = _extract_po_string(msgid_node)
            if not key:
                continue
            val = _extract_po_string(msgstr_node) if msgstr_node is not None else ""
            qn = f"{prefix}.{key}"
            if qn not in seen_qns:
                seen_qns.add(qn)
                symbols.append(
                    ParsedSymbol(
                        name=key,
                        qualified_name=qn,
                        bare_name=key,
                        symbol_kind="type_alias",
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                        signature=f'msgid "{key}"',
                        signature_skeleton=f'msgid "{key}"',
                        docstring=f"@i18n_key {key}",
                        repo_name=repo_name,
                    )
                )
            for param in _extract_placeholders(f"{key} {val}"):
                ekey = (qn, param, "USES_TYPE")
                if ekey not in seen_edges:
                    seen_edges.add(ekey)
                    edges.append(
                        ParsedEdge(
                            source_qualified_name=qn,
                            target_qualified_name=param,
                            edge_kind="USES_TYPE",
                        )
                    )

        return symbols, edges

    def _extract_locale_json(
        self,
        code: str,
        file_path: str,
        repo_name: str,
        prefix: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        parser = self._get_parser("json")
        code_bytes = code.encode("utf-8")
        tree = parser.parse(code_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            return self._abstain(
                "tree-sitter-json",
                f"Syntax error in locale JSON '{file_path or '<memory>'}'",
                error_count=err_cnt,
                missing_count=miss_cnt,
            )
        try:
            json.loads(code)
        except json.JSONDecodeError as exc:
            return self._abstain("tree-sitter-json", f"Invalid JSON: {exc}")

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-json")
        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        seen_qns: set[str] = set()
        seen_edges: set[tuple[str, str, str]] = set()

        def _json_str(str_node: Node) -> str:
            content = next((c for c in str_node.children if c.type == "string_content"), None)
            if content is not None:
                return _node_text(content, code_bytes)
            return _unquote(_node_text(str_node, code_bytes))

        def _walk_obj(obj_node: Node, key_prefix: str) -> None:
            for child in obj_node.children:
                if child.type != "pair":
                    continue
                named = [c for c in child.children if c.is_named]
                if len(named) < 2:
                    continue
                k_nd, v_nd = named[0], named[1]
                if k_nd.type != "string":
                    continue
                k_str = _json_str(k_nd)
                full_key = f"{key_prefix}.{k_str}" if key_prefix else k_str
                if v_nd.type == "object":
                    _walk_obj(v_nd, full_key)
                elif v_nd.type == "string":
                    val_str = _json_str(v_nd)
                    qn = f"{prefix}.{full_key}"
                    if qn not in seen_qns:
                        seen_qns.add(qn)
                        symbols.append(
                            ParsedSymbol(
                                name=full_key,
                                qualified_name=qn,
                                bare_name=full_key,
                                symbol_kind="type_alias",
                                start_line=child.start_point[0] + 1,
                                end_line=child.end_point[0] + 1,
                                start_byte=child.start_byte,
                                end_byte=child.end_byte,
                                signature=f'"{full_key}"',
                                signature_skeleton=f'"{full_key}"',
                                docstring=f"@i18n_key {full_key}",
                                repo_name=repo_name,
                            )
                        )
                    for param in _extract_placeholders(val_str):
                        ekey = (qn, param, "USES_TYPE")
                        if ekey not in seen_edges:
                            seen_edges.add(ekey)
                            edges.append(
                                ParsedEdge(
                                    source_qualified_name=qn,
                                    target_qualified_name=param,
                                    edge_kind="USES_TYPE",
                                )
                            )

        root_obj = next((c for c in tree.root_node.children if c.type == "object"), None)
        if root_obj is not None:
            _walk_obj(root_obj, "")

        return symbols, edges

    def _extract_locale_yaml(
        self,
        code: str,
        file_path: str,
        repo_name: str,
        prefix: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        parser = self._get_parser("yaml")
        code_bytes = code.encode("utf-8")
        tree = parser.parse(code_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            return self._abstain(
                "tree-sitter-yaml",
                f"Syntax error in locale YAML '{file_path or '<memory>'}'",
                error_count=err_cnt,
                missing_count=miss_cnt,
            )

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-yaml")
        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        seen_qns: set[str] = set()
        seen_edges: set[tuple[str, str, str]] = set()

        def _unwrap_flow_scalar(node: Node) -> str:
            txt = _node_text(node, code_bytes).strip()
            return _unquote(txt)

        def _walk_mapping(mapping_node: Node, key_prefix: str) -> None:
            for child in mapping_node.children:
                if child.type != "block_mapping_pair":
                    continue
                named = [c for c in child.children if c.is_named]
                if len(named) < 2:
                    continue
                k_nd, v_nd = named[0], named[1]
                k_str = _unwrap_flow_scalar(k_nd)
                if not k_str:
                    continue
                full_key = f"{key_prefix}.{k_str}" if key_prefix else k_str
                nested_mapping = None
                if v_nd.type == "block_node":
                    nested_mapping = next(
                        (c for c in v_nd.children if c.type == "block_mapping"),
                        None,
                    )
                elif v_nd.type == "block_mapping":
                    nested_mapping = v_nd
                if nested_mapping is not None:
                    _walk_mapping(nested_mapping, full_key)
                else:
                    val_str = _unwrap_flow_scalar(v_nd)
                    qn = f"{prefix}.{full_key}"
                    if qn not in seen_qns:
                        seen_qns.add(qn)
                        symbols.append(
                            ParsedSymbol(
                                name=full_key,
                                qualified_name=qn,
                                bare_name=full_key,
                                symbol_kind="type_alias",
                                start_line=child.start_point[0] + 1,
                                end_line=child.end_point[0] + 1,
                                start_byte=child.start_byte,
                                end_byte=child.end_byte,
                                signature=f"{full_key}:",
                                signature_skeleton=f"{full_key}:",
                                docstring=f"@i18n_key {full_key}",
                                repo_name=repo_name,
                            )
                        )
                    for param in _extract_placeholders(val_str):
                        ekey = (qn, param, "USES_TYPE")
                        if ekey not in seen_edges:
                            seen_edges.add(ekey)
                            edges.append(
                                ParsedEdge(
                                    source_qualified_name=qn,
                                    target_qualified_name=param,
                                    edge_kind="USES_TYPE",
                                )
                            )

        def _find_mappings(node: Node) -> list[Node]:
            found: list[Node] = []
            stack: list[Node] = [node]
            while stack:
                cur = stack.pop()
                if cur.type == "comment":
                    continue
                if cur.type == "block_mapping":
                    found.append(cur)
                    continue
                stack.extend(reversed(cur.children))
            return found

        for mapping in _find_mappings(tree.root_node):
            _walk_mapping(mapping, "")

        return symbols, edges
