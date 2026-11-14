"""Storybook (.stories.* and .mdx) language plugin."""

from __future__ import annotations

import logging
import re

import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Node, Parser

from context_scout.ast_engine.extractor import (
    ParseDiagnostics,
    _node_text,
    inspect_tree_errors,
)
from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol
from context_scout.plugins.base import DependencyEdge

logger = logging.getLogger(__name__)

_TS_LANGUAGE = Language(tstypescript.language_typescript())
_TSX_LANGUAGE = Language(tstypescript.language_tsx())

_SKIPPED_IMPORT_NAMES = frozenset({"React", "FC", "Props", "Story", "StoryObj", "Meta"})


def _unquote(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and (
        (text[0] == '"' and text[-1] == '"')
        or (text[0] == "'" and text[-1] == "'")
        or (text[0] == "`" and text[-1] == "`")
    ):
        return text[1:-1]
    return text


def _unwrap_object_literal(node: Node | None) -> Node | None:
    current = node
    while current is not None:
        if current.type == "object":
            return current
        if current.type in {"as_expression", "satisfies_expression", "type_assertion"}:
            inner = current.child_by_field_name("expression")
            if inner is None:
                named = [c for c in current.children if c.is_named]
                inner = named[0] if named else None
            current = inner
            continue
        if current.type == "parenthesized_expression":
            named = [c for c in current.children if c.is_named]
            current = named[0] if named else None
            continue
        break
    return None


def _extract_meta_type_component(node: Node | None, code_bytes: bytes) -> str:
    """Look for Meta<typeof Component> inside a type annotation / as / satisfies node."""
    if node is None:
        return ""
    if node.type == "type_query":
        named = [c for c in node.children if c.is_named]
        if named:
            return _node_text(named[-1], code_bytes)
    for child in node.children:
        found = _extract_meta_type_component(child, code_bytes)
        if found:
            return found
    return ""


class StorybookPlugin:
    """Language plugin for Storybook stories (.stories.tsx/.ts/.jsx/.js and .mdx)."""

    def __init__(self) -> None:
        self._ts_parser = Parser(_TS_LANGUAGE)
        self._tsx_parser = Parser(_TSX_LANGUAGE)
        self.last_parse_diagnostics: ParseDiagnostics | None = None

    @property
    def language_id(self) -> str:
        return "storybook"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".stories.tsx", ".stories.ts", ".stories.jsx", ".stories.js", ".mdx")

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ()

    def can_handle(self, file_path: str) -> bool:
        lower = file_path.lower()
        return any(lower.endswith(ext) for ext in self.file_extensions)

    def _derive_module_prefix(self, file_path: str) -> str:
        if not file_path:
            return "storybook"
        stem = re.sub(
            r"(?:\.stories\.(?:tsx|ts|jsx|js)|\.mdx)$",
            "",
            file_path,
            flags=re.IGNORECASE,
        )
        normalized = stem.replace("\\", "/").strip("/")
        dotted = ".".join(part for part in normalized.split("/") if part)
        return dotted or "storybook"

    def extract_manifest_dependencies(
        self,
        file_path: str,
        content: str,
    ) -> list[DependencyEdge]:
        return []

    def extract_ast(
        self,
        code: str,
        file_path: str = "",
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Extract Storybook Meta and story symbols using tree-sitter-typescript (or MDX)."""
        if not code or not code.strip():
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-typescript")
            return [], []

        prefix = module_prefix or self._derive_module_prefix(file_path)

        if file_path.lower().endswith(".mdx"):
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="mdx_heuristic_fallback",
                has_parse_error=False,
                abstained=False,
                low_confidence_fallback=True,
                reason="MDX uses heuristic extraction (no tree-sitter-mdx grammar)",
            )
            return self._extract_mdx(code, prefix, repo_name)

        lower = file_path.lower()
        is_ts_only = lower.endswith((".stories.ts", ".stories.js"))
        parser = self._ts_parser if is_ts_only else self._tsx_parser
        code_bytes = code.encode("utf-8")
        tree = parser.parse(code_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            logger.warning(
                "StorybookPlugin parse error in %s (errors=%d, missing=%d); abstaining.",
                file_path or "<memory>",
                err_cnt,
                miss_cnt,
            )
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="tree-sitter-typescript",
                has_parse_error=True,
                abstained=True,
                error_node_count=err_cnt,
                missing_node_count=miss_cnt,
                reason="Syntax/parse error detected by tree-sitter-typescript",
            )
            return [], []

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-typescript")
        imported_components: list[str] = []
        meta_title = ""
        component_name = ""
        meta_node: Node | None = None
        var_objects: dict[str, tuple[Node, str]] = {}
        exported_stories: list[tuple[str, Node, str]] = []

        for child in tree.root_node.children:
            if child.type == "import_statement":
                stmt_text = _node_text(child, code_bytes)
                if re.match(r"^\s*import\s+type\b", stmt_text):
                    continue
                src_node = child.child_by_field_name("source")
                src_str = _unquote(_node_text(src_node, code_bytes)) if src_node is not None else ""
                if "@storybook" in src_str or src_str == "react":
                    continue
                for clause in child.children:
                    if clause.type == "import_clause":
                        for cnode in clause.children:
                            if cnode.type == "identifier":
                                cname = _node_text(cnode, code_bytes)
                                if cname not in _SKIPPED_IMPORT_NAMES:
                                    imported_components.append(cname)
                            elif cnode.type == "named_imports":
                                for spec in cnode.children:
                                    if spec.type == "import_specifier":
                                        alias = spec.child_by_field_name("alias")
                                        name_nd = spec.child_by_field_name("name")
                                        picked = alias or name_nd
                                        if picked is not None:
                                            cname = _node_text(picked, code_bytes)
                                            if cname not in _SKIPPED_IMPORT_NAMES:
                                                imported_components.append(cname)

            elif child.type in {"lexical_declaration", "variable_declaration"}:
                for decl in child.children:
                    if decl.type != "variable_declarator":
                        continue
                    name_nd = decl.child_by_field_name("name")
                    val_nd = decl.child_by_field_name("value")
                    if name_nd is None or name_nd.type != "identifier":
                        continue
                    vname = _node_text(name_nd, code_bytes)
                    obj = _unwrap_object_literal(val_nd)
                    type_hint = _extract_meta_type_component(decl, code_bytes)
                    if obj is not None:
                        var_objects[vname] = (obj, type_hint)

            elif child.type == "export_statement":
                export_text = _node_text(child, code_bytes).lstrip()
                is_default = export_text.startswith("export default")
                if is_default:
                    val_nd = child.child_by_field_name("value")
                    if val_nd is None:
                        val_nd = next(
                            (
                                c
                                for c in child.children
                                if c.is_named and c.type not in {"decorator"}
                            ),
                            None,
                        )
                    obj = _unwrap_object_literal(val_nd)
                    if obj is not None:
                        meta_node = obj
                        component_name = _extract_meta_type_component(val_nd, code_bytes)
                    elif val_nd is not None and val_nd.type == "identifier":
                        ident = _node_text(val_nd, code_bytes)
                        if ident in var_objects:
                            meta_node, hint = var_objects[ident]
                            component_name = hint
                else:
                    for sub in child.children:
                        if sub.type in {"lexical_declaration", "variable_declaration"}:
                            for decl in sub.children:
                                if decl.type != "variable_declarator":
                                    continue
                                name_nd = decl.child_by_field_name("name")
                                val_nd = decl.child_by_field_name("value")
                                if name_nd is None or name_nd.type != "identifier":
                                    continue
                                sname = _node_text(name_nd, code_bytes)
                                if sname.lower() == "meta" or sname == "default":
                                    obj = _unwrap_object_literal(val_nd)
                                    if obj is not None:
                                        meta_node = obj
                                        if not component_name:
                                            component_name = _extract_meta_type_component(
                                                decl, code_bytes
                                            )
                                else:
                                    sig = " ".join(_node_text(decl, code_bytes).split())[:120]
                                    exported_stories.append((sname, decl, sig))

        if meta_node is None and "meta" in var_objects:
            meta_node, hint = var_objects["meta"]
            if not component_name:
                component_name = hint

        if meta_node is not None:
            for item in meta_node.children:
                if item.type == "pair":
                    knd = item.child_by_field_name("key")
                    vnd = item.child_by_field_name("value")
                    if knd is None or vnd is None:
                        continue
                    ktext = _unquote(_node_text(knd, code_bytes))
                    if ktext == "title" and vnd.type in {"string", "template_string"}:
                        meta_title = _unquote(_node_text(vnd, code_bytes))
                    elif ktext == "component" and vnd.type in {"identifier", "member_expression"}:
                        component_name = _node_text(vnd, code_bytes)
                elif item.type == "shorthand_property_identifier":
                    if _node_text(item, code_bytes) == "component" and imported_components:
                        component_name = imported_components[0]

        if not component_name and imported_components:
            component_name = imported_components[0]

        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        seen_edges: set[tuple[str, str, str]] = set()

        def add_doc_edge(src: str, tgt: str) -> None:
            if not src or not tgt:
                return
            key = (src, tgt, "DOCUMENTS_COMPONENT")
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append(
                    ParsedEdge(
                        source_qualified_name=src,
                        target_qualified_name=tgt,
                        edge_kind="DOCUMENTS_COMPONENT",
                    )
                )

        if meta_node is not None or meta_title:
            anchor = meta_node if meta_node is not None else tree.root_node
            meta_qn = f"{prefix}.Meta"
            meta_sig = f'Meta(title="{meta_title}", component="{component_name}")'
            symbols.append(
                ParsedSymbol(
                    name="Meta",
                    qualified_name=meta_qn,
                    bare_name="Meta",
                    symbol_kind="story",
                    start_line=anchor.start_point[0] + 1,
                    end_line=anchor.end_point[0] + 1,
                    start_byte=anchor.start_byte,
                    end_byte=anchor.end_byte,
                    signature=meta_sig,
                    signature_skeleton=meta_sig,
                    repo_name=repo_name,
                )
            )
            if component_name:
                add_doc_edge(meta_qn, component_name)

        for story_name, story_node, sig in exported_stories:
            story_qn = f"{prefix}.{story_name}"
            symbols.append(
                ParsedSymbol(
                    name=story_name,
                    qualified_name=story_qn,
                    bare_name=story_name,
                    symbol_kind="story",
                    start_line=story_node.start_point[0] + 1,
                    end_line=story_node.end_point[0] + 1,
                    start_byte=story_node.start_byte,
                    end_byte=story_node.end_byte,
                    signature=sig,
                    signature_skeleton=sig,
                    repo_name=repo_name,
                )
            )
            if component_name:
                add_doc_edge(story_qn, component_name)

        return symbols, edges

    def _extract_mdx(
        self,
        code: str,
        module_prefix: str,
        repo_name: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        lines = code.splitlines(keepends=True)
        line_offsets: list[int] = []
        offset = 0
        for line in lines:
            line_offsets.append(offset)
            offset += len(line.encode("utf-8"))

        meta_match = re.search(
            r'<Meta\b[^>]*?title=["\']([^"\']+)["\'][^>]*?/?>',
            code,
        )
        of_match = re.search(
            r"<Meta\b[^>]*?(?:of|component)=\{([A-Za-z_]\w*)\}[^>]*?/?>",
            code,
        )

        comp_name = ""
        if of_match:
            comp_name = of_match.group(1)

        if meta_match:
            title = meta_match.group(1)
            start_line = code[: meta_match.start()].count("\n") + 1
            start_byte = len(code[: meta_match.start()].encode("utf-8"))
            end_byte = len(code[: meta_match.end()].encode("utf-8"))
            meta_qn = f"{module_prefix}.Meta"
            symbols.append(
                ParsedSymbol(
                    name="Meta",
                    qualified_name=meta_qn,
                    bare_name="Meta",
                    symbol_kind="story",
                    start_line=start_line,
                    end_line=start_line,
                    start_byte=start_byte,
                    end_byte=end_byte,
                    signature=f'<Meta title="{title}" />',
                    signature_skeleton=f'<Meta title="{title}" />',
                    repo_name=repo_name,
                )
            )
            if comp_name:
                edges.append(
                    ParsedEdge(
                        source_qualified_name=meta_qn,
                        target_qualified_name=comp_name,
                        edge_kind="DOCUMENTS_COMPONENT",
                    )
                )

        for smatch in re.finditer(r'<Story\b[^>]*?name=["\']([^"\']+)["\']', code):
            sname = smatch.group(1)
            start_line = code[: smatch.start()].count("\n") + 1
            start_byte = len(code[: smatch.start()].encode("utf-8"))
            end_byte = len(code[: smatch.end()].encode("utf-8"))
            sqn = f"{module_prefix}.{sname}"
            symbols.append(
                ParsedSymbol(
                    name=sname,
                    qualified_name=sqn,
                    bare_name=sname,
                    symbol_kind="story",
                    start_line=start_line,
                    end_line=start_line,
                    start_byte=start_byte,
                    end_byte=end_byte,
                    signature=f'<Story name="{sname}">',
                    signature_skeleton=f'<Story name="{sname}">',
                    repo_name=repo_name,
                )
            )
            if comp_name:
                edges.append(
                    ParsedEdge(
                        source_qualified_name=sqn,
                        target_qualified_name=comp_name,
                        edge_kind="DOCUMENTS_COMPONENT",
                    )
                )

        return symbols, edges
