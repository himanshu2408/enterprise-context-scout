"""React / Redux Toolkit language plugin using tree-sitter-typescript AST traversal."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

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

_WRAPPER_CALLEES = frozenset({"memo", "React.memo", "forwardRef", "React.forwardRef"})

_IGNORED_HOOKS = frozenset(
    {
        "useState",
        "useEffect",
        "useCallback",
        "useMemo",
        "useRef",
        "useContext",
        "useReducer",
        "useLayoutEffect",
        "useImperativeHandle",
        "useDebugValue",
        "useId",
        "useTransition",
        "useDeferredValue",
        "useSyncExternalStore",
        "useAppDispatch",
        "useAppSelector",
        "useSelector",
        "useDispatch",
    }
)

_RE_COMPONENT_NAME = re.compile(r"^[A-Z][A-Za-z0-9_]*$")
_RE_HOOK_NAME = re.compile(r"^use[A-Z][A-Za-z0-9_]*$")
_RE_SELECTOR_NAME = re.compile(r"^select[A-Z][A-Za-z0-9_]*$")


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def _callee_text(func_node: Node | None, code_bytes: bytes) -> str:
    if func_node is None:
        return ""
    if func_node.type in {"identifier", "member_expression"}:
        return _node_text(func_node, code_bytes)
    return ""


def _callee_leaf(func_node: Node | None, code_bytes: bytes) -> str:
    if func_node is None:
        return ""
    if func_node.type == "identifier":
        return _node_text(func_node, code_bytes)
    if func_node.type == "member_expression":
        prop = func_node.child_by_field_name("property")
        if prop is not None:
            return _node_text(prop, code_bytes)
    return _node_text(func_node, code_bytes)


def _unwrap_react_wrappers(expr: Node | None, code_bytes: bytes) -> Node | None:
    """Peel nested memo/React.memo/forwardRef/React.forwardRef call wrappers."""
    current = expr
    while current is not None:
        if current.type == "parenthesized_expression":
            named = [c for c in current.children if c.is_named]
            current = named[0] if named else None
            continue
        if current.type in {"as_expression", "type_assertion", "satisfies_expression"}:
            inner = current.child_by_field_name("expression")
            if inner is None:
                named = [c for c in current.children if c.is_named]
                inner = named[0] if named else None
            current = inner
            continue
        if current.type == "call_expression":
            func_node = current.child_by_field_name("function")
            callee = _callee_text(func_node, code_bytes)
            if callee in _WRAPPER_CALLEES:
                args = current.child_by_field_name("arguments")
                if args is not None:
                    first_arg = next((c for c in args.children if c.is_named), None)
                    if first_arg is not None:
                        current = first_arg
                        continue
        break
    return current


def _is_function_like(node: Node | None) -> bool:
    return node is not None and node.type in {
        "arrow_function",
        "function_expression",
        "function_declaration",
    }


def _find_object_property_value(obj_node: Node, key_name: str, code_bytes: bytes) -> Node | None:
    for child in obj_node.children:
        if child.type == "pair":
            key = child.child_by_field_name("key")
            val = child.child_by_field_name("value")
            if key is not None and _node_text(key, code_bytes).strip("\"'") == key_name:
                return val
    return None


def _extract_leading_doc_comment(node: Node, code_bytes: bytes) -> str:
    target = node
    if target.parent is not None and target.parent.type in {
        "lexical_declaration",
        "variable_declaration",
    }:
        target = target.parent
    if target.parent is not None and target.parent.type == "export_statement":
        target = target.parent
    comments: list[str] = []
    sibling = target.prev_sibling
    while sibling is not None and sibling.type == "comment":
        text = _node_text(sibling, code_bytes).strip()
        if text.startswith("//"):
            comments.append(text[2:].strip())
        elif text.startswith("/*"):
            cleaned = text.strip("/* \t\n").replace("*", "").strip()
            if cleaned:
                comments.append(cleaned)
        sibling = sibling.prev_sibling
    if not comments:
        return ""
    comments.reverse()
    return "\n".join(line for line in comments if line)


class ReactReduxPlugin:
    """Language plugin for React (.tsx/.jsx) and Redux Toolkit (.ts/.js)."""

    def __init__(self) -> None:
        self._ts_parser = Parser(_TS_LANGUAGE)
        self._tsx_parser = Parser(_TSX_LANGUAGE)
        self.last_parse_diagnostics: ParseDiagnostics | None = None

    @property
    def language_id(self) -> str:
        return "react_redux"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".tsx", ".jsx", ".ts", ".js")

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("package.json",)

    def can_handle(self, file_path: str) -> bool:
        lower = file_path.lower()
        return any(lower.endswith(ext) for ext in self.file_extensions)

    def _derive_module_prefix(self, file_path: str) -> str:
        if not file_path:
            return "react_module"
        stem = re.sub(r"\.(?:tsx|jsx|ts|js)$", "", file_path, flags=re.IGNORECASE)
        normalized = stem.replace("\\", "/").strip("/")
        dotted = ".".join(part for part in normalized.split("/") if part)
        return dotted or "react_module"

    def extract_manifest_dependencies(
        self,
        file_path: str,
        content: str,
    ) -> list[DependencyEdge]:
        """Extract npm dependencies from package.json with explicit error logging."""
        if Path(file_path).name.lower() != "package.json":
            return []
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Failed to parse package.json manifest %s: %s", file_path, exc)
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="json",
                has_parse_error=True,
                abstained=True,
                reason=f"Manifest parse error: {exc}",
            )
            return []

        if not isinstance(data, dict):
            logger.warning("Unexpected non-object package.json manifest %s", file_path)
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="json",
                has_parse_error=True,
                abstained=True,
                reason="package.json root must be a JSON object",
            )
            return []

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="json")
        deps: list[DependencyEdge] = []
        scope_map = {
            "dependencies": "runtime",
            "devDependencies": "dev",
            "peerDependencies": "compile",
        }
        for section, scope in scope_map.items():
            block = data.get(section)
            if not isinstance(block, dict):
                continue
            for pkg, ver in block.items():
                if isinstance(pkg, str) and isinstance(ver, str):
                    deps.append(
                        DependencyEdge(
                            source_identifier=file_path,
                            target_package=pkg,
                            version_constraint=ver or "*",
                            scope=scope,
                            manifest_path=file_path,
                        )
                    )
        return deps

    def extract_ast(
        self,
        code: str,
        file_path: str = "",
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Extract React components, custom hooks, Redux slices/reducers/selectors, and edges."""
        if not code or not code.strip():
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-typescript")
            return [], []

        lower = file_path.lower()
        parser = self._ts_parser if lower.endswith((".ts", ".js")) else self._tsx_parser
        code_bytes = code.encode("utf-8")
        tree = parser.parse(code_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            logger.warning(
                "ReactReduxPlugin parse error in %s (errors=%d, missing=%d); abstaining.",
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
        prefix = module_prefix or self._derive_module_prefix(file_path)

        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        seen_edges: set[tuple[str, str, str]] = set()
        seen_symbols: set[str] = set()
        scope_stack: list[tuple[str, str]] = []

        def add_edge(src: str, tgt: str, kind: str) -> None:
            if not src or not tgt:
                return
            key = (src, tgt, kind)
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append(
                    ParsedEdge(
                        source_qualified_name=src,
                        target_qualified_name=tgt,
                        edge_kind=kind,
                    )
                )

        def add_symbol(
            name: str,
            qn: str,
            kind: str,
            node: Node,
            sig: str,
            bare_name: str | None = None,
            doc: str = "",
        ) -> None:
            if qn in seen_symbols:
                return
            seen_symbols.add(qn)
            symbols.append(
                ParsedSymbol(
                    name=name,
                    qualified_name=qn,
                    bare_name=bare_name or name,
                    symbol_kind=kind,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    signature=sig,
                    signature_skeleton=sig,
                    docstring=doc,
                    repo_name=repo_name,
                )
            )

        def record_slice_call(slice_name: str, declarator: Node, call_node: Node) -> None:
            slice_qn = f"{prefix}.{slice_name}"
            doc = _extract_leading_doc_comment(declarator, code_bytes)
            add_symbol(
                name=slice_name,
                qn=slice_qn,
                kind="slice",
                node=declarator,
                sig=f"createSlice({slice_name})",
                doc=doc,
            )
            args = call_node.child_by_field_name("arguments")
            if args is None:
                return
            config_obj = next((c for c in args.children if c.type == "object"), None)
            if config_obj is None:
                return
            reducers_val = _find_object_property_value(config_obj, "reducers", code_bytes)
            if reducers_val is None or reducers_val.type != "object":
                return
            for child in reducers_val.children:
                if child.type == "method_definition":
                    rname_node = child.child_by_field_name("name")
                    if rname_node is not None:
                        rname = _node_text(rname_node, code_bytes).strip("\"'")
                        if rname:
                            rqn = f"{slice_qn}.reducers.{rname}"
                            add_symbol(
                                name=rname,
                                qn=rqn,
                                kind="reducer",
                                node=child,
                                sig=_collapse_ws(_node_text(child, code_bytes).split("{", 1)[0]),
                                bare_name=f"{slice_name}.reducers.{rname}",
                            )
                            scope_stack.append((rname, rqn))
                            walk(child)
                            scope_stack.pop()
                elif child.type == "pair":
                    key_node = child.child_by_field_name("key")
                    val_node = child.child_by_field_name("value")
                    if key_node is not None:
                        rname = _node_text(key_node, code_bytes).strip("\"'")
                        if rname:
                            rqn = f"{slice_qn}.reducers.{rname}"
                            add_symbol(
                                name=rname,
                                qn=rqn,
                                kind="reducer",
                                node=child,
                                sig=f"{rname}: reducer",
                                bare_name=f"{slice_name}.reducers.{rname}",
                            )
                            if val_node is not None:
                                scope_stack.append((rname, rqn))
                                walk(val_node)
                                scope_stack.pop()

        def handle_call_expression(node: Node) -> bool:
            """Process a call_expression; returns True if children were already visited."""
            func_node = node.child_by_field_name("function")
            callee = _callee_text(func_node, code_bytes)
            callee_leaf = _callee_leaf(func_node, code_bytes)
            args = node.child_by_field_name("arguments")
            caller_qn = scope_stack[-1][1] if scope_stack else f"{prefix}.standalone"
            caller_name = scope_stack[-1][0] if scope_stack else ""

            if callee_leaf == "dispatch" and args is not None:
                first_arg = next((c for c in args.children if c.is_named), None)
                if first_arg is not None:
                    if first_arg.type == "call_expression":
                        action_fn = first_arg.child_by_field_name("function")
                        action_name = _callee_leaf(action_fn, code_bytes)
                        if action_name:
                            add_edge(caller_qn, action_name, "DISPATCHES")
                        action_args = first_arg.child_by_field_name("arguments")
                        if action_args is not None:
                            walk(action_args)
                        return True
                    if first_arg.type in {"identifier", "member_expression"}:
                        action_name = _callee_leaf(first_arg, code_bytes)
                        if action_name:
                            add_edge(caller_qn, action_name, "DISPATCHES")
                        return True

            elif (
                callee_leaf in {"useSelector", "useAppSelector"}
                and scope_stack
                and args is not None
            ):
                first_arg = next((c for c in args.children if c.is_named), None)
                if first_arg is not None and first_arg.type in {"identifier", "member_expression"}:
                    sel_name = _node_text(first_arg, code_bytes)
                    if sel_name:
                        add_edge(caller_qn, sel_name, "SELECTS")

            elif (
                _RE_HOOK_NAME.match(callee_leaf)
                and callee_leaf not in _IGNORED_HOOKS
                and callee_leaf != caller_name
                and scope_stack
            ):
                add_edge(caller_qn, callee_leaf, "CALLS")
            elif (
                scope_stack
                and callee_leaf
                and callee_leaf != caller_name
                and callee not in _WRAPPER_CALLEES
                and callee_leaf not in {"createSlice", "createSelector", "connect"}
            ):
                if callee_leaf[0].islower() and "." not in callee:
                    add_edge(caller_qn, callee_leaf, "CALLS")
            return False

        def handle_function_declaration(node: Node) -> None:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            name = _node_text(name_node, code_bytes)
            if not name:
                return
            if _RE_HOOK_NAME.match(name):
                kind = "hook"
            elif _RE_COMPONENT_NAME.match(name):
                kind = "class"
            elif _RE_SELECTOR_NAME.match(name):
                kind = "selector"
            else:
                if scope_stack:
                    body = node.child_by_field_name("body")
                    if body is not None:
                        walk(body)
                    return
                kind = "function"

            qn = f"{prefix}.{name}"
            sig = _collapse_ws(_node_text(node, code_bytes).split("{", 1)[0])
            doc = _extract_leading_doc_comment(node, code_bytes)
            add_symbol(name=name, qn=qn, kind=kind, node=node, sig=sig, doc=doc)
            scope_stack.append((name, qn))
            body = node.child_by_field_name("body")
            if body is not None:
                walk(body)
            scope_stack.pop()

        def handle_variable_declarator(node: Node) -> None:
            name_node = node.child_by_field_name("name")
            value_node = node.child_by_field_name("value")
            if name_node is None or name_node.type != "identifier" or value_node is None:
                if value_node is not None:
                    visit(value_node)
                return

            var_name = _node_text(name_node, code_bytes)
            if not var_name:
                return

            if value_node.type == "call_expression":
                callee_leaf = _callee_leaf(value_node.child_by_field_name("function"), code_bytes)
                if callee_leaf == "createSlice":
                    record_slice_call(var_name, node, value_node)
                    return
                if callee_leaf == "createSelector" or _RE_SELECTOR_NAME.match(var_name):
                    qn = f"{prefix}.{var_name}"
                    add_symbol(
                        name=var_name,
                        qn=qn,
                        kind="selector",
                        node=node,
                        sig=f"selector {var_name}",
                        doc=_extract_leading_doc_comment(node, code_bytes),
                    )
                    scope_stack.append((var_name, qn))
                    visit(value_node)
                    scope_stack.pop()
                    return

            unwrapped = _unwrap_react_wrappers(value_node, code_bytes)
            if _is_function_like(unwrapped):
                assert unwrapped is not None
                if _RE_HOOK_NAME.match(var_name):
                    kind = "hook"
                elif _RE_COMPONENT_NAME.match(var_name):
                    kind = "class"
                elif _RE_SELECTOR_NAME.match(var_name):
                    kind = "selector"
                else:
                    if scope_stack:
                        # Local helper inside a component or hook inherits parent scope
                        visit(unwrapped)
                        return
                    kind = "function"
                qn = f"{prefix}.{var_name}"
                sig = _collapse_ws(_node_text(node, code_bytes).split("=>", 1)[0])
                add_symbol(
                    name=var_name,
                    qn=qn,
                    kind=kind,
                    node=node,
                    sig=sig,
                    doc=_extract_leading_doc_comment(node, code_bytes),
                )
                scope_stack.append((var_name, qn))
                if unwrapped.type == "function_declaration":
                    body = unwrapped.child_by_field_name("body")
                    walk(body if body is not None else unwrapped)
                else:
                    body = unwrapped.child_by_field_name("body")
                    if body is not None:
                        visit(body)
                    else:
                        walk(unwrapped)
                scope_stack.pop()
                return

            if _RE_SELECTOR_NAME.match(var_name):
                qn = f"{prefix}.{var_name}"
                add_symbol(
                    name=var_name,
                    qn=qn,
                    kind="selector",
                    node=node,
                    sig=f"selector {var_name}",
                    doc=_extract_leading_doc_comment(node, code_bytes),
                )
            visit(value_node)

        def visit(node: Node) -> None:
            if node.type == "function_declaration":
                handle_function_declaration(node)
                return
            if node.type == "variable_declarator":
                handle_variable_declarator(node)
                return
            if node.type == "call_expression":
                handled = handle_call_expression(node)
                if not handled:
                    walk(node)
                return
            walk(node)

        def walk(node: Node) -> None:
            for child in node.children:
                visit(child)

        walk(tree.root_node)
        return symbols, edges
