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

_REACT_EMAIL_TAGS = frozenset({"Html", "Body", "Container", "Section", "Button", "Link", "Preview"})


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


def _unquote_ts(text: str) -> str:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"', "`"}:
        return stripped[1:-1]
    return stripped


def _extract_ts_url_expr(node: Node | None, code_bytes: bytes) -> str:
    """Extract a URL string or template literal path from a TS AST expression node."""
    if node is None:
        return ""
    if node.type in {"string", "template_string"}:
        raw = _unquote_ts(_node_text(node, code_bytes))
        return re.sub(r"\$\{[^}]+\}", "{param}", raw)
    if node.type == "binary_expression":
        left = _extract_ts_url_expr(node.child_by_field_name("left"), code_bytes)
        right = _extract_ts_url_expr(node.child_by_field_name("right"), code_bytes)
        if left or right:
            return f"{left}{right}"
    if node.type == "parenthesized_expression":
        for c in node.children:
            if c.is_named:
                return _extract_ts_url_expr(c, code_bytes)
    return ""


def _join_url_paths(base_url: str, sub_path: str) -> str:
    base = re.sub(r"^https?://[^/]+", "", base_url.strip())
    sub = re.sub(r"^https?://[^/]+", "", sub_path.strip())
    if sub.startswith("/") and base and not sub.startswith(base.rstrip("/")):
        return f"/{base.strip('/')}/{sub.lstrip('/')}" if base.strip("/") else sub
    if base and sub:
        return f"/{base.strip('/')}/{sub.lstrip('/')}"
    return "/" + (sub or base).lstrip("/") if (sub or base) else ""


def _extract_ts_dto_fields(node: Node, code_bytes: bytes) -> list[str]:
    """Extract property_signature names from an interface_body or object_type."""
    fields: list[str] = []

    def walk_body(cur: Node) -> None:
        for child in cur.children:
            if child.type == "property_signature":
                name_nd = child.child_by_field_name("name")
                if name_nd is not None:
                    fname = _unquote_ts(_node_text(name_nd, code_bytes))
                    if fname:
                        fields.append(fname)
            elif child.type in {"interface_body", "object_type"}:
                walk_body(child)

    walk_body(node)
    return list(dict.fromkeys(fields))


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

        return self._extract_ast_impl(
            code=code,
            file_path=file_path,
            repo_name=repo_name,
            module_prefix=module_prefix,
        )

    def _extract_ast_impl(
        self,
        code: str,
        file_path: str = "",
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
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
        seen_symbols: dict[str, int] = {}
        scope_stack: list[tuple[str, str]] = []
        react_email_sources: dict[str, str] = {}
        has_react_email_import = False

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

        def attach_note(qn: str, note: str) -> None:
            if not qn or not note:
                return
            idx = seen_symbols.get(qn)
            if idx is None:
                return
            prev = symbols[idx]
            if note in prev.docstring:
                return
            merged = f"{prev.docstring} {note}".strip()
            symbols[idx] = prev.model_copy(
                update={"docstring": merged, "contextual_summary": merged}
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
            seen_symbols[qn] = len(symbols)
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

        def _extract_rtk_query_spec(ep_call: Node, base_url: str) -> tuple[str, str, list[str]]:
            """Return (http_method, full_url, referenced_dto_types) from builder.query/mutation."""
            fn = ep_call.child_by_field_name("function")
            leaf = _callee_leaf(fn, code_bytes)
            http_method = "POST" if leaf == "mutation" else "GET"
            ref_types: list[str] = []
            type_args = ep_call.child_by_field_name("type_arguments")
            if type_args is None:
                type_args = next((c for c in ep_call.children if c.type == "type_arguments"), None)
            if type_args is not None:
                for tc in type_args.children:
                    if tc.type == "type_identifier":
                        ref_types.append(_node_text(tc, code_bytes))
                    elif tc.type == "generic_type":
                        inner_targs = next(
                            (c for c in tc.children if c.type == "type_arguments"), None
                        )
                        if inner_targs is not None:
                            for sub_t in inner_targs.children:
                                if sub_t.type == "type_identifier":
                                    ref_types.append(_node_text(sub_t, code_bytes))
            url_sub = ""
            ep_args = ep_call.child_by_field_name("arguments")
            ep_cfg = (
                next((c for c in ep_args.children if c.type == "object"), None)
                if ep_args is not None
                else None
            )
            if ep_cfg is not None:
                q_node = _find_object_property_value(ep_cfg, "query", code_bytes)
                if q_node is not None:
                    q_body = (
                        q_node.child_by_field_name("body")
                        if q_node.type == "arrow_function"
                        else q_node
                    )
                    if q_body is not None and q_body.type == "parenthesized_expression":
                        q_body = next((c for c in q_body.children if c.is_named), q_body)
                    if q_body is not None and q_body.type == "object":
                        url_val = _find_object_property_value(q_body, "url", code_bytes)
                        meth_val = _find_object_property_value(q_body, "method", code_bytes)
                        url_sub = _extract_ts_url_expr(url_val, code_bytes)
                        if meth_val is not None:
                            m_txt = _unquote_ts(_node_text(meth_val, code_bytes)).upper()
                            if m_txt in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
                                http_method = m_txt
                    elif q_body is not None and q_body.type == "statement_block":
                        ret_nd = next(
                            (c for c in q_body.children if c.type == "return_statement"), None
                        )
                        if ret_nd is not None:
                            ret_expr = next((c for c in ret_nd.children if c.is_named), None)
                            if ret_expr is not None and ret_expr.type == "object":
                                url_val = _find_object_property_value(ret_expr, "url", code_bytes)
                                meth_val = _find_object_property_value(
                                    ret_expr, "method", code_bytes
                                )
                                url_sub = _extract_ts_url_expr(url_val, code_bytes)
                                if meth_val is not None:
                                    m_txt = _unquote_ts(_node_text(meth_val, code_bytes)).upper()
                                    if m_txt in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
                                        http_method = m_txt
                            else:
                                url_sub = _extract_ts_url_expr(ret_expr, code_bytes)
                    else:
                        url_sub = _extract_ts_url_expr(q_body, code_bytes)
            full_url = _join_url_paths(base_url, url_sub) if (base_url or url_sub) else ""
            return http_method, full_url, list(dict.fromkeys(ref_types))

        def record_create_api(api_name: str, declarator: Node, call_node: Node) -> None:
            api_qn = f"{prefix}.{api_name}"
            doc = _extract_leading_doc_comment(declarator, code_bytes)
            add_symbol(
                name=api_name,
                qn=api_qn,
                kind="slice",
                node=declarator,
                sig=f"createApi({api_name})",
                doc=doc,
            )
            args = call_node.child_by_field_name("arguments")
            if args is None:
                return
            cfg_obj = next((c for c in args.children if c.type == "object"), None)
            if cfg_obj is None:
                return
            base_url = ""
            bq_val = _find_object_property_value(cfg_obj, "baseQuery", code_bytes)
            if bq_val is not None and bq_val.type == "call_expression":
                bq_args = bq_val.child_by_field_name("arguments")
                bq_obj = (
                    next((c for c in bq_args.children if c.type == "object"), None)
                    if bq_args is not None
                    else None
                )
                if bq_obj is not None:
                    bu_nd = _find_object_property_value(bq_obj, "baseUrl", code_bytes)
                    base_url = _extract_ts_url_expr(bu_nd, code_bytes)

            ep_val = _find_object_property_value(cfg_obj, "endpoints", code_bytes)
            if ep_val is None:
                return
            ep_body = (
                ep_val.child_by_field_name("body") if ep_val.type == "arrow_function" else ep_val
            )
            if ep_body is not None and ep_body.type == "parenthesized_expression":
                ep_body = next((c for c in ep_body.children if c.is_named), ep_body)
            if ep_body is None or ep_body.type != "object":
                return
            for child in ep_body.children:
                if child.type != "pair":
                    continue
                k_nd = child.child_by_field_name("key")
                v_nd = child.child_by_field_name("value")
                if k_nd is None or v_nd is None or v_nd.type != "call_expression":
                    continue
                ep_name = _unquote_ts(_node_text(k_nd, code_bytes))
                if not ep_name:
                    continue
                fn_leaf = _callee_leaf(v_nd.child_by_field_name("function"), code_bytes)
                if fn_leaf not in {"query", "mutation"}:
                    continue
                method, full_url, ref_types = _extract_rtk_query_spec(v_nd, base_url)
                ep_qn = f"{api_qn}.{ep_name}"
                notes: list[str] = []
                if full_url:
                    notes.append(f"@route {method} {full_url}")
                for rt in ref_types:
                    notes.append(f"@dto_ref {rt}")
                    add_edge(ep_qn, rt, "USES_TYPE")
                add_symbol(
                    name=ep_name,
                    qn=ep_qn,
                    kind="function",
                    node=child,
                    sig=f"{ep_name}: builder.{fn_leaf}({method} {full_url})"
                    if full_url
                    else f"{ep_name}: builder.{fn_leaf}",
                    bare_name=ep_name,
                    doc=" ".join(notes),
                )

        def handle_call_expression(node: Node) -> bool:
            """Process a call_expression; returns True if children were already visited."""
            func_node = node.child_by_field_name("function")
            callee = _callee_text(func_node, code_bytes)
            callee_leaf = _callee_leaf(func_node, code_bytes)
            args = node.child_by_field_name("arguments")
            caller_qn = scope_stack[-1][1] if scope_stack else f"{prefix}.standalone"
            caller_name = scope_stack[-1][0] if scope_stack else ""

            if callee_leaf == "fetch" and scope_stack and args is not None:
                named_args = [c for c in args.children if c.is_named]
                if named_args:
                    url = _extract_ts_url_expr(named_args[0], code_bytes)
                    method = "GET"
                    if len(named_args) > 1 and named_args[1].type == "object":
                        m_nd = _find_object_property_value(named_args[1], "method", code_bytes)
                        if m_nd is not None:
                            m_txt = _unquote_ts(_node_text(m_nd, code_bytes)).upper()
                            if m_txt in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
                                method = m_txt
                    if url:
                        attach_note(caller_qn, f"@route {method} {url}")

            elif callee_leaf == "EventSource" and scope_stack and args is not None:
                named_args = [c for c in args.children if c.is_named]
                if named_args:
                    url = _extract_ts_url_expr(named_args[0], code_bytes)
                    if url:
                        attach_note(caller_qn, f"@route GET {url}")

            elif callee.startswith("axios") and scope_stack and args is not None:
                named_args = [c for c in args.children if c.is_named]
                axios_method = callee_leaf.upper() if callee != "axios" else "GET"
                if axios_method in {"GET", "POST", "PUT", "DELETE", "PATCH"} and named_args:
                    url = _extract_ts_url_expr(named_args[0], code_bytes)
                    if not url and named_args[0].type == "object":
                        u_nd = _find_object_property_value(named_args[0], "url", code_bytes)
                        m_nd = _find_object_property_value(named_args[0], "method", code_bytes)
                        url = _extract_ts_url_expr(u_nd, code_bytes)
                        if m_nd is not None:
                            axios_method = _unquote_ts(_node_text(m_nd, code_bytes)).upper()
                    if url:
                        attach_note(caller_qn, f"@route {axios_method} {url}")

            elif callee_leaf in {"useQuery", "useMutation"} and scope_stack and args is not None:
                named_args = [c for c in args.children if c.is_named]
                if named_args and named_args[0].type == "identifier":
                    q_ident = _node_text(named_args[0], code_bytes)
                    if q_ident:
                        add_edge(caller_qn, q_ident, "CALLS")

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

            elif callee_leaf == "t" and scope_stack and args is not None:
                first_arg = next((c for c in args.children if c.is_named), None)
                if first_arg is not None and first_arg.type == "string":
                    i18n_key = _unquote_ts(_node_text(first_arg, code_bytes))
                    if i18n_key:
                        attach_note(caller_qn, f"@i18n_key {i18n_key}")
                        add_edge(caller_qn, i18n_key, "USES_TYPE")

            elif callee_leaf == "formatMessage" and scope_stack and args is not None:
                first_arg = next((c for c in args.children if c.is_named), None)
                if first_arg is not None and first_arg.type == "object":
                    id_nd = _find_object_property_value(first_arg, "id", code_bytes)
                    if id_nd is not None and id_nd.type == "string":
                        i18n_key = _unquote_ts(_node_text(id_nd, code_bytes))
                        if i18n_key:
                            attach_note(caller_qn, f"@i18n_key {i18n_key}")
                            add_edge(caller_qn, i18n_key, "USES_TYPE")

            elif (
                _RE_HOOK_NAME.match(callee_leaf)
                and callee_leaf not in _IGNORED_HOOKS
                and callee_leaf != caller_name
                and scope_stack
            ):
                if callee_leaf == "useTranslation" and args is not None:
                    first_arg = next((c for c in args.children if c.is_named), None)
                    if first_arg is not None and first_arg.type == "string":
                        ns = _unquote_ts(_node_text(first_arg, code_bytes))
                        if ns:
                            attach_note(caller_qn, f"@i18n_ns {ns}")
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
                if callee_leaf == "createApi":
                    record_create_api(var_name, node, value_node)
                    return
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
            elif value_node.type == "call_expression":
                callee_leaf = _callee_leaf(value_node.child_by_field_name("function"), code_bytes)
                if callee_leaf in {"gql", "graphql"}:
                    gql_raw = _node_text(value_node, code_bytes)
                    m_gql = re.search(r"\b(query|mutation)\s+([A-Za-z_]\w*)", gql_raw)
                    if m_gql:
                        qn = f"{prefix}.{var_name}"
                        add_symbol(
                            name=var_name,
                            qn=qn,
                            kind="type_alias",
                            node=node,
                            sig=f"{var_name} = {m_gql.group(1)} {m_gql.group(2)}",
                            doc=f"@graphql {m_gql.group(1)} {m_gql.group(2)}",
                        )
            visit(value_node)

        def handle_interface_or_type(node: Node) -> None:
            name_nd = node.child_by_field_name("name")
            if name_nd is None:
                return
            tname = _node_text(name_nd, code_bytes)
            if not tname:
                return
            kind = "interface" if node.type == "interface_declaration" else "type_alias"
            qn = f"{prefix}.{tname}"
            fields = _extract_ts_dto_fields(node, code_bytes)
            doc = _extract_leading_doc_comment(node, code_bytes)
            if fields:
                doc = f"{doc} @dto {tname} fields={','.join(fields)}".strip()
            sig = _collapse_ws(_node_text(node, code_bytes))[:200]
            add_symbol(name=tname, qn=qn, kind=kind, node=node, sig=sig, doc=doc)

        def handle_import_statement(node: Node) -> None:
            nonlocal has_react_email_import
            src_node = node.child_by_field_name("source")
            if src_node is None:
                return
            src_str = _unquote_ts(_node_text(src_node, code_bytes))
            if not (
                src_str == "react-email"
                or src_str == "@react-email/components"
                or src_str.startswith("@react-email/")
            ):
                return
            has_react_email_import = True
            for clause in node.children:
                if clause.type == "import_clause":
                    for cnode in clause.children:
                        if cnode.type == "identifier":
                            cname = _node_text(cnode, code_bytes)
                            react_email_sources[cname] = src_str
                        elif cnode.type == "named_imports":
                            for spec in cnode.children:
                                if spec.type == "import_specifier":
                                    alias = spec.child_by_field_name("alias")
                                    name_nd = spec.child_by_field_name("name")
                                    picked = alias or name_nd
                                    if picked is not None:
                                        react_email_sources[_node_text(picked, code_bytes)] = (
                                            src_str
                                        )

        def handle_jsx_tag(node: Node) -> None:
            tag_nd = node.child_by_field_name("name")
            if tag_nd is None:
                tag_nd = next(
                    (c for c in node.children if c.type in {"identifier", "member_expression"}),
                    None,
                )
            if tag_nd is None:
                return
            tag_name = _node_text(tag_nd, code_bytes)
            caller_qn = scope_stack[-1][1] if scope_stack else prefix

            if tag_name == "Trans" and scope_stack:
                for attr in node.children:
                    if attr.type != "jsx_attribute":
                        continue
                    attr_name_nd = next(
                        (c for c in attr.children if c.type == "property_identifier"), None
                    )
                    if attr_name_nd is None or _node_text(attr_name_nd, code_bytes) != "i18nKey":
                        continue
                    val_nd = next(
                        (c for c in attr.children if c.type in {"string", "jsx_expression"}),
                        None,
                    )
                    if val_nd is not None:
                        if val_nd.type == "jsx_expression":
                            str_nd = next((c for c in val_nd.children if c.type == "string"), None)
                            val_nd = str_nd if str_nd is not None else val_nd
                        if val_nd.type == "string":
                            i18n_key = _unquote_ts(_node_text(val_nd, code_bytes))
                            if i18n_key:
                                attach_note(caller_qn, f"@i18n_key {i18n_key}")
                                add_edge(caller_qn, i18n_key, "USES_TYPE")

            if has_react_email_import and tag_name in _REACT_EMAIL_TAGS:
                email_qn = f"{caller_qn}.email.{tag_name}"
                add_symbol(
                    name=tag_name,
                    qn=email_qn,
                    kind="component",
                    node=node,
                    sig=f"<{tag_name}>",
                    bare_name=tag_name,
                )
                add_edge(caller_qn, tag_name, "USES_TYPE")
                email_pkg = react_email_sources.get(tag_name, "@react-email/components")
                add_edge(caller_qn, email_pkg, "IMPORTS")

        def visit(node: Node) -> None:
            if node.type == "comment":
                return
            if node.type == "import_statement":
                handle_import_statement(node)
                return
            if node.type in {"jsx_opening_element", "jsx_self_closing_element"}:
                handle_jsx_tag(node)
            if node.type in {"interface_declaration", "type_alias_declaration"}:
                handle_interface_or_type(node)
                return
            if node.type == "template_string" and scope_stack:
                parent = node.parent
                if parent is not None and parent.type == "call_expression":
                    fn_txt = _callee_leaf(parent.child_by_field_name("function"), code_bytes)
                    if fn_txt in {"gql", "graphql"}:
                        m_gql = re.search(
                            r"\b(query|mutation)\s+([A-Za-z_]\w*)",
                            _node_text(node, code_bytes),
                        )
                        if m_gql:
                            attach_note(
                                scope_stack[-1][1],
                                f"@graphql {m_gql.group(1)} {m_gql.group(2)}",
                            )
            if node.type == "member_expression" and scope_stack:
                expr_txt = _node_text(node, code_bytes)
                m_env = re.match(r"^(?:process\.env|import\.meta\.env)\.([A-Z0-9_]+)$", expr_txt)
                if m_env:
                    attach_note(scope_stack[-1][1], f"@env {m_env.group(1)}")
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
