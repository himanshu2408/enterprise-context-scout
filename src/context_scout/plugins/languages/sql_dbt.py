"""SQL (PostgreSQL 17, BigQuery) & dbt Jinja language plugin using tree-sitter-sql."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import jinja2
import jinja2.nodes
import yaml
from tree_sitter import Node, Parser
from tree_sitter_language_pack import get_parser

from context_scout.ast_engine.extractor import (
    ParseDiagnostics,
    _node_text,
    inspect_tree_errors,
)
from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol
from context_scout.plugins.base import DependencyEdge

logger = logging.getLogger(__name__)

_SQL_PARSER: Parser | None = None
_YAML_PARSER: Parser | None = None

_MANIFEST_BASENAMES: frozenset[str] = frozenset({"dbt_project.yml", "packages.yml", "schema.yml"})

_BUILTIN_SQL_TYPES: frozenset[str] = frozenset(
    {
        "INT",
        "INTEGER",
        "BIGINT",
        "SMALLINT",
        "INT64",
        "FLOAT",
        "FLOAT64",
        "NUMERIC",
        "DECIMAL",
        "REAL",
        "DOUBLE",
        "TEXT",
        "STRING",
        "VARCHAR",
        "CHAR",
        "BOOLEAN",
        "BOOL",
        "BYTES",
        "BYTEA",
        "DATE",
        "TIME",
        "TIMESTAMP",
        "TIMESTAMPTZ",
        "DATETIME",
        "JSON",
        "JSONB",
        "UUID",
        "VECTOR",
        "HALFVEC",
        "SPARSEVEC",
        "TSVECTOR",
        "ARRAY",
        "STRUCT",
        "GEOGRAPHY",
    }
)

# Regexes for Jinja masking (preserving newline counts)
_JINJA_COMMENT_RE = re.compile(r"\{#.*?#\}", re.DOTALL)
_JINJA_STMT_RE = re.compile(r"\{%-?.*?-?%\}", re.DOTALL)
_JINJA_EXPR_RE = re.compile(r"\{\{-?.*?-?\}\}", re.DOTALL)

# Regexes for PG17 / BigQuery normalization (preserving line counts)
_PG_VECTOR_TYPE_RE = re.compile(r"\b(vector|halfvec|sparsevec)\s*\(\s*\d+\s*\)", re.IGNORECASE)
_PG_USING_INDEX_RE = re.compile(r"\bUSING\s+(?:hnsw|ivfflat)\s*\(([^)]+)\)", re.IGNORECASE)
_PG_OPCLASS_RE = re.compile(
    r"\b(?:vector_cosine_ops|vector_l2_ops|vector_ip_ops|gin_trgm_ops)\b",
    re.IGNORECASE,
)
_PG_ALTER_RLS_RE = re.compile(
    r"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?([A-Za-z_][\w.\"]*)\s+"
    r"(?:ENABLE|DISABLE|FORCE|NO\s+FORCE)\s+ROW\s+LEVEL\s+SECURITY\s*;",
    re.IGNORECASE,
)
_PG_DROP_POLICY_RE = re.compile(
    r"DROP\s+POLICY\s+(?:IF\s+EXISTS\s+)?([A-Za-z_][\w\"]*)\s+ON\s+([A-Za-z_][\w.\"]*)\s*"
    r"(?:CASCADE|RESTRICT)?\s*;",
    re.IGNORECASE,
)
_PG_CREATE_POLICY_RE = re.compile(
    r"CREATE\s+POLICY\s+([A-Za-z_][\w\"]*)\s+ON\s+([A-Za-z_][\w.\"]*)\b[^;]*;",
    re.IGNORECASE | re.DOTALL,
)
_PG_DO_BLOCK_RE = re.compile(
    r"\bDO\s+\$\$(.*?)\$\$\s*(?:LANGUAGE\s+\w+\s*)?;",
    re.IGNORECASE | re.DOTALL,
)
_PG_GRANT_RE = re.compile(
    r"^\s*GRANT\s+[^;]+;",
    re.IGNORECASE | re.MULTILINE,
)
_BQ_BACKTICK_RE = re.compile(r"`([A-Za-z0-9_.-]+)`")
_BQ_CREATE_OR_REPLACE_TABLE_RE = re.compile(r"\bCREATE\s+OR\s+REPLACE\s+TABLE\b", re.IGNORECASE)
_BQ_GENERIC_TYPE_RE = re.compile(r"\b(?:ARRAY|STRUCT)\s*<[^<>]+>", re.IGNORECASE)
_BQ_OPTIONS_RE = re.compile(r"\bOPTIONS\s*\([^)]*\)", re.IGNORECASE | re.DOTALL)
_BQ_QUALIFY_RE = re.compile(r"\bQUALIFY\b[^;]+?(?=;|\Z)", re.IGNORECASE | re.DOTALL)


def _get_sql_parser() -> Parser:
    global _SQL_PARSER
    if _SQL_PARSER is None:
        _SQL_PARSER = get_parser("sql")
    return _SQL_PARSER


def _get_yaml_parser() -> Parser:
    global _YAML_PARSER
    if _YAML_PARSER is None:
        _YAML_PARSER = get_parser("yaml")
    return _YAML_PARSER


def _preserve_newlines(text: str, replacement: str = " ") -> str:
    """Replace text while preserving exact newline count for line-number alignment."""
    nl_count = text.count("\n")
    if nl_count == 0:
        return replacement
    return replacement + ("\n" * nl_count)


def _line_of_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _extract_jinja_calls(
    code: str,
) -> tuple[list[tuple[str, str]], list[str], str | None]:
    """Validate dbt Jinja syntax and extract ref/source/config calls.

    Returns ``(refs_and_sources, configs, error_message)``.
    Each item in ``refs_and_sources`` is ``("ref", model)`` or ``("source", "src.tbl")``.
    """
    if "{{" not in code and "{%" not in code and "{#" not in code:
        return [], [], None

    env = jinja2.Environment()
    try:
        ast_node = env.parse(code)
    except jinja2.TemplateSyntaxError as exc:
        return [], [], f"Jinja syntax error at line {exc.lineno}: {exc.message}"

    refs_and_sources: list[tuple[str, str]] = []
    configs: list[str] = []

    def _const_str(node: jinja2.nodes.Node) -> str | None:
        if isinstance(node, jinja2.nodes.Const) and isinstance(node.value, str):
            return node.value
        return None

    for call in ast_node.find_all(jinja2.nodes.Call):
        if isinstance(call.node, jinja2.nodes.Name):
            fn_name = call.node.name
            if fn_name == "ref" and call.args:
                if len(call.args) >= 2:
                    pkg = _const_str(call.args[0])
                    mdl = _const_str(call.args[1])
                    if pkg and mdl:
                        refs_and_sources.append(("ref", f"{pkg}.{mdl}"))
                else:
                    mdl = _const_str(call.args[0])
                    if mdl:
                        refs_and_sources.append(("ref", mdl))
            elif fn_name == "source" and len(call.args) >= 2:
                src = _const_str(call.args[0])
                tbl = _const_str(call.args[1])
                if src and tbl:
                    refs_and_sources.append(("source", f"{src}.{tbl}"))
            elif fn_name == "config":
                kv = [f"{kw.key}={_const_str(kw.value) or '...'}" for kw in call.kwargs]
                if kv:
                    configs.append(", ".join(kv))

    return refs_and_sources, configs, None


def _mask_jinja(code: str) -> str:
    """Mask validated Jinja tags into SQL-safe placeholders preserving line numbers."""
    masked = _JINJA_COMMENT_RE.sub(lambda m: _preserve_newlines(m.group(0), " "), code)
    masked = _JINJA_STMT_RE.sub(lambda m: _preserve_newlines(m.group(0), " "), masked)

    def _replace_expr(match: re.Match[str]) -> str:
        raw = match.group(0)
        # Keep ref/source/var as valid SQL identifiers so FROM {{ ref('x') }} parses
        ref_m = re.search(r"ref\s*\(\s*['\"]([^'\"]+)['\"]", raw)
        if ref_m:
            ident = re.sub(r"[^A-Za-z0-9_]", "_", ref_m.group(1))
            return _preserve_newlines(raw, f"dbt_ref_{ident}")
        src_m = re.search(r"source\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]", raw)
        if src_m:
            s_id = re.sub(r"[^A-Za-z0-9_]", "_", src_m.group(1))
            t_id = re.sub(r"[^A-Za-z0-9_]", "_", src_m.group(2))
            return _preserve_newlines(raw, f"dbt_src_{s_id}__{t_id}")
        if "config(" in raw:
            return _preserve_newlines(raw, " ")
        return _preserve_newlines(raw, "dbt_expr")

    return _JINJA_EXPR_RE.sub(_replace_expr, masked)


def _normalize_pg17_and_bigquery(
    sql: str,
) -> tuple[str, list[tuple[str, str, int, int]]]:
    """Normalize valid PG17 & BigQuery extensions that tree-sitter-sql 0.13 flags.

    Returns ``(normalized_sql, extracted_policies)`` where each policy tuple is
    ``(policy_name, table_name, start_line, end_line)``.
    """
    policies: list[tuple[str, str, int, int]] = []

    # 1. Strip PG17 RLS statements (valid DDL not in tree-sitter-sql 0.13)
    out = _PG_ALTER_RLS_RE.sub(lambda m: _preserve_newlines(m.group(0), "SELECT 1;"), sql)
    out = _PG_DROP_POLICY_RE.sub(lambda m: _preserve_newlines(m.group(0), "SELECT 1;"), out)

    def _collect_policy(match: re.Match[str]) -> str:
        p_name = match.group(1).strip('"')
        t_name = match.group(2).strip('"')
        start_ln = _line_of_offset(sql, match.start())
        end_ln = _line_of_offset(sql, match.end())
        policies.append((p_name, t_name, start_ln, end_ln))
        return _preserve_newlines(match.group(0), "SELECT 1;")

    out = _PG_CREATE_POLICY_RE.sub(_collect_policy, out)

    # 2. PG17 DO $$ ... END $$; and GRANT statements
    out = _PG_DO_BLOCK_RE.sub(lambda m: _preserve_newlines(m.group(0), "SELECT 1;"), out)
    out = _PG_GRANT_RE.sub(lambda m: _preserve_newlines(m.group(0), "SELECT 1;"), out)

    # 3. pgvector column types & HNSW/IVFFlat index clauses
    out = _PG_VECTOR_TYPE_RE.sub(lambda m: m.group(1), out)
    out = _PG_USING_INDEX_RE.sub(lambda m: f"({_PG_OPCLASS_RE.sub('', m.group(1)).strip()})", out)
    out = _PG_OPCLASS_RE.sub("", out)

    # 4. BigQuery CREATE OR REPLACE TABLE, backticked identifiers, generic types, OPTIONS(), QUALIFY
    out = _BQ_CREATE_OR_REPLACE_TABLE_RE.sub(
        lambda m: _preserve_newlines(m.group(0), "CREATE TABLE"), out
    )
    out = _BQ_BACKTICK_RE.sub(lambda m: m.group(1), out)
    # Iteratively collapse nested ARRAY<STRUCT<...>>
    for _ in range(3):
        next_out = _BQ_GENERIC_TYPE_RE.sub("TEXT", out)
        if next_out == out:
            break
        out = next_out
    out = _BQ_OPTIONS_RE.sub(lambda m: _preserve_newlines(m.group(0), " "), out)
    out = _BQ_QUALIFY_RE.sub(lambda m: _preserve_newlines(m.group(0), " "), out)

    return out, policies


def _clean_ident(raw: str) -> str:
    return raw.strip().strip('"').strip("`").strip("'")


def _find_descendants(node: Node, node_types: set[str]) -> list[Node]:
    found: list[Node] = []
    stack: list[Node] = [node]
    while stack:
        curr = stack.pop()
        if curr.type in node_types:
            found.append(curr)
        stack.extend(reversed(curr.children))
    return found


def _extract_object_name(node: Node, code_bytes: bytes) -> str:
    """Extract table/view/function/index identifier from an object_reference or identifier."""
    for child in node.children:
        if child.type == "object_reference":
            return _clean_ident(_node_text(child, code_bytes))
    for child in node.children:
        if child.type == "identifier":
            return _clean_ident(_node_text(child, code_bytes))
    return ""


class SqlDbtPlugin:
    """Language plugin for PostgreSQL 17, BigQuery SQL, and dbt Jinja SQL."""

    def __init__(self, dialect: str = "auto") -> None:
        self.dialect = dialect
        self.last_parse_diagnostics: ParseDiagnostics | None = None

    @property
    def language_id(self) -> str:
        return "sql"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".sql",)

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("dbt_project.yml", "packages.yml", "schema.yml")

    def can_handle(self, file_path: str) -> bool:
        lower = file_path.lower()
        if lower.endswith(".sql"):
            return True
        name = Path(file_path).name.lower()
        return name in _MANIFEST_BASENAMES

    def _derive_module_prefix(self, file_path: str) -> str:
        if not file_path:
            return "sql"
        stem = re.sub(r"\.sql$", "", file_path, flags=re.IGNORECASE)
        normalized = stem.replace("\\", "/").strip("/")
        dotted = ".".join(part for part in normalized.split("/") if part)
        return dotted or "sql"

    def _detect_dialect(self, code: str) -> str:
        if self.dialect != "auto":
            return self.dialect
        if "`" in code or re.search(r"\b(INT64|FLOAT64|QUALIFY|OPTIONS)\b", code):
            return "bigquery"
        return "postgres17"

    def extract_ast(
        self,
        code: str,
        file_path: str,
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        if not code or not code.strip():
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-sql")
            return [], []

        refs_and_sources, configs, jinja_err = _extract_jinja_calls(code)
        if jinja_err is not None:
            logger.warning(
                "Jinja template syntax error in '%s': %s; abstaining",
                file_path or "<memory>",
                jinja_err,
            )
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="tree-sitter-sql+jinja2",
                has_parse_error=True,
                abstained=True,
                low_confidence_fallback=False,
                error_node_count=1,
                reason=jinja_err,
            )
            return [], []

        masked = _mask_jinja(code)
        normalized, policies = _normalize_pg17_and_bigquery(masked)
        code_bytes = normalized.encode("utf-8")
        tree = _get_sql_parser().parse(code_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)

        if has_err:
            reason = (
                f"Partial/invalid SQL syntax in '{file_path or '<memory>'}': "
                f"error_nodes={err_cnt}, missing_nodes={miss_cnt}"
            )
            logger.warning("%s; abstaining from partial AST extraction", reason)
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="tree-sitter-sql",
                has_parse_error=True,
                abstained=True,
                low_confidence_fallback=False,
                error_node_count=err_cnt,
                missing_node_count=miss_cnt,
                reason=reason,
            )
            return [], []

        prefix = module_prefix or self._derive_module_prefix(file_path)
        dialect = self._detect_dialect(code)

        def _qualify(bare: str) -> str:
            if repo_name:
                return f"{repo_name}::{prefix}::{bare}"
            return f"{prefix}.{bare}" if prefix else bare

        def _repo_qual(bare: str) -> str | None:
            return f"{repo_name}::{prefix}::{bare}" if repo_name else None

        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        seen_edges: set[tuple[str, str, str]] = set()

        def _add_edge(src: str, tgt: str, kind: str) -> None:
            key = (src, tgt, kind)
            if key not in seen_edges and src and tgt:
                seen_edges.add(key)
                edges.append(
                    ParsedEdge(
                        source_qualified_name=src,
                        target_qualified_name=tgt,
                        edge_kind=kind,
                    )
                )

        # Emit policy symbols extracted from normalized CREATE POLICY statements
        for p_name, t_name, start_ln, end_ln in policies:
            qname = _qualify(p_name)
            symbols.append(
                ParsedSymbol(
                    name=p_name,
                    qualified_name=qname,
                    symbol_kind="module",
                    start_line=start_ln,
                    end_line=max(start_ln, end_ln),
                    start_byte=0,
                    end_byte=0,
                    signature=f"CREATE POLICY {p_name} ON {t_name}",
                    signature_skeleton=f"POLICY {p_name} ON {t_name}",
                    contextual_summary=f"@policy {p_name} table={t_name} dialect={dialect}",
                    repo_name=repo_name,
                    repo_qualified_id=_repo_qual(p_name),
                    bare_name=p_name,
                )
            )
            _add_edge(qname, _qualify(t_name), "USES_TYPE")

        for stmt in _find_descendants(tree.root_node, {"statement"}):
            for child in stmt.children:
                if child.type == "create_table":
                    self._handle_create_table(
                        child,
                        code_bytes,
                        dialect,
                        repo_name,
                        _qualify,
                        _repo_qual,
                        symbols,
                        _add_edge,
                    )
                elif child.type in {"create_view", "create_materialized_view"}:
                    self._handle_create_view(
                        child,
                        code_bytes,
                        dialect,
                        repo_name,
                        _qualify,
                        _repo_qual,
                        symbols,
                        _add_edge,
                    )
                elif child.type in {"create_function", "create_procedure"}:
                    self._handle_create_function(
                        child,
                        code_bytes,
                        dialect,
                        repo_name,
                        _qualify,
                        _repo_qual,
                        symbols,
                        _add_edge,
                    )
                elif child.type == "create_index":
                    self._handle_create_index(
                        child,
                        code_bytes,
                        dialect,
                        repo_name,
                        _qualify,
                        _repo_qual,
                        symbols,
                        _add_edge,
                    )
                elif child.type == "create_type":
                    self._handle_create_type(
                        child, code_bytes, dialect, repo_name, _qualify, _repo_qual, symbols
                    )

        # Handle CTEs and top-level SELECT / dbt models
        cte_nodes = _find_descendants(tree.root_node, {"cte"})
        model_name = Path(file_path).stem if file_path else "sql_model"
        model_qname = _qualify(model_name)

        if not symbols or refs_and_sources or cte_nodes:
            # Ensure a model/module symbol exists for dbt models or standalone queries
            has_model_sym = any(s.qualified_name == model_qname for s in symbols)
            if not has_model_sym and (refs_and_sources or cte_nodes or not symbols):
                total_lines = max(1, len(code.splitlines()))
                cfg_note = f" config=({'; '.join(configs)})" if configs else ""
                symbols.append(
                    ParsedSymbol(
                        name=model_name,
                        qualified_name=model_qname,
                        symbol_kind="module",
                        start_line=1,
                        end_line=total_lines,
                        start_byte=0,
                        end_byte=len(code.encode("utf-8")),
                        signature=f"SQL model {model_name}",
                        signature_skeleton=f"SQL model {model_name}",
                        contextual_summary=f"@sql_model {model_name} dialect={dialect}{cfg_note}",
                        repo_name=repo_name,
                        repo_qualified_id=_repo_qual(model_name),
                        bare_name=model_name,
                    )
                )

        # Extract CTEs and their table references
        for cte in cte_nodes:
            cte_id_node = next((c for c in cte.children if c.type == "identifier"), None)
            if cte_id_node is None:
                continue
            cte_name = _clean_ident(_node_text(cte_id_node, code_bytes))
            if not cte_name:
                continue
            cte_qname = _qualify(cte_name)
            symbols.append(
                ParsedSymbol(
                    name=cte_name,
                    qualified_name=cte_qname,
                    symbol_kind="type_alias",
                    start_line=cte.start_point[0] + 1,
                    end_line=max(cte.start_point[0] + 1, cte.end_point[0] + 1),
                    start_byte=cte.start_byte,
                    end_byte=cte.end_byte,
                    signature=f"WITH {cte_name} AS (...)",
                    signature_skeleton=f"CTE {cte_name}",
                    contextual_summary=f"@cte {cte_name} dialect={dialect}",
                    repo_name=repo_name,
                    repo_qualified_id=_repo_qual(cte_name),
                    bare_name=cte_name,
                )
            )
            _add_edge(model_qname, cte_qname, "USES_TYPE")
            for rel in _find_descendants(cte, {"relation"}):
                obj_ref = next((c for c in rel.children if c.type == "object_reference"), None)
                if obj_ref is not None:
                    tbl = _clean_ident(_node_text(obj_ref, code_bytes))
                    if tbl and tbl != cte_name:
                        _add_edge(cte_qname, _resolve_dbt_or_table(tbl, _qualify), "USES_TYPE")

        # Extract dbt ref() and source() edges
        for kind, target in refs_and_sources:
            _add_edge(model_qname, f"dbt::{kind}::{target}", "IMPORTS")
        if configs:
            _add_edge(model_qname, "dbt::config", "USES_TYPE")

        # Extract top-level FROM / JOIN relation references outside CTEs
        for rel in _find_descendants(tree.root_node, {"relation"}):
            obj_ref = next((c for c in rel.children if c.type == "object_reference"), None)
            if obj_ref is not None:
                tbl = _clean_ident(_node_text(obj_ref, code_bytes))
                if tbl:
                    _add_edge(model_qname, _resolve_dbt_or_table(tbl, _qualify), "USES_TYPE")

        self.last_parse_diagnostics = ParseDiagnostics(
            parser_backend="tree-sitter-sql+jinja2"
            if refs_and_sources or configs
            else "tree-sitter-sql"
        )
        return symbols, edges

    def _handle_create_table(
        self,
        node: Node,
        code_bytes: bytes,
        dialect: str,
        repo_name: str,
        qualify: Any,
        repo_qual: Any,
        symbols: list[ParsedSymbol],
        add_edge: Any,
    ) -> None:
        table_name = _extract_object_name(node, code_bytes)
        if not table_name:
            return
        qname = qualify(table_name)
        cols: list[str] = []
        fk_targets: list[str] = []

        for col_def in _find_descendants(node, {"column_definition"}):
            col_id = next((c for c in col_def.children if c.type == "identifier"), None)
            if col_id is not None:
                cols.append(_clean_ident(_node_text(col_id, code_bytes)))

        # Foreign key REFERENCES targets
        for kw_node in _find_descendants(node, {"keyword_references"}):
            parent = kw_node.parent
            if parent is None:
                continue
            for sibling in parent.children:
                if sibling.type == "object_reference":
                    fk_targets.append(_clean_ident(_node_text(sibling, code_bytes)))

        col_summary = ", ".join(cols[:12])
        if len(cols) > 12:
            col_summary += f", +{len(cols) - 12} more"

        symbols.append(
            ParsedSymbol(
                name=table_name,
                qualified_name=qname,
                symbol_kind="class",
                start_line=node.start_point[0] + 1,
                end_line=max(node.start_point[0] + 1, node.end_point[0] + 1),
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                signature=f"CREATE TABLE {table_name} ({col_summary})",
                signature_skeleton=f"TABLE {table_name} ({col_summary})",
                contextual_summary=f"@dto {table_name} fields=[{col_summary}] dialect={dialect}",
                repo_name=repo_name,
                repo_qualified_id=repo_qual(table_name),
                bare_name=table_name,
            )
        )
        for fk in fk_targets:
            add_edge(qname, qualify(fk), "USES_TYPE")

    def _handle_create_view(
        self,
        node: Node,
        code_bytes: bytes,
        dialect: str,
        repo_name: str,
        qualify: Any,
        repo_qual: Any,
        symbols: list[ParsedSymbol],
        add_edge: Any,
    ) -> None:
        view_name = _extract_object_name(node, code_bytes)
        if not view_name:
            return
        qname = qualify(view_name)
        symbols.append(
            ParsedSymbol(
                name=view_name,
                qualified_name=qname,
                symbol_kind="interface",
                start_line=node.start_point[0] + 1,
                end_line=max(node.start_point[0] + 1, node.end_point[0] + 1),
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                signature=f"CREATE VIEW {view_name}",
                signature_skeleton=f"VIEW {view_name}",
                contextual_summary=f"@view {view_name} dialect={dialect}",
                repo_name=repo_name,
                repo_qualified_id=repo_qual(view_name),
                bare_name=view_name,
            )
        )
        for rel in _find_descendants(node, {"relation"}):
            obj_ref = next((c for c in rel.children if c.type == "object_reference"), None)
            if obj_ref is not None:
                tbl = _clean_ident(_node_text(obj_ref, code_bytes))
                if tbl and tbl != view_name:
                    add_edge(qname, qualify(tbl), "USES_TYPE")

    def _handle_create_function(
        self,
        node: Node,
        code_bytes: bytes,
        dialect: str,
        repo_name: str,
        qualify: Any,
        repo_qual: Any,
        symbols: list[ParsedSymbol],
        add_edge: Any,
    ) -> None:
        fn_name = _extract_object_name(node, code_bytes)
        if not fn_name:
            return
        qname = qualify(fn_name)
        kind_kw = "PROCEDURE" if node.type == "create_procedure" else "FUNCTION"
        symbols.append(
            ParsedSymbol(
                name=fn_name,
                qualified_name=qname,
                symbol_kind="function",
                start_line=node.start_point[0] + 1,
                end_line=max(node.start_point[0] + 1, node.end_point[0] + 1),
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                signature=f"CREATE {kind_kw} {fn_name}(...)",
                signature_skeleton=f"{kind_kw} {fn_name}(...)",
                contextual_summary=f"@sql_function {fn_name} dialect={dialect}",
                repo_name=repo_name,
                repo_qualified_id=repo_qual(fn_name),
                bare_name=fn_name,
            )
        )
        for inv in _find_descendants(node, {"invocation"}):
            obj_ref = next((c for c in inv.children if c.type == "object_reference"), None)
            if obj_ref is not None:
                called = _clean_ident(_node_text(obj_ref, code_bytes))
                if called and called != fn_name:
                    add_edge(qname, qualify(called), "CALLS")

    def _handle_create_index(
        self,
        node: Node,
        code_bytes: bytes,
        dialect: str,
        repo_name: str,
        qualify: Any,
        repo_qual: Any,
        symbols: list[ParsedSymbol],
        add_edge: Any,
    ) -> None:
        idx_name = ""
        tbl_name = ""
        for child in node.children:
            if child.type == "identifier" and not idx_name:
                idx_name = _clean_ident(_node_text(child, code_bytes))
            elif child.type == "object_reference":
                tbl_name = _clean_ident(_node_text(child, code_bytes))
        if not idx_name:
            return
        qname = qualify(idx_name)
        symbols.append(
            ParsedSymbol(
                name=idx_name,
                qualified_name=qname,
                symbol_kind="module",
                start_line=node.start_point[0] + 1,
                end_line=max(node.start_point[0] + 1, node.end_point[0] + 1),
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                signature=f"CREATE INDEX {idx_name} ON {tbl_name}",
                signature_skeleton=f"INDEX {idx_name} ON {tbl_name}",
                contextual_summary=f"@index {idx_name} table={tbl_name} dialect={dialect}",
                repo_name=repo_name,
                repo_qualified_id=repo_qual(idx_name),
                bare_name=idx_name,
            )
        )
        if tbl_name:
            add_edge(qname, qualify(tbl_name), "USES_TYPE")

    def _handle_create_type(
        self,
        node: Node,
        code_bytes: bytes,
        dialect: str,
        repo_name: str,
        qualify: Any,
        repo_qual: Any,
        symbols: list[ParsedSymbol],
    ) -> None:
        type_name = _extract_object_name(node, code_bytes)
        if not type_name:
            return
        symbols.append(
            ParsedSymbol(
                name=type_name,
                qualified_name=qualify(type_name),
                symbol_kind="type_alias",
                start_line=node.start_point[0] + 1,
                end_line=max(node.start_point[0] + 1, node.end_point[0] + 1),
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                signature=f"CREATE TYPE {type_name}",
                signature_skeleton=f"TYPE {type_name}",
                contextual_summary=f"@type {type_name} dialect={dialect}",
                repo_name=repo_name,
                repo_qualified_id=repo_qual(type_name),
                bare_name=type_name,
            )
        )

    def extract_manifest_dependencies(
        self,
        file_path: str,
        content: str,
    ) -> list[DependencyEdge]:
        """Extract dependencies from dbt_project.yml, packages.yml, and schema.yml."""
        if not content or not content.strip():
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-yaml")
            return []

        tree = _get_yaml_parser().parse(content.encode("utf-8"))
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            reason = (
                f"Malformed YAML manifest '{file_path or '<memory>'}': "
                f"error_nodes={err_cnt}, missing_nodes={miss_cnt}"
            )
            logger.warning("%s; abstaining", reason)
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="tree-sitter-yaml",
                has_parse_error=True,
                abstained=True,
                low_confidence_fallback=False,
                error_node_count=err_cnt,
                missing_node_count=miss_cnt,
                reason=reason,
            )
            return []

        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            reason = f"YAML parse error in '{file_path or '<memory>'}': {exc}"
            logger.warning("%s; abstaining", reason)
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="tree-sitter-yaml",
                has_parse_error=True,
                abstained=True,
                low_confidence_fallback=False,
                error_node_count=1,
                reason=reason,
            )
            return []

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-yaml")
        if not isinstance(data, dict):
            return []

        edges: list[DependencyEdge] = []
        basename = Path(file_path).name.lower()

        if basename == "packages.yml" or "packages" in data:
            pkgs = data.get("packages") or []
            if isinstance(pkgs, list):
                for pkg in pkgs:
                    if not isinstance(pkg, dict):
                        continue
                    target = (
                        pkg.get("package")
                        or pkg.get("git")
                        or pkg.get("local")
                        or pkg.get("tarball")
                    )
                    if not target:
                        continue
                    version = str(pkg.get("version") or pkg.get("revision") or "*")
                    edges.append(
                        DependencyEdge(
                            source_identifier="dbt:packages",
                            target_package=str(target),
                            version_constraint=version,
                            scope="compile",
                            manifest_path=file_path,
                        )
                    )

        if basename == "dbt_project.yml" or "name" in data:
            proj_name = str(data.get("name") or "dbt_project")
            dbt_ver = data.get("require-dbt-version")
            if dbt_ver:
                edges.append(
                    DependencyEdge(
                        source_identifier=proj_name,
                        target_package="dbt-core",
                        version_constraint=str(dbt_ver),
                        scope="runtime",
                        manifest_path=file_path,
                    )
                )

        if basename == "schema.yml" or "sources" in data:
            sources = data.get("sources") or []
            if isinstance(sources, list):
                for src in sources:
                    if not isinstance(src, dict) or not src.get("name"):
                        continue
                    src_name = str(src["name"])
                    for tbl in src.get("tables") or []:
                        if isinstance(tbl, dict) and tbl.get("name"):
                            edges.append(
                                DependencyEdge(
                                    source_identifier=f"dbt:source:{src_name}",
                                    target_package=f"{src_name}.{tbl['name']}",
                                    version_constraint="*",
                                    scope="runtime",
                                    manifest_path=file_path,
                                )
                            )

        return edges


def _resolve_dbt_or_table(tbl: str, qualify: Any) -> str:
    if tbl.startswith("dbt_ref_"):
        return f"dbt::ref::{tbl.removeprefix('dbt_ref_')}"
    if tbl.startswith("dbt_src_"):
        rest = tbl.removeprefix("dbt_src_").replace("__", ".", 1)
        return f"dbt::source::{rest}"
    return str(qualify(tbl))
