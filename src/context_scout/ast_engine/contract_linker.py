"""Cross-boundary contract linker for polyglot multi-repo AST graphs."""

from __future__ import annotations

import re
from collections.abc import Sequence
from uuid import uuid4

from context_scout.db.schema import ASTEdgeRecord, ASTSymbolRecord, CodeFileRecord, DocChunkRecord

_RE_ROUTE_TAG = re.compile(r"@route\s+(GET|POST|PUT|DELETE|PATCH|ANY)\s+([^\s]+)")
_RE_GQL_TAG = re.compile(r"@graphql\s+(query|mutation)\s+([A-Za-z_][A-Za-z0-9_]*)")
_RE_DTO_TAG = re.compile(r"@dto\s+([A-Za-z_][A-Za-z0-9_]*)\s+fields=([^\s]*)")
_RE_DTO_REF_TAG = re.compile(r"@dto_ref\s+([A-Za-z_][A-Za-z0-9_]*)")
_RE_ENV_TAG = re.compile(r"@env\s+([A-Za-z0-9_]+)")
_RE_SERVICE_TAG = re.compile(r"@service\s+([^\s]+)")
_RE_STORY_COMP_TAG = re.compile(r"@story_component\s+([A-Za-z0-9_.]+)")

_BACKEND_LANGUAGES = frozenset({"csharp", "python", "go", "java", "rust"})
_FRONTEND_LANGUAGES = frozenset({"typescript", "tsx", "javascript", "jsx", "react_redux"})


def normalize_route_template(raw_route: str) -> str:
    """Normalize REST path templates across ASP.NET Core, TS template literals, and Express."""
    cleaned = raw_route.strip().split("?", 1)[0].split("#", 1)[0]
    cleaned = re.sub(r"^https?://[^/]+", "", cleaned)
    if not cleaned:
        return "/"
    segments = [seg for seg in cleaned.split("/") if seg]
    normalized_segments: list[str] = []
    for seg in segments:
        if (
            (seg.startswith("{") and seg.endswith("}"))
            or (seg.startswith("${") and seg.endswith("}"))
            or seg.startswith(":")
        ):
            normalized_segments.append("{param}")
        else:
            normalized_segments.append(seg.lower())
    return "/" + "/".join(normalized_segments)


def _norm_field(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _norm_dto_name(name: str) -> str:
    lower = name.lower()
    for suffix in ("dto", "model", "payload"):
        if lower.endswith(suffix) and len(lower) > len(suffix) + 1:
            return lower[: -len(suffix)]
    return lower


def _repo_qualified(sym: ASTSymbolRecord, repo_name: str) -> str:
    if sym.repo_qualified_id:
        return sym.repo_qualified_id
    if repo_name:
        return f"{repo_name}::{sym.qualified_name}"
    return sym.qualified_name


def _intersect_scopes(src: ASTSymbolRecord, tgt: ASTSymbolRecord) -> list[str]:
    tgt_set = set(tgt.allowed_scopes)
    shared = [s for s in src.allowed_scopes if s in tgt_set]
    return shared or list(src.allowed_scopes)


class CrossBoundaryContractLinker:
    """Synthesizes cross-repo AST edges across backend, frontend, Storybook, and Terraform."""

    def link_contracts(
        self,
        domain_id: str,
        symbols: Sequence[ASTSymbolRecord],
        code_files: Sequence[CodeFileRecord],
        doc_chunks: Sequence[DocChunkRecord] = (),
    ) -> list[ASTEdgeRecord]:
        """Link cross-boundary REST, GraphQL, DTO, Storybook, and Terraform contracts."""
        _ = doc_chunks
        file_by_id = {cf.file_id: cf for cf in code_files}
        edges: list[ASTEdgeRecord] = []
        seen_keys: set[tuple[str, str, str]] = set()

        def sym_repo(sym: ASTSymbolRecord) -> str:
            cf = file_by_id.get(sym.file_id)
            if cf is not None:
                return cf.repo_name
            if sym.repo_qualified_id and "::" in sym.repo_qualified_id:
                return sym.repo_qualified_id.split("::", 1)[0]
            return ""

        def sym_lang(sym: ASTSymbolRecord) -> str:
            cf = file_by_id.get(sym.file_id)
            return cf.language.lower() if cf is not None else ""

        def sym_meta_text(sym: ASTSymbolRecord) -> str:
            return f"{sym.contextual_summary}\n{sym.signature_skeleton}"

        def emit_edge(src: ASTSymbolRecord, tgt: ASTSymbolRecord, edge_kind: str) -> None:
            if src.symbol_id == tgt.symbol_id:
                return
            src_repo = sym_repo(src)
            tgt_repo = sym_repo(tgt)
            tgt_rqid = _repo_qualified(tgt, tgt_repo)
            key = (str(src.symbol_id), str(tgt.symbol_id), edge_kind)
            if key in seen_keys:
                return
            seen_keys.add(key)
            edges.append(
                ASTEdgeRecord(
                    edge_id=uuid4(),
                    domain_id=domain_id,
                    source_symbol_id=src.symbol_id,
                    target_qualified_name=tgt.qualified_name,
                    target_symbol_id=tgt.symbol_id,
                    edge_kind=edge_kind,
                    allowed_scopes=_intersect_scopes(src, tgt),
                    target_repo_qualified_id=tgt_rqid,
                    is_cross_repo=bool(src_repo and tgt_repo and src_repo != tgt_repo),
                )
            )

        backend_routes: dict[tuple[str, str], list[ASTSymbolRecord]] = {}
        frontend_routes: list[tuple[ASTSymbolRecord, str, str]] = []
        backend_gql: dict[tuple[str, str], list[ASTSymbolRecord]] = {}
        frontend_gql: list[tuple[ASTSymbolRecord, str, str]] = []
        backend_dtos: list[tuple[ASTSymbolRecord, str, set[str]]] = []
        frontend_dtos: list[tuple[ASTSymbolRecord, str, set[str]]] = []
        dto_refs: list[tuple[ASTSymbolRecord, str]] = []
        tf_envs: dict[str, list[ASTSymbolRecord]] = {}
        tf_services: dict[str, list[ASTSymbolRecord]] = {}
        app_envs: list[tuple[ASTSymbolRecord, str]] = []
        story_components: list[tuple[ASTSymbolRecord, str]] = []
        ui_components_by_name: dict[str, list[ASTSymbolRecord]] = {}

        for sym in symbols:
            lang = sym_lang(sym)
            text = sym_meta_text(sym)

            if lang in _FRONTEND_LANGUAGES and sym.symbol_kind in {"class", "function", "hook"}:
                leaf = sym.bare_name or sym.qualified_name.rsplit(".", 1)[-1]
                ui_components_by_name.setdefault(leaf, []).append(sym)
                ui_components_by_name.setdefault(sym.qualified_name, []).append(sym)

            for method, raw_route in _RE_ROUTE_TAG.findall(text):
                norm_route = normalize_route_template(raw_route)
                is_fe = lang in _FRONTEND_LANGUAGES or "builder." in sym.signature_skeleton
                if is_fe and lang not in _BACKEND_LANGUAGES:
                    frontend_routes.append((sym, method, norm_route))
                else:
                    backend_routes.setdefault((method, norm_route), []).append(sym)

            for op_type, op_name in _RE_GQL_TAG.findall(text):
                key = (op_type.lower(), op_name.lower())
                if lang in _FRONTEND_LANGUAGES:
                    frontend_gql.append((sym, key[0], key[1]))
                else:
                    backend_gql.setdefault(key, []).append(sym)

            for dto_name, raw_fields in _RE_DTO_TAG.findall(text):
                field_set = {_norm_field(f) for f in raw_fields.split(",") if f.strip()}
                if lang in _FRONTEND_LANGUAGES:
                    frontend_dtos.append((sym, dto_name, field_set))
                else:
                    backend_dtos.append((sym, dto_name, field_set))

            for ref_name in _RE_DTO_REF_TAG.findall(text):
                dto_refs.append((sym, ref_name))

            for env_key in _RE_ENV_TAG.findall(text):
                upper_key = env_key.upper()
                if lang == "terraform":
                    tf_envs.setdefault(upper_key, []).append(sym)
                    if upper_key.startswith("TF_VAR_"):
                        tf_envs.setdefault(upper_key[len("TF_VAR_") :], []).append(sym)
                else:
                    app_envs.append((sym, upper_key))

            if lang == "terraform":
                for svc_name in _RE_SERVICE_TAG.findall(text):
                    tf_services.setdefault(svc_name.lower(), []).append(sym)

            if sym.symbol_kind == "story":
                for comp_name in _RE_STORY_COMP_TAG.findall(text):
                    story_components.append((sym, comp_name))

        for fe_sym, method, norm_route in frontend_routes:
            targets = backend_routes.get((method, norm_route), []) + backend_routes.get(
                ("ANY", norm_route), []
            )
            for be_sym in targets:
                emit_edge(fe_sym, be_sym, "CALLS")

        for fe_sym, op_type, op_name in frontend_gql:
            for be_sym in backend_gql.get((op_type, op_name), []):
                emit_edge(fe_sym, be_sym, "CALLS")

        backend_dto_by_norm: dict[str, list[ASTSymbolRecord]] = {}
        for be_sym, be_name, be_fields in backend_dtos:
            backend_dto_by_norm.setdefault(be_name.lower(), []).append(be_sym)
            backend_dto_by_norm.setdefault(_norm_dto_name(be_name), []).append(be_sym)
            for fe_sym, fe_name, fe_fields in frontend_dtos:
                exact_match = fe_name.lower() == be_name.lower()
                stem_match = _norm_dto_name(fe_name) == _norm_dto_name(be_name)
                overlap = fe_fields & be_fields
                min_fields = min(len(fe_fields), len(be_fields))
                field_match = min_fields > 0 and (len(overlap) * 2 >= min_fields)
                if exact_match or (stem_match and (field_match or not fe_fields or not be_fields)):
                    emit_edge(fe_sym, be_sym, "USES_TYPE")

        for ref_sym, ref_name in dto_refs:
            for be_sym in backend_dto_by_norm.get(ref_name.lower(), []):
                emit_edge(ref_sym, be_sym, "USES_TYPE")

        for story_sym, comp_name in story_components:
            candidates = ui_components_by_name.get(comp_name) or ui_components_by_name.get(
                comp_name.rsplit(".", 1)[-1], []
            )
            for comp_sym in candidates:
                emit_edge(story_sym, comp_sym, "DOCUMENTS_COMPONENT")

        for app_sym, env_key in app_envs:
            for tf_sym in tf_envs.get(env_key, []):
                emit_edge(app_sym, tf_sym, "PROVISIONED_BY")

        if tf_services:
            for sym in symbols:
                if sym_lang(sym) == "terraform":
                    continue
                rname = sym_repo(sym).lower()
                if rname in tf_services and sym.symbol_kind in {"class", "method", "function"}:
                    for tf_sym in tf_services[rname]:
                        emit_edge(sym, tf_sym, "PROVISIONED_BY")

        return edges
