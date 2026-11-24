"""Cross-boundary contract linker v2 for polyglot multi-repo AST & spec graphs."""

from __future__ import annotations

import re
from collections.abc import Sequence
from uuid import UUID, uuid4

from context_scout.db.schema import ASTEdgeRecord, ASTSymbolRecord, CodeFileRecord, DocChunkRecord
from context_scout.schemas.contracts import DEFAULT_TENANT_ID

_RE_ROUTE_TAG = re.compile(r"@route\s+(GET|POST|PUT|DELETE|PATCH|ANY)\s+([^\s]+)")
_RE_GQL_TAG = re.compile(r"@graphql\s+(query|mutation)\s+([A-Za-z_][A-Za-z0-9_]*)")
_RE_DTO_TAG = re.compile(r"@dto\s+([A-Za-z_][A-Za-z0-9_.]*)\s+fields=(?:\[([^\]]*)\]|([^\s]*))")
_RE_DTO_REF_TAG = re.compile(r"@dto_ref\s+([A-Za-z_][A-Za-z0-9_.]*)")
_RE_ENV_TAG = re.compile(r"@env\s+([A-Za-z0-9_]+)")
_RE_SERVICE_TAG = re.compile(r"@service\s+([^\s]+)")
_RE_STORY_COMP_TAG = re.compile(r"@story_component\s+([A-Za-z0-9_.]+)")

_RE_GRPC_SVC_TAG = re.compile(r"@grpc_service\s+([A-Za-z0-9_.]+)")
_RE_CHANNEL_TAG = re.compile(r"@channel\s+([^\s]+)")
_RE_ASYNC_OP_TAG = re.compile(r"@async_op\s+([A-Za-z0-9_-]+)\s+([^\s]+)")
_RE_EVENT_PUB_TAG = re.compile(r"@(?:event_publish|publish_event|pubsub_topic)\s+([^\s]+)")
_RE_EVENT_SUB_TAG = re.compile(r"@(?:event_subscribe|subscribe_event|kafka_topic)\s+([^\s]+)")
_RE_SQL_MODEL_TAG = re.compile(r"@sql_model\s+([A-Za-z0-9_.]+)")
_RE_SQL_VIEW_TAG = re.compile(r"@view\s+([A-Za-z0-9_.]+)")
_RE_DB_WRITE_TAG = re.compile(r"@db_write\s+([A-Za-z0-9_.]+)")
_RE_DB_READ_TAG = re.compile(r"@db_read\s+([A-Za-z0-9_.]+)")
_RE_I18N_KEY_TAG = re.compile(r"@i18n_key\s+([A-Za-z0-9_.-]+)")
_RE_TEMPLATE_TAG = re.compile(r"@(?:template|renders_template)\s+([^\s]+)")
_RE_JIRA_KEY = re.compile(r"\b([A-Z][A-Z0-9]{1,15}-[0-9]{1,7})\b")

_BACKEND_LANGUAGES = frozenset({"csharp", "python", "go", "java", "java_spring", "kotlin", "rust"})
_FRONTEND_LANGUAGES = frozenset({"typescript", "tsx", "javascript", "jsx", "react_redux"})
_SPEC_LANGUAGES = frozenset(
    {"api_specs", "protobuf", "protobuf_grpc", "openapi", "asyncapi", "graphql"}
)
_DATA_LANGUAGES = frozenset({"sql", "sql_dbt", "dbt"})
_TEMPLATE_LANGUAGES = frozenset({"templates_i18n", "mjml", "handlebars", "jinja2"})


def normalize_route_template(raw_route: str) -> str:
    """Normalize REST path templates across ASP.NET Core, Spring, TS, OpenAPI, and gRPC."""
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
    base = name.split(":", 1)[0].strip()
    return re.sub(r"[^a-z0-9]", "", base.lower())


def _norm_dto_name(name: str) -> str:
    leaf = name.rsplit(".", 1)[-1].lower()
    for suffix in ("dto", "model", "payload", "entity", "record", "table", "msg", "message"):
        if leaf.endswith(suffix) and len(leaf) > len(suffix) + 1:
            return leaf[: -len(suffix)]
    return leaf


def _norm_topic(raw: str) -> str:
    cleaned = raw.strip().strip("'\"").strip("/")
    if "/topics/" in cleaned:
        cleaned = cleaned.rsplit("/topics/", 1)[-1]
    elif "/subscriptions/" in cleaned:
        cleaned = cleaned.rsplit("/subscriptions/", 1)[-1]
    return re.sub(r"[^a-z0-9]+", ".", cleaned.lower()).strip(".")


def _norm_template_name(raw: str) -> str:
    cleaned = raw.strip().strip("'\"").replace("\\", "/")
    leaf = cleaned.rsplit("/", 1)[-1]
    for ext in (".mjml", ".hbs", ".handlebars", ".j2", ".jinja2", ".jinja", ".html", ".tsx"):
        if leaf.lower().endswith(ext):
            leaf = leaf[: -len(ext)]
            break
    return leaf.lower()


def _repo_qualified(sym: ASTSymbolRecord, repo_name: str) -> str:
    if sym.repo_qualified_id:
        return sym.repo_qualified_id
    if repo_name:
        return f"{repo_name}::{sym.qualified_name}"
    return sym.qualified_name


def _intersect_scopes(src: ASTSymbolRecord, tgt: ASTSymbolRecord) -> list[str]:
    """Return strict scope intersection; disjoint enclaves return [] (fail-closed)."""
    tgt_set = set(tgt.allowed_scopes)
    return [s for s in src.allowed_scopes if s in tgt_set]


class CrossBoundaryContractLinker:
    """Synthesizes v2 cross-repo AST edges across services, specs, SQL, templates, and Jira."""

    def link_contracts(
        self,
        domain_id: str,
        symbols: Sequence[ASTSymbolRecord],
        code_files: Sequence[CodeFileRecord],
        doc_chunks: Sequence[DocChunkRecord] = (),
        *,
        tenant_id: UUID = DEFAULT_TENANT_ID,
    ) -> list[ASTEdgeRecord]:
        """Link cross-boundary REST, gRPC/OpenAPI, AsyncAPI, SQL/dbt, i18n, and Jira contracts."""
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

        def sym_path(sym: ASTSymbolRecord) -> str:
            cf = file_by_id.get(sym.file_id)
            return cf.file_path.lower() if cf is not None else ""

        def sym_meta_text(sym: ASTSymbolRecord) -> str:
            return f"{sym.contextual_summary}\n{sym.signature_skeleton}"

        def emit_edge(src: ASTSymbolRecord, tgt: ASTSymbolRecord, edge_kind: str) -> None:
            if src.symbol_id == tgt.symbol_id:
                return
            scopes = _intersect_scopes(src, tgt)
            if not scopes:
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
                    allowed_scopes=scopes,
                    target_repo_qualified_id=tgt_rqid,
                    is_cross_repo=bool(src_repo and tgt_repo and src_repo != tgt_repo),
                    tenant_id=tenant_id,
                )
            )

        backend_routes: dict[tuple[str, str], list[ASTSymbolRecord]] = {}
        spec_routes: dict[tuple[str, str], list[ASTSymbolRecord]] = {}
        frontend_routes: list[tuple[ASTSymbolRecord, str, str]] = []
        backend_gql: dict[tuple[str, str], list[ASTSymbolRecord]] = {}
        frontend_gql: list[tuple[ASTSymbolRecord, str, str]] = []
        backend_dtos: list[tuple[ASTSymbolRecord, str, set[str]]] = []
        frontend_dtos: list[tuple[ASTSymbolRecord, str, set[str]]] = []
        spec_dtos: list[tuple[ASTSymbolRecord, str, set[str]]] = []
        data_tables: list[tuple[ASTSymbolRecord, str, set[str]]] = []
        dto_refs: list[tuple[ASTSymbolRecord, str]] = []
        tf_envs: dict[str, list[ASTSymbolRecord]] = {}
        tf_services: dict[str, list[ASTSymbolRecord]] = {}
        tf_topics: dict[str, list[ASTSymbolRecord]] = {}
        app_envs: list[tuple[ASTSymbolRecord, str]] = []
        story_components: list[tuple[ASTSymbolRecord, str]] = []
        ui_components_by_name: dict[str, list[ASTSymbolRecord]] = {}

        # v2 registries
        grpc_spec_services: dict[str, list[ASTSymbolRecord]] = {}
        grpc_impl_symbols: list[tuple[ASTSymbolRecord, str]] = []
        async_channels: dict[str, list[ASTSymbolRecord]] = {}
        event_publishers: dict[str, list[ASTSymbolRecord]] = {}
        event_subscribers: dict[str, list[ASTSymbolRecord]] = {}
        sql_targets_by_name: dict[str, list[ASTSymbolRecord]] = {}
        db_writers: list[tuple[ASTSymbolRecord, str]] = []
        db_readers: list[tuple[ASTSymbolRecord, str]] = []
        i18n_catalog_by_key: dict[str, list[ASTSymbolRecord]] = {}
        i18n_consumers: list[tuple[ASTSymbolRecord, str]] = []
        template_defs_by_name: dict[str, list[ASTSymbolRecord]] = {}
        template_renderers: list[tuple[ASTSymbolRecord, str]] = []

        # Jira keys present in visible doc_chunks
        jira_doc_keys: dict[str, list[DocChunkRecord]] = {}
        for chunk in doc_chunks:
            if chunk.source_type == "jira" or chunk.doc_id.startswith("jira:"):
                for jk in _RE_JIRA_KEY.findall(f"{chunk.doc_id} {chunk.doc_title}"):
                    jira_doc_keys.setdefault(jk.upper(), []).append(chunk)

        for sym in symbols:
            lang = sym_lang(sym)
            fpath = sym_path(sym)
            text = sym_meta_text(sym)

            is_spec = lang in _SPEC_LANGUAGES or fpath.endswith(
                (".proto", ".openapi.yaml", ".openapi.yml", ".openapi.json", ".graphql", ".gql")
            )
            is_data = lang in _DATA_LANGUAGES or fpath.endswith(".sql")
            is_template = lang in _TEMPLATE_LANGUAGES or fpath.endswith(
                (".mjml", ".hbs", ".handlebars", ".j2", ".jinja2", ".po")
            )

            if lang in _FRONTEND_LANGUAGES and sym.symbol_kind in {"class", "function", "hook"}:
                leaf = sym.bare_name or sym.qualified_name.rsplit(".", 1)[-1]
                ui_components_by_name.setdefault(leaf, []).append(sym)
                ui_components_by_name.setdefault(sym.qualified_name, []).append(sym)

            for method, raw_route in _RE_ROUTE_TAG.findall(text):
                norm_route = normalize_route_template(raw_route)
                is_fe = lang in _FRONTEND_LANGUAGES or "builder." in sym.signature_skeleton
                if is_spec:
                    spec_routes.setdefault((method, norm_route), []).append(sym)
                    backend_routes.setdefault((method, norm_route), []).append(sym)
                elif is_fe and lang not in _BACKEND_LANGUAGES:
                    frontend_routes.append((sym, method, norm_route))
                else:
                    backend_routes.setdefault((method, norm_route), []).append(sym)

            for op_type, op_name in _RE_GQL_TAG.findall(text):
                key = (op_type.lower(), op_name.lower())
                if lang in _FRONTEND_LANGUAGES:
                    frontend_gql.append((sym, key[0], key[1]))
                else:
                    backend_gql.setdefault(key, []).append(sym)

            for dto_name, raw_bracket, raw_plain in _RE_DTO_TAG.findall(text):
                raw_fields = (
                    raw_bracket if raw_bracket is not None and raw_bracket != "" else raw_plain
                )
                field_set = {_norm_field(f) for f in raw_fields.split(",") if f.strip()}
                field_set.discard("")
                if is_data:
                    data_tables.append((sym, dto_name, field_set))
                    sql_targets_by_name.setdefault(_norm_dto_name(dto_name), []).append(sym)
                    sql_targets_by_name.setdefault(dto_name.rsplit(".", 1)[-1].lower(), []).append(
                        sym
                    )
                elif is_spec:
                    spec_dtos.append((sym, dto_name, field_set))
                    backend_dtos.append((sym, dto_name, field_set))
                elif lang in _FRONTEND_LANGUAGES:
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
                if (
                    "pubsub" in text.lower()
                    or "kafka" in text.lower()
                    or "sns_topic" in text.lower()
                ):
                    topic_cand = sym.bare_name or sym.qualified_name.rsplit(".", 1)[-1]
                    tf_topics.setdefault(_norm_topic(topic_cand), []).append(sym)

            if sym.symbol_kind == "story":
                for comp_name in _RE_STORY_COMP_TAG.findall(text):
                    story_components.append((sym, comp_name))

            # gRPC service specs vs implementations
            for svc_name in _RE_GRPC_SVC_TAG.findall(text):
                norm_svc = svc_name.lower()
                leaf_svc = svc_name.rsplit(".", 1)[-1].lower()
                if is_spec:
                    grpc_spec_services.setdefault(norm_svc, []).append(sym)
                    grpc_spec_services.setdefault(leaf_svc, []).append(sym)
                else:
                    grpc_impl_symbols.append((sym, norm_svc))

            # AsyncAPI channels & event producers/subscribers
            for ch_name in _RE_CHANNEL_TAG.findall(text):
                async_channels.setdefault(_norm_topic(ch_name), []).append(sym)
            for action_name, op_target in _RE_ASYNC_OP_TAG.findall(text):
                async_channels.setdefault(_norm_topic(op_target), []).append(sym)
                act_low = action_name.lower()
                if act_low in {"send", "publish", "pub"}:
                    event_publishers.setdefault(_norm_topic(op_target), []).append(sym)
                elif act_low in {"receive", "subscribe", "sub"}:
                    event_subscribers.setdefault(_norm_topic(op_target), []).append(sym)
            for pub_topic in _RE_EVENT_PUB_TAG.findall(text):
                event_publishers.setdefault(_norm_topic(pub_topic), []).append(sym)
            for sub_topic in _RE_EVENT_SUB_TAG.findall(text):
                event_subscribers.setdefault(_norm_topic(sub_topic), []).append(sym)

            # SQL/dbt lineage targets & readers/writers
            for model_name in _RE_SQL_MODEL_TAG.findall(text):
                sql_targets_by_name.setdefault(model_name.lower(), []).append(sym)
                sql_targets_by_name.setdefault(_norm_dto_name(model_name), []).append(sym)
            for view_name in _RE_SQL_VIEW_TAG.findall(text):
                sql_targets_by_name.setdefault(view_name.lower(), []).append(sym)
                sql_targets_by_name.setdefault(_norm_dto_name(view_name), []).append(sym)
            for tbl in _RE_DB_WRITE_TAG.findall(text):
                db_writers.append((sym, tbl))
            for tbl in _RE_DB_READ_TAG.findall(text):
                db_readers.append((sym, tbl))

            # i18n keys & templates
            for ikey in _RE_I18N_KEY_TAG.findall(text):
                if is_template:
                    i18n_catalog_by_key.setdefault(ikey.lower(), []).append(sym)
                else:
                    i18n_consumers.append((sym, ikey.lower()))
            if is_template and sym.signature_skeleton.startswith('msgid "'):
                raw_k = sym.bare_name or sym.qualified_name.rsplit(".", 1)[-1]
                if raw_k:
                    i18n_catalog_by_key.setdefault(raw_k.lower(), []).append(sym)

            if is_template and sym.symbol_kind in {"module", "component", "function"}:
                tname = _norm_template_name(fpath or sym.bare_name or sym.qualified_name)
                if tname:
                    template_defs_by_name.setdefault(tname, []).append(sym)
            for tpl_ref in _RE_TEMPLATE_TAG.findall(text):
                template_renderers.append((sym, _norm_template_name(tpl_ref)))

        # 1. Frontend -> Backend & Spec REST routes
        for fe_sym, method, norm_route in frontend_routes:
            targets = backend_routes.get((method, norm_route), []) + backend_routes.get(
                ("ANY", norm_route), []
            )
            for be_sym in targets:
                emit_edge(fe_sym, be_sym, "CALLS")

        # 2. Backend -> OpenAPI/Protobuf spec routes (IMPLEMENTS_SPEC)
        for (method, norm_route), spec_syms in spec_routes.items():
            impls = backend_routes.get((method, norm_route), []) + backend_routes.get(
                ("ANY", norm_route), []
            )
            for impl_sym in impls:
                if sym_lang(impl_sym) in _SPEC_LANGUAGES:
                    continue
                for sp_sym in spec_syms:
                    emit_edge(impl_sym, sp_sym, "IMPLEMENTS_SPEC")

        for impl_sym, svc_key in grpc_impl_symbols:
            for sp_sym in grpc_spec_services.get(svc_key, []) or grpc_spec_services.get(
                svc_key.rsplit(".", 1)[-1], []
            ):
                emit_edge(impl_sym, sp_sym, "IMPLEMENTS_SPEC")

        # 3. GraphQL CALLS
        for fe_sym, op_type, op_name in frontend_gql:
            for be_sym in backend_gql.get((op_type, op_name), []):
                emit_edge(fe_sym, be_sym, "CALLS")

        # 4. DTO parity (Frontend <-> Backend/Spec/SQL)
        backend_dto_by_norm: dict[str, list[ASTSymbolRecord]] = {}
        for be_sym, be_name, be_fields in backend_dtos:
            leaf_be = be_name.rsplit(".", 1)[-1].lower()
            backend_dto_by_norm.setdefault(be_name.lower(), []).append(be_sym)
            backend_dto_by_norm.setdefault(leaf_be, []).append(be_sym)
            backend_dto_by_norm.setdefault(_norm_dto_name(be_name), []).append(be_sym)
            for fe_sym, fe_name, fe_fields in frontend_dtos:
                leaf_fe = fe_name.rsplit(".", 1)[-1].lower()
                exact_match = leaf_fe == leaf_be
                stem_match = _norm_dto_name(fe_name) == _norm_dto_name(be_name)
                overlap = fe_fields & be_fields
                min_fields = min(len(fe_fields), len(be_fields))
                field_match = min_fields > 0 and (len(overlap) * 2 >= min_fields)
                if exact_match or (stem_match and (field_match or not fe_fields or not be_fields)):
                    emit_edge(fe_sym, be_sym, "USES_TYPE")

        for be_sym, be_name, be_fields in backend_dtos:
            if sym_lang(be_sym) in _SPEC_LANGUAGES:
                continue
            for sp_sym, sp_name, sp_fields in spec_dtos:
                exact_match = (
                    be_name.rsplit(".", 1)[-1].lower() == sp_name.rsplit(".", 1)[-1].lower()
                )
                stem_match = _norm_dto_name(be_name) == _norm_dto_name(sp_name)
                overlap = be_fields & sp_fields
                min_fields = min(len(be_fields), len(sp_fields))
                field_match = min_fields > 0 and (len(overlap) * 2 >= min_fields)
                if exact_match or (stem_match and (field_match or not be_fields or not sp_fields)):
                    emit_edge(be_sym, sp_sym, "IMPLEMENTS_SPEC")

        for ref_sym, ref_name in dto_refs:
            for be_sym in backend_dto_by_norm.get(ref_name.lower(), []):
                emit_edge(ref_sym, be_sym, "USES_TYPE")

        # 5. SQL/dbt lineage (READS_TABLE, WRITES_TABLE, DTO -> SQL table)
        for wr_sym, tbl_name in db_writers:
            targets = sql_targets_by_name.get(tbl_name.lower()) or sql_targets_by_name.get(
                _norm_dto_name(tbl_name), []
            )
            for tbl_sym in targets:
                emit_edge(wr_sym, tbl_sym, "WRITES_TABLE")

        for rd_sym, tbl_name in db_readers:
            targets = sql_targets_by_name.get(tbl_name.lower()) or sql_targets_by_name.get(
                _norm_dto_name(tbl_name), []
            )
            for tbl_sym in targets:
                emit_edge(rd_sym, tbl_sym, "READS_TABLE")

        for be_sym, be_name, _be_fields in backend_dtos:
            for tbl_sym, tbl_name, _tbl_cols in data_tables:
                if _norm_dto_name(be_name) == _norm_dto_name(tbl_name):
                    emit_edge(be_sym, tbl_sym, "USES_TYPE")

        # 6. AsyncAPI / Event Bus (PUBLISHES_EVENT, SUBSCRIBES_EVENT, DISPATCHES, PROVISIONED_BY)
        for topic_key, pub_syms in event_publishers.items():
            for pub_sym in pub_syms:
                for ch_sym in async_channels.get(topic_key, []):
                    emit_edge(pub_sym, ch_sym, "PUBLISHES_EVENT")
                for sub_sym in event_subscribers.get(topic_key, []):
                    emit_edge(pub_sym, sub_sym, "DISPATCHES")
                for tf_sym in tf_topics.get(topic_key, []):
                    emit_edge(pub_sym, tf_sym, "PROVISIONED_BY")

        for topic_key, sub_syms in event_subscribers.items():
            for sub_sym in sub_syms:
                for ch_sym in async_channels.get(topic_key, []):
                    emit_edge(sub_sym, ch_sym, "SUBSCRIBES_EVENT")

        # 7. i18n keys & Templates (USES_TYPE / RENDERS_TEMPLATE)
        for cons_sym, ikey in i18n_consumers:
            for cat_sym in i18n_catalog_by_key.get(ikey, []):
                emit_edge(cons_sym, cat_sym, "USES_TYPE")

        for rnd_sym, tpl_key in template_renderers:
            for tpl_sym in template_defs_by_name.get(tpl_key, []):
                emit_edge(rnd_sym, tpl_sym, "RENDERS_TEMPLATE")

        # 8. Storybook -> UI Component
        for story_sym, comp_name in story_components:
            candidates = ui_components_by_name.get(comp_name) or ui_components_by_name.get(
                comp_name.rsplit(".", 1)[-1], []
            )
            for comp_sym in candidates:
                emit_edge(story_sym, comp_sym, "DOCUMENTS_COMPONENT")

        # 9. Terraform Env & Service PROVISIONED_BY
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

        # 10. Jira ticket references across code symbols -> Jira/Spec peer symbols
        if jira_doc_keys:
            symbols_by_jira: dict[str, list[ASTSymbolRecord]] = {}
            for sym in symbols:
                for jk in _RE_JIRA_KEY.findall(sym_meta_text(sym)):
                    jk_up = jk.upper()
                    if jk_up in jira_doc_keys:
                        symbols_by_jira.setdefault(jk_up, []).append(sym)
            for _jk, syms_for_issue in symbols_by_jira.items():
                if len(syms_for_issue) > 1:
                    anchor = syms_for_issue[0]
                    for peer in syms_for_issue[1:]:
                        emit_edge(anchor, peer, "IMPLEMENTS_ISSUE")

        return edges
