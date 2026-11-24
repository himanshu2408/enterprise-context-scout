"""Living Repository Wiki Builder synthesizing enclave-isolated Markdown docs (M2-03)."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_DNS, UUID, uuid5

from context_scout.db.schema import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
    SideEffectRecord,
)
from context_scout.retrieval.embedder import get_embedding_provider
from context_scout.schemas.contracts import DEFAULT_TENANT_ID

logger = logging.getLogger(__name__)

_ROUTE_RE = re.compile(r"@route\s+(GET|POST|PUT|DELETE|PATCH|ANY)\s+([^\s]+)")
_GRPC_RE = re.compile(r"@grpc_service\s+([^\s]+)")
_CHANNEL_RE = re.compile(r"@channel\s+([^\s]+)")
_ENV_RE = re.compile(r"@env\s+([A-Za-z0-9_]+)")


def _enclave_slug(scopes: tuple[str, ...]) -> str:
    joined = "|".join(scopes)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def _sanitize_mermaid_id(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    if not cleaned or cleaned[0].isdigit():
        return f"n_{cleaned}"
    return cleaned


class RepoWikiBuilder:
    """Synthesizes living repository wiki chunks grouped strictly by ACL scope enclaves."""

    def build_wiki_chunks(
        self,
        *,
        domain_id: str,
        repo_name: str,
        code_files: Sequence[CodeFileRecord],
        symbols: Sequence[ASTSymbolRecord] = (),
        edges: Sequence[ASTEdgeRecord] = (),
        side_effects: Sequence[SideEffectRecord] = (),
        commit_sha: str = "HEAD",
        tenant_id: UUID = DEFAULT_TENANT_ID,
        embed: bool = True,
    ) -> list[DocChunkRecord]:
        """Builds living Markdown wiki DocChunkRecords isolated by allowed_scopes enclaves."""
        repo_files = [
            cf
            for cf in code_files
            if cf.repo_name == repo_name and cf.domain_id == domain_id and cf.tenant_id == tenant_id
        ]
        if not repo_files:
            return []

        file_by_id = {cf.file_id: cf for cf in repo_files}
        enclave_files: dict[tuple[str, ...], list[CodeFileRecord]] = {}
        for cf in repo_files:
            scope_key = tuple(sorted(set(s.strip() for s in cf.allowed_scopes if s and s.strip())))
            if not scope_key:
                continue
            enclave_files.setdefault(scope_key, []).append(cf)

        if not enclave_files:
            return []

        enclave_syms: dict[tuple[str, ...], list[ASTSymbolRecord]] = {}
        sym_by_id: dict[UUID, ASTSymbolRecord] = {}
        for sym in symbols:
            if sym.domain_id != domain_id or sym.tenant_id != tenant_id:
                continue
            sym_cf = file_by_id.get(sym.file_id)
            if sym_cf is None:
                continue
            sym_by_id[sym.symbol_id] = sym
            scope_key = tuple(
                sorted(set(s.strip() for s in sym.allowed_scopes if s and s.strip()))
            ) or tuple(sorted(set(s.strip() for s in sym_cf.allowed_scopes if s and s.strip())))
            enclave_syms.setdefault(scope_key, []).append(sym)

        enclave_se: dict[tuple[str, ...], list[SideEffectRecord]] = {}
        for se in side_effects:
            if se.repo_name != repo_name or se.domain_id != domain_id or se.tenant_id != tenant_id:
                continue
            scope_key = tuple(sorted(set(s.strip() for s in se.allowed_scopes if s and s.strip())))
            if not scope_key:
                continue
            enclave_se.setdefault(scope_key, []).append(se)

        enclave_edges: dict[tuple[str, ...], list[ASTEdgeRecord]] = {}
        for edge in edges:
            if edge.domain_id != domain_id or edge.tenant_id != tenant_id:
                continue
            src = sym_by_id.get(edge.source_symbol_id)
            if src is None:
                continue
            scope_key = tuple(
                sorted(set(s.strip() for s in edge.allowed_scopes if s and s.strip()))
            )
            if not scope_key:
                continue
            enclave_edges.setdefault(scope_key, []).append(edge)

        pending_sections: list[
            tuple[str, tuple[str, ...], str, str, str, str, dict[str, object]]
        ] = []
        today = datetime.now(UTC).date()
        ver_str = commit_sha[:12] if commit_sha else "HEAD"

        for scope_key in sorted(enclave_files.keys()):
            files_in_enclave = enclave_files[scope_key]
            syms_in_enclave = enclave_syms.get(scope_key, [])
            se_in_enclave = enclave_se.get(scope_key, [])
            edges_in_enclave = enclave_edges.get(scope_key, [])
            slug = _enclave_slug(scope_key)
            doc_id = f"repo_wiki:{repo_name}:{slug}"
            source_uri = f"urn:repo_wiki:{domain_id}:{repo_name}:{slug}"
            doc_title = f"Repository Wiki: {repo_name} ({slug})"

            # Section 1: Architecture, Language Topology, Ownership & Mermaid Module Graph
            lang_counts: dict[str, int] = {}
            for cf in files_in_enclave:
                lang_counts[cf.language] = lang_counts.get(cf.language, 0) + 1
            lang_lines = [f"- `{lang}`: {cnt} file(s)" for lang, cnt in sorted(lang_counts.items())]

            ownership_lines: list[str] = []
            mermaid_edges: list[str] = []
            for e in edges_in_enclave:
                src_s = sym_by_id.get(e.source_symbol_id)
                src_name = src_s.qualified_name if src_s is not None else str(e.source_symbol_id)
                if e.edge_kind in {"OWNED_BY_TEAM", "PART_OF_SYSTEM"}:
                    ownership_lines.append(
                        f"- `{src_name}` --({e.edge_kind})--> `{e.target_qualified_name}`"
                    )
                if len(mermaid_edges) < 30:
                    s_id = _sanitize_mermaid_id(src_name)
                    t_id = _sanitize_mermaid_id(e.target_qualified_name)
                    mermaid_edges.append(
                        f'  {s_id}["{src_name}"] -->|{e.edge_kind}| '
                        f'{t_id}["{e.target_qualified_name}"]'
                    )

            arch_parts = [
                f"## Architecture & Ownership ({repo_name} @ {ver_str})",
                "",
                f"Indexed {len(files_in_enclave)} file(s), {len(syms_in_enclave)} AST symbol(s), "
                f"and {len(edges_in_enclave)} dependency edge(s).",
                "",
                "### Language Breakdown",
                *(lang_lines or ["- None recorded"]),
                "",
                "### Ownership & System Mapping",
                *(ownership_lines or ["- No explicit CODEOWNERS / Backstage edges recorded"]),
            ]
            if mermaid_edges:
                arch_parts.extend(
                    [
                        "",
                        "### Module Dependency Graph",
                        "```mermaid",
                        "graph LR",
                        *mermaid_edges,
                        "```",
                    ]
                )
            arch_md = "\n".join(arch_parts).strip()
            pending_sections.append(
                (
                    "arch",
                    scope_key,
                    doc_id,
                    doc_title,
                    source_uri,
                    "Architecture & Ownership",
                    {
                        "repo_name": repo_name,
                        "commit_sha": commit_sha,
                        "enclave_slug": slug,
                        "section_slug": "arch",
                        "wiki_content": arch_md,
                    },
                )
            )

            # Section 2: Cross-Boundary API, gRPC, & Event Contracts
            contract_lines: list[str] = []
            for sym in syms_in_enclave:
                cf_opt = file_by_id.get(sym.file_id)
                loc = (
                    f"{cf_opt.file_path}:L{sym.start_line}-L{sym.end_line}"
                    if cf_opt is not None
                    else sym.qualified_name
                )
                meta_txt = f"{sym.contextual_summary}\n{sym.signature_skeleton}"
                for method, path in _ROUTE_RE.findall(meta_txt):
                    contract_lines.append(
                        f"- REST/RPC `{method} {path}` -> `{sym.qualified_name}` (`{loc}`)"
                    )
                for gsvc in _GRPC_RE.findall(meta_txt):
                    contract_lines.append(f"- gRPC Service `{gsvc}` (`{loc}`)")
                for ch in _CHANNEL_RE.findall(meta_txt):
                    contract_lines.append(f"- Async Channel `{ch}` (`{loc}`)")
            for e in edges_in_enclave:
                if e.is_cross_repo or e.edge_kind in {
                    "CALLS",
                    "IMPLEMENTS_SPEC",
                    "PUBLISHES_EVENT",
                    "SUBSCRIBES_EVENT",
                    "DISPATCHES",
                    "DOCUMENTS_COMPONENT",
                    "IMPLEMENTS_ISSUE",
                }:
                    src_s = sym_by_id.get(e.source_symbol_id)
                    src_name = (
                        src_s.qualified_name if src_s is not None else str(e.source_symbol_id)
                    )
                    contract_lines.append(
                        f"- Edge `{e.edge_kind}`: `{src_name}` -> `{e.target_qualified_name}`"
                    )
            contracts_md = "\n".join(
                [
                    f"## Cross-Boundary Contracts ({repo_name})",
                    "",
                    *(
                        list(dict.fromkeys(contract_lines))[:80]
                        or ["- No external HTTP/gRPC/Event contracts detected in this enclave."]
                    ),
                ]
            )
            pending_sections.append(
                (
                    "contracts",
                    scope_key,
                    doc_id,
                    doc_title,
                    source_uri,
                    "Cross-Boundary Contracts",
                    {
                        "repo_name": repo_name,
                        "commit_sha": commit_sha,
                        "enclave_slug": slug,
                        "section_slug": "contracts",
                        "wiki_content": contracts_md,
                    },
                )
            )

            # Section 3: Side-Effect & Operational Risk Matrix
            se_rows: list[str] = []
            for se in sorted(
                se_in_enclave, key=lambda x: (x.effect_category, x.file_path, x.start_line)
            ):
                safe_tgt = se.target_identifier.replace("|", "/")
                safe_drv = (se.provider_or_driver or "-").replace("|", "/")
                se_rows.append(
                    f"| `{se.effect_category}` | `{safe_tgt}` | `{safe_drv}` | "
                    f"`{se.severity_tier}` | `{se.qualified_symbol}` | "
                    f"`{se.file_path}:L{se.start_line}-L{se.end_line}` |"
                )
            se_md_lines = [
                f"## Side-Effect Catalog ({repo_name})",
                "",
            ]
            if se_rows:
                se_md_lines.extend(
                    [
                        "| Category | Target | Driver | Severity | Symbol | Location |",
                        "|---|---|---|---|---|---|",
                        *se_rows,
                    ]
                )
            else:
                se_md_lines.append("No external side effects detected in this enclave.")
            se_md = "\n".join(se_md_lines)
            pending_sections.append(
                (
                    "side_effects",
                    scope_key,
                    doc_id,
                    doc_title,
                    source_uri,
                    "Side-Effect Catalog",
                    {
                        "repo_name": repo_name,
                        "commit_sha": commit_sha,
                        "enclave_slug": slug,
                        "section_slug": "side_effects",
                        "side_effect_count": len(se_in_enclave),
                        "wiki_content": se_md,
                    },
                )
            )

            # Section 4: Data Lineage & Infrastructure Provisioning
            lineage_lines: list[str] = []
            for se in se_in_enclave:
                if se.effect_category == "db_write":
                    lineage_lines.append(
                        f"- DB Write `{se.target_identifier}` via `{se.provider_or_driver}` "
                        f"(`{se.file_path}:L{se.start_line}`)"
                    )
            for e in edges_in_enclave:
                if e.edge_kind in {"READS_TABLE", "WRITES_TABLE", "PROVISIONED_BY"}:
                    src_s = sym_by_id.get(e.source_symbol_id)
                    src_name = (
                        src_s.qualified_name if src_s is not None else str(e.source_symbol_id)
                    )
                    lineage_lines.append(
                        f"- `{e.edge_kind}`: `{src_name}` -> `{e.target_qualified_name}`"
                    )
            for sym in syms_in_enclave:
                for env_k in _ENV_RE.findall(f"{sym.contextual_summary}\n{sym.signature_skeleton}"):
                    lineage_lines.append(
                        f"- Config Env `{env_k}` referenced by `{sym.qualified_name}`"
                    )
            lineage_md = "\n".join(
                [
                    f"## Data Lineage & Infrastructure ({repo_name})",
                    "",
                    *(
                        list(dict.fromkeys(lineage_lines))[:80]
                        or ["- No database writes or Terraform provisioning links detected."]
                    ),
                ]
            )
            pending_sections.append(
                (
                    "lineage_infra",
                    scope_key,
                    doc_id,
                    doc_title,
                    source_uri,
                    "Data Lineage & Infrastructure",
                    {
                        "repo_name": repo_name,
                        "commit_sha": commit_sha,
                        "enclave_slug": slug,
                        "section_slug": "lineage_infra",
                        "wiki_content": lineage_md,
                    },
                )
            )

        headers: list[str] = []
        embed_inputs: list[str] = []
        for _sec_slug, _scope_key, _doc_id, _title, _uri, sec_heading, meta in pending_sections:
            content = str(meta.pop("wiki_content"))
            meta["wiki_body"] = content
            hdr = (
                f"This living repository wiki documents '{repo_name}' "
                f"(Commit: {ver_str}, Domain: {domain_id}, Section: '{sec_heading}', "
                f"Enclave: {meta['enclave_slug']})."
            )
            headers.append(hdr)
            embed_inputs.append(f"{hdr}\n{content}")

        if embed and embed_inputs:
            embeddings = get_embedding_provider().embed_batch(embed_inputs, is_query=False)
        else:
            dim = get_embedding_provider().dimensions
            val = 1.0 / (dim**0.5)
            embeddings = [[val] * dim for _ in embed_inputs]

        records: list[DocChunkRecord] = []
        for idx, (
            sec_slug,
            scope_key,
            doc_id,
            doc_title,
            source_uri,
            sec_heading,
            meta,
        ) in enumerate(pending_sections):
            content = str(meta.pop("wiki_body"))
            chunk_id = uuid5(
                NAMESPACE_DNS,
                f"{tenant_id}:{domain_id}:{doc_id}:{ver_str}:{sec_slug}",
            )
            records.append(
                DocChunkRecord(
                    chunk_id=chunk_id,
                    domain_id=domain_id,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    source_uri=source_uri,
                    market_scope=None,
                    published_date=today,
                    authority_weight=1.1,
                    section_heading=sec_heading,
                    contextual_header=headers[idx],
                    verbatim_content=content,
                    metadata_json=meta,
                    allowed_scopes=list(scope_key),
                    embedding=embeddings[idx],
                    source_type="repo_wiki",
                    version_str=ver_str,
                    is_stale=False,
                    stale_reason=None,
                    tenant_id=tenant_id,
                )
            )
        return records

    def sync_repo_wiki_to_repository(
        self,
        *,
        domain_id: str,
        repo_name: str,
        repository: Any,
        conn: Any = None,
        principal: Any = None,
        code_files: Sequence[CodeFileRecord] | None = None,
        symbols: Sequence[ASTSymbolRecord] | None = None,
        edges: Sequence[ASTEdgeRecord] | None = None,
        side_effects: Sequence[SideEffectRecord] | None = None,
        commit_sha: str = "HEAD",
        tenant_id: UUID = DEFAULT_TENANT_ID,
    ) -> list[DocChunkRecord]:
        """Builds living repo wiki chunks, supersedes older commits, and persists to repo."""
        is_pg = hasattr(repository, "insert_doc_chunks_batch") and conn is not None

        if code_files is None:
            if is_pg:
                if hasattr(repository, "list_visible_code_file_records"):
                    code_files = repository.list_visible_code_file_records(
                        conn, domain_id, repo_name=repo_name
                    )
                else:
                    code_files = []
            elif principal is not None and hasattr(repository, "list_visible_code_file_records"):
                code_files = repository.list_visible_code_file_records(
                    principal, domain_id, repo_name=repo_name
                )
            elif principal is not None and hasattr(repository, "list_visible_code_files"):
                code_files = [
                    f
                    for f in repository.list_visible_code_files(principal, domain_id)
                    if f.repo_name == repo_name
                ]
            else:
                code_files = []

        if symbols is None:
            if is_pg:
                symbols = (
                    repository.list_visible_ast_symbol_records(conn, domain_id)
                    if hasattr(repository, "list_visible_ast_symbol_records")
                    else []
                )
            elif principal is not None and hasattr(repository, "list_visible_ast_symbol_records"):
                symbols = repository.list_visible_ast_symbol_records(principal, domain_id)
            elif principal is not None and hasattr(repository, "list_visible_ast_symbols"):
                symbols = repository.list_visible_ast_symbols(principal, domain_id)
            else:
                symbols = []

        if edges is None:
            if is_pg:
                edges = (
                    repository.list_visible_ast_edge_records(conn, domain_id)
                    if hasattr(repository, "list_visible_ast_edge_records")
                    else []
                )
            elif principal is not None and hasattr(repository, "list_visible_ast_edge_records"):
                edges = repository.list_visible_ast_edge_records(principal, domain_id)
            elif principal is not None and hasattr(repository, "list_visible_ast_edges"):
                edges = repository.list_visible_ast_edges(principal, domain_id)
            else:
                edges = []

        if side_effects is None:
            if is_pg and hasattr(repository, "list_visible_side_effects"):
                side_effects = repository.list_visible_side_effects(
                    conn, domain_id, repo_name=repo_name
                )
            elif principal is not None and hasattr(repository, "list_visible_side_effects"):
                side_effects = repository.list_visible_side_effects(
                    principal, domain_id, repo_name=repo_name
                )
            else:
                side_effects = []

        chunks = self.build_wiki_chunks(
            domain_id=domain_id,
            repo_name=repo_name,
            code_files=code_files,
            symbols=symbols,
            edges=edges,
            side_effects=side_effects,
            commit_sha=commit_sha,
            tenant_id=tenant_id,
            embed=True,
        )
        if not chunks:
            return []

        doc_ids = sorted({c.doc_id for c in chunks})
        ver_str = chunks[0].version_str

        if is_pg:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM doc_chunks
                    WHERE domain_id = %s AND doc_id = ANY(%s) AND version_str = %s;
                    """,
                    (domain_id, doc_ids, ver_str),
                )
                cur.execute(
                    """
                    UPDATE doc_chunks
                    SET is_stale = TRUE,
                        stale_reason = %s
                    WHERE domain_id = %s
                      AND doc_id = ANY(%s)
                      AND version_str <> %s
                      AND NOT is_stale;
                    """,
                    (f"Superseded by commit {ver_str}", domain_id, doc_ids, ver_str),
                )
            repository.insert_doc_chunks_batch(conn, chunks)
        else:
            if hasattr(repository, "_docs"):
                for cid, existing in list(repository._docs.items()):
                    if (
                        existing.tenant_id == tenant_id
                        and existing.domain_id == domain_id
                        and existing.doc_id in doc_ids
                    ):
                        if existing.version_str == ver_str:
                            repository._docs.pop(cid, None)
                        elif not existing.is_stale:
                            repository._docs[cid] = dataclasses.replace(
                                existing,
                                is_stale=True,
                                stale_reason=f"Superseded by commit {ver_str}",
                            )
            if hasattr(repository, "insert_doc_chunks_batch") and principal is not None:
                repository.insert_doc_chunks_batch(principal, chunks)
            elif hasattr(repository, "insert_doc_chunk") and principal is not None:
                for chk in chunks:
                    repository.insert_doc_chunk(principal, chk)
            elif hasattr(repository, "_docs"):
                for chk in chunks:
                    repository._docs[chk.chunk_id] = chk

        return chunks


__all__ = ["RepoWikiBuilder"]
