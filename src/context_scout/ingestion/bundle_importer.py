"""Offline Air-Gapped Bundle Importer (`context-scout import-bundle`).

Safely ingests signed/hashed `.tar.gz` / `.tgz` / `.tar` archives or local
directories containing a ``manifest.json``, enforcing path-traversal, symlink,
hardlink, archive-size bomb, and SHA-256 integrity checks before running
pre-index secret/PII scrubbing, Docling extraction, Tree-sitter AST/side-effect
extraction, cross-boundary contract linking, living repo wiki generation,
object-store archiving, and bundle watermark recording.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import posixpath
import tarfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import NAMESPACE_DNS, UUID, uuid4, uuid5

from context_scout.ast_engine.contract_linker import CrossBoundaryContractLinker
from context_scout.ast_engine.extractor import ASTExtractor
from context_scout.ast_engine.side_effect_extractor import SideEffectExtractor
from context_scout.db.repositories import InMemoryRLSRepository, PostgresRLSRepository
from context_scout.db.schema import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    ConnectorWatermarkRecord,
    DocChunkRecord,
    RepositoryRecord,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.ingestion.doc_parser import chunk_custom_domain_document
from context_scout.ingestion.docling_extractor import DocExtractorSPI, get_doc_extractor
from context_scout.ingestion.git_merkle_indexer import compute_git_blob_sha
from context_scout.ingestion.repo_wiki_builder import RepoWikiBuilder
from context_scout.plugins.registry import PluginRegistry
from context_scout.retrieval.embedder import get_embedding_provider
from context_scout.schemas.contracts import DEFAULT_TENANT_ID, PrincipalContext, RoleType
from context_scout.security.pre_index_scrubber import PreIndexScrubber
from context_scout.storage.object_store import (
    ArtifactObjectStore,
    build_tenant_domain_key,
    get_artifact_object_store,
)

logger = logging.getLogger(__name__)

_RICH_DOC_EXTENSIONS = frozenset({".pdf", ".docx", ".pptx", ".html", ".htm"})
_TEXT_DOC_EXTENSIONS = frozenset({".md", ".markdown", ".txt", ".rst", ".tex", ".adoc"})


class BundleSecurityError(ValueError):
    """Raised when an offline bundle violates path, link, ACL, or size invariants."""


class BundleIntegrityError(ValueError):
    """Raised when bundle manifest structure or SHA-256 verification fails."""


@dataclass(frozen=True)
class BundleImportReport:
    """Structured summary emitted after an air-gapped bundle import."""

    bundle_id: str
    domain_id: str
    tenant_id: str
    repositories_imported: list[str]
    code_files_indexed: int
    ast_symbols_indexed: int
    ast_edges_created: int
    side_effects_indexed: int
    doc_chunks_indexed: int
    wiki_chunks_created: int
    scrubbed_findings_count: int
    manifest_uri: str
    manifest_sha256: str
    duration_ms: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_bundle_relpath(raw_path: str) -> str:
    """Validate and normalize an archive/bundle member path (fail-closed against traversal)."""
    if not raw_path or "\x00" in raw_path or "\\" in raw_path:
        raise BundleSecurityError(f"Invalid bundle member path: {raw_path!r}")
    stripped = raw_path.strip()
    if stripped.startswith("/") or (len(stripped) >= 2 and stripped[1] == ":"):
        raise BundleSecurityError(f"Absolute bundle member path rejected: {raw_path!r}")
    # Allow leading './' only before strict component validation
    while stripped.startswith("./"):
        stripped = stripped[2:]
    if not stripped:
        raise BundleSecurityError(f"Empty normalized bundle path from {raw_path!r}")
    parts = PurePosixPath(stripped).parts
    if any(part in ("", ".", "..") for part in parts):
        raise BundleSecurityError(f"Path traversal segment rejected in bundle: {raw_path!r}")
    normalized = posixpath.normpath(stripped)
    if normalized.startswith(("/", "..")):
        raise BundleSecurityError(f"Normalized path escapes bundle root: {raw_path!r}")
    return normalized


def _infer_language(file_path: str, explicit_lang: str | None = None) -> str:
    if explicit_lang and explicit_lang.strip():
        return explicit_lang.strip().lower()
    lang_plugin = PluginRegistry.get_plugin_for_file(file_path)
    if lang_plugin is not None:
        return lang_plugin.language_id
    low = file_path.lower()
    ext_map = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".js": "javascript",
        ".jsx": "tsx",
        ".go": "go",
        ".java": "java",
        ".kt": "kotlin",
        ".rs": "rust",
        ".cs": "csharp",
        ".sql": "sql",
        ".tf": "hcl",
        ".hcl": "hcl",
        ".yaml": "yaml",
        ".yml": "yaml",
    }
    for ext, lang in ext_map.items():
        if low.endswith(ext):
            return lang
    return "text"


def _is_doc_entry(rel_path: str, explicit_kind: str | None) -> bool:
    if explicit_kind:
        k = explicit_kind.strip().lower()
        if k in {"doc", "docs", "document", "documentation"}:
            return True
        if k in {"code", "source"}:
            return False
    suffix = Path(rel_path).suffix.lower()
    return suffix in (_RICH_DOC_EXTENSIONS | _TEXT_DOC_EXTENSIONS)


class AirGappedBundleImporter:
    """Verified air-gapped archive & directory importer for Enterprise Context Scout."""

    def __init__(
        self,
        *,
        repository: Any | None = None,
        connection_manager: RLSConnectionManager | None = None,
        zoekt_index: Any | None = None,
        object_store: ArtifactObjectStore | None = None,
        doc_extractor: DocExtractorSPI | None = None,
        scrubber: PreIndexScrubber | None = None,
        max_files: int = 5000,
        max_total_bytes: int = 250_000_000,
        max_total_uncompressed_bytes: int | None = None,
        max_member_bytes: int = 100_000_000,
    ) -> None:
        self.repository: Any = repository or InMemoryRLSRepository()
        self.connection_manager = connection_manager
        self.zoekt_index = zoekt_index
        self.object_store = object_store or get_artifact_object_store()
        self.doc_extractor = doc_extractor or get_doc_extractor()
        self.scrubber = scrubber or PreIndexScrubber()
        self.max_files = max(1, max_files)
        effective_max_bytes = (
            max_total_uncompressed_bytes
            if max_total_uncompressed_bytes is not None
            else max_total_bytes
        )
        self.max_total_bytes = max(1024, effective_max_bytes)
        self.max_member_bytes = max(1024, min(max_member_bytes, self.max_total_bytes))

    def _read_bundle_payloads(self, bundle_path: Path) -> dict[str, bytes]:
        if not bundle_path.exists():
            raise FileNotFoundError(f"Bundle path does not exist: {bundle_path}")

        files_map: dict[str, bytes] = {}
        total_bytes = 0

        if bundle_path.is_dir():
            if bundle_path.is_symlink():
                raise BundleSecurityError("Symlinked bundle root directory rejected.")
            root_resolved = bundle_path.resolve()
            file_count = 0
            for dirpath, dirnames, filenames in os.walk(bundle_path, followlinks=False):
                current_dir = Path(dirpath)
                # Reject symlinks inside directory tree
                for dname in list(dirnames):
                    sub = current_dir / dname
                    if sub.is_symlink():
                        raise BundleSecurityError(f"Symlink directory rejected in bundle: {sub}")
                for fname in filenames:
                    fpath = current_dir / fname
                    if fpath.is_symlink():
                        raise BundleSecurityError(f"Symlink file rejected in bundle: {fpath}")
                    resolved = fpath.resolve()
                    if not resolved.is_relative_to(root_resolved):
                        raise BundleSecurityError(f"Path escape rejected in bundle: {fpath}")
                    rel = fpath.relative_to(bundle_path).as_posix()
                    norm_rel = _normalize_bundle_relpath(rel)
                    file_count += 1
                    if file_count > self.max_files:
                        raise BundleSecurityError(
                            f"Bundle exceeds max_files limit ({self.max_files})"
                        )
                    raw = fpath.read_bytes()
                    total_bytes += len(raw)
                    if total_bytes > self.max_total_bytes:
                        raise BundleSecurityError(
                            f"Bundle exceeds max_total_bytes limit ({self.max_total_bytes})"
                        )
                    files_map[norm_rel] = raw
            return files_map

        try:
            with tarfile.open(bundle_path, mode="r:*") as tf:
                file_count = 0
                for member in tf.getmembers():
                    if member.issym() or member.islnk():
                        raise BundleSecurityError(
                            f"Symlink/hardlink entry rejected in bundle archive: {member.name!r}"
                        )
                    if not (member.isreg() or member.isdir()):
                        raise BundleSecurityError(
                            f"Non-regular file entry rejected in bundle archive: {member.name!r}"
                        )
                    if member.isdir():
                        _normalize_bundle_relpath(
                            member.name.rstrip("/") or "."
                        ) if member.name not in (".", "./") else None
                        continue
                    try:
                        filtered = tarfile.data_filter(member, "/bundle")
                    except tarfile.FilterError as exc:
                        raise BundleSecurityError(
                            f"Tar data_filter rejected archive member {member.name!r}: {exc}"
                        ) from exc
                    norm_rel = _normalize_bundle_relpath(filtered.name)
                    file_count += 1
                    if file_count > self.max_files:
                        raise BundleSecurityError(
                            f"Bundle archive exceeds max_files limit ({self.max_files})"
                        )
                    if (
                        filtered.size < 0
                        or filtered.size > self.max_member_bytes
                        or total_bytes + filtered.size > self.max_total_bytes
                    ):
                        raise BundleSecurityError(
                            f"Bundle archive exceeds byte limit ({self.max_total_bytes})"
                        )
                    extracted = tf.extractfile(filtered)
                    if extracted is None:
                        raise BundleIntegrityError(f"Failed reading archive member '{member.name}'")
                    read_cap = min(self.max_member_bytes, self.max_total_bytes - total_bytes)
                    raw = extracted.read(read_cap + 1)
                    if len(raw) > self.max_member_bytes:
                        raise BundleSecurityError(
                            f"Bundle archive member '{norm_rel}' exceeds "
                            f"max_member_bytes ({self.max_member_bytes})"
                        )
                    total_bytes += len(raw)
                    if total_bytes > self.max_total_bytes:
                        raise BundleSecurityError(
                            f"Bundle archive exceeds max_total_bytes limit ({self.max_total_bytes})"
                        )
                    files_map[norm_rel] = raw
        except tarfile.TarError as exc:
            raise BundleIntegrityError(
                f"Invalid or unreadable bundle archive '{bundle_path}': {exc}"
            ) from exc

        return files_map

    def import_bundle(
        self,
        bundle_path: Path | str,
        *,
        domain_id: str,
        tenant_id: UUID | None = None,
        default_allowed_scopes: Sequence[str] = (),
        principal: PrincipalContext | None = None,
    ) -> BundleImportReport:
        """Validate, scrub, extract, index, and archive an offline air-gapped bundle."""
        t0 = time.perf_counter()
        resolved_path = Path(bundle_path)
        files_map = self._read_bundle_payloads(resolved_path)

        manifest_bytes = files_map.get("manifest.json")
        if manifest_bytes is None:
            raise BundleIntegrityError("Bundle missing required root 'manifest.json'")

        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BundleIntegrityError(f"Malformed manifest.json: {exc}") from exc

        if not isinstance(manifest, dict):
            raise BundleIntegrityError("manifest.json root must be a JSON object")

        bundle_id = str(manifest.get("bundle_id") or "").strip()
        if not bundle_id:
            raise BundleIntegrityError("manifest.json missing non-empty 'bundle_id'")

        manifest_domain = str(manifest.get("domain_id") or "").strip()
        if manifest_domain and manifest_domain != domain_id:
            raise BundleIntegrityError(
                f"Bundle domain_id mismatch: manifest='{manifest_domain}' "
                f"!= requested='{domain_id}'"
            )

        manifest_tid_raw = manifest.get("tenant_id")
        manifest_tid = UUID(str(manifest_tid_raw)) if manifest_tid_raw else None
        if manifest_tid is not None and tenant_id is not None and manifest_tid != tenant_id:
            raise BundleSecurityError(
                f"Tenant mismatch between bundle manifest ({manifest_tid}) and caller ({tenant_id})"
            )

        eff_tid = tenant_id or (
            principal.tenant_id if principal is not None else (manifest_tid or DEFAULT_TENANT_ID)
        )
        manifest_default_scopes = [
            str(s).strip()
            for s in (manifest.get("allowed_scopes") or default_allowed_scopes or ())
            if str(s).strip()
        ]

        raw_entries = manifest.get("files")
        if not isinstance(raw_entries, list) or not raw_entries:
            raise BundleIntegrityError("manifest.json must contain a non-empty 'files' array")

        validated_entries: list[dict[str, Any]] = []
        all_entry_scopes: set[str] = set(manifest_default_scopes)

        for entry in raw_entries:
            if not isinstance(entry, Mapping):
                raise BundleIntegrityError("Each manifest files[] entry must be an object")
            rel = _normalize_bundle_relpath(str(entry.get("path") or ""))
            expected_sha = str(entry.get("sha256") or "").strip().lower()
            if len(expected_sha) != 64:
                raise BundleIntegrityError(f"Manifest entry '{rel}' missing 64-char sha256 digest")
            raw_bytes = files_map.get(rel)
            if raw_bytes is None:
                raise BundleIntegrityError(f"Manifest references missing bundle file '{rel}'")
            actual_sha = hashlib.sha256(raw_bytes).hexdigest().lower()
            if actual_sha != expected_sha:
                raise BundleIntegrityError(
                    f"SHA-256 mismatch for '{rel}': expected={expected_sha}, actual={actual_sha}"
                )
            entry_scopes = [
                str(s).strip()
                for s in (entry.get("allowed_scopes") or manifest_default_scopes)
                if str(s).strip()
            ]
            if not entry_scopes:
                raise BundleSecurityError(f"Fail-closed: entry '{rel}' has empty allowed_scopes")
            all_entry_scopes.update(entry_scopes)
            validated_entries.append(
                {
                    "rel_path": rel,
                    "raw_bytes": raw_bytes,
                    "sha256": actual_sha,
                    "kind": str(entry.get("kind") or ""),
                    "language": str(entry.get("language") or ""),
                    "repo_name": str(
                        entry.get("repo_name") or manifest.get("repo_name") or f"bundle-{bundle_id}"
                    ),
                    "commit_sha": str(
                        entry.get("commit_sha") or manifest.get("commit_sha") or actual_sha[:12]
                    ),
                    "branch": str(entry.get("branch") or manifest.get("default_branch") or "main"),
                    "allowed_scopes": entry_scopes,
                }
            )

        eff_principal = principal or PrincipalContext(
            user_id="airgap_bundle_importer",
            role=RoleType.ENGINEER,
            allowed_scopes=sorted(all_entry_scopes) or ["public"],
            tenant_id=eff_tid,
        )

        extractor = ASTExtractor()
        se_extractor = SideEffectExtractor()
        embedder = get_embedding_provider()

        code_files_by_repo: dict[str, list[CodeFileRecord]] = {}
        symbols_by_repo: dict[str, list[ASTSymbolRecord]] = {}
        edges_by_repo: dict[str, list[ASTEdgeRecord]] = {}
        side_effects_by_repo: dict[str, list[Any]] = {}
        repos_meta: dict[str, tuple[str, str, set[str]]] = {}
        doc_chunks_all: list[DocChunkRecord] = []
        warnings: list[str] = []
        scrubbed_findings = 0

        for item in validated_entries:
            rel_path = item["rel_path"]
            raw_bytes = item["raw_bytes"]
            scopes = item["allowed_scopes"]

            if _is_doc_entry(rel_path, item["kind"]):
                suffix = Path(rel_path).suffix.lower()
                if suffix in _RICH_DOC_EXTENSIONS:
                    extracted_doc = self.doc_extractor.extract(
                        raw_bytes, filename=Path(rel_path).name
                    )
                    raw_text = extracted_doc.markdown_text or extracted_doc.plain_text
                else:
                    raw_text = raw_bytes.decode("utf-8", errors="replace")

                scrub_res = self.scrubber.scrub_text(raw_text, source_name=rel_path)
                scrubbed_findings += scrub_res.secret_count + scrub_res.pii_count
                scrubbed_text = scrub_res.scrubbed_text
                if not scrubbed_text.strip():
                    continue

                parsed_chunks = chunk_custom_domain_document(
                    file_path=Path(rel_path),
                    text=scrubbed_text,
                    raw_bytes=scrubbed_text.encode("utf-8"),
                    domain_id=domain_id,
                    rel_path=rel_path,
                    blob_sha=item["sha256"],
                    tenant_id=eff_tid,
                )
                if not parsed_chunks:
                    continue

                headers = [
                    (
                        f"Air-gapped bundle '{bundle_id}' document '{c.doc_title}' "
                        f"(Section: '{c.section_heading}', Path: '{rel_path}')."
                    )
                    for c in parsed_chunks
                ]
                emb_inputs = [
                    f"{hdr} {c.verbatim_content}"
                    for hdr, c in zip(headers, parsed_chunks, strict=True)
                ]
                embeddings = embedder.embed_batch(emb_inputs, is_query=False)
                for idx_c, (chk, hdr, emb_vec) in enumerate(
                    zip(parsed_chunks, headers, embeddings, strict=True)
                ):
                    meta = dict(chk.metadata_json)
                    meta.update(
                        {
                            "bundle_id": bundle_id,
                            "rel_path": rel_path,
                            "sha256": item["sha256"],
                            "tables_markdown": chk.tables_markdown,
                        }
                    )
                    doc_chunks_all.append(
                        DocChunkRecord(
                            chunk_id=uuid5(
                                NAMESPACE_DNS,
                                f"{eff_tid}:{domain_id}:bundle:{bundle_id}:{rel_path}:{idx_c}",
                            ),
                            domain_id=domain_id,
                            doc_id=f"bundle:{bundle_id}:{rel_path}",
                            doc_title=chk.doc_title,
                            source_uri=f"bundle://{bundle_id}/{rel_path}",
                            market_scope=None,
                            published_date=date.today(),
                            authority_weight=1.0,
                            section_heading=chk.section_heading,
                            contextual_header=hdr,
                            verbatim_content=chk.verbatim_content,
                            metadata_json=meta,
                            allowed_scopes=scopes,
                            embedding=emb_vec,
                            source_type="airgap_bundle",
                            version_str=bundle_id,
                            tenant_id=eff_tid,
                        )
                    )
                continue

            # Code file ingestion
            repo_name = item["repo_name"]
            commit_sha = item["commit_sha"]
            branch = item["branch"]
            if repo_name not in repos_meta:
                repos_meta[repo_name] = (branch, commit_sha, set(scopes))
            else:
                repos_meta[repo_name][2].update(scopes)

            raw_text = raw_bytes.decode("utf-8", errors="replace")
            scrub_res = self.scrubber.scrub_text(raw_text, source_name=f"{repo_name}:{rel_path}")
            scrubbed_findings += scrub_res.secret_count + scrub_res.pii_count
            scrubbed_code = scrub_res.scrubbed_text

            lang = _infer_language(rel_path, item["language"])
            blob_sha = compute_git_blob_sha(scrubbed_code.encode("utf-8"))
            repo_id = uuid5(NAMESPACE_DNS, f"{eff_tid}:{domain_id}:{repo_name}")
            file_id = uuid5(NAMESPACE_DNS, f"{eff_tid}:{domain_id}:{repo_name}:{rel_path}")
            cf_rec = CodeFileRecord(
                file_id=file_id,
                domain_id=domain_id,
                repo_name=repo_name,
                file_path=rel_path,
                git_commit_sha=commit_sha,
                git_blob_sha=blob_sha,
                language=lang,
                allowed_scopes=scopes,
                raw_content=scrubbed_code,
                line_count=len(scrubbed_code.splitlines()),
                repo_id=repo_id,
                tenant_id=eff_tid,
            )
            code_files_by_repo.setdefault(repo_name, []).append(cf_rec)

            parsed_syms, parsed_edges = extractor.parse_source(
                scrubbed_code,
                lang,
                file_path=rel_path,
                repo_name=repo_name,
            )
            file_sym_recs: list[ASTSymbolRecord] = []
            sym_id_by_name: dict[str, UUID] = {}
            if parsed_syms:
                sym_texts = [
                    f"{repo_name}::{s.name} {s.signature or s.name}".strip() for s in parsed_syms
                ]
                sym_vecs = embedder.embed_batch(sym_texts, is_query=False)
                for sym, svec in zip(parsed_syms, sym_vecs, strict=True):
                    qname = f"{repo_name}::{sym.name}"
                    sid = uuid5(NAMESPACE_DNS, f"{file_id}:{qname}:{sym.start_line}")
                    srec = ASTSymbolRecord(
                        symbol_id=sid,
                        file_id=file_id,
                        domain_id=domain_id,
                        qualified_name=qname,
                        symbol_kind=sym.symbol_kind,
                        start_line=sym.start_line,
                        end_line=sym.end_line,
                        signature_skeleton=sym.signature or sym.name,
                        contextual_summary="",
                        allowed_scopes=scopes,
                        embedding=svec,
                        repo_id=repo_id,
                        bare_name=sym.name.split(".")[-1],
                        tenant_id=eff_tid,
                    )
                    file_sym_recs.append(srec)
                    sym_id_by_name[sym.name] = sid
                    sym_id_by_name[qname] = sid
                symbols_by_repo.setdefault(repo_name, []).extend(file_sym_recs)

            for pe in parsed_edges:
                src_id = sym_id_by_name.get(pe.source_qualified_name) or (
                    file_sym_recs[0].symbol_id if file_sym_recs else None
                )
                if src_id is None:
                    continue
                tgt_sym_id = sym_id_by_name.get(pe.target_qualified_name)
                edges_by_repo.setdefault(repo_name, []).append(
                    ASTEdgeRecord(
                        edge_id=uuid4(),
                        domain_id=domain_id,
                        source_symbol_id=src_id,
                        target_qualified_name=pe.target_qualified_name,
                        target_symbol_id=tgt_sym_id,
                        edge_kind=pe.edge_kind,
                        allowed_scopes=scopes,
                        target_repo_qualified_id=pe.target_repo_qualified_id,
                        is_cross_repo=pe.is_cross_repo,
                        tenant_id=eff_tid,
                    )
                )

            se_records = se_extractor.extract_from_code_file(cf_rec, file_sym_recs)
            if se_records:
                side_effects_by_repo.setdefault(repo_name, []).extend(se_records)

            if self.zoekt_index is not None:
                self.zoekt_index.index_file(
                    tenant_id=eff_tid,
                    domain_id=domain_id,
                    repo_name=repo_name,
                    file_path=rel_path,
                    content=scrubbed_code,
                    allowed_scopes=scopes,
                    language=lang,
                    branch=branch,
                    checksum=blob_sha,
                )

        all_code_files = [cf for lst in code_files_by_repo.values() for cf in lst]
        all_symbols = [sym for lst in symbols_by_repo.values() for sym in lst]
        contract_edges = CrossBoundaryContractLinker().link_contracts(
            domain_id,
            all_symbols,
            all_code_files,
            doc_chunks_all,
            tenant_id=eff_tid,
        )
        for cedge in contract_edges:
            edges_by_repo.setdefault("_contracts", []).append(cedge)

        all_edges = [e for lst in edges_by_repo.values() for e in lst]
        all_side_effects = [se for lst in side_effects_by_repo.values() for se in lst]

        # Store verified manifest artifact in Object Store
        manifest_obj_key = build_tenant_domain_key(
            eff_tid, domain_id, f"bundles/{bundle_id}/manifest.json"
        )
        manifest_meta = self.object_store.put_bytes(
            manifest_obj_key,
            manifest_bytes,
            content_type="application/json",
            expected_tenant_id=eff_tid,
        )

        # Persist to PG17 or InMemoryRLSRepository cleanly
        is_pg = isinstance(self.repository, PostgresRLSRepository) or (
            self.connection_manager is not None
        )
        wiki_chunks_created = 0

        def _persist(target_arg: Any) -> None:
            nonlocal wiki_chunks_created
            for rname, (branch, commit_sha, rscopes) in repos_meta.items():
                rid = uuid5(NAMESPACE_DNS, f"{eff_tid}:{domain_id}:{rname}")
                repo_rec = RepositoryRecord(
                    repo_id=rid,
                    domain_id=domain_id,
                    repo_name=rname,
                    source_type="airgap_bundle",
                    remote_url=f"bundle://{bundle_id}/{rname}",
                    default_branch=branch,
                    head_commit_sha=commit_sha,
                    allowed_scopes=sorted(rscopes),
                    metadata_json={
                        "bundle_id": bundle_id,
                        "manifest_uri": manifest_meta.uri,
                        "manifest_sha256": manifest_meta.sha256,
                    },
                    tenant_id=eff_tid,
                )
                self.repository.upsert_repository(target_arg, repo_rec)

            for cf in all_code_files:
                if hasattr(self.repository, "delete_code_file_and_symbols"):
                    if is_pg:
                        self.repository.delete_code_file_and_symbols(
                            target_arg, domain_id, cf.repo_name, cf.file_path
                        )
                    else:
                        self.repository.delete_code_file_and_symbols(
                            domain_id, cf.repo_name, cf.file_path
                        )
                self.repository.insert_code_file(target_arg, cf)

            for sym in all_symbols:
                self.repository.insert_ast_symbol(target_arg, sym)

            seen_edge_triples: set[tuple[UUID, str, str]] = set()
            for edge in all_edges:
                ekey = (edge.source_symbol_id, edge.target_qualified_name, edge.edge_kind)
                if ekey in seen_edge_triples:
                    continue
                seen_edge_triples.add(ekey)
                self.repository.insert_ast_edge(target_arg, edge)

            if all_side_effects and hasattr(self.repository, "upsert_side_effects_batch"):
                self.repository.upsert_side_effects_batch(target_arg, all_side_effects)

            if doc_chunks_all:
                if hasattr(self.repository, "purge_domain_artifacts"):
                    for did in sorted({d.doc_id for d in doc_chunks_all}):
                        self.repository.purge_domain_artifacts(target_arg, domain_id, doc_id=did)
                self.repository.insert_doc_chunks_batch(target_arg, doc_chunks_all)

            wiki_builder = RepoWikiBuilder()
            for rname, (_branch, commit_sha, _rscopes) in repos_meta.items():
                r_files = code_files_by_repo.get(rname, [])
                r_file_ids = {f.file_id for f in r_files}
                r_syms = [s for s in all_symbols if s.file_id in r_file_ids]
                r_sym_ids = {s.symbol_id for s in r_syms}
                r_edges = [e for e in all_edges if e.source_symbol_id in r_sym_ids]
                r_se = side_effects_by_repo.get(rname, [])
                built_wiki = wiki_builder.sync_repo_wiki_to_repository(
                    domain_id=domain_id,
                    repo_name=rname,
                    repository=self.repository,
                    conn=target_arg if is_pg else None,
                    principal=None if is_pg else target_arg,
                    code_files=r_files,
                    symbols=r_syms,
                    edges=r_edges,
                    side_effects=r_se,
                    commit_sha=commit_sha,
                    tenant_id=eff_tid,
                )
                wiki_chunks_created += len(built_wiki)

            if hasattr(self.repository, "upsert_connector_watermark"):
                self.repository.upsert_connector_watermark(
                    target_arg,
                    ConnectorWatermarkRecord(
                        tenant_id=eff_tid,
                        domain_id=domain_id,
                        source_id="airgap_bundle",
                        container_key=bundle_id,
                        watermark_cursor=manifest_meta.sha256,
                        sync_tier="bundle",
                        items_synced=len(all_code_files) + len(doc_chunks_all),
                        metadata_json={
                            "manifest_uri": manifest_meta.uri,
                            "manifest_sha256": manifest_meta.sha256,
                            "repositories": sorted(repos_meta),
                        },
                        updated_at=datetime.now(UTC),
                    ),
                )

        if is_pg and self.connection_manager is not None:
            with self.connection_manager.scoped_transaction(eff_principal) as tx_conn:
                _persist(tx_conn)
        elif is_pg:
            raise RuntimeError("PostgresRLSRepository import_bundle requires connection_manager")
        else:
            _persist(eff_principal)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return BundleImportReport(
            bundle_id=bundle_id,
            domain_id=domain_id,
            tenant_id=str(eff_tid),
            repositories_imported=sorted(repos_meta),
            code_files_indexed=len(all_code_files),
            ast_symbols_indexed=len(all_symbols),
            ast_edges_created=len(all_edges),
            side_effects_indexed=len(all_side_effects),
            doc_chunks_indexed=len(doc_chunks_all),
            wiki_chunks_created=wiki_chunks_created,
            scrubbed_findings_count=scrubbed_findings,
            manifest_uri=manifest_meta.uri,
            manifest_sha256=manifest_meta.sha256,
            duration_ms=elapsed_ms,
            warnings=warnings,
        )


__all__ = [
    "AirGappedBundleImporter",
    "BundleImportReport",
    "BundleIntegrityError",
    "BundleSecurityError",
]
