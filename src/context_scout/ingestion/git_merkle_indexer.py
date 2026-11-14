"""Git Blob-SHA Merkle incremental indexer with Tree-Sitter AST extraction and RLS persistence."""

import hashlib
import logging
import math
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from uuid import UUID, uuid4

from psycopg import Connection
from pydantic import BaseModel

from context_scout.ast_engine.extractor import ASTExtractor
from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    InMemoryRLSRepository,
    PostgresRLSRepository,
)
from context_scout.schemas.contracts import PrincipalContext

logger = logging.getLogger(__name__)

CODE_EXTENSIONS: Final[set[str]] = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".sql",
    ".yaml",
    ".yml",
    ".tf",
    ".go",
    ".rs",
    ".java",
}


def compute_git_blob_sha(raw_bytes: bytes) -> str:
    """Computes the exact 40-hex Git blob SHA-1 for arbitrary file bytes."""
    header = f"blob {len(raw_bytes)}\x00".encode("ascii")
    return hashlib.sha1(header + raw_bytes).hexdigest()


def generate_deterministic_embedding(text: str, dim: int = 768) -> list[float]:
    """Generates a repeatable 768-dim normalized float vector from text hash."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed_int = int.from_bytes(h[:4], "big")
    extended: list[float] = []
    for i in range(dim):
        seed_int = (1103515245 * seed_int + 12345 + i) & 0x7FFFFFFF
        val = (seed_int / 0x7FFFFFFF) * 2.0 - 1.0
        extended.append(val)
    norm = math.sqrt(sum(x * x for x in extended))
    if norm == 0.0:
        return [1.0 / math.sqrt(dim)] * dim
    return [round(x / norm, 6) for x in extended]


LANG_MAP: Final[dict[str, str]] = {
    ".py": "python",
    ".tsx": "tsx",
    ".ts": "typescript",
    ".jsx": "jsx",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".tf": "terraform",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
}

_SYMBOL_EMBEDDING_CACHE: dict[tuple[str, str], list[float]] = {}
_TRACKED_IN_MEMORY_REPOS: dict[
    tuple[int, str, str], tuple[Path, list[str] | None, str]
] = {}

SKIP_BINARY_EXTENSIONS: Final[set[str]] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".bmp",
    ".pdf",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".onnx",
    ".gguf",
    ".npz",
    ".npy",
    ".whl",
    ".tar",
    ".gz",
    ".zip",
    ".7z",
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
}


# Cache immutable git ls-tree -r output by (resolved_repo_dir, commit_sha).
_COMMIT_TREE_CACHE: dict[tuple[str, str], dict[str, str]] = {}

# Cache working-tree dirty file blob SHAs by (abs_path, st_mtime_ns, st_size).
_WT_BLOB_CACHE: dict[tuple[str, int, int], str] = {}


@dataclass(frozen=True)
class MerkleBlobDiff:
    """Commit-to-commit tree diff partitioned by file operation (F08)."""

    added: dict[str, str]  # file_path -> new_blob_sha
    modified: dict[str, tuple[str, str]]  # file_path -> (old_blob_sha, new_blob_sha)
    deleted: dict[str, str]  # file_path -> old_blob_sha
    renamed: list[tuple[str, str, str, str]]  # (old_path, new_path, old_blob, new_blob)


class MerkleSyncStats(BaseModel):
    """Statistics emitted by Merkle incremental synchronization."""

    total_files_scanned: int
    unchanged_files_skipped: int
    new_files_indexed: int
    modified_files_reindexed: int
    deleted_files_removed: int
    total_symbols_stored: int
    total_edges_stored: int
    duration_ms: float
    commit_sha: str
    cache_hit: bool


def _resolve_commit_sha(repo_dir: Path, ref: str = "HEAD") -> str:
    """Fast resolution of commit SHA from .git directory with subprocess fallback."""
    git_dir = repo_dir / ".git"
    if git_dir.is_dir():
        try:
            if ref == "HEAD":
                head_file = git_dir / "HEAD"
                if head_file.is_file():
                    content = head_file.read_text().strip()
                    if not content.startswith("ref:"):
                        is_hex = all(c in "0123456789abcdefABCDEF" for c in content)
                        if len(content) == 40 and is_hex:
                            return content
                    else:
                        ref_path = content.split(":", 1)[1].strip()
                        target_file = git_dir / ref_path
                        if target_file.is_file():
                            ref_sha = target_file.read_text().strip()
                            if len(ref_sha) == 40:
                                return ref_sha
                        packed = git_dir / "packed-refs"
                        if packed.is_file():
                            for line in packed.read_text().splitlines():
                                if line.endswith(" " + ref_path):
                                    cand = line.split()[0]
                                    if len(cand) == 40:
                                        return cand
        except Exception as exc:
            logger.debug(
                "Direct .git ref lookup failed for %s (%s): %s; falling back to git rev-parse",
                ref,
                type(exc).__name__,
                exc,
            )

    rev_cmd = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    return rev_cmd.stdout.strip()


class GitMerkleIndexer:
    """Git Blob-SHA Merkle incremental indexer with Tree-Sitter AST extraction."""

    EMPTY_TREE_SHA: Final[str] = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

    def __init__(
        self,
        extractor: ASTExtractor | None = None,
        embedder: Any | None = None,
        use_semantic_embeddings: bool = True,
        include_working_tree: bool = True,
    ) -> None:
        self.extractor = extractor or ASTExtractor()
        self.embedder = embedder
        self.use_semantic_embeddings = use_semantic_embeddings
        self.include_working_tree = include_working_tree

    @staticmethod
    def _is_indexable_text_path(file_path: str) -> bool:
        suffix = Path(file_path).suffix.lower()
        return suffix not in SKIP_BINARY_EXTENSIONS

    def _compute_symbol_embeddings(
        self,
        symbols: list[ParsedSymbol],
        blob_sha: str = "",
        *,
        git_blob_sha: str = "",
    ) -> list[list[float]]:
        blob_sha = blob_sha or git_blob_sha
        if not symbols:
            return []
        if self.use_semantic_embeddings and self.embedder is not None:
            results: list[list[float] | None] = [None] * len(symbols)
            missing_indices: list[int] = []
            missing_texts: list[str] = []
            for idx, s in enumerate(symbols):
                cache_key = (blob_sha, s.qualified_name)
                if blob_sha and cache_key in _SYMBOL_EMBEDDING_CACHE:
                    results[idx] = _SYMBOL_EMBEDDING_CACHE[cache_key]
                else:
                    missing_indices.append(idx)
                    missing_texts.append(
                        f"{s.qualified_name} {s.signature_skeleton} {s.contextual_summary}".strip()
                    )
            if missing_texts:
                batch_vecs = self.embedder.embed_batch(missing_texts, is_query=False)
                if len(batch_vecs) == len(missing_indices):
                    for m_idx, vec in zip(missing_indices, batch_vecs, strict=True):
                        flt_vec = [float(x) for x in vec]
                        results[m_idx] = flt_vec
                        if blob_sha:
                            _SYMBOL_EMBEDDING_CACHE[(blob_sha, symbols[m_idx].qualified_name)] = (
                                flt_vec
                            )
            if all(r is not None for r in results):
                return [r for r in results if r is not None]
        out_vecs: list[list[float]] = []
        for s in symbols:
            cache_key = (blob_sha, s.qualified_name)
            if blob_sha and cache_key in _SYMBOL_EMBEDDING_CACHE:
                out_vecs.append(_SYMBOL_EMBEDDING_CACHE[cache_key])
            else:
                det_v = generate_deterministic_embedding(s.signature_skeleton + " " + s.docstring)
                if blob_sha:
                    _SYMBOL_EMBEDDING_CACHE[cache_key] = det_v
                out_vecs.append(det_v)
        return out_vecs

    def get_git_tree(
        self,
        repo_dir: Path,
        ref: str = "HEAD",
        include_working_tree: bool | None = None,
    ) -> tuple[str, dict[str, str]]:
        """Extracts current commit SHA and mapping of {file_path: blob_sha} via git ls-tree."""
        commit_sha = _resolve_commit_sha(repo_dir, ref)
        cache_key = (str(repo_dir.resolve()), commit_sha)
        cached_base = _COMMIT_TREE_CACHE.get(cache_key)
        if cached_base is not None:
            tree: dict[str, str] = dict(cached_base)
        else:
            ls_cmd = subprocess.run(
                ["git", "ls-tree", "-r", ref],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=True,
            )

            tree = {}
            for line in ls_cmd.stdout.strip().splitlines():
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                meta, file_path = parts
                if not self._is_indexable_text_path(file_path):
                    continue
                meta_parts = meta.split()
                if len(meta_parts) >= 3:
                    blob_sha = meta_parts[2]
                    tree[file_path] = blob_sha
            _COMMIT_TREE_CACHE[cache_key] = dict(tree)

        overlay_wt = (
            self.include_working_tree if include_working_tree is None else include_working_tree
        )
        if overlay_wt and ref == "HEAD":
            st_cmd = subprocess.run(
                ["git", "--no-optional-locks", "status", "--porcelain", "-unormal"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            if st_cmd.returncode == 0 and st_cmd.stdout.strip():
                dirty_mutations: list[str] = []
                for raw_ln in st_cmd.stdout.splitlines():
                    if len(raw_ln) < 4:
                        continue
                    xy = raw_ln[:2]
                    rel_part = raw_ln[3:].strip().strip('"')
                    if " -> " in rel_part:
                        old_rel, new_rel = rel_part.split(" -> ", 1)
                        old_rel = old_rel.strip().strip('"')
                        new_rel = new_rel.strip().strip('"')
                        tree.pop(old_rel, None)
                        dirty_mutations.append(f"D:{old_rel}")
                        rel_part = new_rel
                    if not self._is_indexable_text_path(rel_part):
                        continue
                    disk_f = repo_dir / rel_part
                    if "D" in xy or not disk_f.exists():
                        if rel_part in tree:
                            tree.pop(rel_part, None)
                            dirty_mutations.append(f"D:{rel_part}")
                    elif disk_f.is_file():
                        try:
                            st = disk_f.stat()
                            wt_key = (str(disk_f), st.st_mtime_ns, st.st_size)
                            wt_blob = _WT_BLOB_CACHE.get(wt_key)
                            if wt_blob is None:
                                wt_blob = compute_git_blob_sha(disk_f.read_bytes())
                                _WT_BLOB_CACHE[wt_key] = wt_blob
                            if tree.get(rel_part) != wt_blob:
                                tree[rel_part] = wt_blob
                                dirty_mutations.append(f"M:{rel_part}:{wt_blob}")
                        except Exception as exc:
                            logger.warning(
                                "Failed to hash working-tree file %s (%s): %s",
                                rel_part,
                                type(exc).__name__,
                                exc,
                            )
                if dirty_mutations:
                    wt_digest = hashlib.sha1(
                        "|".join(sorted(dirty_mutations)).encode("utf-8")
                    ).hexdigest()
                    commit_sha = f"{commit_sha[:32]}dddd{wt_digest[:4]}"

        return commit_sha, tree

    def diff_merkle_tree(
        self, current: dict[str, str], existing: dict[str, str]
    ) -> tuple[set[str], set[str], set[str], set[str]]:
        """Partitions repository files into (unchanged, new, modified, deleted)."""
        unchanged = {p for p, s in current.items() if p in existing and existing[p] == s}
        new = set(current.keys()) - set(existing.keys())
        modified = {p for p, s in current.items() if p in existing and existing[p] != s}
        deleted = set(existing.keys()) - set(current.keys())
        return unchanged, new, modified, deleted

    def diff_commit_blobs(
        self,
        repo_dir: Path,
        before_sha: str,
        after_sha: str,
    ) -> MerkleBlobDiff:
        """Diffs commit trees using git diff-tree -r --raw in <25ms (F08)."""
        effective_before = (
            self.EMPTY_TREE_SHA
            if not before_sha or before_sha == "0" * 40 or before_sha.strip() == ""
            else before_sha
        )
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "diff-tree", "-r", "--raw", effective_before, after_sha],
            capture_output=True,
            text=True,
            check=True,
        )

        added: dict[str, str] = {}
        modified: dict[str, tuple[str, str]] = {}
        deleted: dict[str, str] = {}
        renamed: list[tuple[str, str, str, str]] = []

        for line in proc.stdout.splitlines():
            if not line.startswith(":"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            meta = parts[0]
            file_path = parts[1]
            meta_tokens = meta[1:].split()
            if len(meta_tokens) < 5:
                continue
            old_blob = meta_tokens[2]
            new_blob = meta_tokens[3]
            status = meta_tokens[4]

            if not self._is_indexable_text_path(file_path):
                continue

            if status == "A":
                added[file_path] = new_blob
            elif status == "M":
                modified[file_path] = (old_blob, new_blob)
            elif status == "D":
                deleted[file_path] = old_blob
            elif status.startswith("R"):
                old_path = file_path
                new_path = parts[2] if len(parts) > 2 else file_path
                renamed.append((old_path, new_path, old_blob, new_blob))

        return MerkleBlobDiff(added=added, modified=modified, deleted=deleted, renamed=renamed)

    def _detect_language(self, file_path: str) -> str | None:
        for ext, lang in LANG_MAP.items():
            if file_path.endswith(ext):
                return lang
        return None

    @staticmethod
    def _read_blob_or_disk(
        repo_dir: Path,
        file_path: str,
        blob_sha: str,
        ref: str = "HEAD",
        prefer_working_tree: bool = False,
    ) -> str | None:
        """Read immutable Git blob content via git cat-file or live working-tree file."""
        _ = ref
        disk_file = repo_dir / file_path
        if prefer_working_tree and disk_file.is_file():
            try:
                return disk_file.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
            except Exception as exc:
                logger.debug(
                    "Working-tree read failed for %s (%s): %s; falling back to git cat-file",
                    file_path,
                    type(exc).__name__,
                    exc,
                )
        if blob_sha:
            proc = subprocess.run(
                ["git", "-C", str(repo_dir), "cat-file", "-p", blob_sha],
                capture_output=True,
                check=False,
            )
            if proc.returncode == 0:
                return proc.stdout.decode("utf-8", errors="replace").replace("\x00", "")
        if not disk_file.exists():
            return None
        return disk_file.read_text(encoding="utf-8", errors="replace").replace("\x00", "")

    def sync_in_memory(
        self,
        repo_dir: Path,
        repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        domain_id: str,
        repo_name: str,
        allowed_scopes: list[str] | None = None,
        ref: str = "HEAD",
        isolate_by_commit: bool = False,
        include_working_tree: bool | None = None,
    ) -> MerkleSyncStats:
        """Incrementally indexes git repository into InMemoryRLSRepository."""
        start_time = time.perf_counter()
        use_wt = (
            (self.include_working_tree if include_working_tree is None else include_working_tree)
            and ref == "HEAD"
            and not isolate_by_commit
        )
        commit_sha, current_tree = self.get_git_tree(
            repo_dir, ref, include_working_tree=use_wt
        )
        effective_repo_name = f"{repo_name}@{commit_sha[:12]}" if isolate_by_commit else repo_name
        if ref == "HEAD" and not isolate_by_commit:
            _TRACKED_IN_MEMORY_REPOS[(id(repo), domain_id, effective_repo_name)] = (
                repo_dir.resolve(),
                allowed_scopes,
                commit_sha,
            )
        existing_hashes = repo.get_file_blob_hashes(principal, domain_id, effective_repo_name)
        existing_map = {path: blob_sha for path, (_, blob_sha) in existing_hashes.items()}

        unchanged, new, modified, deleted = self.diff_merkle_tree(current_tree, existing_map)

        # Fast-exit cache hit (<50ms)
        if not new and not modified and not deleted:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return MerkleSyncStats(
                total_files_scanned=len(current_tree),
                unchanged_files_skipped=len(unchanged),
                new_files_indexed=0,
                modified_files_reindexed=0,
                deleted_files_removed=0,
                total_symbols_stored=0,
                total_edges_stored=0,
                duration_ms=duration_ms,
                commit_sha=commit_sha,
                cache_hit=True,
            )

        # 1. Remove deleted files
        for d in deleted:
            if d in existing_hashes:
                file_id, _ = existing_hashes[d]
                repo.delete_code_file(principal, file_id)

        # 2. Remove modified files prior to re-indexing
        for m in modified:
            if m in existing_hashes:
                file_id, _ = existing_hashes[m]
                repo.delete_code_file(principal, file_id)

        scopes = allowed_scopes or principal.allowed_scopes
        to_index = sorted(new | modified)

        total_symbols = 0
        total_edges = 0

        # Pass 1: Parse and insert code files and ast symbols
        parsed_files_data: list[
            tuple[CodeFileRecord, list[ParsedSymbol], list[ParsedEdge], dict[str, UUID]]
        ] = []
        global_symbol_map: dict[str, UUID] = {}
        short_name_map: dict[str, list[UUID]] = defaultdict(list)

        # Also populate global_symbol_map with existing visible symbols for cross-file linking
        for s in repo.list_visible_ast_symbols(principal, domain_id):
            global_symbol_map[s.qualified_name] = s.symbol_id
            short_name_map[s.qualified_name.split(".")[-1]].append(s.symbol_id)

        for file_path in to_index:
            blob_sha = current_tree[file_path]
            raw_content = self._read_blob_or_disk(
                repo_dir, file_path, blob_sha, ref, prefer_working_tree=use_wt
            )
            if raw_content is None:
                continue

            lang = self._detect_language(file_path)
            lines = raw_content.splitlines()
            file_id = uuid4()

            file_rec = CodeFileRecord(
                file_id=file_id,
                domain_id=domain_id,
                repo_name=effective_repo_name,
                file_path=file_path,
                git_commit_sha=commit_sha,
                git_blob_sha=blob_sha,
                language=lang or "text",
                allowed_scopes=scopes,
                raw_content=raw_content,
                line_count=len(lines),
            )
            repo.insert_code_file(principal, file_rec)

            if not lang:
                continue

            symbols, edges = self.extractor.parse_source(raw_content, lang, file_path=file_path)
            file_symbol_map: dict[str, UUID] = {}
            symbol_embeddings = self._compute_symbol_embeddings(symbols, blob_sha=blob_sha)

            for parsed_sym, emb in zip(symbols, symbol_embeddings, strict=True):
                sym_id = uuid4()
                sym_rec = ASTSymbolRecord(
                    symbol_id=sym_id,
                    file_id=file_id,
                    domain_id=domain_id,
                    qualified_name=parsed_sym.qualified_name,
                    symbol_kind=parsed_sym.symbol_kind,
                    start_line=parsed_sym.start_line,
                    end_line=parsed_sym.end_line,
                    signature_skeleton=parsed_sym.signature_skeleton,
                    contextual_summary=parsed_sym.contextual_summary,
                    allowed_scopes=scopes,
                    embedding=emb,
                )
                repo.insert_ast_symbol(principal, sym_rec)
                file_symbol_map[parsed_sym.qualified_name] = sym_id
                global_symbol_map[parsed_sym.qualified_name] = sym_id
                short_name_map[parsed_sym.name].append(sym_id)
                total_symbols += 1

            parsed_files_data.append((file_rec, symbols, edges, file_symbol_map))

        # Pass 2: Directed Edge Resolution
        for file_rec, _symbols, edges, file_symbol_map in parsed_files_data:
            for edge in edges:
                src_id = file_symbol_map.get(edge.source_qualified_name)
                if not src_id:
                    # Look up by exact qualified name or short name (for cross-file TESTED_BY)
                    src_id = global_symbol_map.get(edge.source_qualified_name)
                if not src_id:
                    src_short = edge.source_qualified_name.split(".")[-1]
                    src_matches = short_name_map.get(src_short, [])
                    if src_matches:
                        src_id = src_matches[0]
                if not src_id:
                    continue

                target_id = global_symbol_map.get(edge.target_qualified_name)
                if not target_id:
                    target_short = edge.target_qualified_name.split(".")[-1]
                    matches = short_name_map.get(target_short, [])
                    if len(matches) == 1:
                        target_id = matches[0]

                edge_rec = ASTEdgeRecord(
                    edge_id=uuid4(),
                    domain_id=domain_id,
                    source_symbol_id=src_id,
                    target_qualified_name=edge.target_qualified_name,
                    target_symbol_id=target_id,
                    edge_kind=edge.edge_kind,
                    allowed_scopes=file_rec.allowed_scopes,
                )
                repo.insert_ast_edge(principal, edge_rec)
                total_edges += 1

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return MerkleSyncStats(
            total_files_scanned=len(current_tree),
            unchanged_files_skipped=len(unchanged),
            new_files_indexed=len(new),
            modified_files_reindexed=len(modified),
            deleted_files_removed=len(deleted),
            total_symbols_stored=total_symbols,
            total_edges_stored=total_edges,
            duration_ms=duration_ms,
            commit_sha=commit_sha,
            cache_hit=False,
        )

    def sync_postgres(
        self,
        repo_dir: Path,
        conn: Connection[tuple[object, ...]],
        repo: PostgresRLSRepository,
        principal: PrincipalContext,
        domain_id: str,
        repo_name: str,
        allowed_scopes: list[str] | None = None,
        ref: str = "HEAD",
        isolate_by_commit: bool = False,
        include_working_tree: bool | None = None,
    ) -> MerkleSyncStats:
        """Incrementally indexes git repository into live PostgreSQL 17 under Forced RLS."""
        start_time = time.perf_counter()
        use_wt = (
            (self.include_working_tree if include_working_tree is None else include_working_tree)
            and ref == "HEAD"
            and not isolate_by_commit
        )
        commit_sha, current_tree = self.get_git_tree(
            repo_dir, ref, include_working_tree=use_wt
        )
        effective_repo_name = f"{repo_name}@{commit_sha[:12]}" if isolate_by_commit else repo_name
        existing_hashes = repo.get_file_blob_hashes(conn, domain_id, effective_repo_name)
        existing_map = {path: blob_sha for path, (_, blob_sha) in existing_hashes.items()}

        unchanged, new, modified, deleted = self.diff_merkle_tree(current_tree, existing_map)

        # Fast-exit cache hit (<50ms)
        if not new and not modified and not deleted:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return MerkleSyncStats(
                total_files_scanned=len(current_tree),
                unchanged_files_skipped=len(unchanged),
                new_files_indexed=0,
                modified_files_reindexed=0,
                deleted_files_removed=0,
                total_symbols_stored=0,
                total_edges_stored=0,
                duration_ms=duration_ms,
                commit_sha=commit_sha,
                cache_hit=True,
            )

        # 1. Remove deleted files
        for d in deleted:
            if d in existing_hashes:
                file_id, _ = existing_hashes[d]
                repo.delete_code_file(conn, file_id)

        # 2. Remove modified files prior to re-indexing
        for m in modified:
            if m in existing_hashes:
                file_id, _ = existing_hashes[m]
                repo.delete_code_file(conn, file_id)

        scopes = allowed_scopes or principal.allowed_scopes
        to_index = sorted(new | modified)

        total_symbols = 0
        total_edges = 0

        parsed_files_data: list[
            tuple[CodeFileRecord, list[ParsedSymbol], list[ParsedEdge], dict[str, UUID]]
        ] = []
        global_symbol_map: dict[str, UUID] = {}
        short_name_map: dict[str, list[UUID]] = defaultdict(list)

        # Query existing symbols in DB for cross-file linking
        existing_sym_rows = conn.execute(
            "SELECT symbol_id, qualified_name FROM ast_symbols WHERE domain_id = %s;",
            (domain_id,),
        ).fetchall()
        for s_row in existing_sym_rows:
            s_id = UUID(str(s_row[0]))
            s_qual = str(s_row[1])
            global_symbol_map[s_qual] = s_id
            short_name_map[s_qual.split(".")[-1]].append(s_id)

        for file_path in to_index:
            blob_sha = current_tree[file_path]
            raw_content = self._read_blob_or_disk(
                repo_dir, file_path, blob_sha, ref, prefer_working_tree=use_wt
            )
            if raw_content is None:
                continue

            lang = self._detect_language(file_path)
            lines = raw_content.splitlines()
            file_id = uuid4()

            file_rec = CodeFileRecord(
                file_id=file_id,
                domain_id=domain_id,
                repo_name=effective_repo_name,
                file_path=file_path,
                git_commit_sha=commit_sha,
                git_blob_sha=blob_sha,
                language=lang or "text",
                allowed_scopes=scopes,
                raw_content=raw_content,
                line_count=len(lines),
            )
            repo.insert_code_file(conn, file_rec)

            if not lang:
                continue

            symbols, edges = self.extractor.parse_source(raw_content, lang, file_path=file_path)
            file_symbol_map: dict[str, UUID] = {}
            symbol_embeddings = self._compute_symbol_embeddings(symbols, blob_sha=blob_sha)

            for parsed_sym, emb in zip(symbols, symbol_embeddings, strict=True):
                sym_id = uuid4()
                sym_rec = ASTSymbolRecord(
                    symbol_id=sym_id,
                    file_id=file_id,
                    domain_id=domain_id,
                    qualified_name=parsed_sym.qualified_name,
                    symbol_kind=parsed_sym.symbol_kind,
                    start_line=parsed_sym.start_line,
                    end_line=parsed_sym.end_line,
                    signature_skeleton=parsed_sym.signature_skeleton,
                    contextual_summary=parsed_sym.contextual_summary,
                    allowed_scopes=scopes,
                    embedding=emb,
                )
                repo.insert_ast_symbol(conn, sym_rec)
                file_symbol_map[parsed_sym.qualified_name] = sym_id
                global_symbol_map[parsed_sym.qualified_name] = sym_id
                short_name_map[parsed_sym.name].append(sym_id)
                total_symbols += 1

            parsed_files_data.append((file_rec, symbols, edges, file_symbol_map))

        # Pass 2: Directed Edge Resolution
        for file_rec, _symbols, edges, file_symbol_map in parsed_files_data:
            for edge in edges:
                src_id = file_symbol_map.get(edge.source_qualified_name)
                if not src_id:
                    src_id = global_symbol_map.get(edge.source_qualified_name)
                if not src_id:
                    src_short = edge.source_qualified_name.split(".")[-1]
                    src_matches = short_name_map.get(src_short, [])
                    if src_matches:
                        src_id = src_matches[0]
                if not src_id:
                    continue

                target_id = global_symbol_map.get(edge.target_qualified_name)
                if not target_id:
                    target_short = edge.target_qualified_name.split(".")[-1]
                    matches = short_name_map.get(target_short, [])
                    if len(matches) == 1:
                        target_id = matches[0]

                edge_rec = ASTEdgeRecord(
                    edge_id=uuid4(),
                    domain_id=domain_id,
                    source_symbol_id=src_id,
                    target_qualified_name=edge.target_qualified_name,
                    target_symbol_id=target_id,
                    edge_kind=edge.edge_kind,
                    allowed_scopes=file_rec.allowed_scopes,
                )
                repo.insert_ast_edge(conn, edge_rec)
                total_edges += 1

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return MerkleSyncStats(
            total_files_scanned=len(current_tree),
            unchanged_files_skipped=len(unchanged),
            new_files_indexed=len(new),
            modified_files_reindexed=len(modified),
            deleted_files_removed=len(deleted),
            total_symbols_stored=total_symbols,
            total_edges_stored=total_edges,
            duration_ms=duration_ms,
            commit_sha=commit_sha,
            cache_hit=False,
        )

    def sync_push_incremental_postgres(
        self,
        repo_dir: Path,
        conn: Connection[tuple[object, ...]],
        repo: PostgresRLSRepository,
        principal: PrincipalContext,
        domain_id: str,
        repo_name: str,
        before_sha: str,
        after_sha: str,
        allowed_scopes: list[str] | None = None,
    ) -> MerkleSyncStats:
        """Executes surgical incremental update on push in <50ms without dropping inward edges."""
        start_time = time.perf_counter()
        scopes = allowed_scopes or principal.allowed_scopes

        diff = self.diff_commit_blobs(repo_dir, before_sha, after_sha)

        # 1. Process deletions
        all_deleted_paths = set(diff.deleted.keys()) | {r[0] for r in diff.renamed}
        if all_deleted_paths:
            conn.execute(
                """
                DELETE FROM code_files
                WHERE domain_id = %s AND repo_name = %s AND file_path = ANY(%s);
                """,
                (domain_id, repo_name, list(all_deleted_paths)),
            )

        # 2. Process modified files: delete old symbols (cascades outward edges, NULLs inward edges)
        all_modified_paths = set(diff.modified.keys())
        if all_modified_paths:
            conn.execute(
                """
                DELETE FROM ast_symbols
                WHERE domain_id = %s
                  AND file_id IN (
                      SELECT file_id FROM code_files
                      WHERE domain_id = %s AND repo_name = %s AND file_path = ANY(%s)
                  );
                """,
                (domain_id, domain_id, repo_name, list(all_modified_paths)),
            )

        # 3. Parse and insert added and modified files
        to_index_paths = set(diff.added.keys()) | all_modified_paths | {r[1] for r in diff.renamed}
        total_symbols = 0
        total_edges = 0

        repo_symbol_map: dict[str, UUID] = {}
        existing_sym_rows = conn.execute(
            """
            SELECT s.symbol_id, s.qualified_name
            FROM ast_symbols s
            JOIN code_files f ON f.file_id = s.file_id
            WHERE s.domain_id = %s AND f.repo_name = %s;
            """,
            (domain_id, repo_name),
        ).fetchall()
        for s_row in existing_sym_rows:
            repo_symbol_map[str(s_row[1])] = UUID(str(s_row[0]))

        newly_inserted_file_ids: list[UUID] = []
        parsed_files_data = []

        for f_path in sorted(to_index_paths):
            new_blob_sha = (
                diff.added.get(f_path)
                or (diff.modified[f_path][1] if f_path in diff.modified else None)
                or next((r[3] for r in diff.renamed if r[1] == f_path), "")
            )
            raw_content = self._read_blob_or_disk(repo_dir, f_path, new_blob_sha or "", after_sha)
            if raw_content is None:
                continue
            lang = self._detect_language(f_path)
            lines = raw_content.splitlines()

            file_id = uuid4()
            row = conn.execute(
                """
                INSERT INTO code_files (
                    file_id, domain_id, repo_name, file_path,
                    git_commit_sha, git_blob_sha, language,
                    allowed_scopes, raw_content, line_count
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (domain_id, repo_name, file_path) DO UPDATE SET
                    git_commit_sha = EXCLUDED.git_commit_sha,
                    git_blob_sha = EXCLUDED.git_blob_sha,
                    raw_content = EXCLUDED.raw_content,
                    line_count = EXCLUDED.line_count,
                    indexed_at = NOW()
                RETURNING file_id;
                """,
                (
                    file_id,
                    domain_id,
                    repo_name,
                    f_path,
                    after_sha,
                    new_blob_sha or "",
                    lang or "text",
                    scopes,
                    raw_content,
                    len(lines),
                ),
            ).fetchone()
            actual_file_id = UUID(str(row[0])) if row else file_id
            newly_inserted_file_ids.append(actual_file_id)

            if not lang:
                continue

            symbols, edges = self.extractor.parse_source(
                raw_content, lang, file_path=f_path, repo_name=repo_name
            )
            file_symbol_map: dict[str, UUID] = {}
            symbol_embeddings = self._compute_symbol_embeddings(
                symbols, blob_sha=new_blob_sha or ""
            )

            for parsed_sym, emb in zip(symbols, symbol_embeddings, strict=True):
                sym_id = uuid4()
                sym_rec = ASTSymbolRecord(
                    symbol_id=sym_id,
                    file_id=actual_file_id,
                    domain_id=domain_id,
                    qualified_name=parsed_sym.qualified_name,
                    symbol_kind=parsed_sym.symbol_kind,
                    start_line=parsed_sym.start_line,
                    end_line=parsed_sym.end_line,
                    signature_skeleton=parsed_sym.signature_skeleton,
                    contextual_summary=parsed_sym.contextual_summary,
                    allowed_scopes=scopes,
                    embedding=emb,
                )
                repo.insert_ast_symbol(conn, sym_rec)
                file_symbol_map[parsed_sym.qualified_name] = sym_id
                repo_symbol_map[parsed_sym.qualified_name] = sym_id
                total_symbols += 1

            parsed_files_data.append((actual_file_id, scopes, edges, file_symbol_map))

        # 4. Insert outward edges
        for _actual_file_id, file_scopes, edges, file_symbol_map in parsed_files_data:
            for parsed_edge in edges:
                src_id = file_symbol_map.get(
                    parsed_edge.source_qualified_name
                ) or repo_symbol_map.get(parsed_edge.source_qualified_name)
                if not src_id:
                    continue

                target_id = repo_symbol_map.get(parsed_edge.target_qualified_name)
                target_qual = parsed_edge.target_qualified_name
                if not target_id:
                    target_qual = f"UNRESOLVED::{parsed_edge.target_qualified_name}"

                edge_rec = ASTEdgeRecord(
                    edge_id=uuid4(),
                    domain_id=domain_id,
                    source_symbol_id=src_id,
                    target_qualified_name=target_qual,
                    target_symbol_id=target_id,
                    edge_kind=parsed_edge.edge_kind,
                    allowed_scopes=file_scopes,
                    target_repo_qualified_id=parsed_edge.target_repo_qualified_id,
                    is_cross_repo=parsed_edge.is_cross_repo or target_id is None,
                )
                repo.insert_ast_edge(conn, edge_rec)
                total_edges += 1

        # 5. Inward Edge Re-link: restore existing inward edges targeting updated symbols
        if newly_inserted_file_ids:
            conn.execute(
                """
                UPDATE ast_edges e
                SET target_symbol_id = s.symbol_id,
                    target_qualified_name = s.qualified_name
                FROM ast_symbols s
                WHERE s.domain_id = %s
                  AND s.file_id = ANY(%s)
                  AND e.domain_id = s.domain_id
                  AND e.target_symbol_id IS NULL
                  AND (
                      e.target_qualified_name = s.qualified_name
                      OR e.target_qualified_name = ('UNRESOLVED::' || s.qualified_name)
                      OR ('UNRESOLVED::' || e.target_qualified_name) = s.qualified_name
                      OR s.qualified_name LIKE ('%%::' || e.target_qualified_name)
                  );
                """,
                (domain_id, newly_inserted_file_ids),
            )

        # 6. Reconcile cross-repo edges
        self.reconcile_cross_repo_edges_postgres(conn, domain_id)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return MerkleSyncStats(
            total_files_scanned=len(to_index_paths),
            unchanged_files_skipped=0,
            new_files_indexed=len(diff.added),
            modified_files_reindexed=len(diff.modified),
            deleted_files_removed=len(all_deleted_paths),
            total_symbols_stored=total_symbols,
            total_edges_stored=total_edges,
            duration_ms=duration_ms,
            commit_sha=after_sha,
            cache_hit=False,
        )

    def reconcile_cross_repo_edges_postgres(
        self, conn: Connection[tuple[object, ...]], domain_id: str
    ) -> int:
        """Phase 2: Resolves UNRESOLVED cross-repo edges across repositories under RLS (F09)."""
        res = conn.execute(
            """
            WITH candidate_matches AS (
                SELECT
                    e.edge_id,
                    s.symbol_id AS matched_symbol_id,
                    s.qualified_name AS matched_qual_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY e.edge_id
                        ORDER BY (s.allowed_scopes && e.allowed_scopes) DESC, s.symbol_id
                    ) AS match_rank
                FROM ast_edges e
                JOIN ast_symbols s ON s.domain_id = e.domain_id
                WHERE e.domain_id = %s
                  AND e.target_symbol_id IS NULL
                  AND e.target_qualified_name LIKE 'UNRESOLVED::%%'
                  AND (
                      s.qualified_name = SUBSTRING(e.target_qualified_name FROM 13)
                      OR s.qualified_name LIKE (
                          '%%::' || SUBSTRING(e.target_qualified_name FROM 13)
                      )
                      OR s.qualified_name LIKE (
                          '%%' || REPLACE(SUBSTRING(e.target_qualified_name FROM 13), '.', '::%%')
                      )
                  )
                  AND e.allowed_scopes && s.allowed_scopes
            )
            UPDATE ast_edges e
            SET target_symbol_id = cm.matched_symbol_id,
                target_qualified_name = cm.matched_qual_id,
                is_cross_repo = TRUE
            FROM candidate_matches cm
            WHERE e.edge_id = cm.edge_id AND cm.match_rank = 1;
            """,
            (domain_id,),
        )
        return res.rowcount if hasattr(res, "rowcount") and res.rowcount != -1 else 0

    def sync_push_incremental_inmemory(
        self,
        repo_dir: Path,
        repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        domain_id: str,
        repo_name: str,
        before_sha: str,
        after_sha: str,
        allowed_scopes: list[str] | None = None,
    ) -> MerkleSyncStats:
        """Executes surgical incremental update on push in InMemoryRLSRepository (F08, F09)."""
        start_time = time.perf_counter()
        scopes = allowed_scopes or principal.allowed_scopes

        diff = self.diff_commit_blobs(repo_dir, before_sha, after_sha)

        all_deleted_paths = set(diff.deleted.keys()) | {r[0] for r in diff.renamed}
        all_modified_paths = set(diff.modified.keys())

        # 1. Process deletions
        for file_id, file_rec in list(repo._files.items()):
            if (
                file_rec.domain_id == domain_id
                and file_rec.repo_name == repo_name
                and file_rec.file_path in all_deleted_paths
            ):
                repo.delete_code_file(principal, file_id)

        # 2. Process modified files: delete old symbols & outward edges, preserve inward edges
        for file_id, file_rec in list(repo._files.items()):
            if (
                file_rec.domain_id == domain_id
                and file_rec.repo_name == repo_name
                and file_rec.file_path in all_modified_paths
            ):
                sym_ids_to_del = {
                    s_id for s_id, sym in repo._symbols.items() if sym.file_id == file_id
                }
                for s_id in sym_ids_to_del:
                    repo._symbols.pop(s_id, None)

                # Outward edges
                outward_edge_ids = {
                    e_id for e_id, edge in repo._edges.items()
                    if edge.source_symbol_id in sym_ids_to_del
                }
                for e_id in outward_edge_ids:
                    repo._edges.pop(e_id, None)

                # Inward edges -> target_symbol_id = None
                for e_id, edge in list(repo._edges.items()):
                    if edge.target_symbol_id in sym_ids_to_del:
                        repo._edges[e_id] = ASTEdgeRecord(
                            edge_id=edge.edge_id,
                            domain_id=edge.domain_id,
                            source_symbol_id=edge.source_symbol_id,
                            target_qualified_name=edge.target_qualified_name,
                            target_symbol_id=None,
                            edge_kind=edge.edge_kind,
                            allowed_scopes=edge.allowed_scopes,
                            target_repo_qualified_id=edge.target_repo_qualified_id,
                            is_cross_repo=edge.is_cross_repo,
                        )

        # 3. Parse and insert added and modified files
        to_index_paths = set(diff.added.keys()) | all_modified_paths | {r[1] for r in diff.renamed}
        total_symbols = 0
        total_edges = 0

        repo_symbol_map: dict[str, UUID] = {}
        for s_id, s_rec in repo._symbols.items():
            if s_rec.domain_id == domain_id:
                f_rec = repo._files.get(s_rec.file_id)
                if f_rec and f_rec.repo_name == repo_name:
                    repo_symbol_map[s_rec.qualified_name] = s_id

        newly_inserted_file_ids: list[UUID] = []
        parsed_files_data = []

        for f_path in sorted(to_index_paths):
            new_blob_sha = (
                diff.added.get(f_path)
                or (diff.modified[f_path][1] if f_path in diff.modified else None)
                or next((r[3] for r in diff.renamed if r[1] == f_path), "")
            )
            raw_content = self._read_blob_or_disk(repo_dir, f_path, new_blob_sha or "", after_sha)
            if raw_content is None:
                continue
            lang = self._detect_language(f_path)
            lines = raw_content.splitlines()

            # Existing file_id if modifying
            existing_fid = next(
                (
                    f_id
                    for f_id, f_rec in repo._files.items()
                    if f_rec.domain_id == domain_id
                    and f_rec.repo_name == repo_name
                    and f_rec.file_path == f_path
                ),
                None,
            )
            actual_file_id = existing_fid or uuid4()

            file_rec = CodeFileRecord(
                file_id=actual_file_id,
                domain_id=domain_id,
                repo_name=repo_name,
                file_path=f_path,
                git_commit_sha=after_sha,
                git_blob_sha=new_blob_sha or "",
                language=lang or "text",
                allowed_scopes=scopes,
                raw_content=raw_content,
                line_count=len(lines),
            )
            repo.insert_code_file(principal, file_rec)
            newly_inserted_file_ids.append(actual_file_id)

            if not lang:
                continue

            symbols, edges = self.extractor.parse_source(
                raw_content, lang, file_path=f_path, repo_name=repo_name
            )
            file_symbol_map: dict[str, UUID] = {}
            symbol_embeddings = self._compute_symbol_embeddings(
                symbols, blob_sha=new_blob_sha or ""
            )

            for parsed_sym, emb in zip(symbols, symbol_embeddings, strict=True):
                sym_id = uuid4()
                sym_rec = ASTSymbolRecord(
                    symbol_id=sym_id,
                    file_id=actual_file_id,
                    domain_id=domain_id,
                    qualified_name=parsed_sym.qualified_name,
                    symbol_kind=parsed_sym.symbol_kind,
                    start_line=parsed_sym.start_line,
                    end_line=parsed_sym.end_line,
                    signature_skeleton=parsed_sym.signature_skeleton,
                    contextual_summary=parsed_sym.contextual_summary,
                    allowed_scopes=scopes,
                    embedding=emb,
                )
                repo.insert_ast_symbol(principal, sym_rec)
                file_symbol_map[parsed_sym.qualified_name] = sym_id
                repo_symbol_map[parsed_sym.qualified_name] = sym_id
                total_symbols += 1

            parsed_files_data.append((actual_file_id, scopes, edges, file_symbol_map))

        # 4. Insert outward edges
        for _actual_file_id, file_scopes, edges, file_symbol_map in parsed_files_data:
            for parsed_edge in edges:
                src_id = file_symbol_map.get(
                    parsed_edge.source_qualified_name
                ) or repo_symbol_map.get(parsed_edge.source_qualified_name)
                if not src_id:
                    continue

                target_id = repo_symbol_map.get(parsed_edge.target_qualified_name)
                target_qual = parsed_edge.target_qualified_name
                if not target_id:
                    target_qual = f"UNRESOLVED::{parsed_edge.target_qualified_name}"

                edge_rec = ASTEdgeRecord(
                    edge_id=uuid4(),
                    domain_id=domain_id,
                    source_symbol_id=src_id,
                    target_qualified_name=target_qual,
                    target_symbol_id=target_id,
                    edge_kind=parsed_edge.edge_kind,
                    allowed_scopes=file_scopes,
                    target_repo_qualified_id=parsed_edge.target_repo_qualified_id,
                    is_cross_repo=parsed_edge.is_cross_repo or target_id is None,
                )
                repo.insert_ast_edge(principal, edge_rec)
                total_edges += 1

        # 5. Inward Edge Re-link in memory
        newly_inserted_syms = [
            s
            for s in repo._symbols.values()
            if s.domain_id == domain_id and s.file_id in newly_inserted_file_ids
        ]
        for s in newly_inserted_syms:
            for e_id, edge in list(repo._edges.items()):
                if edge.domain_id == domain_id and edge.target_symbol_id is None:
                    t_qual = edge.target_qualified_name
                    if (
                        t_qual == s.qualified_name
                        or t_qual == f"UNRESOLVED::{s.qualified_name}"
                        or f"UNRESOLVED::{t_qual}" == s.qualified_name
                        or s.qualified_name.endswith(f"::{t_qual}")
                    ):
                        repo._edges[e_id] = ASTEdgeRecord(
                            edge_id=edge.edge_id,
                            domain_id=edge.domain_id,
                            source_symbol_id=edge.source_symbol_id,
                            target_qualified_name=s.qualified_name,
                            target_symbol_id=s.symbol_id,
                            edge_kind=edge.edge_kind,
                            allowed_scopes=edge.allowed_scopes,
                            target_repo_qualified_id=edge.target_repo_qualified_id,
                            is_cross_repo=edge.is_cross_repo,
                        )

        # 6. Reconcile cross-repo edges in memory
        self.reconcile_cross_repo_edges_inmemory(repo, domain_id)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return MerkleSyncStats(
            total_files_scanned=len(to_index_paths),
            unchanged_files_skipped=0,
            new_files_indexed=len(diff.added),
            modified_files_reindexed=len(diff.modified),
            deleted_files_removed=len(all_deleted_paths),
            total_symbols_stored=total_symbols,
            total_edges_stored=total_edges,
            duration_ms=duration_ms,
            commit_sha=after_sha,
            cache_hit=False,
        )

    def reconcile_cross_repo_edges_inmemory(
        self, repo: InMemoryRLSRepository, domain_id: str
    ) -> int:
        """Phase 2: Resolves UNRESOLVED cross-repo edges in InMemoryRLSRepository (F09)."""
        from context_scout.db.repositories import _scopes_intersect

        resolved_count = 0
        all_symbols = [s for s in repo._symbols.values() if s.domain_id == domain_id]

        for e_id, edge in list(repo._edges.items()):
            if (
                edge.domain_id == domain_id
                and edge.target_symbol_id is None
                and edge.target_qualified_name.startswith("UNRESOLVED::")
            ):
                unresolved_target = edge.target_qualified_name[12:]
                candidate: ASTSymbolRecord | None = None
                for s in all_symbols:
                    if not _scopes_intersect(edge.allowed_scopes, s.allowed_scopes):
                        continue
                    if (
                        s.qualified_name == unresolved_target
                        or s.qualified_name.endswith(f"::{unresolved_target}")
                        or s.qualified_name.endswith(f"::{unresolved_target.replace('.', '::')}")
                        or unresolved_target in s.qualified_name
                    ):
                        candidate = s
                        break

                if candidate is not None:
                    repo._edges[e_id] = ASTEdgeRecord(
                        edge_id=edge.edge_id,
                        domain_id=edge.domain_id,
                        source_symbol_id=edge.source_symbol_id,
                        target_qualified_name=candidate.qualified_name,
                        target_symbol_id=candidate.symbol_id,
                        edge_kind=edge.edge_kind,
                        allowed_scopes=edge.allowed_scopes,
                        target_repo_qualified_id=candidate.qualified_name,
                        is_cross_repo=True,
                    )
                    resolved_count += 1

        return resolved_count


def sync_tracked_in_memory_repos(
    repo: Any, principal: PrincipalContext, domain_id: str
) -> list[MerkleSyncStats]:
    """Automatically re-syncs tracked InMemoryRLSRepository Git repos on HEAD/WT change."""
    if not isinstance(repo, InMemoryRLSRepository):
        return []
    matching_keys = [
        k for k in list(_TRACKED_IN_MEMORY_REPOS.keys()) if k[0] == id(repo) and k[1] == domain_id
    ]
    if not matching_keys:
        return []
    indexer = GitMerkleIndexer()
    results: list[MerkleSyncStats] = []
    for key in matching_keys:
        _repo_id, _dom_id, repo_name = key
        repo_dir, allowed_scopes, _last_sha = _TRACKED_IN_MEMORY_REPOS[key]
        if not repo_dir.exists() or not (repo_dir / ".git").exists():
            continue
        try:
            st = indexer.sync_in_memory(
                repo_dir=repo_dir,
                repo=repo,
                principal=principal,
                domain_id=domain_id,
                repo_name=repo_name,
                allowed_scopes=allowed_scopes,
                ref="HEAD",
                isolate_by_commit=False,
            )
            results.append(st)
        except Exception as exc:
            logger.warning(
                "Incremental repo sync skipped %s (%s): %s",
                repo_dir,
                type(exc).__name__,
                exc,
            )
    return results


__all__ = [
    "CODE_EXTENSIONS",
    "GitMerkleIndexer",
    "MerkleBlobDiff",
    "MerkleSyncStats",
    "compute_git_blob_sha",
    "generate_deterministic_embedding",
    "sync_tracked_in_memory_repos",
]
