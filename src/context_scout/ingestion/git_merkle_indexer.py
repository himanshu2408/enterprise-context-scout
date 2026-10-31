"""Git Blob-SHA Merkle incremental indexer with Tree-Sitter AST extraction and RLS persistence."""

import hashlib
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Final, Literal
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

CODE_EXTENSIONS: Final[set[str]] = {".py", ".ts", ".tsx"}


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
}


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
        except Exception:
            pass

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

    def __init__(self, extractor: ASTExtractor | None = None) -> None:
        self.extractor = extractor or ASTExtractor()

    def get_git_tree(self, repo_dir: Path, ref: str = "HEAD") -> tuple[str, dict[str, str]]:
        """Extracts current commit SHA and mapping of {file_path: blob_sha} via git ls-tree."""
        commit_sha = _resolve_commit_sha(repo_dir, ref)

        ls_cmd = subprocess.run(
            ["git", "ls-tree", "-r", ref],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        tree: dict[str, str] = {}
        for line in ls_cmd.stdout.strip().splitlines():
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            meta, file_path = parts
            meta_parts = meta.split()
            if len(meta_parts) >= 3:
                blob_sha = meta_parts[2]
                tree[file_path] = blob_sha

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

    def _detect_language(self, file_path: str) -> str | None:
        for ext, lang in LANG_MAP.items():
            if file_path.endswith(ext):
                return lang
        return None

    def sync_in_memory(
        self,
        repo_dir: Path,
        repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        domain_id: str,
        repo_name: str,
        allowed_scopes: list[str] | None = None,
        ref: str = "HEAD",
    ) -> MerkleSyncStats:
        """Incrementally indexes git repository into InMemoryRLSRepository."""
        start_time = time.perf_counter()
        commit_sha, current_tree = self.get_git_tree(repo_dir, ref)
        existing_hashes = repo.get_file_blob_hashes(principal, domain_id, repo_name)
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
            disk_file = repo_dir / file_path
            if not disk_file.exists():
                continue

            lang = self._detect_language(file_path)
            raw_content = disk_file.read_text(encoding="utf-8", errors="replace")
            lines = raw_content.splitlines()
            file_id = uuid4()
            blob_sha = current_tree[file_path]

            file_rec = CodeFileRecord(
                file_id=file_id,
                domain_id=domain_id,
                repo_name=repo_name,
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

            for parsed_sym in symbols:
                sym_id = uuid4()
                emb = generate_deterministic_embedding(
                    parsed_sym.signature_skeleton + " " + parsed_sym.docstring
                )
                kind: Literal["function", "class", "method", "interface", "type_alias"] = (
                    parsed_sym.symbol_kind
                )
                sym_rec = ASTSymbolRecord(
                    symbol_id=sym_id,
                    file_id=file_id,
                    domain_id=domain_id,
                    qualified_name=parsed_sym.qualified_name,
                    symbol_kind=kind,
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
                    # Look up by exact qualified name
                    src_id = global_symbol_map.get(edge.source_qualified_name)
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
    ) -> MerkleSyncStats:
        """Incrementally indexes git repository into live PostgreSQL 17 under Forced RLS."""
        start_time = time.perf_counter()
        commit_sha, current_tree = self.get_git_tree(repo_dir, ref)
        existing_hashes = repo.get_file_blob_hashes(conn, domain_id, repo_name)
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
            disk_file = repo_dir / file_path
            if not disk_file.exists():
                continue

            lang = self._detect_language(file_path)
            raw_content = disk_file.read_text(encoding="utf-8", errors="replace")
            lines = raw_content.splitlines()
            file_id = uuid4()
            blob_sha = current_tree[file_path]

            file_rec = CodeFileRecord(
                file_id=file_id,
                domain_id=domain_id,
                repo_name=repo_name,
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

            for parsed_sym in symbols:
                sym_id = uuid4()
                emb = generate_deterministic_embedding(
                    parsed_sym.signature_skeleton + " " + parsed_sym.docstring
                )
                kind: Literal["function", "class", "method", "interface", "type_alias"] = (
                    parsed_sym.symbol_kind
                )
                sym_rec = ASTSymbolRecord(
                    symbol_id=sym_id,
                    file_id=file_id,
                    domain_id=domain_id,
                    qualified_name=parsed_sym.qualified_name,
                    symbol_kind=kind,
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


__all__ = [
    "CODE_EXTENSIONS",
    "GitMerkleIndexer",
    "MerkleSyncStats",
    "generate_deterministic_embedding",
]
