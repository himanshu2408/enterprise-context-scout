"""Ingestion package: Git Merkle indexer and incremental source code graph synchronization."""

from context_scout.ingestion.git_merkle_indexer import (
    CODE_EXTENSIONS,
    GitMerkleIndexer,
    MerkleSyncStats,
    generate_deterministic_embedding,
)

__all__ = [
    "CODE_EXTENSIONS",
    "GitMerkleIndexer",
    "MerkleSyncStats",
    "generate_deterministic_embedding",
]
