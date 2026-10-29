"""Retrieval package: Semantic vector embeddings and hybrid search."""

from context_scout.retrieval.embedder import (
    DEFAULT_GOOGLE_EMBEDDING_MODEL,
    EXPECTED_EMBEDDING_DIM,
    LocalSemanticEmbedder,
    resolve_torch_device,
)

__all__ = [
    "DEFAULT_GOOGLE_EMBEDDING_MODEL",
    "EXPECTED_EMBEDDING_DIM",
    "LocalSemanticEmbedder",
    "resolve_torch_device",
]
