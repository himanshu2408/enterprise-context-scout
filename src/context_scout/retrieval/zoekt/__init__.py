"""Modular Zoekt trigram client package."""

from __future__ import annotations

from context_scout.retrieval.zoekt.acl_bitmap import RepoACLBitmapRegistry, RepoACLResult
from context_scout.retrieval.zoekt.grpc_json_client import ZoektGrpcJsonClient
from context_scout.retrieval.zoekt.local_trigram_index import LocalTrigramZoektIndex
from context_scout.retrieval.zoekt.query_builder import (
    UNCAPPED_ZOEK_DISPLAY_COUNT,
    RE2Match,
    RE2Pattern,
    RE2SyntaxError,
    TrigramQueryPlan,
    ZoektClientSPI,
    ZoektFileMatch,
    ZoektLineMatch,
    ZoektSearchResponse,
    ZoektSearchStats,
    ZoektUnavailableError,
    compile_re2,
    extract_required_trigrams,
    format_zoekt_shard_repo_name,
    strip_zoekt_shard_repo_name,
    validate_re2_pattern,
)

__all__ = [
    "UNCAPPED_ZOEK_DISPLAY_COUNT",
    "LocalTrigramZoektIndex",
    "RE2Match",
    "RE2Pattern",
    "RE2SyntaxError",
    "RepoACLBitmapRegistry",
    "RepoACLResult",
    "TrigramQueryPlan",
    "ZoektClientSPI",
    "ZoektFileMatch",
    "ZoektGrpcJsonClient",
    "ZoektLineMatch",
    "ZoektSearchResponse",
    "ZoektSearchStats",
    "ZoektUnavailableError",
    "compile_re2",
    "extract_required_trigrams",
    "format_zoekt_shard_repo_name",
    "strip_zoekt_shard_repo_name",
    "validate_re2_pattern",
]
