"""Zoekt gRPC/JSON trigram search client with ACL bitmaps and O(n) RE2 regex."""

from __future__ import annotations

from context_scout.retrieval.zoekt.acl_bitmap import (
    RepoACLBitmapRegistry as RepoACLBitmapRegistry,
)
from context_scout.retrieval.zoekt.acl_bitmap import (
    RepoACLResult as RepoACLResult,
)
from context_scout.retrieval.zoekt.grpc_json_client import (
    ZoektGrpcJsonClient as ZoektGrpcJsonClient,
)
from context_scout.retrieval.zoekt.local_trigram_index import (
    LocalTrigramZoektIndex as LocalTrigramZoektIndex,
)
from context_scout.retrieval.zoekt.query_builder import (
    UNCAPPED_ZOEK_DISPLAY_COUNT as UNCAPPED_ZOEK_DISPLAY_COUNT,
)
from context_scout.retrieval.zoekt.query_builder import (
    RE2Match as RE2Match,
)
from context_scout.retrieval.zoekt.query_builder import (
    RE2Pattern as RE2Pattern,
)
from context_scout.retrieval.zoekt.query_builder import (
    RE2SyntaxError as RE2SyntaxError,
)
from context_scout.retrieval.zoekt.query_builder import (
    TrigramQueryPlan as TrigramQueryPlan,
)
from context_scout.retrieval.zoekt.query_builder import (
    ZoektClientSPI as ZoektClientSPI,
)
from context_scout.retrieval.zoekt.query_builder import (
    ZoektFileMatch as ZoektFileMatch,
)
from context_scout.retrieval.zoekt.query_builder import (
    ZoektLineMatch as ZoektLineMatch,
)
from context_scout.retrieval.zoekt.query_builder import (
    ZoektSearchResponse as ZoektSearchResponse,
)
from context_scout.retrieval.zoekt.query_builder import (
    ZoektSearchStats as ZoektSearchStats,
)
from context_scout.retrieval.zoekt.query_builder import (
    ZoektUnavailableError as ZoektUnavailableError,
)
from context_scout.retrieval.zoekt.query_builder import (
    compile_re2 as compile_re2,
)
from context_scout.retrieval.zoekt.query_builder import (
    extract_required_trigrams as extract_required_trigrams,
)
from context_scout.retrieval.zoekt.query_builder import (
    format_zoekt_shard_repo_name as format_zoekt_shard_repo_name,
)
from context_scout.retrieval.zoekt.query_builder import (
    strip_zoekt_shard_repo_name as strip_zoekt_shard_repo_name,
)
from context_scout.retrieval.zoekt.query_builder import (
    validate_re2_pattern as validate_re2_pattern,
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
