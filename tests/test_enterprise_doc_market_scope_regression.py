"""Regression tests for non-finance DocChunk market_scope parsing and Data Engineering queries."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient

from context_scout.agents.scout_workflow import LocalGemmaClient
from context_scout.api.dependencies import (
    DEFAULT_ROLE_SCOPES,
    create_signed_principal_token,
    get_local_gemma_client,
    get_repository,
)
from context_scout.api.main import app
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
)
from context_scout.mcp_servers.common import PostgresScoutGraphAdapter
from context_scout.schemas.contracts import DocumentScope, PrincipalContext, RoleType

_ACID_QUOTE = (
    "ACID stands for Atomicity, Consistency, Isolation, and Durability, "
    "guaranteeing reliable transactional processing in relational databases."
)


def _make_doc_row(
    *,
    chunk_id: uuid.UUID,
    doc_id: str,
    doc_title: str,
    section_heading: str,
    verbatim_content: str,
    raw_market_scope: object,
) -> tuple[object, ...]:
    return (
        chunk_id,
        "data_engineering_scout",
        doc_id,
        doc_title,
        f"urn:data_engineering_scout:{doc_id}",
        raw_market_scope,
        date(2026, 1, 15),
        1.0,
        section_heading,
        f"{doc_title} — {section_heading}",
        verbatim_content,
        {"rel_path": f"{doc_id}.md", "split": "train"},
        ["public_eng", "domain:data_engineering_scout"],
    )


class _FakeEnterpriseDocRepo:
    """Deterministic in-memory repo backed by PostgresScoutGraphAdapter._row_to_doc_chunk."""

    def __init__(self, chunks: list[DocChunkRecord]) -> None:
        self._chunks = chunks

    def list_visible_code_files(
        self,
        principal: PrincipalContext,
        domain_id: str,
    ) -> list[CodeFileRecord]:
        _ = (principal, domain_id)
        return []

    def list_visible_ast_symbols(
        self,
        principal: PrincipalContext,
        domain_id: str,
    ) -> list[ASTSymbolRecord]:
        _ = (principal, domain_id)
        return []

    def list_visible_ast_edges(
        self,
        principal: PrincipalContext,
        domain_id: str,
    ) -> list[ASTEdgeRecord]:
        _ = (principal, domain_id)
        return []

    def list_visible_doc_chunks(
        self,
        principal: PrincipalContext,
        domain_id: str,
        market_scope: DocumentScope | str | None = None,
    ) -> list[DocChunkRecord]:
        _ = (principal, market_scope)
        return [c for c in self._chunks if c.domain_id == domain_id]

    def search_visible_doc_chunks(
        self,
        principal: PrincipalContext,
        domain_id: str,
        query_text: str,
        market_scope: DocumentScope | str | None = None,
        limit: int = 50,
    ) -> list[DocChunkRecord]:
        _ = (principal, query_text, market_scope)
        return [c for c in self._chunks if c.domain_id == domain_id][:limit]

    def get_doc_chunks_by_ids(
        self,
        principal: PrincipalContext,
        domain_id: str,
        chunk_ids: list[str],
    ) -> list[DocChunkRecord]:
        _ = principal
        wanted = set(chunk_ids)
        return [c for c in self._chunks if c.domain_id == domain_id and str(c.chunk_id) in wanted]


class _FakeCursor:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._rows


class _RecordingConnection:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self._rows = rows
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> _FakeCursor:
        self.executed.append((sql, params))
        return _FakeCursor(self._rows)


class _FakeConnectionManager:
    def __init__(self, conn: _RecordingConnection) -> None:
        self._conn = conn

    @contextmanager
    def scoped_transaction(
        self,
        principal: PrincipalContext,
        enable_hnsw_iterative_scan: bool = True,
    ) -> Iterator[_RecordingConnection]:
        _ = (principal, enable_hnsw_iterative_scan)
        yield self._conn


def test_row_to_doc_chunk_parses_enterprise_doc_and_negative_control() -> None:
    """Proves DocumentScope parses ENTERPRISE_DOC and REGIONAL_US while preserving None."""
    assert DocumentScope("ENTERPRISE_DOC") == DocumentScope.ENTERPRISE_DOC

    enterprise_chunk = PostgresScoutGraphAdapter._row_to_doc_chunk(
        _make_doc_row(
            chunk_id=uuid.uuid4(),
            doc_id="acid_tx",
            doc_title="ACID Transactions",
            section_heading="ACID Properties",
            verbatim_content=_ACID_QUOTE,
            raw_market_scope="ENTERPRISE_DOC",
        )
    )
    regional_chunk = PostgresScoutGraphAdapter._row_to_doc_chunk(
        _make_doc_row(
            chunk_id=uuid.uuid4(),
            doc_id="reg_doc",
            doc_title="US Regional Compliance",
            section_heading="Overview",
            verbatim_content="US regional compliance semantics.",
            raw_market_scope="REGIONAL_US",
        )
    )
    none_chunk = PostgresScoutGraphAdapter._row_to_doc_chunk(
        _make_doc_row(
            chunk_id=uuid.uuid4(),
            doc_id="none_doc",
            doc_title="Unscoped Doc",
            section_heading="Overview",
            verbatim_content="Unscoped metadata check.",
            raw_market_scope=None,
        )
    )

    assert enterprise_chunk.market_scope == DocumentScope.ENTERPRISE_DOC
    assert regional_chunk.market_scope == DocumentScope.REGIONAL_US
    assert none_chunk.market_scope is None


def test_adapter_search_filters_acronym_stopwords_and_parses_enterprise_doc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies fallback keyword SQL drops 'does'/'stand' noise and returns ENTERPRISE_DOC rows."""
    monkeypatch.setenv("SCOUT_USE_HASH_EMBEDDINGS_FOR_TESTS", "1")
    row = _make_doc_row(
        chunk_id=uuid.uuid4(),
        doc_id="acid_tx",
        doc_title="Transaction Management",
        section_heading="ACID Properties",
        verbatim_content=_ACID_QUOTE,
        raw_market_scope="ENTERPRISE_DOC",
    )
    conn = _RecordingConnection([row])
    adapter = PostgresScoutGraphAdapter(connection_manager=_FakeConnectionManager(conn))  # type: ignore[arg-type]
    principal = PrincipalContext(
        user_id="eng_test",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )

    results = adapter.search_visible_doc_chunks(
        principal=principal,
        domain_id="data_engineering_scout",
        query_text="what does ACID stand for?",
        limit=5,
    )

    assert len(results) == 1
    assert results[0].market_scope == "ENTERPRISE_DOC"
    assert results[0].verbatim_content == _ACID_QUOTE
    assert conn.executed
    executed_sql, executed_params = conn.executed[0]
    assert "ORDER BY 0" not in executed_sql
    tsquery_arg = str(executed_params[0])
    assert tsquery_arg == "acid"


def test_fastapi_query_and_stream_data_engineering_enterprise_doc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercises /api/v1/scout/query and /api/v1/scout/stream with ENTERPRISE_DOC ACID chunks."""
    monkeypatch.delenv("CONTEXT_SCOUT_USE_LIVE_MLX", raising=False)
    acid_chunk = PostgresScoutGraphAdapter._row_to_doc_chunk(
        _make_doc_row(
            chunk_id=uuid.uuid4(),
            doc_id="pg_acid",
            doc_title="Relational Transaction Guarantees",
            section_heading="ACID Overview",
            verbatim_content=_ACID_QUOTE,
            raw_market_scope="ENTERPRISE_DOC",
        )
    )
    fake_repo = _FakeEnterpriseDocRepo([acid_chunk])
    offline_client = LocalGemmaClient(offline_mode=True)

    app.dependency_overrides[get_repository] = lambda: fake_repo
    app.dependency_overrides[get_local_gemma_client] = lambda: offline_client

    token = create_signed_principal_token(
        user_id="de_engineer",
        role=RoleType.ENGINEER,
        allowed_scopes=DEFAULT_ROLE_SCOPES[RoleType.ENGINEER],
    )
    client = TestClient(app, headers={"Authorization": f"Bearer {token}"})

    try:
        query_resp = client.post(
            "/api/v1/scout/query",
            json={
                "query": "what does ACID stand for?",
                "domain_id": "data_engineering_scout",
                "normalized_rrf_score": 0.9,
            },
        )
        assert query_resp.status_code == 200, query_resp.text
        query_data: dict[str, Any] = query_resp.json()
        assert query_data["passed"] is True
        citations = query_data.get("citations", [])
        assert len(citations) >= 1
        assert any(_ACID_QUOTE in c.get("verbatim_quote", "") for c in citations)

        with client.stream(
            "POST",
            "/api/v1/scout/stream",
            json={
                "query": "what does ACID stand for?",
                "domain_id": "data_engineering_scout",
                "normalized_rrf_score": 0.9,
            },
        ) as stream_resp:
            assert stream_resp.status_code == 200
            raw_sse = stream_resp.read().decode("utf-8")

        assert "event: context_pack_ready" in raw_sse
        ready_payload: dict[str, Any] | None = None
        for block in raw_sse.split("\n\n"):
            if "event: context_pack_ready" in block:
                for line in block.splitlines():
                    if line.startswith("data: "):
                        ready_payload = json.loads(line[len("data: ") :])
                        break
        assert ready_payload is not None
        pack_citations = ready_payload["pack"]["supporting_doc_citations"]
        assert len(pack_citations) >= 1
        assert any(_ACID_QUOTE in c.get("verbatim_quote", "") for c in pack_citations)
    finally:
        app.dependency_overrides.clear()
