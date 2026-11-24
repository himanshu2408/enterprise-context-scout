"""Tests for the Data Engineering domain plugin, multi-format chunking, and 6-check gate."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.preflight_gate import (
    build_custom_doc_chunk_records,
    embed_custom_domain_texts,
    extract_document_text,
    sync_custom_domain_corpus_if_needed,
)
from context_scout.agents.scout_workflow import ScoutWorkflow
from context_scout.cli import cmd_distill_custom_domain, cmd_init_domain
from context_scout.db.repositories import DocChunkRecord, InMemoryRLSRepository
from context_scout.ingestion.doc_parser import (
    MarkdownTableChunker,
    chunk_custom_domain_document,
    extract_sgml_docbook_text,
    parse_asciidoc_to_markdown,
    parse_rst_to_markdown,
    strip_mdx_boilerplate,
)
from context_scout.plugins.domains.data_engineering import (
    DATA_ENG_REALISM_CHECK_NAMES,
    DataEngineeringDomainPlugin,
    DataEngineeringRealismGate,
    extract_clean_quote,
    is_boilerplate_text,
)
from context_scout.plugins.registry import PluginRegistry
from context_scout.schemas.contracts import (
    DocCitationSpan,
    DocumentScope,
    DomainKnowledgeBrief,
    ImplementationContextPack,
    PrincipalContext,
    RoleType,
    build_domain_knowledge_brief_prompt,
    parse_domain_knowledge_brief_json,
)


def _sample_chunk(chunk_id_seed: str, text: str) -> DocChunkRecord:
    return DocChunkRecord(
        chunk_id=uuid4(),
        domain_id="data_engineering_scout",
        doc_id=f"data_engineering_scout:{chunk_id_seed}",
        doc_title=f"Doc {chunk_id_seed}",
        source_uri=f"urn:data_engineering_scout:{chunk_id_seed}",
        market_scope="ENTERPRISE_DOC",
        published_date=None,
        authority_weight=0.95,
        section_heading="Architecture",
        contextual_header=f"Domain: data_engineering_scout | {chunk_id_seed}",
        verbatim_content=text,
        metadata_json={"rel_path": chunk_id_seed},
        allowed_scopes=["GLOBAL", "ENTERPRISE_DOC", "INTERNAL"],
        embedding=[0.01] * 768,
        source_type="enterprise_doc",
    )


def _make_valid_brief(
    quote: str,
    chunk_id: str,
    doc_id: str = "data_engineering_scout:doc1",
    source_uri: str = "urn:data_engineering_scout:doc1",
) -> DomainKnowledgeBrief:
    return DomainKnowledgeBrief(
        brief_title="Iceberg Merge-on-Read & Streaming Sink Trade-offs",
        domain_id="data_engineering_scout",
        executive_summary=(
            "Merge-on-read tables write equality deletes on ingestion and rely on periodic "
            "compaction (`rewrite_data_files`) to bound read amplification under partitioned "
            "scans. Streaming sinks use idempotent upserts and bounded WAL replication slots."
        ),
        formal_expression_or_rule=(
            "Total Read Cost = Base Parquet Scan + Delete Reconciliation; "
            "compact when delete-file ratio exceeds threshold."
        ),
        core_findings_and_rules=[
            "Equality deletes trade write speed for read amplification until "
            "rewrite_data_files compacts them.",
            "Partition pruning and Z-order sort keys bound analytical table scan latency.",
        ],
        domain_constraints_and_caveats=[
            "Requires scheduled compaction and snapshot expiration (`expire_snapshots`) "
            "for bounded storage.",
            "Flink two-phase commit requires transactional or idempotent sinks under "
            "checkpoint recovery.",
        ],
        known_risks_and_failure_modes=[
            "Uncompacted equality deletes cause OOM and severe read amplification across "
            "wide scans.",
            "Unbounded PostgreSQL replication slots accumulate WAL until disk exhaustion "
            "without max_slot_wal_keep_size.",
        ],
        structured_metadata={"table_format": "iceberg_v2", "mode": "merge-on-read"},
        verified_citations=[
            DocCitationSpan(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_title="Iceberg Spec",
                source_uri=source_uri,
                section_heading="Architecture",
                verbatim_quote=quote,
            )
        ],
    )


def test_markdown_chunker_preserves_fenced_sql_and_code_blocks_atomically() -> None:
    padding = "comment padding to cross token threshold without splitting fence " * 25
    sql_block = "\n".join(
        [
            "```sql",
            "MERGE INTO analytics.dim_customer AS target",
            "USING staging.customer_updates AS src",
            "ON target.customer_id = src.customer_id AND target.is_current = TRUE",
            "WHEN MATCHED AND target.email IS DISTINCT FROM src.email THEN",
            "  UPDATE SET valid_to = src.updated_at, is_current = FALSE",
            "WHEN NOT MATCHED THEN",
            "  INSERT (customer_id, email, valid_from, valid_to, is_current)",
            "  VALUES (src.customer_id, src.email, src.updated_at, '9999-12-31', TRUE);",
            f"-- {padding}",
            "```",
        ]
    )
    prose_prefix = "## Temporal MERGE Pattern\n\n" + (
        "SCD Type 2 temporal updates require atomic boundaries. " * 25
    )
    prose_suffix = "\n\n## Compaction Follow-Up\n\n" + (
        "Schedule rewrite_data_files after bulk merges. " * 15
    )

    chunks = MarkdownTableChunker().parse_and_chunk(
        doc_id="doc_fence_1",
        doc_title="Temporal MERGE Pattern",
        source_uri="urn:test:merge.md",
        market_scope=DocumentScope.GLOBAL_MULTI,
        markdown_content=prose_prefix + "\n\n" + sql_block + prose_suffix,
        min_chunk_tokens=150,
        max_chunk_tokens=300,
    )

    assert len(chunks) >= 2
    fence_chunks = [c for c in chunks if "MERGE INTO analytics.dim_customer" in c.verbatim_content]
    assert len(fence_chunks) == 1
    assert fence_chunks[0].verbatim_content.count("```") == 2
    assert "VALUES (src.customer_id, src.email" in fence_chunks[0].verbatim_content


def test_multi_format_extractors_and_chunker_handle_sgml_mdx_adoc_rst_and_sql(
    tmp_path: Path,
) -> None:
    sgml_raw = """<!DOCTYPE sect1>
<sect1 id="sql-merge">
  <title>PostgreSQL MERGE &amp; MVCC</title>
  <para>Use <command>MERGE</command> when predicates have cardinality &lt;= 1.</para>
  <programlisting>
MERGE INTO accounts t USING tx s ON t.id = s.id
WHEN MATCHED THEN UPDATE SET balance = t.balance + s.delta;
  </programlisting>
</sect1>"""
    sgml_clean = extract_sgml_docbook_text(sgml_raw)
    assert "## PostgreSQL MERGE & MVCC" in sgml_clean
    assert "<=" in sgml_clean
    assert "```sql\nMERGE INTO accounts" in sgml_clean
    assert "<sect1" not in sgml_clean

    mdx_raw = (
        "import Tabs from '@theme/Tabs';\n"
        "export const meta = { title: 'Flink EOS' };\n\n"
        "# Checkpointing\n\n"
        "Aligned checkpoints enforce barrier synchronization.\n"
    )
    assert "import Tabs" not in strip_mdx_boilerplate(mdx_raw)
    assert "# Checkpointing" in strip_mdx_boilerplate(mdx_raw)

    adoc_raw = (
        "= Apache Iceberg Spec\n"
        "== Partition Evolution\n"
        "Hidden partitioning avoids physical rewrites.\n"
    )
    adoc_md = parse_asciidoc_to_markdown(adoc_raw)
    assert "# Apache Iceberg Spec" in adoc_md
    assert "## Partition Evolution" in adoc_md

    rst_raw = "Spark AQE\n=========\n\nSkew Join\n---------\nCoalesces post-shuffle partitions.\n"
    rst_md = parse_rst_to_markdown(rst_raw)
    assert "# Spark AQE" in rst_md
    assert "## Skew Join" in rst_md

    sql_path = tmp_path / "patterns" / "scd2.sql"
    sql_path.parent.mkdir(parents=True, exist_ok=True)
    sql_body = (
        "MERGE INTO warehouse.orders t USING stage.orders s "
        "ON t.id = s.id WHEN MATCHED THEN UPDATE SET status = s.status;"
    )
    sql_path.write_text(sql_body, encoding="utf-8")

    sql_chunks = chunk_custom_domain_document(
        file_path=sql_path,
        text=sql_body,
        raw_bytes=sql_body.encode("utf-8"),
        domain_id="data_engineering_scout",
        rel_path="patterns/scd2.sql",
        blob_sha="abc123",
    )
    assert len(sql_chunks) == 1
    assert "```sql" in sql_chunks[0].verbatim_content
    assert sql_body in sql_chunks[0].verbatim_content

    sgml_path = tmp_path / "pg_merge.sgml"
    sgml_path.write_text(sgml_raw, encoding="utf-8")
    assert "## PostgreSQL MERGE & MVCC" in extract_document_text(sgml_path)


def test_embed_custom_domain_texts_default_and_test_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCOUT_USE_HASH_EMBEDDINGS_FOR_TESTS", "1")
    hash_vecs = embed_custom_domain_texts(["iceberg compaction"], batch_size=16)
    assert len(hash_vecs) == 1
    assert len(hash_vecs[0]) == 768

    monkeypatch.delenv("SCOUT_USE_HASH_EMBEDDINGS_FOR_TESTS", raising=False)
    mock_embedder = SimpleNamespace(
        embed_batch=lambda texts, is_query=False, batch_size=64: [[0.25] * 768 for _ in texts]
    )
    with patch(
        "context_scout.agents.preflight_gate._get_custom_doc_embedder",
        return_value=mock_embedder,
    ):
        real_vecs = embed_custom_domain_texts(["iceberg compaction", "spark aqe"], batch_size=32)

    assert len(real_vecs) == 2
    assert real_vecs[0][0] == pytest.approx(0.25)


def test_multi_chunk_build_and_sync_deletes_stale_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCOUT_USE_HASH_EMBEDDINGS_FOR_TESTS", "1")

    doc_path = tmp_path / "docs" / "iceberg.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    long_doc = (
        "# Iceberg Architecture\n\n"
        + ("Apache Iceberg uses immutable manifest lists and atomic snapshot pointers. " * 65)
        + "\n\n## Compaction\n\n"
        + (
            "Merge-on-read produces equality deletes that rewrite_data_files compacts cleanly. "
            * 65
        )
    )
    doc_path.write_text(long_doc, encoding="utf-8")

    records = build_custom_doc_chunk_records(
        domain_id="data_engineering_scout",
        file_entries=[(doc_path, "docs/iceberg.md", long_doc, long_doc.encode("utf-8"), "sha1")],
        batch_size=32,
    )
    assert len(records) >= 2
    assert len({r.chunk_id for r in records}) == len(records)
    assert all(r.market_scope == "ENTERPRISE_DOC" for r in records)

    # Verify incremental multi-chunk sync tracks and removes all stale chunk_ids
    stored_docs: dict[UUID, DocChunkRecord] = {}

    class _FakeCursor:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self._rows = rows

        def fetchall(self) -> list[tuple[object, ...]]:
            return self._rows

        def fetchone(self) -> tuple[object, ...] | None:
            return self._rows[0] if self._rows else None

    class _FakeConn:
        def execute(self, sql: str, params: tuple[object, ...] = ()) -> _FakeCursor:
            _ = params
            if "FROM code_files" in sql:
                return _FakeCursor([])
            if "COUNT(*)" in sql:
                return _FakeCursor([(0,)])
            if "FROM doc_chunks" in sql:
                rows: list[tuple[object, ...]]
                if "coalesce(metadata_json->>'rel_path'" in sql:
                    rows = [
                        (
                            str(rec.chunk_id),
                            str(
                                rec.metadata_json.get("rel_path")
                                or rec.metadata_json.get("filename")
                                or ""
                            ),
                            str(rec.metadata_json.get("blob_sha") or ""),
                            "" if rec.metadata_json.get("blob_sha") else rec.verbatim_content,
                        )
                        for rec in stored_docs.values()
                    ]
                else:
                    rows = [
                        (rec.chunk_id, json.dumps(rec.metadata_json), rec.verbatim_content)
                        for rec in stored_docs.values()
                    ]
                return _FakeCursor(rows)
            return _FakeCursor([])

    class _FakeRLSManager:
        def __init__(self, *args: object, **kwargs: object) -> None:
            _ = (args, kwargs)

        @contextmanager
        def scoped_transaction(self, principal: object):  # type: ignore[no-untyped-def]
            _ = principal
            yield _FakeConn()

    def _fake_persist(
        *,
        domain_id: str,
        display_name: str,
        kind: str,
        manifest_config: dict[str, object],
        code_files: list[object],
        ast_symbols: list[object],
        ast_edges: list[object],
        doc_chunks: list[DocChunkRecord],
        deleted_doc_ids: list[UUID] | None = None,
        **kwargs: object,
    ) -> tuple[int, int, int, int]:
        _ = (domain_id, display_name, kind, manifest_config, code_files, ast_symbols, ast_edges)
        _ = kwargs
        for del_id in deleted_doc_ids or []:
            stored_docs.pop(del_id, None)
        for ch in doc_chunks:
            stored_docs[ch.chunk_id] = ch
        return 0, 0, 0, len(stored_docs)

    domains_dir = tmp_path / "domains"
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = corpus_dir / "iceberg.md"
    corpus_file.write_text(long_doc, encoding="utf-8")

    monkeypatch.setattr(
        "context_scout.cli.persist_custom_domain_to_pg",
        lambda *a, **kw: (0, 0, 0, 0),
    )
    cmd_init_domain(
        domain_id="eval_de_sync_tmp",
        kind="knowledge",
        display_name="Eval DE Sync",
        domains_dir=domains_dir,
        source_dir=corpus_dir,
    )
    monkeypatch.setenv("SCOUT_DOMAINS_DIR", str(domains_dir))

    with (
        patch("context_scout.agents.preflight_gate.RLSConnectionManager", _FakeRLSManager),
        patch(
            "context_scout.agents.preflight_gate.get_ingestion_pool",
            lambda *a, **kw: _FakeRLSManager(),
        ),
        patch(
            "context_scout.agents.preflight_gate.get_query_pool",
            lambda *a, **kw: _FakeRLSManager(),
        ),
        patch("context_scout.agents.preflight_gate.persist_custom_domain_to_pg", _fake_persist),
    ):
        res1 = sync_custom_domain_corpus_if_needed(domain_id="eval_de_sync_tmp")
        assert res1 is not None
        assert res1["added"] == 1
        assert len(stored_docs) >= 2
        initial_chunk_count = len(stored_docs)

        corpus_file.unlink()
        res2 = sync_custom_domain_corpus_if_needed(domain_id="eval_de_sync_tmp")
        assert res2 is not None
        assert res2["deleted"] == 1
        assert initial_chunk_count >= 2
        assert stored_docs == {}


def test_cli_distill_custom_domain_handles_multi_chunk_and_unreadable_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCOUT_USE_HASH_EMBEDDINGS_FOR_TESTS", "1")
    domains_dir = tmp_path / "domains"
    data_dir = tmp_path / "data"
    corpus_dir = tmp_path / "src_docs"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    (corpus_dir / "lakehouse.md").write_text(
        "# Lakehouse Storage\n\n"
        + ("Apache Iceberg compaction rewrite_data_files removes equality deletes safely. " * 60),
        encoding="utf-8",
    )
    (corpus_dir / "broken.pdf").write_bytes(b"not-a-valid-pdf")

    monkeypatch.setattr(
        "context_scout.cli.persist_custom_domain_to_pg",
        lambda *a, **kw: (0, 0, 0, 0),
    )
    cmd_init_domain(
        domain_id="eval_de_distill_tmp",
        kind="knowledge",
        display_name="Eval DE Distill",
        domains_dir=domains_dir,
        source_dir=corpus_dir,
    )

    def _fake_cli_persist(**kwargs: object) -> tuple[int, int, int, int]:
        doc_chunks_arg = kwargs.get("doc_chunks") or []
        return 0, 0, 0, len(doc_chunks_arg) if isinstance(doc_chunks_arg, list) else 0

    class _DummyCursor:
        def fetchone(self) -> None:
            return None

        def fetchall(self) -> list[tuple[object, ...]]:
            return []

    @contextmanager
    def _fake_tx(*_a: object, **_kw: object):  # type: ignore[no-untyped-def]
        class _DummyConn:
            def execute(self, *_args: object, **_kwargs: object) -> _DummyCursor:
                return _DummyCursor()

        yield _DummyConn()

    class _FakePool:
        scoped_transaction = staticmethod(_fake_tx)

    with (
        patch("context_scout.cli.persist_custom_domain_to_pg", side_effect=_fake_cli_persist),
        patch("context_scout.agents.preflight_gate.get_ingestion_pool", return_value=_FakePool()),
        patch("context_scout.agents.preflight_gate.get_query_pool", return_value=_FakePool()),
    ):
        report = cmd_distill_custom_domain(
            domain_id="eval_de_distill_tmp",
            source_dir=corpus_dir,
            num_samples=4,
            domains_dir=domains_dir,
            data_dir=data_dir,
        )

    assert report["files_indexed"] >= 1
    assert report["train_samples"] >= 1
    assert (data_dir / "eval_de_distill_tmp" / "train.jsonl").exists()


def test_data_engineering_realism_gate_positive_and_negative_controls() -> None:
    gate = DataEngineeringRealismGate()
    verbatim_quote = (
        "Iceberg merge-on-read tables write equality deletes on commit and require "
        "rewrite_data_files compaction, partition pruning, atomic MERGE transactions, "
        "max_slot_wal_keep_size bounds, and benchmark workload specifications."
    )
    chunk = _sample_chunk("docs/spec.md", verbatim_quote)

    valid_brief = _make_valid_brief(
        quote=verbatim_quote,
        chunk_id=str(chunk.chunk_id),
        doc_id=chunk.doc_id,
        source_uri=chunk.source_uri,
    )
    ok_res = gate.evaluate_brief(valid_brief, corpus_chunks=[chunk])
    assert ok_res.passed is True
    assert ok_res.checks_evaluated == 6
    assert ok_res.failed_checks == []
    assert ok_res.citations_valid is True

    empty_brief = DomainKnowledgeBrief(
        brief_title="   ",
        domain_id="data_engineering_scout",
        executive_summary="   ",
        core_findings_and_rules=["placeholder"],
        domain_constraints_and_caveats=["placeholder"],
        known_risks_and_failure_modes=["placeholder"],
        verified_citations=[
            DocCitationSpan(
                chunk_id=str(chunk.chunk_id),
                doc_id=chunk.doc_id,
                doc_title="Iceberg Spec",
                source_uri=chunk.source_uri,
                section_heading="Architecture",
                verbatim_quote=verbatim_quote,
            )
        ],
    )
    empty_res = gate.evaluate_brief(empty_brief, corpus_chunks=[chunk])
    assert empty_res.passed is False
    assert "EMPTY_OR_UNGROUNDED_BRIEF" in empty_res.failed_checks

    neutral_quote = "Data platform architecture design note for internal review."
    neutral_chunk = _sample_chunk("docs/neutral.md", neutral_quote)

    cases: list[tuple[str, str, str, list[str]]] = [
        (
            "EOS_WITHOUT_SINK_IDEMPOTENCY",
            "Streaming Pipeline",
            "Flink provides end-to-end guarantee with exactly-once processing across jobs.",
            ["Two-phase commit coordinates checkpoints across operators."],
        ),
        (
            "UNPARTITIONED_SCAN_RISK",
            "Warehouse Aggregation",
            "Queries perform a full scan on the raw unpartitioned events table.",
            ["Shuffle join aggregates daily metrics across the whole dataset."],
        ),
        (
            "ICEBERG_EQUALITY_DELETE_UNCOMPACTED",
            "Iceberg Ingestion",
            "Tables run in merge-on-read mode writing equality delete files per commit.",
            ["Deletion vector markers accumulate across commits."],
        ),
        (
            "SCD2_NON_ATOMIC_OR_OVERLAP",
            "Customer Dimension",
            "Daily batch jobs overwrite SCD Type 2 rows in dbt snapshot tables.",
            ["Slowly changing dimension updates append new rows after deletes."],
        ),
        (
            "CDC_SLOT_WAL_UNBOUNDED",
            "CDC Pipeline",
            "Debezium uses a PostgreSQL pgoutput logical replication slot to stream events.",
            ["Change data capture reads directly from primary instance."],
        ),
        (
            "UNVERIFIED_THROUGHPUT_CLAIM",
            "Engine Speedup",
            "The new vectorized engine achieves 5x faster scans and 2.5 GB/s throughput.",
            ["Maintains sub-second latency for dashboard queries."],
        ),
    ]

    assert tuple(c[0] for c in cases) == DATA_ENG_REALISM_CHECK_NAMES

    for expected_check, title, summary, findings in cases:
        bad_brief = DomainKnowledgeBrief(
            brief_title=title,
            domain_id="data_engineering_scout",
            executive_summary=summary,
            core_findings_and_rules=findings,
            domain_constraints_and_caveats=["Operational caveat pending review."],
            known_risks_and_failure_modes=["Operational risk pending review."],
            verified_citations=[
                DocCitationSpan(
                    chunk_id=str(neutral_chunk.chunk_id),
                    doc_id=neutral_chunk.doc_id,
                    doc_title="Neutral Doc",
                    source_uri=neutral_chunk.source_uri,
                    section_heading="Architecture",
                    verbatim_quote=neutral_quote,
                )
            ],
        )
        res = gate.evaluate_brief(bad_brief, corpus_chunks=[neutral_chunk])
        assert res.passed is False, f"Expected failure for {expected_check}"
        assert expected_check in res.failed_checks, (
            f"Expected {expected_check} in {res.failed_checks}"
        )


def test_scout_workflow_synthesizes_domain_knowledge_brief_on_data_engineering_scout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCOUT_USE_HASH_EMBEDDINGS_FOR_TESTS", "1")
    repo = InMemoryRLSRepository()
    plugin = PluginRegistry.get_domain_plugin("data_engineering_scout")
    assert isinstance(plugin, DataEngineeringDomainPlugin)

    quote_iceberg = (
        "Apache Iceberg merge-on-read tables emit equality deletes during streaming upserts and "
        "require scheduled rewrite_data_files compaction and snapshot expiration to bound read "
        "amplification."
    )
    quote_sql = (
        "MERGE INTO warehouse.customer_scd2 AS tgt USING stage.customers AS src "
        "ON tgt.customer_id = src.customer_id AND tgt.is_current = TRUE "
        "WHEN MATCHED THEN UPDATE SET valid_to = src.ts, is_current = FALSE."
    )
    c1 = _sample_chunk("iceberg/compaction.md", quote_iceberg)
    c2 = _sample_chunk("patterns/scd2.sql", quote_sql)

    principal = PrincipalContext(
        user_id="test_de_engineer",
        role=RoleType.ENGINEER,
        allowed_scopes=["GLOBAL", "ENTERPRISE_DOC", "INTERNAL"],
    )
    repo.insert_doc_chunk(principal, c1)
    repo.insert_doc_chunk(principal, c2)

    workflow = ScoutWorkflow(registry=DomainRegistry(domains_dir=Path("domains")), repo=repo)
    pack = workflow.run(
        user_prompt=(
            "Compare Iceberg merge-on-read equality deletes compaction "
            "against SQL MERGE SCD2 patterns."
        ),
        principal=principal,
        domain_id="data_engineering_scout",
    )

    assert isinstance(pack, ImplementationContextPack)
    assert pack.domain_knowledge_brief is not None
    assert pack.domain_knowledge_brief.verified_citations
    assert pack.preflight_report.passed is True


def test_data_engineering_prompt_parity_boilerplate_filter_and_raw_decode() -> None:
    assert is_boilerplate_text("Licensed to the Apache Software Foundation (ASF) under one")
    assert is_boilerplate_text("* [SELECT Main](sql-ref-syntax-qry-select.html)")
    assert not is_boilerplate_text(
        "Apache Iceberg compaction merges small Parquet files and applies position deletes."
    )

    chunk_text = (
        "Licensed to the Apache Software Foundation (ASF) under one or more agreements.\n"
        "* [Overview](overview.html)\n"
        "Merge-on-Read defers Parquet rewrites by writing delta logs that readers merge on demand."
    )
    quote = extract_clean_quote(chunk_text, max_chars=220)
    assert "Merge-on-Read defers Parquet rewrites" in quote
    assert len(quote) <= 220

    plugin = DataEngineeringDomainPlugin()
    c1 = _sample_chunk("hudi/mor.md", chunk_text)
    sample = plugin.build_multi_chunk_distillation_sample(
        domain_id="data_engineering_scout",
        chunks=[c1],
        quotes=[quote],
        rng=None,
    )
    expected_prompt = build_domain_knowledge_brief_prompt(
        user_prompt=sample["val_user_prompt"],
        domain_id="data_engineering_scout",
        chunk_lines=[
            f"Retrieved Corpus Chunk: [chunk_id={c1.chunk_id} | doc_id={c1.doc_id} | "
            f"title={c1.doc_title} | uri={c1.source_uri} | section={c1.section_heading} | "
            f"quote={json.dumps(quote)}]"
        ],
    )
    assert sample["user_prompt"] == expected_prompt

    # raw_decode handles trailing notes and <turn|>; negative control rejects list metadata strictly
    payload = sample["brief"].model_dump()
    payload["structured_metadata"] = {"tool": "Hudi", "topics": ["CoW", "MoR"]}
    raw_with_note = json.dumps(payload) + "\n*(Note: trailing explanation.)*<turn|>"

    strict_brief, strict_ok = parse_domain_knowledge_brief_json(
        raw_with_note,
        domain_id="data_engineering_scout",
        normalize_scalars=False,
    )
    assert strict_brief is None
    assert strict_ok is False

    norm_brief, norm_strict_ok = parse_domain_knowledge_brief_json(
        raw_with_note,
        domain_id="data_engineering_scout",
        normalize_scalars=True,
    )
    assert norm_brief is not None
    assert norm_strict_ok is False
    assert norm_brief.structured_metadata["topics"] == "CoW, MoR"


def test_teacher_distiller_split_filter_excludes_eval_holdout() -> None:
    from dataclasses import replace

    from context_scout.tuning.teacher_distiller import TeacherDistiller

    train_c = replace(
        _sample_chunk(
            "train/repos/apache__iceberg/compaction.md",
            "Iceberg rewrite_data_files merges small Parquet files and converts equality deletes "
            "to keep scan latency predictable across large partitioned lakehouse tables.",
        ),
        doc_id="data_engineering_scout:train/repos/apache__iceberg/compaction.md",
        metadata_json={"split": "train", "sub_scope": "STORAGE_LAKEHOUSE"},
    )
    train_c2 = replace(
        _sample_chunk(
            "train/repos/apache__iceberg/snapshots.md",
            "Expire snapshots removes orphaned Iceberg manifest lists after compaction finishes "
            "while preserving active time-travel checkpoints cleanly.",
        ),
        doc_id="data_engineering_scout:train/repos/apache__iceberg/snapshots.md",
        metadata_json={"split": "train", "sub_scope": "STORAGE_LAKEHOUSE"},
    )
    holdout_c = replace(
        _sample_chunk(
            "eval_holdout/repos/SQLMesh__sqlmesh/environments.md",
            "SQLMesh virtual data environments isolate backfill plans from production views "
            "via versioned snapshots and atomic pointer swaps.",
        ),
        doc_id="data_engineering_scout:eval_holdout/repos/SQLMesh__sqlmesh/environments.md",
        metadata_json={"split": "eval_holdout", "sub_scope": "MODELING_QUALITY_GOV"},
    )

    with (
        patch.object(TeacherDistiller, "get_all_code_files", return_value=[]),
        patch.object(TeacherDistiller, "get_all_ast_symbols", return_value=[]),
        patch.object(TeacherDistiller, "get_all_ast_edges", return_value=[]),
        patch.object(
            TeacherDistiller,
            "get_all_doc_chunks",
            return_value=[train_c, train_c2, holdout_c],
        ),
    ):
        distiller = TeacherDistiller(research_domain_id="data_engineering_scout")
        train_exs, valid_exs = distiller.distill_research_domain(
            domain_id="data_engineering_scout",
            num_samples=6,
            force_offline=True,
            split_filter="train",
        )
    combined_text = json.dumps(train_exs + valid_exs)
    assert "SQLMesh" not in combined_text
    assert "eval_holdout" not in combined_text
    assert "Iceberg" in combined_text
