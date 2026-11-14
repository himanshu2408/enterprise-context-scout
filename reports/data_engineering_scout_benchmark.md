# Data Engineering Concepts Scout (`data_engineering_scout`) — Wave 2 Benchmark Report

## 1. Corpus, Adapter & Disjointness Statement
- **Domain ID:** `data_engineering_scout` (`primary_truth_source: document`, `output_contract: DomainKnowledgeBrief`)
- **PostgreSQL 17 Visible Corpus:** `31,327` `doc_chunks` (`24,581` `train` + `6,746` `eval_holdout` across 7 unseen repos & 60 post-July-2026 arXiv papers).
- **Strict Disjointness:** `train.jsonl` (`187` trajectories) and `valid.jsonl` (`33` trajectories; `39/220 = 17.73%` 4-turn bounce-backs) were distilled strictly from `metadata_json->>'split' = 'train'` (`0` `doc_id` overlap between `train` and `valid`; `0` `eval_holdout` docs; `0` truncated sequences at `max_seq_length=4096`).
- **Frozen 4-bit MLX QLoRA Adapter (`gemma4_26b_data_engineering`):** `SHA-256 = 2e70a0b4c90bda12b4f675fa40cf25f5e2e830d177ca6afe1154e7df28cd9237` (verified unchanged before and after evaluation), `initial_loss=0.8028 -> final_loss=0.0829`, `val_loss=0.2138`, `peak_memory_gb=20.66`.

## 2. Benchmark A — Unseen Held-Out Corpus (N=40 Non-Hinted Queries)
- **Measurement Method:** Live `ScoutWorkflow` + `CONTEXT_SCOUT_USE_LIVE_MLX=1` against PG17 (`train` + `eval_holdout`), probing 24 queries across all 7 unseen repos (`SQLMesh`, `Hudi`, `Trino`, `Polars`, `Debezium`, `Dagster`, `Nessie`) and 16 queries across 2026H2 `eval_holdout` arXiv papers (`STORAGE_LAKEHOUSE`, `BATCH_MPP_ENGINES`, `STREAMING_CDC`, `MODELING_QUALITY_GOV`). Token savings compare `pack_token_count` against reading the top-5 full retrieved documents.

| Metric | Measured Result |
|---|---|
| **Recall@5 (Target Repo/Paper in Top-5)** | **37/40** (`92.50%`) |
| **Live QLoRA Schema Validity (Strict Raw)** | **13/40** (`32.50%`) |
| **Live QLoRA Schema Validity (Normalized)** | **13/40** (`32.50%`) |
| **Pre-Flight Gate Pass Rate** | **40/40** (`100.00%`) |
| **Verbatim Quote Faithfulness (>=0.98 NFKC)** | **105/105** (`100.00%`) |
| **DataEngineeringRealismGate Pass Rate** | **39/40** (`97.50%`) |
| **Median End-to-End Latency** | **21298.86 ms** |
| **Median Token Savings vs. Top-5 Full Docs** | **99.81%** |

## 3. Benchmark B — Senior Data Engineer Stress-Test (N=25 Scenarios)
- **Measurement Method:** End-to-end live-MLX `ScoutWorkflow` over `domains/data_engineering_scout/evals/senior_de_25_stress_test.json` (`DE-01..DE-25`).

| Metric | Measured Result |
|---|---|
| **Pre-Flight Gate Pass Rate** | **25/25** (`100.00%`) |
| **Live QLoRA Schema Validity (Strict Raw)** | **3/25** (`12.00%`) |
| **Live QLoRA Schema Validity (Normalized)** | **3/25** (`12.00%`) |
| **`must_cover_concepts` Coverage** | **23/140** (`16.43%`) |
| **Verbatim Quote Faithfulness (>=0.98 NFKC)** | **71/71** (`100.00%`) |
| **DataEngineeringRealismGate Pass Rate** | **22/25** (`88.00%`) |
| **Median End-to-End Latency** | **34628.04 ms** |

## 4. FastAPI `TestClient` Verification
- `GET /api/v1/domains`: HTTP `200`, `data_engineering_scout` `adapter_verified=True`.
- `POST /api/v1/scout/query`: HTTP `200`, `brief_title='Towards Unified File Management for Data Lakehouse Systems — Architectural Synthesis'`, `verified_citations=3`, `preflight_passed=True`.

## 5. Honest Misses & Limitations
- **Benchmark A Misses / Anomalies (29):**
  - `A-01` (`SQLMesh__sqlmesh`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-02` (`SQLMesh__sqlmesh`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-04` (`SQLMesh__sqlmesh`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-05` (`apache__hudi`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-06` (`apache__hudi`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-07` (`apache__hudi`): recall@5=`False`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-08` (`trinodb__trino`): recall@5=`False`, strict=`True`, norm=`True`, preflight=`True`, gate=`True` ([])
  - `A-09` (`trinodb__trino`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-10` (`trinodb__trino`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-11` (`trinodb__trino`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-12` (`pola-rs__polars`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-14` (`pola-rs__polars`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-15` (`debezium__debezium`): recall@5=`False`, strict=`True`, norm=`True`, preflight=`True`, gate=`True` ([])
  - `A-16` (`debezium__debezium`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-17` (`debezium__debezium`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-18` (`debezium__debezium`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-20` (`dagster-io__dagster`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-21` (`dagster-io__dagster`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-22` (`projectnessie__nessie`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-23` (`projectnessie__nessie`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-24` (`projectnessie__nessie`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-26` (`2611.12169`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`False` (['ICEBERG_EQUALITY_DELETE_UNCOMPACTED'])
  - `A-28` (`2609.14370`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-32` (`2610.11525`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-33` (`2611.10810`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-34` (`2609.28779`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-36` (`2610.02259`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-38` (`2608.14643`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
  - `A-39` (`2610.10919`): recall@5=`True`, strict=`False`, norm=`False`, preflight=`True`, gate=`True` ([])
- **Benchmark B Misses / Partial Concept Coverage (25):**
  - `DE-01`: preflight=`True`, strict=`True`, gate=`True`, concepts=`3/6` missing=`['read_process_write', 'transactional outbox', 'epoch fencing']`
  - `DE-02`: preflight=`True`, strict=`False`, gate=`True`, concepts=`2/6` missing=`['position deletes', 'rewrite_data_files', 'rewrite_position_delete_files', 'compaction']`
  - `DE-03`: preflight=`True`, strict=`True`, gate=`True`, concepts=`1/6` missing=`['non-exchange stage', 'UDF', 'whole-stage codegen', 'skewed join', 'structured streaming']`
  - `DE-04`: preflight=`True`, strict=`False`, gate=`True`, concepts=`0/5` missing=`['partition pruning', 'non-sargable predicate', 'require_partition_filter', 'range predicate', 'clustering']`
  - `DE-05`: preflight=`True`, strict=`False`, gate=`False`, concepts=`2/6` missing=`['lookback window', 'NULL = NULL', 'insert_overwrite', 'microbatch']`
  - `DE-06`: preflight=`True`, strict=`False`, gate=`True`, concepts=`3/6` missing=`['half-open interval', 'LEAD window', 'EXCLUDE USING gist']`
  - `DE-07`: preflight=`True`, strict=`False`, gate=`False`, concepts=`1/6` missing=`['pg_wal', 'WAL retention', 'max_slot_wal_keep_size', 'disk full', 'snapshot fallback']`
  - `DE-08`: preflight=`True`, strict=`False`, gate=`True`, concepts=`1/5` missing=`['background merge', 'eventual deduplication', 'argMax', 'sorting key']`
  - `DE-09`: preflight=`True`, strict=`False`, gate=`True`, concepts=`1/5` missing=`['retention threshold', 'snapshot isolation', 'time travel', 'orphan files']`
  - `DE-10`: preflight=`True`, strict=`False`, gate=`True`, concepts=`1/5` missing=`['idle partition', 'withIdleness', 'minimum watermark', 'event-time skew']`
  - `DE-11`: preflight=`True`, strict=`False`, gate=`True`, concepts=`3/6` missing=`['incremental checkpoint', 'in-flight buffers', 'cgroup OOM']`
  - `DE-12`: preflight=`True`, strict=`False`, gate=`True`, concepts=`0/6` missing=`['small files', 'manifest list', 'S3 LIST/GET overhead', 'bin-packing', 'target-file-size-bytes', 'expire_snapshots']`
  - `DE-13`: preflight=`True`, strict=`False`, gate=`True`, concepts=`1/5` missing=`['column mapping', 'name-based resolution', 'schema evolution', 'type widening']`
  - `DE-14`: preflight=`True`, strict=`False`, gate=`True`, concepts=`0/5` missing=`['single-writer lock', 'read_only', 'out-of-core spill', 'memory_limit', 'temp_directory']`
  - `DE-15`: preflight=`True`, strict=`False`, gate=`True`, concepts=`1/5` missing=`['binlog_format=ROW', 'binlog_row_image=FULL', 'FLUSH TABLES WITH READ LOCK', 'incremental snapshot']`
  - `DE-16`: preflight=`True`, strict=`False`, gate=`True`, concepts=`0/6` missing=`['idempotency', 'data_interval_start', 'data_interval_end', 'DELETE+INSERT', 'MERGE', 'non-deterministic NOW()']`
  - `DE-17`: preflight=`True`, strict=`False`, gate=`True`, concepts=`0/5` missing=`['micro-partition overlap', 'clustering depth', 'automatic reclustering', 'COPY INTO metadata', '64-day load history']`
  - `DE-18`: preflight=`True`, strict=`False`, gate=`True`, concepts=`0/5` missing=`['immutable Parquet', 'time travel', 'expire_snapshots', 'compaction', 'crypto-shredding']`
  - `DE-19`: preflight=`True`, strict=`False`, gate=`True`, concepts=`0/5` missing=`['compression ratio', 'JVM object overhead', 'deserialized size', 'hash table', 'spark.sql.autoBroadcastJoinThreshold']`
  - `DE-20`: preflight=`True`, strict=`False`, gate=`False`, concepts=`0/6` missing=`['TIMESTAMPTZ', 'UTC normalization', 'DECIMAL(18,4)', 'IEEE-754 rounding', 'data contracts', 'distribution checks']`
  - `DE-21`: preflight=`True`, strict=`False`, gate=`True`, concepts=`1/6` missing=`['Kimball star schema', 'Hub Link Satellite', 'PIT table', 'fan-out trap', 'conformed dimensions']`
  - `DE-22`: preflight=`True`, strict=`False`, gate=`True`, concepts=`0/6` missing=`['Write-Audit-Publish', 'WAP', 'Iceberg branching', 'Nessie', 'atomic fast-forward', 'data quality gate']`
  - `DE-23`: preflight=`True`, strict=`False`, gate=`True`, concepts=`0/6` missing=`['federated computational governance', 'data contracts', 'ODCS', 'OpenLineage', 'domain-oriented ownership', 'semantic drift']`
  - `DE-24`: preflight=`True`, strict=`False`, gate=`True`, concepts=`1/6` missing=`['cardinality misestimation', 'correlated predicates', 'multi-column statistics', 'dynamic runtime filtering', 'AQE']`
  - `DE-25`: preflight=`True`, strict=`True`, gate=`True`, concepts=`1/6` missing=`['RUM conjecture', 'write amplification', 'read amplification', 'space amplification', 'Bloom filter']`
