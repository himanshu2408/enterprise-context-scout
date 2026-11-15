# Data Engineering Document Contextualizer Prompt

Given a document chunk (`doc_title`, `section_heading`, `source_format`, `sub_scope`, and raw prose/SQL/table content), produce a concise 1-2 sentence retrieval header that locates this chunk inside its broader Data Engineering architecture without altering the chunk's verbatim text:

- Identify the system or specification (e.g., PostgreSQL 17, Apache Iceberg v2/v3, Delta Lake, Apache Spark AQE, Apache Flink, Debezium CDC, dbt/SQLMesh, DuckDB, Trino, ClickHouse).
- State the concrete mechanism, invariant, or trade-off covered (e.g., MVCC tuple visibility & `VACUUM` freeze, copy-on-write vs merge-on-read equality deletes, watermark lateness & checkpoint barriers, Kimball star schema vs Data Vault vs OBT, incremental `merge` vs `microbatch` atomicity).
- Keep the contextual header factual, concise (`<= 45` words), and faithful to the chunk.
