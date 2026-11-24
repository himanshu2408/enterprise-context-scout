# Streaming Watermarking, dbt Incremental Merges, and Parquet Evolution

## Spark Structured Streaming & Flink Watermarks
Event-time streaming engines track `watermark = max_event_time - delay_threshold` via
`withWatermark("event_time", "10 minutes")`. Records arriving with `event_time >= watermark`
update active windows in RocksDB state; records older than the watermark are dropped and
finalized windows are evicted to bound memory. In multi-partition Kafka consumers, an idle
partition stalls global min-watermark progression unless `withIdleness()` is configured.

## dbt Incremental Merge Models & Schema Evolution
dbt incremental models guard incremental reads with `{% if is_incremental() %}` and use
`incremental_strategy='merge'` with a non-null `unique_key` (or `insert_overwrite` by partition)
to achieve idempotent re-runs across late-arriving lookback windows. Declarative schema evolution
via `on_schema_change='append_new_columns'` paired with enforced model contracts prevents silent
data loss.
