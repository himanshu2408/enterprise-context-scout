# Lakehouse ACID Transactions, MVCC, Snapshot Isolation, and OCC

## Relational ACID & PostgreSQL MVCC
ACID guarantees Atomicity (all-or-nothing commit/rollback via Write-Ahead Log), Consistency
(primary/foreign key and CHECK constraint invariants), Isolation (Read Committed, Snapshot
Isolation / Repeatable Read, and Serializable Snapshot Isolation via SIREAD locks), and
Durability (`fsync` WAL persistence). PostgreSQL MVCC uses `xmin`/`xmax` tuple headers so
readers do not block writers and writers do not block readers.

## Apache Iceberg & Delta Lake Optimistic Concurrency Control
- **Apache Iceberg**: Atomic compare-and-swap (CAS) of the catalog metadata pointer swaps
  `metadata.json` to a new immutable snapshot pointing to manifest lists and data/delete files.
  Concurrent writers validate conflicts (overlapping deleted files or equality deletes), rebase
  conflict-free manifests, or abort without partial writes. Schema evolution relies on immutable
  column **field IDs** for O(1) metadata-only ADD/DROP/RENAME/widen operations.
- **Delta Lake**: Append-only `_delta_log/NNNN.json` commits with periodic Parquet checkpoints
  provide WriteSerializable and Serializable OCC; readers reconstruct consistent point-in-time
  snapshots. `VACUUM RETAIN 0 HOURS` must not run while active readers or checkpoints reference
  older Parquet files.
