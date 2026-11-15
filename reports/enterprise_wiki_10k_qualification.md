# Suite 3.1: 10,000-Page Enterprise Wiki Scale & Lifecycle Qualification

## Summary (5d85e3e Baseline vs Post-Fix HEAD at 10,000 Pages)

- **Evaluation Query Sample Size**: `N = 71` balanced queries (`16` per role across `ENGINEER`, `PRODUCT_MANAGER`, `COMPLIANCE_OFFICER`, `QUANT_RESEARCHER`)
- **Measurement Methods**: `5d85e3e worktree baseline` (temporary detached worktree against isolated PG17 `eval_base5d85_*`) vs `deterministic PG17 + gate` (live PG17 `FORCE ROW LEVEL SECURITY` Hybrid RRF + `EmbeddingGemma 2` + `ScoutWorkflow` / `PreFlightValidator(min_rrf_score=0.55)`)
- **5d85e3e Non-GLOBAL `market_scope` Probe**: `ValueError: 'CONFLUENCE' is not a valid MarketRegime` (baseline `1K/5K/10K` vector comparison below used `market_scope='GLOBAL'` so SQL RRF could execute)
- **5d85e3e Federated Coordinator Custom-Domain Probe**: `abstained=False` (``)

| Metric | 5d85e3e Baseline (10K) | Post-Fix HEAD (10K) | 95% Wilson CI (HEAD) |
|---|---|---|---|
| p50 Latency (ms) | 322.28 ms | 418.99 ms | — |
| p95 Latency (ms) | 355.36 ms | 533.2 ms | — |
| Top-5 Noise & Stale Rejection | 0 of 12 | 12 of 12 | [0.7575, 1.0] |
| Multi-Version v3 Freshness | Not enforced (stale v1/v2 indexed) | 25 of 27 | [0.7663, 0.9794] |
| 2026 vs 2023 Conflict Disambiguation | 0 of 10 | 2 of 10 | [0.0567, 0.5098] |
| Table Value Lookup Accuracy | Not tested (`CONFLUENCE` crash) | 7 of 10 | [0.3968, 0.8922] |
| Multi-Role RBAC Zero-Leak Enforcement | 12 of 12 | 12 of 12 | [0.7575, 1.0] |
| In-Scope Wrongful Refusal Rate | N/A | 0 of 51 (0.0%) | — |
| Engineer Pure-Doc `code_slices == []` | N/A | 10 of 10 | [0.7225, 1.0] |

## Scale Curve Across 1,000 -> 5,000 -> 10,000 Indexed Pages (HEAD)

| Pages | p50 Latency (ms) | p95 Latency (ms) | Noise Rejection | Conflicts | Tables | RBAC | Refusals |
|---|---|---|---|---|---|---|---|
| 1,000 | 106.76 | 121.41 | 12/12 | 3/10 | 5/10 | 12/12 | 0/51 (0.0%) |
| 5,000 | 258.95 | 364.78 | 12/12 | 3/10 | 7/10 | 12/12 | 0/51 (0.0%) |
| 10,000 | 418.99 | 533.2 | 12/12 | 2/10 | 7/10 | 12/12 | 0/51 (0.0%) |

## Methodology & Limitations

- On-disk real public documents and tables were chunked verbatim via `ConfluenceConnectorPlugin` and embedded with cached `google/embeddinggemma-2` (768-d) in 500-row batches.
- To avoid $O(N^2)$ Python pair-scanning across 10,000 chunks inside `ConflictDetector`, Stage 1 pushes down Hybrid RRF (`LIMIT 12`) to PG17 (`FORCE ROW LEVEL SECURITY` + HNSW + FTS + GIN trigram) and passes the retrieved active chunks to `ScoutWorkflow`.
- Provenance manifest saved in [enterprise_wiki_10k_provenance_summary.md](enterprise_wiki_10k_provenance_summary.md).
