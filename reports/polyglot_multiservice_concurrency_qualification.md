# Suite 3.2: Polyglot Cross-Boundary Contracts & 40-Way Concurrent SSE Qualification

## Corpus Composition (Real Cached Multi-Language Repositories)

- **Indexed Code Files**: `110` across `10` repositories (`{'python': 49, 'tsx': 13, 'typescript': 31, 'sql': 3, 'terraform': 1, 'go': 13}`)
- **AST Symbols & Edges**: `1673` symbols, `13981` edges (`3` cross-boundary links)
- **Architecture & Policy ADRs**: `9` chunks (`market_scope='INTERNAL_ADR'`)

## Polyglot Graph & Drift Accuracy (5d85e3e Baseline vs Post-Fix HEAD)

| Metric | 5d85e3e Baseline | Post-Fix HEAD | 95% Wilson CI (HEAD) |
|---|---|---|---|
| Cross-Boundary Contract 1-Hop Recall | 0 of 3 | 2 of 3 | [0.2077, 0.9385] |
| Code-vs-Stale-RFC Conflict Detection | 0 of 5 | 5 of 5 | [0.5655, 1.0] |
| Colliding Same-Name Symbol Isolation (`0` false leaks target) | 3 of 3 clean (0 leaks) | 3 of 3 clean (0 leaks) | [0.4385, 1.0] |

## 40-Concurrent-User Mixed-Persona FastAPI ASGI SSE Load Test (HEAD)

- **Concurrent Sessions**: `40` signed Bearer sessions (`{'engineer': 10, 'quant_researcher': 10, 'product_manager': 10, 'compliance_officer': 10}`)
- **HTTP 200 SSE Completion**: `40 of 40` (`95% CI [0.9124, 1.0]`)
- **Stage Telemetry Validity**: `40 of 40`
- **Zero Cross-Principal ContextVar / RLS Leaks**: `40 of 40` (`95% CI [0.9124, 1.0]`)
- **SSE Latency**: `p50 = 1201.6 ms`, `p95 = 1203.21 ms`

## Methodology & Limitations

- Cross-boundary contract linking (`CrossBoundaryContractLinker`) used extracted symbols from `scout-api`, `scout-ui`, `scout-sql`, and `gitlab-ops` plus real cached libraries (`flask`, `structlog`, `zod`, `ky`, `chi`).
- The 40-concurrent SSE test ran over `httpx.AsyncClient(transport=ASGITransport(app=app))` against real PostgreSQL 17 `FORCE ROW LEVEL SECURITY` connection pooling with `LocalGemmaClient(offline_mode=True)` deterministic synthesis.
