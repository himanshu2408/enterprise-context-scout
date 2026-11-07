# Domain Readiness Benchmark Report: `polyglot_enterprise_v2`

- **Repositories Evaluated (5):** `enterprise_gitlab_ops`, `more-itertools`, `rich`, `tenacity`, `typer`
- **Languages Indexed (7):** `go`, `javascript`, `python`, `sql`, `terraform`, `tsx`, `yaml`
- **Corpus Size:** `990` files, `6333` AST symbols, `38398` directed AST edges (`~883,895` tokens)
- **File Recall@5:** `25/25` (`100.0%`)
- **Symbol Recall@5:** `24/25` (`96.0%`)
- **Pre-Flight Gate Pass Rate:** `25/25` (`100.0%`)
- **Byte-Exact Slice Faithfulness (VAL-02):** `160/160` (`100.0%`)
- **Mean Token Savings vs Full Repo:** `99.84%`
- **Median Query Latency:** `2451.17 ms`
- **Incremental Sync Latency:** `144.23 ms`
- **Out-of-Scope $0 Abstention:** `True`
- **Prompt Injection Blocked:** `True`

| # | Category | Repo | Lang | Target Symbol | File@5 | Sym@5 | Tokens | Latency |
|---|---|---|---|---|---|---|---|---|
| 1 | `exact_symbol_localization` | `enterprise_gitlab_ops` | `yaml` | `gitlab-ci..base_python_job` | `True` | `True` | `162` | `2492.03 ms` |
| 2 | `cross_symbol_dependency_closure` | `enterprise_gitlab_ops` | `terraform` | `infra.main.google_cloud_run_v2_service.context_scout_gateway` | `True` | `True` | `478` | `3859.58 ms` |
| 3 | `polyglot_sql_query` | `enterprise_gitlab_ops` | `sql` | `schema.ledger_rls.tenant_accounts` | `True` | `True` | `260` | `2684.47 ms` |
| 4 | `exact_symbol_localization` | `enterprise_gitlab_ops` | `tsx` | `services.AuditPanel.AuditRowProps` | `True` | `True` | `210` | `2320.42 ms` |
| 5 | `polyglot_go_query` | `enterprise_gitlab_ops` | `go` | `services.gateway.GatewayConfig` | `True` | `True` | `144` | `2480.53 ms` |
| 6 | `polyglot_python_query` | `more-itertools` | `python` | `more_itertools.more.chunked` | `True` | `True` | `6552` | `2406.76 ms` |
| 7 | `exact_symbol_localization` | `rich` | `python` | `rich.get_console` | `True` | `True` | `2146` | `2384.77 ms` |
| 8 | `polyglot_python_query` | `tenacity` | `python` | `conf` | `True` | `True` | `1649` | `2391.54 ms` |
| 9 | `polyglot_yaml_query` | `typer` | `yaml` | `github.workflows.guard-dependencies.jobs` | `True` | `True` | `2381` | `2341.89 ms` |
| 10 | `exact_symbol_localization` | `typer` | `python` | `docs_src.app_dir.tutorial001_py310.main` | `True` | `True` | `575` | `2510.54 ms` |
| 11 | `cross_symbol_dependency_closure` | `enterprise_gitlab_ops` | `yaml` | `gitlab-ci.lint_and_typecheck` | `True` | `True` | `162` | `2400.4 ms` |
| 12 | `polyglot_terraform_query` | `enterprise_gitlab_ops` | `terraform` | `infra.main.alloydb_cluster` | `True` | `True` | `478` | `3943.76 ms` |
| 13 | `exact_symbol_localization` | `enterprise_gitlab_ops` | `sql` | `schema.ledger_rls.execution_orders` | `True` | `True` | `222` | `2507.58 ms` |
| 14 | `cross_symbol_dependency_closure` | `enterprise_gitlab_ops` | `tsx` | `services.AuditPanel.formatAuditSummary` | `True` | `True` | `203` | `2450.09 ms` |
| 15 | `polyglot_go_query` | `enterprise_gitlab_ops` | `go` | `services.gateway.GatewayServer` | `True` | `True` | `144` | `2287.07 ms` |
| 16 | `exact_symbol_localization` | `more-itertools` | `python` | `more_itertools.more.first` | `True` | `True` | `6418` | `2451.17 ms` |
| 17 | `cross_symbol_dependency_closure` | `rich` | `python` | `rich.reconfigure` | `True` | `True` | `3319` | `2417.99 ms` |
| 18 | `polyglot_python_query` | `tenacity` | `python` | `tenacity._has_tornado` | `True` | `False` | `1873` | `3779.78 ms` |
| 19 | `exact_symbol_localization` | `typer` | `yaml` | `mkdocs.extra_javascript` | `True` | `True` | `69` | `2508.77 ms` |
| 20 | `cross_symbol_dependency_closure` | `typer` | `python` | `docs_src.arguments.default.tutorial001_an_py310.main` | `True` | `True` | `580` | `2432.45 ms` |
| 21 | `polyglot_yaml_query` | `enterprise_gitlab_ops` | `yaml` | `gitlab-ci.run_postgres_rls_tests` | `True` | `True` | `274` | `2415.24 ms` |
| 22 | `exact_symbol_localization` | `enterprise_gitlab_ops` | `sql` | `schema.ledger_rls.verify_tenant_margin` | `True` | `True` | `242` | `2509.38 ms` |
| 23 | `cross_symbol_dependency_closure` | `enterprise_gitlab_ops` | `tsx` | `services.AuditPanel.renderAuditBadge` | `True` | `True` | `200` | `2397.8 ms` |
| 24 | `polyglot_go_query` | `enterprise_gitlab_ops` | `go` | `services.gateway.NewGatewayServer` | `True` | `True` | `138` | `2561.98 ms` |
| 25 | `exact_symbol_localization` | `more-itertools` | `python` | `more_itertools.more.last` | `True` | `True` | `6376` | `2597.11 ms` |
