# Domain Readiness Benchmark Report: `polyglot_unseen_v2`

- **Repositories Evaluated (50):** `anyio`, `attrs`, `build`, `chardet`, `chi`, `click`, `databases`, `dirty-equals`, `enterprise_gitlab_ops`, `flask`, `flask-sqlalchemy`, `flit`, `h11`, `h2`, `hpack`, `httpcore`, `httpx`, `hyperframe`, `idna`, `itsdangerous`, `jinja`, `ky`, `loguru`, `markdown-it-py`, `markupsafe`, `marshmallow`, `more-itertools`, `orm`, `packaging`, `pydantic-settings`, `pyjwt`, `pyproject-hooks`, `python-certifi`, `python-dotenv`, `requests`, `rich`, `sniffio`, `stamina`, `starlette`, `structlog`, `svcs`, `tenacity`, `tomli`, `tomli-w`, `typer`, `urllib3`, `uvicorn`, `watchfiles`, `werkzeug`, `zod`
- **Languages Indexed (9):** `go`, `javascript`, `python`, `rust`, `sql`, `terraform`, `tsx`, `typescript`, `yaml`
- **Corpus Size:** `4232` files, `51130` AST symbols, `290759` directed AST edges (`~7,486,164` tokens)
- **File Recall@5:** `58/60` (`96.67%`)
- **Symbol Recall@5:** `57/60` (`95.0%`)
- **Pre-Flight Gate Pass Rate:** `60/60` (`100.0%`)
- **Byte-Exact Slice Faithfulness (VAL-02):** `366/366` (`100.0%`)
- **Mean Token Savings vs Full Repo:** `99.98%`
- **Median Query Latency:** `200.65 ms`
- **Incremental Sync Latency:** `1653.42 ms`
- **Out-of-Scope $0 Abstention:** `True`
- **Prompt Injection Blocked:** `True`

| # | Category | Repo | Lang | Target Symbol | File@5 | Sym@5 | Tokens | Latency |
|---|---|---|---|---|---|---|---|---|
| 1 | `repo_scoped_symbol_localization` | `anyio` | `python` | `anyio._backends._asyncio.AsyncIOBackend.check_cancelled` | `True` | `True` | `921` | `3504.86 ms` |
| 2 | `cross_symbol_dependency_closure` | `attrs` | `python` | `attr._cmp._make_operator` | `True` | `True` | `825` | `167.69 ms` |
| 3 | `polyglot_python_query` | `build` | `python` | `build.__main__._ArtifactReport` | `True` | `True` | `871` | `175.96 ms` |
| 4 | `symptom_and_signature_discovery` | `chardet` | `python` | `chardet._kernel.dot_packed` | `True` | `True` | `5846` | `222.28 ms` |
| 5 | `repo_scoped_symbol_localization` | `chi` | `go` | `chain.chain` | `True` | `True` | `550` | `167.35 ms` |
| 6 | `cross_symbol_dependency_closure` | `click` | `python` | `click._compat._AtomicFile` | `True` | `True` | `483` | `735.95 ms` |
| 7 | `polyglot_python_query` | `databases` | `python` | `databases.backends.aiopg.AiopgBackend` | `True` | `True` | `3646` | `241.01 ms` |
| 8 | `symptom_and_signature_discovery` | `dirty-equals` | `python` | `dirty_equals._base.AnyThing` | `True` | `True` | `2132` | `465.67 ms` |
| 9 | `repo_scoped_symbol_localization` | `enterprise_gitlab_ops` | `yaml` | `gitlab-ci.lint_and_typecheck` | `True` | `True` | `182` | `170.63 ms` |
| 10 | `cross_symbol_dependency_closure` | `flask` | `python` | `flask.app.Flask.app_context` | `True` | `True` | `2042` | `178.67 ms` |
| 11 | `polyglot_python_query` | `flask-sqlalchemy` | `python` | `flask_sqlalchemy.cli.add_models_to_shell` | `True` | `True` | `1464` | `722.75 ms` |
| 12 | `symptom_and_signature_discovery` | `flit` | `python` | `flit._get_dirs.get_dirs` | `True` | `True` | `1495` | `230.56 ms` |
| 13 | `repo_scoped_symbol_localization` | `h11` | `python` | `fuzz.afl-server.process_all` | `True` | `True` | `501` | `168.29 ms` |
| 14 | `cross_symbol_dependency_closure` | `h2` | `python` | `h2._typing.Buffer` | `True` | `True` | `81` | `169.98 ms` |
| 15 | `polyglot_python_query` | `hpack` | `python` | `hpack.exceptions.InvalidTableIndex` | `True` | `True` | `1053` | `168.41 ms` |
| 16 | `symptom_and_signature_discovery` | `httpcore` | `python` | `httpcore._api.request` | `True` | `True` | `3944` | `713.13 ms` |
| 17 | `repo_scoped_symbol_localization` | `httpx` | `python` | `httpx._api.delete` | `True` | `True` | `3264` | `505.13 ms` |
| 18 | `cross_symbol_dependency_closure` | `hyperframe` | `python` | `hyperframe.exceptions.HyperframeError` | `True` | `True` | `48` | `221.0 ms` |
| 19 | `polyglot_python_query` | `idna` | `python` | `idna.cli._build_parser` | `True` | `True` | `1389` | `743.05 ms` |
| 20 | `symptom_and_signature_discovery` | `itsdangerous` | `python` | `itsdangerous.encoding.base64_decode` | `True` | `True` | `1527` | `163.67 ms` |
| 21 | `repo_scoped_symbol_localization` | `jinja` | `python` | `jinja2.async_utils.async_variant` | `True` | `True` | `1273` | `179.11 ms` |
| 22 | `cross_symbol_dependency_closure` | `ky` | `typescript` | `headers` | `True` | `True` | `809` | `146.78 ms` |
| 23 | `polyglot_python_query` | `loguru` | `python` | `loguru._asyncio_loop.load_loop_functions` | `True` | `True` | `1053` | `161.77 ms` |
| 24 | `symptom_and_signature_discovery` | `markdown-it-py` | `python` | `benchmarking.conftest.pytest_benchmark_update_machine_info` | `True` | `True` | `1358` | `182.61 ms` |
| 25 | `repo_scoped_symbol_localization` | `markupsafe` | `python` | `markupsafe.EscapeFormatter` | `True` | `True` | `634` | `155.14 ms` |
| 26 | `cross_symbol_dependency_closure` | `marshmallow` | `python` | `marshmallow.class_registry.get_class` | `True` | `True` | `738` | `577.84 ms` |
| 27 | `polyglot_python_query` | `more-itertools` | `python` | `more_itertools.more.SequenceView` | `True` | `True` | `822` | `201.03 ms` |
| 28 | `symptom_and_signature_discovery` | `orm` | `python` | `orm.fields.AutoNowMixin` | `True` | `True` | `725` | `214.73 ms` |
| 29 | `repo_scoped_symbol_localization` | `packaging` | `python` | `noxfile.Project` | `True` | `True` | `1936` | `297.35 ms` |
| 30 | `cross_symbol_dependency_closure` | `pydantic-settings` | `python` | `pydantic_settings.exceptions.IncompleteFieldDefinitionWarning` | `True` | `True` | `2622` | `209.97 ms` |
| 31 | `polyglot_python_query` | `pyjwt` | `python` | `jwt.algorithms.Algorithm` | `True` | `True` | `1488` | `699.66 ms` |
| 32 | `symptom_and_signature_discovery` | `pyproject-hooks` | `python` | `pyproject_hooks._impl.BackendUnavailable` | `True` | `True` | `1447` | `206.65 ms` |
| 33 | `repo_scoped_symbol_localization` | `python-certifi` | `python` | `certifi.core.exit_cacert_ctx` | `True` | `True` | `610` | `165.85 ms` |
| 34 | `cross_symbol_dependency_closure` | `python-dotenv` | `python` | `dotenv.cli.enumerate_env` | `True` | `True` | `4934` | `271.36 ms` |
| 35 | `polyglot_python_query` | `requests` | `python` | `requests._check_cryptography` | `True` | `True` | `790` | `182.12 ms` |
| 36 | `symptom_and_signature_discovery` | `rich` | `python` | `rich.__main__.ColorBox` | `True` | `True` | `3572` | `613.46 ms` |
| 37 | `repo_scoped_symbol_localization` | `sniffio` | `python` | `sniffio._impl.current_async_library` | `True` | `True` | `1350` | `161.42 ms` |
| 38 | `cross_symbol_dependency_closure` | `stamina` | `python` | `noxfile._get_pkg` | `True` | `True` | `444` | `174.32 ms` |
| 39 | `polyglot_python_query` | `starlette` | `python` | `starlette._exception_handler.wrap_app_handling_exceptions` | `True` | `True` | `7367` | `208.42 ms` |
| 40 | `symptom_and_signature_discovery` | `structlog` | `python` | `show_off.make_call_stack_more_impressive` | `True` | `False` | `1048` | `181.76 ms` |
| 41 | `repo_scoped_symbol_localization` | `svcs` | `python` | `svcs._autowire._default_or_raise` | `True` | `True` | `2023` | `212.11 ms` |
| 42 | `cross_symbol_dependency_closure` | `tenacity` | `python` | `tenacity.BaseAction` | `True` | `True` | `249` | `587.63 ms` |
| 43 | `polyglot_python_query` | `tomli` | `python` | `benchmark.run.benchmark` | `True` | `True` | `1660` | `233.88 ms` |
| 44 | `symptom_and_signature_discovery` | `tomli-w` | `python` | `benchmark.run.benchmark` | `True` | `True` | `2199` | `231.22 ms` |
| 45 | `repo_scoped_symbol_localization` | `typer` | `python` | `scripts.agent_banner.AgentBannerPreprocessor` | `True` | `True` | `1277` | `253.48 ms` |
| 46 | `cross_symbol_dependency_closure` | `urllib3` | `python` | `noxfile.downstream_botocore` | `True` | `True` | `2112` | `180.2 ms` |
| 47 | `polyglot_python_query` | `uvicorn` | `python` | `uvicorn._ansi.style` | `True` | `True` | `2069` | `643.59 ms` |
| 48 | `symptom_and_signature_discovery` | `watchfiles` | `python` | `watchfiles.cli.build_filter` | `True` | `True` | `4004` | `200.65 ms` |
| 49 | `repo_scoped_symbol_localization` | `werkzeug` | `python` | `werkzeug._header_property._DSProto` | `True` | `True` | `843` | `160.96 ms` |
| 50 | `cross_symbol_dependency_closure` | `zod` | `typescript` | `array` | `True` | `True` | `1231` | `153.58 ms` |
| 51 | `polyglot_go_query` | `enterprise_gitlab_ops` | `go` | `services.gateway.GatewayConfig` | `True` | `True` | `123` | `154.06 ms` |
| 52 | `symptom_and_signature_discovery` | `enterprise_gitlab_ops` | `sql` | `schema.ledger_rls.execution_orders` | `True` | `True` | `223` | `156.0 ms` |
| 53 | `repo_scoped_symbol_localization` | `enterprise_gitlab_ops` | `terraform` | `infra.main.alloydb_cluster` | `True` | `True` | `686` | `678.16 ms` |
| 54 | `cross_symbol_dependency_closure` | `enterprise_gitlab_ops` | `tsx` | `services.AuditPanel.formatAuditSummary` | `True` | `True` | `570` | `170.29 ms` |
| 55 | `polyglot_yaml_query` | `markdown-it-py` | `yaml` | `port` | `True` | `True` | `1470` | `157.83 ms` |
| 56 | `symptom_and_signature_discovery` | `typer` | `yaml` | `mkdocs.extra_javascript` | `False` | `False` | `4755` | `175.89 ms` |
| 57 | `repo_scoped_symbol_localization` | `urllib3` | `javascript` | `emscripten_fetch_worker` | `True` | `True` | `3145` | `159.49 ms` |
| 58 | `symptom_and_signature_discovery` | `watchfiles` | `rust` | `lib.RustNotify` | `True` | `True` | `3750` | `160.53 ms` |
| 59 | `polyglot_javascript_query` | `werkzeug` | `javascript` | `werkzeug.debug.shared.debugger.addConsoleIconToFrames` | `True` | `True` | `2074` | `593.76 ms` |
| 60 | `symptom_and_signature_discovery` | `zod` | `javascript` | `scripts.chrome-devtools-mcp.headlessShellPath` | `False` | `False` | `269` | `187.15 ms` |
