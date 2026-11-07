# Domain Readiness Benchmark Report: `polyglot_unseen_v2`

- **Repositories Evaluated (50):** `anyio`, `attrs`, `build`, `chardet`, `chi`, `click`, `databases`, `dirty-equals`, `enterprise_gitlab_ops`, `flask`, `flask-sqlalchemy`, `flit`, `h11`, `h2`, `hpack`, `httpcore`, `httpx`, `hyperframe`, `idna`, `itsdangerous`, `jinja`, `ky`, `loguru`, `markdown-it-py`, `markupsafe`, `marshmallow`, `more-itertools`, `orm`, `packaging`, `pydantic-settings`, `pyjwt`, `pyproject-hooks`, `python-certifi`, `python-dotenv`, `requests`, `rich`, `sniffio`, `stamina`, `starlette`, `structlog`, `svcs`, `tenacity`, `tomli`, `tomli-w`, `typer`, `urllib3`, `uvicorn`, `watchfiles`, `werkzeug`, `zod`
- **Languages Indexed (9):** `go`, `javascript`, `python`, `rust`, `sql`, `terraform`, `tsx`, `typescript`, `yaml`
- **Corpus Size:** `4232` files, `51130` AST symbols, `290759` directed AST edges (`~7,486,164` tokens)
- **File Recall@5:** `59/60` (`98.33%`)
- **Symbol Recall@5:** `59/60` (`98.33%`)
- **Pre-Flight Gate Pass Rate:** `60/60` (`100.0%`)
- **Byte-Exact Slice Faithfulness (VAL-02):** `371/371` (`100.0%`)
- **Median Input Tokens (Without Scout vs. With Scout):** `32,486` vs `1,923` (`92.66%` median agent token savings)
- **Total Input Tokens Across Benchmark ($N=60$):** `1,926,044` Without Scout vs `137,162` With Scout
- **Mean Token Savings vs Full Repo:** `99.98%`
- **Median Query Latency:** `199.7 ms`
- **Incremental Sync Latency:** `1541.0 ms`
- **Out-of-Scope $0 Abstention:** `True`
- **Prompt Injection Blocked:** `True`

| # | Category | Repo | Lang | Target Symbol | File@5 | Sym@5 | Without Scout | With Scout | Saved % | Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `repo_scoped_symbol_localization` | `anyio` | `python` | `anyio._backends._asyncio.AsyncIOBackend.check_cancelled` | `True` | `True` | `86,482` | `1,571` | `98.18%` | `3588.39 ms` |
| 2 | `cross_symbol_dependency_closure` | `attrs` | `python` | `attr._cmp._make_operator` | `True` | `True` | `24,552` | `1,475` | `93.99%` | `157.36 ms` |
| 3 | `polyglot_python_query` | `build` | `python` | `build.__main__._ArtifactReport` | `True` | `True` | `39,848` | `1,521` | `96.18%` | `182.29 ms` |
| 4 | `symptom_and_signature_discovery` | `chardet` | `python` | `chardet._kernel.dot_packed` | `True` | `True` | `54,337` | `6,496` | `88.04%` | `231.62 ms` |
| 5 | `repo_scoped_symbol_localization` | `chi` | `go` | `chain.chain` | `True` | `True` | `61,476` | `1,200` | `98.05%` | `171.0 ms` |
| 6 | `cross_symbol_dependency_closure` | `click` | `python` | `click._compat._AtomicFile` | `True` | `True` | `31,577` | `1,133` | `96.41%` | `764.47 ms` |
| 7 | `polyglot_python_query` | `databases` | `python` | `databases.backends.aiopg.AiopgBackend` | `True` | `True` | `31,387` | `4,296` | `86.31%` | `251.15 ms` |
| 8 | `symptom_and_signature_discovery` | `dirty-equals` | `python` | `dirty_equals._base.AnyThing` | `True` | `True` | `42,885` | `2,782` | `93.51%` | `472.96 ms` |
| 9 | `repo_scoped_symbol_localization` | `enterprise_gitlab_ops` | `yaml` | `gitlab-ci.lint_and_typecheck` | `True` | `True` | `13,550` | `832` | `93.86%` | `163.49 ms` |
| 10 | `cross_symbol_dependency_closure` | `flask` | `python` | `flask.app.Flask.app_context` | `True` | `True` | `49,750` | `2,692` | `94.59%` | `609.22 ms` |
| 11 | `polyglot_python_query` | `flask-sqlalchemy` | `python` | `flask_sqlalchemy.cli.add_models_to_shell` | `True` | `True` | `36,034` | `2,114` | `94.13%` | `311.66 ms` |
| 12 | `symptom_and_signature_discovery` | `flit` | `python` | `flit._get_dirs.get_dirs` | `True` | `True` | `37,438` | `2,145` | `94.27%` | `236.48 ms` |
| 13 | `repo_scoped_symbol_localization` | `h11` | `python` | `h11._connection.Connection._clean_up_response_headers_for_sending` | `True` | `True` | `47,060` | `4,061` | `91.37%` | `188.22 ms` |
| 14 | `cross_symbol_dependency_closure` | `h2` | `python` | `h2._typing.Buffer` | `True` | `True` | `4,084` | `731` | `82.1%` | `177.83 ms` |
| 15 | `polyglot_python_query` | `hpack` | `python` | `hpack.exceptions.InvalidTableIndex` | `True` | `True` | `11,642` | `1,703` | `85.37%` | `589.31 ms` |
| 16 | `symptom_and_signature_discovery` | `httpcore` | `python` | `httpcore._api.request` | `True` | `True` | `37,123` | `4,594` | `87.62%` | `263.47 ms` |
| 17 | `repo_scoped_symbol_localization` | `httpx` | `python` | `httpx._api.delete` | `True` | `True` | `69,234` | `3,914` | `94.35%` | `532.76 ms` |
| 18 | `cross_symbol_dependency_closure` | `hyperframe` | `python` | `hyperframe.exceptions.HyperframeError` | `True` | `True` | `3,799` | `698` | `81.63%` | `644.82 ms` |
| 19 | `polyglot_python_query` | `idna` | `python` | `idna.cli._build_parser` | `True` | `True` | `34,500` | `2,039` | `94.09%` | `317.02 ms` |
| 20 | `symptom_and_signature_discovery` | `itsdangerous` | `python` | `itsdangerous.encoding.base64_decode` | `True` | `True` | `15,714` | `2,177` | `86.15%` | `165.62 ms` |
| 21 | `repo_scoped_symbol_localization` | `jinja` | `python` | `jinja2.async_utils.async_variant` | `True` | `True` | `34,672` | `1,923` | `94.45%` | `184.95 ms` |
| 22 | `cross_symbol_dependency_closure` | `ky` | `typescript` | `headers` | `True` | `True` | `49,142` | `1,459` | `97.03%` | `144.98 ms` |
| 23 | `polyglot_python_query` | `loguru` | `python` | `loguru._asyncio_loop.load_loop_functions` | `True` | `True` | `23,207` | `1,703` | `92.66%` | `162.73 ms` |
| 24 | `symptom_and_signature_discovery` | `markdown-it-py` | `python` | `markdown_it._punycode.map_domain` | `True` | `True` | `30,006` | `2,133` | `92.89%` | `607.11 ms` |
| 25 | `repo_scoped_symbol_localization` | `markupsafe` | `python` | `markupsafe.EscapeFormatter` | `True` | `True` | `11,916` | `1,284` | `89.22%` | `168.12 ms` |
| 26 | `cross_symbol_dependency_closure` | `marshmallow` | `python` | `marshmallow.class_registry.get_class` | `True` | `True` | `33,064` | `1,388` | `95.8%` | `162.19 ms` |
| 27 | `polyglot_python_query` | `more-itertools` | `python` | `more_itertools.more.SequenceView` | `True` | `True` | `39,702` | `1,472` | `96.29%` | `199.7 ms` |
| 28 | `symptom_and_signature_discovery` | `orm` | `python` | `orm.fields.AutoNowMixin` | `True` | `True` | `9,296` | `1,375` | `85.21%` | `215.92 ms` |
| 29 | `repo_scoped_symbol_localization` | `packaging` | `python` | `packaging._elffile.ELFFile` | `True` | `True` | `44,774` | `3,632` | `91.89%` | `228.68 ms` |
| 30 | `cross_symbol_dependency_closure` | `pydantic-settings` | `python` | `pydantic_settings.exceptions.IncompleteFieldDefinitionWarning` | `True` | `True` | `11,800` | `3,272` | `72.27%` | `631.51 ms` |
| 31 | `polyglot_python_query` | `pyjwt` | `python` | `jwt.algorithms.Algorithm` | `True` | `True` | `68,066` | `2,138` | `96.86%` | `275.22 ms` |
| 32 | `symptom_and_signature_discovery` | `pyproject-hooks` | `python` | `pyproject_hooks._impl.BackendUnavailable` | `True` | `True` | `16,975` | `2,097` | `87.65%` | `205.43 ms` |
| 33 | `repo_scoped_symbol_localization` | `python-certifi` | `python` | `certifi.core.exit_cacert_ctx` | `True` | `True` | `14,456` | `1,260` | `91.28%` | `167.6 ms` |
| 34 | `cross_symbol_dependency_closure` | `python-dotenv` | `python` | `dotenv.cli.enumerate_env` | `True` | `True` | `24,153` | `5,584` | `76.88%` | `703.72 ms` |
| 35 | `polyglot_python_query` | `requests` | `python` | `requests._check_cryptography` | `True` | `True` | `19,568` | `1,440` | `92.64%` | `186.88 ms` |
| 36 | `symptom_and_signature_discovery` | `rich` | `python` | `rich.__main__.ColorBox` | `True` | `True` | `38,338` | `4,222` | `88.99%` | `192.6 ms` |
| 37 | `repo_scoped_symbol_localization` | `sniffio` | `python` | `sniffio._impl.current_async_library` | `True` | `True` | `14,192` | `2,000` | `85.91%` | `167.18 ms` |
| 38 | `cross_symbol_dependency_closure` | `stamina` | `python` | `stamina._config._Config` | `True` | `True` | `41,714` | `2,790` | `93.31%` | `185.74 ms` |
| 39 | `polyglot_python_query` | `starlette` | `python` | `starlette._exception_handler.wrap_app_handling_exceptions` | `True` | `True` | `35,387` | `8,017` | `77.34%` | `205.42 ms` |
| 40 | `symptom_and_signature_discovery` | `structlog` | `python` | `structlog._base.BoundLoggerBase._process_event` | `True` | `True` | `48,258` | `2,726` | `94.35%` | `675.21 ms` |
| 41 | `repo_scoped_symbol_localization` | `svcs` | `python` | `svcs._autowire._default_or_raise` | `True` | `True` | `46,418` | `2,673` | `94.24%` | `207.34 ms` |
| 42 | `cross_symbol_dependency_closure` | `tenacity` | `python` | `tenacity.BaseAction` | `True` | `True` | `24,708` | `899` | `96.36%` | `171.13 ms` |
| 43 | `polyglot_python_query` | `tomli` | `python` | `fuzzer.fuzz.normalize_toml_obj` | `True` | `True` | `13,057` | `1,730` | `86.75%` | `187.18 ms` |
| 44 | `symptom_and_signature_discovery` | `tomli-w` | `python` | `tomli_w._writer.format_decimal` | `True` | `True` | `33,893` | `3,066` | `90.95%` | `192.43 ms` |
| 45 | `repo_scoped_symbol_localization` | `typer` | `python` | `typer._click._compat._AtomicFile` | `True` | `True` | `103,605` | `1,753` | `98.31%` | `858.15 ms` |
| 46 | `cross_symbol_dependency_closure` | `urllib3` | `python` | `urllib3._base_connection._ResponseOptions` | `True` | `True` | `22,509` | `956` | `95.75%` | `156.3 ms` |
| 47 | `polyglot_python_query` | `uvicorn` | `python` | `uvicorn._ansi.style` | `True` | `True` | `47,634` | `2,719` | `94.29%` | `236.15 ms` |
| 48 | `symptom_and_signature_discovery` | `watchfiles` | `python` | `watchfiles.cli.build_filter` | `True` | `True` | `32,486` | `4,654` | `85.67%` | `204.9 ms` |
| 49 | `repo_scoped_symbol_localization` | `werkzeug` | `python` | `werkzeug._header_property._DSProto` | `True` | `True` | `8,117` | `1,493` | `81.61%` | `164.4 ms` |
| 50 | `cross_symbol_dependency_closure` | `zod` | `typescript` | `array` | `True` | `True` | `49,394` | `1,881` | `96.19%` | `567.68 ms` |
| 51 | `polyglot_go_query` | `enterprise_gitlab_ops` | `go` | `services.gateway.GatewayConfig` | `True` | `True` | `5,670` | `773` | `86.37%` | `155.98 ms` |
| 52 | `symptom_and_signature_discovery` | `enterprise_gitlab_ops` | `sql` | `schema.ledger_rls.execution_orders` | `True` | `True` | `5,746` | `873` | `84.81%` | `157.26 ms` |
| 53 | `repo_scoped_symbol_localization` | `enterprise_gitlab_ops` | `terraform` | `infra.main.alloydb_cluster` | `True` | `True` | `10,730` | `1,336` | `87.55%` | `275.91 ms` |
| 54 | `cross_symbol_dependency_closure` | `enterprise_gitlab_ops` | `tsx` | `services.AuditPanel.formatAuditSummary` | `True` | `True` | `10,782` | `1,220` | `88.68%` | `176.78 ms` |
| 55 | `polyglot_yaml_query` | `markdown-it-py` | `yaml` | `port` | `True` | `True` | `18,326` | `2,120` | `88.43%` | `167.73 ms` |
| 56 | `symptom_and_signature_discovery` | `urllib3` | `javascript` | `emscripten_fetch_worker` | `False` | `False` | `51,824` | `1,917` | `96.3%` | `712.54 ms` |
| 57 | `repo_scoped_symbol_localization` | `watchfiles` | `rust` | `lib.RustNotify` | `True` | `True` | `27,506` | `1,042` | `96.21%` | `156.97 ms` |
| 58 | `cross_symbol_dependency_closure` | `werkzeug` | `javascript` | `werkzeug.debug.shared.debugger.addConsoleIconToFrames` | `True` | `True` | `8,407` | `1,408` | `83.25%` | `174.55 ms` |
| 59 | `polyglot_javascript_query` | `zod` | `javascript` | `rollup.config` | `True` | `True` | `39,957` | `2,203` | `94.49%` | `172.02 ms` |
| 60 | `symptom_and_signature_discovery` | `chi` | `go` | `chi.Router` | `True` | `True` | `24,117` | `2,357` | `90.23%` | `158.2 ms` |
