# Domain Readiness Benchmark Report: `polyglot_unseen_v2`

- **Repositories Evaluated (50):** `anyio`, `attrs`, `build`, `chardet`, `chi`, `click`, `databases`, `dirty-equals`, `enterprise_gitlab_ops`, `flask`, `flask-sqlalchemy`, `flit`, `h11`, `h2`, `hpack`, `httpcore`, `httpx`, `hyperframe`, `idna`, `itsdangerous`, `jinja`, `ky`, `loguru`, `markdown-it-py`, `markupsafe`, `marshmallow`, `more-itertools`, `orm`, `packaging`, `pydantic-settings`, `pyjwt`, `pyproject-hooks`, `python-certifi`, `python-dotenv`, `requests`, `rich`, `sniffio`, `stamina`, `starlette`, `structlog`, `svcs`, `tenacity`, `tomli`, `tomli-w`, `typer`, `urllib3`, `uvicorn`, `watchfiles`, `werkzeug`, `zod`
- **Languages Indexed (9):** `go`, `javascript`, `python`, `rust`, `sql`, `terraform`, `tsx`, `typescript`, `yaml`
- **Corpus Size:** `4232` files, `51130` AST symbols, `290759` directed AST edges (`~7,486,164` tokens)
- **File Recall@5:** `59/60` (`98.33%`)
- **Symbol Recall@5:** `59/60` (`98.33%`)
- **Pre-Flight Gate Pass Rate:** `60/60` (`100.0%`)
- **Live Local QLoRA (`Gemma-4-26B-A4B-it`) Schema Validity:** `59/60` (`98.33%`, median GPU latency `1688.26 ms`)
- **Byte-Exact Slice Faithfulness (VAL-02):** `371/371` (`100.0%`)
- **Median Input Tokens (Without Scout vs. With Scout):** `32,486` vs `1,923` (`92.66%` median agent token savings)
- **Total Input Tokens Across Benchmark ($N=60$):** `1,926,044` Without Scout vs `137,162` With Scout
- **Mean Token Savings vs Full Repo:** `99.98%`
- **Median Query Latency:** `629.81 ms`
- **Incremental Sync Latency:** `2288.63 ms`
- **Out-of-Scope $0 Abstention:** `True`
- **Prompt Injection Blocked:** `True`

| # | Category | Repo | Lang | Target Symbol | File@5 | Sym@5 | Without Scout | With Scout | Saved % | Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `repo_scoped_symbol_localization` | `anyio` | `python` | `anyio._backends._asyncio.AsyncIOBackend.check_cancelled` | `True` | `True` | `86,482` | `1,571` | `98.18%` | `3830.77 ms` |
| 2 | `cross_symbol_dependency_closure` | `attrs` | `python` | `attr._cmp._make_operator` | `True` | `True` | `24,552` | `1,475` | `93.99%` | `580.05 ms` |
| 3 | `polyglot_python_query` | `build` | `python` | `build.__main__._ArtifactReport` | `True` | `True` | `39,848` | `1,521` | `96.18%` | `590.13 ms` |
| 4 | `symptom_and_signature_discovery` | `chardet` | `python` | `chardet._kernel.dot_packed` | `True` | `True` | `54,337` | `6,496` | `88.04%` | `641.79 ms` |
| 5 | `repo_scoped_symbol_localization` | `chi` | `go` | `chain.chain` | `True` | `True` | `61,476` | `1,200` | `98.05%` | `586.56 ms` |
| 6 | `cross_symbol_dependency_closure` | `click` | `python` | `click._compat._AtomicFile` | `True` | `True` | `31,577` | `1,133` | `96.41%` | `751.59 ms` |
| 7 | `polyglot_python_query` | `databases` | `python` | `databases.backends.aiopg.AiopgBackend` | `True` | `True` | `31,387` | `4,296` | `86.31%` | `765.6 ms` |
| 8 | `symptom_and_signature_discovery` | `dirty-equals` | `python` | `dirty_equals._base.AnyThing` | `True` | `True` | `42,885` | `2,782` | `93.51%` | `979.64 ms` |
| 9 | `repo_scoped_symbol_localization` | `enterprise_gitlab_ops` | `yaml` | `gitlab-ci.lint_and_typecheck` | `True` | `True` | `13,550` | `832` | `93.86%` | `620.1 ms` |
| 10 | `cross_symbol_dependency_closure` | `flask` | `python` | `flask.app.Flask.app_context` | `True` | `True` | `49,750` | `2,692` | `94.59%` | `628.74 ms` |
| 11 | `polyglot_python_query` | `flask-sqlalchemy` | `python` | `flask_sqlalchemy.cli.add_models_to_shell` | `True` | `True` | `36,034` | `2,114` | `94.13%` | `729.74 ms` |
| 12 | `symptom_and_signature_discovery` | `flit` | `python` | `flit._get_dirs.get_dirs` | `True` | `True` | `37,438` | `2,145` | `94.27%` | `657.0 ms` |
| 13 | `repo_scoped_symbol_localization` | `h11` | `python` | `h11._connection.Connection._clean_up_response_headers_for_sending` | `True` | `True` | `47,060` | `4,061` | `91.37%` | `657.76 ms` |
| 14 | `cross_symbol_dependency_closure` | `h2` | `python` | `h2._typing.Buffer` | `True` | `True` | `4,084` | `731` | `82.1%` | `605.92 ms` |
| 15 | `polyglot_python_query` | `hpack` | `python` | `hpack.exceptions.InvalidTableIndex` | `True` | `True` | `11,642` | `1,703` | `85.37%` | `586.12 ms` |
| 16 | `symptom_and_signature_discovery` | `httpcore` | `python` | `httpcore._api.request` | `True` | `True` | `37,123` | `4,594` | `87.62%` | `687.53 ms` |
| 17 | `repo_scoped_symbol_localization` | `httpx` | `python` | `httpx._api.delete` | `True` | `True` | `69,234` | `3,914` | `94.35%` | `948.92 ms` |
| 18 | `cross_symbol_dependency_closure` | `hyperframe` | `python` | `hyperframe.exceptions.HyperframeError` | `True` | `True` | `3,799` | `698` | `81.63%` | `641.34 ms` |
| 19 | `polyglot_python_query` | `idna` | `python` | `idna.cli._build_parser` | `True` | `True` | `34,500` | `2,039` | `94.09%` | `755.23 ms` |
| 20 | `symptom_and_signature_discovery` | `itsdangerous` | `python` | `itsdangerous.encoding.base64_decode` | `True` | `True` | `15,714` | `2,177` | `86.15%` | `610.4 ms` |
| 21 | `repo_scoped_symbol_localization` | `jinja` | `python` | `jinja2.async_utils.async_variant` | `True` | `True` | `34,672` | `1,923` | `94.45%` | `636.97 ms` |
| 22 | `cross_symbol_dependency_closure` | `ky` | `typescript` | `headers` | `True` | `True` | `49,142` | `1,459` | `97.03%` | `572.42 ms` |
| 23 | `polyglot_python_query` | `loguru` | `python` | `loguru._asyncio_loop.load_loop_functions` | `True` | `True` | `23,207` | `1,703` | `92.66%` | `590.41 ms` |
| 24 | `symptom_and_signature_discovery` | `markdown-it-py` | `python` | `markdown_it._punycode.map_domain` | `True` | `True` | `30,006` | `2,133` | `92.89%` | `609.21 ms` |
| 25 | `repo_scoped_symbol_localization` | `markupsafe` | `python` | `markupsafe.EscapeFormatter` | `True` | `True` | `11,916` | `1,284` | `89.22%` | `586.85 ms` |
| 26 | `cross_symbol_dependency_closure` | `marshmallow` | `python` | `marshmallow.class_registry.get_class` | `True` | `True` | `33,064` | `1,388` | `95.8%` | `588.69 ms` |
| 27 | `polyglot_python_query` | `more-itertools` | `python` | `more_itertools.more.SequenceView` | `True` | `True` | `39,702` | `1,472` | `96.29%` | `639.0 ms` |
| 28 | `symptom_and_signature_discovery` | `orm` | `python` | `orm.fields.AutoNowMixin` | `True` | `True` | `9,296` | `1,375` | `85.21%` | `643.51 ms` |
| 29 | `repo_scoped_symbol_localization` | `packaging` | `python` | `packaging._elffile.ELFFile` | `True` | `True` | `44,774` | `3,632` | `91.89%` | `662.97 ms` |
| 30 | `cross_symbol_dependency_closure` | `pydantic-settings` | `python` | `pydantic_settings.exceptions.IncompleteFieldDefinitionWarning` | `True` | `True` | `11,800` | `3,272` | `72.27%` | `644.11 ms` |
| 31 | `polyglot_python_query` | `pyjwt` | `python` | `jwt.algorithms.Algorithm` | `True` | `True` | `68,066` | `2,138` | `96.86%` | `714.19 ms` |
| 32 | `symptom_and_signature_discovery` | `pyproject-hooks` | `python` | `pyproject_hooks._impl.BackendUnavailable` | `True` | `True` | `16,975` | `2,097` | `87.65%` | `636.05 ms` |
| 33 | `repo_scoped_symbol_localization` | `python-certifi` | `python` | `certifi.core.exit_cacert_ctx` | `True` | `True` | `14,456` | `1,260` | `91.28%` | `629.81 ms` |
| 34 | `cross_symbol_dependency_closure` | `python-dotenv` | `python` | `dotenv.cli.enumerate_env` | `True` | `True` | `24,153` | `5,584` | `76.88%` | `720.1 ms` |
| 35 | `polyglot_python_query` | `requests` | `python` | `requests._check_cryptography` | `True` | `True` | `19,568` | `1,440` | `92.64%` | `612.88 ms` |
| 36 | `symptom_and_signature_discovery` | `rich` | `python` | `rich.__main__.ColorBox` | `True` | `True` | `38,338` | `4,222` | `88.99%` | `617.97 ms` |
| 37 | `repo_scoped_symbol_localization` | `sniffio` | `python` | `sniffio._impl.current_async_library` | `True` | `True` | `14,192` | `2,000` | `85.91%` | `592.04 ms` |
| 38 | `cross_symbol_dependency_closure` | `stamina` | `python` | `stamina._config._Config` | `True` | `True` | `41,714` | `2,790` | `93.31%` | `607.49 ms` |
| 39 | `polyglot_python_query` | `starlette` | `python` | `starlette._exception_handler.wrap_app_handling_exceptions` | `True` | `True` | `35,387` | `8,017` | `77.34%` | `637.67 ms` |
| 40 | `symptom_and_signature_discovery` | `structlog` | `python` | `structlog._base.BoundLoggerBase._process_event` | `True` | `True` | `48,258` | `2,726` | `94.35%` | `686.33 ms` |
| 41 | `repo_scoped_symbol_localization` | `svcs` | `python` | `svcs._autowire._default_or_raise` | `True` | `True` | `46,418` | `2,673` | `94.24%` | `657.79 ms` |
| 42 | `cross_symbol_dependency_closure` | `tenacity` | `python` | `tenacity.BaseAction` | `True` | `True` | `24,708` | `899` | `96.36%` | `602.75 ms` |
| 43 | `polyglot_python_query` | `tomli` | `python` | `fuzzer.fuzz.normalize_toml_obj` | `True` | `True` | `13,057` | `1,730` | `86.75%` | `620.8 ms` |
| 44 | `symptom_and_signature_discovery` | `tomli-w` | `python` | `tomli_w._writer.format_decimal` | `True` | `True` | `33,893` | `3,066` | `90.95%` | `619.92 ms` |
| 45 | `repo_scoped_symbol_localization` | `typer` | `python` | `typer._click._compat._AtomicFile` | `True` | `True` | `103,605` | `1,753` | `98.31%` | `858.7 ms` |
| 46 | `cross_symbol_dependency_closure` | `urllib3` | `python` | `urllib3._base_connection._ResponseOptions` | `True` | `True` | `22,509` | `956` | `95.75%` | `583.68 ms` |
| 47 | `polyglot_python_query` | `uvicorn` | `python` | `uvicorn._ansi.style` | `True` | `True` | `47,634` | `2,719` | `94.29%` | `662.45 ms` |
| 48 | `symptom_and_signature_discovery` | `watchfiles` | `python` | `watchfiles.cli.build_filter` | `True` | `True` | `32,486` | `4,654` | `85.67%` | `637.46 ms` |
| 49 | `repo_scoped_symbol_localization` | `werkzeug` | `python` | `werkzeug._header_property._DSProto` | `True` | `True` | `8,117` | `1,493` | `81.61%` | `595.15 ms` |
| 50 | `cross_symbol_dependency_closure` | `zod` | `typescript` | `array` | `True` | `True` | `49,394` | `1,881` | `96.19%` | `589.21 ms` |
| 51 | `polyglot_go_query` | `enterprise_gitlab_ops` | `go` | `services.gateway.GatewayConfig` | `True` | `True` | `5,670` | `773` | `86.37%` | `601.35 ms` |
| 52 | `symptom_and_signature_discovery` | `enterprise_gitlab_ops` | `sql` | `schema.ledger_rls.execution_orders` | `True` | `True` | `5,746` | `873` | `84.81%` | `593.8 ms` |
| 53 | `repo_scoped_symbol_localization` | `enterprise_gitlab_ops` | `terraform` | `infra.main.alloydb_cluster` | `True` | `True` | `10,730` | `1,336` | `87.55%` | `702.1 ms` |
| 54 | `cross_symbol_dependency_closure` | `enterprise_gitlab_ops` | `tsx` | `services.AuditPanel.formatAuditSummary` | `True` | `True` | `10,782` | `1,220` | `88.68%` | `598.33 ms` |
| 55 | `polyglot_yaml_query` | `markdown-it-py` | `yaml` | `port` | `True` | `True` | `18,326` | `2,120` | `88.43%` | `593.39 ms` |
| 56 | `symptom_and_signature_discovery` | `urllib3` | `javascript` | `emscripten_fetch_worker` | `False` | `False` | `51,824` | `1,917` | `96.3%` | `719.33 ms` |
| 57 | `repo_scoped_symbol_localization` | `watchfiles` | `rust` | `lib.RustNotify` | `True` | `True` | `27,506` | `1,042` | `96.21%` | `588.63 ms` |
| 58 | `cross_symbol_dependency_closure` | `werkzeug` | `javascript` | `werkzeug.debug.shared.debugger.addConsoleIconToFrames` | `True` | `True` | `8,407` | `1,408` | `83.25%` | `637.5 ms` |
| 59 | `polyglot_javascript_query` | `zod` | `javascript` | `rollup.config` | `True` | `True` | `39,957` | `2,203` | `94.49%` | `605.43 ms` |
| 60 | `symptom_and_signature_discovery` | `chi` | `go` | `chi.Router` | `True` | `True` | `24,117` | `2,357` | `90.23%` | `599.17 ms` |
