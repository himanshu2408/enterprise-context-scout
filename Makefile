.PHONY: install lint format-check typecheck test-fast test-full test-ui ci

PY_DIRS := src tests
PER_FILE_TIMEOUT_S := 120

install:
	uv sync --extra dev
	cd web-ui && npm ci

lint:
	uv run ruff check $(PY_DIRS)
	cd web-ui && npm run lint

format-check:
	uv run ruff format --check $(PY_DIRS)

typecheck:
	uv run mypy --strict $(PY_DIRS)
	cd web-ui && npm run typecheck

test-fast:
	uv run pytest -q -m "not postgres and not live and not benchmark"

test-full:
	@uv run python -c '\
	import pathlib, re, subprocess, sys; \
	files = sorted(pathlib.Path("tests").rglob("test_*.py")); \
	totals = {"files_run": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0, "timed_out": 0}; \
	failures = []; \
	summary_re = re.compile(r"(\d+)\s+(passed|failed|skipped|errors?)\b"); \
	for f in files: \
	    totals["files_run"] += 1; \
	    try: \
	        r = subprocess.run(["uv", "run", "pytest", "-rs", "--tb=short", str(f)], capture_output=True, text=True, timeout=$(PER_FILE_TIMEOUT_S)); \
	        out = r.stdout + "\n" + r.stderr; \
	        lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]; \
	        tail = lines[-1] if lines else f"exit={r.returncode}"; \
	        counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}; \
	        for n_str, kind in summary_re.findall(tail): \
	            key = "errors" if kind.startswith("error") else kind; \
	            counts[key] += int(n_str); \
	        for k, v in counts.items(): totals[k] += v; \
	        print(f"=== {f} === {tail}", flush=True); \
	        if r.returncode != 0: \
	            failures.append((str(f), f"exit={r.returncode}")); \
	            print(out, file=sys.stderr, flush=True); \
	    except subprocess.TimeoutExpired: \
	        totals["timed_out"] += 1; \
	        failures.append((str(f), "TIMEOUT")); \
	        print(f"=== {f} === TIMEOUT ($(PER_FILE_TIMEOUT_S)s)", file=sys.stderr, flush=True); \
	summary = " ".join(f"{k}={v}" for k, v in totals.items()); \
	print(f"GRAND TOTAL: {summary}", flush=True); \
	if failures: \
	    print(f"FAILED FILES ({len(failures)}): {failures}", file=sys.stderr); \
	    sys.exit(1)'

test-ui:
	cd web-ui && npm run test && npm run build

ci: lint format-check typecheck test-fast test-ui
