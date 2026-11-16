.PHONY: install lint format-check typecheck test-fast test-full test-ui ci

PY_DIRS := src tests
PER_FILE_TIMEOUT_S := 120

define _TEST_FULL_PY
import os
import pathlib
import re
import signal
import subprocess
import sys

timeout_s = int(sys.argv[1])
files = sorted(pathlib.Path("tests").rglob("test_*.py"))
totals = {"files_run": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0, "timed_out": 0}
failures = []
summary_re = re.compile(r"(\d+)\s+(passed|failed|skipped|errors?)\b")

for f in files:
    totals["files_run"] += 1
    proc = subprocess.Popen(
        ["uv", "run", "pytest", "-rs", "--tb=short", str(f)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env={**os.environ, "SCOUT_EMBEDDING_DEVICE": "cpu"},
    )
    try:
        stdout_s, stderr_s = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        totals["timed_out"] += 1
        failures.append((str(f), "TIMEOUT"))
        print(f"=== {f} === TIMEOUT ({timeout_s}s)", file=sys.stderr, flush=True)
        continue

    out = stdout_s + "\n" + stderr_s
    lines = [ln.strip() for ln in stdout_s.splitlines() if ln.strip()]
    tail = lines[-1] if lines else f"exit={proc.returncode}"
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for n_str, kind in summary_re.findall(tail):
        key = "errors" if kind.startswith("error") else kind
        counts[key] += int(n_str)
    for k, v in counts.items():
        totals[k] += v
    print(f"=== {f} === {tail}", flush=True)
    if proc.returncode != 0:
        failures.append((str(f), f"exit={proc.returncode}"))
        print(out, file=sys.stderr, flush=True)

summary = " ".join(f"{k}={v}" for k, v in totals.items())
print(f"GRAND TOTAL: {summary}", flush=True)
if failures:
    print(f"FAILED FILES ({len(failures)}): {failures}", file=sys.stderr)
    sys.exit(1)
endef
export _TEST_FULL_PY

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
	@uv run python -c "$$_TEST_FULL_PY" $(PER_FILE_TIMEOUT_S)

test-ui:
	cd web-ui && npm run test && npm run build

ci: lint format-check typecheck test-fast test-ui
