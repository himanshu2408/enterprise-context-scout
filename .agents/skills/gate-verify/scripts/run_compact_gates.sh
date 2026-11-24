#!/usr/bin/env bash
set -euo pipefail

export CI=true
export GIT_TERMINAL_PROMPT=0
export CONTEXT_SCOUT_USE_LIVE_MLX="${CONTEXT_SCOUT_USE_LIVE_MLX:-0}"

SCOPE="all"
PYTEST_TARGET="tests/"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scope) SCOPE="$2"; shift 2 ;;
    --pytest-target) PYTEST_TARGET="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

STEP_TIMEOUT="${GATE_STEP_TIMEOUT:-900}"
if [[ "$SCOPE" == "fast" ]]; then
  STEP_TIMEOUT="${GATE_STEP_TIMEOUT:-180}"
fi

run_step() {
  local label="$1"
  shift
  local tmp_out
  tmp_out="$(mktemp "${TMPDIR:-/tmp}/gate_verify.XXXXXX")"
  echo "[RUN ] ${label} ..."
  set +e
  python3 - "$STEP_TIMEOUT" "$tmp_out" "$@" <<'PY'
import os
import signal
import subprocess
import sys

timeout = int(sys.argv[1])
out_path = sys.argv[2]
cmd = sys.argv[3:]

with open(out_path, "wb") as f:
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        rc = proc.wait(timeout=timeout)
        sys.exit(rc)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        f.write(f"\n[TIMEOUT] Command exceeded {timeout}s and was killed.\n".encode())
        sys.exit(124)
PY
  local rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    local summary
    summary="$(tail -n 3 "$tmp_out" | tr '\n' ' ' | cut -c1-140)"
    echo "[PASS] ${label} :: ${summary}"
    rm -f "$tmp_out"
  else
    echo "[FAIL] ${label} (exit ${rc}) — last 45 lines:"
    tail -n 45 "$tmp_out"
    rm -f "$tmp_out"
    exit "$rc"
  fi
}

echo "=== Compact Gate Runner (scope=${SCOPE}) ==="

if git check-ignore -q AGENTS.md 2>/dev/null; then
  echo "[FAIL] gitignore-hygiene :: AGENTS.md is ignored by .gitignore!"
  exit 1
else
  echo "[PASS] gitignore-hygiene :: AGENTS.md is tracked cleanly"
fi

if [[ "$SCOPE" == "all" || "$SCOPE" == "py" || "$SCOPE" == "fast" ]]; then
  run_step "ruff-check" uv run ruff check .
  run_step "ruff-format-check" uv run ruff format --check .
  if grep -q "\[tool.mypy\]" pyproject.toml 2>/dev/null && uv run mypy --version >/dev/null 2>&1; then
    run_step "mypy-src" uv run mypy --strict src tests
  fi
  run_step "squawk-migrations" uv run python -m context_scout.db.migrations_runner --lint-squawk
  if [[ "$SCOPE" == "fast" ]]; then
    local_batch_timeout="$STEP_TIMEOUT"
    STEP_TIMEOUT="${GATE_FAST_TOTAL_TIMEOUT:-900}"
    FAST_PY_CODE="$(cat <<'PY'
import os
import re
import signal
import subprocess
import sys

batch_timeout = int(sys.argv[1])
target = sys.argv[2]

ignore_args = [
    "--ignore-glob=tests/e2e/*.py",
    "--ignore-glob=tests/test_live_*.py",
    "--ignore-glob=tests/test_step*.py",
    "--ignore-glob=tests/test_ws1_*.py",
    "--ignore-glob=tests/test_real_paper_*.py",
    "--ignore-glob=tests/test_zero_spend_*.py",
    "--ignore-glob=tests/test_postgres_*.py",
]

collect = subprocess.run(
    ["uv", "run", "pytest", target, *ignore_args, "--collect-only", "-q"],
    capture_output=True,
    text=True,
    check=False,
)
if collect.returncode != 0:
    sys.stdout.write(collect.stdout)
    sys.stderr.write(collect.stderr)
    sys.exit(collect.returncode)

node_files: list[str] = []
seen: set[str] = set()
for line in collect.stdout.splitlines():
    stripped = line.strip()
    fpath = ""
    if "::" in stripped:
        fpath = stripped.split("::", 1)[0].strip()
    elif re.match(r"^[^:\s]+\.py:\s*\d+$", stripped):
        fpath = stripped.rsplit(":", 1)[0].strip()
    if fpath and fpath not in seen:
        seen.add(fpath)
        node_files.append(fpath)

if not node_files:
    print("No test files collected for fast scope.", flush=True)
    sys.exit(1)

batch_size = 10
batches = [node_files[i : i + batch_size] for i in range(0, len(node_files), batch_size)]
totals = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
summary_re = re.compile(r"(\d+)\s+(passed|failed|skipped|errors?)\b")

for b_idx, batch in enumerate(batches, start=1):
    cmd = ["uv", "run", "pytest", *batch, "-v", "-rs", "--tb=short", "--maxfail=5"]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=batch_timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        print(f"[TIMEOUT] pytest-fast batch {b_idx}/{len(batches)} exceeded {batch_timeout}s: {batch}", flush=True)
        sys.exit(124)

    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    tail = lines[-1] if lines else f"exit={proc.returncode}"
    for n_str, kind in summary_re.findall(tail):
        key = "errors" if kind.startswith("error") else kind
        totals[key] += int(n_str)

    if proc.returncode != 0:
        print(f"[FAIL] pytest-fast batch {b_idx}/{len(batches)} ({batch[0]}..{batch[-1]}):", flush=True)
        print("\n".join(out.splitlines()[-45:]), flush=True)
        sys.exit(proc.returncode)

    print(f"[BATCH {b_idx}/{len(batches)}] {len(batch)} files :: {tail}", flush=True)

if totals["failed"] > 0 or totals["errors"] > 0 or totals["skipped"] > 0 or totals["passed"] == 0:
    print(f"[FAIL] pytest-fast totals violated zero-fail/zero-skip gate: {totals}", flush=True)
    sys.exit(1)

print(
    f"{totals['passed']} passed, {totals['failed']} failed, "
    f"{totals['skipped']} skipped, {totals['errors']} errors across {len(node_files)} files",
    flush=True,
)
PY
)"
    run_step "pytest-fast (${PYTEST_TARGET})" env PYTHONUNBUFFERED=1 python3 -c "$FAST_PY_CODE" "$local_batch_timeout" "$PYTEST_TARGET"
    STEP_TIMEOUT="$local_batch_timeout"
  else
    ALL_PY_CODE="$(cat <<'PY'
import re
import subprocess
import sys

target = sys.argv[1]
proc = subprocess.run(
    ["uv", "run", "pytest", target, "-v", "-rs", "--tb=short", "--maxfail=5"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    check=False,
)
sys.stdout.write(proc.stdout)
sys.stdout.flush()
if proc.returncode != 0:
    sys.exit(proc.returncode)

summary_re = re.compile(r"(\d+)\s+(passed|failed|skipped|errors?)\b")
lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
tail = lines[-1] if lines else ""
counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
for n_str, kind in summary_re.findall(tail):
    key = "errors" if kind.startswith("error") else kind
    counts[key] += int(n_str)

if counts["failed"] > 0 or counts["errors"] > 0 or counts["skipped"] > 0 or counts["passed"] == 0:
    print(f"[FAIL] pytest zero-skip/zero-fail enforcement failed: {counts}", file=sys.stderr, flush=True)
    sys.exit(1)

print(
    f"{counts['passed']} passed, {counts['failed']} failed, "
    f"{counts['skipped']} skipped, {counts['errors']} errors",
    flush=True,
)
PY
)"
    run_step "pytest (${PYTEST_TARGET})" env PYTHONUNBUFFERED=1 python3 -c "$ALL_PY_CODE" "$PYTEST_TARGET"
  fi
fi

if [[ "$SCOPE" == "all" || "$SCOPE" == "ui" || "$SCOPE" == "fast" ]]; then
  if [[ -d "web-ui" && -f "web-ui/package.json" ]]; then
    run_step "web-ui:lint" npm --prefix web-ui run lint
    run_step "web-ui:typecheck" npm --prefix web-ui run typecheck
    run_step "web-ui:test" npm --prefix web-ui run test
    run_step "web-ui:build" npm --prefix web-ui run build
  fi
fi

if command -v gitleaks >/dev/null 2>&1; then
  run_step "gitleaks" gitleaks detect --no-banner --redact
fi

echo "=== ALL REQUESTED GATES PASSED ==="
