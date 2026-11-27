#!/usr/bin/env python3
"""Duration-tiered LPT parallel pytest batch runner for local and CI verification."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Final

DEFAULT_FALLBACK_SECONDS: Final[float] = 6.0

INITIAL_HEAVY_ESTIMATES_S: Final[dict[str, float]] = {
    "tests/test_m2_code_engine.py": 120.0,
    "tests/test_m6_eval_harness.py": 95.0,
    "tests/test_m0_02_governance_and_eval_fixes.py": 85.0,
    "tests/test_release_readiness_and_seed_demo.py": 70.0,
    "tests/test_m6_02_pitr_reconciler_and_chaos_dr.py": 90.0,
    "tests/test_m4_01_vllm_sidecar_and_asi05.py": 75.0,
    "tests/test_m5_01_02_zero_trust_iap_oidc_scim_audit_mcp.py": 75.0,
    "tests/test_m4_02_specialization_distiller_peft.py": 65.0,
    "tests/test_owasp_asi01_to_asi10_matrix.py": 60.0,
    "tests/test_tier5_scale_statistical_adversarial.py": 55.0,
    "tests/test_zero_spend_abstention.py": 55.0,
    "tests/test_m2_schemathesis_openapi_fuzz.py": 55.0,
    "tests/test_m5_adk_workflow.py": 45.0,
    "tests/test_m8_fastapi_sse.py": 40.0,
    "tests/test_m7_qlora_tuning.py": 35.0,
    "tests/test_m7_teacher_distiller.py": 30.0,
    "tests/test_l9_mixed_tenant_50_sse_mcp_concurrency.py": 30.0,
    "tests/test_live_unscripted_semantic_eval.py": 25.0,
    "tests/test_live_stack_e2e_smoke.py": 25.0,
    "tests/test_m1_03_gcs_embedder_halfvec.py": 20.0,
    "tests/test_adversarial_ast_rls.py": 20.0,
    "tests/test_postgres_m2_live.py": 20.0,
}

_PASSED_RE: Final[re.Pattern[str]] = re.compile(r"(\d+)\s+passed")


def discover_test_files(repo_root: Path) -> list[str]:
    """Dynamically discovers 100% of ``tests/**/test_*.py`` files in sorted order."""
    tests_dir = repo_root / "tests"
    if not tests_dir.is_dir():
        raise FileNotFoundError(f"Tests directory not found: {tests_dir}")
    discovered = sorted(
        {p.relative_to(repo_root).as_posix() for p in tests_dir.rglob("test_*.py") if p.is_file()}
    )
    if not discovered:
        raise RuntimeError(f"No test_*.py files discovered under {tests_dir}")
    return discovered


def _resolve_duration_cache_path(repo_root: Path) -> Path:
    proc = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        common_dir = Path(proc.stdout.strip())
        if not common_dir.is_absolute():
            common_dir = (repo_root / common_dir).resolve()
        return common_dir / "remediation" / "pytest_durations.json"
    return repo_root / ".git" / "remediation" / "pytest_durations.json"


def _load_duration_cache(cache_path: Path) -> dict[str, float]:
    if not cache_path.is_file():
        return {}
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    except (OSError, ValueError, TypeError):
        return {}
    return {}


def _save_duration_cache(cache_path: Path, durations: dict[str, float]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(durations, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(cache_path)


def estimate_file_duration(repo_root: Path, rel_path: str, cache: dict[str, float]) -> float:
    """Returns cached elapsed seconds or a byte-size / whale-table initial estimate."""
    if rel_path in cache and cache[rel_path] > 0.0:
        return cache[rel_path]
    if rel_path in INITIAL_HEAVY_ESTIMATES_S:
        return INITIAL_HEAVY_ESTIMATES_S[rel_path]
    fp = repo_root / rel_path
    if fp.is_file():
        kb = fp.stat().st_size / 1024.0
        return max(2.0, min(18.0, 2.0 + kb * 0.15))
    return DEFAULT_FALLBACK_SECONDS


def build_batches(
    files: list[str],
    estimates: dict[str, float],
    *,
    mode: str,
    workers: int,
) -> list[tuple[list[str], float]]:
    """Groups files into duration-tiered batches sorted longest-estimated-first (LPT)."""
    ordered = sorted(files, key=lambda f: (-estimates[f], f))
    batches: list[tuple[list[str], float]] = []

    if mode == "files" and len(ordered) <= workers:
        for f in ordered:
            batches.append(([f], estimates[f]))
        return batches

    heavy: list[str] = []
    medium: list[str] = []
    fast: list[str] = []
    for f in ordered:
        est = estimates[f]
        if est >= 20.0:
            heavy.append(f)
        elif est >= 10.0:
            medium.append(f)
        else:
            fast.append(f)

    for f in heavy:
        batches.append(([f], estimates[f]))

    med_size = 3 if mode == "unit-offline" else 4
    for idx in range(0, len(medium), med_size):
        chunk = medium[idx : idx + med_size]
        batches.append((chunk, sum(estimates[f] for f in chunk)))

    fast_size = 8 if mode in ("unit-offline", "fast") else 10
    for idx in range(0, len(fast), fast_size):
        chunk = fast[idx : idx + fast_size]
        batches.append((chunk, sum(estimates[f] for f in chunk)))

    batches.sort(key=lambda item: (-item[1], item[0][0]))
    return batches


def _verify_zero_drop(expected: list[str], batches: list[tuple[list[str], float]]) -> None:
    assigned: list[str] = []
    for chunk, _ in batches:
        assigned.extend(chunk)
    if len(assigned) != len(set(assigned)):
        raise RuntimeError("Duplicate test file detected in batch assignment.")
    if set(assigned) != set(expected):
        missing = sorted(set(expected) - set(assigned))
        extra = sorted(set(assigned) - set(expected))
        raise RuntimeError(f"Zero-drop invariant violated: missing={missing}, extra={extra}")


def _build_worker_env(mode: str) -> dict[str, str]:
    env = dict(os.environ)
    env["SCOUT_PARALLEL_PYTEST"] = "1"
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if mode in ("unit-offline", "fast"):
        env["SCOUT_STORAGE_BACKEND"] = "memory"
        env["SCOUT_DB_APP_URL"] = "postgresql://scout_app_user:pw@127.0.0.1:59999/context_scout"
        env["SCOUT_DB_ADMIN_URL"] = "postgresql://postgres:pw@127.0.0.1:59999/context_scout"
        env["SCOUT_PG_CONNECT_TIMEOUT"] = "1"
    return env


def _prewarm_pg17_once(repo_root: Path) -> None:
    """Pre-warms PG17 schema and neural seed corpora once in the parent process."""
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))
    from tests.fixtures.pg_isolation import ensure_pg17_schema_and_seed_corpora

    from context_scout.config import ScoutSettings

    settings = ScoutSettings()
    ensure_pg17_schema_and_seed_corpora(
        db_admin_url=settings.db_admin_url,
        db_app_url=settings.db_app_url,
        repo_root=repo_root,
    )


def _run_batch(
    batch_id: int,
    files: list[str],
    *,
    marker_expr: str,
    repo_root: Path,
    env: dict[str, str],
) -> tuple[int, list[str], float, int, str, int]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *files,
        "-m",
        marker_expr,
        "--tb=short",
    ]
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.monotonic() - t0
    combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    passed_matches = _PASSED_RE.findall(combined)
    passed_count = sum(int(m) for m in passed_matches)
    return batch_id, files, elapsed, proc.returncode, combined, passed_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Duration-tiered LPT parallel pytest runner.")
    parser.add_argument(
        "--mode",
        choices=("unit-offline", "fast", "files", "full"),
        default="unit-offline",
        help="Execution mode (unit-offline, fast, files, full).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 4),
        help="Maximum concurrent pytest worker processes.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Explicit test files when --mode files is selected.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    cache_path = _resolve_duration_cache_path(repo_root)
    cache = _load_duration_cache(cache_path)

    all_files = discover_test_files(repo_root)
    if args.mode == "files":
        if not args.files:
            print("ERROR: --mode files requires at least one test file.", file=sys.stderr)
            return 2
        selected_files = [
            Path(f).resolve().relative_to(repo_root).as_posix() if Path(f).is_absolute() else f
            for f in args.files
        ]
        marker_expr = "not live"
    elif args.mode == "fast":
        selected_files = [
            f for f in all_files if estimate_file_duration(repo_root, f, cache) < 15.0
        ]
        marker_expr = "not postgres and not live and not quant_dogfood"
    elif args.mode == "unit-offline":
        selected_files = all_files
        marker_expr = "not postgres and not live and not quant_dogfood"
    else:
        selected_files = all_files
        marker_expr = "not live"

    estimates = {f: estimate_file_duration(repo_root, f, cache) for f in selected_files}
    workers = max(1, int(args.workers))
    batches = build_batches(selected_files, estimates, mode=args.mode, workers=workers)
    _verify_zero_drop(selected_files, batches)

    if args.mode == "full":
        t_warm = time.monotonic()
        print("Pre-warming PG17 schema + neural seed once in parent process...")
        _prewarm_pg17_once(repo_root)
        print(f"Pre-warm completed in {time.monotonic() - t_warm:.2f}s.")

    env = _build_worker_env(args.mode)
    print(
        f"Running mode={args.mode} workers={workers} "
        f"files={len(selected_files)} batches={len(batches)} marker={marker_expr!r}"
    )

    t_start = time.monotonic()
    completed_results: list[tuple[int, list[str], float, int]] = []
    total_passed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[Future[tuple[int, list[str], float, int, str, int]], int] = {}
        for idx, (chunk, _est) in enumerate(batches, start=1):
            fut = executor.submit(
                _run_batch,
                idx,
                chunk,
                marker_expr=marker_expr,
                repo_root=repo_root,
                env=env,
            )
            futures[fut] = idx

        pending = set(futures.keys())
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                batch_id, chunk, elapsed, rc, output, passed_count = fut.result()
                total_passed += passed_count
                is_deselected_empty = rc == 5 and "deselected" in output.lower()
                if rc != 0 and not is_deselected_empty:
                    for p_fut in pending:
                        p_fut.cancel()
                    print(
                        f"\n[FAIL] Batch {batch_id}/{len(batches)} failed "
                        f"(rc={rc}, {elapsed:.2f}s): {' '.join(chunk)}",
                        file=sys.stderr,
                    )
                    print(output, file=sys.stderr)
                    return rc if rc != 0 else 1
                if rc == 0 and passed_count == 0:
                    for p_fut in pending:
                        p_fut.cancel()
                    print(
                        f"\n[FAIL] Batch {batch_id}/{len(batches)} passed 0 tests "
                        f"unexpectedly: {' '.join(chunk)}",
                        file=sys.stderr,
                    )
                    print(output, file=sys.stderr)
                    return 1

                completed_results.append((batch_id, chunk, elapsed, passed_count))
                label = "deselected" if is_deselected_empty else f"{passed_count} passed"
                print(
                    f"[OK] Batch {batch_id}/{len(batches)} ({len(chunk)} files, "
                    f"{label}, {elapsed:.2f}s)"
                )

                if len(chunk) == 1:
                    cache[chunk[0]] = round(elapsed, 3)
                else:
                    est_sum = sum(estimates[f] for f in chunk) or float(len(chunk))
                    for f in chunk:
                        cache[f] = round(elapsed * (estimates[f] / est_sum), 3)

    _save_duration_cache(cache_path, cache)
    total_wall = time.monotonic() - t_start

    if total_passed == 0:
        print("ERROR: Total passed tests across all batches was 0.", file=sys.stderr)
        return 1

    print("\n=== Timing Leaderboard (Slowest 10 Batches) ===")
    for batch_id, chunk, elapsed, passed_count in sorted(completed_results, key=lambda x: -x[2])[
        :10
    ]:
        preview = chunk[0] if len(chunk) == 1 else f"{chunk[0]} (+{len(chunk) - 1} more)"
        print(f"  {elapsed:6.2f}s | {passed_count:4d} passed | batch #{batch_id}: {preview}")
    print(
        f"TOTAL: {total_passed} passed across {len(selected_files)} files "
        f"in {total_wall:.2f}s wall-clock (workers={workers})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
