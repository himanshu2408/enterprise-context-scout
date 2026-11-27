#!/usr/bin/env python3
"""Deterministic Longest-Processing-Time (LPT) pytest shard allocator for CI.

Partitions 100% of ``tests/**/test_*.py`` across ``N`` balanced CI shards using
empirical Ubuntu GitHub Actions CPU durations and in-process embedding-cache
affinity groups while guaranteeing zero missed or duplicated test modules.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

DEFAULT_FILE_WEIGHT_SECONDS: Final[int] = 6

# Empirical pure-test durations (in seconds) measured directly via pytest --durations
# on GitHub Actions ubuntu-latest (4-vCPU x86_64 CPU).
MEASURED_UBUNTU_CI_SECONDS: Final[dict[str, int]] = {
    "tests/test_m6_eval_harness.py": 870,
    "tests/test_m0_02_governance_and_eval_fixes.py": 745,
    "tests/test_release_readiness_and_seed_demo.py": 490,
    "tests/test_unseen_repo_ab_benchmark.py": 240,
    "tests/test_m4_02_specialization_distiller_peft.py": 180,
    "tests/test_step2_3_intent_federated_routing.py": 180,
    "tests/test_zero_spend_abstention.py": 150,
    "tests/test_m2_schemathesis_openapi_fuzz.py": 145,
    "tests/test_tier5_scale_statistical_adversarial.py": 90,
    "tests/test_live_unscripted_semantic_eval.py": 90,
    "tests/test_tier5_adversarial_coverage.py": 60,
    "tests/test_real_paper_harvest_live.py": 60,
    "tests/test_end_to_end_feature_build.py": 55,
    "tests/test_m2_code_engine.py": 45,
    "tests/test_backend_middleware_authenticity.py": 45,
    "tests/test_postgres_m2_live.py": 25,
    "tests/test_m7_teacher_distiller.py": 22,
    "tests/test_adversarial_ast_rls.py": 20,
    "tests/test_live_stack_e2e_smoke.py": 18,
    "tests/test_m1_03_gcs_embedder_halfvec.py": 16,
    "tests/test_m7_qlora_tuning.py": 14,
    "tests/test_claim_verifier.py": 14,
    "tests/test_m3_03_nonce_guardrails_and_mutation_gate.py": 12,
    "tests/test_m4_mcp_servers.py": 12,
    "tests/test_sec_vuln_01_regression.py": 12,
    "tests/test_owasp_asi01_to_asi10_matrix.py": 12,
    "tests/test_adversarial_security.py": 12,
    "tests/test_auth_fail_closed_hmac.py": 10,
    "tests/test_m8_fastapi_sse.py": 10,
    "tests/test_m6_01_containers_helm_tf_aibom.py": 10,
    "tests/test_step3_enterprise_qualification.py": 10,
    "tests/test_m2_03_side_effects_contracts_wiki.py": 10,
}

# Co-located module groups that share large in-RAM embedding caches (_SYMBOL_EMBEDDING_CACHE
# or _TEXT_VECTOR_CACHE), eliminating ~30m of duplicate cold CPU re-embedding across shards.
CACHE_AFFINITY_GROUPS: Final[tuple[tuple[str, ...], ...]] = (
    (
        "tests/test_m2_code_engine.py",
        "tests/test_postgres_m2_live.py",
    ),
    (
        "tests/test_m0_02_governance_and_eval_fixes.py",
        "tests/test_zero_spend_abstention.py",
    ),
)


def discover_test_files(repo_root: Path) -> list[str]:
    """Discovers all pytest test modules under tests/ in deterministic sorted order."""
    tests_dir = repo_root / "tests"
    if not tests_dir.is_dir():
        raise FileNotFoundError(f"Tests directory not found: {tests_dir}")
    discovered = sorted(
        {p.relative_to(repo_root).as_posix() for p in tests_dir.rglob("test_*.py") if p.is_file()}
    )
    if not discovered:
        raise RuntimeError(f"No test_*.py files discovered under {tests_dir}")
    return discovered


def estimate_weight_seconds(rel_path: str) -> int:
    """Returns estimated Ubuntu CI execution time in seconds for a test module."""
    return MEASURED_UBUNTU_CI_SECONDS.get(rel_path, DEFAULT_FILE_WEIGHT_SECONDS)


def compute_shards(
    repo_root: Path,
    total_shards: int,
) -> tuple[list[list[str]], list[int]]:
    """Partitions all discovered test modules across ``total_shards`` via LPT bin-packing."""
    all_files = discover_test_files(repo_root)
    all_set = set(all_files)
    if total_shards < 1 or total_shards > len(all_files):
        raise ValueError(f"total_shards must be in [1, {len(all_files)}], got {total_shards}")

    grouped_members: set[str] = set()
    units: list[tuple[int, tuple[str, ...]]] = []
    for grp in CACHE_AFFINITY_GROUPS:
        present = tuple(p for p in grp if p in all_set and p not in grouped_members)
        if present:
            grouped_members.update(present)
            units.append((sum(estimate_weight_seconds(p) for p in present), present))

    for rel_path in all_files:
        if rel_path not in grouped_members:
            units.append((estimate_weight_seconds(rel_path), (rel_path,)))

    units.sort(key=lambda item: (-item[0], item[1]))

    bins: list[list[str]] = [[] for _ in range(total_shards)]
    loads: list[int] = [0 for _ in range(total_shards)]

    for unit_weight, unit_paths in units:
        target_idx = min(
            range(total_shards),
            key=lambda idx: (loads[idx], len(bins[idx]), idx),
        )
        bins[target_idx].extend(unit_paths)
        loads[target_idx] += unit_weight

    for shard_files in bins:
        shard_files.sort()

    seen: set[str] = set()
    for idx, shard_files in enumerate(bins, start=1):
        if not shard_files:
            raise AssertionError(f"Shard {idx}/{total_shards} was assigned 0 test files.")
        overlap = seen.intersection(shard_files)
        if overlap:
            raise AssertionError(f"Duplicate test assignment across shards: {sorted(overlap)}")
        seen.update(shard_files)

    if seen != all_set:
        missing = sorted(all_set - seen)
        extra = sorted(seen - all_set)
        raise AssertionError(f"Shard partition mismatch: missing={missing}, extra={extra}")

    return bins, loads


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute balanced LPT pytest shards.")
    parser.add_argument("--total-shards", type=int, default=6)
    parser.add_argument("--shard", type=int, default=None, help="1-based shard index to emit.")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print human-readable shard balance summary and verify partition invariants.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args(argv)

    bins, loads = compute_shards(args.repo_root.resolve(), args.total_shards)

    if args.summary:
        total_files = sum(len(b) for b in bins)
        total_sec = sum(loads)
        print(
            f"Verified {total_files} test modules across {args.total_shards} shards "
            f"(estimated total={total_sec / 60:.1f}m, max={max(loads) / 60:.1f}m, "
            f"min={min(loads) / 60:.1f}m):"
        )
        for idx, (shard_files, load_sec) in enumerate(zip(bins, loads, strict=True), start=1):
            heavy = sorted(
                shard_files,
                key=lambda p: (-estimate_weight_seconds(p), p),
            )[:3]
            heavy_desc = ", ".join(f"{Path(p).name}({estimate_weight_seconds(p)}s)" for p in heavy)
            print(
                f"  Shard {idx}/{args.total_shards}: {len(shard_files):2d} files | "
                f"~{load_sec:4d}s ({load_sec / 60:4.1f}m) | top: {heavy_desc}"
            )

    if args.shard is not None:
        if args.shard < 1 or args.shard > args.total_shards:
            raise SystemExit(f"--shard must be between 1 and {args.total_shards}, got {args.shard}")
        print(" ".join(bins[args.shard - 1]))

    if not args.summary and args.shard is None:
        parser.error("Specify --shard <1..N> and/or --summary.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
