#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

from context_scout.tuning.specialization_compiler import (
    SpecializationCompilationError,
    SpecializationCompiler,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    domains_dir = Path("domains")
    if not domains_dir.exists():
        print("[FAIL] domains/ directory not found.")
        return 1

    compiler = SpecializationCompiler(domains_dir=domains_dir)
    failures: list[str] = []
    checked = 0

    for dom_dir in sorted(p for p in domains_dir.iterdir() if p.is_dir()):
        checked += 1
        spec_file = dom_dir / "specialization.yaml"
        if spec_file.exists():
            spec = yaml.safe_load(spec_file.read_text(encoding="utf-8")) or {}
            lora_pol = spec.get("lora_policy") or {}
            if lora_pol.get("enabled", False):
                ceiling = spec.get("training_scope_ceiling") or []
                if not ceiling:
                    failures.append(f"{dom_dir.name}: missing training_scope_ceiling")
                if lora_pol.get("use_dora") is True:
                    failures.append(f"{dom_dir.name}: use_dora=True breaks vLLM")
                if lora_pol.get("use_rslora") is not True:
                    failures.append(f"{dom_dir.name}: use_rslora=True is required")
                if lora_pol.get("assistant_only_loss") is not True:
                    failures.append(f"{dom_dir.name}: assistant_only_loss=True is required")
                rank = int(lora_pol.get("rank", 16))
                if rank > 32 or rank not in {8, 16, 32}:
                    failures.append(f"{dom_dir.name}: rank={rank} violates {{8,16,32}} policy")
                replay_ratio = float(lora_pol.get("general_replay_ratio", 0.0))
                if not (0.10 <= replay_ratio <= 0.50):
                    failures.append(
                        f"{dom_dir.name}: general_replay_ratio={replay_ratio} outside [0.10, 0.50]"
                    )
            try:
                domain_yaml = dom_dir / "domain.yaml"
                compiler.compile(
                    spec_file,
                    domain_yaml_path=domain_yaml if domain_yaml.is_file() else None,
                )
            except (SpecializationCompilationError, ValueError) as exc:
                failures.append(f"{dom_dir.name}: specialization compile failed: {exc}")

        tuning_dir = dom_dir / "tuning"
        if tuning_dir.exists():
            hashes: dict[str, str] = {}
            for split_name in ("train.jsonl", "valid.jsonl", "eval.jsonl"):
                fp = tuning_dir / split_name
                if fp.exists() and fp.stat().st_size > 0:
                    hashes[split_name] = sha256_file(fp)
            seen: dict[str, str] = {}
            for split_name, digest in hashes.items():
                if digest in seen:
                    failures.append(
                        f"{dom_dir.name}: DATA LEAKAGE! {split_name} == {seen[digest]} "
                        f"({digest[:12]}...)"
                    )
                seen[digest] = split_name

        adapters_dir = dom_dir / "adapters"
        if adapters_dir.is_dir():
            for manifest_fp in sorted(adapters_dir.rglob("artifact_manifest.json")):
                payload = json.loads(manifest_fp.read_text(encoding="utf-8"))
                if int(payload.get("canary_leak_count", 0)) != 0:
                    failures.append(
                        f"{dom_dir.name}: canary leak detected in {manifest_fp}: "
                        f"{payload.get('canary_leak_count')}"
                    )
            for metrics_fp in sorted(adapters_dir.rglob("training_metrics.json")):
                metrics = json.loads(metrics_fp.read_text(encoding="utf-8"))
                canary_audit = metrics.get("canary_audit")
                if (
                    isinstance(canary_audit, dict)
                    and int(canary_audit.get("canary_leak_count", 0)) != 0
                ):
                    failures.append(f"{dom_dir.name}: canary_audit leak detected in {metrics_fp}")

    if failures:
        print("[FAIL] BYO-LoRA & Dataset Split Audit failed:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print(f"[PASS] Audited {checked} domain pack(s): zero split collisions or LoRA issues.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
