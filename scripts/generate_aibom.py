#!/usr/bin/env python3
"""CycloneDX v1.6 SBOM & AI-BOM Generator with Supply-Chain Gates (M5-06).

Covers Python (`pyproject.toml`), Frontend (`web-ui/package.json`), governed AI
models (`config/model_registry.yaml`), and domain LoRA cartridges
(`domains/*/domain.yaml`, `domains/*/specialization.yaml`, `domains/*/adapters/**`).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import struct
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from context_scout.config import UNSAFE_WEIGHT_EXTENSIONS, ModelGovernancePolicy
from context_scout.serving.adapter_sync_sidecar import (
    CosignBlobVerifier,
    sign_blob_for_cosign,
)

MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024
_PEP508_SPLIT_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+(?:\[[A-Za-z0-9_,.\-]+\])?)\s*(.*)$")


class SupplyChainGateError(ValueError):
    """Raised fail-closed when SBOM/AI-BOM governance or security gates fail."""


@dataclass(frozen=True)
class CartridgeAuditResult:
    """Validated metadata and supply-chain status for one domain LoRA cartridge."""

    domain_id: str
    adapter_id: str
    base_model_id: str
    spdx_license: str
    rank: int
    use_rslora: bool
    use_dora: bool
    training_scope_ceiling: list[str]
    safetensors_files: list[dict[str, Any]]
    split_counts: dict[str, int]
    modelscan_status: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def _validate_safetensors_header(weight_path: Path) -> dict[str, Any]:
    """Validates a `.safetensors` header JSON without executing or deserializing tensors."""
    size = weight_path.stat().st_size
    if size < 8:
        raise SupplyChainGateError(f"Truncated .safetensors file (<8 bytes): {weight_path}")
    with weight_path.open("rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        if header_len == 0 or header_len > MAX_SAFETENSORS_HEADER_BYTES or 8 + header_len > size:
            raise SupplyChainGateError(
                f"Invalid .safetensors header length ({header_len}) in {weight_path}"
            )
        header_bytes = fh.read(header_len)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupplyChainGateError(
            f"Malformed .safetensors JSON header in {weight_path}: {exc}"
        ) from exc
    if not isinstance(header, dict):
        raise SupplyChainGateError(f".safetensors header must be a JSON object: {weight_path}")
    return header


def _run_modelscan(target_dir: Path) -> str:
    """Runs `modelscan` CLI or Python library if available; enforces .safetensors-only."""
    cli = shutil.which("modelscan")
    if cli:
        proc = subprocess.run(
            [cli, "-p", str(target_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode not in (0, 3):
            raise SupplyChainGateError(
                f"modelscan CLI failed for {target_dir} (rc={proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        if "issues found" in proc.stdout.lower() and "no issues found" not in proc.stdout.lower():
            raise SupplyChainGateError(f"modelscan detected unsafe artifacts in {target_dir}")
        return "modelscan-cli-passed"

    if importlib.util.find_spec("modelscan") is not None:
        from modelscan.modelscan import ModelScan

        scanner = ModelScan()
        results = scanner.scan(target_dir)
        issues = results.get("issues") or []
        if issues:
            raise SupplyChainGateError(
                f"modelscan detected {len(issues)} issue(s) in {target_dir}: {issues}"
            )
        return "modelscan-lib-passed"

    return "safetensors-header-verified"


def _parse_python_dependencies(pyproject_path: Path) -> list[dict[str, Any]]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project") or {}
    raw_deps: list[tuple[str, str]] = []
    for dep in project.get("dependencies") or []:
        raw_deps.append(("required", str(dep)))
    for group, deps in (project.get("optional-dependencies") or {}).items():
        for dep in deps or []:
            raw_deps.append((f"optional:{group}", str(dep)))

    seen: set[str] = set()
    components: list[dict[str, Any]] = []
    for scope, spec in raw_deps:
        spec_no_marker = spec.split(";", 1)[0].strip()
        match = _PEP508_SPLIT_RE.match(spec_no_marker)
        if not match:
            continue
        raw_name, constraint = match.group(1), match.group(2).strip()
        pkg_name = raw_name.split("[", 1)[0].lower()
        if pkg_name in seen:
            continue
        seen.add(pkg_name)
        components.append(
            {
                "type": "library",
                "bom-ref": f"pkg:pypi/{pkg_name}",
                "name": pkg_name,
                "version": constraint or "unspecified",
                "purl": f"pkg:pypi/{pkg_name}",
                "scope": "required" if scope == "required" else "optional",
                "properties": [{"name": "scout:dependency_group", "value": scope}],
            }
        )
    return components


def _parse_frontend_dependencies(package_json_path: Path) -> list[dict[str, Any]]:
    if not package_json_path.is_file():
        return []
    data = json.loads(package_json_path.read_text(encoding="utf-8"))
    components: list[dict[str, Any]] = []
    for section, scope in (("dependencies", "required"), ("devDependencies", "optional")):
        for name, version in sorted((data.get(section) or {}).items()):
            components.append(
                {
                    "type": "library",
                    "bom-ref": f"pkg:npm/{name}@{version}",
                    "name": name,
                    "version": str(version),
                    "purl": f"pkg:npm/{name}",
                    "scope": scope,
                    "properties": [{"name": "scout:ecosystem", "value": "npm"}],
                }
            )
    return components


def _collect_governed_models(
    registry_path: Path,
    policy: ModelGovernancePolicy,
) -> list[dict[str, Any]]:
    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    components: list[dict[str, Any]] = []

    for category, entries in (raw.get("local_defaults") or {}).items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("id"):
                continue
            model_id = policy.validate_local_model_id(str(entry["id"]))
            components.append(
                {
                    "type": "machine-learning-model",
                    "bom-ref": f"model:{model_id}",
                    "name": model_id,
                    "version": str(entry.get("tier") or "default"),
                    "modelCard": {
                        "modelParameters": {
                            "task": str(category),
                            "architectureFamily": str(entry.get("family") or "unknown"),
                        }
                    },
                    "properties": [
                        {"name": "scout:model_tier", "value": "local_default"},
                        {"name": "scout:trust_remote_code", "value": str(policy.trust_remote_code)},
                        {
                            "name": "scout:require_safetensors_only",
                            "value": str(policy.require_safetensors_only),
                        },
                    ],
                }
            )

    for category, groups in (raw.get("byo_open_weights") or {}).items():
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict) or not group.get("family"):
                continue
            family = str(group["family"])
            patterns = [str(p) for p in (group.get("patterns") or [])]
            components.append(
                {
                    "type": "machine-learning-model",
                    "bom-ref": f"byo-family:{family}",
                    "name": family,
                    "version": "allowlisted-byo-family",
                    "properties": [
                        {"name": "scout:model_tier", "value": "byo_open_weights"},
                        {"name": "scout:category", "value": str(category)},
                        {"name": "scout:patterns", "value": ",".join(patterns)},
                    ],
                }
            )

    return components


def _verify_tuning_disjointness(domain_dir: Path) -> dict[str, int]:
    tuning_dir = domain_dir / "tuning"
    split_hashes: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for split_name in ("train", "valid", "eval"):
        split_path = tuning_dir / f"{split_name}.jsonl"
        if not split_path.is_file():
            split_hashes[split_name] = set()
            counts[split_name] = 0
            continue
        hashes: set[str] = set()
        line_count = 0
        for raw_line in split_path.read_text(encoding="utf-8").splitlines():
            cleaned = raw_line.strip()
            if not cleaned:
                continue
            line_count += 1
            hashes.add(hashlib.sha256(cleaned.encode("utf-8")).hexdigest())
        split_hashes[split_name] = hashes
        counts[split_name] = line_count

    for left, right in (("train", "valid"), ("train", "eval"), ("valid", "eval")):
        overlap = split_hashes[left] & split_hashes[right]
        if overlap:
            raise SupplyChainGateError(
                f"Domain '{domain_dir.name}' tuning splits '{left}' and '{right}' "
                f"share {len(overlap)} overlapping SHA-256 record(s)."
            )
    return counts


def _audit_domain_cartridge(
    domain_dir: Path,
    policy: ModelGovernancePolicy,
) -> CartridgeAuditResult | None:
    domain_yaml_path = domain_dir / "domain.yaml"
    spec_yaml_path = domain_dir / "specialization.yaml"
    if not domain_yaml_path.is_file() or not spec_yaml_path.is_file():
        return None

    domain_cfg = yaml.safe_load(domain_yaml_path.read_text(encoding="utf-8")) or {}
    spec_cfg = yaml.safe_load(spec_yaml_path.read_text(encoding="utf-8")) or {}
    domain_id = str(domain_cfg.get("domain_id") or domain_dir.name).strip()

    base_model_id = str(
        spec_cfg.get("base_model_id") or domain_cfg.get("base_model_id") or ""
    ).strip()
    policy.validate_local_model_id(base_model_id)

    spdx_license = str(spec_cfg.get("license") or domain_cfg.get("license") or "Apache-2.0").strip()
    policy.validate_spdx_license(spdx_license)

    scope_ceiling = [
        str(s).strip() for s in (spec_cfg.get("training_scope_ceiling") or []) if str(s).strip()
    ]
    if not scope_ceiling:
        raise SupplyChainGateError(
            f"Domain '{domain_id}' specialization.yaml has empty training_scope_ceiling."
        )

    lora_policy = spec_cfg.get("lora_policy") or {}
    use_rslora = bool(lora_policy.get("use_rslora", False))
    use_dora = bool(lora_policy.get("use_dora", False))
    rank = int(lora_policy.get("rank") or 0)
    if not use_rslora:
        raise SupplyChainGateError(f"Domain '{domain_id}' must enforce use_rslora=True.")
    if use_dora:
        raise SupplyChainGateError(f"Domain '{domain_id}' must enforce use_dora=False.")
    if rank <= 0 or rank > 32:
        raise SupplyChainGateError(
            f"Domain '{domain_id}' LoRA rank={rank} violates 1 <= rank <= 32."
        )

    split_counts = _verify_tuning_disjointness(domain_dir)

    adapters_root = domain_dir / "adapters"
    adapter_id = str(domain_cfg.get("lora_adapter_id") or domain_id).strip()
    safetensors_records: list[dict[str, Any]] = []
    scan_status = "no-adapter-dir"

    if adapters_root.is_dir():
        all_files = sorted(p for p in adapters_root.rglob("*") if p.is_file())
        weight_candidates = [
            p
            for p in all_files
            if p.suffix.lower() in UNSAFE_WEIGHT_EXTENSIONS or p.suffix.lower() == ".safetensors"
        ]
        config_payload: dict[str, Any] = {}
        for cfg_name in ("adapter_config.json", "config.json"):
            for cfg_path in adapters_root.rglob(cfg_name):
                loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    config_payload.update(loaded)
                    if loaded.get("trust_remote_code") is True or "auto_map" in loaded:
                        raise SupplyChainGateError(
                            f"Prohibited trust_remote_code/auto_map in {cfg_path}"
                        )
                    cfg_rank = loaded.get("r")
                    if cfg_rank is not None and int(cfg_rank) > 32:
                        raise SupplyChainGateError(f"Adapter rank r={cfg_rank} > 32 in {cfg_path}")

        policy.validate_weights_safety(
            [str(p) for p in weight_candidates],
            trust_remote_code=bool(config_payload.get("trust_remote_code", False)),
            config_payload=config_payload,
        )
        scan_status = _run_modelscan(adapters_root)

        for st_path in sorted(adapters_root.rglob("*.safetensors")):
            header = _validate_safetensors_header(st_path)
            safetensors_records.append(
                {
                    "path": st_path.relative_to(domain_dir.parent.parent).as_posix(),
                    "sha256": _sha256_file(st_path),
                    "size_bytes": st_path.stat().st_size,
                    "tensor_count": len([k for k in header if k != "__metadata__"]),
                }
            )

    return CartridgeAuditResult(
        domain_id=domain_id,
        adapter_id=adapter_id,
        base_model_id=base_model_id,
        spdx_license=spdx_license,
        rank=rank,
        use_rslora=use_rslora,
        use_dora=use_dora,
        training_scope_ceiling=scope_ceiling,
        safetensors_files=safetensors_records,
        split_counts=split_counts,
        modelscan_status=scan_status,
    )


def build_aibom(repo_root: Path) -> dict[str, Any]:
    """Audits all dependencies, models, and cartridges and builds CycloneDX v1.6 JSON."""
    policy = ModelGovernancePolicy.from_registry_file(repo_root / "config" / "model_registry.yaml")
    if policy.trust_remote_code:
        raise SupplyChainGateError(
            "Global ModelGovernancePolicy must enforce trust_remote_code=False."
        )

    pyproject_path = repo_root / "pyproject.toml"
    pyproject_data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project_meta = pyproject_data.get("project") or {}
    project_version = str(project_meta.get("version") or "0.1.0")
    project_license = str((project_meta.get("license") or {}).get("text") or "Apache-2.0")
    policy.validate_spdx_license(project_license)

    components: list[dict[str, Any]] = []
    components.extend(_parse_python_dependencies(pyproject_path))
    components.extend(_parse_frontend_dependencies(repo_root / "web-ui" / "package.json"))
    components.extend(
        _collect_governed_models(repo_root / "config" / "model_registry.yaml", policy)
    )

    cartridges: list[CartridgeAuditResult] = []
    domains_dir = repo_root / "domains"
    if domains_dir.is_dir():
        for domain_dir in sorted(p for p in domains_dir.iterdir() if p.is_dir()):
            audit = _audit_domain_cartridge(domain_dir, policy)
            if audit is None:
                continue
            cartridges.append(audit)
            components.append(
                {
                    "type": "machine-learning-model",
                    "bom-ref": f"cartridge:{audit.domain_id}:{audit.adapter_id}",
                    "name": f"{audit.domain_id}/{audit.adapter_id}",
                    "version": f"rslora-r{audit.rank}",
                    "licenses": [{"license": {"id": audit.spdx_license}}],
                    "hashes": [
                        {"alg": "SHA-256", "content": sf["sha256"]}
                        for sf in audit.safetensors_files
                    ],
                    "properties": [
                        {"name": "scout:base_model_id", "value": audit.base_model_id},
                        {"name": "scout:use_rslora", "value": str(audit.use_rslora)},
                        {"name": "scout:use_dora", "value": str(audit.use_dora)},
                        {"name": "scout:lora_rank", "value": str(audit.rank)},
                        {
                            "name": "scout:training_scope_ceiling",
                            "value": ",".join(audit.training_scope_ceiling),
                        },
                        {"name": "scout:modelscan_status", "value": audit.modelscan_status},
                        {
                            "name": "scout:tuning_splits",
                            "value": json.dumps(audit.split_counts, sort_keys=True),
                        },
                    ],
                }
            )

    serial_uuid = uuid5(NAMESPACE_URL, f"enterprise-context-scout:{project_version}")
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": f"urn:uuid:{serial_uuid}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "component": {
                "type": "application",
                "bom-ref": "enterprise-context-scout",
                "name": str(project_meta.get("name") or "enterprise-context-scout"),
                "version": project_version,
                "licenses": [{"license": {"id": project_license}}],
            },
            "properties": [
                {"name": "scout:trust_remote_code", "value": "false"},
                {"name": "scout:require_safetensors_only", "value": "true"},
                {"name": "scout:cartridges_audited", "value": str(len(cartridges))},
            ],
        },
        "components": components,
    }


def sign_and_verify_aibom(
    bom_bytes: bytes,
    output_path: Path,
    *,
    signing_key_path: Path | None = None,
) -> Path:
    """Signs `bom_bytes` as a Sigstore-compatible bundle and verifies it fail-closed."""
    if signing_key_path is not None:
        raw_key = signing_key_path.read_bytes()
        private_key = serialization.load_pem_private_key(raw_key, password=None)
    else:
        private_key = ec.generate_private_key(ec.SECP256R1())

    pub_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    bundle_json = sign_blob_for_cosign(private_key, bom_bytes, as_sigstore_bundle=True)
    bundle_path = Path(f"{output_path}.sigstore.json")
    bundle_path.write_text(bundle_json + "\n", encoding="utf-8")

    verifier = CosignBlobVerifier(trusted_public_keys_pem={"aibom-key": pub_pem})
    verifier.verify(bom_bytes, bundle_json, key_id="aibom-key")
    return bundle_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate CycloneDX v1.6 SBOM & AI-BOM with supply-chain verification."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("dist/aibom.cdx.json"))
    parser.add_argument("--signing-key", type=Path, default=None)
    parser.add_argument("--no-sign", action="store_true")
    args = parser.parse_args(argv)

    bom = build_aibom(args.repo_root.resolve())
    bom_bytes = (json.dumps(bom, indent=2, sort_keys=True) + "\n").encode("utf-8")

    if str(args.output) == "-":
        sys.stdout.buffer.write(bom_bytes)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(bom_bytes)

    sig_info = "unsigned"
    if not args.no_sign:
        bundle_path = sign_and_verify_aibom(
            bom_bytes, args.output, signing_key_path=args.signing_key
        )
        sig_info = f"signed+verified:{bundle_path}"

    print(
        f"Generated CycloneDX {bom['specVersion']} AI-BOM: {args.output} "
        f"(components={len(bom['components'])}, {sig_info})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
