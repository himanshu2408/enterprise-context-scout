"""M6-01 verification: containers, Helm chart, Terraform, airgap bundle, and AI-BOM."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from context_scout.ingestion.bundle_importer import AirGappedBundleImporter
from context_scout.plugins.languages.terraform import TerraformPlugin
from context_scout.serving.adapter_sync_sidecar import (
    CosignBlobVerifier,
    CosignVerificationError,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_generate_aibom_module() -> Any:
    script_path = REPO_ROOT / "scripts" / "generate_aibom.py"
    spec = importlib.util.spec_from_file_location("generate_aibom", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_aibom"] = module
    spec.loader.exec_module(module)
    return module


generate_aibom = _load_generate_aibom_module()

# ---------------------------------------------------------------------------
# 1. Distroless Dockerfiles, nginx-airgap.conf, and docker-compose.enterprise.yml
# ---------------------------------------------------------------------------


def test_distroless_dockerfiles_and_nginx_airgap_hardening() -> None:
    docker_dir = REPO_ROOT / "deploy" / "docker"
    required_files = [
        docker_dir / "Dockerfile.api",
        docker_dir / "Dockerfile.worker",
        docker_dir / "Dockerfile.adapter-sidecar",
        docker_dir / "Dockerfile.webui",
        docker_dir / "nginx-airgap.conf",
        docker_dir / "25-scout-resolver.sh",
        docker_dir / ".dockerignore",
    ]
    for path in required_files:
        assert path.is_file(), f"Missing container artifact: {path}"

    for py_dockerfile in (
        docker_dir / "Dockerfile.api",
        docker_dir / "Dockerfile.worker",
        docker_dir / "Dockerfile.adapter-sidecar",
    ):
        text = py_dockerfile.read_text(encoding="utf-8")
        assert "ARG PYTHON_BUILDER_IMAGE=python:3.12-slim-bookworm" in text
        assert "ARG DISTROLESS_PYTHON_IMAGE=gcr.io/distroless/python3-debian12:nonroot" in text
        assert "FROM ${PYTHON_BUILDER_IMAGE} AS builder" in text
        assert "FROM ${DISTROLESS_PYTHON_IMAGE} AS runtime" in text
        assert "COPY --from=builder /usr/local/bin/python3.12 /usr/bin/python3" in text
        assert "libpython3.12" in text
        assert "LD_LIBRARY_PATH=/usr/local/lib" in text
        assert "USER 65532:65532" in text
        assert "PYTHONDONTWRITEBYTECODE=1" in text
        assert "PYTHONUNBUFFERED=1" in text
        assert "TMPDIR=/tmp" in text

    api_text = (docker_dir / "Dockerfile.api").read_text(encoding="utf-8")
    assert "--extra api" in api_text
    assert "EXPOSE 8080" in api_text
    assert "context_scout.api.main:app" in api_text
    assert "--no-server-header" in api_text
    assert "alembic.ini" in api_text
    assert "domains /app/domains" in api_text

    worker_text = (docker_dir / "Dockerfile.worker").read_text(encoding="utf-8")
    assert "--extra worker" in worker_text
    assert "HF_HUB_OFFLINE=1" in worker_text
    assert "TRANSFORMERS_OFFLINE=1" in worker_text
    assert "HF_HUB_DISABLE_TELEMETRY=1" in worker_text
    assert "context_scout.cli" in worker_text
    assert "alembic.ini" in worker_text
    assert "domains /app/domains" in worker_text

    sidecar_text = (docker_dir / "Dockerfile.adapter-sidecar").read_text(encoding="utf-8")
    assert "SCOUT_ADAPTER_STAGING_DIR=/var/run/scout/adapters" in sidecar_text
    assert "chown -R 65532:65532 /var/run/scout/adapters" in sidecar_text
    assert "context_scout.serving.adapter_sync_sidecar" in sidecar_text

    webui_text = (docker_dir / "Dockerfile.webui").read_text(encoding="utf-8")
    assert "ARG NODE_BUILDER_IMAGE=node:22-bookworm-slim" in webui_text
    assert "FROM ${NODE_BUILDER_IMAGE} AS builder" in webui_text
    assert "npm ci --ignore-scripts" in webui_text
    assert "npm run build" in webui_text
    assert "USER 65532:65532" in webui_text
    assert "8080" in webui_text
    assert "/docker-entrypoint.d/25-scout-resolver.sh" in webui_text

    nginx_conf = (docker_dir / "nginx-airgap.conf").read_text(encoding="utf-8")
    assert "listen 8080" in nginx_conf
    assert "/tmp/" in nginx_conf
    assert "Content-Security-Policy" in nginx_conf
    assert "default-src 'self'" in nginx_conf
    assert "connect-src 'self'" in nginx_conf
    assert "frame-ancestors 'none'" in nginx_conf
    assert "include /tmp/nginx-scout-resolver*.conf;" in nginx_conf
    assert "proxy_pass $scout_api_upstream;" in nginx_conf
    assert "proxy_pass $scout_api_upstream/;" not in nginx_conf
    assert "proxy_pass http://context-scout-api" not in nginx_conf
    assert 'proxy_set_header Connection "";' in nginx_conf
    assert "proxy_set_header Host $http_host;" in nginx_conf
    for expected_loc in (
        "location /api/",
        "location /mcp/",
        "location = /.well-known/mcp",
        "location = /.well-known/oauth-protected-resource",
    ):
        assert expected_loc in nginx_conf

    dockerignore = (docker_dir / ".dockerignore").read_text(encoding="utf-8")
    for excluded in (".git", ".venv", "node_modules", ".env", "examples/quant_dogfood"):
        assert excluded in dockerignore


def test_docker_compose_enterprise_security_and_isolation() -> None:
    compose_path = REPO_ROOT / "docker-compose.enterprise.yml"
    assert compose_path.is_file()
    raw_text = compose_path.read_text(encoding="utf-8")

    # Never pass admin DB URL or hardcoded weak dev passwords in compose.
    assert "SCOUT_DB_ADMIN_URL" not in raw_text
    assert "scout_local_dev_pw" not in raw_text

    data = yaml.safe_load(raw_text)
    assert isinstance(data, dict)
    services = data.get("services", {})
    expected_services = {
        "postgres",
        "db-migrate",
        "redis",
        "zoekt",
        "minio",
        "vllm",
        "adapter-sync-sidecar",
        "context-scout-api",
        "context-scout-worker",
        "context-scout-webui",
    }
    assert expected_services.issubset(set(services.keys()))

    assert "pgvector/pgvector:pg17" in str(services["postgres"]["image"])
    assert "redis:7.4-alpine" in str(services["redis"]["image"])
    assert "vllm/vllm-openai:v0.34.0" in str(services["vllm"]["image"])
    vllm_cmd = " ".join(str(x) for x in services["vllm"].get("command", []))
    assert "--guided-decoding-backend" not in vllm_cmd
    for flag in (
        "--enable-lora",
        "--max-lora-rank",
        "--structured-outputs-config.backend",
        "xgrammar",
        "--enable-prefix-caching",
    ):
        assert flag in vllm_cmd

    runtime_services = (
        "adapter-sync-sidecar",
        "context-scout-api",
        "context-scout-worker",
        "context-scout-webui",
    )
    for svc_name in runtime_services:
        svc = services[svc_name]
        assert str(svc.get("user")) == "65532:65532", f"{svc_name} must run as 65532:65532"
        assert svc.get("read_only") is True, f"{svc_name} must set read_only: true"
        assert "no-new-privileges:true" in (svc.get("security_opt") or [])
        assert "ALL" in (svc.get("cap_drop") or [])
        assert svc.get("tmpfs"), f"{svc_name} must mount writable /tmp tmpfs"

    for db_consumer in ("adapter-sync-sidecar", "context-scout-api", "context-scout-worker"):
        env = services[db_consumer].get("environment", {})
        assert "SCOUT_DB_APP_URL" in env
        assert "SCOUT_DB_APP_URL" in str(env["SCOUT_DB_APP_URL"])
        assert "SCOUT_DB_ADMIN_URL" not in env
        assert str(env.get("SCOUT_ALLOW_EXTERNAL_CLOUD_EGRESS")).lower() == "false"

    networks = data.get("networks", {})
    assert "scout_data_plane" in networks
    assert networks["scout_data_plane"].get("internal") is True

    docker_bin = shutil.which("docker")
    if docker_bin:
        env = os.environ.copy()
        env.update(
            {
                "POSTGRES_ADMIN_PASSWORD": "test_admin_pw_ci",
                "POSTGRES_APP_PASSWORD": "test_app_pw_ci",
                "REDIS_PASSWORD": "test_redis_pw_ci",
                "SCOUT_AUTH_HMAC_SECRET": "test_hmac_secret_ci",
                "MINIO_ROOT_USER": "test_minio_user",
                "MINIO_ROOT_PASSWORD": "test_minio_pw_ci",
            }
        )
        proc = subprocess.run(
            [docker_bin, "compose", "-f", str(compose_path), "config", "--quiet"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"docker compose config failed: {proc.stderr}"


# ---------------------------------------------------------------------------
# 2. Helm Chart (values.yaml + values-gcp-gke.yaml + templates)
# ---------------------------------------------------------------------------


def _count_template_blocks(text: str) -> tuple[int, int]:
    """Counts balanced Go template open/close tags and block/end keywords."""
    opens = len(re.findall(r"\{\{-?", text))
    closes = len(re.findall(r"-?\}\}", text))
    assert opens == closes, f"Unbalanced template braces: {opens} != {closes}"

    block_open = len(re.findall(r"\{\{-?\s*(?:if|range|define|with)\b", text))
    block_end = len(re.findall(r"\{\{-?\s*end\b", text))
    return block_open, block_end


def test_helm_chart_values_templates_and_security_invariants() -> None:
    chart_dir = REPO_ROOT / "deploy" / "helm" / "context-scout"
    assert (chart_dir / "Chart.yaml").is_file()
    chart_meta = yaml.safe_load((chart_dir / "Chart.yaml").read_text(encoding="utf-8"))
    assert chart_meta["name"] == "context-scout"

    base_values = yaml.safe_load((chart_dir / "values.yaml").read_text(encoding="utf-8"))
    gke_values = yaml.safe_load((chart_dir / "values-gcp-gke.yaml").read_text(encoding="utf-8"))

    # Verify GKE Workload Identity & Gateway/IAP & Cloud SQL Auth Proxy in values-gcp-gke.yaml
    assert gke_values["serviceAccount"]["workloadIdentity"]["provider"] == "gcp"
    assert gke_values["serviceAccount"]["workloadIdentity"]["gcp"]["api"]
    assert gke_values["cloudSqlAuthProxy"]["enabled"] is True
    assert gke_values["gateway"]["enabled"] is True
    assert gke_values["gateway"]["iap"]["enabled"] is True

    # Verify NVIDIA L4/A100/H100 & AMD ROCm support in base values
    assert base_values["vllm"]["gpu"]["vendor"] in ("nvidia", "amd")
    assert "adapterSyncSidecar" in base_values["vllm"]
    assert base_values["vllm"]["adapterSyncSidecar"]["baseModelRevisionSha256"] == ""
    for secret_key in (
        "cosignPublicKeysJsonKey",
        "cosignPublicKeyPemKey",
        "cosignCaCertsPemKey",
    ):
        assert secret_key in base_values["secrets"]
    assert gke_values["config"]["objectStoreScheme"] == "gs"

    templates_dir = chart_dir / "templates"
    required_templates = [
        "_helpers.tpl",
        "serviceaccount.yaml",
        "configmap.yaml",
        "api-deployment.yaml",
        "api-service.yaml",
        "worker-deployment.yaml",
        "webui-deployment.yaml",
        "webui-service.yaml",
        "vllm-statefulset.yaml",
        "vllm-service.yaml",
        "zoekt-statefulset.yaml",
        "zoekt-service.yaml",
        "gateway-iap.yaml",
        "keda-scaledobjects.yaml",
        "networkpolicy.yaml",
        "pdb.yaml",
    ]
    for tpl_name in required_templates:
        tpl_path = templates_dir / tpl_name
        assert tpl_path.is_file(), f"Missing Helm template: {tpl_name}"
        tpl_text = tpl_path.read_text(encoding="utf-8")
        assert "SCOUT_DB_ADMIN_URL" not in tpl_text
        block_open, block_end = _count_template_blocks(tpl_text)
        assert block_open == block_end, f"Unbalanced if/range/define/with/end in {tpl_name}"

    helpers_text = (templates_dir / "_helpers.tpl").read_text(encoding="utf-8")
    sa_text = (templates_dir / "serviceaccount.yaml").read_text(encoding="utf-8")
    assert "iam.gke.io/gcp-service-account" in helpers_text
    assert "workloadIdentityAnnotations" in sa_text

    gateway_text = (templates_dir / "gateway-iap.yaml").read_text(encoding="utf-8")
    assert "GCPBackendPolicy" in gateway_text
    assert "iap:" in gateway_text
    assert "/.well-known/oauth-protected-resource" in gateway_text
    assert "/.well-known/oauth-protected-resource" in base_values["ingress"]["hosts"][0]["apiPaths"]

    vllm_text = (templates_dir / "vllm-statefulset.yaml").read_text(encoding="utf-8")
    assert "--structured-outputs-config.backend" in vllm_text
    assert "--guided-decoding-backend" not in vllm_text
    assert "nvidia.com/gpu" in vllm_text
    assert "amd.com/gpu" in vllm_text
    assert "adapter-sync-sidecar" in vllm_text
    assert "gs://" in vllm_text or "s3://" in vllm_text or "SCOUT_ADAPTER_STAGING_DIR" in vllm_text
    for env_name in (
        "SCOUT_BASE_MODEL_REVISION_SHA256",
        "SCOUT_COSIGN_PUBLIC_KEYS_JSON",
        "SCOUT_COSIGN_PUBLIC_KEY_PEM",
        "SCOUT_COSIGN_CA_CERTS_PEM",
    ):
        assert env_name in vllm_text

    zoekt_text = (templates_dir / "zoekt-statefulset.yaml").read_text(encoding="utf-8")
    assert "volumeClaimTemplates" in zoekt_text

    keda_text = (templates_dir / "keda-scaledobjects.yaml").read_text(encoding="utf-8")
    assert "kind: ScaledObject" in keda_text

    netpol_text = (templates_dir / "networkpolicy.yaml").read_text(encoding="utf-8")
    assert "kind: NetworkPolicy" in netpol_text

    pdb_text = (templates_dir / "pdb.yaml").read_text(encoding="utf-8")
    assert "kind: PodDisruptionBudget" in pdb_text

    # All workload templates must enforce restricted PodSecurity + Cloud SQL proxy on api/worker.
    assert base_values["podSecurityContext"]["runAsUser"] == 65532
    assert base_values["podSecurityContext"]["runAsNonRoot"] is True
    assert base_values["containerSecurityContext"]["readOnlyRootFilesystem"] is True
    assert base_values["containerSecurityContext"]["allowPrivilegeEscalation"] is False
    assert "ALL" in base_values["containerSecurityContext"]["capabilities"]["drop"]

    workload_templates = [
        "api-deployment.yaml",
        "worker-deployment.yaml",
        "webui-deployment.yaml",
        "vllm-statefulset.yaml",
        "zoekt-statefulset.yaml",
    ]
    for wl in workload_templates:
        wl_text = (templates_dir / wl).read_text(encoding="utf-8")
        assert ".Values.containerSecurityContext" in wl_text, (
            f"{wl} missing containerSecurityContext"
        )
        assert ".Values.podSecurityContext" in wl_text, f"{wl} missing podSecurityContext"
        if wl in ("api-deployment.yaml", "worker-deployment.yaml"):
            assert "cloudSqlAuthProxy.enabled" in wl_text, (
                f"{wl} missing cloudSqlAuthProxy.enabled toggle"
            )

    helm_bin = shutil.which("helm")
    if helm_bin:
        for values_file in ("values.yaml", "values-gcp-gke.yaml"):
            lint_proc = subprocess.run(
                [helm_bin, "lint", str(chart_dir), "-f", str(chart_dir / values_file)],
                capture_output=True,
                text=True,
                check=False,
            )
            assert lint_proc.returncode == 0, (
                f"helm lint failed ({values_file}): {lint_proc.stderr}"
            )
            tpl_proc = subprocess.run(
                [
                    helm_bin,
                    "template",
                    "scout-test",
                    str(chart_dir),
                    "-f",
                    str(chart_dir / values_file),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            assert tpl_proc.returncode == 0, (
                f"helm template failed ({values_file}): {tpl_proc.stderr}"
            )


# ---------------------------------------------------------------------------
# 3. Terraform Modules (gcp-gke, onprem-k8s, aws-eks, azure-aks)
# ---------------------------------------------------------------------------


def test_terraform_modules_ast_valid_and_no_static_sa_keys(tmp_path: Path) -> None:
    tf_root = REPO_ROOT / "deploy" / "terraform"
    modules = ("gcp-gke", "onprem-k8s", "aws-eks", "azure-aks")
    plugin = TerraformPlugin()

    for mod in modules:
        mod_dir = tf_root / mod
        assert mod_dir.is_dir(), f"Missing Terraform module: {mod_dir}"
        for fname in ("versions.tf", "variables.tf", "main.tf", "outputs.tf"):
            tf_file = mod_dir / fname
            assert tf_file.is_file(), f"Missing Terraform file: {tf_file}"
            raw = tf_file.read_text(encoding="utf-8")
            assert "google_service_account_key" not in raw, (
                f"Forbidden static google_service_account_key found in {tf_file}"
            )
            symbols, _edges = plugin.extract_ast(raw, file_path=str(tf_file))
            if fname in ("main.tf", "variables.tf", "outputs.tf"):
                assert len(symbols) > 0, f"Expected parsed HCL symbols in {tf_file}"

    gcp_main = (tf_root / "gcp-gke" / "main.tf").read_text(encoding="utf-8")
    required_gcp_resources = (
        "google_compute_network",
        "google_compute_subnetwork",
        "google_compute_global_address",
        "google_kms_key_ring",
        "google_kms_crypto_key",
        "google_container_cluster",
        "google_container_node_pool",
        "google_sql_database_instance",
        "google_sql_user",
        "google_redis_instance",
        "google_storage_bucket",
        "google_artifact_registry_repository",
        "google_compute_security_policy",
    )
    for res_type in required_gcp_resources:
        assert f'resource "{res_type}"' in gcp_main, f"gcp-gke/main.tf missing {res_type}"

    gcp_vars = (tf_root / "gcp-gke" / "variables.tf").read_text(encoding="utf-8")
    assert "configured-via-iap-secret" not in gcp_main
    assert "configured-via-secret-manager" not in gcp_main
    assert 'dynamic "iap"' in gcp_main
    assert "var.iap_oauth2_client_id" in gcp_main
    assert "var.iap_oauth2_client_secret" in gcp_main
    assert 'variable "iap_oauth2_client_id"' in gcp_vars
    assert 'variable "iap_oauth2_client_secret"' in gcp_vars
    assert "sensitive   = true" in gcp_vars or "sensitive = true" in gcp_vars

    tf_bin = shutil.which("terraform")
    if os.environ.get("CI", "").lower() == "true":
        assert tf_bin is not None, "terraform CLI must be installed in CI"
    if tf_bin:
        tf_env = os.environ.copy()
        tf_env.setdefault("TF_PLUGIN_CACHE_DIR", "/tmp/tf-plugin-cache")
        Path(tf_env["TF_PLUGIN_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        for mod in modules:
            fmt_proc = subprocess.run(
                [tf_bin, "fmt", "-check", str(tf_root / mod)],
                capture_output=True,
                text=True,
                check=False,
            )
            assert fmt_proc.returncode == 0, f"terraform fmt failed for {mod}: {fmt_proc.stdout}"
            mod_env = dict(tf_env)
            mod_env["TF_DATA_DIR"] = str(tmp_path / f"tf_{mod}")
            mod_copy = tmp_path / f"tf_src_{mod}"
            shutil.copytree(tf_root / mod, mod_copy)
            init_proc = subprocess.run(
                [tf_bin, f"-chdir={mod_copy}", "init", "-backend=false", "-input=false"],
                env=mod_env,
                capture_output=True,
                text=True,
                check=False,
            )
            assert init_proc.returncode == 0, (
                f"terraform init failed for {mod}: {init_proc.stderr or init_proc.stdout}"
            )
            val_proc = subprocess.run(
                [tf_bin, f"-chdir={mod_copy}", "validate"],
                env=mod_env,
                capture_output=True,
                text=True,
                check=False,
            )
            assert val_proc.returncode == 0, (
                f"terraform validate failed for {mod}: {val_proc.stderr or val_proc.stdout}"
            )


# ---------------------------------------------------------------------------
# 4. Air-Gap Bundle Script + AirGappedBundleImporter End-to-End
# ---------------------------------------------------------------------------


def test_airgap_bundle_script_e2e_and_negative_controls(tmp_path: Path) -> None:
    script = REPO_ROOT / "deploy" / "airgap" / "create_airgap_bundle.sh"
    assert script.is_file()
    assert os.access(script, os.X_OK), "create_airgap_bundle.sh must be executable"

    src_dir = tmp_path / "source_corpus"
    src_dir.mkdir()
    (src_dir / "service.py").write_text(
        "def verify_token(token: str) -> bool:\n    return bool(token)\n",
        encoding="utf-8",
    )
    (src_dir / "architecture.md").write_text(
        "# Air-Gapped Architecture\nAll workloads run offline inside VPC.\n",
        encoding="utf-8",
    )

    out_bundle = tmp_path / "out" / "airgap-corpus.tar.gz"
    proc = subprocess.run(
        [
            str(script),
            "--domain-id",
            "enterprise_docs_scout",
            "--bundle-id",
            "ci-airgap-001",
            str(src_dir),
            str(out_bundle),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"create_airgap_bundle.sh failed: {proc.stderr or proc.stdout}"
    assert out_bundle.is_file()
    sha_file = Path(f"{out_bundle}.sha256")
    assert sha_file.is_file()

    expected_sha = hashlib.sha256(out_bundle.read_bytes()).hexdigest()
    assert expected_sha in sha_file.read_text(encoding="utf-8")

    importer = AirGappedBundleImporter()
    report = importer.import_bundle(
        out_bundle,
        domain_id="enterprise_docs_scout",
        tenant_id=uuid4(),
        default_allowed_scopes=["public"],
    )
    assert report.bundle_id == "ci-airgap-001"
    assert len(report.repositories_imported) >= 1
    assert report.code_files_indexed >= 1
    assert report.doc_chunks_indexed >= 1

    # Negative control: reject unsafe pickle weights in source dir.
    bad_dir = tmp_path / "bad_corpus"
    bad_dir.mkdir()
    (bad_dir / "unsafe_model.pkl").write_bytes(b"pickle-payload")
    bad_proc = subprocess.run(
        [
            str(script),
            "--domain-id",
            "enterprise_docs_scout",
            str(bad_dir),
            str(tmp_path / "out" / "bad.tar.gz"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_proc.returncode != 0, "create_airgap_bundle.sh must reject .pkl files"


# ---------------------------------------------------------------------------
# 5. AI-BOM Generator (CycloneDX 1.6, modelscan, SPDX, cosign + negative controls)
# ---------------------------------------------------------------------------


def _clone_minimal_repo_for_aibom(dest: Path) -> Path:
    """Copies only the files needed by `generate_aibom.build_aibom` for fast mutation tests."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "pyproject.toml", dest / "pyproject.toml")
    if (REPO_ROOT / "uv.lock").is_file():
        shutil.copy2(REPO_ROOT / "uv.lock", dest / "uv.lock")
    (dest / "web-ui").mkdir(exist_ok=True)
    shutil.copy2(REPO_ROOT / "web-ui" / "package.json", dest / "web-ui" / "package.json")
    if (REPO_ROOT / "web-ui" / "package-lock.json").is_file():
        shutil.copy2(
            REPO_ROOT / "web-ui" / "package-lock.json", dest / "web-ui" / "package-lock.json"
        )
    (dest / "config").mkdir(exist_ok=True)
    shutil.copy2(
        REPO_ROOT / "config" / "model_registry.yaml",
        dest / "config" / "model_registry.yaml",
    )
    shutil.copytree(REPO_ROOT / "domains", dest / "domains")
    return dest


def test_aibom_generation_signing_and_negative_controls(tmp_path: Path) -> None:
    out_bom = tmp_path / "aibom.cdx.json"
    rc = generate_aibom.main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--output",
            str(out_bom),
        ]
    )
    assert rc == 0
    assert out_bom.is_file()
    bundle_file = Path(f"{out_bom}.sigstore.json")
    pub_file = Path(f"{out_bom}.pub")
    assert bundle_file.is_file()
    assert pub_file.is_file()

    bom = json.loads(out_bom.read_text(encoding="utf-8"))
    assert bom["bomFormat"] == "CycloneDX"
    assert bom["specVersion"] == "1.6"
    props = {p["name"]: p["value"] for p in bom["metadata"]["properties"]}
    assert props["scout:trust_remote_code"] == "false"
    assert props["scout:require_safetensors_only"] == "true"
    assert props["scout:cartridges_audited"] == "4"
    assert props["scout:signing_key_ephemeral"] == "true"
    assert props["scout:scanner_backend"] in {
        "modelscan-cli",
        "modelscan-lib",
        "native-safetensors-header-inspector",
    }
    assert len(bom["components"]) >= 50
    hashed_libs = [c for c in bom["components"] if c.get("type") == "library" and c.get("hashes")]
    assert len(hashed_libs) >= 20, "Lockfile SHA hashes must populate library components"

    # Cryptographically verify the ACTUAL emitted Sigstore bundle + public key.
    bom_bytes = out_bom.read_bytes()
    actual_bundle = bundle_file.read_text(encoding="utf-8")
    actual_pub_pem = pub_file.read_bytes()
    verifier = CosignBlobVerifier(trusted_public_keys_pem={"aibom": actual_pub_pem})
    verifier.verify(bom_bytes, actual_bundle, key_id="aibom")

    with pytest.raises(CosignVerificationError):
        verifier.verify(bom_bytes + b"tampered", actual_bundle, key_id="aibom")

    # Negative control 2: unsafe .pkl / .pt weights in a cartridge directory must fail closed.
    repo_pkl = _clone_minimal_repo_for_aibom(tmp_path / "repo_pkl")
    (repo_pkl / "domains" / "data_engineering_scout" / "adapters" / "evil.pkl").write_bytes(b"bad")
    with pytest.raises((generate_aibom.SupplyChainGateError, ValueError)):
        generate_aibom.build_aibom(repo_pkl)

    # Negative control 3: trust_remote_code=True in model_registry.yaml must fail closed.
    repo_trc = _clone_minimal_repo_for_aibom(tmp_path / "repo_trc")
    reg_path = repo_trc / "config" / "model_registry.yaml"
    reg_data: dict[str, Any] = yaml.safe_load(reg_path.read_text(encoding="utf-8"))
    reg_data["governance"]["trust_remote_code"] = True
    reg_path.write_text(yaml.safe_dump(reg_data), encoding="utf-8")
    with pytest.raises((generate_aibom.SupplyChainGateError, ValueError)):
        generate_aibom.build_aibom(repo_trc)

    # Negative control 4: unapproved SPDX license in domain specialization.yaml must fail closed.
    repo_spdx = _clone_minimal_repo_for_aibom(tmp_path / "repo_spdx")
    spec_spdx_path = repo_spdx / "domains" / "data_engineering_scout" / "specialization.yaml"
    spec_spdx: dict[str, Any] = yaml.safe_load(spec_spdx_path.read_text(encoding="utf-8"))
    spec_spdx["license"] = "GPL-3.0-only"
    spec_spdx_path.write_text(yaml.safe_dump(spec_spdx), encoding="utf-8")
    with pytest.raises((generate_aibom.SupplyChainGateError, ValueError)):
        generate_aibom.build_aibom(repo_spdx)


# ---------------------------------------------------------------------------
# 6. Keyless GCP / Multi-Cloud TF-Helm-Compose-.env.example Parity & Entrypoints
# ---------------------------------------------------------------------------


def test_gcp_gke_terraform_helm_parity() -> None:
    """Verifies gcp-gke Terraform aligns with Helm values-gcp-gke.yaml and API runtime."""
    chart_dir = REPO_ROOT / "deploy" / "helm" / "context-scout"
    gke_values = yaml.safe_load((chart_dir / "values-gcp-gke.yaml").read_text(encoding="utf-8"))
    tf_dir = REPO_ROOT / "deploy" / "terraform" / "gcp-gke"
    gcp_main = (tf_dir / "main.tf").read_text(encoding="utf-8")
    gcp_vars = (tf_dir / "variables.tf").read_text(encoding="utf-8")
    gcp_outputs = (tf_dir / "outputs.tf").read_text(encoding="utf-8")

    plugin = TerraformPlugin()
    main_syms, _ = plugin.extract_ast(gcp_main, file_path=str(tf_dir / "main.tf"))
    assert len(main_syms) > 0

    # All 3 Helm GCP Workload Identity GSAs must exist and bind to their K8s SAs.
    gcp_wi = gke_values["serviceAccount"]["workloadIdentity"]["gcp"]
    for component in ("api", "worker", "vllm"):
        expected_gsa = gcp_wi[component].split("@", 1)[0]
        assert f'account_id   = "{expected_gsa}"' in gcp_main
        assert f"/context-scout-{component}]" in gcp_main
        assert f'output "scout_{component}_service_account_email"' in gcp_outputs

    assert "roles/iam.workloadIdentityUser" in gcp_main
    assert 'variable "enable_vertex_ai"' in gcp_vars
    assert gke_values["config"]["useVertexAi"] == "true"
    assert "roles/aiplatform.user" in gcp_main
    assert "roles/storage.objectAdmin" in gcp_main
    assert "roles/storage.objectViewer" in gcp_main
    assert "google_service_account_key" not in gcp_main

    # All 3 workloads (api, worker, vllm) must have Cloud SQL IAM + CLOUD_IAM_SERVICE_ACCOUNT user.
    for role_name in ("scout_api", "scout_worker", "scout_vllm"):
        assert f'resource "google_project_iam_member" "{role_name}_sql_client"' in gcp_main
        assert f'resource "google_project_iam_member" "{role_name}_sql_user"' in gcp_main
        assert f'resource "google_sql_user" "{role_name}_db_user"' in gcp_main
        assert f"google_service_account.{role_name}.email" in gcp_main

    # Cloud Armor policy name and Cloud SQL instance name must match TF cluster_name default.
    cluster_match = re.search(
        r'variable\s+"cluster_name"\s*\{[^}]*default\s*=\s*"([^"]+)"',
        gcp_vars,
        re.DOTALL,
    )
    assert cluster_match is not None
    default_cluster = cluster_match.group(1)
    assert 'name        = "${var.cluster_name}-cloud-armor"' in gcp_main
    assert 'name                = "${var.cluster_name}-pg17"' in gcp_main
    assert gke_values["gateway"]["cloudArmor"]["securityPolicyName"] == (
        f"{default_cluster}-cloud-armor"
    )
    assert (
        gke_values["cloudSqlAuthProxy"]["instanceConnectionName"].rsplit(":", 1)[-1]
        == f"{default_cluster}-pg17"
    )

    # Health check & backend service must match API runtime HTTP :8080 /api/v1/health.
    api_tpl = (chart_dir / "templates" / "api-deployment.yaml").read_text(encoding="utf-8")
    dockerfile_api = (REPO_ROOT / "deploy" / "docker" / "Dockerfile.api").read_text(
        encoding="utf-8"
    )
    assert "path: /api/v1/health/live" in api_tpl
    assert "path: /api/v1/health/ready" in api_tpl
    assert "--port" in dockerfile_api and "8080" in dockerfile_api
    assert "http_health_check {" in gcp_main
    assert "https_health_check" not in gcp_main
    assert 'request_path = "/api/v1/health"' in gcp_main
    assert "port         = 8080" in gcp_main
    assert 'protocol        = "HTTP"' in gcp_main

    # vllm-statefulset must include Cloud SQL Auth Proxy parity when enabled.
    vllm_tpl = (chart_dir / "templates" / "vllm-statefulset.yaml").read_text(encoding="utf-8")
    assert "cloudSqlAuthProxy.enabled" in vllm_tpl


def test_multicloud_terraform_workload_identity_parity() -> None:
    """Verifies AWS EKS, Azure AKS, and On-Prem K8s bind api, worker, and vllm SAs."""
    tf_root = REPO_ROOT / "deploy" / "terraform"
    values = yaml.safe_load(
        (REPO_ROOT / "deploy" / "helm" / "context-scout" / "values.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert set(values["serviceAccount"]["workloadIdentity"]["aws"]) == {
        "apiRoleArn",
        "workerRoleArn",
        "vllmRoleArn",
    }
    assert set(values["serviceAccount"]["workloadIdentity"]["azure"]) == {
        "apiClientId",
        "workerClientId",
        "vllmClientId",
    }

    aws_main = (tf_root / "aws-eks" / "main.tf").read_text(encoding="utf-8")
    aws_outputs = (tf_root / "aws-eks" / "outputs.tf").read_text(encoding="utf-8")
    azure_main = (tf_root / "azure-aks" / "main.tf").read_text(encoding="utf-8")
    onprem_main = (tf_root / "onprem-k8s" / "main.tf").read_text(encoding="utf-8")
    onprem_outputs = (tf_root / "onprem-k8s" / "outputs.tf").read_text(encoding="utf-8")

    for sa_name in ("context-scout-api", "context-scout-worker", "context-scout-vllm"):
        assert sa_name in aws_main, f"aws-eks/main.tf missing {sa_name} IRSA binding"
        assert sa_name in azure_main, f"azure-aks/main.tf missing {sa_name} federated credential"
        assert sa_name in onprem_main, f"onprem-k8s/main.tf missing {sa_name} ServiceAccount"

    for out_name in (
        "scout_api_irsa_role_arn",
        "scout_worker_irsa_role_arn",
        "scout_vllm_irsa_role_arn",
    ):
        assert f'output "{out_name}"' in aws_outputs
    assert "vllm" in onprem_outputs


def test_docker_compose_enterprise_alembic_migration_and_rls_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies Compose db-migrate ordering and worker/sidecar entrypoints."""
    from urllib.parse import quote

    from context_scout.db.migrations_runner import (
        _resolve_app_role_password,
        build_admin_url_from_postgres_env,
    )
    from context_scout.serving.adapter_sync_sidecar import (
        AdapterVerificationError,
        build_sidecar_from_env,
        parse_vllm_endpoints_from_env,
    )

    compose_path = REPO_ROOT / "docker-compose.enterprise.yml"
    raw_text = compose_path.read_text(encoding="utf-8")
    assert "SCOUT_DB_ADMIN_URL" not in raw_text
    data = yaml.safe_load(raw_text)
    services = data["services"]

    db_migrate = services["db-migrate"]
    assert str(db_migrate.get("restart")) == "no"
    assert str(db_migrate.get("user")) == "65532:65532"
    assert db_migrate.get("read_only") is True
    assert "no-new-privileges:true" in (db_migrate.get("security_opt") or [])
    assert "ALL" in (db_migrate.get("cap_drop") or [])
    assert db_migrate.get("command") == ["db", "upgrade", "--from-postgres-env"]
    assert db_migrate["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert "POSTGRES_ADMIN_PASSWORD" in db_migrate["environment"]
    assert "SCOUT_DB_APP_PASSWORD" in db_migrate["environment"]

    for runtime_svc in ("adapter-sync-sidecar", "context-scout-api", "context-scout-worker"):
        svc = services[runtime_svc]
        assert svc["depends_on"]["db-migrate"]["condition"] == "service_completed_successfully"
        env = svc.get("environment", {})
        assert "POSTGRES_ADMIN_PASSWORD" not in env
        assert "SCOUT_DB_ADMIN_URL" not in env

    assert services["context-scout-worker"].get("command") == ["worker"]

    sidecar_env = services["adapter-sync-sidecar"]["environment"]
    assert "SCOUT_PRIMARY_MODEL_ID" in sidecar_env
    for fail_closed_var in (
        "SCOUT_BASE_MODEL_REVISION_SHA256",
        "SCOUT_COSIGN_PUBLIC_KEYS_JSON",
        "SCOUT_COSIGN_PUBLIC_KEY_PEM",
        "SCOUT_COSIGN_CA_CERTS_PEM",
    ):
        assert fail_closed_var in sidecar_env
        assert sidecar_env[fail_closed_var] == f"${{{fail_closed_var}:-}}"

    # Unit checks for build_admin_url_from_postgres_env (fail-closed + escaping).
    with pytest.raises(RuntimeError, match="POSTGRES_ADMIN_PASSWORD"):
        build_admin_url_from_postgres_env({})
    with pytest.raises(RuntimeError, match="POSTGRES_ADMIN_PASSWORD"):
        build_admin_url_from_postgres_env({"POSTGRES_ADMIN_PASSWORD": "   "})
    with pytest.raises(RuntimeError, match="POSTGRES_ADMIN_PASSWORD"):
        build_admin_url_from_postgres_env({"POSTGRES_ADMIN_PASSWORD": "   \t\n  "})

    dsn = build_admin_url_from_postgres_env(
        {
            "POSTGRES_HOST": "postgres",
            "POSTGRES_PORT": "5432",
            "POSTGRES_USER": "postgres",
            "POSTGRES_ADMIN_PASSWORD": "p@ss:w/rd#1",
            "POSTGRES_DB": "context_scout",
        }
    )
    assert dsn == "postgresql://postgres:p%40ss%3Aw%2Frd%231@postgres:5432/context_scout"

    from psycopg.conninfo import conninfo_to_dict
    from sqlalchemy.engine import make_url

    dsn_space = build_admin_url_from_postgres_env(
        {
            "POSTGRES_HOST": "postgres",
            "POSTGRES_PORT": "5432",
            "POSTGRES_USER": "postgres",
            "POSTGRES_ADMIN_PASSWORD": "pass word@1",
            "POSTGRES_DB": "context_scout",
        }
    )
    assert "pass%20word%401" in dsn_space
    assert "+" not in dsn_space
    assert conninfo_to_dict(dsn_space)["password"] == "pass word@1"

    lead_pw = "  lead space  "
    dsn_lead_space = build_admin_url_from_postgres_env(
        {
            "POSTGRES_HOST": "postgres",
            "POSTGRES_PORT": "5432",
            "POSTGRES_USER": "postgres",
            "POSTGRES_ADMIN_PASSWORD": lead_pw,
            "POSTGRES_DB": "context_scout",
        }
    )
    assert "%20%20lead%20space%20%20" in dsn_lead_space
    assert conninfo_to_dict(dsn_lead_space)["password"] == lead_pw
    assert make_url(dsn_lead_space).password == lead_pw

    # _resolve_app_role_password must preserve leading/trailing whitespace identically
    # to SCOUT_DB_APP_URL unquote(parsed.password) while ignoring whitespace-only values.
    monkeypatch.delenv("SCOUT_DB_APP_PASSWORD", raising=False)
    monkeypatch.delenv("SCOUT_DB_APP_URL", raising=False)
    assert _resolve_app_role_password(lead_pw) == lead_pw
    monkeypatch.setenv("SCOUT_DB_APP_PASSWORD", lead_pw)
    assert _resolve_app_role_password(None) == lead_pw
    assert _resolve_app_role_password("   \t\n  ") == lead_pw
    monkeypatch.setenv("SCOUT_DB_APP_PASSWORD", "   \t\n  ")
    assert _resolve_app_role_password(None) is None
    monkeypatch.setenv(
        "SCOUT_DB_APP_URL",
        f"postgresql://scout_app_user:{quote(lead_pw, safe='')}@postgres:5432/context_scout",
    )
    assert _resolve_app_role_password(None) == lead_pw

    # CLI worker subcommand and --from-postgres-env must parse cleanly.
    cli_proc = subprocess.run(
        [sys.executable, "-m", "context_scout.cli", "worker", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cli_proc.returncode == 0, cli_proc.stderr
    assert "--once" in cli_proc.stdout

    # AdapterSyncSidecar entrypoint must support both Helm & Compose env vars and fail closed.
    assert parse_vllm_endpoints_from_env({"SCOUT_VLLM_ENDPOINTS": "http://vllm:8000"}) == (
        "http://vllm:8000",
    )
    assert parse_vllm_endpoints_from_env(
        {"SCOUT_VLLM_ADMIN_ENDPOINT": "http://127.0.0.1:8001"}
    ) == ("http://127.0.0.1:8001",)
    with pytest.raises(AdapterVerificationError, match="SCOUT_DB_APP_URL"):
        build_sidecar_from_env({})
    sidecar_proc = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-m",
            "context_scout.serving.adapter_sync_sidecar",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert sidecar_proc.returncode == 0, sidecar_proc.stderr
    assert "RuntimeWarning" not in sidecar_proc.stderr

    # ECS-145 regression guard: sidecar MUST import cleanly when pruned ML/tree_sitter wheels
    # are absent from site-packages (Dockerfile.adapter-sidecar).
    pruned_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import sys",
                    "for m in (",
                    "    'tree_sitter', 'tree_sitter_hcl', 'tree_sitter_python',",
                    "    'tree_sitter_typescript', 'tree_sitter_c_sharp',",
                    "    'tree_sitter_language_pack', 'torch', 'transformers',",
                    "    'sentence_transformers', 'scipy', 'sklearn', 'onnx',",
                    "):",
                    "    sys.modules[m] = None",
                    "import context_scout.serving.adapter_sync_sidecar as s",
                    "assert hasattr(s, 'AdapterSyncSidecar')",
                ]
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert pruned_proc.returncode == 0, pruned_proc.stderr


@pytest.mark.postgres
def test_adapter_sync_sidecar_close_respects_rls_manager_ownership(tmp_path: Path) -> None:
    """AdapterSyncSidecar.close() closes owned RLSConnectionManager but leaves shared ones."""
    from context_scout.db.repositories import InMemoryRLSRepository
    from context_scout.serving.adapter_sync_sidecar import AdapterSyncSidecar

    class _FakeMgr:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    priv_key = ec.generate_private_key(ec.SECP256R1())
    pub_pem = priv_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    verifier = CosignBlobVerifier(trusted_public_keys_pem={"default": pub_pem})
    rev_sha = "a" * 64

    shared_mgr = _FakeMgr()
    sidecar_shared = AdapterSyncSidecar(
        expected_base_model_id="google/gemma-4-26B-A4B-it",
        expected_base_model_revision_sha256=rev_sha,
        staging_root=tmp_path / "shared",
        cosign_verifier=verifier,
        rls_conn_manager=shared_mgr,  # type: ignore[arg-type]
        in_memory_repo=InMemoryRLSRepository(),
        owns_rls_conn_manager=False,
    )
    sidecar_shared.close()
    assert shared_mgr.closed is False

    owned_mgr = _FakeMgr()
    sidecar_owned = AdapterSyncSidecar(
        expected_base_model_id="google/gemma-4-26B-A4B-it",
        expected_base_model_revision_sha256=rev_sha,
        staging_root=tmp_path / "owned",
        cosign_verifier=verifier,
        rls_conn_manager=owned_mgr,  # type: ignore[arg-type]
        in_memory_repo=InMemoryRLSRepository(),
        owns_rls_conn_manager=True,
    )
    sidecar_owned.close()
    assert owned_mgr.closed is True


def test_api_main_import_does_not_block_when_pg_unreachable() -> None:
    """Importing context_scout.api.main must fail fast (<5s) when PG17 port is unreachable."""
    import socket
    import time

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    closed_port = sock.getsockname()[1]
    sock.close()

    env = os.environ.copy()
    env["SCOUT_DB_APP_URL"] = (
        f"postgresql://scout_app_user:pw@127.0.0.1:{closed_port}/context_scout"
    )
    env.setdefault("SCOUT_AUTH_HMAC_SECRET", "test-only-hmac-secret-at-least-32-chars")

    t0 = time.monotonic()
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import context_scout.api.main; print(context_scout.api.main.app.title)",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    elapsed = time.monotonic() - t0
    assert proc.returncode == 0, proc.stderr
    assert "Enterprise Context Scout API" in proc.stdout
    assert elapsed < 5.0, f"api.main import stalled for {elapsed:.2f}s on unreachable PG"


def test_env_example_byo_and_keyless_gcp_parity() -> None:
    """Verifies .env.example documents BYO open-weights, fail-closed egress, and WIF."""
    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "Google-Only Model Stack" not in env_example
    assert "BYO Open-Weights" in env_example
    assert "SCOUT_RERANKER_MODEL_ID=BAAI/bge-reranker-v2-m3" in env_example
    assert "SCOUT_ALLOW_EXTERNAL_CLOUD_EGRESS=false" in env_example
    assert "SCOUT_USE_VERTEX_AI=true" in env_example
    assert "SCOUT_GCP_LOCATION=europe-west3" in env_example
    assert "Keyless GCP Vertex AI" in env_example
    assert "gemini-3.8-flash / gemini-4-argon worktree" not in env_example


def test_scout_resolver_rejects_multiline_injection_and_accepts_valid_upstream(
    tmp_path: Path,
) -> None:
    """25-scout-resolver.sh must reject multi-line/control-char injection."""
    resolver_script = REPO_ROOT / "deploy" / "docker" / "25-scout-resolver.sh"
    sh_bin = shutil.which("sh") or "/bin/sh"
    conf_dir = tmp_path / "nginx_conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    resolv_conf = tmp_path / "resolv.conf"
    resolv_conf.write_text("nameserver 10.96.0.10\nnameserver fd00::10\n", encoding="utf-8")
    resolver_conf = conf_dir / "nginx-scout-resolver.conf"
    override_conf = conf_dir / "nginx-scout-api-override.conf"

    def _cleanup() -> None:
        resolver_conf.unlink(missing_ok=True)
        override_conf.unlink(missing_ok=True)

    base_env = os.environ.copy()
    base_env["SCOUT_NGINX_CONF_DIR"] = str(conf_dir)
    base_env["SCOUT_RESOLV_CONF"] = str(resolv_conf)

    invalid_upstreams = (
        'http://context-scout-api:8080\n"; add_header X-Injected pwned always; return 200 "pwned',
        "http://context-scout-api:8080\n",
        "http://context-scout-api:8080\r; return 200 pwned",
        "http://context-scout-api:8080; return 200 pwned;",
        "http://context-scout-api:8080/extra",
        "http://context-scout-api:0",
        "http://context-scout-api:65536",
        "http://context-scout-api:000001",
        "   ",
    )

    try:
        for bad_upstream in invalid_upstreams:
            _cleanup()
            env = dict(base_env)
            env["SCOUT_API_UPSTREAM"] = bad_upstream
            proc = subprocess.run(
                [sh_bin, str(resolver_script)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            assert proc.returncode == 1, (
                f"Expected exit 1 for {bad_upstream!r}, got {proc.returncode}"
            )
            assert "Invalid SCOUT_API_UPSTREAM value" in proc.stderr
            assert not override_conf.exists(), (
                f"Override conf must not be written for invalid upstream {bad_upstream!r}"
            )

        for valid_upstream in (
            "http://my-api.scout.svc:9000",
            "https://context-scout-api:8443",
            "http://context-scout-api",
        ):
            _cleanup()
            env = dict(base_env)
            env["SCOUT_API_UPSTREAM"] = valid_upstream
            proc = subprocess.run(
                [sh_bin, str(resolver_script)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            assert proc.returncode == 0, proc.stderr
            assert resolver_conf.is_file()
            resolver_text = resolver_conf.read_text(encoding="utf-8")
            assert resolver_text.startswith("resolver ")
            assert "[fd00::10]" in resolver_text
            assert override_conf.is_file()
            assert (
                override_conf.read_text(encoding="utf-8")
                == f'set $scout_api_upstream "{valid_upstream}";\n'
            )

        _cleanup()
        env_unset = {k: v for k, v in base_env.items() if k != "SCOUT_API_UPSTREAM"}
        proc_unset = subprocess.run(
            [sh_bin, str(resolver_script)],
            env=env_unset,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc_unset.returncode == 0, proc_unset.stderr
        assert resolver_conf.is_file()
        assert not override_conf.exists()

        # Stale override conf must be deleted when upstream is unset/empty/rejected.
        override_conf.write_text(
            'set $scout_api_upstream "http://stale-api:8080";\n', encoding="utf-8"
        )
        proc_stale_unset = subprocess.run(
            [sh_bin, str(resolver_script)],
            env=env_unset,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc_stale_unset.returncode == 0, proc_stale_unset.stderr
        assert not override_conf.exists()

        override_conf.write_text(
            'set $scout_api_upstream "http://stale-api:8080";\n', encoding="utf-8"
        )
        env_empty = dict(env_unset)
        env_empty["SCOUT_API_UPSTREAM"] = ""
        proc_stale_empty = subprocess.run(
            [sh_bin, str(resolver_script)],
            env=env_empty,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc_stale_empty.returncode == 0, proc_stale_empty.stderr
        assert not override_conf.exists()

        override_conf.write_text(
            'set $scout_api_upstream "http://stale-api:8080";\n', encoding="utf-8"
        )
        env_bad_with_stale = dict(env_unset)
        env_bad_with_stale["SCOUT_API_UPSTREAM"] = "http://context-scout-api:65536"
        proc_stale_bad = subprocess.run(
            [sh_bin, str(resolver_script)],
            env=env_bad_with_stale,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc_stale_bad.returncode == 1
        assert not override_conf.exists()
    finally:
        _cleanup()


def test_webui_self_hosted_woff2_fonts_valid_ots_brotli() -> None:
    """All 5 self-hosted web-ui .woff2 font files decode cleanly via TTFont/Brotli."""
    from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

    fonts_dir = REPO_ROOT / "web-ui" / "public" / "fonts"
    expected_fonts = (
        "PlusJakartaSans-Regular.woff2",
        "PlusJakartaSans-SemiBold.woff2",
        "PlusJakartaSans-Bold.woff2",
        "JetBrainsMono-Regular.woff2",
        "JetBrainsMono-Medium.woff2",
    )
    for fname in expected_fonts:
        fpath = fonts_dir / fname
        assert fpath.is_file(), f"Missing self-hosted font: {fpath}"
        assert fpath.stat().st_size > 1000, f"Stub font detected ({fpath.stat().st_size}B): {fpath}"
        font = TTFont(str(fpath))
        assert font.flavor == "woff2"
        assert len(font.getGlyphOrder()) >= 100
        assert "cmap" in font and "glyf" in font


def test_dockerfiles_dockerignore_and_uv_lock_cve_hardening() -> None:
    """Guard Dockerfiles, root .dockerignore, and uv.lock against CVE regressions."""
    import tomllib

    docker_dir = REPO_ROOT / "deploy" / "docker"
    for py_name in ("Dockerfile.api", "Dockerfile.worker", "Dockerfile.adapter-sidecar"):
        text = (docker_dir / py_name).read_text(encoding="utf-8")
        builder_stage, _sep, runtime_stage = text.partition("AS runtime")
        assert runtime_stage, f"{py_name} missing distroless runtime stage"
        assert "uv>=0.5.0,<0.7.0" not in text
        assert "uv>=0.13.4" in builder_stage
        assert "UV_COMPILE_BYTECODE=0" in builder_stage
        assert "pip uninstall -y uv" in builder_stage
        assert "__pycache__" in builder_stage and "rm -rf" in builder_stage

    webui_text = (docker_dir / "Dockerfile.webui").read_text(encoding="utf-8")
    assert "apk upgrade --no-cache" in webui_text
    user_lines = [
        line.strip() for line in webui_text.splitlines() if line.strip().startswith("USER ")
    ]
    assert user_lines and user_lines[-1] == "USER 65532:65532"

    root_ignore = REPO_ROOT / ".dockerignore"
    nested_ignore = docker_dir / ".dockerignore"
    assert root_ignore.is_file(), "Missing root ./.dockerignore for BuildKit context=."
    assert root_ignore.read_text(encoding="utf-8") == nested_ignore.read_text(encoding="utf-8")
    ignore_text = root_ignore.read_text(encoding="utf-8")
    for token in (
        "__pycache__",
        "*.pyc",
        ".venv",
        "examples/quant_dogfood",
        "tests",
        ".env",
        "**/.DS_Store",
    ):
        assert token in ignore_text

    lock_data = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = {pkg["name"]: pkg["version"] for pkg in lock_data.get("package", [])}
    assert "fsspec" in packages
    fsspec_ver = tuple(int(p) for p in str(packages["fsspec"]).split(".")[:3])
    assert fsspec_ver >= (2026, 6, 0), f"Vulnerable fsspec in uv.lock: {packages['fsspec']}"
    assert "datasets" in packages
    datasets_ver = tuple(int(p) for p in str(packages["datasets"]).split(".")[:3])
    assert datasets_ver >= (5, 2, 0), f"Legacy datasets in uv.lock: {packages['datasets']}"


def test_pr5_supply_chain_health_probes_and_corpus_hygiene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies CI SHA pinning, SECURITY.md, Makefile, health probes, pickle, and corpus."""
    from fastapi.testclient import TestClient

    from context_scout.api.main import app
    from context_scout.api.routes.health import reset_health_probe_cache

    ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    uses_lines = [ln.strip() for ln in ci_text.splitlines() if "uses:" in ln]
    assert len(uses_lines) >= 15
    for ln in uses_lines:
        assert not re.search(r"@v\d+", ln.split("#", 1)[0]), (
            f"Mutable @v* action ref found in ci.yml: {ln}"
        )

    assert (REPO_ROOT / "SECURITY.md").is_file()
    makefile_text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "scripts/run_test_batches.py --mode fast" in makefile_text
    assert "scripts/run_test_batches.py --mode full" in makefile_text

    # Split health probes: /live returns 200 without DB; /ready returns 200 in memory mode
    reset_health_probe_cache()
    monkeypatch.setenv("SCOUT_STORAGE_BACKEND", "memory")
    client = TestClient(app)
    live_resp = client.get("/api/v1/health/live")
    assert live_resp.status_code == 200
    assert live_resp.json()["status"] == "alive"

    ready_resp = client.get("/api/v1/health/ready")
    assert ready_resp.status_code == 200
    assert ready_resp.json()["status"] == "ok"
    reset_health_probe_cache()

    # Air-gap bundle script must reject disguised pickle magic bytes regardless of extension
    script = REPO_ROOT / "deploy" / "airgap" / "create_airgap_bundle.sh"
    bad_dir = tmp_path / "disguised_pickle"
    bad_dir.mkdir()
    (bad_dir / "weights.safetensors").write_bytes(b"\x80\x04\x95fake-pickle")
    proc = subprocess.run(
        [
            str(script),
            "--domain-id",
            "enterprise_docs_scout",
            str(bad_dir),
            str(tmp_path / "b.tgz"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0

    # Corpus enrichment script must never use db_admin_url and must fail closed on offline egress
    corpus_script = REPO_ROOT / "scripts" / "clean_and_enrich_corpus.py"
    corpus_text = corpus_script.read_text(encoding="utf-8")
    assert "db_admin_url" not in corpus_text
    assert "RLSConnectionManager" in corpus_text
    assert "embedding_halfvec" in corpus_text

    import importlib.util

    spec = importlib.util.spec_from_file_location("clean_and_enrich_corpus", corpus_script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from context_scout.config import ScoutSettings

    with pytest.raises(RuntimeError, match="SCOUT_ALLOW_EXTERNAL_CLOUD_EGRESS"):
        mod.fetch_wikipedia_plaintext(
            "ACID", settings=ScoutSettings(allow_external_cloud_egress=False)
        )
