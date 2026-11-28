#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: create_airgap_bundle.sh [OPTIONS] <source_dir> <output_bundle.tar.gz>

Packages a source directory into a deterministic, AirGappedBundleImporter-compatible
.tar.gz archive with root manifest.json, <bundle>.tar.gz.sha256, and optional
cosign signature bundle.

Options:
  --bundle-id ID          Bundle identifier (default: derived from output filename)
  --domain-id ID          Target domain_id (default: code_architecture_scout)
  --tenant-id UUID        Optional tenant UUID
  --scopes CSV            Comma-separated allowed_scopes (default: public)
  --cosign-key PATH       Optional cosign private key for signing the .tar.gz bundle
  -h, --help              Show this help message
EOF
}

BUNDLE_ID=""
DOMAIN_ID="code_architecture_scout"
TENANT_ID=""
SCOPES_CSV="public"
COSIGN_KEY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-id)
      BUNDLE_ID="${2:?Missing value for --bundle-id}"
      shift 2
      ;;
    --domain-id)
      DOMAIN_ID="${2:?Missing value for --domain-id}"
      shift 2
      ;;
    --tenant-id)
      TENANT_ID="${2:?Missing value for --tenant-id}"
      shift 2
      ;;
    --scopes)
      SCOPES_CSV="${2:?Missing value for --scopes}"
      shift 2
      ;;
    --cosign-key)
      COSIGN_KEY="${2:?Missing value for --cosign-key}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

SOURCE_DIR="$1"
OUTPUT_BUNDLE="$2"

if [[ ! -d "$SOURCE_DIR" ]] || [[ -L "$SOURCE_DIR" ]]; then
  echo "ERROR: source_dir must be a real directory (not a symlink): $SOURCE_DIR" >&2
  exit 1
fi

if [[ -z "$BUNDLE_ID" ]]; then
  base_name="$(basename "$OUTPUT_BUNDLE")"
  BUNDLE_ID="${base_name%.tar.gz}"
  BUNDLE_ID="${BUNDLE_ID%.tgz}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python3 is required for deterministic bundle packaging." >&2
  exit 1
fi

"$PYTHON_BIN" - "$SOURCE_DIR" "$OUTPUT_BUNDLE" "$BUNDLE_ID" "$DOMAIN_ID" "$TENANT_ID" "$SCOPES_CSV" <<'PYEOF'
import gzip
import hashlib
import io
import json
import os
import posixpath
import stat
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

UNSAFE_EXTENSIONS = frozenset({".pkl", ".pickle", ".pt", ".pth", ".bin", ".h5", ".ckpt"})
PICKLE_MAGIC_PREFIXES = (b"\x80\x01", b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05")


def _assert_safe_payload(norm_rel: str, raw_bytes: bytes) -> None:
    if any(raw_bytes.startswith(prefix) for prefix in PICKLE_MAGIC_PREFIXES):
        raise SystemExit(
            f"ERROR: Pickle protocol magic bytes rejected regardless of extension: {norm_rel}"
        )
    if raw_bytes.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                for member in zf.namelist():
                    lowered = member.lower()
                    if lowered.endswith((".pkl", ".pickle")) or "data.pkl" in lowered:
                        raise SystemExit(
                            f"ERROR: ZIP-wrapped pickle archive rejected ({member}): {norm_rel}"
                        )
        except zipfile.BadZipFile:
            pass

source_dir = Path(sys.argv[1])
output_bundle = Path(sys.argv[2])
bundle_id = sys.argv[3].strip()
domain_id = sys.argv[4].strip()
tenant_id = sys.argv[5].strip()
scopes_csv = sys.argv[6]

if not bundle_id:
    raise SystemExit("ERROR: bundle_id must be non-empty.")
if not domain_id:
    raise SystemExit("ERROR: domain_id must be non-empty.")

allowed_scopes = [s.strip() for s in scopes_csv.split(",") if s.strip()]
if not allowed_scopes:
    raise SystemExit("ERROR: allowed_scopes must be non-empty.")

root_resolved = source_dir.resolve()
collected_files: list[tuple[str, bytes]] = []

for dirpath, dirnames, filenames in os.walk(source_dir, followlinks=False):
    dirnames.sort()
    filenames.sort()
    current_dir = Path(dirpath)

    for dname in dirnames:
        sub = current_dir / dname
        if sub.is_symlink():
            raise SystemExit(f"ERROR: Symlink directory rejected: {sub}")

    for fname in filenames:
        fpath = current_dir / fname
        if fpath.is_symlink():
            raise SystemExit(f"ERROR: Symlink file rejected: {fpath}")

        st = os.lstat(fpath)
        if stat.S_ISLNK(st.st_mode):
            raise SystemExit(f"ERROR: Symlink rejected: {fpath}")
        if st.st_nlink > 1:
            raise SystemExit(f"ERROR: Hardlink rejected (st_nlink={st.st_nlink}): {fpath}")
        if not stat.S_ISREG(st.st_mode):
            raise SystemExit(f"ERROR: Non-regular file rejected: {fpath}")

        resolved = fpath.resolve()
        if not resolved.is_relative_to(root_resolved):
            raise SystemExit(f"ERROR: Path escapes source root: {fpath}")

        rel = fpath.relative_to(source_dir).as_posix()
        if not rel or "\x00" in rel or "\\" in rel or rel.startswith("/"):
            raise SystemExit(f"ERROR: Invalid relative path: {rel!r}")
        parts = PurePosixPath(rel).parts
        if any(part in ("", ".", "..") for part in parts):
            raise SystemExit(f"ERROR: Path traversal segment rejected: {rel!r}")
        norm_rel = posixpath.normpath(rel)
        if norm_rel == "manifest.json":
            continue

        suffix = Path(norm_rel).suffix.lower()
        if suffix in UNSAFE_EXTENSIONS:
            raise SystemExit(f"ERROR: Unsafe weight/pickle extension rejected ({suffix}): {norm_rel}")

        raw_bytes = fpath.read_bytes()
        _assert_safe_payload(norm_rel, raw_bytes)
        collected_files.append((norm_rel, raw_bytes))

if not collected_files:
    raise SystemExit(f"ERROR: No eligible payload files found in {source_dir}")

collected_files.sort(key=lambda item: item[0])

manifest_entries = []
for rel_path, raw_bytes in collected_files:
    digest = hashlib.sha256(raw_bytes).hexdigest().lower()
    manifest_entries.append(
        {
            "path": rel_path,
            "sha256": digest,
            "size_bytes": len(raw_bytes),
            "allowed_scopes": allowed_scopes,
        }
    )

manifest: dict[str, object] = {
    "schema_version": "1.0",
    "bundle_id": bundle_id,
    "domain_id": domain_id,
    "allowed_scopes": allowed_scopes,
    "files": manifest_entries,
}
if tenant_id:
    manifest["tenant_id"] = tenant_id

manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

output_bundle.parent.mkdir(parents=True, exist_ok=True)
tar_buf = io.BytesIO()
with tarfile.open(fileobj=tar_buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
    all_members = [("manifest.json", manifest_bytes), *collected_files]
    for member_name, payload in all_members:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(payload)
        info.mode = 0o644
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = 0
        tf.addfile(info, io.BytesIO(payload))

gz_buf = io.BytesIO()
with gzip.GzipFile(fileobj=gz_buf, mode="wb", mtime=0, compresslevel=9) as gz:
    gz.write(tar_buf.getvalue())

bundle_bytes = gz_buf.getvalue()
output_bundle.write_bytes(bundle_bytes)

bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest().lower()
sha_file = Path(f"{output_bundle}.sha256")
sha_file.write_text(f"{bundle_sha256}  {output_bundle.name}\n", encoding="utf-8")

print(
    f"Created deterministic air-gap bundle: {output_bundle} "
    f"(files={len(collected_files)}, sha256={bundle_sha256})"
)
PYEOF

if [[ -n "$COSIGN_KEY" ]]; then
  if ! command -v cosign >/dev/null 2>&1; then
    echo "ERROR: --cosign-key specified but 'cosign' CLI not found in PATH." >&2
    exit 1
  fi
  cosign sign-blob --yes --key "$COSIGN_KEY" \
    --bundle "${OUTPUT_BUNDLE}.sigstore.json" \
    "$OUTPUT_BUNDLE"
  echo "Signed bundle with cosign: ${OUTPUT_BUNDLE}.sigstore.json"
fi
