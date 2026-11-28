# Security Policy

## Supported Versions

| Version | Supported |
| --- | --- |
| `1.0.x` | Yes |
| `< 1.0` | No |

## Reporting a Vulnerability

Please report suspected security vulnerabilities privately via GitHub Security Advisories for this repository rather than opening a public issue. Include:

1. Affected component, file path, or API endpoint.
2. Minimal reproduction steps or proof-of-concept.
3. Impact assessment (for example: tenant isolation, RBAC/RLS, secret exposure, or supply-chain integrity).

Target acknowledgment SLA is 2 business days, with remediation prioritized by severity (`SEV-0` critical isolation/RCE flaws immediately).

## Supply-Chain & AI-BOM Verification

- GitHub Actions workflows pin third-party actions to immutable 40-character commit SHAs.
- Releases ship a CycloneDX v1.6 AI-BOM (`aibom.cdx.json`) backed by `uv.lock` and `web-ui/package-lock.json` hashes, signed as a Sigstore bundle (`aibom.cdx.json.sigstore.json`) with accompanying public key (`aibom.cdx.json.pub`).
- LoRA adapters are restricted to `.safetensors` weights (`trust_remote_code=false`) verified by `CosignBlobVerifier` before loading into `vLLM`.
