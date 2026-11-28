"""Mutation verification probes for ClaimVerifier and AdapterRouter."""

from __future__ import annotations

import types
import uuid
from pathlib import Path
from typing import Any

from context_scout.config import ClaimVerificationPolicy, ClaimVerifierConfig
from context_scout.db.repositories import DocChunkRecord
from context_scout.schemas.contracts import (
    AuthorityRules,
    DomainPackManifest,
    PrincipalContext,
    RoleType,
)


def _probe_claim_verifier(mod: types.ModuleType) -> None:
    assert mod.normalize_claim_text("  Hello   \ufb01le  ") == "Hello file"
    assert mod.normalize_claim_text("") == ""

    assert mod.detect_homoglyph_spoofing("!!!") == (False, "")
    hg_mixed, reason_m = mod.detect_homoglyph_spoofing("Verify s\u0435cret")
    assert hg_mixed is True and "Mixed-script" in reason_m
    hg_greek_mix, _ = mod.detect_homoglyph_spoofing("Verify s\u03bfme")
    assert hg_greek_mix is True
    hg_pure, reason_p = mod.detect_homoglyph_spoofing("Valid Latin \u0430\u0441")
    assert hg_pure is True and "Pure homoglyph" in reason_p
    hg_pure_gr, _ = mod.detect_homoglyph_spoofing("Valid Latin \u03bf\u03bd")
    assert hg_pure_gr is True
    hg_ru, _ = mod.detect_homoglyph_spoofing(
        "\u041f\u0440\u0438\u0432\u0435\u0442 123 \u043c\u0438\u0440"
    )
    assert hg_ru is False
    hg_gr, _ = mod.detect_homoglyph_spoofing("\u0391\u03b2\u03b3 \u03b4\u03b5\u03b6")
    assert hg_gr is False
    hg_nfkc, _ = mod.detect_homoglyph_spoofing("The \uff21PI")
    assert hg_nfkc is True
    assert mod.detect_homoglyph_spoofing("Valid Latin text") == (False, "")
    assert mod.detect_homoglyph_spoofing(
        "\u041f\u0440\u0438\u0432\u0435\u0442 \u0430 \u043c\u0438\u0440"
    ) == (False, "")
    assert mod.detect_homoglyph_spoofing(
        "Hello \u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440 \u0430"
    ) == (False, "")
    assert mod.detect_homoglyph_spoofing("Hello \u0391\u03b2\u03b3 \u03b4\u03b5\u03b6 \u03bf") == (
        False,
        "",
    )

    verifier = mod.ClaimVerifier()
    sources = {"doc1": "PreFlightValidator enforces FORCE RLS and 100% verbatim citations."}

    r_ok = verifier.verify_single_claim("enforces FORCE RLS", sources)
    assert r_ok.is_verified is True and r_ok.matched_source_id == "doc1"
    r_tup = verifier.verify_single_claim("enforces FORCE RLS", [("tup1", sources["doc1"])])
    assert r_tup.is_verified is True and r_tup.matched_source_id == "tup1"
    chunk_obj = DocChunkRecord(
        chunk_id=uuid.uuid4(),
        domain_id="code_architecture_scout",
        doc_id="d1",
        doc_title="D1",
        source_uri="https://docs.internal/d1",
        market_scope=None,
        published_date=None,
        authority_weight=1.0,
        section_heading="S1",
        contextual_header="D1",
        verbatim_content=sources["doc1"],
        metadata_json={},
        allowed_scopes=["public_eng"],
        embedding=[0.1] * 768,
    )
    r_chunk = verifier.verify_single_claim("enforces FORCE RLS", [chunk_obj])
    assert r_chunk.is_verified is True and r_chunk.matched_source_id == str(chunk_obj.chunk_id)
    r_str = verifier.verify_single_claim("enforces FORCE RLS", [sources["doc1"]])
    assert r_str.is_verified is True

    r_sub = verifier.verify_single_claim("enforces optional RLS", sources)
    assert r_sub.is_verified is False
    r_empty = verifier.verify_single_claim("   ", sources)
    assert r_empty.is_verified is False
    r_hg = verifier.verify_single_claim("enforces F\u041eRCE RLS", sources)
    assert r_hg.is_verified is False and r_hg.is_homoglyph_attack is True
    r_nosrc = verifier.verify_single_claim("enforces FORCE RLS", [])
    assert r_nosrc.is_verified is False

    strict_pass = verifier.verify_claims(["enforces FORCE RLS"], sources)
    assert (
        strict_pass.passed is True
        and strict_pass.paid_tokens_spent == 0
        and strict_pass.diagnostic is None
    )
    strict_zero = verifier.verify_claims([], sources)
    assert strict_zero.passed is False
    strict_fail = verifier.verify_claims(["enforces FORCE RLS", "hallucinated"], sources)
    assert strict_fail.passed is False
    assert strict_fail.verified_claims == ["enforces FORCE RLS"]
    assert strict_fail.pruned_claims == ["hallucinated"]
    assert strict_fail.diagnostic is not None
    assert strict_fail.diagnostic.paid_tokens_spent == 0
    assert any("hallucinated" in sym for sym in strict_fail.diagnostic.missing_symbols)
    assert "hallucinated" in (strict_fail.diagnostic_reason or "")

    prune_v = mod.ClaimVerifier(
        ClaimVerifierConfig(
            policy=ClaimVerificationPolicy.PRUNE_AND_MARK_LOW_TRUST,
            min_verified_claims=1,
        )
    )
    assert prune_v.verify_claims(["enforces FORCE RLS", "bad"], sources).passed is True
    assert prune_v.verify_claims(["bad"], sources).passed is False

    min_v = mod.ClaimVerifier(
        ClaimVerifierConfig(
            policy=ClaimVerificationPolicy.MIN_VERIFIED_COUNT,
            min_verified_claims=2,
        )
    )
    assert min_v.verify_claims(["enforces FORCE RLS", "verbatim citations"], sources).passed is True
    assert min_v.verify_claims(["enforces FORCE RLS"], sources).passed is False


def _probe_adapter_router(mod: types.ModuleType) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        domains_dir = Path(tmp) / "domains"
        dom_dir = domains_dir / "test_dom"
        adapter_dir = dom_dir / "adapters" / "lora_test_dom"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapters.safetensors").write_bytes(b"safetensors")

        spec = {
            "domain_id": "test_dom",
            "base_model_id": "google/gemma-4-26B-A4B-it",
            "training_scope_ceiling": ["tenant_common", "enclave:eng"],
            "lora_adapter_id": "lora_test_dom",
            "lora_adapter_path": str(adapter_dir),
            "lora_policy": {
                "enabled": True,
                "rank": 16,
                "use_rslora": True,
                "use_dora": False,
            },
        }
        router = mod.AdapterRouter(domains_dir=domains_dir, spec_overrides={"test_dom": spec})
        ok_p = PrincipalContext(
            user_id="eng1",
            role=RoleType.ENGINEER,
            allowed_scopes=["tenant_common", "enclave:eng", "extra"],
        )
        bad_p = PrincipalContext(
            user_id="eng2",
            role=RoleType.ENGINEER,
            allowed_scopes=["tenant_common"],
        )

        # Subset allowed -> routes to LoRA
        d_ok = router.route("test_dom", ok_p)
        assert (
            d_ok.routed_to_lora is True
            and d_ok.lora_request == "lora_test_dom"
            and d_ok.lora_adapter_id == "lora_test_dom"
            and d_ok.base_model_id == "google/gemma-4-26B-A4B-it"
        )

        # Relative adapter path + adapter_config.json + default base_model_id/lora_adapter_id
        rel_dir = domains_dir / "rel_dom" / "adapters" / "lora_rel_dom"
        rel_dir.mkdir(parents=True, exist_ok=True)
        (rel_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        rel_spec = {
            "domain_id": "rel_dom",
            "training_scope_ceiling": ["tenant_common"],
            "lora_policy": {"enabled": True, "rank": 8, "use_rslora": True},
        }
        r_rel = mod.AdapterRouter(domains_dir=domains_dir, spec_overrides={"rel_dom": rel_spec})
        d_rel = r_rel.route("rel_dom", ok_p)
        assert d_rel.routed_to_lora is True and d_rel.lora_adapter_id == "lora_rel_dom"

        # AdapterPool integration (verify_adapter + get_slot + error fallback)
        class _FakeSlot:
            adapter_id = "pool_adapter_id"
            adapter_path = str(adapter_dir)

        class _FakePool:
            def __init__(self, valid: bool = True, legacy_sig: bool = False) -> None:
                self._valid = valid
                self._legacy_sig = legacy_sig

            def verify_adapter(self, domain_id: str, tenant_id: Any = None) -> dict[str, Any]:
                if self._legacy_sig and tenant_id is not None:
                    raise TypeError("legacy signature")
                return {"is_valid": self._valid}

            def get_slot(self, domain_id: str) -> _FakeSlot:
                return _FakeSlot()

        r_pool = mod.AdapterRouter(
            domains_dir=domains_dir,
            adapter_pool=_FakePool(valid=True),
            spec_overrides={"test_dom": {**spec, "lora_adapter_path": ""}},
        )
        d_pool = r_pool.route("test_dom", ok_p, verify_weights=True)
        assert d_pool.routed_to_lora is True and d_pool.lora_adapter_id == "pool_adapter_id"

        r_pool_legacy = mod.AdapterRouter(
            domains_dir=domains_dir,
            adapter_pool=_FakePool(valid=True, legacy_sig=True),
            spec_overrides={"test_dom": {**spec, "lora_adapter_path": ""}},
        )
        assert r_pool_legacy.route("test_dom", ok_p, verify_weights=True).routed_to_lora is True

        assert router._verify_adapter_weights("test_dom", None, ok_p) is False
        r_empty_path = mod.AdapterRouter(
            domains_dir=domains_dir,
            spec_overrides={
                "empty_dom": {**spec, "domain_id": "empty_dom", "lora_adapter_path": ""}
            },
        )
        assert r_empty_path.route("empty_dom", ok_p, verify_weights=True).routed_to_lora is False

        # Superset ceiling -> routes to frozen base
        d_bad_p = router.route("test_dom", bad_p)
        assert d_bad_p.routed_to_lora is False and d_bad_p.lora_request is None

        # Runtime backend tensor-key format verification (mlx_lm vs hf_peft vs corrupt vs missing)
        import json as _json
        import struct as _struct

        def _make_st(keys: list[str]) -> bytes:
            hdr = _json.dumps(
                {k: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for k in keys}
            ).encode("utf-8")
            return _struct.pack("<Q", len(hdr)) + hdr + b"\x00\x00\x00\x00"

        mlx_dir = domains_dir / "fmt_mlx" / "adapters" / "lora_mlx"
        mlx_dir.mkdir(parents=True, exist_ok=True)
        (mlx_dir / "adapters.safetensors").write_bytes(
            _make_st(["model.layers.0.q_proj.lora_a", "model.layers.0.q_proj.lora_b"])
        )
        peft_dir = domains_dir / "fmt_peft" / "adapters" / "lora_peft"
        peft_dir.mkdir(parents=True, exist_ok=True)
        (peft_dir / "adapter_model.safetensors").write_bytes(
            _make_st(["base_model.layers.0.q_proj.lora_A.weight"])
        )
        corrupt_dir = domains_dir / "fmt_corrupt" / "adapters" / "lora_corrupt"
        corrupt_dir.mkdir(parents=True, exist_ok=True)
        (corrupt_dir / "adapters.safetensors").write_bytes(b"\xff\x00bad")
        missing_dir = domains_dir / "fmt_missing" / "adapters" / "lora_missing"
        missing_dir.mkdir(parents=True, exist_ok=True)
        (missing_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

        r_fmt = mod.AdapterRouter(
            domains_dir=domains_dir,
            spec_overrides={
                "fmt_mlx": {**spec, "domain_id": "fmt_mlx", "lora_adapter_path": str(mlx_dir)},
                "fmt_peft": {**spec, "domain_id": "fmt_peft", "lora_adapter_path": str(peft_dir)},
                "fmt_corrupt": {
                    **spec,
                    "domain_id": "fmt_corrupt",
                    "lora_adapter_path": str(corrupt_dir),
                },
                "fmt_missing": {
                    **spec,
                    "domain_id": "fmt_missing",
                    "lora_adapter_path": str(missing_dir),
                },
            },
        )
        assert (
            r_fmt.route(
                "fmt_mlx", ok_p, verify_weights=True, runtime_backend="mlx_lm"
            ).routed_to_lora
            is True
        )
        assert (
            r_fmt.route("fmt_mlx", ok_p, verify_weights=True, runtime_backend="vllm").routed_to_lora
            is False
        )
        assert (
            r_fmt.route(
                "fmt_peft", ok_p, verify_weights=True, runtime_backend="vllm"
            ).routed_to_lora
            is True
        )
        assert (
            r_fmt.route(
                "fmt_peft", ok_p, verify_weights=True, runtime_backend="mlx_lm"
            ).routed_to_lora
            is False
        )
        assert (
            r_fmt.route(
                "fmt_corrupt", ok_p, verify_weights=True, runtime_backend="mlx_lm"
            ).routed_to_lora
            is False
        )
        assert (
            r_fmt.route(
                "fmt_missing", ok_p, verify_weights=True, runtime_backend="mlx_lm"
            ).routed_to_lora
            is False
        )

        # Missing principal / domain / manifest disabled
        assert router.route("test_dom", None).routed_to_lora is False
        assert router.route(None, ok_p).routed_to_lora is False
        assert router.route("missing_dom", ok_p).routed_to_lora is False

        manifest_off = DomainPackManifest(
            domain_id="test_dom",
            display_name="Test",
            base_model_id="google/gemma-4-26B-A4B-it",
            lora_adapter_id="lora_test_dom",
            lora_adapter_path=str(adapter_dir),
            authority_rules=AuthorityRules(primary_truth_source="code"),
            output_contract="ImplementationContextPack",
            mcp_read_tools_allowed=["code_repo_mcp:search_code"],
            description="Test",
            lora_enabled=False,
        )
        assert router.route("test_dom", ok_p, manifest=manifest_off).routed_to_lora is False

        # Policy failures (use_dora=True, use_rslora=False/None, bad rank, empty ceiling)
        for mutated_spec in (
            {**spec, "lora_policy": {"enabled": False, "rank": 16, "use_rslora": True}},
            {
                **spec,
                "lora_policy": {"enabled": True, "rank": 16, "use_rslora": True, "use_dora": True},
            },
            {**spec, "lora_policy": {"enabled": True, "rank": 16, "use_rslora": False}},
            {**spec, "lora_policy": {"enabled": True, "rank": 16}},
            {**spec, "lora_policy": {"enabled": True, "rank": 64, "use_rslora": True}},
            {**spec, "training_scope_ceiling": []},
        ):
            r2 = mod.AdapterRouter(
                domains_dir=domains_dir, spec_overrides={"test_dom": mutated_spec}
            )
            assert r2.route("test_dom", ok_p).routed_to_lora is False

        # Cross-tenant isolation probe on AdapterRouter
        tenant_scoped_spec = {**spec, "tenant_id": str(uuid.uuid4())}
        r_tenant = mod.AdapterRouter(domains_dir=domains_dir)
        r_tenant._spec_overrides["test_dom"] = tenant_scoped_spec
        assert r_tenant.route("test_dom", ok_p).routed_to_lora is False
        assert r_tenant.route("test_dom", None).routed_to_lora is False

        # Missing weights with verify_weights=True -> base
        no_weights_spec = {**spec, "lora_adapter_path": str(dom_dir / "nonexistent")}
        r3 = mod.AdapterRouter(
            domains_dir=domains_dir, spec_overrides={"test_dom": no_weights_spec}
        )
        assert r3.route("test_dom", ok_p, verify_weights=True).routed_to_lora is False
        assert r3.route("test_dom", ok_p, verify_weights=False).routed_to_lora is True
