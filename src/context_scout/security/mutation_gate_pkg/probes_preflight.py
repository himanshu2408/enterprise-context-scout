"""Mutation verification probe for PreFlightValidator (preflight_gate.py)."""

from __future__ import annotations

import types
import uuid

from context_scout.config import DomainConfig
from context_scout.db.repositories import (
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
    InMemoryRLSRepository,
)
from context_scout.schemas.contracts import (
    DocCitationSpan,
    ImplementationContextPack,
    ModificationIntent,
    PreFlightValidationReport,
    PrincipalContext,
    RoleType,
    VerbatimCodeSlice,
)


def _probe_preflight_gate(mod: types.ModuleType) -> None:
    repo = InMemoryRLSRepository()
    principal = PrincipalContext(
        user_id="pf_mut_eng",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )
    admin_principal = PrincipalContext(
        user_id="pf_mut_admin",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "exec_only"],
    )
    code_content = (
        "class OrderService:\n"
        "    def process(self, order_id: str) -> bool:\n"
        "        return bool(order_id)\n"
    )
    file_uuid = uuid.uuid4()
    f_rec = CodeFileRecord(
        file_id=file_uuid,
        domain_id="code_architecture_scout",
        repo_name="core",
        file_path="src/orders.py",
        git_commit_sha="1234567890abcdef",
        git_blob_sha="abcdef1234567890",
        language="python",
        allowed_scopes=["public_eng"],
        raw_content=code_content,
        line_count=3,
    )
    repo.insert_code_file(principal, f_rec)
    sym_rec = ASTSymbolRecord(
        symbol_id=uuid.uuid4(),
        file_id=file_uuid,
        domain_id="code_architecture_scout",
        qualified_name="OrderService.process",
        symbol_kind="method",
        start_line=2,
        end_line=3,
        signature_skeleton="def process(self, order_id: str) -> bool:",
        contextual_summary="Process order",
        allowed_scopes=["public_eng"],
    )
    repo.insert_ast_symbol(principal, sym_rec)
    c_rec = DocChunkRecord(
        chunk_id=uuid.uuid4(),
        domain_id="code_architecture_scout",
        doc_id="runbook",
        doc_title="Runbook",
        source_uri="https://docs.internal/runbook",
        market_scope=None,
        published_date=None,
        authority_weight=1.0,
        section_heading="SLA",
        contextual_header="Runbook SLA",
        verbatim_content="OrderService must complete processing within 50ms.",
        metadata_json={"resolution": "Fixed"},
        allowed_scopes=["public_eng"],
        embedding=[0.1] * 768,
        version_str="v1",
        is_stale=False,
    )
    stale_rec = DocChunkRecord(
        chunk_id=uuid.uuid4(),
        domain_id="code_architecture_scout",
        doc_id="old_runbook",
        doc_title="Old Runbook",
        source_uri="https://docs.internal/old_runbook",
        market_scope=None,
        published_date=None,
        authority_weight=0.5,
        section_heading="Old",
        contextual_header="Old Runbook",
        verbatim_content="Legacy SLA 500ms.",
        metadata_json={},
        allowed_scopes=["public_eng"],
        embedding=[0.2] * 768,
        version_str="v0",
        is_stale=True,
    )
    restricted_rec = DocChunkRecord(
        chunk_id=uuid.uuid4(),
        domain_id="code_architecture_scout",
        doc_id="secret_doc",
        doc_title="Secret Spec",
        source_uri="https://docs.internal/secret",
        market_scope=None,
        published_date=None,
        authority_weight=1.0,
        section_heading="Secret",
        contextual_header="Secret",
        verbatim_content="Restricted secret spec.",
        metadata_json={},
        allowed_scopes=["exec_only"],
        embedding=[0.3] * 768,
        version_str="v1",
        is_stale=False,
    )
    repo.insert_doc_chunk(principal, c_rec)
    repo.insert_doc_chunk(principal, stale_rec)
    repo.insert_doc_chunk(admin_principal, restricted_rec)

    validator = mod.PreFlightValidator(
        repo=repo,
        domain_id="code_architecture_scout",
        domain_config=DomainConfig(
            domain_id="code_architecture_scout", min_retrieval_confidence=0.65
        ),
    )
    bound = validator._resolve_graph_repo(repo)

    # 1. check_rrf_threshold
    assert validator.check_rrf_threshold(0.80) == (True, 0.8)
    assert validator.check_rrf_threshold(1.50) == (True, 1.0)
    assert validator.check_rrf_threshold(-0.50) == (False, 0.0)
    assert validator.check_rrf_threshold(0.30) == (False, 0.3)
    assert validator.check_rrf_threshold(0.30, cross_encoder_score=0.70) == (True, 0.3)
    assert validator.check_rrf_threshold(None, cross_encoder_score=0.75) == (True, 0.75)
    assert validator.check_rrf_threshold(None, cross_encoder_score=0.20) == (False, 0.2)
    assert validator.check_rrf_threshold(None, cross_encoder_score=None) == (True, 1.0)

    good_slice = VerbatimCodeSlice(
        repo_name="core",
        file_path="src/orders.py",
        start_line=1,
        end_line=3,
        qualified_symbol="OrderService.process",
        selection_reason="Target",
        modification_intent=ModificationIntent.MODIFY_TARGET,
        git_commit_sha="1234567890abcdef",
        git_blob_sha="abcdef1234567890",
        verbatim_code_with_lines=(
            "   1 | class OrderService:\n"
            "   2 |     def process(self, order_id: str) -> bool:\n"
            "   3 |         return bool(order_id)\n"
        ),
    )
    good_cit = DocCitationSpan(
        doc_id="runbook",
        doc_title="Runbook",
        source_uri="https://docs.internal/runbook",
        version="v1",
        section_heading="SLA",
        verbatim_quote="OrderService must complete processing within 50ms.",
        chunk_id=str(c_rec.chunk_id),
    )

    # 2. check_ast_syntax (Python, TS, TSX, nesting > 10, partial window truncation, empty)
    ok_syn, syn_errs, _ = validator.check_ast_syntax([good_slice])
    assert ok_syn is True and syn_errs == []
    empty_slice = good_slice.model_copy(update={"verbatim_code_with_lines": "   1 |    \n"})
    assert validator.check_ast_syntax([empty_slice])[0] is True
    bad_syn_slice = good_slice.model_copy(
        update={"verbatim_code_with_lines": "   1 | def broken(:\n   2 |     pass\n"}
    )
    ok_bad_syn, bad_syn_errs, _ = validator.check_ast_syntax([bad_syn_slice])
    assert ok_bad_syn is False and len(bad_syn_errs) == 1
    assert "Unclosed syntax delimiters" in bad_syn_errs[0]

    # Balanced multi-group parens (max depth 6 <= 10) must pass
    valid_nested = good_slice.model_copy(
        update={"verbatim_code_with_lines": "   1 | x = f((((((1)))))) + g((((((2))))))\n"}
    )
    assert validator.check_ast_syntax([valid_nested])[0] is True

    # Sub-file Python context window starting inside an indented else block (balanced delimiters,
    # invalid standalone/wrapped, passes only when verbatim lines match a valid parent module)
    win_code = (
        "class OrderService:\n"
        "    def run(self, x: int) -> str:\n"
        "        if x > 0:\n"
        "            return 'pos'\n"
        "        else:\n"
        "            return 'neg'\n"
    )
    win_file = CodeFileRecord(
        file_id=uuid.uuid4(),
        domain_id="code_architecture_scout",
        repo_name="core",
        file_path="src/order_win.py",
        git_commit_sha="1234567890abcdef",
        git_blob_sha="abcdef1234567891",
        language="python",
        allowed_scopes=["public_eng"],
        raw_content=win_code,
        line_count=6,
    )
    repo.insert_code_file(principal, win_file)
    win_slice = VerbatimCodeSlice(
        repo_name="core",
        file_path="src/order_win.py",
        start_line=5,
        end_line=6,
        qualified_symbol="OrderService.run",
        selection_reason="Target",
        modification_intent=ModificationIntent.REFERENCE_ONLY,
        git_commit_sha="1234567890abcdef",
        git_blob_sha="abcdef1234567891",
        verbatim_code_with_lines=("   5 |         else:\n   6 |             return 'neg'\n"),
    )
    assert (
        validator.check_ast_syntax(
            [win_slice], bound_repo=bound, principal=principal, domain_id="code_architecture_scout"
        )[0]
        is True
    )
    # Without parent repo context, orphan else branch must fail closed
    assert validator.check_ast_syntax([win_slice])[0] is False
    # Tampered line not matching parent module must fail closed
    tampered_win = win_slice.model_copy(
        update={
            "verbatim_code_with_lines": (
                "   5 |         else:\n   6 |             return 'tampered'\n"
            )
        }
    )
    assert (
        validator.check_ast_syntax(
            [tampered_win],
            bound_repo=bound,
            principal=principal,
            domain_id="code_architecture_scout",
        )[0]
        is False
    )
    # Broken parent module must fail closed even when slice matches parent lines
    broken_parent = CodeFileRecord(
        file_id=uuid.uuid4(),
        domain_id="code_architecture_scout",
        repo_name="core",
        file_path="src/broken_parent.py",
        git_commit_sha="1234567890abcdef",
        git_blob_sha="abcdef1234567892",
        language="python",
        allowed_scopes=["public_eng"],
        raw_content="class Bad:\n    def f(self):\n        else:\n            return 1\n",
        line_count=4,
    )
    repo.insert_code_file(principal, broken_parent)
    broken_parent_slice = win_slice.model_copy(
        update={
            "file_path": "src/broken_parent.py",
            "start_line": 3,
            "end_line": 4,
            "verbatim_code_with_lines": "   3 |         else:\n   4 |             return 1\n",
        }
    )
    assert (
        validator.check_ast_syntax(
            [broken_parent_slice],
            bound_repo=bound,
            principal=principal,
            domain_id="code_architecture_scout",
        )[0]
        is False
    )

    ts_good = good_slice.model_copy(
        update={
            "file_path": "src/app.ts",
            "verbatim_code_with_lines": (
                "   1 | export function add(a: number, b: number): number { return a + b; }\n"
            ),
        }
    )
    assert validator.check_ast_syntax([ts_good])[0] is True
    ts_bad = good_slice.model_copy(
        update={
            "file_path": "src/app.ts",
            "verbatim_code_with_lines": "   1 | export function add( { return ;\n",
        }
    )
    assert validator.check_ast_syntax([ts_bad])[0] is False

    tsx_good = good_slice.model_copy(
        update={
            "file_path": "src/app.tsx",
            "verbatim_code_with_lines": "   1 | export const Box = () => <div>ok</div>;\n",
        }
    )
    assert validator.check_ast_syntax([tsx_good])[0] is True
    tsx_bad = good_slice.model_copy(
        update={
            "file_path": "src/app.tsx",
            "verbatim_code_with_lines": "   1 | export const Box = () => <div>\n",
        }
    )
    assert validator.check_ast_syntax([tsx_bad])[0] is False

    deep_slice = good_slice.model_copy(
        update={"verbatim_code_with_lines": "   1 | x = " + "(" * 11 + "1" + ")" * 11 + "\n"}
    )
    deep_ok, deep_errs, _ = validator.check_ast_syntax([deep_slice])
    assert deep_ok is False and "Nesting depth" in deep_errs[0]

    partial_ok_slice = good_slice.model_copy(
        update={
            "modification_intent": ModificationIntent.DIRECT_CALLER,
            "verbatim_code_with_lines": "   1 | call(\n   2 |     arg1,\n",
        }
    )
    assert validator.check_ast_syntax([partial_ok_slice])[0] is True
    partial_bad_slice = good_slice.model_copy(
        update={
            "file_path": "src/unindexed_partial.py",
            "qualified_symbol": "Unindexed.call",
            "modification_intent": ModificationIntent.DIRECT_CALLER,
            "verbatim_code_with_lines": "   1 | call(arg1\n",
        }
    )
    assert validator.check_ast_syntax([partial_bad_slice])[0] is False

    # Balanced-delimiter syntax errors reaching Tree-sitter parser checks + unclosed quote
    for lang_ext, bad_code in (
        (".py", "   1 | def broken() : = 1\n"),
        (".ts", "   1 | export function add() { return = ; }\n"),
        (".tsx", "   1 | export const Box = () => { return === ; };\n"),
    ):
        ts_err_slice = good_slice.model_copy(
            update={"file_path": f"src/bad{lang_ext}", "verbatim_code_with_lines": bad_code}
        )
        ok_ts, errs_ts, _ = validator.check_ast_syntax([ts_err_slice])
        assert ok_ts is False and errs_ts

    unclosed_q_slice = good_slice.model_copy(
        update={"qualified_symbol": "", "verbatim_code_with_lines": "   1 | x = 'unclosed\n"}
    )
    ok_q, errs_q, items_q = validator.check_ast_syntax([unclosed_q_slice])
    assert ok_q is False and errs_q and items_q[0].identifier == "src/orders.py"

    # 3. check_principal_scopes (admin, stale chunks, mixed slices, cross-domain lookup)
    other_dom_rec = DocChunkRecord(
        chunk_id=uuid.uuid4(),
        domain_id="other_dom",
        doc_id="cross_dom_doc",
        doc_title="Cross Domain",
        source_uri="https://docs.internal/cross_dom",
        market_scope=None,
        published_date=None,
        authority_weight=1.0,
        section_heading="Cross",
        contextual_header="Cross",
        verbatim_content="Cross-domain verified policy.",
        metadata_json={},
        allowed_scopes=["public_eng"],
        embedding=[0.15] * 768,
        version_str="v1",
        is_stale=False,
    )
    cross_dom_principal = PrincipalContext(
        user_id="pf_cross",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "domain:other_dom"],
    )
    repo.insert_doc_chunk(cross_dom_principal, other_dom_rec)
    other_dom_cit = DocCitationSpan(
        doc_id="cross_dom_doc",
        doc_title="Cross Domain",
        source_uri="https://docs.internal/cross_dom",
        version="v1",
        section_heading="Cross",
        verbatim_quote="Cross-domain verified policy.",
        chunk_id=str(other_dom_rec.chunk_id),
    )

    sc_ok, p_sl, p_ct, ex, _ = validator.check_principal_scopes(
        [good_slice], [good_cit], bound, principal, "code_architecture_scout"
    )
    assert sc_ok is True and len(p_sl) == 1 and len(p_ct) == 1 and ex == []

    # Citations only (empty code_slices) should pass when citations permitted
    sc_cit_only, _, p_ct_only, _, _ = validator.check_principal_scopes(
        [], [good_cit], bound, principal, "code_architecture_scout"
    )
    assert sc_cit_only is True and len(p_ct_only) == 1

    # Stale chunk stays visible for scope pass (freshness gate rejects staleness separately)
    stale_cit_scope = DocCitationSpan(
        doc_id="old_runbook",
        doc_title="Old Runbook",
        source_uri="https://docs.internal/old_runbook",
        version="v0",
        section_heading="Old",
        verbatim_quote="Legacy SLA 500ms.",
        chunk_id=str(stale_rec.chunk_id),
    )
    sc_stale_ok, _, p_stale_ct, _, _ = validator.check_principal_scopes(
        [good_slice],
        [stale_cit_scope],
        bound,
        principal,
        "code_architecture_scout",
        doc_chunks=None,
    )
    assert sc_stale_ok is True and len(p_stale_ct) == 1

    # Cross-domain chunk lookup in check_principal_scopes
    sc_cross_ok, _, p_cross_ct, _, _ = validator.check_principal_scopes(
        [good_slice], [other_dom_cit], bound, cross_dom_principal, "code_architecture_scout"
    )
    assert sc_cross_ok is True and len(p_cross_ct) == 1

    unauth_cit = DocCitationSpan(
        doc_id="secret_doc",
        doc_title="Secret Spec",
        source_uri="https://docs.internal/secret",
        version="v1",
        section_heading="Secret",
        verbatim_quote="Restricted secret spec.",
        chunk_id=str(restricted_rec.chunk_id),
    )

    admin_scope_p = PrincipalContext(
        user_id="pf_admin_scope", role=RoleType.ENGINEER, allowed_scopes=["public_eng", "admin"]
    )
    sc_admin, p_adm, p_adm_c, _, _ = validator.check_principal_scopes(
        [good_slice],
        [unauth_cit],
        bound,
        admin_scope_p,
        "code_architecture_scout",
        doc_chunks=[restricted_rec],
    )
    assert sc_admin is True and len(p_adm) == 1 and len(p_adm_c) == 1

    wildcard_p = PrincipalContext(
        user_id="pf_wild", role=RoleType.ENGINEER, allowed_scopes=["public_eng", "*"]
    )
    sc_wild, p_wild, p_wild_c, _, _ = validator.check_principal_scopes(
        [good_slice],
        [unauth_cit],
        bound,
        wildcard_p,
        "code_architecture_scout",
        doc_chunks=[restricted_rec],
    )
    assert sc_wild is True and len(p_wild) == 1 and len(p_wild_c) == 1

    missing_file_slice = good_slice.model_copy(update={"file_path": "src/other.py"})
    sc_missing_file, p_sl2, _, ex_mf, _ = validator.check_principal_scopes(
        [missing_file_slice], [good_cit], bound, principal, "code_architecture_scout"
    )
    assert sc_missing_file is False and p_sl2 == [] and ex_mf

    # Mixed slices: non-target unauthorized caller excised, valid target kept -> passed is True
    unauth_caller = good_slice.model_copy(
        update={
            "file_path": "src/other.py",
            "modification_intent": ModificationIntent.DIRECT_CALLER,
        }
    )
    sc_mixed_caller, p_sl_mix, _, ex_mix, _ = validator.check_principal_scopes(
        [good_slice, unauth_caller], [], bound, principal, "code_architecture_scout"
    )
    assert sc_mixed_caller is True and len(p_sl_mix) == 1 and len(ex_mix) == 1

    # Single unauthorized caller even with permitted citation -> passed is False (all code excised)
    sc_unauth_caller_with_cit, p_sl_uc, _, _, _ = validator.check_principal_scopes(
        [unauth_caller], [good_cit], bound, principal, "code_architecture_scout"
    )
    assert sc_unauth_caller_with_cit is False and p_sl_uc == []

    # Single unauthorized caller with no citations -> passed is False
    sc_only_unauth_caller, p_sl_u, _, _, _ = validator.check_principal_scopes(
        [unauth_caller], [], bound, principal, "code_architecture_scout"
    )
    assert sc_only_unauth_caller is False and p_sl_u == []

    # Mixed slices: unauthorized MODIFY_TARGET excised even if another slice remains -> False
    unauth_target = good_slice.model_copy(update={"file_path": "src/other.py"})
    sc_mixed_target, p_sl_mt, _, _, _ = validator.check_principal_scopes(
        [good_slice, unauth_target], [], bound, principal, "code_architecture_scout"
    )
    assert sc_mixed_target is False and len(p_sl_mt) == 1

    sc_bad, _, _, ex_bad, _ = validator.check_principal_scopes(
        [good_slice], [unauth_cit], bound, principal, "code_architecture_scout"
    )
    assert sc_bad is False and ex_bad

    sc_unauth_list, _, _, ex_list, _ = validator.check_principal_scopes(
        [good_slice],
        [unauth_cit],
        bound,
        principal,
        "code_architecture_scout",
        doc_chunks=[restricted_rec],
    )
    assert sc_unauth_list is False and ex_list

    # Disjoint-scope file returned by repo stub exercises L4584 ("*" / "admin" checks)
    class _StubBoundRepo:
        def list_visible_code_files(self, _p: object, _d: str) -> list[CodeFileRecord]:
            return [
                CodeFileRecord(
                    file_id=uuid.uuid4(),
                    domain_id="code_architecture_scout",
                    repo_name="core",
                    file_path="src/orders.py",
                    git_commit_sha="1234567890abcdef",
                    git_blob_sha="abcdef1234567890",
                    language="python",
                    allowed_scopes=["exec_only"],
                    raw_content="x = 1\n",
                    line_count=1,
                )
            ]

        def list_visible_doc_chunks(
            self, _p: object, _d: str, include_stale: bool = True
        ) -> list[DocChunkRecord]:
            return []

    stub_repo = _StubBoundRepo()
    assert (
        validator.check_principal_scopes(
            [good_slice], [], stub_repo, principal, "code_architecture_scout"
        )[0]
        is False
    )
    assert (
        validator.check_principal_scopes(
            [good_slice], [], stub_repo, wildcard_p, "code_architecture_scout"
        )[0]
        is True
    )
    assert (
        validator.check_principal_scopes(
            [good_slice], [], stub_repo, admin_scope_p, "code_architecture_scout"
        )[0]
        is True
    )

    # 4. check_cryptographic_provenance (including short <7 blob SHA ljust padding)
    short_blob_file = CodeFileRecord(
        file_id=uuid.uuid4(),
        domain_id="code_architecture_scout",
        repo_name="core",
        file_path="src/short_blob.py",
        git_commit_sha="1234567890abcdef",
        git_blob_sha="abcd",
        language="python",
        allowed_scopes=["public_eng"],
        raw_content="x = 1\n",
        line_count=1,
    )
    repo.insert_code_file(principal, short_blob_file)
    bound_fresh = mod.BoundGraphRepository(repo)
    short_blob_slice = good_slice.model_copy(
        update={"file_path": "src/short_blob.py", "git_blob_sha": "abcd000"}
    )
    assert (
        validator.check_cryptographic_provenance(
            [short_blob_slice], bound_fresh, principal, "code_architecture_scout"
        )[0]
        is True
    )
    pv_ok, pv_errs, _ = validator.check_cryptographic_provenance(
        [good_slice], bound, principal, "code_architecture_scout"
    )
    assert pv_ok is True and pv_errs == []
    for bad_update, expected_substr in (
        ({"git_commit_sha": "placeholder_sha"}, "Invalid commit SHA"),
        ({"git_commit_sha": "abc"}, "Invalid commit SHA"),
        ({"git_blob_sha": "placeholder_blob"}, "Invalid blob SHA"),
        ({"git_blob_sha": "abc"}, "Invalid blob SHA"),
        ({"end_line": 1, "start_line": 3}, "Invalid line bounds"),
        ({"file_path": "src/phantom.py"}, "Phantom file"),
        ({"git_blob_sha": "wrongblob123"}, "Blob SHA mismatch"),
    ):
        bad_pv_slice = good_slice.model_copy(update=bad_update)
        pv_bad, pv_bad_errs, _ = validator.check_cryptographic_provenance(
            [bad_pv_slice], bound, principal, "code_architecture_scout"
        )
        assert pv_bad is False and pv_bad_errs and expected_substr in pv_bad_errs[0]

    # 5. check_document_freshness & check_citations (repo stale & cross-domain lookup)
    assert (
        validator.check_document_freshness([], None, bound, principal, "code_architecture_scout")[0]
        is True
    )
    fr_ok, _, _ = validator.check_document_freshness(
        [good_cit], [c_rec], bound, principal, "code_architecture_scout"
    )
    assert fr_ok is True
    stale_cit = DocCitationSpan(
        doc_id="old_runbook",
        doc_title="Old Runbook",
        source_uri="https://docs.internal/old_runbook",
        version="v0",
        section_heading="Old",
        verbatim_quote="Legacy SLA 500ms.",
        chunk_id=str(stale_rec.chunk_id),
    )
    fr_bad, stale_items, _ = validator.check_document_freshness(
        [stale_cit], [stale_rec], bound, principal, "code_architecture_scout"
    )
    assert fr_bad is False and stale_items
    fr_repo_stale, stale_repo_items, _ = validator.check_document_freshness(
        [stale_cit], None, bound, principal, "code_architecture_scout"
    )
    assert fr_repo_stale is False and stale_repo_items

    neg_res_rec = DocChunkRecord(
        chunk_id=uuid.uuid4(),
        domain_id="code_architecture_scout",
        doc_id="jira_1",
        doc_title="JIRA-1",
        source_uri="https://jira.internal/JIRA-1",
        market_scope=None,
        published_date=None,
        authority_weight=0.5,
        section_heading="Status",
        contextual_header="JIRA-1",
        verbatim_content="Closed ticket.",
        metadata_json={"resolution": "Won't Fix"},
        allowed_scopes=["public_eng"],
        embedding=[0.2] * 768,
        is_stale=False,
    )
    neg_cit = good_cit.model_copy(
        update={
            "doc_id": "jira_1",
            "chunk_id": str(neg_res_rec.chunk_id),
            "verbatim_quote": "Closed ticket.",
        }
    )
    fr_neg, neg_items, _ = validator.check_document_freshness(
        [neg_cit], [neg_res_rec], bound, principal, "code_architecture_scout"
    )
    assert fr_neg is False and neg_items

    assert validator.check_citations([], None, bound, principal, "code_architecture_scout") == (
        True,
        [],
    )
    cit_ok, _ = validator.check_citations(
        [good_cit], [c_rec], bound, principal, "code_architecture_scout"
    )
    assert cit_ok is True
    cit_dict_ok, _ = validator.check_citations(
        [good_cit],
        {str(c_rec.chunk_id): c_rec.verbatim_content},
        bound,
        principal,
        "code_architecture_scout",
    )
    assert cit_dict_ok is True
    cit_cross_list_ok, _ = validator.check_citations(
        [other_dom_cit], [], bound, cross_dom_principal, "code_architecture_scout"
    )
    assert cit_cross_list_ok is True
    cit_cross_dict_ok, _ = validator.check_citations(
        [other_dom_cit], {}, bound, cross_dom_principal, "code_architecture_scout"
    )
    assert cit_cross_dict_ok is True
    cit_repo_ok, _ = validator.check_citations(
        [good_cit], None, bound, principal, "code_architecture_scout"
    )
    assert cit_repo_ok is True
    bad_cit = good_cit.model_copy(update={"verbatim_quote": "Fabricated quote text"})
    cit_bad, cit_errs = validator.check_citations(
        [bad_cit], [c_rec], bound, principal, "code_architecture_scout"
    )
    assert cit_bad is False and cit_errs
    assert "Quote failed verbatim verification:" not in cit_errs[0]

    missing_chunk_cit = good_cit.model_copy(
        update={"doc_id": "unmapped_doc", "chunk_id": str(uuid.uuid4())}
    )
    cit_miss, cit_miss_errs = validator.check_citations(
        [missing_chunk_cit], [c_rec], bound, principal, "code_architecture_scout"
    )
    assert cit_miss is False and cit_miss_errs
    assert "Quote failed verbatim verification:" not in cit_miss_errs[0]

    # 6. partition_prompt_identifiers
    missing_refs, novel_targets = validator.partition_prompt_identifiers(
        "Refactor OrderService.process and add new helper validate_order_payload",
        [sym_rec],
        [f_rec],
        [good_slice],
    )
    assert missing_refs == []
    assert "validate_order_payload" in novel_targets

    missing_refs2, _ = validator.partition_prompt_identifiers(
        "Refactor OrderService.process and inspect MissingLegacyHandler",
        [sym_rec],
        [f_rec],
        [good_slice],
    )
    assert "MissingLegacyHandler" in missing_refs2

    # 7. validate + evaluate_gate end-to-end (including pack fallback branches)
    gate_pass = validator.evaluate_gate(
        code_slices=[good_slice],
        user_prompt="Update OrderService.process",
        supporting_doc_citations=[good_cit],
        normalized_rrf_score=0.90,
        principal=principal,
        domain_id="code_architecture_scout",
    )
    assert gate_pass.passed is True

    sample_pack = ImplementationContextPack(
        trace_id="trace_mut_pack",
        domain_id="code_architecture_scout",
        user_task="Update OrderService.process",
        architectural_execution_plan=["Update OrderService.process"],
        code_slices=[good_slice],
        supporting_doc_citations=[good_cit],
        preflight_report=PreFlightValidationReport(
            passed=True,
            normalized_rrf_score=0.9,
            citations_verified_100_pct=True,
        ),
        local_tokens_consumed=1000,
        pack_token_count=200,
    )
    gate_pack_fail = validator.evaluate_gate(
        pack=sample_pack,
        normalized_rrf_score=0.10,
        principal=principal,
    )
    assert gate_pack_fail.passed is False
    assert gate_pack_fail.diagnostic.trace_id == "trace_mut_pack"
    assert gate_pack_fail.diagnostic.domain_id == "code_architecture_scout"
    assert gate_pack_fail.diagnostic.partial_candidate_files == ["src/orders.py"]

    gate_low_rrf = validator.evaluate_gate(
        code_slices=[good_slice],
        user_prompt="Update OrderService.process",
        supporting_doc_citations=[good_cit],
        normalized_rrf_score=0.20,
        principal=principal,
        domain_id=None,
    )
    assert gate_low_rrf.passed is False
    assert gate_low_rrf.diagnostic.paid_tokens_spent == 0
    assert gate_low_rrf.diagnostic.failure_check == "check_3_rrf"
    assert gate_low_rrf.diagnostic.domain_id == "code_architecture_scout"

    gate_empty = validator.evaluate_gate(
        code_slices=[],
        user_prompt="Update something",
        supporting_doc_citations=[],
        normalized_rrf_score=0.90,
        principal=principal,
        domain_id="code_architecture_scout",
    )
    assert gate_empty.passed is False
    assert gate_empty.diagnostic.paid_tokens_spent == 0
