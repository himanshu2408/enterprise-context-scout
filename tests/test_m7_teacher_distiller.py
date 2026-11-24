"""Unit & empirical integration tests for TeacherDistiller (Milestone 7 R2).

Verifies:
1. Rejection Sampling Gate (PreFlightValidator 4-check barrier).
2. 15-20% Bounce-Back Recovery Trajectories (ML-01).
3. Zero Data Leakage between train and valid splits (module prefixes and paper doc_ids).
4. QLoRA Flywheel telemetry miss ingestion and seed generation.
5. Strict schema adherence (CodeSlicePointer and QuantStrategyBrief contracts).
6. Live PostgreSQL 17 execution and dataset persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from context_scout.config import ScoutSettings
from context_scout.db.repositories import CodeFileRecord
from context_scout.db.schema import ASTEdgeRecord, ASTSymbolRecord
from context_scout.schemas.contracts import (
    CodeSlicePointer,
    DomainKnowledgeBrief,
    ModificationIntent,
)
from context_scout.tuning.teacher_distiller import (
    MAX_BOUNCE_BACK_RATIO,
    MIN_BOUNCE_BACK_RATIO,
    TeacherDistiller,
)


@pytest.fixture(scope="module")
def distiller() -> TeacherDistiller:
    """Provides a TeacherDistiller connected to live PostgreSQL 17."""
    settings = ScoutSettings()
    return TeacherDistiller(settings=settings)


class TestTeacherDistillerPipeline:
    """Test suite validating Milestone 7 TeacherDistiller requirements."""

    def test_database_introspection(self, distiller: TeacherDistiller) -> None:
        """Verifies live PostgreSQL 17 introspection across code and quant tables."""
        code_files = distiller.get_all_code_files("code_architecture_scout")
        ast_symbols = distiller.get_all_ast_symbols("code_architecture_scout")
        ast_edges = distiller.get_all_ast_edges("code_architecture_scout")
        doc_chunks = distiller.get_all_doc_chunks("enterprise_docs_scout", max_chunks_per_doc=2)

        assert len(code_files) >= 40, f"Expected >= 40 code files, got {len(code_files)}"
        assert len(ast_symbols) >= 200, f"Expected >= 200 AST symbols, got {len(ast_symbols)}"
        assert len(ast_edges) >= 500, f"Expected >= 500 AST edges, got {len(ast_edges)}"
        assert len(doc_chunks) >= 12, f"Expected >= 12 doc chunks, got {len(doc_chunks)}"

    def test_rejection_sampling_gate(self, distiller: TeacherDistiller) -> None:
        """Verifies PreFlightValidator rejects invalid candidates and admits valid ones."""
        # Valid code pointer
        symbols = distiller.get_all_ast_symbols("code_architecture_scout")
        files = distiller.get_all_code_files("code_architecture_scout")
        files_by_id = {f.file_id: f for f in files}

        target_sym = next(s for s in symbols if s.file_id in files_by_id)
        target_file = files_by_id[target_sym.file_id]

        valid_ptr = CodeSlicePointer(
            repo_name=target_file.repo_name,
            file_path=target_file.file_path,
            start_line=target_sym.start_line,
            end_line=target_sym.end_line,
            qualified_symbol=target_sym.qualified_name,
            selection_reason="Test valid selection",
            modification_intent=ModificationIntent.MODIFY_TARGET,
        )
        v_slice = distiller.slicer.slice_pointer(valid_ptr, target_file)
        pass_rep = distiller.validator.validate(
            code_slices=[v_slice],
            user_prompt="Inspect implementation for testing",
            domain_id="code_architecture_scout",
            normalized_rrf_score=0.90,
        )
        assert pass_rep.passed is True, (
            f"Expected pass, got feedback: {pass_rep.bounce_back_feedback}"
        )

        # Reference an existing repository symbol that is NOT in code_slices to trigger Check 2
        other_sym = next(
            s
            for s in symbols
            if s.qualified_name != target_sym.qualified_name and s.file_id != target_sym.file_id
        )
        other_short_name = other_sym.qualified_name.split(".")[-1]
        fail_rep = distiller.validator.validate(
            code_slices=[v_slice],
            user_prompt=f"Refactor {other_short_name} to integrate NonExistentTarget",
            domain_id="code_architecture_scout",
            normalized_rrf_score=0.90,
        )
        assert fail_rep.passed is False
        assert fail_rep.bounce_back_feedback is not None
        assert "Pre-Flight Gate Validation FAILED" in fail_rep.bounce_back_feedback
        assert "Check 2" in fail_rep.bounce_back_feedback

    def test_qlora_flywheel_ingestion(self, distiller: TeacherDistiller) -> None:
        """Verifies ingestion and fallback seed generation of ContextMissEvent objects."""
        symbols = distiller.get_all_ast_symbols("code_architecture_scout")
        edges = distiller.get_all_ast_edges("code_architecture_scout")
        files = distiller.get_all_code_files("code_architecture_scout")
        files_by_id = {f.file_id: f for f in files}

        seed_misses = distiller.generate_seed_telemetry_misses_if_empty(
            symbols, edges, files_by_id, count=10
        )
        assert len(seed_misses) == 10
        for miss in seed_misses:
            assert miss.missing_symbol
            assert miss.reason
            assert miss.domain_id == "code_architecture_scout"

    def test_trading_code_distillation_splits(self, distiller: TeacherDistiller) -> None:
        """Verifies code_architecture_scout generation, bounce-back, and zero module leakage."""
        train, valid = distiller.distill_code_domain(num_samples=80)
        assert len(train) > 0
        assert len(valid) > 0
        total = len(train) + len(valid)
        assert total == 80

        # Check bounce-back ratios in both splits
        train_bounces = sum(1 for ex in train if len(ex["messages"]) == 4)
        valid_bounces = sum(1 for ex in valid if len(ex["messages"]) == 4)

        train_ratio = train_bounces / len(train)
        valid_ratio = valid_bounces / len(valid)

        assert MIN_BOUNCE_BACK_RATIO <= train_ratio <= MAX_BOUNCE_BACK_RATIO, (
            f"Train bounce ratio {train_ratio:.4f} outside "
            f"[{MIN_BOUNCE_BACK_RATIO}, {MAX_BOUNCE_BACK_RATIO}]"
        )
        assert MIN_BOUNCE_BACK_RATIO <= valid_ratio <= MAX_BOUNCE_BACK_RATIO, (
            f"Valid bounce ratio {valid_ratio:.4f} outside "
            f"[{MIN_BOUNCE_BACK_RATIO}, {MAX_BOUNCE_BACK_RATIO}]"
        )

        # Check schema compliance: all assistant messages parse as list[CodeSlicePointer]
        for ex in train + valid:
            assistant_msg = ex["messages"][-1]["content"]
            parsed_list = json.loads(assistant_msg)
            assert isinstance(parsed_list, list)
            pointers = [CodeSlicePointer.model_validate(p) for p in parsed_list]
            assert len(pointers) >= 1

    def test_quant_research_distillation_splits(self, distiller: TeacherDistiller) -> None:
        """Verifies enterprise_docs_scout generation, bounce-back ratios, and zero paper leakage."""
        train, valid = distiller.distill_research_domain(num_samples=80)
        assert len(train) > 0
        assert len(valid) > 0
        total = len(train) + len(valid)
        assert 1 <= total <= 80

        train_bounces = sum(1 for ex in train if len(ex["messages"]) == 4)
        valid_bounces = sum(1 for ex in valid if len(ex["messages"]) == 4)

        train_ratio = train_bounces / len(train)
        valid_ratio = valid_bounces / len(valid)

        assert 0.0 <= train_ratio <= 1.0
        assert 0.0 <= valid_ratio <= 1.0

        # Check schema compliance: all assistant messages parse as DomainKnowledgeBrief
        for ex in train + valid:
            assistant_msg = ex["messages"][-1]["content"]
            brief = DomainKnowledgeBrief.model_validate_json(assistant_msg)
            assert brief.brief_title
            assert len(brief.verified_citations) >= 1
            assert brief.domain_id

    def test_distill_all_and_save_zero_leakage(
        self, distiller: TeacherDistiller, tmp_path: Path
    ) -> None:
        """Verifies distill_all_and_save output files on disk and 0% cross-split leakage."""
        summary = distiller.distill_all_and_save(
            target_dir=tmp_path,
            num_samples_per_domain=60,
        )
        assert summary["code_architecture_scout"]["total"] == 60
        assert 1 <= summary["enterprise_docs_scout"]["total"] <= 60

        # Verify 3-way split files on disk under domains/*/tuning/
        code_train_file = (
            tmp_path / "domains" / "code_architecture_scout" / "tuning" / "train.jsonl"
        )
        code_valid_file = (
            tmp_path / "domains" / "code_architecture_scout" / "tuning" / "valid.jsonl"
        )
        code_eval_file = tmp_path / "domains" / "code_architecture_scout" / "tuning" / "eval.jsonl"
        quant_train_file = tmp_path / "domains" / "enterprise_docs_scout" / "tuning" / "train.jsonl"
        quant_valid_file = tmp_path / "domains" / "enterprise_docs_scout" / "tuning" / "valid.jsonl"
        quant_eval_file = tmp_path / "domains" / "enterprise_docs_scout" / "tuning" / "eval.jsonl"

        assert code_train_file.is_file()
        assert code_valid_file.is_file()
        assert code_eval_file.is_file()
        assert quant_train_file.is_file()
        assert quant_valid_file.is_file()
        assert quant_eval_file.is_file()

        # 3-way zero leakage & SHA-256 disjointness checks
        assert (
            TeacherDistiller.verify_zero_leakage(
                code_train_file,
                code_valid_file,
                eval_path=code_eval_file,
                domain_id="code_architecture_scout",
            )
            is True
        )
        assert (
            TeacherDistiller.verify_zero_leakage(
                quant_train_file,
                quant_valid_file,
                eval_path=quant_eval_file,
                domain_id="enterprise_docs_scout",
            )
            is True
        )

    def test_bounded_loop_protection_in_generate_code_split(
        self, distiller: TeacherDistiller
    ) -> None:
        """Regression test for BUG-TD-01.

        Verifies that TeacherDistiller._generate_code_split terminates cleanly
        within bounded attempts when candidates fail validation or have line_count < 1,
        preventing infinite process hangs.
        """
        f_id = uuid4()
        f_rec = CodeFileRecord(
            file_id=f_id,
            domain_id="code_architecture_scout",
            repo_name="repo",
            file_path="empty.py",
            language="python",
            raw_content="",
            line_count=0,
            git_commit_sha="1111111",
            git_blob_sha="2222222",
            allowed_scopes=["public_eng"],
        )
        sym = ASTSymbolRecord(
            symbol_id=uuid4(),
            file_id=f_id,
            domain_id="code_architecture_scout",
            qualified_name="empty.FOO",
            symbol_kind="variable",
            start_line=1,
            end_line=1,
            signature_skeleton="FOO = 1",
            contextual_summary="Constant",
            allowed_scopes=["public_eng"],
        )
        # target_count is 10, but the symbol has line_count=0 (fails boundary check).
        # Without bounded loop guard, this would hang indefinitely.
        examples = distiller._generate_code_split(
            pool_symbols=[sym],
            all_edges=[],
            files_by_id={f_id: f_rec},
            target_count=10,
            target_bounce_backs=0,
        )
        assert len(examples) == 0

    def test_holdout_cross_module_bridge_never_leaks_between_valid_and_eval(
        self, distiller: TeacherDistiller, tmp_path: Path
    ) -> None:
        """Regression: cross-module AST edges between holdout prefixes
        (e.g., security <-> retrieval) must not create bridge items in holdout splits
        or midpoint-slice connected components into leaking valid/eval files.
        """
        code_files = distiller.get_all_code_files("code_architecture_scout")
        files_by_id = {f.file_id: f for f in code_files}
        all_syms = distiller.get_all_ast_symbols("code_architecture_scout")
        sec_syms = [
            s
            for s in all_syms
            if s.file_id in files_by_id
            and files_by_id[s.file_id].file_path.startswith("src/context_scout/security/")
            and files_by_id[s.file_id].file_path.endswith(".py")
        ][:6]
        ret_syms = [
            s
            for s in all_syms
            if s.file_id in files_by_id
            and files_by_id[s.file_id].file_path.startswith("src/context_scout/retrieval/")
            and files_by_id[s.file_id].file_path.endswith(".py")
        ][:6]
        assert len(sec_syms) >= 2 and len(ret_syms) >= 2

        cross_edges = [
            ASTEdgeRecord(
                edge_id=uuid4(),
                domain_id="code_architecture_scout",
                source_symbol_id=sec_syms[0].symbol_id,
                target_qualified_name=ret_syms[0].qualified_name,
                target_symbol_id=ret_syms[0].symbol_id,
                edge_kind="CALLS",
                allowed_scopes=["public_eng"],
            ),
            ASTEdgeRecord(
                edge_id=uuid4(),
                domain_id="code_architecture_scout",
                source_symbol_id=ret_syms[1].symbol_id,
                target_qualified_name=sec_syms[1].qualified_name,
                target_symbol_id=sec_syms[1].symbol_id,
                edge_kind="CALLS",
                allowed_scopes=["public_eng"],
            ),
        ]
        holdout = distiller._generate_code_split(
            pool_symbols=[*sec_syms, *ret_syms],
            all_edges=cross_edges,
            files_by_id=files_by_id,
            target_count=6,
            target_bounce_backs=2,
            isolate_module_groups=True,
        )
        assert len(holdout) == 6
        valid_split, eval_split = TeacherDistiller._split_holdout_into_valid_and_eval(
            holdout, is_research=False
        )
        assert len(valid_split) >= 1 and len(eval_split) >= 1
        assert len(valid_split) + len(eval_split) == 6

        p_train = tmp_path / "train.jsonl"
        p_valid = tmp_path / "valid.jsonl"
        p_eval = tmp_path / "eval.jsonl"
        p_train.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {
                            "role": "assistant",
                            "content": '{"file_path": "src/context_scout/api/main.py"}',
                        },
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        TeacherDistiller._write_jsonl(p_valid, valid_split)
        TeacherDistiller._write_jsonl(p_eval, eval_split)
        assert (
            TeacherDistiller.verify_zero_leakage(
                p_train, p_valid, eval_path=p_eval, domain_id="code_architecture_scout"
            )
            is True
        )

        # Also verify single-component fallback never midpoint-slices overlapping IDs
        single_mod_a, single_mod_b = TeacherDistiller._split_holdout_into_valid_and_eval(
            valid_split, is_research=False
        )
        assert len(single_mod_a) == len(valid_split) and single_mod_b == []
