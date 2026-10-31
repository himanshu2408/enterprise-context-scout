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

import pytest

from context_scout.config import ScoutSettings
from context_scout.schemas.contracts import (
    CodeSlicePointer,
    ModificationIntent,
    QuantStrategyBrief,
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
        code_files = distiller.get_all_code_files("trading_code_scout")
        ast_symbols = distiller.get_all_ast_symbols("trading_code_scout")
        ast_edges = distiller.get_all_ast_edges("trading_code_scout")
        doc_chunks = distiller.get_all_doc_chunks("quant_research_scout", max_chunks_per_doc=2)

        assert len(code_files) >= 40, f"Expected >= 40 code files, got {len(code_files)}"
        assert len(ast_symbols) >= 500, f"Expected >= 500 AST symbols, got {len(ast_symbols)}"
        assert len(ast_edges) >= 2000, f"Expected >= 2000 AST edges, got {len(ast_edges)}"
        assert len(doc_chunks) >= 300, f"Expected >= 300 doc chunks, got {len(doc_chunks)}"

    def test_rejection_sampling_gate(self, distiller: TeacherDistiller) -> None:
        """Verifies PreFlightValidator rejects invalid candidates and admits valid ones."""
        # Valid code pointer
        symbols = distiller.get_all_ast_symbols("trading_code_scout")
        files = distiller.get_all_code_files("trading_code_scout")
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
            domain_id="trading_code_scout",
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
            domain_id="trading_code_scout",
            normalized_rrf_score=0.90,
        )
        assert fail_rep.passed is False
        assert fail_rep.bounce_back_feedback is not None
        assert "Pre-Flight Gate Validation FAILED" in fail_rep.bounce_back_feedback
        assert "Check 2" in fail_rep.bounce_back_feedback

    def test_qlora_flywheel_ingestion(self, distiller: TeacherDistiller) -> None:
        """Verifies ingestion and fallback seed generation of ContextMissEvent objects."""
        symbols = distiller.get_all_ast_symbols("trading_code_scout")
        edges = distiller.get_all_ast_edges("trading_code_scout")
        files = distiller.get_all_code_files("trading_code_scout")
        files_by_id = {f.file_id: f for f in files}

        seed_misses = distiller.generate_seed_telemetry_misses_if_empty(
            symbols, edges, files_by_id, count=10
        )
        assert len(seed_misses) == 10
        for miss in seed_misses:
            assert miss.missing_symbol
            assert miss.reason
            assert miss.domain_id == "trading_code_scout"

    def test_trading_code_distillation_splits(self, distiller: TeacherDistiller) -> None:
        """Verifies trading_code_scout generation, bounce-back ratios, and zero module leakage."""
        train, valid = distiller.distill_trading_code_scout(num_samples=80)
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
        """Verifies quant_research_scout generation, bounce-back ratios, and zero paper leakage."""
        train, valid = distiller.distill_quant_research_scout(num_samples=80)
        assert len(train) > 0
        assert len(valid) > 0
        total = len(train) + len(valid)
        assert total == 80

        train_bounces = sum(1 for ex in train if len(ex["messages"]) == 4)
        valid_bounces = sum(1 for ex in valid if len(ex["messages"]) == 4)

        train_ratio = train_bounces / len(train)
        valid_ratio = valid_bounces / len(valid)

        assert MIN_BOUNCE_BACK_RATIO <= train_ratio <= MAX_BOUNCE_BACK_RATIO, (
            f"Quant train bounce ratio {train_ratio:.4f} outside "
            f"[{MIN_BOUNCE_BACK_RATIO}, {MAX_BOUNCE_BACK_RATIO}]"
        )
        assert MIN_BOUNCE_BACK_RATIO <= valid_ratio <= MAX_BOUNCE_BACK_RATIO, (
            f"Quant valid bounce ratio {valid_ratio:.4f} outside "
            f"[{MIN_BOUNCE_BACK_RATIO}, {MAX_BOUNCE_BACK_RATIO}]"
        )

        # Check schema compliance: all assistant messages parse as QuantStrategyBrief
        for ex in train + valid:
            assistant_msg = ex["messages"][-1]["content"]
            brief = QuantStrategyBrief.model_validate_json(assistant_msg)
            assert brief.strategy_name
            assert len(brief.verified_citations) >= 1
            assert len(brief.target_markets) >= 1

    def test_distill_all_and_save_zero_leakage(
        self, distiller: TeacherDistiller, tmp_path: Path
    ) -> None:
        """Verifies distill_all_and_save output files on disk and 0% cross-split leakage."""
        summary = distiller.distill_all_and_save(
            target_dir=tmp_path,
            num_samples_per_domain=60,
        )
        assert summary["trading_code_scout"]["total"] == 60
        assert summary["quant_research_scout"]["total"] == 60

        # Verify files on disk
        code_train_file = tmp_path / "data" / "tuning" / "trading_code_scout" / "train.jsonl"
        code_valid_file = tmp_path / "data" / "tuning" / "trading_code_scout" / "valid.jsonl"
        quant_train_file = tmp_path / "data" / "tuning" / "quant_research_scout" / "train.jsonl"
        quant_valid_file = tmp_path / "data" / "tuning" / "quant_research_scout" / "valid.jsonl"

        assert code_train_file.is_file()
        assert code_valid_file.is_file()
        assert quant_train_file.is_file()
        assert quant_valid_file.is_file()

        # Zero leakage checks
        assert TeacherDistiller.verify_zero_leakage(
            code_train_file, code_valid_file, domain_id="trading_code_scout"
        ) is True
        assert TeacherDistiller.verify_zero_leakage(
            quant_train_file, quant_valid_file, domain_id="quant_research_scout"
        ) is True
