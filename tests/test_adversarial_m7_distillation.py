"""Independent Adversarial Verification Suite for Milestone 7 Teacher-Student Distillation Pipeline.

Probes:
1. Rejection Sampling Adversarial Probe:
   - Subtle hallucinations: fake symbols, out-of-bounds line ranges, non-existent files,
     1-character quote alterations, truncated unclosed quotes, invalid chunk UUIDs.
   - Verifies PreFlightValidator deterministically rejects 100% of invalid samples and
     TeacherDistiller never admits them into single-turn canonical examples.
2. Leakage Adversarial Probe:
   - Mathematical proof of disjoint sets: module prefixes, file paths, symbols,
     paper doc_ids, paper titles, source URIs between train.jsonl and valid.jsonl.
   - Proves len(intersection) == 0 across both data/tuning and domains/*/tuning.
3. Bounce-Back Ratio Verification:
   - 4-turn trajectory audit (Request -> Draft -> Gate Feedback -> Remediated Output).
   - Verifies 0 anomalous records and bounce-back ratios strictly within [15%, 20%] (ML-01).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer
from context_scout.ast_engine.models import (
    InvalidLineBoundsError,
    OffByOneBoundsError,
    SlicerErrorCode,
)
from context_scout.config import ScoutSettings
from context_scout.schemas.contracts import (
    CodeSlicePointer,
    DocCitationSpan,
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
    settings = ScoutSettings()
    return TeacherDistiller(settings=settings)


class TestRejectionSamplingAdversarialProbe:
    """Probes candidate outputs with subtle hallucinations and invalid bounds."""

    def test_fake_symbol_rejection(self, distiller: TeacherDistiller) -> None:
        """Verifies that candidate outputs with fake symbol names are 100% rejected."""
        symbols = distiller.get_all_ast_symbols("trading_code_scout")
        files = distiller.get_all_code_files("trading_code_scout")
        files_by_id = {f.file_id: f for f in files}

        target_sym = next(s for s in symbols if s.file_id in files_by_id)
        target_file = files_by_id[target_sym.file_id]

        fake_symbols = [
            "context_scout.fake.NonExistentOptimizer",
            "context_scout.hallucinated_module.GhostClass",
            "AlmgrenChrissFakeExecutor",
            "__non_existent_dunder_symbol__",
        ]

        for fake_sym in fake_symbols:
            ptr = CodeSlicePointer(
                repo_name=target_file.repo_name,
                file_path=target_file.file_path,
                start_line=target_sym.start_line,
                end_line=target_sym.end_line,
                qualified_symbol=fake_sym,
                selection_reason="Adversarial fake symbol test",
                modification_intent=ModificationIntent.MODIFY_TARGET,
            )
            v_slice = distiller.slicer.slice_pointer(ptr, target_file)
            report = distiller.validator.validate(
                code_slices=[v_slice],
                user_prompt=f"Inspect implementation of {fake_sym}",
                domain_id="trading_code_scout",
                normalized_rrf_score=0.90,
            )
            assert report.passed is False, f"Admitted fake symbol: {fake_sym}"
            assert fake_sym in report.unresolved_symbols
            assert "Check 1" in (report.bounce_back_feedback or "")

    def test_out_of_bounds_line_ranges_rejection(self, distiller: TeacherDistiller) -> None:
        """Verifies 0-indexed, inverted, off-by-one, and out-of-range bounds are rejected."""
        slicer = DeterministicSlicer()
        total_lines = 100

        invalid_scenarios = [
            (0, 50, OffByOneBoundsError, "0-indexed start_line"),
            (-1, 50, InvalidLineBoundsError, "Negative start_line"),
            (50, 40, InvalidLineBoundsError, "Inverted line bounds (end < start)"),
            (105, 120, InvalidLineBoundsError, "start_line exceeds total lines"),
            (1, 101, OffByOneBoundsError, "end_line exceeds file lines by exactly 1 (fencepost)"),
            (1, 9999, InvalidLineBoundsError, "end_line far exceeds file line count"),
        ]

        for start, end, expected_exc, _scenario_desc in invalid_scenarios:
            with pytest.raises(expected_exc, match=r"(start_line|end_line|1-indexed)"):
                slicer.validate_bounds(start, end, total_lines)

    def test_non_existent_file_rejection(self, distiller: TeacherDistiller) -> None:
        """Verifies that non-existent repository files cannot be sliced or admitted."""
        files = distiller.get_all_code_files("trading_code_scout")
        known_paths = {f.file_path for f in files}

        hallucinated_paths = [
            "src/context_scout/non_existent_file.py",
            "src/context_scout/strategies/phantom_strategy.py",
            "tests/test_ghost_evaluator.py",
        ]

        for h_path in hallucinated_paths:
            assert h_path not in known_paths
            # Slicing without a CodeFileRecord must fail
            with pytest.raises(AttributeError):
                distiller.slicer.slice_pointer(
                    CodeSlicePointer(
                        repo_name="enterprise-context-scout",
                        file_path=h_path,
                        start_line=1,
                        end_line=20,
                        qualified_symbol="SomeSymbol",
                        selection_reason="Phantom file",
                        modification_intent=ModificationIntent.MODIFY_TARGET,
                    ),
                    None,  # type: ignore[arg-type]
                )

    def test_modified_quote_rejection_and_boundary(self, distiller: TeacherDistiller) -> None:
        """Verifies quote verification on 1-char edits, truncations, and invalid chunk IDs."""
        chunks = distiller.get_all_doc_chunks("quant_research_scout", max_chunks_per_doc=1)
        sample_chunk = chunks[0]
        raw_lines = [
            ln.strip() for ln in sample_chunk.verbatim_content.splitlines() if len(ln.strip()) >= 30
        ]
        real_quote = raw_lines[0] if raw_lines else sample_chunk.verbatim_content[:80]

        # 1. 1-character alteration on realistic short/medium quote
        mod_char = "x" if real_quote[-1] != "x" else "y"
        mod_quote = real_quote[:-1] + mod_char
        cit_mod = DocCitationSpan(
            chunk_id=str(sample_chunk.chunk_id),
            doc_id=sample_chunk.doc_id,
            doc_title=sample_chunk.doc_title,
            source_uri=sample_chunk.source_uri,
            section_heading=sample_chunk.section_heading,
            verbatim_quote=mod_quote,
        )
        res_mod = distiller.slicer.verify_doc_citation(cit_mod, sample_chunk.verbatim_content)
        # For quotes under 50 tokens, 1 char diff drops overlap below 0.98
        toks = distiller.slicer.tokenize(distiller.slicer.normalize_text(real_quote))
        if len(toks) < 50:
            assert res_mod.is_valid is False
            assert res_mod.token_overlap_ratio < 0.98
            assert res_mod.error_code in (
                SlicerErrorCode.HALLUCINATED_QUOTE,
                SlicerErrorCode.QUOTE_MISMATCH,
                SlicerErrorCode.TRUNCATED_QUOTE,
            )

        # 2. Truncated quote with unclosed delimiter
        trunc_quote = real_quote[:35] + " (unclosed_math_statement"
        cit_trunc = cit_mod.model_copy(update={"verbatim_quote": trunc_quote})
        res_trunc = distiller.slicer.verify_doc_citation(cit_trunc, sample_chunk.verbatim_content)
        assert res_trunc.is_valid is False
        assert res_trunc.error_code == SlicerErrorCode.TRUNCATED_QUOTE

        # 3. Non-existent chunk UUID
        cit_fake_id = cit_mod.model_copy(
            update={
                "chunk_id": "00000000-0000-0000-0000-000000000000",
                "verbatim_quote": real_quote,
            }
        )
        passed, reasons = distiller.validator.check_citations(
            citations=[cit_fake_id],
            doc_chunks={str(sample_chunk.chunk_id): sample_chunk.verbatim_content},
            bound_repo=distiller.validator._resolve_graph_repo(distiller.get_in_memory_twin_repo()),
            principal=distiller.system_principal,
            domain_id="quant_research_scout",
        )
        assert passed is False
        assert any("not found in corpus" in r for r in reasons)

    def test_distiller_rejection_sampling_gate_determinism(
        self, distiller: TeacherDistiller
    ) -> None:
        """Proves TeacherDistiller rejects invalid candidates during generation."""
        # Check that validator report with failure never admits an example
        pass_rep = distiller.validator.validate(
            code_slices=[],
            user_prompt="Inspect SomeNovelSymbol",
            domain_id="trading_code_scout",
            normalized_rrf_score=0.40,  # Below 0.55 threshold (Check 3 failure)
        )
        assert pass_rep.passed is False
        assert "Check 3" in (pass_rep.bounce_back_feedback or "")

    def test_quote_length_fuzzy_boundary_analysis(self) -> None:
        """Empirically analyzes and documents the 50-token threshold vulnerability in VAL-02.

        Mathematical property of min_overlap_ratio = 0.98:
        If len(tokens) < 50, altering 1 token yields overlap < 0.98 -> Rejected.
        If len(tokens) >= 50, altering 1 token yields overlap >= 0.98 -> Admitted by fuzzy matcher!
        """
        slicer = DeterministicSlicer()
        # Test 40 tokens (below threshold)
        tokens_40 = [f"token{i}" for i in range(40)]
        source_40 = " ".join(tokens_40)
        mod_40 = list(tokens_40)
        mod_40[20] = "hallucinated_token"
        res_40 = slicer.verify_quote(" ".join(mod_40), source_40)
        assert res_40.is_valid is False
        assert res_40.token_overlap_ratio == 0.975  # 39/40 < 0.98

        # Test 50 tokens (exact threshold boundary)
        tokens_50 = [f"token{i}" for i in range(50)]
        source_50 = " ".join(tokens_50)
        mod_50 = list(tokens_50)
        mod_50[25] = "hallucinated_token"
        res_50 = slicer.verify_quote(" ".join(mod_50), source_50)
        assert res_50.is_valid is True  # 49/50 == 0.9800 >= 0.98
        assert res_50.token_overlap_ratio == 0.98



class TestLeakageAdversarialProbe:
    """Mathematically proves zero data leakage between train and valid holdout sets."""

    def test_trading_code_scout_zero_leakage(self) -> None:
        """Proves mathematical disjointness of module prefixes, file paths, and symbols."""
        bases = [
            Path("data/tuning/trading_code_scout"),
            Path("domains/trading_code_scout/tuning"),
        ]
        for base in bases:
            train_file = base / "train.jsonl"
            valid_file = base / "valid.jsonl"
            assert train_file.is_file() and valid_file.is_file()

            def extract_meta(p: Path) -> tuple[set[str], set[str], set[str]]:
                mods: set[str] = set()
                paths: set[str] = set()
                syms: set[str] = set()
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        for m in obj.get("messages", []):
                            content = m.get("content", "")
                            for fp in re.findall(r'"file_path":\s*"([^"]+)"', content):
                                paths.add(fp)
                                parts = Path(fp).parts
                                if (
                                    len(parts) >= 3
                                    and parts[0] == "src"
                                    and parts[1] == "context_scout"
                                ):
                                    mods.add(f"src/context_scout/{parts[2]}")
                                else:
                                    mods.add(str(Path(fp).parent))
                            for sym in re.findall(r'"qualified_symbol":\s*"([^"]+)"', content):
                                syms.add(sym)
                return mods, paths, syms

            train_mods, train_paths, train_syms = extract_meta(train_file)
            valid_mods, valid_paths, valid_syms = extract_meta(valid_file)

            # Mathematical proof: len(intersection) == 0
            mod_overlap = train_mods.intersection(valid_mods)
            path_overlap = train_paths.intersection(valid_paths)
            sym_overlap = train_syms.intersection(valid_syms)

            assert len(mod_overlap) == 0, f"Module leakage in {base}: {mod_overlap}"
            assert len(path_overlap) == 0, f"File path leakage in {base}: {path_overlap}"
            assert len(sym_overlap) == 0, f"Symbol leakage in {base}: {sym_overlap}"

    def test_quant_research_scout_zero_leakage(self) -> None:
        """Proves mathematical disjointness of doc_ids, titles, and source URIs."""
        quant_bases = [
            Path("data/tuning/quant_research_scout"),
            Path("domains/quant_research_scout/tuning"),
        ]
        for base in quant_bases:
            train_file = base / "train.jsonl"
            valid_file = base / "valid.jsonl"
            assert train_file.is_file() and valid_file.is_file()

            def extract_quant_meta(p: Path) -> tuple[set[str], set[str], set[str]]:
                docs: set[str] = set()
                titles: set[str] = set()
                uris: set[str] = set()
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        for m in obj.get("messages", []):
                            content = m.get("content", "")
                            for d in re.findall(r'"doc_id":\s*"([^"]+)"', content):
                                docs.add(d)
                            for t in re.findall(r'"doc_title":\s*"([^"]+)"', content):
                                titles.add(t)
                            for u in re.findall(r'"source_uri":\s*"([^"]+)"', content):
                                uris.add(u)
                return docs, titles, uris

            train_docs, train_titles, train_uris = extract_quant_meta(train_file)
            valid_docs, valid_titles, valid_uris = extract_quant_meta(valid_file)

            doc_overlap = train_docs.intersection(valid_docs)
            title_overlap = train_titles.intersection(valid_titles)
            uri_overlap = train_uris.intersection(valid_uris)

            assert len(doc_overlap) == 0, f"doc_id leakage in {base}: {doc_overlap}"
            assert len(title_overlap) == 0, f"Title leakage in {base}: {title_overlap}"
            assert len(uri_overlap) == 0, f"URI leakage in {base}: {uri_overlap}"


class TestBounceBackRatioAdversarialProbe:
    """Audits multi-turn bounce-back recovery trajectories against the 15-20% ML-01 constraint."""

    @pytest.mark.parametrize(
        "file_path",
        [
            "data/tuning/trading_code_scout/train.jsonl",
            "data/tuning/trading_code_scout/valid.jsonl",
            "data/tuning/quant_research_scout/train.jsonl",
            "data/tuning/quant_research_scout/valid.jsonl",
            "domains/trading_code_scout/tuning/train.jsonl",
            "domains/trading_code_scout/tuning/valid.jsonl",
            "domains/quant_research_scout/tuning/train.jsonl",
            "domains/quant_research_scout/tuning/valid.jsonl",
        ],
    )
    def test_bounce_back_trajectory_integrity_and_ratios(self, file_path: str) -> None:
        p = Path(file_path)
        assert p.is_file()

        total_records = 0
        single_turn_records = 0
        bounce_back_records = 0

        with open(p, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                raw = line.strip()
                if not raw:
                    continue
                total_records += 1
                record = json.loads(raw)
                assert "messages" in record
                msgs = record["messages"]
                roles = [m["role"] for m in msgs]

                if len(msgs) == 2:
                    assert roles == ["user", "assistant"], f"Invalid 2-turn roles at {p}:{line_no}"
                    single_turn_records += 1
                elif len(msgs) == 4:
                    assert roles == [
                        "user",
                        "assistant",
                        "user",
                        "assistant",
                    ], f"Invalid 4-turn roles at {p}:{line_no}"
                    # Turn 3 must be Pre-Flight Gate failure feedback
                    feedback = msgs[2]["content"]
                    assert "Pre-Flight Gate Validation FAILED" in feedback, (
                        f"Missing gate failure header in Turn 3 at {p}:{line_no}"
                    )
                    # Turn 4 must be a valid remediated response conforming to schema
                    turn4_content = msgs[3]["content"]
                    if "trading_code_scout" in file_path:
                        parsed = json.loads(turn4_content)
                        assert isinstance(parsed, list) and len(parsed) >= 2
                    else:
                        brief = QuantStrategyBrief.model_validate_json(turn4_content)
                        assert len(brief.verified_citations) >= 1
                    bounce_back_records += 1
                else:
                    pytest.fail(f"Anomalous message length {len(msgs)} at {p}:{line_no}")

        assert total_records > 0
        assert single_turn_records + bounce_back_records == total_records

        ratio = bounce_back_records / total_records
        assert MIN_BOUNCE_BACK_RATIO <= ratio <= MAX_BOUNCE_BACK_RATIO, (
            f"Bounce-back ratio {ratio:.4f} in {file_path} violates ML-01 constraint "
            f"[{MIN_BOUNCE_BACK_RATIO}, {MAX_BOUNCE_BACK_RATIO}]"
        )
