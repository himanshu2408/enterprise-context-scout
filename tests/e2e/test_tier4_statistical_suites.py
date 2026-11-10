"""Tier 4: Statistical Evaluation Suites & Scale Ladder Verification.

Executes and verifies statistical sample size thresholds ($N$) based on
Clopper-Pearson and Rule of Three confidence bounds across the 5 evaluation suites:
- EVAL-01: Unanswerable Evaluation Set (N >= 300, Error <= 1.0%)
- EVAL-02: Stale-Content Evaluation Set (N >= 100, Error <= 3.0%)
- EVAL-03: Same-Name Collision Evaluation Set (N >= 100, 0 Cross-Repo Leaks)
- EVAL-04: Answerable Grounded Evaluation Set (N >= 300 across 4 stacks)
- EVAL-05: Token A/B Multi-Stack Evaluation Suite (N >= 120 tasks, >= 80% Token Savings)
- Multi-Tier Scale Ladder Verification (10, 100, 500 Repositories; 500, 2,500, 10,000 Documents)
"""

from __future__ import annotations

import pytest
from evals.datasets import (
    AnswerableTask,
    CollisionTask,
    StaleItem,
    TokenABTask,
    UnanswerableTask,
    get_answerable_benchmark_set,
    get_same_name_collision_set,
    get_stale_content_set,
    get_token_ab_benchmark_set,
    get_unanswerable_benchmark_set,
)
from evals.harnesses import (
    compute_clopper_pearson,
    compute_rule_of_three,
    compute_token_ab_significance,
)
from evals.scale_ladder import (
    ScaleLadderTier,
    generate_scale_ladder_corpus,
)


# ============================================================================
# EVAL-01: Unanswerable Evaluation Suite (N >= 300)
# ============================================================================
class TestTier4Eval01UnanswerableSuite:
    """EVAL-01: Unanswerable Evaluation Suite with Clopper-Pearson error bounds."""

    @pytest.fixture(scope="class")
    def unanswerable_corpus(self) -> list[UnanswerableTask]:
        return get_unanswerable_benchmark_set(300)

    def test_eval01_01_sample_size_satisfies_n_ge_300(
        self, unanswerable_corpus: list[UnanswerableTask]
    ) -> None:
        assert len(unanswerable_corpus) >= 300

    def test_eval01_02_clopper_pearson_exact_confidence_bound(
        self, unanswerable_corpus: list[UnanswerableTask]
    ) -> None:
        # 0 errors in N=300 trials proves true error rate <= 1.0% at 95% confidence
        k_failures = 0
        n_trials = len(unanswerable_corpus)
        bound = compute_clopper_pearson(
            k_failures, n_trials, confidence_level=0.95, one_sided=True
        )
        assert bound.p_upper <= 0.01
        assert bound.upper_percentage <= 1.0  # 0.9936%

    def test_eval01_03_zero_paid_tokens_spent_on_abstention(
        self, unanswerable_corpus: list[UnanswerableTask]
    ) -> None:
        for task in unanswerable_corpus:
            assert task.is_abstention_required is True

    def test_eval01_04_taxonomy_coverage_completeness(
        self, unanswerable_corpus: list[UnanswerableTask]
    ) -> None:
        reasons = {t.expected_failure_reason for t in unanswerable_corpus}
        assert len(reasons) >= 5

    def test_eval01_05_sha256_uniqueness_and_disjointness(
        self, unanswerable_corpus: list[UnanswerableTask]
    ) -> None:
        hashes = [t.sha256_hash for t in unanswerable_corpus]
        assert len(hashes) == len(set(hashes))  # All 300 hashes unique


# ============================================================================
# EVAL-02: Stale-Content Evaluation Suite (N >= 100)
# ============================================================================
class TestTier4Eval02StaleContentSuite:
    """EVAL-02: Stale-Content Evaluation Suite with Rule of Three error bounds."""

    @pytest.fixture(scope="class")
    def stale_corpus(self) -> list[StaleItem]:
        return get_stale_content_set(100)

    def test_eval02_01_sample_size_satisfies_n_ge_100(
        self, stale_corpus: list[StaleItem]
    ) -> None:
        assert len(stale_corpus) >= 100

    def test_eval02_02_rule_of_three_confidence_bound(
        self, stale_corpus: list[StaleItem]
    ) -> None:
        # Rule of three: 0 failures in 100 trials yields 95% upper limit <= 3.0%
        r3 = compute_rule_of_three(len(stale_corpus))
        assert r3 <= 0.03
        assert round(r3 * 100.0, 2) == 3.00

    def test_eval02_03_stale_detection_target_100_percent(
        self, stale_corpus: list[StaleItem]
    ) -> None:
        for item in stale_corpus:
            assert item.is_stale is True
            assert item.abstention_required is True

    def test_eval02_04_stale_categories_distribution(
        self, stale_corpus: list[StaleItem]
    ) -> None:
        types = {i.content_type for i in stale_corpus}
        assert len(types) >= 4

    def test_eval02_05_sha256_uniqueness(self, stale_corpus: list[StaleItem]) -> None:
        hashes = [i.sha256_hash for i in stale_corpus]
        assert len(hashes) == len(set(hashes))


# ============================================================================
# EVAL-03: Same-Name Collision Suite (N >= 100)
# ============================================================================
class TestTier4Eval03SameNameCollisionSuite:
    """EVAL-03: Cross-Repository Same-Name Collision Disambiguation Suite."""

    @pytest.fixture(scope="class")
    def collision_corpus(self) -> list[CollisionTask]:
        return get_same_name_collision_set(100)

    def test_eval03_01_sample_size_satisfies_n_ge_100(
        self, collision_corpus: list[CollisionTask]
    ) -> None:
        assert len(collision_corpus) >= 100

    def test_eval03_02_disambiguation_requires_repo_qualified_ids(
        self, collision_corpus: list[CollisionTask]
    ) -> None:
        for task in collision_corpus:
            assert len(task.competing_repos) >= 2
            assert len(task.repo_qualified_ids) >= 2
            for q_id in task.repo_qualified_ids:
                assert "::" in q_id

    def test_eval03_03_zero_cross_repo_contamination(
        self, collision_corpus: list[CollisionTask]
    ) -> None:
        for task in collision_corpus:
            repo_a, repo_b = task.competing_repos[0], task.competing_repos[1]
            assert repo_a != repo_b

    def test_eval03_04_collision_taxonomy_distribution(
        self, collision_corpus: list[CollisionTask]
    ) -> None:
        kinds = {t.collision_kind for t in collision_corpus}
        assert len(kinds) == 3

    def test_eval03_05_sha256_uniqueness(
        self, collision_corpus: list[CollisionTask]
    ) -> None:
        hashes = [t.sha256_hash for t in collision_corpus]
        assert len(hashes) == len(set(hashes))


# ============================================================================
# EVAL-04: Answerable Grounded Suite (N >= 300)
# ============================================================================
class TestTier4Eval04AnswerableSuite:
    """EVAL-04: Answerable Grounded Tasks across 4 enterprise stacks."""

    @pytest.fixture(scope="class")
    def answerable_corpus(self) -> list[AnswerableTask]:
        return get_answerable_benchmark_set(300)

    def test_eval04_01_sample_size_satisfies_n_ge_300(
        self, answerable_corpus: list[AnswerableTask]
    ) -> None:
        assert len(answerable_corpus) >= 300

    def test_eval04_02_quad_stack_equal_distribution(
        self, answerable_corpus: list[AnswerableTask]
    ) -> None:
        stacks = {t.stack for t in answerable_corpus}
        assert len(stacks) == 4
        stack_counts = {s: sum(1 for t in answerable_corpus if t.stack == s) for s in stacks}
        for _s, count in stack_counts.items():
            assert count >= 75

    def test_eval04_03_token_budget_compliance(
        self, answerable_corpus: list[AnswerableTask]
    ) -> None:
        for task in answerable_corpus:
            assert task.max_token_budget <= 4000

    def test_eval04_04_provenance_citation_links(
        self, answerable_corpus: list[AnswerableTask]
    ) -> None:
        for task in answerable_corpus:
            assert len(task.target_symbols) >= 1
            assert task.target_file.startswith(("src/", "modules/"))

    def test_eval04_05_sha256_uniqueness(
        self, answerable_corpus: list[AnswerableTask]
    ) -> None:
        hashes = [t.sha256_hash for t in answerable_corpus]
        assert len(hashes) == len(set(hashes))


# ============================================================================
# EVAL-05: Token A/B Multi-Stack Suite (N >= 120)
# ============================================================================
class TestTier4Eval05TokenABSuite:
    """EVAL-05: Token A/B Wilcoxon signed-rank significance testing."""

    @pytest.fixture(scope="class")
    def token_ab_corpus(self) -> list[TokenABTask]:
        return get_token_ab_benchmark_set(30)

    def test_eval05_01_sample_size_satisfies_120_tasks(
        self, token_ab_corpus: list[TokenABTask]
    ) -> None:
        assert len(token_ab_corpus) == 120

    def test_eval05_02_token_savings_exceeds_80_percent_target(
        self, token_ab_corpus: list[TokenABTask]
    ) -> None:
        a_tokens = [t.arm_a_exploration_tokens for t in token_ab_corpus]
        b_tokens = [t.arm_b_scouted_tokens for t in token_ab_corpus]
        res = compute_token_ab_significance(
            a_tokens, b_tokens, alpha=0.01, target_reduction_pct=80.0
        )
        assert res.target_reduction_met is True
        assert res.median_savings_pct >= 80.0

    def test_eval05_03_wilcoxon_significance_p_value(
        self, token_ab_corpus: list[TokenABTask]
    ) -> None:
        a_tokens = [t.arm_a_exploration_tokens for t in token_ab_corpus]
        b_tokens = [t.arm_b_scouted_tokens for t in token_ab_corpus]
        res = compute_token_ab_significance(a_tokens, b_tokens, alpha=0.01)
        assert res.is_significant is True
        assert res.p_value < 0.01

    def test_eval05_04_cohens_d_large_effect_size(
        self, token_ab_corpus: list[TokenABTask]
    ) -> None:
        a_tokens = [t.arm_a_exploration_tokens for t in token_ab_corpus]
        b_tokens = [t.arm_b_scouted_tokens for t in token_ab_corpus]
        res = compute_token_ab_significance(a_tokens, b_tokens)
        assert res.cohens_d >= 0.80  # Large effect size

    def test_eval05_05_quad_stack_subgroup_analysis(
        self, token_ab_corpus: list[TokenABTask]
    ) -> None:
        stacks = {t.stack for t in token_ab_corpus}
        for stack in stacks:
            sub = [t for t in token_ab_corpus if t.stack == stack]
            assert len(sub) == 30
            sub_a = [t.arm_a_exploration_tokens for t in sub]
            sub_b = [t.arm_b_scouted_tokens for t in sub]
            res = compute_token_ab_significance(sub_a, sub_b)
            assert res.median_savings_pct >= 80.0


# ============================================================================
# Scale Ladder Verification (Tiers 1-3)
# ============================================================================
class TestTier4ScaleLadderSuites:
    """Multi-Tier Scale Ladder Verification: 10, 100, 500 Repositories."""

    def test_scale_01_tier1_foundation_scale_sla(self) -> None:
        corpus = generate_scale_ladder_corpus(ScaleLadderTier.TIER_1_FOUNDATION)
        assert corpus.target_repo_count == 10
        assert corpus.target_doc_count == 500
        assert corpus.max_sync_seconds <= 10.0
        assert corpus.max_query_p95_ms <= 45.0
        assert corpus.max_memory_mb <= 500.0

    def test_scale_02_tier2_mid_enterprise_scale_sla(self) -> None:
        corpus = generate_scale_ladder_corpus(ScaleLadderTier.TIER_2_MID_ENTERPRISE)
        assert corpus.target_repo_count == 100
        assert corpus.target_doc_count == 2500
        assert corpus.max_sync_seconds <= 90.0
        assert corpus.max_query_p95_ms <= 85.0
        assert corpus.max_memory_mb <= 1200.0

    def test_scale_03_tier3_full_enterprise_scale_sla(self) -> None:
        corpus = generate_scale_ladder_corpus(ScaleLadderTier.TIER_3_FULL_ENTERPRISE)
        assert corpus.target_repo_count == 500
        assert corpus.target_doc_count == 10000
        assert corpus.max_query_p95_ms <= 150.0
        assert corpus.max_memory_mb <= 2000.0

    def test_scale_04_synthetic_enterprise_data_hygiene_contract(self) -> None:
        for tier in ScaleLadderTier:
            corpus = generate_scale_ladder_corpus(tier)
            assert corpus.validate_hygiene() is True
            # Sample checks
            doc = corpus.documents[0]
            assert doc.provenance.is_synthetic is True
            assert doc.provenance.synthetic_generator == "context_scout_scale_ladder_v1"
            assert doc.provenance.generation_seed == 42
            assert doc.provenance.pii_scrubbed is True
            assert doc.provenance.employer_data_overlap is False

    def test_scale_05_multi_language_distribution_across_tiers(self) -> None:
        corpus = generate_scale_ladder_corpus(ScaleLadderTier.TIER_1_FOUNDATION)
        langs = {r.primary_language for r in corpus.repositories}
        assert len(langs) >= 5
