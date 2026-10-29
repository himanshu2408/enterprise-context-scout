"""Evaluation suite, golden datasets, and CI/CD quality gate harness."""

from context_scout.evals.golden_dataset import (
    GoldenBenchmarkSuite,
    GoldenCodeTask,
    GoldenFederatedTask,
    GoldenQuantTask,
    PoisonedInjectionPayload,
    RedTeamRBACQuery,
    TaskCategory,
    get_default_golden_suite,
)
from context_scout.evals.harness import (
    EvaluationGateThresholds,
    EvaluationHarness,
    EvaluationRunResult,
    MetricResult,
)
from context_scout.evals.runner import generate_markdown_report, main

__all__ = [
    "EvaluationGateThresholds",
    "EvaluationHarness",
    "EvaluationRunResult",
    "GoldenBenchmarkSuite",
    "GoldenCodeTask",
    "GoldenFederatedTask",
    "GoldenQuantTask",
    "MetricResult",
    "PoisonedInjectionPayload",
    "RedTeamRBACQuery",
    "TaskCategory",
    "generate_markdown_report",
    "get_default_golden_suite",
    "main",
]
