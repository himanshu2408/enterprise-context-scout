"""Evaluation datasets for statistical benchmarking (Tiers 1-4)."""

from context_scout.evals.datasets.answerable_dataset import (
    AnswerableTask,
    get_answerable_benchmark_set,
)
from context_scout.evals.datasets.same_name_collision_dataset import (
    CollisionTask,
    get_same_name_collision_set,
)
from context_scout.evals.datasets.stale_content_dataset import (
    StaleItem,
    get_stale_content_set,
)
from context_scout.evals.datasets.token_ab_dataset import (
    TokenABTask,
    get_token_ab_benchmark_set,
)
from context_scout.evals.datasets.unanswerable_dataset import (
    UnanswerableTask,
    get_unanswerable_benchmark_set,
)

__all__ = [
    "AnswerableTask",
    "CollisionTask",
    "StaleItem",
    "TokenABTask",
    "UnanswerableTask",
    "get_answerable_benchmark_set",
    "get_same_name_collision_set",
    "get_stale_content_set",
    "get_token_ab_benchmark_set",
    "get_unanswerable_benchmark_set",
]
