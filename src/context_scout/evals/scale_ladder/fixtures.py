"""Multi-Tier Scale Ladder Corpus Fixtures and Synthetic Enterprise Data Generators.

Adheres strictly to the Synthetic Enterprise Data Hygiene Contract:
- is_synthetic: True
- synthetic_generator: "context_scout_scale_ladder_v1"
- generation_seed: 42
- pii_scrubbed: True
- employer_data_overlap: False
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_GENERATION_SEED: Final[int] = 42
DEFAULT_GENERATOR_NAME: Final[str] = "context_scout_scale_ladder_v1"


class ScaleLadderTier(IntEnum):
    TIER_1_FOUNDATION = 1  # 10 Repos, 500 Docs
    TIER_2_MID_ENTERPRISE = 2  # 100 Repos, 2,500 Docs
    TIER_3_FULL_ENTERPRISE = 3  # 500 Repos, 10,000 Docs


class SyntheticProvenance(BaseModel):
    """Hygiene metadata attached to all synthetic artifacts."""

    model_config = ConfigDict(frozen=True)

    is_synthetic: bool = True
    synthetic_generator: str = DEFAULT_GENERATOR_NAME
    generation_seed: int = DEFAULT_GENERATION_SEED
    pii_scrubbed: bool = True
    employer_data_overlap: bool = False
    corpus_tier: int = Field(ge=1, le=3)


class SyntheticRepoMeta(BaseModel):
    repo_id: str
    repo_name: str
    primary_language: str
    estimated_files: int
    provenance: SyntheticProvenance


class SyntheticDocMeta(BaseModel):
    doc_id: str
    doc_type: str  # confluence, jira, rfc
    title: str
    verbatim_text: str
    provenance: SyntheticProvenance


class ScaleLadderCorpus(BaseModel):
    """Container for multi-tier scale ladder datasets and performance SLAs."""

    tier: ScaleLadderTier
    target_repo_count: int
    target_doc_count: int
    max_sync_seconds: float
    max_query_p95_ms: float
    max_memory_mb: float
    repositories: list[SyntheticRepoMeta]
    documents: list[SyntheticDocMeta]

    def validate_hygiene(self) -> bool:
        """Verifies that all documents in the corpus adhere to synthetic hygiene standards."""
        for doc in self.documents:
            if not doc.provenance.is_synthetic or doc.provenance.employer_data_overlap:
                return False
        return True


def generate_scale_ladder_corpus(
    tier: ScaleLadderTier,
    seed: int = DEFAULT_GENERATION_SEED,
) -> ScaleLadderCorpus:
    """Generates synthetic scale ladder corpora conforming to the enterprise scale matrix."""
    if tier == ScaleLadderTier.TIER_1_FOUNDATION:
        repo_count, doc_count = 10, 500
        sync_sla, p95_sla, mem_sla = 10.0, 45.0, 500.0
    elif tier == ScaleLadderTier.TIER_2_MID_ENTERPRISE:
        repo_count, doc_count = 100, 2500
        sync_sla, p95_sla, mem_sla = 90.0, 85.0, 1200.0
    elif tier == ScaleLadderTier.TIER_3_FULL_ENTERPRISE:
        repo_count, doc_count = 500, 10000
        sync_sla, p95_sla, mem_sla = 300.0, 150.0, 2000.0
    else:
        raise ValueError(f"Unknown tier: {tier}")

    prov = SyntheticProvenance(
        is_synthetic=True,
        synthetic_generator=DEFAULT_GENERATOR_NAME,
        generation_seed=seed,
        pii_scrubbed=True,
        employer_data_overlap=False,
        corpus_tier=int(tier),
    )

    languages = ["csharp", "typescript", "python", "terraform", "go", "rust", "java"]
    repos: list[SyntheticRepoMeta] = []
    for r in range(repo_count):
        lang = languages[r % len(languages)]
        repos.append(
            SyntheticRepoMeta(
                repo_id=f"repo_t{tier}_{r:04d}",
                repo_name=f"enterprise/service-cluster-{lang}-{r:04d}",
                primary_language=lang,
                estimated_files=20 + ((r * 13) % 80),
                provenance=prov,
            )
        )

    docs: list[SyntheticDocMeta] = []
    doc_types = ["confluence", "jira", "rfc"]
    for d in range(doc_count):
        dtype = doc_types[d % len(doc_types)]
        docs.append(
            SyntheticDocMeta(
                doc_id=f"doc_t{tier}_{dtype}_{d:05d}",
                doc_type=dtype,
                title=f"Enterprise Architecture Spec {d:05d} ({dtype.upper()})",
                verbatim_text=(
                    f"The distributed lock acquisition timeout for partition {d % 16} "
                    f"must be calibrated to 250ms under peak throughput."
                ),
                provenance=prov,
            )
        )

    return ScaleLadderCorpus(
        tier=tier,
        target_repo_count=repo_count,
        target_doc_count=doc_count,
        max_sync_seconds=sync_sla,
        max_query_p95_ms=p95_sla,
        max_memory_mb=mem_sla,
        repositories=repos,
        documents=docs,
    )
