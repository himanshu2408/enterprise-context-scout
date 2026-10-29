"""Core Pydantic v2 contracts and deterministic scoring/friction validators."""

from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from context_scout.config import validate_google_model_id


class RoleType(StrEnum):
    ENGINEER = "engineer"
    QUANT_RESEARCHER = "quant_researcher"
    PRODUCT_MANAGER = "product_manager"


class MarketRegime(StrEnum):
    INDIA_NSE_BSE = "IN_NSE"
    GERMANY_XETRA_DAX = "DE_XETRA"
    US_NYSE_NASDAQ = "US_EQUITY"
    GLOBAL_MULTI = "GLOBAL"
    INTERNAL_ADR = "INTERNAL_ADR"


class ExecutionMode(StrEnum):
    CHAT_QA = "chat_qa"
    GEMINI_HANDOFF = "gemini_handoff"


class ModificationIntent(StrEnum):
    MODIFY_TARGET = "modify_target"
    IMPORTED_TYPE = "imported_type"
    DIRECT_CALLER = "direct_caller"
    TEST_FIXTURE = "test_fixture"
    REFERENCE_ONLY = "reference_only"


class FrictionVerdict(StrEnum):
    REALISTIC_FRICTION = "REALISTIC_FRICTION"
    UNVERIFIED_FRICTION_WARNING = "UNVERIFIED_FRICTION_WARNING"
    FAILS_AFTER_COSTS = "FAILS_AFTER_COSTS"


# FIN-01: Realistic minimum round-trip transaction cost floors (bps) per market regime
MARKET_MIN_FRICTION_BPS: Final[dict[MarketRegime, float]] = {
    MarketRegime.INDIA_NSE_BSE: 15.0,
    MarketRegime.GERMANY_XETRA_DAX: 10.0,
    MarketRegime.US_NYSE_NASDAQ: 5.0,
    MarketRegime.GLOBAL_MULTI: 10.0,
    MarketRegime.INTERNAL_ADR: 0.0,
}


def compute_normalized_rrf(
    ranks_by_modality: list[int | None],
    num_modalities: int = 3,
    k: int = 60,
) -> float:
    """Computes Normalized Reciprocal Rank Fusion in [0.0, 1.0] (BLK-01 remediation).

    Raw RRF score is sum(1 / (k + rank_i)). Since the theoretical maximum when a
    item is rank #1 across all M modalities is M / (k + 1), dividing by M / (k + 1)
    scales the score cleanly onto [0.0, 1.0].
    """
    if num_modalities <= 0 or k < 0:
        raise ValueError("num_modalities must be >= 1 and k must be >= 0")

    raw_score = 0.0
    for rank in ranks_by_modality:
        if rank is not None:
            if rank < 1:
                raise ValueError(f"Rank must be >= 1, got {rank}")
            raw_score += 1.0 / (k + rank)

    max_possible = num_modalities * (1.0 / (k + 1.0))
    normalized = raw_score / max_possible
    return round(min(max(normalized, 0.0), 1.0), 6)


def evaluate_market_friction_floor(
    target_markets: list[MarketRegime],
    assumed_bps: float | None,
    is_intraday_futures: bool = False,
) -> FrictionVerdict:
    """Evaluates whether a paper's transaction cost assumption meets market reality (FIN-01)."""
    if assumed_bps is None:
        return FrictionVerdict.UNVERIFIED_FRICTION_WARNING

    required_bps = 5.0
    for mkt in target_markets:
        floor = MARKET_MIN_FRICTION_BPS.get(mkt, 10.0)
        if mkt == MarketRegime.INDIA_NSE_BSE and is_intraday_futures:
            floor = 5.0
        required_bps = max(required_bps, floor)

    if assumed_bps < required_bps:
        return FrictionVerdict.UNVERIFIED_FRICTION_WARNING
    return FrictionVerdict.REALISTIC_FRICTION


class PrincipalContext(BaseModel):
    """Server-side verified identity and RBAC/ABAC scopes (never LLM-supplied)."""

    user_id: str = Field(min_length=1)
    role: RoleType
    allowed_scopes: list[str] = Field(min_length=1)
    max_session_paid_tokens: int = Field(default=16_000, ge=1_000, le=100_000)

    @field_validator("allowed_scopes")
    @classmethod
    def _validate_scopes(cls, scopes: list[str]) -> list[str]:
        cleaned = [s.strip() for s in scopes if s.strip()]
        if not cleaned:
            raise ValueError("allowed_scopes must contain at least one non-empty scope")
        for s in cleaned:
            if "," in s or ";" in s or " " in s:
                raise ValueError(f"Invalid scope token '{s}'")
        return cleaned

    def to_postgres_scope_csv(self) -> str:
        """Serializes scopes for SET LOCAL app.principal_scopes."""
        return ",".join(self.allowed_scopes)


class AuthorityRules(BaseModel):
    primary_truth_source: Literal["code", "document"]
    require_out_of_sample_check: bool = False
    require_transaction_cost_check: bool = False
    freshness_half_life_days: int = Field(default=180, ge=1)


class DomainPackManifest(BaseModel):
    """Declarative plugin manifest loaded from domains/<domain_id>/domain.yaml."""

    domain_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    base_model_id: str
    contextualizer_model_id: str = Field(default="google/gemma-4-E4B-it")
    lora_adapter_id: str = Field(min_length=1)
    lora_adapter_path: str = Field(min_length=1)
    embedding_model_id: str = Field(default="google/embeddinggemma-2")
    embedding_dimensions: int = Field(default=768, ge=128, le=3072)
    mcp_read_tools_allowed: list[str] = Field(min_length=1)
    authority_rules: AuthorityRules
    output_contract: Literal["ImplementationContextPack", "QuantStrategyBrief"]

    @field_validator("base_model_id", "contextualizer_model_id", "embedding_model_id")
    @classmethod
    def _check_google_model(cls, value: str) -> str:
        return validate_google_model_id(value)


class CodeSlicePointer(BaseModel):
    """Pointer emitted by the Gemma 4 Scout (never re-typed code from memory)."""

    repo_name: str = Field(min_length=1)
    file_path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    qualified_symbol: str = Field(min_length=1)
    selection_reason: str = Field(min_length=1)
    modification_intent: ModificationIntent

    @model_validator(mode="after")
    def _validate_line_order(self) -> "CodeSlicePointer":
        if self.end_line < self.start_line:
            raise ValueError(
                f"end_line ({self.end_line}) must be >= start_line ({self.start_line})"
            )
        return self


class VerbatimCodeSlice(CodeSlicePointer):
    """Hydrated deterministically by the Python Slicer from Git/PostgreSQL."""

    git_commit_sha: str = Field(min_length=7)
    git_blob_sha: str = Field(min_length=7)
    verbatim_code_with_lines: str = Field(min_length=1)
    auto_expanded_by_ast: bool = False


class DocCitationSpan(BaseModel):
    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    doc_title: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    section_heading: str = Field(min_length=1)
    verbatim_quote: str = Field(min_length=1)


class QuantStrategyBrief(BaseModel):
    """Structured extraction contract for Model 2 with Market-Regime Friction Gate (FIN-01)."""

    strategy_name: str = Field(min_length=1)
    target_markets: list[MarketRegime] = Field(min_length=1)
    is_intraday_futures: bool = False
    mathematical_signal_latex: str = Field(min_length=1)
    entry_exit_rules: list[str] = Field(min_length=1)
    sample_period_years: str = Field(min_length=1)
    reported_sharpe_ratio: float | None = None
    reported_max_drawdown_pct: float | None = None
    transaction_cost_bps_assumed: float | None = None
    friction_verdict: FrictionVerdict = FrictionVerdict.UNVERIFIED_FRICTION_WARNING
    market_specific_caveats: list[str] = Field(min_length=1)
    known_failure_regimes: list[str] = Field(min_length=1)
    verified_citations: list[DocCitationSpan] = Field(min_length=1)

    @model_validator(mode="after")
    def _enforce_scepticism_gate(self) -> "QuantStrategyBrief":
        if self.friction_verdict != FrictionVerdict.FAILS_AFTER_COSTS:
            computed_verdict = evaluate_market_friction_floor(
                target_markets=self.target_markets,
                assumed_bps=self.transaction_cost_bps_assumed,
                is_intraday_futures=self.is_intraday_futures,
            )
            if computed_verdict == FrictionVerdict.UNVERIFIED_FRICTION_WARNING:
                self.friction_verdict = FrictionVerdict.UNVERIFIED_FRICTION_WARNING
        return self


class PreFlightValidationReport(BaseModel):
    passed: bool
    unresolved_symbols: list[str] = Field(default_factory=list)
    missing_reference_identifiers: list[str] = Field(default_factory=list)
    novel_target_identifiers: list[str] = Field(default_factory=list)
    normalized_rrf_score: float = Field(ge=0.0, le=1.0)
    citations_verified_100_pct: bool
    auto_attached_ast_slices_count: int = Field(default=0, ge=0)
    bounce_back_feedback: str | None = None


class InsufficientContextDiagnostic(BaseModel):
    """Terminal exit payload returned when local bounce-back still fails Pre-Flight."""

    trace_id: str
    domain_id: str
    reason: str
    missing_symbols: list[str]
    partial_candidate_files: list[str]
    paid_tokens_spent: Literal[0] = 0


class ImplementationContextPack(BaseModel):
    """Compiler-complete context bundle handed to paid Gemini 2.5 / 3.8 Flash."""

    trace_id: str = Field(min_length=1)
    domain_id: str = Field(min_length=1)
    user_task: str = Field(min_length=1)
    architectural_execution_plan: list[str] = Field(min_length=1)
    doc_vs_code_conflicts_detected: list[str] = Field(default_factory=list)
    code_slices: list[VerbatimCodeSlice] = Field(min_length=1)
    supporting_doc_citations: list[DocCitationSpan] = Field(default_factory=list)
    quant_strategy_spec: QuantStrategyBrief | None = None
    preflight_report: PreFlightValidationReport
    local_tokens_consumed: int = Field(ge=0)
    pack_token_count: int = Field(ge=1)
