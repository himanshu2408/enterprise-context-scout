"""Quantitative Finance Domain Plug-in (FIN-01, F18).

Encapsulates institutional market regimes, statutory transaction cost floors,
turnover/volatility Sharpe penalty formulas, and maximum drawdown risk gates.
Composes with DocumentCitationGate for decoupled, modular evaluation.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

from context_scout.agents.citation_gate import (
    DocumentCitationGate,
    DocumentCitationGateResult,
)
from context_scout.db.repositories import ASTSymbolRecord, CodeFileRecord, DocChunkRecord
from context_scout.schemas.contracts import (
    DocCitationSpan,
    DomainKnowledgeBrief,
    PrincipalContext,
    RoleType,
)
from examples.quant_dogfood.contracts import FrictionVerdict, MarketRegime, QuantStrategyBrief

if TYPE_CHECKING:
    from context_scout.ingestion.doc_parser import DocumentChunk

logger = logging.getLogger(__name__)

CODE_DOMAIN_ID_FINANCE: Final[str] = "trading_code_scout"
DOMAIN_ID_FINANCE: Final[str] = "quant_research_scout"
FEDERATED_DOMAIN_ID_FINANCE: Final[str] = "federated_coordinator"

AUTO_RESEARCH_KEYWORDS: Final[tuple[str, ...]] = (
    "equity market",
    "stock market",
    "indian",
    "actor-critic",
    "reinforcement learning",
    "momentum strategy",
    "hierarchical risk parity",
    "minimum variance",
    "outperform",
    "hawkes process",
)
AUTO_RESEARCH_REGEX: Final[re.Pattern[str]] = re.compile(
    r"\b(?:nifty|nse|xetra|dax|finrl)\b", re.IGNORECASE
)

# Institutional minimum round-trip transaction cost floors (bps) per market regime (FIN-01)
MARKET_MIN_FRICTION_BPS: Final[dict[MarketRegime, float]] = {
    MarketRegime.INDIA_NSE_BSE: 15.0,  # 10 bps delivery STT + GST + turnover
    MarketRegime.GERMANY_XETRA_DAX: 10.0,  # Clearing, exchange fee, bid-ask spread
    MarketRegime.US_NYSE_NASDAQ: 5.0,  # SEC Section 31, FINRA TAF, spread
    MarketRegime.GLOBAL_MULTI: 10.0,
    MarketRegime.INTERNAL_ADR: 0.0,
}

# Institutional risk boundaries
MAX_ALLOWABLE_DRAWDOWN_PCT: Final[float] = 30.0
MIN_NET_SHARPE_THRESHOLD: Final[float] = 0.5
DEFAULT_ANNUAL_TURNOVER: Final[float] = 12.0
DEFAULT_ANNUAL_VOLATILITY: Final[float] = 0.285714  # ~3.5x multiplier


def calculate_effective_friction_floor(
    target_markets: list[MarketRegime],
    is_intraday_futures: bool = False,
) -> float:
    """Calculates the strict minimum transaction cost floor (bps) across target markets."""
    if not target_markets:
        return 10.0

    floor = 0.0
    for mkt in target_markets:
        mkt_floor = MARKET_MIN_FRICTION_BPS.get(mkt, 10.0)
        if mkt == MarketRegime.INDIA_NSE_BSE and is_intraday_futures:
            mkt_floor = 5.0
        floor = max(floor, mkt_floor)
    return floor


def estimate_net_sharpe(
    reported_sharpe: float | None,
    assumed_bps: float | None,
    floor_bps: float,
    turnover_annual: float = DEFAULT_ANNUAL_TURNOVER,
    assumed_annual_volatility: float | None = None,
) -> float | None:
    """Penalizes gross Sharpe if assumed friction omitted institutional realities."""
    if reported_sharpe is None:
        return None

    actual_cost_bps = assumed_bps if assumed_bps is not None else 0.0
    friction_deficit_bps = max(0.0, floor_bps - actual_cost_bps)
    # Annualized friction drag as a fraction of capital
    annual_drag = (friction_deficit_bps * turnover_annual) / 10000.0
    # Dimensional reduction: Delta SR = annual_drag / sigma_annual
    inv_vol_multiplier = (
        (1.0 / max(assumed_annual_volatility, 0.01))
        if assumed_annual_volatility is not None
        else (1.0 / DEFAULT_ANNUAL_VOLATILITY)
    )
    net_sharpe = reported_sharpe - (annual_drag * inv_vol_multiplier)
    return round(max(-2.0, net_sharpe), 2)


@dataclass(frozen=True)
class FinancialEvaluationResult:
    """Result of pure financial parameter evaluation (costs, Sharpe, drawdown)."""

    passed: bool
    friction_verdict: FrictionVerdict
    effective_friction_floor_bps: float
    assumed_friction_bps: float | None
    estimated_net_sharpe: float | None
    reason: str


@dataclass(frozen=True)
class ScepticismEvaluationResult:
    """Comprehensive evaluation payload combining financial rules and citation gating."""

    passed: bool
    friction_verdict: FrictionVerdict
    citations_valid: bool
    citations_verified_count: int
    failed_citations: list[tuple[DocCitationSpan, str]]
    reason: str
    effective_friction_floor_bps: float
    assumed_friction_bps: float | None
    estimated_net_sharpe: float | None = None
    citation_gate_result: DocumentCitationGateResult | None = None


class FinanceDomainPlugin:
    """Quantitative Finance Domain Plug-in (F18).

    Evaluates:
    1. Statutory transaction cost friction floors across global market regimes (FIN-01).
    2. Net Sharpe ratio degradation under turnover and volatility drag.
    3. Institutional risk boundaries (max drawdown <= 30%, net Sharpe > 0.5).
    4. Composes cleanly with DocumentCitationGate for decoupled verbatim quote verification.
    """

    domain_id: Final[str] = DOMAIN_ID_FINANCE
    code_domain_id: Final[str] = CODE_DOMAIN_ID_FINANCE
    federated_domain_id: Final[str] = FEDERATED_DOMAIN_ID_FINANCE

    def __init__(
        self,
        citation_gate: DocumentCitationGate | None = None,
        min_citation_overlap_ratio: float = 0.98,
    ) -> None:
        self.citation_gate = citation_gate or DocumentCitationGate(
            min_overlap_ratio=min_citation_overlap_ratio,
            require_citations_default=True,
            strict_freshness=False,
        )

    def supported_domain_ids(self) -> tuple[str, ...]:
        """All domain IDs (code, research, federated) handled by this plugin."""
        return (
            CODE_DOMAIN_ID_FINANCE,
            DOMAIN_ID_FINANCE,
            FEDERATED_DOMAIN_ID_FINANCE,
        )

    def get_role_domain_ids(self) -> dict[str, str]:
        """Returns role-to-domain mapping for finance code, research, and federated domains."""
        return {
            "code": CODE_DOMAIN_ID_FINANCE,
            "research": DOMAIN_ID_FINANCE,
            "federated": FEDERATED_DOMAIN_ID_FINANCE,
        }

    def get_default_scopes(self) -> list[str]:
        """Default principal scopes associated with the finance domain packs."""
        return ["public_eng", "repo:trading_strategies", "corpus:quant_papers"]

    def get_adapter_specs(self) -> list[dict[str, str]]:
        """Adapter specifications for finance code and research LoRA adapters."""
        return [
            {
                "domain_id": CODE_DOMAIN_ID_FINANCE,
                "adapter_rel_path": (
                    f"domains/{CODE_DOMAIN_ID_FINANCE}/adapters/gemma4_26b_code_scout"
                ),
                "train_jsonl": f"domains/{CODE_DOMAIN_ID_FINANCE}/tuning/train.jsonl",
                "valid_jsonl": f"domains/{CODE_DOMAIN_ID_FINANCE}/tuning/valid.jsonl",
                "role": "code",
                "label": "Model 1",
            },
            {
                "domain_id": DOMAIN_ID_FINANCE,
                "adapter_rel_path": (
                    f"domains/{DOMAIN_ID_FINANCE}/adapters/gemma4_26b_quant_research"
                ),
                "train_jsonl": f"domains/{DOMAIN_ID_FINANCE}/tuning/train.jsonl",
                "valid_jsonl": f"domains/{DOMAIN_ID_FINANCE}/tuning/valid.jsonl",
                "role": "research",
                "label": "Model 2",
            },
        ]

    def is_federated_domain(self, domain_id: str) -> bool:
        """Returns True if domain_id is the finance federated coordinator."""
        return domain_id == FEDERATED_DOMAIN_ID_FINANCE

    def is_research_domain(self, domain_id: str) -> bool:
        """Returns True if domain_id is the quantitative research domain."""
        return domain_id == DOMAIN_ID_FINANCE

    def should_auto_route_research(
        self,
        prompt: str,
        domain_id: str,
        allowed_scopes: list[str] | None = None,
    ) -> bool:
        """Detects quantitative research terms when querying the finance code domain."""
        if domain_id != CODE_DOMAIN_ID_FINANCE:
            return False
        if allowed_scopes is not None:
            research_scopes = {"public_quant", "corpus:quant_papers", "quant_restricted"}
            if not (research_scopes & set(allowed_scopes)):
                return False
        prompt_l = prompt.lower()
        code_term_overrides = (
            "kalman",
            "garch",
            "kelly",
            "almgren",
            "chriss",
            "ledoit",
            "slippage",
            "orderflowimbalance",
            "order_flow_imbalance",
        )
        if any(term in prompt_l for term in code_term_overrides):
            return False
        return any(k in prompt_l for k in AUTO_RESEARCH_KEYWORDS) or bool(
            AUTO_RESEARCH_REGEX.search(prompt_l)
        )

    def elevate_principal_for_research(self, principal: PrincipalContext) -> PrincipalContext:
        """Adds finance research domain scope when principal already holds quant corpus access."""
        research_scopes = {"public_quant", "corpus:quant_papers", "quant_restricted"}
        domain_scope = f"domain:{DOMAIN_ID_FINANCE}"
        if (research_scopes & set(principal.allowed_scopes)) and (
            domain_scope not in principal.allowed_scopes
        ):
            return PrincipalContext(
                user_id=principal.user_id,
                role=principal.role,
                allowed_scopes=[*principal.allowed_scopes, domain_scope],
                session_budget_tokens=principal.session_budget_tokens,
                max_session_paid_tokens=principal.max_session_paid_tokens,
            )
        return principal

    def infer_friction_label(self, market_scope: str) -> str:
        """Infers statutory transaction-cost floor label for contextual chunk headers."""
        scope_upper = market_scope.upper()
        if "NSE" in scope_upper:
            return "15 bps STT floor"
        if "XETRA" in scope_upper:
            return "10 bps floor"
        return "5 bps floor"

    def infer_market_scopes(self, prompt: str) -> list[MarketRegime]:
        """Infers target MarketRegime scopes from user prompt keywords."""
        prompt_lower = prompt.lower()
        target_markets: list[MarketRegime] = []
        if any(k in prompt_lower for k in ("nse", "bse", "nifty", "sensex", "india")):
            target_markets.append(MarketRegime.INDIA_NSE_BSE)
        if any(k in prompt_lower for k in ("xetra", "dax", "german", "eurex")):
            target_markets.append(MarketRegime.GERMANY_XETRA_DAX)
        if (
            any(k in prompt_lower for k in ("us", "nyse", "nasdaq", "cme", "mnq"))
            or not target_markets
        ):
            target_markets.append(MarketRegime.US_NYSE_NASDAQ)
        return target_markets

    def infer_explicit_market_scopes(self, prompt: str) -> list[str]:
        """Infer explicit market scope values only when named in the query (no default)."""
        q_low = prompt.lower()
        inferred: list[str] = []
        if re.search(r"\b(nse|in_nse|india|indian|nifty|bse|sensex)\b", q_low):
            inferred.append(MarketRegime.INDIA_NSE_BSE.value)
        if re.search(r"\b(xetra|de_xetra|dax|frankfurt|german|germany|eurex)\b", q_low):
            inferred.append(MarketRegime.GERMANY_XETRA_DAX.value)
        if re.search(r"\b(nyse|nasdaq|us_equity|sp500|cme|mnq)\b", q_low):
            inferred.append(MarketRegime.US_NYSE_NASDAQ.value)
        return inferred

    def get_market_scope_alias_map(self) -> dict[str, str]:
        """Return mapping from normalized MarketRegime code to searchable exchange/region terms."""
        return {
            MarketRegime.INDIA_NSE_BSE.value: "IN_NSE NSE India NIFTY BSE",
            MarketRegime.GERMANY_XETRA_DAX.value: "DE_XETRA XETRA Germany DAX",
            MarketRegime.US_NYSE_NASDAQ.value: "US_EQUITY NYSE NASDAQ US equity",
        }

    def get_market_scope_aliases(self, market_scope: MarketRegime | str | None) -> str:
        """Expand a MarketRegime code into searchable geographic and exchange aliases."""
        raw_val = market_scope.value if isinstance(market_scope, MarketRegime) else market_scope
        key = str(raw_val or "").strip().upper()
        return self.get_market_scope_alias_map().get(key, key)

    def synthesize_research_brief(
        self,
        user_prompt: str,
        domain_id: str,
        chunks: list[Any],
        verified_citations: list[Any],
    ) -> QuantStrategyBrief:
        """Synthesizes a deterministic QuantStrategyBrief grounded in verified chunks."""
        prompt_lower = user_prompt.lower()
        target_markets = self.infer_market_scopes(user_prompt)
        top_chk = chunks[0] if chunks else None

        from context_scout.agents.intent_router import (
            decompose_query_facets,
            evaluate_facet_grounding,
        )
        from context_scout.plugins.domains.data_engineering import (
            clean_brief_title,
            clean_display_text,
            is_slop_text,
        )
        from examples.quant_dogfood.contracts import evaluate_market_friction_floor

        clean_top_title = (
            clean_brief_title(str(top_chk.doc_title), default="Retrieved Document Context")
            if top_chk is not None
            else "No Retrieved Document Context"
        )
        clean_sec = (
            clean_brief_title(str(getattr(top_chk, "section_heading", "") or ""), default="")
            if top_chk is not None
            else ""
        )
        strategy_name = (
            f"{clean_top_title} — {clean_sec}"
            if top_chk is not None and clean_sec
            else (f"{clean_top_title} Synthesis" if top_chk is not None else "Insufficient Context")
        )

        combined_text = "\n".join(str(getattr(c, "verbatim_content", "") or "") for c in chunks)
        math_match = re.search(r"\$\$?([^\$\n]{4,160})\$\$?", combined_text)
        signal_latex = (
            math_match.group(1).strip()
            if math_match
            else r"\text{Derived strictly from retrieved evidence (no explicit equation in chunk)}"
        )

        # Parse quantitative metrics strictly from retrieved text; never fabricate numbers.
        bps_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:bps|basis\s+points?)", combined_text, re.IGNORECASE
        )
        assumed_bps: float | None = float(bps_match.group(1)) if bps_match else None

        sharpe_match = re.search(
            r"Sharpe(?:\s+ratio)?\s*(?:of|=|:)?\s*(-?\d+(?:\.\d+)?)",
            combined_text,
            re.IGNORECASE,
        )
        sharpe_ratio: float | None = float(sharpe_match.group(1)) if sharpe_match else None

        dd_match = re.search(
            r"max(?:imum)?\s+drawdown\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*%",
            combined_text,
            re.IGNORECASE,
        )
        max_dd_pct: float | None = float(dd_match.group(1)) if dd_match else None

        period_match = re.search(
            r"\b((?:19|20)\d{2}\s*[-\u2013]\s*(?:19|20)\d{2})\b", combined_text
        )
        sample_period_years: str = (
            period_match.group(1).replace("\u2013", "-") if period_match else "unspecified"
        )

        # Extract real empirical findings, caveats, and failure regimes from retrieved chunks
        extracted_rules: list[str] = []
        extracted_caveats: list[str] = []
        extracted_failures: list[str] = []
        for chk in chunks:
            for raw_ln in str(getattr(chk, "verbatim_content", "") or "").splitlines():
                s_ln = raw_ln.strip().lstrip("-*•0123456789. ")
                if len(s_ln) < 30 or s_ln.endswith(":") or is_slop_text(s_ln):
                    continue
                cleaned_ln = clean_display_text(s_ln)
                if len(cleaned_ln) < 30 or len(re.findall(r"[A-Za-z]{3,}", cleaned_ln)) < 4:
                    continue
                s_lower = cleaned_ln.lower()
                if any(
                    fk in s_lower
                    for fk in (
                        "off-target",
                        "failure",
                        "fail",
                        "degrad",
                        "instability",
                        "toxic",
                        "error",
                        "loss of",
                        "breach",
                        "violation",
                        "penalty",
                        "outage",
                        "drawdown",
                        "crash",
                        "slippage",
                        "overfit",
                        "non-stationar",
                    )
                ):
                    if cleaned_ln not in extracted_failures and len(extracted_failures) < 6:
                        extracted_failures.append(cleaned_ln)
                elif any(
                    ck in s_lower
                    for ck in (
                        "caveat",
                        "constraint",
                        "require",
                        "threshold",
                        "control",
                        "concentration",
                        "temperature",
                        "condition",
                        "must",
                        "mandatory",
                        "compliance",
                        "friction",
                        "transaction cost",
                        "liquidity",
                    )
                ):
                    if cleaned_ln not in extracted_caveats and len(extracted_caveats) < 6:
                        extracted_caveats.append(cleaned_ln)
                elif cleaned_ln not in extracted_rules and len(extracted_rules) < 8:
                    extracted_rules.append(cleaned_ln)

        for cit in reversed(verified_citations):
            q = str(getattr(cit, "verbatim_quote", "") or "").strip()
            cleaned_q = clean_display_text(q)
            if cleaned_q and not is_slop_text(q) and cleaned_q not in extracted_rules:
                extracted_rules.insert(0, cleaned_q)

        entry_exit_rules = (
            extracted_rules[:6]
            if extracted_rules
            else [
                f"Retrieved evidence from {clean_top_title} "
                "did not yield explicit entry/exit rules."
            ]
        )
        market_caveats = (
            extracted_caveats[:5]
            if extracted_caveats
            else [
                "No explicit market friction or constraint clauses extracted from retrieved chunks."
            ]
        )
        failure_regimes = (
            extracted_failures[:5]
            if extracted_failures
            else ["No explicit failure regime clauses extracted from retrieved chunks."]
        )

        if domain_id != DOMAIN_ID_FINANCE and top_chk is not None:
            target_markets = [MarketRegime.GLOBAL_MULTI]

        plan = decompose_query_facets(user_prompt)
        facet_covs = [
            evaluate_facet_grounding(
                facet,
                doc_chunks=chunks,
                code_slices=[],
                domain_id=domain_id,
            )
            for facet in plan.facets
        ]
        grounded_count = sum(1 for cov, _ in facet_covs if cov.status == "grounded")
        partial_count = sum(1 for cov, _ in facet_covs if cov.status == "partial")
        missing_aspects = [msg for _, msg in facet_covs if msg]

        if not chunks or not verified_citations or (grounded_count == 0 and partial_count == 0):
            sufficiency_verdict: Literal["sufficient", "partial", "insufficient"] = "insufficient"
        elif missing_aspects or partial_count > 0:
            sufficiency_verdict = "partial"
        else:
            sufficiency_verdict = "sufficient"

        if (
            target_markets == [MarketRegime.GLOBAL_MULTI]
            and assumed_bps is None
            and sharpe_ratio is None
        ):
            friction_verdict = FrictionVerdict.REALISTIC_FRICTION
        else:
            friction_verdict = evaluate_market_friction_floor(target_markets, assumed_bps)

        _ = prompt_lower
        return QuantStrategyBrief(
            strategy_name=strategy_name,
            target_markets=target_markets,
            is_intraday_futures=False,
            mathematical_signal_latex=signal_latex,
            entry_exit_rules=entry_exit_rules,
            sample_period_years=sample_period_years,
            reported_sharpe_ratio=sharpe_ratio,
            reported_max_drawdown_pct=max_dd_pct,
            transaction_cost_bps_assumed=assumed_bps,
            friction_verdict=friction_verdict,
            market_specific_caveats=market_caveats,
            known_failure_regimes=failure_regimes,
            verified_citations=verified_citations,
            sufficiency_verdict=sufficiency_verdict,
            missing_aspects=missing_aspects,
            synthesis_mode="deterministic_extractive_fallback",
        )

    def extract_brief_metadata(self, brief: Any) -> dict[str, Any]:
        """Extracts neutral structured metadata from a QuantStrategyBrief."""
        if not isinstance(brief, QuantStrategyBrief):
            return {}
        metadata: dict[str, Any] = {
            "target_markets": [m.value for m in brief.target_markets],
            "friction_verdict": brief.friction_verdict.value,
        }
        if brief.transaction_cost_bps_assumed is not None:
            metadata["transaction_cost_bps_assumed"] = brief.transaction_cost_bps_assumed
        if brief.reported_sharpe_ratio is not None:
            metadata["reported_sharpe_ratio"] = brief.reported_sharpe_ratio
        return metadata

    def validate_research_brief(
        self,
        brief: Any,
        doc_chunks: list[Any],
    ) -> tuple[bool, str]:
        """Validates financial friction floors and citations for QuantStrategyBrief."""
        if not isinstance(brief, QuantStrategyBrief):
            return True, ""
        min_floor = calculate_effective_friction_floor(
            target_markets=brief.target_markets,
            is_intraday_futures=brief.is_intraday_futures,
        )
        assumed = brief.transaction_cost_bps_assumed or 0.0
        if assumed < min_floor:
            return (
                False,
                f"QuantStrategyBrief assumed {assumed} bps, "
                f"which fails statutory friction floor for: {brief.target_markets}",
            )
        return True, ""

    def build_distillation_prompt_and_brief(
        self,
        domain_id: str,
        doc_id: str,
        title: str,
        sec_head: str,
        quote: str,
        chk_id: str,
        manifest: dict[str, Any],
    ) -> tuple[str, str]:
        """Builds a training prompt and structured brief JSON for custom CLI distillation."""
        doc_slug = doc_id.split(":")[-1]
        cit = DocCitationSpan(
            chunk_id=chk_id,
            doc_id=doc_id,
            doc_title=title,
            source_uri=f"urn:{domain_id}:{doc_slug}",
            section_heading=sec_head,
            verbatim_quote=quote,
        )
        is_quant_domain = domain_id == DOMAIN_ID_FINANCE or bool(
            (manifest.get("authority_rules") or {}).get("require_transaction_cost_check")
            and domain_id.startswith("quant")
        )
        use_knowledge_contract = manifest.get("output_contract") == "DomainKnowledgeBrief"
        if use_knowledge_contract:
            kb_brief = DomainKnowledgeBrief(
                brief_title=f"{title} — {sec_head}",
                domain_id=domain_id,
                executive_summary=f"Grounded synthesis for {title}: {quote}",
                formal_expression_or_rule=None,
                core_findings_and_rules=[quote],
                domain_constraints_and_caveats=[f"Grounded in {doc_id} ({sec_head})"],
                known_risks_and_failure_modes=["Out-of-distribution domain regime shift"],
                structured_metadata={
                    "domain_id": domain_id,
                    "verified_citations_count": 1,
                },
                verified_citations=[cit],
            )
            assistant_json = kb_brief.model_dump_json(indent=2)
            strategy_label = kb_brief.brief_title
        else:
            brief = QuantStrategyBrief(
                strategy_name=f"{title} — {sec_head}",
                target_markets=[MarketRegime.GLOBAL_MULTI],
                is_intraday_futures=False,
                mathematical_signal_latex=r"E[Y \mid X] = \alpha + \beta X",
                entry_exit_rules=[quote],
                sample_period_years="2020-2026",
                reported_sharpe_ratio=1.75 if is_quant_domain else None,
                reported_max_drawdown_pct=8.5 if is_quant_domain else None,
                transaction_cost_bps_assumed=12.0 if is_quant_domain else None,
                friction_verdict=FrictionVerdict.REALISTIC_FRICTION,
                market_specific_caveats=[f"Grounded in {doc_id} ({sec_head})"],
                known_failure_regimes=["Out-of-distribution domain regime shift"],
                verified_citations=[cit],
            )
            assistant_json = brief.model_dump_json(indent=2)
            strategy_label = brief.strategy_name

        if is_quant_domain:
            user_prompt = (
                f"Synthesize a verified research brief for `{title}`.\n"
                f"Retrieved Corpus Chunk: [chunk_id={chk_id} | doc_id={doc_id} | "
                f"title={title} | uri=urn:{domain_id}:{doc_slug} | section={sec_head} | "
                f"strategy={strategy_label} | sharpe=1.75 | mdd=8.5 | bps=12.0 | "
                f"quote={json.dumps(quote)}]"
            )
        else:
            user_prompt = (
                f"Synthesize a verified domain brief for `{title}`.\n"
                f"Retrieved Corpus Chunk: [chunk_id={chk_id} | doc_id={doc_id} | "
                f"title={title} | uri=urn:{domain_id}:{doc_slug} | section={sec_head} | "
                f"topic={strategy_label} | quote={json.dumps(quote)}]"
            )
        return user_prompt, assistant_json

    def get_demo_seed_payloads(self) -> dict[str, Any]:
        """Returns finance-domain in-memory demo code/doc seed data for API demo fallback."""
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[4]
        strat_dir = repo_root / "src" / "context_scout" / "trading_strategies"
        code_files: list[CodeFileRecord] = []
        ast_symbols: list[ASTSymbolRecord] = []
        if strat_dir.is_dir():
            for py_file in sorted(strat_dir.glob("*.py")):
                if py_file.name == "__init__.py":
                    continue
                content = py_file.read_text(encoding="utf-8")
                lines = content.splitlines()
                rel_path = str(py_file.relative_to(repo_root))
                f_id = uuid.uuid4()
                code_files.append(
                    CodeFileRecord(
                        file_id=f_id,
                        domain_id=CODE_DOMAIN_ID_FINANCE,
                        repo_name="enterprise-context-scout",
                        file_path=rel_path,
                        git_commit_sha="c0ffee123456",
                        git_blob_sha="b10b123456",
                        language="python",
                        allowed_scopes=["public_eng"],
                        raw_content=content,
                        line_count=len(lines),
                    )
                )
                module_parts = py_file.relative_to(repo_root / "src").with_suffix("").parts
                mod_dotted = ".".join(module_parts)
                import ast as _ast

                from context_scout.retrieval.embedder import get_embedding_provider

                demo_embedder = get_embedding_provider()

                try:
                    tree = _ast.parse(content)
                    for node in _ast.walk(tree):
                        if isinstance(node, _ast.ClassDef):
                            cls_name = node.name
                            q_name = f"{mod_dotted}.{cls_name}"
                            ast_symbols.append(
                                ASTSymbolRecord(
                                    symbol_id=uuid.uuid4(),
                                    file_id=f_id,
                                    domain_id=CODE_DOMAIN_ID_FINANCE,
                                    qualified_name=q_name,
                                    symbol_kind="class",
                                    start_line=node.lineno,
                                    end_line=node.end_lineno or min(node.lineno + 50, len(lines)),
                                    signature_skeleton=f"class {cls_name}:",
                                    contextual_summary=f"Class definition for {cls_name}",
                                    allowed_scopes=["public_eng"],
                                    embedding=demo_embedder.embed_text(
                                        f"{q_name} class {cls_name}", is_query=False
                                    ),
                                )
                            )
                except SyntaxError as exc:
                    logger.debug(
                        "AST parse failed while seeding demo symbols for %s (%s): %s",
                        rel_path,
                        type(exc).__name__,
                        exc,
                    )
                    for idx, line in enumerate(lines, start=1):
                        cls_match = re.match(r"^class\s+([A-Za-z0-9_]+)", line)
                        if cls_match:
                            cls_name = cls_match.group(1)
                            q_name = f"{mod_dotted}.{cls_name}"
                            ast_symbols.append(
                                ASTSymbolRecord(
                                    symbol_id=uuid.uuid4(),
                                    file_id=f_id,
                                    domain_id=CODE_DOMAIN_ID_FINANCE,
                                    qualified_name=q_name,
                                    symbol_kind="class",
                                    start_line=idx,
                                    end_line=min(idx + 50, len(lines)),
                                    signature_skeleton=f"class {cls_name}:",
                                    contextual_summary=f"Class definition for {cls_name}",
                                    allowed_scopes=["public_eng"],
                                    embedding=demo_embedder.embed_text(
                                        f"{q_name} class {cls_name}", is_query=False
                                    ),
                                )
                            )

        from context_scout.retrieval.embedder import get_embedding_provider

        doc_embedder = get_embedding_provider()
        doc_chunks = [
            DocChunkRecord(
                chunk_id=uuid.uuid4(),
                domain_id=DOMAIN_ID_FINANCE,
                doc_id="arxiv:2606.31387",
                doc_title="Signature-Based Optimal Execution for Statistical Arbitrage",
                source_uri="urn:arxiv:2606.31387.pdf",
                market_scope=MarketRegime.US_NYSE_NASDAQ,
                published_date=datetime.date(2026, 6, 15),
                authority_weight=1.0,
                section_heading="Optimal Liquidation Trajectory",
                contextual_header="Quantitative optimal execution analysis.",
                verbatim_content=(
                    "Optimal execution requires balancing market impact and inventory risk. "
                    "Order flow imbalance exhibits mean reversion over sub-second horizons."
                ),
                metadata_json={},
                allowed_scopes=["public_eng", "public_quant", "quant_restricted"],
                embedding=doc_embedder.embed_text(
                    "Signature-Based Optimal Execution for Statistical Arbitrage",
                    is_query=False,
                ),
            ),
            DocChunkRecord(
                chunk_id=uuid.uuid4(),
                domain_id=DOMAIN_ID_FINANCE,
                doc_id="arxiv:2607.12944",
                doc_title="Microstructure Alpha in European Equities and XETRA Order Books",
                source_uri="urn:arxiv:2607.12944.pdf",
                market_scope=MarketRegime.GERMANY_XETRA_DAX,
                published_date=datetime.date(2026, 7, 20),
                authority_weight=1.0,
                section_heading="Tick-Level Dynamics",
                contextual_header="European microstructure alpha modeling.",
                verbatim_content=(
                    "XETRA continuous trading auctions enforce a minimum round-trip friction "
                    "floor. Statistical arbitrage strategies must clear at least 10 bps to remain "
                    "profitable."
                ),
                metadata_json={},
                allowed_scopes=["public_eng", "public_quant", "quant_restricted"],
                embedding=doc_embedder.embed_text(
                    "Microstructure Alpha in European Equities and XETRA Order Books",
                    is_query=False,
                ),
            ),
            DocChunkRecord(
                chunk_id=uuid.uuid4(),
                domain_id=DOMAIN_ID_FINANCE,
                doc_id="arxiv:2302.13245",
                doc_title="Physical Momentum in the Indian Stock Market",
                source_uri="https://arxiv.org/abs/2302.13245",
                market_scope=MarketRegime.INDIA_NSE_BSE,
                published_date=datetime.date(2024, 1, 1),
                authority_weight=1.0,
                section_heading="Abstract",
                contextual_header="Indian equity momentum analysis.",
                verbatim_content=(
                    "Physical and cross-sectional momentum strategy portfolios across National "
                    "Stock Exchange (NSE) NIFTY constituents clear >= 18 bps round-trip friction."
                ),
                metadata_json={},
                allowed_scopes=["public_quant"],
                embedding=doc_embedder.embed_text(
                    "Physical Momentum in the Indian Stock Market",
                    is_query=False,
                ),
            ),
            DocChunkRecord(
                chunk_id=uuid.uuid4(),
                domain_id=DOMAIN_ID_FINANCE,
                doc_id="arxiv:2211.07080",
                doc_title=(
                    "Designing Efficient Pair-Trading and Intraday Momentum Strategies "
                    "on NSE Nifty 50"
                ),
                source_uri="https://arxiv.org/abs/2211.07080",
                market_scope=MarketRegime.INDIA_NSE_BSE,
                published_date=datetime.date(2024, 1, 1),
                authority_weight=1.0,
                section_heading="Abstract",
                contextual_header="NSE Nifty 50 intraday momentum.",
                verbatim_content=(
                    "High-frequency intraday momentum and order flow strategies on National Stock "
                    "Exchange of India (NSE) Nifty 50 constituents require >= 18 bps friction."
                ),
                metadata_json={},
                allowed_scopes=["public_quant"],
                embedding=doc_embedder.embed_text(
                    "Designing Efficient Pair-Trading and Intraday Momentum Strategies "
                    "on NSE Nifty 50",
                    is_query=False,
                ),
            ),
            DocChunkRecord(
                chunk_id=uuid.uuid4(),
                domain_id=DOMAIN_ID_FINANCE,
                doc_id="arxiv:2202.02728",
                doc_title=(
                    "Hierarchical Risk Parity and Minimum Variance Portfolio Design "
                    "on NIFTY 50 Stocks"
                ),
                source_uri="https://arxiv.org/abs/2202.02728",
                market_scope=MarketRegime.INDIA_NSE_BSE,
                published_date=datetime.date(2024, 1, 1),
                authority_weight=1.0,
                section_heading="Abstract",
                contextual_header="NIFTY 50 HRP portfolio optimization.",
                verbatim_content=(
                    "Hierarchical Risk Parity (HRP) and minimum variance portfolio optimization "
                    "across Indian NIFTY sectors on NSE outperform under >= 16 bps friction."
                ),
                metadata_json={},
                allowed_scopes=["public_quant"],
                embedding=doc_embedder.embed_text(
                    "Hierarchical Risk Parity and Minimum Variance Portfolio Design "
                    "on NIFTY 50 Stocks",
                    is_query=False,
                ),
            ),
            DocChunkRecord(
                chunk_id=uuid.uuid4(),
                domain_id=DOMAIN_ID_FINANCE,
                doc_id="arxiv:2607.18001",
                doc_title=(
                    "High-Dimensional Factor Statistical Arbitrage on Deutsche Boerse XETRA DAX"
                ),
                source_uri="https://arxiv.org/pdf/2607.18001v1",
                market_scope=MarketRegime.GERMANY_XETRA_DAX,
                published_date=datetime.date(2026, 7, 20),
                authority_weight=1.0,
                section_heading="Empirical Evaluation",
                contextual_header="XETRA DAX factor statistical arbitrage.",
                verbatim_content=(
                    "High-dimensional factor statistical arbitrage and order book liquidity "
                    "herding models on XETRA DAX equities clear a 12 bps friction floor."
                ),
                metadata_json={},
                allowed_scopes=["public_quant"],
                embedding=doc_embedder.embed_text(
                    "High-Dimensional Factor Statistical Arbitrage on Deutsche Boerse XETRA DAX",
                    is_query=False,
                ),
            ),
            DocChunkRecord(
                chunk_id=uuid.uuid4(),
                domain_id=DOMAIN_ID_FINANCE,
                doc_id="arxiv:2502.07393",
                doc_title=(
                    "FinRL-DeepSeek: Risk-Sensitive Deep Reinforcement Learning Trading Agents"
                ),
                source_uri="https://arxiv.org/pdf/2502.07393v1",
                market_scope=MarketRegime.US_NYSE_NASDAQ,
                published_date=datetime.date(2025, 2, 11),
                authority_weight=1.0,
                section_heading="Abstract",
                contextual_header="US equity reinforcement learning trading agents.",
                verbatim_content=(
                    "Risk-sensitive deep reinforcement learning actor-critic trading agents "
                    "evaluated on US equity Nasdaq-100 benchmarks clear >= 5 bps execution costs."
                ),
                metadata_json={},
                allowed_scopes=["public_quant"],
                embedding=doc_embedder.embed_text(
                    "FinRL-DeepSeek: Risk-Sensitive Deep Reinforcement Learning Trading Agents",
                    is_query=False,
                ),
            ),
        ]
        admin_principal = PrincipalContext(
            user_id="demo_admin",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng", "public_quant", "quant_restricted"],
        )
        return {
            "admin_principal": admin_principal,
            "code_domain_id": CODE_DOMAIN_ID_FINANCE,
            "research_domain_id": DOMAIN_ID_FINANCE,
            "code_files": code_files,
            "ast_symbols": ast_symbols,
            "doc_chunks": doc_chunks,
            "federated_fallback_chunks": {
                "chk_quant_default": (
                    "Order flow imbalance exhibits mean reversion over sub-second horizons."
                )
            },
        }

    def get_tuning_specs(self) -> dict[str, Any]:
        """Returns finance-specific distillation, training, and benchmark metadata."""
        return {
            "code_domain_id": CODE_DOMAIN_ID_FINANCE,
            "research_domain_id": DOMAIN_ID_FINANCE,
            "federated_domain_id": FEDERATED_DOMAIN_ID_FINANCE,
            "fallback_symbol_file": (
                "src/context_scout/trading_strategies/almgren_chriss_execution.py"
            ),
            "system_scopes": [
                "public_eng",
                "public_quant",
                "repo:trading_strategies",
                "corpus:quant_papers",
                "quant_restricted",
            ],
            "adapter_specs": self.get_adapter_specs(),
            "market_configs": [
                {
                    "market": MarketRegime.US_NYSE_NASDAQ,
                    "bps": 12.0,
                    "topics": [
                        (
                            "Order Flow Imbalance High-Frequency Alpha",
                            r"\Delta P_t = \lambda \cdot \text{OFI}_t + \varepsilon_t",
                            ["Enter when OFI z-score > 2.0", "Exit on quote imbalance exhaustion"],
                        ),
                        (
                            "Cross-Sectional Volatility-Managed Momentum",
                            (
                                r"w_{i,t} = \frac{\sigma_{\text{target}}}{\hat{\sigma}_{i,t}} "
                                r"\cdot R_{i,t-12:t-1}"
                            ),
                            [
                                "Rebalance at monthly boundary",
                                "Cut position if drawdown exceeds 5%",
                            ],
                        ),
                        (
                            "Almgren-Chriss Optimal Liquidation Trajectory",
                            r"x_j = \frac{\sinh(\kappa (T - t_j))}{\sinh(\kappa T)} X_0",
                            ["Execute TWAP intervals", "Halt on volume spike > 3 sigma"],
                        ),
                    ],
                },
                {
                    "market": MarketRegime.GERMANY_XETRA_DAX,
                    "bps": 15.0,
                    "topics": [
                        (
                            "DAX Continuous Auction Spread Mean-Reversion",
                            r"S_t = \alpha + \beta_1 \cdot \text{DepthRatio}_t + \varepsilon_t",
                            [
                                "Quote bid-ask spread at 95th percentile",
                                "Cancel unexecuted orders in 500ms",
                            ],
                        ),
                        (
                            "XETRA Microstructure Limit Order Book Imbalance",
                            r"\text{LOB}_t = \frac{V_t^b - V_t^a}{V_t^b + V_t^a}",
                            [
                                "Enter aggressive market order when |LOB| > 0.4",
                                "Exit on book replenishment",
                            ],
                        ),
                    ],
                },
                {
                    "market": MarketRegime.INDIA_NSE_BSE,
                    "bps": 25.0,
                    "topics": [
                        (
                            "NSE Nifty 50 Intraday Futures Momentum with STT Barrier",
                            r"\text{Signal}_t = \text{EMA}_{10}(P_t) - \text{EMA}_{30}(P_t)",
                            [
                                "Execute only if expected alpha > 25 bps STT floor",
                                "Square off at 15:15 IST",
                            ],
                        ),
                        (
                            "NSE Continuous Double Auction Liquidity Replenishment",
                            r"R_t = \frac{\sum Q_{\text{replenish}}}{\sum Q_{\text{consumed}}}",
                            [
                                "Place passive limit orders inside spread",
                                "Halt during opening 15 minutes",
                            ],
                        ),
                    ],
                },
            ],
        }

    def build_research_distillation_sample(
        self,
        domain_id: str,
        target_chunk: Any,
        quote_text: str,
        rng: Any,
    ) -> dict[str, Any]:
        """Builds valid/invalid QuantStrategyBrief sample payloads for teacher distillation."""
        market_configs = self.get_tuning_specs()["market_configs"]
        cfg = next(
            (m for m in market_configs if m["market"] == target_chunk.market_scope),
            market_configs[0],
        )
        topic_title, math_latex, entry_exit = rng.choice(cfg["topics"])
        scope_label = (
            target_chunk.market_scope.value
            if hasattr(target_chunk.market_scope, "value")
            else str(target_chunk.market_scope or MarketRegime.GLOBAL_MULTI.value)
        )
        citation = DocCitationSpan(
            chunk_id=str(target_chunk.chunk_id),
            doc_id=target_chunk.doc_id,
            doc_title=target_chunk.doc_title,
            source_uri=target_chunk.source_uri,
            section_heading=target_chunk.section_heading,
            verbatim_quote=quote_text,
        )
        brief = QuantStrategyBrief(
            strategy_name=f"{topic_title} ({scope_label})",
            target_markets=[target_chunk.market_scope or MarketRegime.GLOBAL_MULTI],
            is_intraday_futures=False,
            mathematical_signal_latex=math_latex,
            entry_exit_rules=entry_exit,
            sample_period_years="2018-2024",
            reported_sharpe_ratio=round(rng.uniform(1.4, 2.2), 2),
            reported_max_drawdown_pct=round(rng.uniform(6.0, 14.0), 1),
            transaction_cost_bps_assumed=cfg["bps"],
            friction_verdict=FrictionVerdict.REALISTIC_FRICTION,
            market_specific_caveats=[
                f"Requires minimum {cfg['bps']} bps friction barrier to overcome exchange costs",
                "Excludes high-volatility scheduled macroeconomic announcement windows",
            ],
            known_failure_regimes=[
                "Severe liquidity dry-ups and widening bid-ask spreads",
                "Structural regulatory margin increases",
            ],
            verified_citations=[citation],
        )
        bad_citation = DocCitationSpan(
            chunk_id=str(target_chunk.chunk_id),
            doc_id=target_chunk.doc_id,
            doc_title=target_chunk.doc_title,
            source_uri=target_chunk.source_uri,
            section_heading=target_chunk.section_heading,
            verbatim_quote="Hallucinated econometric quote not present in paper text.",
        )
        bad_brief = brief.model_copy(update={"verified_citations": [bad_citation]})
        val_user_prompt = (
            f"Synthesize a market-friction verified QuantStrategyBrief for "
            f"`{target_chunk.doc_title}` in the {scope_label} regime."
        )
        user_prompt = (
            f"{val_user_prompt}\n"
            f"Retrieved Corpus Chunk: [chunk_id={target_chunk.chunk_id} | "
            f"doc_id={target_chunk.doc_id} | title={target_chunk.doc_title} | "
            f"uri={target_chunk.source_uri} | section={target_chunk.section_heading} | "
            f"strategy={brief.strategy_name} | sharpe={brief.reported_sharpe_ratio} | "
            f"mdd={brief.reported_max_drawdown_pct} | bps={cfg['bps']} | "
            f"quote={json.dumps(quote_text)}]"
        )
        return {
            "brief": brief,
            "bad_brief": bad_brief,
            "brief_kwarg": {"quant_strategy_spec": brief},
            "bad_brief_kwarg": {"quant_strategy_spec": bad_brief},
            "citation": citation,
            "bad_citation": bad_citation,
            "val_user_prompt": val_user_prompt,
            "user_prompt": user_prompt,
        }

    def verify_research_brief_output(
        self,
        raw_output: str,
    ) -> tuple[bool, list[str], list[str]]:
        """Validates raw QuantStrategyBrief or DomainKnowledgeBrief JSON output."""
        cleaned = raw_output.strip()
        try:
            brief = QuantStrategyBrief.model_validate_json(cleaned)
            is_valid = bool(
                brief.strategy_name
                and brief.target_markets
                and brief.mathematical_signal_latex
                and brief.friction_verdict
            )
            doc_ids = [c.doc_id for c in brief.verified_citations[:5] if c.doc_id]
            titles = [c.doc_title for c in brief.verified_citations[:5] if c.doc_title]
            return is_valid, doc_ids, titles
        except Exception as exc:
            logger.debug(
                "QuantStrategyBrief validation fell back to DomainKnowledgeBrief (%s): %s",
                type(exc).__name__,
                exc,
            )
        try:
            kb_brief = DomainKnowledgeBrief.model_validate_json(cleaned)
            is_valid = bool(kb_brief.brief_title and kb_brief.executive_summary)
            doc_ids = [c.doc_id for c in kb_brief.verified_citations[:5] if c.doc_id]
            titles = [c.doc_title for c in kb_brief.verified_citations[:5] if c.doc_title]
            return is_valid, doc_ids, titles
        except Exception as exc:
            logger.debug(
                "Research brief JSON validation failed (%s): %s",
                type(exc).__name__,
                exc,
            )
            return False, [], []

    def evaluate_financial_rules(
        self,
        brief: QuantStrategyBrief,
    ) -> FinancialEvaluationResult:
        """Evaluates transaction cost assumptions, net Sharpe, and max drawdown."""
        floor_bps = calculate_effective_friction_floor(
            target_markets=brief.target_markets,
            is_intraday_futures=brief.is_intraday_futures,
        )
        assumed_bps = brief.transaction_cost_bps_assumed
        net_sharpe = estimate_net_sharpe(
            reported_sharpe=brief.reported_sharpe_ratio,
            assumed_bps=assumed_bps,
            floor_bps=floor_bps,
        )

        # 1. Determine friction verdict
        if assumed_bps is None or assumed_bps < floor_bps:
            verdict = FrictionVerdict.UNVERIFIED_FRICTION_WARNING
            friction_reason = (
                f"Assumed friction ({assumed_bps} bps) is below market floor "
                f"({floor_bps:.1f} bps) for {brief.target_markets}."
            )
        elif net_sharpe is not None and net_sharpe <= MIN_NET_SHARPE_THRESHOLD:
            verdict = FrictionVerdict.FAILS_AFTER_COSTS
            friction_reason = (
                f"Net Sharpe ratio ({net_sharpe}) after institutional friction "
                f"({floor_bps:.1f} bps) falls <= {MIN_NET_SHARPE_THRESHOLD} threshold."
            )
        elif (
            brief.reported_max_drawdown_pct is not None
            and abs(brief.reported_max_drawdown_pct) > MAX_ALLOWABLE_DRAWDOWN_PCT
        ):
            verdict = FrictionVerdict.FAILS_AFTER_COSTS
            friction_reason = (
                f"Reported maximum drawdown ({brief.reported_max_drawdown_pct}%) "
                f"exceeds {MAX_ALLOWABLE_DRAWDOWN_PCT:.1f}% institutional risk threshold."
            )
        else:
            verdict = FrictionVerdict.REALISTIC_FRICTION
            friction_reason = (
                f"Transaction cost assumptions ({assumed_bps} bps) satisfy "
                f"market floor ({floor_bps:.1f} bps)."
            )

        passed = verdict == FrictionVerdict.REALISTIC_FRICTION
        return FinancialEvaluationResult(
            passed=passed,
            friction_verdict=verdict,
            effective_friction_floor_bps=floor_bps,
            assumed_friction_bps=assumed_bps,
            estimated_net_sharpe=net_sharpe,
            reason=friction_reason,
        )

    def evaluate_strategy_brief(
        self,
        brief: QuantStrategyBrief,
        corpus_chunks: list[DocChunkRecord] | list[DocumentChunk] | dict[str, str],
    ) -> ScepticismEvaluationResult:
        """Composes pure financial evaluation with generic DocumentCitationGate."""
        fin_res = self.evaluate_financial_rules(brief)

        citation_res = self.citation_gate.verify_citations(
            citations=brief.verified_citations,
            corpus_chunks=corpus_chunks,
            domain_id=self.domain_id,
            require_citations=True,
        )

        passed = fin_res.passed and citation_res.passed

        reasons = [fin_res.reason]
        if not citation_res.passed:
            reasons.append(
                f"{citation_res.failed_count} of {citation_res.total_citations} "
                "citations failed verbatim verification."
            )

        return ScepticismEvaluationResult(
            passed=passed,
            friction_verdict=fin_res.friction_verdict,
            citations_valid=citation_res.passed,
            citations_verified_count=citation_res.verified_count,
            failed_citations=citation_res.failed_citations,
            reason=" | ".join(reasons),
            effective_friction_floor_bps=fin_res.effective_friction_floor_bps,
            assumed_friction_bps=fin_res.assumed_friction_bps,
            estimated_net_sharpe=fin_res.estimated_net_sharpe,
            citation_gate_result=citation_res,
        )


__all__ = [
    "AUTO_RESEARCH_KEYWORDS",
    "AUTO_RESEARCH_REGEX",
    "CODE_DOMAIN_ID_FINANCE",
    "DOMAIN_ID_FINANCE",
    "FEDERATED_DOMAIN_ID_FINANCE",
    "MARKET_MIN_FRICTION_BPS",
    "MAX_ALLOWABLE_DRAWDOWN_PCT",
    "MIN_NET_SHARPE_THRESHOLD",
    "FinanceDomainPlugin",
    "FinancialEvaluationResult",
    "ScepticismEvaluationResult",
    "calculate_effective_friction_floor",
    "estimate_net_sharpe",
]
