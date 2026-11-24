"""Declarative SpecializationCompiler for all 5 domain specialization archetypes.

Compiles ``domains/<id>/specialization.yaml`` into a strongly typed, architecture-aware
``CompiledSpecialization`` bundle with:
- Full support for all 5 archetypes:
  1. ``pointer_localization`` (``CodeSlicePointer`` arrays/envelopes)
  2. ``extractive_synthesis`` (``DomainKnowledgeBrief`` with verbatim ``DocCitationSpan``s)
  3. ``compliance_security_audit`` (``ComplianceSecurityAuditReport``)
  4. ``change_impact_mr_review`` (``ChangeImpactMRReviewReport``)
  5. ``custom_json_contract`` (arbitrary JSON Schema + safe ``validator_plugin`` hook)
- Fail-closed OWASP ASI06 & vLLM multi-LoRA policy enforcement (non-empty
  ``training_scope_ceiling``, ``rank in {8, 16, 32}``, ``use_rslora=True``,
  ``use_dora=False``, ``assistant_only_loss=True``, ``general_replay_ratio >= 0.10``).
- Vendor-neutral prompt formatting delegated to ``ModelArchitectureAdapter`` and
  ``AutoTokenizer.apply_chat_template`` with optional ``glossary.yaml`` injection.
- Deterministic output validation and actionable ``bounce_back_feedback`` generation.
"""

from __future__ import annotations

import importlib
import json
import logging
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Final, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from context_scout.agents.open_weights_vllm_executor import (
    ModelArchitectureAdapter,
    ModelArchitectureProfile,
    detect_architecture_profile,
)
from context_scout.config import ModelGovernancePolicy
from context_scout.db.repositories import ASTSymbolRecord, CodeFileRecord, DocChunkRecord
from context_scout.retrieval.reranker import DomainGlossary, load_domain_glossary
from context_scout.schemas.contracts import (
    CodeSlicePointer,
    DocCitationSpan,
    DomainKnowledgeBrief,
    parse_domain_knowledge_brief_json,
)

logger = logging.getLogger(__name__)

VALID_SPECIALIZATION_RANKS: Final[frozenset[int]] = frozenset({8, 16, 32})
_SAFE_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_PLUGIN_MODULE_RE: Final[re.Pattern[str]] = re.compile(
    r"^context_scout(?:\.[A-Za-z_][A-Za-z0-9_]*)+$"
)
_SAFE_PLUGIN_FUNC_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NESTED_QUANTIFIER_RE: Final[re.Pattern[str]] = re.compile(
    r"\([^)]*[+*][^)]*\)[+*?]|\([^)]*\|[^)]*\)[+*]"
)
_MAX_SCHEMA_REGEX_INPUT_LEN: Final[int] = 4096

ValidatorCallable = Callable[[Any], bool | tuple[bool, str | Sequence[str]] | Sequence[str] | None]


class SpecializationCompilationError(ValueError):
    """Raised fail-closed when a specialization.yaml manifest or plugin violates policy."""


class SpecializationArchetype(StrEnum):
    """Supported specialization.yaml archetypes for self-hosted LoRA cartridges."""

    POINTER_LOCALIZATION = "pointer_localization"
    EXTRACTIVE_SYNTHESIS = "extractive_synthesis"
    COMPLIANCE_SECURITY_AUDIT = "compliance_security_audit"
    CHANGE_IMPACT_MR_REVIEW = "change_impact_mr_review"
    CUSTOM_JSON_CONTRACT = "custom_json_contract"


class LoRAPolicySpec(BaseModel):
    """Fail-closed LoRA policy contract aligned with vLLM multi-LoRA & replay safety."""

    enabled: bool = True
    rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.0, ge=0.0, le=0.5)
    use_rslora: bool = True
    use_dora: bool = False
    loraplus_lr_ratio: float = Field(default=8.0, ge=1.0, le=64.0)
    neftune_noise_alpha: float = Field(default=5.0, ge=0.0, le=20.0)
    general_replay_ratio: float = Field(default=0.15, ge=0.0, le=0.5)
    assistant_only_loss: bool = True
    dpo_beta: float = Field(default=0.1, gt=0.0, le=2.0)
    cached_mnrl_enabled: bool = False

    @model_validator(mode="after")
    def _enforce_vllm_and_forgetting_guards(self) -> LoRAPolicySpec:
        if not self.enabled:
            return self
        if self.use_dora:
            raise ValueError("use_dora=True is forbidden (breaks vLLM multi-LoRA kernels)")
        if not self.use_rslora:
            raise ValueError("use_rslora=True is required for rank-stabilized multi-LoRA")
        if not self.assistant_only_loss:
            raise ValueError(
                "assistant_only_loss=True is required to prevent canary/prompt memorization"
            )
        if self.rank not in VALID_SPECIALIZATION_RANKS:
            raise ValueError(
                f"LoRA rank {self.rank} violates vLLM policy (rank <= 32, allowed: "
                f"{sorted(VALID_SPECIALIZATION_RANKS)})"
            )
        if self.general_replay_ratio < 0.10 or self.general_replay_ratio > 0.50:
            raise ValueError(
                f"general_replay_ratio={self.general_replay_ratio} must be in [0.10, 0.50] "
                "(target 0.15 replay ratio)"
            )
        return self


class SpecializationManifest(BaseModel):
    """Validated declarative schema for ``domains/<id>/specialization.yaml``."""

    domain_id: str = Field(min_length=1)
    display_name: str = ""
    archetype: SpecializationArchetype = SpecializationArchetype.POINTER_LOCALIZATION
    base_model_id: str = "google/gemma-4-26B-A4B-it"
    training_scope_ceiling: list[str] = Field(default_factory=list)
    output_schema_path: str | None = None
    output_schema: dict[str, Any] | None = None
    validator_plugin: str | None = None
    glossary_path: str | None = None
    lora_adapter_id: str | None = None
    lora_adapter_path: str | None = None
    lora_policy: LoRAPolicySpec = Field(default_factory=LoRAPolicySpec)

    @field_validator("domain_id")
    @classmethod
    def _validate_domain_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not _SAFE_SLUG_RE.fullmatch(cleaned) or ".." in cleaned:
            raise ValueError(f"Unsafe or invalid domain_id '{value}'")
        return cleaned

    @field_validator("base_model_id")
    @classmethod
    def _validate_governed_base_model(cls, value: str) -> str:
        return ModelGovernancePolicy.default().validate_local_model_id(value)

    @field_validator("training_scope_ceiling")
    @classmethod
    def _validate_scope_tokens(cls, scopes: list[str]) -> list[str]:
        cleaned: list[str] = []
        for raw in scopes:
            tok = str(raw).strip()
            if not tok:
                continue
            if "," in tok or ";" in tok or " " in tok:
                raise ValueError(f"Invalid training_scope_ceiling scope token: {tok!r}")
            if tok not in cleaned:
                cleaned.append(tok)
        return cleaned

    @model_validator(mode="after")
    def _validate_cross_field_contracts(self) -> SpecializationManifest:
        if self.lora_policy.enabled and not self.training_scope_ceiling:
            raise ValueError(
                f"{self.domain_id}: training_scope_ceiling must be non-empty when LoRA is enabled"
            )
        if (
            self.archetype == SpecializationArchetype.CUSTOM_JSON_CONTRACT
            and not self.output_schema_path
            and self.output_schema is None
        ):
            raise ValueError(
                f"{self.domain_id}: archetype 'custom_json_contract' requires "
                "output_schema_path or output_schema"
            )
        return self


class PointerLocalizationEnvelope(BaseModel):
    """Canonical structured envelope wrapping one or more CodeSlicePointers."""

    pointers: list[CodeSlicePointer] = Field(min_length=1)


class ComplianceAuditFinding(BaseModel):
    """Single verifiable compliance or security audit finding with mandatory grounding."""

    control_id: str = Field(min_length=1)
    rule_or_policy: str = Field(min_length=1)
    status: Literal["pass", "fail", "warning"]
    remediation: str = Field(min_length=1)
    evidence_pointer: CodeSlicePointer | None = None
    evidence_citation: DocCitationSpan | None = None

    @model_validator(mode="after")
    def _require_evidence(self) -> ComplianceAuditFinding:
        if self.evidence_pointer is None and self.evidence_citation is None:
            raise ValueError(
                f"ComplianceAuditFinding '{self.control_id}' must attach either "
                "evidence_pointer or evidence_citation"
            )
        return self


class ComplianceSecurityAuditReport(BaseModel):
    """Structured output contract for ``compliance_security_audit`` archetype."""

    domain_id: str = Field(min_length=1)
    audit_verdict: Literal[
        "compliant", "non_compliant", "needs_remediation", "insufficient_evidence"
    ]
    severity: Literal["critical", "high", "medium", "low", "info"]
    executive_summary: str = Field(min_length=1)
    findings: list[ComplianceAuditFinding] = Field(default_factory=list)
    verified_citations: list[DocCitationSpan] = Field(default_factory=list)
    code_pointers: list[CodeSlicePointer] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_findings_unless_insufficient(self) -> ComplianceSecurityAuditReport:
        if self.audit_verdict != "insufficient_evidence" and not self.findings:
            raise ValueError(
                "ComplianceSecurityAuditReport requires at least one finding unless "
                "audit_verdict == 'insufficient_evidence'"
            )
        return self


class ChangeImpactMRReviewReport(BaseModel):
    """Structured output contract for ``change_impact_mr_review`` archetype."""

    domain_id: str = Field(min_length=1)
    mr_or_commit_ref: str = Field(min_length=1)
    risk_level: Literal["critical", "high", "medium", "low"]
    review_summary: str = Field(min_length=1)
    impacted_slices: list[CodeSlicePointer] = Field(min_length=1)
    side_effect_risks: list[str] = Field(default_factory=list)
    linked_ticket_keys: list[str] = Field(default_factory=list)
    spec_drift_warnings: list[str] = Field(default_factory=list)
    supporting_citations: list[DocCitationSpan] = Field(default_factory=list)


@dataclass(frozen=True)
class SpecializationValidationResult:
    """Outcome of deterministic schema + grounding validation for an archetype output."""

    archetype: SpecializationArchetype
    passed: bool
    parsed_payload: Any
    canonical_json: str
    violations: tuple[str, ...] = ()
    bounce_back_feedback: str | None = None


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def extract_json_from_model_output(raw_output: str) -> str:
    """Strips reasoning blocks, structural tags, and Markdown fences around JSON."""
    cleaned = re.sub(r"<\|channel>thought.*?<channel\|>", "", raw_output, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    tag_match = re.search(r"<json_output>(.*?)</json_output>", cleaned, flags=re.DOTALL)
    if tag_match:
        cleaned = tag_match.group(1)
    cleaned = cleaned.replace("<turn|>", "").replace("<eos>", "").strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    return cleaned


def _validate_json_schema_node(
    instance: Any,
    schema: Mapping[str, Any],
    path: str = "$",
) -> list[str]:
    """Deterministic O(n)-bounded JSON Schema validator for ``custom_json_contract``."""
    violations: list[str] = []
    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        type_ok = (
            (expected_type == "object" and isinstance(instance, dict))
            or (expected_type == "array" and isinstance(instance, list))
            or (expected_type == "string" and isinstance(instance, str))
            or (
                expected_type == "integer"
                and isinstance(instance, int)
                and not isinstance(instance, bool)
            )
            or (
                expected_type == "number"
                and isinstance(instance, (int, float))
                and not isinstance(instance, bool)
            )
            or (expected_type == "boolean" and isinstance(instance, bool))
            or (expected_type == "null" and instance is None)
        )
        if not type_ok:
            violations.append(
                f"Schema type mismatch at {path}: expected '{expected_type}', "
                f"got {type(instance).__name__}"
            )
            return violations

    if "enum" in schema:
        allowed_enum = schema["enum"]
        if isinstance(allowed_enum, Sequence) and instance not in allowed_enum:
            violations.append(
                f"Value {instance!r} at {path} not in allowed enum {list(allowed_enum)!r}"
            )

    if isinstance(instance, str):
        min_len = schema.get("minLength")
        if isinstance(min_len, int) and len(instance) < min_len:
            violations.append(f"String length {len(instance)} < minLength {min_len} at {path}")
        max_len = schema.get("maxLength")
        if isinstance(max_len, int) and len(instance) > max_len:
            violations.append(f"String length {len(instance)} > maxLength {max_len} at {path}")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and pattern:
            if _NESTED_QUANTIFIER_RE.search(pattern):
                violations.append(f"Unsafe nested quantifier regex pattern in schema at {path}")
            else:
                bounded_text = instance[:_MAX_SCHEMA_REGEX_INPUT_LEN]
                try:
                    if re.search(pattern, bounded_text) is None:
                        violations.append(f"String at {path} does not match pattern {pattern!r}")
                except re.error as exc:
                    violations.append(f"Invalid regex pattern {pattern!r} at {path}: {exc}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and instance < minimum:
            violations.append(f"Numeric value {instance} < minimum {minimum} at {path}")
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and instance > maximum:
            violations.append(f"Numeric value {instance} > maximum {maximum} at {path}")

    if isinstance(instance, dict):
        required = schema.get("required") or ()
        for req_key in required:
            if req_key not in instance:
                violations.append(f"Missing required property '{req_key}' at {path}")
        props = schema.get("properties") or {}
        if schema.get("additionalProperties") is False and isinstance(props, Mapping):
            extras = sorted(set(instance) - set(props))
            if extras:
                violations.append(f"Unexpected additionalProperties {extras} at {path}")
        if isinstance(props, Mapping):
            for key, sub_schema in props.items():
                if key in instance and isinstance(sub_schema, Mapping):
                    violations.extend(
                        _validate_json_schema_node(instance[key], sub_schema, path=f"{path}.{key}")
                    )
    elif isinstance(instance, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(instance) < min_items:
            violations.append(f"Array length {len(instance)} < minItems {min_items} at {path}")
        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and len(instance) > max_items:
            violations.append(f"Array length {len(instance)} > maxItems {max_items} at {path}")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for idx, elem in enumerate(instance):
                violations.extend(
                    _validate_json_schema_node(elem, item_schema, path=f"{path}[{idx}]")
                )

    return violations


def _build_pointer_localization_schema() -> dict[str, Any]:
    ptr_schema = CodeSlicePointer.model_json_schema()
    defs = dict(ptr_schema.pop("$defs", {}))
    return {
        "type": "array",
        "minItems": 1,
        "items": ptr_schema,
        **({"$defs": defs} if defs else {}),
    }


def _check_code_pointers(
    pointers: Sequence[CodeSlicePointer],
    *,
    code_files: Sequence[CodeFileRecord] | None = None,
    ast_symbols: Sequence[ASTSymbolRecord] | None = None,
) -> list[str]:
    violations: list[str] = []
    if code_files is not None:
        files_by_path: dict[str, list[CodeFileRecord]] = {}
        files_by_repo_path: dict[tuple[str, str], CodeFileRecord] = {}
        for cf in code_files:
            files_by_path.setdefault(cf.file_path, []).append(cf)
            files_by_repo_path[(cf.repo_name, cf.file_path)] = cf

        syms_by_file_id: dict[Any, set[str]] = {}
        if ast_symbols is not None:
            for sym in ast_symbols:
                syms_by_file_id.setdefault(sym.file_id, set()).add(sym.qualified_name)
                syms_by_file_id[sym.file_id].add(sym.qualified_name.split(".")[-1])
                if sym.bare_name:
                    syms_by_file_id[sym.file_id].add(sym.bare_name)

        for idx, ptr in enumerate(pointers):
            matched_file = files_by_repo_path.get((ptr.repo_name, ptr.file_path))
            if matched_file is None:
                cands = files_by_path.get(ptr.file_path, [])
                matched_file = cands[0] if cands else None
            if matched_file is None:
                violations.append(
                    f"Pointer[{idx}] references unindexed file '{ptr.repo_name}/{ptr.file_path}'"
                )
                continue
            if matched_file.line_count < 1 or ptr.end_line > matched_file.line_count:
                violations.append(
                    f"Pointer[{idx}] lines {ptr.start_line}-{ptr.end_line} exceed "
                    f"file '{ptr.file_path}' line_count={matched_file.line_count}"
                )
            if ast_symbols is not None and matched_file.file_id in syms_by_file_id:
                known_syms = syms_by_file_id[matched_file.file_id]
                short_q = ptr.qualified_symbol.split(".")[-1]
                if ptr.qualified_symbol not in known_syms and short_q not in known_syms:
                    violations.append(
                        f"Pointer[{idx}] qualified_symbol '{ptr.qualified_symbol}' "
                        f"not found in AST symbols for '{ptr.file_path}'"
                    )
    return violations


def _check_doc_citations(
    citations: Sequence[DocCitationSpan],
    *,
    doc_chunks: Sequence[DocChunkRecord] | None = None,
    require_non_empty: bool = False,
) -> list[str]:
    violations: list[str] = []
    if require_non_empty and not citations:
        violations.append("Missing verified_citations: at least one grounded citation is required")
        return violations
    if doc_chunks is None:
        return violations

    chunks_by_id = {
        str(chunk.chunk_id): _normalize_whitespace(chunk.verbatim_content) for chunk in doc_chunks
    }
    for idx, cit in enumerate(citations):
        cid = str(cit.chunk_id).strip()
        if cid not in chunks_by_id:
            violations.append(
                f"Citation[{idx}] references missing or unauthorized chunk_id '{cid}'"
            )
            continue
        q_norm = _normalize_whitespace(cit.verbatim_quote)
        if len(q_norm) < 12:
            violations.append(
                f"Citation[{idx}] verbatim_quote length ({len(q_norm)}) is below 12-char floor"
            )
            continue
        chunk_norm = chunks_by_id[cid]
        if q_norm in chunk_norm:
            continue
        q_tokens = re.findall(r"[A-Za-z0-9_]{2,}", q_norm.lower())
        if not q_tokens:
            violations.append(f"Citation[{idx}] verbatim_quote has zero verifiable tokens")
            continue
        chunk_low = chunk_norm.lower()
        overlap = sum(1 for tok in q_tokens if tok in chunk_low) / len(q_tokens)
        if overlap < 0.80:
            violations.append(
                f"Citation[{idx}] verbatim_quote failed grounding check against chunk '{cid}' "
                f"(token_overlap={overlap:.2f} < 0.80)"
            )
    return violations


@dataclass(frozen=True)
class CompiledSpecialization:
    """Compiled runtime & training bundle produced by ``SpecializationCompiler``."""

    manifest: SpecializationManifest
    architecture_profile: ModelArchitectureProfile
    architecture_adapter: ModelArchitectureAdapter
    system_prompt: str
    output_json_schema: dict[str, Any]
    glossary: DomainGlossary | None = None
    validator_callable: ValidatorCallable | None = None

    @property
    def domain_id(self) -> str:
        return self.manifest.domain_id

    @property
    def archetype(self) -> SpecializationArchetype:
        return self.manifest.archetype

    @property
    def training_scope_ceiling(self) -> tuple[str, ...]:
        return tuple(self.manifest.training_scope_ceiling)

    def format_messages(
        self,
        user_prompt: str,
        *,
        assistant_response: str | None = None,
        extra_turns: Sequence[Mapping[str, str]] = (),
        include_system: bool = True,
    ) -> list[dict[str, str]]:
        """Builds canonical chat turns without hardcoded vendor prompt tags."""
        messages: list[dict[str, str]] = []
        if include_system and self.system_prompt.strip():
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        for turn in extra_turns:
            role = str(turn.get("role") or "user").strip()
            content = str(turn.get("content") or "")
            messages.append({"role": role, "content": content})
        if assistant_response is not None:
            messages.append({"role": "assistant", "content": assistant_response})
        return messages

    def render_prompt(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenizer: Any | None = None,
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
    ) -> str:
        """Renders chat turns via ``ModelArchitectureAdapter`` & ``apply_chat_template``."""
        return self.architecture_adapter.format_chat_messages(
            messages,
            tokenizer=tokenizer,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    def expand_query_with_glossary(self, query: str) -> str:
        if self.glossary is None:
            return query
        return self.glossary.expand_query(query).expanded_query

    def validate_output(
        self,
        raw_output: str | Mapping[str, Any] | Sequence[Any],
        *,
        code_files: Sequence[CodeFileRecord] | None = None,
        ast_symbols: Sequence[ASTSymbolRecord] | None = None,
        doc_chunks: Sequence[DocChunkRecord] | None = None,
    ) -> SpecializationValidationResult:
        """Deterministically validates model output against the compiled archetype contract."""
        archetype = self.manifest.archetype
        violations: list[str] = []

        if isinstance(raw_output, str):
            cleaned = extract_json_from_model_output(raw_output)
            try:
                parsed_raw: Any = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                return SpecializationValidationResult(
                    archetype=archetype,
                    passed=False,
                    parsed_payload=None,
                    canonical_json="",
                    violations=(f"Invalid JSON payload: {exc}",),
                    bounce_back_feedback=(
                        "Pre-Flight Gate Validation FAILED ($0 Local Bounce-Back Required):\n"
                        f"- Return strictly valid JSON matching the '{archetype.value}' schema: "
                        f"{exc}"
                    ),
                )
        else:
            parsed_raw = raw_output

        if archetype == SpecializationArchetype.POINTER_LOCALIZATION:
            raw_list: Any = parsed_raw
            if isinstance(parsed_raw, Mapping) and "pointers" in parsed_raw:
                raw_list = parsed_raw["pointers"]
            if not isinstance(raw_list, list) or not raw_list:
                violations.append(
                    "pointer_localization output must be a non-empty JSON array of CodeSlicePointer"
                )
                return self._fail_result(violations, None)
            pointers: list[CodeSlicePointer] = []
            for idx, item in enumerate(raw_list):
                try:
                    pointers.append(CodeSlicePointer.model_validate(item))
                except ValidationError as exc:
                    violations.append(f"Pointer[{idx}] schema error: {exc}")
            if not violations:
                violations.extend(
                    _check_code_pointers(pointers, code_files=code_files, ast_symbols=ast_symbols)
                )
            if violations:
                return self._fail_result(violations, pointers)
            canonical = json.dumps([p.model_dump() for p in pointers], indent=2)
            return SpecializationValidationResult(
                archetype=archetype,
                passed=True,
                parsed_payload=pointers,
                canonical_json=canonical,
            )

        if archetype == SpecializationArchetype.EXTRACTIVE_SYNTHESIS:
            try:
                if isinstance(parsed_raw, Mapping):
                    brief = DomainKnowledgeBrief.model_validate(parsed_raw)
                else:
                    raise TypeError("extractive_synthesis output must be a JSON object")
            except (ValidationError, TypeError, ValueError) as exc:
                if isinstance(raw_output, str):
                    brief_norm, strict_ok = parse_domain_knowledge_brief_json(
                        extract_json_from_model_output(raw_output),
                        domain_id=self.domain_id,
                        normalize_scalars=True,
                    )
                    if brief_norm is not None and strict_ok:
                        brief = brief_norm
                    else:
                        return self._fail_result(
                            [f"DomainKnowledgeBrief validation failed: {exc}"], None
                        )
                else:
                    return self._fail_result(
                        [f"DomainKnowledgeBrief validation failed: {exc}"], None
                    )

            require_cits = brief.sufficiency_verdict != "insufficient"
            violations.extend(
                _check_doc_citations(
                    brief.verified_citations,
                    doc_chunks=doc_chunks,
                    require_non_empty=require_cits,
                )
            )
            if violations:
                return self._fail_result(violations, brief)
            return SpecializationValidationResult(
                archetype=archetype,
                passed=True,
                parsed_payload=brief,
                canonical_json=brief.model_dump_json(indent=2),
            )

        if archetype == SpecializationArchetype.COMPLIANCE_SECURITY_AUDIT:
            try:
                report = ComplianceSecurityAuditReport.model_validate(parsed_raw)
            except (ValidationError, TypeError, ValueError) as exc:
                return self._fail_result(
                    [f"ComplianceSecurityAuditReport validation failed: {exc}"], None
                )
            ptrs: list[CodeSlicePointer] = list(report.code_pointers)
            cits: list[DocCitationSpan] = list(report.verified_citations)
            for finding in report.findings:
                if finding.evidence_pointer is not None:
                    ptrs.append(finding.evidence_pointer)
                if finding.evidence_citation is not None:
                    cits.append(finding.evidence_citation)
            if ptrs:
                violations.extend(
                    _check_code_pointers(ptrs, code_files=code_files, ast_symbols=ast_symbols)
                )
            if cits:
                violations.extend(
                    _check_doc_citations(cits, doc_chunks=doc_chunks, require_non_empty=False)
                )
            if report.audit_verdict != "insufficient_evidence" and not ptrs and not cits:
                violations.append(
                    "Compliance report requires grounded evidence pointers or citations"
                )
            if violations:
                return self._fail_result(violations, report)
            return SpecializationValidationResult(
                archetype=archetype,
                passed=True,
                parsed_payload=report,
                canonical_json=report.model_dump_json(indent=2),
            )

        if archetype == SpecializationArchetype.CHANGE_IMPACT_MR_REVIEW:
            try:
                mr_report = ChangeImpactMRReviewReport.model_validate(parsed_raw)
            except (ValidationError, TypeError, ValueError) as exc:
                return self._fail_result(
                    [f"ChangeImpactMRReviewReport validation failed: {exc}"], None
                )
            violations.extend(
                _check_code_pointers(
                    mr_report.impacted_slices, code_files=code_files, ast_symbols=ast_symbols
                )
            )
            if mr_report.supporting_citations:
                violations.extend(
                    _check_doc_citations(
                        mr_report.supporting_citations,
                        doc_chunks=doc_chunks,
                        require_non_empty=False,
                    )
                )
            if violations:
                return self._fail_result(violations, mr_report)
            return SpecializationValidationResult(
                archetype=archetype,
                passed=True,
                parsed_payload=mr_report,
                canonical_json=mr_report.model_dump_json(indent=2),
            )

        # Archetype 5: custom_json_contract
        violations.extend(_validate_json_schema_node(parsed_raw, self.output_json_schema))
        if not violations and self.validator_callable is not None:
            try:
                plugin_res = self.validator_callable(parsed_raw)
            except Exception as exc:
                violations.append(f"validator_plugin raised {type(exc).__name__}: {exc}")
            else:
                if plugin_res is False:
                    violations.append("validator_plugin returned False")
                elif isinstance(plugin_res, tuple) and len(plugin_res) == 2:
                    ok, msg = plugin_res
                    if not ok:
                        if isinstance(msg, str):
                            violations.append(msg)
                        elif isinstance(msg, Sequence):
                            violations.extend(str(m) for m in msg)
                        else:
                            violations.append("validator_plugin returned failure tuple")
                elif isinstance(plugin_res, Sequence) and not isinstance(plugin_res, (str, bytes)):
                    violations.extend(str(m) for m in plugin_res)

        if violations:
            return self._fail_result(violations, parsed_raw)
        return SpecializationValidationResult(
            archetype=archetype,
            passed=True,
            parsed_payload=parsed_raw,
            canonical_json=json.dumps(parsed_raw, indent=2, sort_keys=True),
        )

    def _fail_result(
        self, violations: Sequence[str], parsed_payload: Any
    ) -> SpecializationValidationResult:
        clean_v = tuple(str(v) for v in violations if str(v).strip())
        bullet_list = "\n".join(f"- {v}" for v in clean_v)
        feedback = (
            "Pre-Flight Gate Validation FAILED ($0 Local Bounce-Back Required):\n"
            f"{bullet_list}\n"
            "Fix every schema or grounding violation above and re-emit strictly grounded JSON."
        )
        return SpecializationValidationResult(
            archetype=self.manifest.archetype,
            passed=False,
            parsed_payload=parsed_payload,
            canonical_json="",
            violations=clean_v,
            bounce_back_feedback=feedback,
        )


class SpecializationCompiler:
    """Compiles declarative ``specialization.yaml`` packs across all 5 enterprise archetypes."""

    _GLOBAL_VALIDATOR_REGISTRY: ClassVar[dict[str, ValidatorCallable]] = {}

    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        domains_dir: Path | None = None,
        validator_registry: Mapping[str, ValidatorCallable] | None = None,
        governance_policy: ModelGovernancePolicy | None = None,
    ) -> None:
        default_root = Path(__file__).resolve().parents[3]
        self.workspace_root = (workspace_root or default_root).resolve()
        self.domains_dir = (domains_dir or (self.workspace_root / "domains")).resolve()
        self.policy = governance_policy or ModelGovernancePolicy.default()
        self._validators: dict[str, ValidatorCallable] = dict(self._GLOBAL_VALIDATOR_REGISTRY)
        if validator_registry:
            self._validators.update(validator_registry)

    @classmethod
    def register_validator_plugin(cls, key: str, fn: ValidatorCallable) -> None:
        """Registers an explicit named validator callable for ``custom_json_contract`` packs."""
        clean_key = key.strip()
        if not clean_key:
            raise ValueError("Validator plugin key cannot be empty")
        cls._GLOBAL_VALIDATOR_REGISTRY[clean_key] = fn

    @classmethod
    def unregister_validator_plugin(cls, key: str) -> None:
        cls._GLOBAL_VALIDATOR_REGISTRY.pop(key.strip(), None)

    def _resolve_safe_path(self, raw_path: str, *, field_name: str) -> Path:
        cleaned = raw_path.strip()
        if not cleaned or ".." in Path(cleaned).parts:
            raise SpecializationCompilationError(
                f"Unsafe path traversal in {field_name}: {raw_path!r}"
            )
        candidate = Path(cleaned)
        if not candidate.is_absolute():
            candidate = (self.workspace_root / candidate).resolve()
        else:
            candidate = candidate.resolve()

        allowed_roots = (self.workspace_root, self.domains_dir)
        if not any(candidate.is_relative_to(root) for root in allowed_roots):
            raise SpecializationCompilationError(
                f"{field_name} path '{candidate}' escapes workspace/domains root"
            )
        return candidate

    def _resolve_validator_plugin(self, plugin_spec: str | None) -> ValidatorCallable | None:
        if plugin_spec is None or not plugin_spec.strip():
            return None
        spec = plugin_spec.strip()
        if spec in self._validators:
            return self._validators[spec]

        if ":" not in spec or "/" in spec or "\\" in spec or ".." in spec:
            raise SpecializationCompilationError(
                f"Untrusted or unregistered validator_plugin '{spec}' (ASI05 fail-closed)"
            )
        mod_name, func_name = spec.split(":", 1)
        if not _SAFE_PLUGIN_MODULE_RE.fullmatch(mod_name) or not _SAFE_PLUGIN_FUNC_RE.fullmatch(
            func_name
        ):
            raise SpecializationCompilationError(
                f"Untrusted validator_plugin '{spec}': only registered plugins or "
                "'context_scout.*' module paths are permitted (ASI05 fail-closed)"
            )
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, func_name, None)
        if not callable(fn):
            raise SpecializationCompilationError(
                f"validator_plugin '{spec}' target is not callable"
            )
        return fn  # type: ignore[no-any-return]

    def _resolve_output_schema(self, manifest: SpecializationManifest) -> dict[str, Any]:
        arch = manifest.archetype
        if arch == SpecializationArchetype.POINTER_LOCALIZATION:
            return _build_pointer_localization_schema()
        if arch == SpecializationArchetype.EXTRACTIVE_SYNTHESIS:
            return DomainKnowledgeBrief.model_json_schema()
        if arch == SpecializationArchetype.COMPLIANCE_SECURITY_AUDIT:
            return ComplianceSecurityAuditReport.model_json_schema()
        if arch == SpecializationArchetype.CHANGE_IMPACT_MR_REVIEW:
            return ChangeImpactMRReviewReport.model_json_schema()

        # custom_json_contract
        if manifest.output_schema is not None:
            return dict(manifest.output_schema)
        assert manifest.output_schema_path is not None
        schema_path = self._resolve_safe_path(
            manifest.output_schema_path, field_name="output_schema_path"
        )
        if not schema_path.is_file():
            raise SpecializationCompilationError(
                f"output_schema_path does not exist: {schema_path}"
            )
        try:
            loaded = json.loads(schema_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SpecializationCompilationError(
                f"Malformed JSON in output_schema_path '{schema_path}': {exc}"
            ) from exc
        if not isinstance(loaded, dict):
            raise SpecializationCompilationError("output_schema_path root must be a JSON object")
        return loaded

    def _resolve_glossary(self, manifest: SpecializationManifest) -> DomainGlossary | None:
        if manifest.glossary_path:
            g_path = self._resolve_safe_path(manifest.glossary_path, field_name="glossary_path")
            if not g_path.is_file():
                raise SpecializationCompilationError(f"glossary_path does not exist: {g_path}")
            return DomainGlossary.from_yaml(g_path, domain_id=manifest.domain_id)
        return load_domain_glossary(manifest.domain_id, domains_dir=self.domains_dir)

    @staticmethod
    def _build_system_prompt(
        manifest: SpecializationManifest,
        output_schema: Mapping[str, Any],
        glossary: DomainGlossary | None,
    ) -> str:
        arch = manifest.archetype
        role_title = manifest.display_name.strip() or manifest.domain_id
        schema_compact = json.dumps(output_schema, separators=(",", ":"), sort_keys=True)

        if arch == SpecializationArchetype.POINTER_LOCALIZATION:
            task_rules = (
                "Emit ONLY a JSON array of CodeSlicePointer objects referencing verified "
                "repo_name, file_path, start_line, end_line, and qualified_symbol. "
                "Never emit raw source code from memory."
            )
        elif arch == SpecializationArchetype.EXTRACTIVE_SYNTHESIS:
            task_rules = (
                "Emit ONLY a valid DomainKnowledgeBrief JSON object. Every claim MUST cite a "
                "chunk_id and verbatim_quote copied verbatim from retrieved chunks. If evidence "
                "is missing or insufficient, set sufficiency_verdict='insufficient' and abstain "
                "at $0 paid tokens."
            )
        elif arch == SpecializationArchetype.COMPLIANCE_SECURITY_AUDIT:
            task_rules = (
                "Emit ONLY a valid ComplianceSecurityAuditReport JSON object. Every finding MUST "
                "attach at least one verified CodeSlicePointer or DocCitationSpan with exact "
                "verbatim quotes. Abstain with audit_verdict='insufficient_evidence' if unverified."
            )
        elif arch == SpecializationArchetype.CHANGE_IMPACT_MR_REVIEW:
            task_rules = (
                "Emit ONLY a valid ChangeImpactMRReviewReport JSON object linking modified "
                "AST CodeSlicePointers, side-effect risks, Jira ticket keys, and specification "
                "drift warnings strictly grounded in retrieved context."
            )
        else:
            task_rules = (
                "Emit ONLY valid JSON strictly conforming to the domain output_schema below. "
                "Every identifier and quote must trace directly to retrieved context."
            )

        glossary_section = ""
        if glossary is not None and glossary.entries:
            terms_preview = "; ".join(
                f"{e.term} ({', '.join(e.aliases[:3])}): {e.definition[:80]}"
                if e.aliases
                else f"{e.term}: {e.definition[:80]}"
                for e in glossary.entries[:12]
            )
            glossary_section = f"\nDomain Glossary: {terms_preview}"

        ceiling_str = ", ".join(manifest.training_scope_ceiling)
        return (
            f"You are the specialized '{role_title}' ({manifest.domain_id}) agent running in "
            f"archetype '{arch.value}' with training_scope_ceiling=[{ceiling_str}].\n"
            f"{task_rules}{glossary_section}\n"
            f"Output JSON Schema: {schema_compact}"
        )

    def load_manifest(
        self,
        manifest_or_path: SpecializationManifest | Mapping[str, Any] | Path | str,
        *,
        domain_yaml_path: Path | None = None,
    ) -> SpecializationManifest:
        """Loads and validates a ``specialization.yaml`` manifest fail-closed."""
        if isinstance(manifest_or_path, SpecializationManifest):
            return manifest_or_path

        if isinstance(manifest_or_path, (Path, str)):
            spec_path = Path(manifest_or_path)
            if not spec_path.is_file():
                raise FileNotFoundError(f"specialization.yaml not found: {spec_path}")
            try:
                raw_obj = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                raise SpecializationCompilationError(f"Invalid YAML in {spec_path}: {exc}") from exc
            if not isinstance(raw_obj, Mapping):
                raise SpecializationCompilationError(f"{spec_path} must contain a YAML mapping")
            data: dict[str, Any] = dict(raw_obj)
            if domain_yaml_path is None:
                cand_domain = spec_path.parent / "domain.yaml"
                if cand_domain.is_file():
                    domain_yaml_path = cand_domain
        elif isinstance(manifest_or_path, Mapping):
            data = dict(manifest_or_path)
        else:
            raise TypeError(
                f"Expected SpecializationManifest, mapping, or path; got {type(manifest_or_path)}"
            )

        if domain_yaml_path is not None and domain_yaml_path.is_file():
            try:
                dom_raw = yaml.safe_load(domain_yaml_path.read_text(encoding="utf-8")) or {}
                if isinstance(dom_raw, Mapping):
                    data.setdefault("domain_id", dom_raw.get("domain_id"))
                    data.setdefault("display_name", dom_raw.get("display_name", ""))
                    data.setdefault("base_model_id", dom_raw.get("base_model_id"))
                    data.setdefault("lora_adapter_id", dom_raw.get("lora_adapter_id"))
                    data.setdefault("lora_adapter_path", dom_raw.get("lora_adapter_path"))
                    if "archetype" not in data:
                        contract = str(dom_raw.get("output_contract") or "")
                        if contract == "DomainKnowledgeBrief":
                            data["archetype"] = SpecializationArchetype.EXTRACTIVE_SYNTHESIS.value
                        else:
                            data["archetype"] = SpecializationArchetype.POINTER_LOCALIZATION.value
            except yaml.YAMLError as exc:
                raise SpecializationCompilationError(
                    f"Invalid domain.yaml at {domain_yaml_path}: {exc}"
                ) from exc

        try:
            return SpecializationManifest.model_validate(data)
        except (ValidationError, ValueError) as exc:
            raise SpecializationCompilationError(f"Invalid specialization manifest: {exc}") from exc

    def compile(
        self,
        manifest_or_path: SpecializationManifest | Mapping[str, Any] | Path | str,
        *,
        domain_yaml_path: Path | None = None,
        tokenizer: Any | None = None,
    ) -> CompiledSpecialization:
        """Compiles ``specialization.yaml`` into a validated ``CompiledSpecialization``."""
        manifest = self.load_manifest(manifest_or_path, domain_yaml_path=domain_yaml_path)
        output_schema = self._resolve_output_schema(manifest)
        validator_fn = self._resolve_validator_plugin(manifest.validator_plugin)
        glossary = self._resolve_glossary(manifest)
        arch_profile = detect_architecture_profile(manifest.base_model_id)
        arch_adapter = ModelArchitectureAdapter(
            manifest.base_model_id,
            tokenizer=tokenizer,
            governance_policy=self.policy,
        )
        system_prompt = self._build_system_prompt(manifest, output_schema, glossary)
        return CompiledSpecialization(
            manifest=manifest,
            architecture_profile=arch_profile,
            architecture_adapter=arch_adapter,
            system_prompt=system_prompt,
            output_json_schema=output_schema,
            glossary=glossary,
            validator_callable=validator_fn,
        )


__all__ = [
    "VALID_SPECIALIZATION_RANKS",
    "ChangeImpactMRReviewReport",
    "CompiledSpecialization",
    "ComplianceAuditFinding",
    "ComplianceSecurityAuditReport",
    "LoRAPolicySpec",
    "PointerLocalizationEnvelope",
    "SpecializationArchetype",
    "SpecializationCompilationError",
    "SpecializationCompiler",
    "SpecializationManifest",
    "SpecializationValidationResult",
    "ValidatorCallable",
    "extract_json_from_model_output",
]
