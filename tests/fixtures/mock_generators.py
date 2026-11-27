"""Explicit test doubles and mock completion generators for unit test isolation.

Extracted from production code per the Zero-Simulation Integrity Mandate.
These test doubles are permitted STRICTLY within test harnesses and unit tests
to verify schema parsing, AST slicing, and preflight gate validation without
requiring live 26B MLX GPU weights.
NEVER permitted on the production path or cited as empirical benchmark evidence.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

import pytest

from context_scout.agents.llm_backend import GPUPriorityClass
from context_scout.agents.scout_workflow import LocalGemmaClient
from context_scout.db.schema import ASTSymbolRecord
from context_scout.schemas.contracts import (
    CodeSlicePointer,
    DocCitationSpan,
    DomainKnowledgeBrief,
    ModificationIntent,
    PrincipalContext,
    RoleType,
)

logger = logging.getLogger(__name__)


def simulate_untuned_base_model_completion(prompt: str) -> str:
    """Simulates Condition A (zero-shot) and Condition B (few-shot) base model completions."""
    is_few_shot = "Example 1:" in prompt or "Example 2:" in prompt
    tail_score = sum(ord(c) for c in prompt[-100:]) if len(prompt) >= 100 else len(prompt)

    if is_few_shot:
        if (tail_score % 6) == 0:
            return '{\n  "error": "Incomplete schema response",\n  "partial_data": [1, 2, 3]\n}'

        if "DomainKnowledgeBrief" in prompt or "domain_id" in prompt.lower():
            base_brief = {
                "brief_title": "In-Context Domain Synthesis",
                "domain_id": "enterprise_docs_scout",
                "executive_summary": "Grounded synthesis of enterprise architecture documentation.",
                "formal_expression_or_rule": "E[Y | X] = f_theta(DomainContext)",
                "core_findings_and_rules": ["Follow documented service boundary invariants"],
                "domain_constraints_and_caveats": ["Verify regional compliance constraints"],
                "known_risks_and_failure_modes": ["Stale index cache during rapid rollouts"],
                "structured_metadata": {"scope": "GLOBAL_MULTI", "confidence": 0.9},
                "verified_citations": [
                    {
                        "chunk_id": "chunk_in_context_01",
                        "doc_id": "doc:enterprise_arch_01",
                        "doc_title": "Enterprise Architecture Principles",
                        "source_uri": "https://docs.internal/arch/principles",
                        "section_heading": "Introduction",
                        "verbatim_quote": "Service boundaries must enforce explicit contracts.",
                    }
                ],
            }
            return json.dumps(base_brief, indent=2)

        generic_pointers = [
            {
                "repo_name": "enterprise-context-scout",
                "file_path": "src/context_scout/db/repositories.py",
                "start_line": 1,
                "end_line": 25,
                "qualified_symbol": "context_scout.db.repositories",
                "selection_reason": "High-level module match from in-context prompt",
                "modification_intent": "modify_target",
            }
        ]
        return json.dumps(generic_pointers, indent=2)

    if (tail_score % 3) != 0:
        return (
            "Here is the analysis based on your request:\n\n"
            "```json\n"
            "[\n"
            "  {\n"
            '    "symbol": "ASTSymbolRecord",\n'
            '    "path": "repositories.py"\n'
            "  }\n"
            "]\n"
            "```\n\n"
            "Note: Please verify line numbers before making modifications."
        )

    return json.dumps(
        [
            {
                "repo_name": "enterprise-context-scout",
                "file_path": "src/context_scout/db/repositories.py",
                "start_line": 1,
                "end_line": 30,
                "qualified_symbol": "ASTSymbolRecord",
                "selection_reason": "Identified by zero-shot base inference",
                "modification_intent": "modify_target",
            }
        ],
        indent=2,
    )


def deterministic_completion_generator(
    prompt: str,
    domain_id: str | None = None,
    system_instruction: str | None = None,
    repo: Any = None,
    **kwargs: Any,
) -> str:
    """Test fixture completion generator simulating empirical capabilities for offline testing."""
    if domain_id is None:
        return simulate_untuned_base_model_completion(prompt)

    is_custom_code_domain = (
        domain_id is not None
        and domain_id != "enterprise_docs_scout"
        and (
            domain_id == "code_architecture_scout"
            or "Retrieved AST Candidate:" in prompt
            or "code" in domain_id
        )
    )

    if is_custom_code_domain:
        pointers: list[CodeSlicePointer] = []
        ast_cand_pat = (
            r"\[?repo=([^\|\]\s]+)\s*\|\s*file=([^\|\]\s]+)\s*\|\s*"
            r"symbol=([^\|\]\s]+)\s*\|\s*lines=(\d+)-(\d+)"
        )
        for idx_cand, cand_match in enumerate(re.finditer(ast_cand_pat, prompt)):
            repo_n, f_path, q_sym, s_ln, e_ln = cand_match.groups()
            pointers.append(
                CodeSlicePointer(
                    repo_name=repo_n.strip(),
                    file_path=f_path.strip(),
                    start_line=int(s_ln),
                    end_line=int(e_ln),
                    qualified_symbol=q_sym.strip(),
                    selection_reason=(
                        f"Target localized from retrieved AST candidate {q_sym.strip()}"
                    ),
                    modification_intent=(
                        ModificationIntent.MODIFY_TARGET
                        if idx_cand == 0
                        else ModificationIntent.REFERENCE_ONLY
                    ),
                )
            )
        if pointers:
            return json.dumps([p.model_dump() for p in pointers], indent=2)

        found_files = list(
            dict.fromkeys(re.findall(r"src/context_scout/[A-Za-z0-9_\-\./]+\.py", prompt))
        )
        raw_syms = re.findall(r"`([^`]+)`", prompt)
        clean_syms = list(dict.fromkeys(s for s in raw_syms if "/" not in s))

        try:
            from context_scout.config import ScoutSettings
            from context_scout.db.session import RLSConnectionManager
            from context_scout.mcp_servers.common import PostgresScoutGraphAdapter

            principal = PrincipalContext(
                user_id="local_gemma_worker",
                role=RoleType.ENGINEER,
                allowed_scopes=[
                    "public_eng",
                    "repo:code_architecture_scout",
                    "corpus:enterprise_docs",
                    "docs_restricted",
                ],
            )
            if repo is not None:
                syms = repo.list_visible_ast_symbols(principal, "code_architecture_scout")
                files = repo.list_visible_code_files(principal, "code_architecture_scout")
            else:
                settings = ScoutSettings()
                with RLSConnectionManager(settings.db_app_url, use_pool=False) as conn_mgr:
                    adapter = PostgresScoutGraphAdapter(conn_mgr, settings)
                    syms = adapter.list_visible_ast_symbols(principal, "code_architecture_scout")
                    files = adapter.list_visible_code_files(principal, "code_architecture_scout")
            file_by_id = {f.file_id: f for f in files}

            for sym_name in clean_syms:
                matched_syms: list[tuple[ASTSymbolRecord, str]] = []
                for s_rec in syms:
                    s_file = file_by_id.get(s_rec.file_id)
                    f_path = s_file.file_path if s_file else ""
                    if f_path in found_files and (
                        s_rec.qualified_name.endswith("." + sym_name)
                        or s_rec.qualified_name == sym_name
                    ):
                        matched_syms.append((s_rec, f_path))
                if not matched_syms:
                    for s_rec in syms:
                        if (
                            s_rec.qualified_name.endswith("." + sym_name)
                            or s_rec.qualified_name == sym_name
                        ):
                            s_file = file_by_id.get(s_rec.file_id)
                            matched_syms.append(
                                (
                                    s_rec,
                                    s_file.file_path
                                    if s_file
                                    else (found_files[0] if found_files else ""),
                                )
                            )
                for s_rec, f_path in matched_syms:
                    pointers.append(
                        CodeSlicePointer(
                            repo_name="enterprise-context-scout",
                            file_path=f_path,
                            start_line=s_rec.start_line,
                            end_line=s_rec.end_line,
                            qualified_symbol=s_rec.qualified_name,
                            selection_reason=(
                                f"Identified symbol {sym_name} for requested inspection"
                            ),
                            modification_intent=ModificationIntent.MODIFY_TARGET
                            if len(pointers) == 0
                            else ModificationIntent.REFERENCE_ONLY,
                        )
                    )
        except Exception as e:
            logger.debug("Database AST resolution fallback error in test double: %s", e)

        if not pointers:
            f_path = found_files[0] if found_files else "src/context_scout/schemas/contracts.py"
            sym = clean_syms[0] if clean_syms else "CodeSlicePointer"
            if not sym.startswith("context_scout.") and "src/context_scout/" in f_path:
                mod = f_path.replace("src/", "").replace(".py", "").replace("/", ".")
                qualified_sym = f"{mod}.{sym}"
            else:
                qualified_sym = sym
            pointers = [
                CodeSlicePointer(
                    repo_name="enterprise-context-scout",
                    file_path=f_path,
                    start_line=10,
                    end_line=50,
                    qualified_symbol=qualified_sym,
                    selection_reason="Target localized by tuned QLoRA adapter",
                    modification_intent=ModificationIntent.MODIFY_TARGET,
                )
            ]
            if len(found_files) > 1 and len(clean_syms) > 1:
                f2 = found_files[1]
                s2 = clean_syms[1]
                if not s2.startswith("context_scout.") and "src/context_scout/" in f2:
                    mod2 = f2.replace("src/", "").replace(".py", "").replace("/", ".")
                    qs2 = f"{mod2}.{s2}"
                else:
                    qs2 = s2
                pointers.append(
                    CodeSlicePointer(
                        repo_name="enterprise-context-scout",
                        file_path=f2,
                        start_line=20,
                        end_line=60,
                        qualified_symbol=qs2,
                        selection_reason="Secondary dependency localized by tuned adapter",
                        modification_intent=ModificationIntent.REFERENCE_ONLY,
                    )
                )
        return json.dumps([p.model_dump() for p in pointers], indent=2)

    if domain_id is not None and not is_custom_code_domain:
        chunk_pat = (
            r"Retrieved Corpus Chunk: \[chunk_id=([^\|]+?) \| doc_id=([^\|]+?) \| "
            r"title=([^\|]+?) \| uri=([^\|]+?) \| section=([^\|]+?) \| .*?quote=([^\]]+)\]"
        )
        retrieved_match = re.search(chunk_pat, prompt)
        if retrieved_match:
            chunk_id = retrieved_match.group(1).strip()
            doc_id = retrieved_match.group(2).strip()
            doc_title = retrieved_match.group(3).strip()
            source_uri = retrieved_match.group(4).strip()
            section_heading = retrieved_match.group(5).strip()
            verbatim_quote = retrieved_match.group(6).strip().strip('"')

            rag_cit = DocCitationSpan(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_title=doc_title,
                source_uri=source_uri,
                section_heading=section_heading,
                verbatim_quote=verbatim_quote,
            )
            rag_brief = DomainKnowledgeBrief(
                brief_title=f"{doc_title} Synthesis",
                domain_id=domain_id,
                executive_summary=f"Grounded synthesis from {doc_title}.",
                formal_expression_or_rule=r"E[Y \mid X] = f_{\theta}(\text{DomainContext})",
                core_findings_and_rules=[f"Grounded in verified citation from {doc_title}"],
                domain_constraints_and_caveats=["Empirical evaluation required"],
                known_risks_and_failure_modes=["High load or stale index windows"],
                structured_metadata={"scope": "GLOBAL_MULTI"},
                verified_citations=[rag_cit],
            )
            return rag_brief.model_dump_json(indent=2)

        titles = re.findall(r"`([^`]+)`", prompt)
        doc_title = titles[0] if titles else "Synthesized Enterprise Brief"
        rag_cit = DocCitationSpan(
            chunk_id="chunk_test_fixture_01",
            doc_id="doc:test_enterprise_doc",
            doc_title=doc_title,
            source_uri="https://docs.internal/test",
            section_heading="Methodology",
            verbatim_quote="Verified empirical citation quote for domain synthesis.",
        )
        rag_brief = DomainKnowledgeBrief(
            brief_title=f"{doc_title} Brief",
            domain_id=domain_id,
            executive_summary=f"Grounded synthesis from {doc_title}.",
            formal_expression_or_rule=r"E[Y \mid X] = f_{\theta}(\text{DomainContext})",
            core_findings_and_rules=[f"Grounded in verified citation from {doc_title}"],
            domain_constraints_and_caveats=["Empirical evaluation required"],
            known_risks_and_failure_modes=["High load or stale index windows"],
            structured_metadata={"scope": "GLOBAL_MULTI"},
            verified_citations=[rag_cit],
        )
        return rag_brief.model_dump_json(indent=2)

    return simulate_untuned_base_model_completion(prompt)


class MockLocalGemmaClient(LocalGemmaClient):
    """Explicit test double for LocalGemmaClient.

    Used STRICTLY in unit tests to verify prompt composition, preflight gate validation,
    and benchmark harness logic without requiring live 26B MLX GPU weights.
    NEVER permitted on the production path or cited as empirical benchmark evidence.
    """

    def __init__(
        self,
        generator_callable: Callable[..., str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("offline_mode", True)
        self.generator_callable = generator_callable or deterministic_completion_generator
        kwargs["completion_fixture"] = self.generator_callable
        super().__init__(*args, **kwargs)

    def complete(
        self,
        prompt: str,
        domain_id: str | None = None,
        system_instruction: str | None = None,
        max_tokens: int = 1024,
        *,
        principal: PrincipalContext | None = None,
        priority: GPUPriorityClass = GPUPriorityClass.P0,
    ) -> str:
        del principal, priority
        if domain_id is None:
            return simulate_untuned_base_model_completion(prompt)
        return self.generator_callable(
            prompt=prompt,
            domain_id=domain_id,
            system_instruction=system_instruction,
        )


@pytest.fixture
def mock_gemma_client() -> MockLocalGemmaClient:
    """Pytest fixture providing an isolated MockLocalGemmaClient test double."""
    return MockLocalGemmaClient()


__all__ = [
    "MockLocalGemmaClient",
    "deterministic_completion_generator",
    "mock_gemma_client",
    "simulate_untuned_base_model_completion",
]
