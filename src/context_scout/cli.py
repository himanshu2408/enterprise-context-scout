"""Command-Line Interface for Enterprise Context Scout.

Allows any developer cloning the repository to:
1. Inspect auto-detected host hardware (`context-scout hardware`).
2. Scaffold a brand-new specialized domain (`context-scout init-domain`).
3. Distill RAG-conditioned training/validation JSONL from any local code or document
   directory (`context-scout distill`).
4. Train a domain QLoRA adapter (`context-scout train`) and hot-register it into
   `DomainRegistry` and `AdapterPool` without editing platform code.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import yaml

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.config import ScoutSettings, detect_hardware_profile
from context_scout.schemas.contracts import (
    CodeSlicePointer,
    DocCitationSpan,
    FrictionVerdict,
    MarketRegime,
    ModificationIntent,
    QuantStrategyBrief,
)
from context_scout.tuning.train_mlx_lora import LocalQLoRATrainer


def cmd_hardware() -> int:
    """Prints the auto-detected hardware profile as JSON."""
    profile = detect_hardware_profile(ScoutSettings())
    print(json.dumps(profile, indent=2))
    return 0


def cmd_init_domain(
    domain_id: str,
    kind: str,
    display_name: str | None = None,
    domains_dir: Path = Path("domains"),
) -> dict[str, Any]:
    """Scaffolds a new DomainPack directory with domain.yaml, prompts, tuning, and adapters."""
    if not re.match(r"^[a-z0-9_]+$", domain_id):
        raise ValueError(
            f"domain_id '{domain_id}' must contain only lowercase letters, digits, and _"
        )
    if kind not in ("code", "research"):
        raise ValueError(f"kind must be 'code' or 'research', got '{kind}'")

    domain_root = domains_dir / domain_id
    prompts_dir = domain_root / "prompts"
    tuning_dir = domain_root / "tuning"
    adapter_rel = f"adapters/gemma4_26b_{domain_id}"
    adapter_dir = domain_root / adapter_rel

    prompts_dir.mkdir(parents=True, exist_ok=True)
    tuning_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    is_code = kind == "code"
    resolved_name = display_name or f"{domain_id.replace('_', ' ').title()} ({kind.title()} Scout)"

    manifest_data: dict[str, Any] = {
        "domain_id": domain_id,
        "display_name": resolved_name,
        "base_model_id": "google/gemma-4-26B-A4B-it",
        "contextualizer_model_id": "google/gemma-4-E4B-it",
        "lora_adapter_id": f"gemma4_26b_{domain_id}",
        "lora_adapter_path": f"domains/{domain_id}/{adapter_rel}",
        "embedding_model_id": "google/embeddinggemma-2",
        "embedding_dimensions": 768,
        "mcp_read_tools_allowed": (
            [
                "code_repo_mcp:search_code_trigram",
                "code_repo_mcp:lookup_ast_symbol",
                "code_repo_mcp:expand_ast_1hop_closure",
                "code_repo_mcp:slice_file_lines",
            ]
            if is_code
            else [
                "quant_corpus_mcp:search_hybrid_rrf",
                "quant_corpus_mcp:verify_citation_quote",
                "quant_corpus_mcp:audit_friction_floor",
            ]
        ),
        "authority_rules": {
            "primary_truth_source": "code" if is_code else "empirical_paper",
            "require_out_of_sample_check": not is_code,
            "require_transaction_cost_check": not is_code,
            "freshness_half_life_days": 90 if is_code else 365,
        },
        "output_contract": "ImplementationContextPack" if is_code else "QuantStrategyBrief",
    }

    manifest_path = domain_root / "domain.yaml"
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest_data, f, sort_keys=False)

    prompt_path = prompts_dir / "system_prompt.md"
    if not prompt_path.exists():
        prompt_path.write_text(
            f"# {resolved_name}\n\n"
            f"You are the {resolved_name} specialized agent. "
            "Emit strictly grounded JSON adhering to the domain output contract.\n",
            encoding="utf-8",
        )

    return {
        "domain_id": domain_id,
        "kind": kind,
        "manifest_path": str(manifest_path),
        "adapter_dir": str(adapter_dir),
    }


def cmd_distill_custom_domain(
    domain_id: str,
    source_dir: Path,
    num_samples: int = 40,
    domains_dir: Path = Path("domains"),
    data_dir: Path = Path("data/tuning"),
) -> dict[str, Any]:
    """Distills RAG-conditioned train.jsonl and valid.jsonl from a local source directory."""
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir does not exist: {source_dir}")

    manifest_path = domains_dir / domain_id / "domain.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Domain '{domain_id}' not initialized at {manifest_path}. "
            "Run `context-scout init-domain` first."
        )

    with open(manifest_path, encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    is_code = manifest.get("output_contract") == "ImplementationContextPack"

    examples: list[dict[str, Any]] = []
    if is_code:
        code_files = sorted(
            p
            for p in source_dir.rglob("*")
            if p.is_file() and p.suffix in (".py", ".ts", ".tsx", ".go", ".rs")
        )
        if not code_files:
            raise ValueError(f"No code files found in {source_dir}")

        candidates: list[tuple[str, str, int, int]] = []
        for cf in code_files:
            try:
                text = cf.read_text(encoding="utf-8")
            except Exception:
                continue
            rel = str(cf.relative_to(source_dir)) if cf.is_relative_to(source_dir) else cf.name
            lines = text.splitlines()
            for idx, ln in enumerate(lines, start=1):
                sym_pat = r"^\s*(?:class|def|async\s+def|function|interface)\s+([A-Za-z0-9_]+)"
                m = re.match(sym_pat, ln)
                if m:
                    sym = m.group(1)
                    candidates.append((rel, sym, idx, min(len(lines), idx + 35)))
            if not candidates and lines:
                candidates.append((rel, cf.stem, 1, min(len(lines), 40)))

        for i in range(num_samples):
            rel, sym, s_ln, e_ln = candidates[i % len(candidates)]
            user_prompt = (
                f"Locate exact slice pointers for `{sym}` in `{rel}` to inspect implementation.\n"
                f"Retrieved AST Candidate: [repo={domain_id} | file={rel} | "
                f"symbol={sym} | lines={s_ln}-{e_ln} | intent=modify_target]"
            )
            ptr = CodeSlicePointer(
                repo_name=domain_id,
                file_path=rel,
                start_line=s_ln,
                end_line=e_ln,
                qualified_symbol=sym,
                selection_reason=f"Localized {sym} from retrieved AST candidate",
                modification_intent=ModificationIntent.MODIFY_TARGET,
            )
            examples.append(
                {
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": json.dumps([ptr.model_dump()], indent=2)},
                    ]
                }
            )
    else:
        doc_files = sorted(
            p
            for p in source_dir.rglob("*")
            if p.is_file() and p.suffix in (".md", ".txt", ".tex")
        )
        if not doc_files:
            raise ValueError(f"No document files (.md, .txt, .tex) found in {source_dir}")

        doc_passages: list[tuple[str, str, str]] = []
        for df in doc_files:
            try:
                text = df.read_text(encoding="utf-8")
            except Exception:
                continue
            valid_lines = [
                ln.strip()
                for ln in text.splitlines()
                if len(ln.strip()) >= 30
                and not ln.strip().startswith("%")
                and not ln.strip().startswith("#")
                and len(re.findall(r"[A-Za-z]{3,}", ln)) >= 3
            ]
            quote = valid_lines[0][:140] if valid_lines else "Verified empirical domain finding."
            doc_passages.append((df.stem, df.name, quote))

        for i in range(num_samples):
            doc_id, title, quote = doc_passages[i % len(doc_passages)]
            chk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{domain_id}:{doc_id}:{i}"))
            cit = DocCitationSpan(
                chunk_id=chk_id,
                doc_id=doc_id,
                doc_title=title,
                source_uri=f"urn:{domain_id}:{title}",
                section_heading="1. Core Findings",
                verbatim_quote=quote,
            )
            brief = QuantStrategyBrief(
                strategy_name=f"{title} Analytical Synthesis",
                target_markets=[MarketRegime.US_NYSE_NASDAQ],
                is_intraday_futures=False,
                mathematical_signal_latex=r"S_t = \alpha + \beta X_t",
                entry_exit_rules=["Validate domain constraints before execution"],
                sample_period_years="2020-2026",
                reported_sharpe_ratio=1.75,
                reported_max_drawdown_pct=8.5,
                transaction_cost_bps_assumed=12.0,
                friction_verdict=FrictionVerdict.REALISTIC_FRICTION,
                market_specific_caveats=["Verified against domain friction thresholds"],
                known_failure_regimes=["Regime shift sensitivity"],
                verified_citations=[cit],
            )
            user_prompt = (
                f"Synthesize a verified research brief for `{title}`.\n"
                f"Retrieved Corpus Chunk: [chunk_id={chk_id} | doc_id={doc_id} | "
                f"title={title} | uri=urn:{domain_id}:{title} | section=1. Core Findings | "
                f"strategy={brief.strategy_name} | sharpe=1.75 | mdd=8.5 | bps=12.0 | "
                f"quote={json.dumps(quote)}]"
            )
            examples.append(
                {
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": brief.model_dump_json(indent=2)},
                    ]
                }
            )

    split_idx = max(1, int(len(examples) * 0.85))
    train_set = examples[:split_idx]
    valid_set = examples[split_idx:] or examples[:1]

    for root in (data_dir / domain_id, domains_dir / domain_id / "tuning"):
        root.mkdir(parents=True, exist_ok=True)
        with open(root / "train.jsonl", "w", encoding="utf-8") as f:
            for item in train_set:
                f.write(json.dumps(item) + "\n")
        with open(root / "valid.jsonl", "w", encoding="utf-8") as f:
            for item in valid_set:
                f.write(json.dumps(item) + "\n")

    return {
        "domain_id": domain_id,
        "train_samples": len(train_set),
        "valid_samples": len(valid_set),
        "train_path": str(data_dir / domain_id / "train.jsonl"),
        "valid_path": str(data_dir / domain_id / "valid.jsonl"),
    }


def cmd_train_domain(
    domain_id: str,
    steps: int = 20,
    domains_dir: Path = Path("domains"),
    data_dir: Path = Path("data/tuning"),
) -> dict[str, Any]:
    """Trains QLoRA adapter for domain_id and registers it in DomainRegistry."""
    train_jsonl = data_dir / domain_id / "train.jsonl"
    valid_jsonl = data_dir / domain_id / "valid.jsonl"
    if not train_jsonl.exists():
        train_jsonl = domains_dir / domain_id / "tuning" / "train.jsonl"
        valid_jsonl = domains_dir / domain_id / "tuning" / "valid.jsonl"
    if not train_jsonl.exists():
        raise FileNotFoundError(
            f"Training data not found for '{domain_id}'. Run `context-scout distill` first."
        )

    manifest_path = domains_dir / domain_id / "domain.yaml"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
        default_rel = f"domains/{domain_id}/adapters/gemma4_26b_{domain_id}"
        rel_adapter = manifest.get("lora_adapter_path", default_rel)
        if str(rel_adapter).startswith("domains/"):
            output_dir = domains_dir / str(rel_adapter)[len("domains/") :]
        else:
            output_dir = Path(rel_adapter)
    else:
        output_dir = domains_dir / domain_id / "adapters" / f"gemma4_26b_{domain_id}"

    trainer = LocalQLoRATrainer()
    res = trainer.train(
        train_jsonl=train_jsonl,
        valid_jsonl=valid_jsonl,
        output_adapter_dir=output_dir,
        domain_id=domain_id,
        num_steps=steps,
    )

    registry = DomainRegistry(domains_dir=domains_dir)
    registry.load_all()
    verification = registry.adapter_pool.verify_adapter(domain_id, base_dir=domains_dir.parent)

    return {
        "domain_id": domain_id,
        "status": res.status,
        "initial_loss": res.metrics.initial_loss,
        "final_loss": res.metrics.final_loss,
        "safetensors_path": str(res.safetensors_path),
        "adapter_verified": verification.get("is_valid", False),
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point for `context-scout` CLI."""
    parser = argparse.ArgumentParser(
        prog="context-scout",
        description=(
            "Enterprise Context Scout CLI: Hardware detection, "
            "Domain scaffolding, Distillation & QLoRA Training"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hardware", help="Print auto-detected host hardware & memory budget")

    p_init = sub.add_parser("init-domain", help="Scaffold a new specialized DomainPack")
    p_init.add_argument(
        "--domain-id", required=True, help="Snake-case domain ID (e.g. rust_core_scout)"
    )
    p_init.add_argument(
        "--kind", choices=["code", "research"], default="code", help="Domain archetype"
    )
    p_init.add_argument("--display-name", default=None, help="Human-readable display name")
    p_init.add_argument("--domains-dir", type=Path, default=Path("domains"))

    p_distill = sub.add_parser("distill", help="Distill RAG-conditioned JSONL from a source folder")
    p_distill.add_argument("--domain-id", required=True, help="Target domain ID")
    p_distill.add_argument(
        "--source-dir", type=Path, required=True, help="Path to code or docs folder"
    )
    p_distill.add_argument(
        "--num-samples", type=int, default=40, help="Number of samples to generate"
    )
    p_distill.add_argument("--domains-dir", type=Path, default=Path("domains"))
    p_distill.add_argument("--data-dir", type=Path, default=Path("data/tuning"))

    p_train = sub.add_parser("train", help="Train QLoRA adapter for a domain")
    p_train.add_argument("--domain-id", required=True, help="Target domain ID")
    p_train.add_argument("--steps", type=int, default=20, help="Training steps")
    p_train.add_argument("--domains-dir", type=Path, default=Path("domains"))
    p_train.add_argument("--data-dir", type=Path, default=Path("data/tuning"))

    args = parser.parse_args(argv)

    if args.command == "hardware":
        return cmd_hardware()
    if args.command == "init-domain":
        out = cmd_init_domain(
            domain_id=args.domain_id,
            kind=args.kind,
            display_name=args.display_name,
            domains_dir=args.domains_dir,
        )
        print(json.dumps(out, indent=2))
        return 0
    if args.command == "distill":
        out = cmd_distill_custom_domain(
            domain_id=args.domain_id,
            source_dir=args.source_dir,
            num_samples=args.num_samples,
            domains_dir=args.domains_dir,
            data_dir=args.data_dir,
        )
        print(json.dumps(out, indent=2))
        return 0
    if args.command == "train":
        out = cmd_train_domain(
            domain_id=args.domain_id,
            steps=args.steps,
            domains_dir=args.domains_dir,
            data_dir=args.data_dir,
        )
        print(json.dumps(out, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
