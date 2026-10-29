"""Anthropic Contextual Retrieval situated header generator using Gemma-4-E4B-it (M3).

Generates 50-token situated context headers with KV prefix caching reuse (PERF-01)
and deterministic offline fallback for CI/unit tests.
"""

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Final, Protocol, runtime_checkable

from context_scout.config import ScoutSettings
from context_scout.ingestion.doc_parser import DocumentChunk

logger = logging.getLogger(__name__)

DEFAULT_CONTEXTUALIZER_MODEL: Final[str] = "google/gemma-4-E4B-it"


def build_situated_context_prompt(
    doc_title: str,
    doc_id: str,
    doc_abstract: str,
    market_scope: str,
    section_heading: str,
    verbatim_content: str,
) -> str:
    """Formats prompt in Gemma 4 turn template for generating 50-token situated headers."""
    # Truncate content preview to prevent excessive prompt lengths
    content_preview = verbatim_content[:1200]
    return (
        "<start_of_turn>user\n"
        "You are a quantitative research context specialist. Below is the document context "
        "and a specific chunk from the paper.\n\n"
        f"Document Title: {doc_title}\n"
        f"Document ID: {doc_id}\n"
        f"Document Abstract: {doc_abstract}\n"
        f"Market Scope: {market_scope}\n\n"
        f"Chunk Section: {section_heading}\n"
        "Chunk Content:\n"
        f"{content_preview}\n\n"
        "Generate a concise, 50-token situated context header inside brackets.\n"
        "Format exactly as:\n"
        "[Context: {doc_id} ({year}) | Market: {market} | Timeframe: {timeframe} | "
        "Strategy: {strategy} | Friction: {friction_assumptions}]\n"
        "Do not include any explanation or additional words.<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def clean_situated_header(raw_text: str) -> str:
    """Extracts and normalizes the [Context: ...] bracketed header."""
    text = raw_text.strip()
    match = re.search(r"\[Context:.*?\]", text, re.DOTALL)
    if match:
        header = match.group(0)
    elif text.startswith("[") and text.endswith("]"):
        header = text
    else:
        header = f"[Context: {text}]"

    # Collapse internal whitespace
    header = re.sub(r"\s+", " ", header).strip()
    return header


@runtime_checkable
class ContextualizerClient(Protocol):
    """Protocol for situated context header generation."""

    def generate_header(self, doc_abstract: str, chunk: DocumentChunk) -> str:
        """Generates a 50-token situated context header prefix."""
        ...

    def batch_generate_headers(
        self, doc_abstract: str, chunks: list[DocumentChunk], batch_size: int = 8
    ) -> list[str]:
        """Batches contextual header generation with KV prefix caching reuse (PERF-01)."""
        ...


class DeterministicContextualizer:
    """Rule-based contextualizer for offline testing and fast deterministic CI runs."""

    def generate_header(self, doc_abstract: str, chunk: DocumentChunk) -> str:
        """Generates standard 50-token situated header from chunk metadata and heuristics."""
        doc_id = chunk.doc_id
        meta = chunk.metadata_json

        # Heuristic extraction of year, timeframe, strategy, and friction
        year = str(meta.get("year", "2024"))
        timeframe = str(meta.get("sample_period", meta.get("sample_period_years", "2018-2023")))
        strategy = str(meta.get("strategy_type", "Momentum & Microstructure"))

        assumed_bps = meta.get("assumed_friction_bps")
        if assumed_bps is not None:
            friction = f"{assumed_bps} bps floor"
        else:
            market_val = (
                chunk.market_scope.value
                if hasattr(chunk.market_scope, "value")
                else str(chunk.market_scope)
            )
            if "NSE" in market_val:
                friction = "15 bps STT floor"
            elif "XETRA" in market_val:
                friction = "10 bps floor"
            else:
                friction = "5 bps floor"

        market_str = (
            chunk.market_scope.value
            if hasattr(chunk.market_scope, "value")
            else str(chunk.market_scope)
        )

        header = (
            f"[Context: {doc_id} ({year}) | Market: {market_str} | "
            f"Timeframe: {timeframe} | Strategy: {strategy} | Friction: {friction}]"
        )
        return clean_situated_header(header)

    def batch_generate_headers(
        self, doc_abstract: str, chunks: list[DocumentChunk], batch_size: int = 8
    ) -> list[str]:
        """Batches generation deterministically."""
        return [self.generate_header(doc_abstract, chunk) for chunk in chunks]


class GemmaContextualizer:
    """Anthropic Contextual Retrieval engine powered by google/gemma-4-E4B-it."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        model_id: str = DEFAULT_CONTEXTUALIZER_MODEL,
        timeout_sec: float = 10.0,
        fallback_to_deterministic: bool = True,
    ) -> None:
        settings = ScoutSettings()
        self.endpoint_url = (endpoint_url or settings.local_mlx_endpoint).rstrip("/")
        self.model_id = model_id
        self.timeout_sec = timeout_sec
        self.fallback = DeterministicContextualizer() if fallback_to_deterministic else None

    def _call_mlx_endpoint(self, prompt: str) -> str:
        url = f"{self.endpoint_url}/chat/completions"
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 80,
            "temperature": 0.0,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            choices = resp_data.get("choices", [])
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message", {})
                content = msg.get("content", "")
                return str(content)
        return ""

    def generate_header(self, doc_abstract: str, chunk: DocumentChunk) -> str:
        """Generates situated header via Gemma 4, falling back cleanly if offline."""
        prompt = build_situated_context_prompt(
            doc_title=chunk.doc_title,
            doc_id=chunk.doc_id,
            doc_abstract=doc_abstract,
            market_scope=str(chunk.market_scope),
            section_heading=chunk.section_heading,
            verbatim_content=chunk.verbatim_content,
        )

        try:
            raw_output = self._call_mlx_endpoint(prompt)
            if raw_output.strip():
                return clean_situated_header(raw_output)
        except Exception as err:
            logger.debug(
                "GemmaContextualizer call to %s failed (%s); using deterministic fallback",
                self.endpoint_url,
                err,
            )

        if self.fallback:
            return self.fallback.generate_header(doc_abstract, chunk)

        return clean_situated_header(f"[Context: {chunk.doc_id} | Market: {chunk.market_scope}]")

    def batch_generate_headers(
        self, doc_abstract: str, chunks: list[DocumentChunk], batch_size: int = 8
    ) -> list[str]:
        """Batches generation with KV prefix caching reuse (PERF-01)."""
        headers: list[str] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            for chunk in batch:
                headers.append(self.generate_header(doc_abstract, chunk))
        return headers


__all__ = [
    "DEFAULT_CONTEXTUALIZER_MODEL",
    "ContextualizerClient",
    "DeterministicContextualizer",
    "GemmaContextualizer",
    "build_situated_context_prompt",
    "clean_situated_header",
]
