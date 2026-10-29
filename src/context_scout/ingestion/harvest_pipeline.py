"""Scaled quantitative research paper harvester and PostgreSQL 17 ingestion pipeline (M3).

Harvests top quantitative trading and market microstructure research papers across
IN_NSE, DE_XETRA, and US_EQUITY markets, caches artifacts locally in gitignored data/papers/,
parses lossless chunks with situated headers, and persists into doc_chunks.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
from collections.abc import Iterator
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Final

from context_scout.config import ScoutSettings
from context_scout.db.repositories import DocChunkRecord, PostgresRLSRepository
from context_scout.db.session import RLSConnectionManager
from context_scout.ingestion.contextualizer import DeterministicContextualizer
from context_scout.ingestion.doc_parser import DocumentChunk, LatexDocParser, PdfDocParser
from context_scout.ingestion.paper_harvester import (
    MARKET_QUERY_PRESETS,
    HarvestedPaper,
    HarvesterSource,
    HttpPaperHarvesterClient,
    QuantRelevanceClassifier,
    build_arxiv_query_url,
    build_openalex_query_url,
    parse_arxiv_atom_feed,
    parse_openalex_response,
)
from context_scout.retrieval.embedder import LocalSemanticEmbedder
from context_scout.schemas.contracts import MarketRegime, PrincipalContext, RoleType

logger = logging.getLogger(__name__)

DEFAULT_PAPERS_DIR: Final[Path] = Path("data/papers")
QUANT_DOMAIN_ID: Final[str] = "quant_research_scout"
QUANT_SCOPE: Final[str] = "public_quant"


def sanitize_filename(identifier: str) -> str:
    """Sanitizes document IDs into safe filesystem file names."""
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", identifier)


class CorpusHarvestPipeline:
    """Manages scaled harvesting, caching, parsing, and database persistence."""

    def __init__(
        self,
        settings: ScoutSettings | None = None,
        base_dir: Path | None = None,
        harvester_client: HttpPaperHarvesterClient | None = None,
    ) -> None:
        self.settings = settings or ScoutSettings()
        self.base_dir = base_dir or DEFAULT_PAPERS_DIR
        self.metadata_dir = self.base_dir / "metadata"
        self.content_dir = self.base_dir / "content"
        self.client = harvester_client or HttpPaperHarvesterClient()
        self.latex_parser = LatexDocParser()
        self.pdf_parser = PdfDocParser()
        self.contextualizer = DeterministicContextualizer()
        self.repo = PostgresRLSRepository()
        self.relevance_classifier = QuantRelevanceClassifier()
        self.embedder = LocalSemanticEmbedder(
            model_id=self.settings.embedding_model_id,
            settings=self.settings,
        )

        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensures required cache directories exist."""
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.content_dir.mkdir(parents=True, exist_ok=True)
        for regime in MarketRegime:
            (self.content_dir / regime.value).mkdir(parents=True, exist_ok=True)

    def fetch_market_candidates_stream(
        self,
        market: MarketRegime,
        seen_ids: set[str] | None = None,
    ) -> Iterator[HarvestedPaper]:
        """Yields candidate papers across cache, arXiv sweeps, and OpenAlex."""
        if seen_ids is None:
            seen_ids = set()

        # 1. Existing cached valid full-text files in content_dir / market.value
        market_dir = self.content_dir / market.value
        if market_dir.exists():
            for f in sorted(market_dir.glob("*.*")):
                if f.suffix == ".txt":
                    # Strictly purge any stale .txt files
                    f.unlink(missing_ok=True)
                    continue
                if f.suffix in (".tex", ".pdf"):
                    raw_id = f.stem
                    doc_id = raw_id.replace("_", ":", 1) if "_" in raw_id else raw_id
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)

                    tex_content: str | None = None
                    pdf_bytes: bytes | None = None
                    if f.suffix == ".tex":
                        tex_content = f.read_text(encoding="utf-8", errors="replace")
                        if len(tex_content.strip()) < 3000:
                            continue
                    elif f.suffix == ".pdf":
                        pdf_bytes = f.read_bytes()
                        if not pdf_bytes.startswith(b"%PDF-") or len(pdf_bytes) < 3000:
                            continue

                    yield HarvestedPaper(
                        doc_id=doc_id,
                        source=HarvesterSource.LOCAL_CACHE,
                        title=raw_id.replace("_", " "),
                        authors=["Quantitative Researcher"],
                        published_date=date(2024, 1, 1),
                        market_scope=market,
                        source_uri=f"file://{f}",
                        abstract="Verified quantitative finance research paper",
                        raw_tex_content=tex_content,
                        raw_pdf_bytes=pdf_bytes,
                        metadata={"cached": True},
                    )

        query_preset = MARKET_QUERY_PRESETS[market]

        # 2. Market-specific arXiv query with preset keywords first (highest market specificity)
        try:
            arxiv_url = build_arxiv_query_url(query_preset)
            xml_body = self.client._execute_http_get(arxiv_url)
            if xml_body:
                papers = parse_arxiv_atom_feed(xml_body, market)
                for p in papers:
                    if p.doc_id not in seen_ids:
                        seen_ids.add(p.doc_id)
                        yield p
        except Exception as err:
            logger.debug("arXiv query preset error for %s: %s", market.value, err)

        # 3. Dedicated arXiv quantitative finance sweeps aligned to market regime
        cat_order: list[str]
        if market == MarketRegime.INDIA_NSE_BSE:
            cat_order = ["q-fin.CP", "q-fin.ST", "q-fin.TR", "q-fin.PM", "q-fin.RM"]
        elif market == MarketRegime.GERMANY_XETRA_DAX:
            cat_order = ["q-fin.RM", "q-fin.PM", "q-fin.TR", "q-fin.CP", "q-fin.ST"]
        else:
            cat_order = ["q-fin.TR", "q-fin.PM", "q-fin.CP", "q-fin.ST", "q-fin.RM"]

        for cat in cat_order:
            for start_idx in (0, 30, 60, 90, 120, 150, 180):
                try:
                    params = {
                        "search_query": f"cat:{cat}",
                        "start": str(start_idx),
                        "max_results": "30",
                        "sortBy": "submittedDate",
                        "sortOrder": "descending",
                    }
                    cat_url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
                    xml_body = self.client._execute_http_get(cat_url)
                    if xml_body:
                        papers = parse_arxiv_atom_feed(xml_body, market)
                        for p in papers:
                            if p.doc_id not in seen_ids:
                                seen_ids.add(p.doc_id)
                                yield p
                except Exception as err:
                    logger.debug("arXiv sweep %s offset %d error: %s", cat, start_idx, err)

        # 4. OpenAlex open-access works (when available)
        try:
            oa_url = build_openalex_query_url(query_preset, sort_by="cited_by_count:desc")
            oa_body = self.client._execute_http_get(oa_url)
            if oa_body:
                data = json.loads(oa_body)
                if isinstance(data, dict):
                    oa_papers = parse_openalex_response(data, market)
                    for p in oa_papers:
                        if p.doc_id not in seen_ids:
                            seen_ids.add(p.doc_id)
                            yield p
        except Exception as err:
            logger.debug("OpenAlex query error for %s: %s", market.value, err)

    def fetch_market_candidates(
        self,
        market: MarketRegime,
        target_count: int = 120,
    ) -> list[HarvestedPaper]:
        """Harvests candidate papers up to target_count."""
        candidates: list[HarvestedPaper] = []
        for p in self.fetch_market_candidates_stream(market):
            candidates.append(p)
            if len(candidates) >= target_count:
                break
        return candidates

    def cache_paper_source(
        self, paper: HarvestedPaper, download_full_text: bool = True
    ) -> HarvestedPaper:
        """Downloads source if missing from cache, storing full-text content to disk."""
        sanitized = sanitize_filename(paper.doc_id)
        market_dir = self.content_dir / paper.market_scope.value
        tex_path = market_dir / f"{sanitized}.tex"
        pdf_path = market_dir / f"{sanitized}.pdf"

        # Check existing disk cache first
        if tex_path.exists():
            content = tex_path.read_text(encoding="utf-8", errors="replace")
            if len(content.strip()) >= 3000:
                return HarvestedPaper(
                    doc_id=paper.doc_id,
                    source=paper.source,
                    title=paper.title,
                    authors=paper.authors,
                    published_date=paper.published_date,
                    market_scope=paper.market_scope,
                    source_uri=paper.source_uri,
                    abstract=paper.abstract,
                    raw_tex_content=content,
                    metadata=paper.metadata,
                )

        if pdf_path.exists():
            pdf_bytes = pdf_path.read_bytes()
            if pdf_bytes.startswith(b"%PDF-") and len(pdf_bytes) >= 3000:
                return HarvestedPaper(
                    doc_id=paper.doc_id,
                    source=paper.source,
                    title=paper.title,
                    authors=paper.authors,
                    published_date=paper.published_date,
                    market_scope=paper.market_scope,
                    source_uri=paper.source_uri,
                    abstract=paper.abstract,
                    raw_pdf_bytes=pdf_bytes,
                    metadata=paper.metadata,
                )

        downloaded = paper
        if download_full_text:
            try:
                downloaded = self.client.download_source(paper)
            except Exception as err:
                logger.warning("Error downloading source for %s: %s", paper.doc_id, err)

        # Only persist to disk if source satisfies strict full-text validation
        if downloaded.raw_tex_content and len(downloaded.raw_tex_content.strip()) >= 3000:
            tex_path.write_text(downloaded.raw_tex_content, encoding="utf-8")
        elif (
            downloaded.raw_pdf_bytes
            and downloaded.raw_pdf_bytes.startswith(b"%PDF-")
            and len(downloaded.raw_pdf_bytes) >= 3000
        ):
            extracted = self.pdf_parser.extract_text(downloaded.raw_pdf_bytes)
            if len(extracted.strip()) >= 3000:
                pdf_path.write_bytes(downloaded.raw_pdf_bytes)
            else:
                logger.debug("PDF for %s text length < 3000 chars; not writing", paper.doc_id)

        # Strictly 0 .txt files permitted
        return downloaded

    def chunk_and_contextualize(self, paper: HarvestedPaper) -> list[DocChunkRecord]:
        """Parses paper into chunks with situated headers and 768-d embeddings."""
        chunks: list[DocumentChunk] = []

        if paper.raw_tex_content:
            tex_clean = paper.raw_tex_content.strip()
            if len(tex_clean) >= 3000:
                sections = self.latex_parser.parse_latex(tex_clean)
                chunks = self.latex_parser.chunk_paper(
                    doc_id=paper.doc_id,
                    doc_title=paper.title,
                    source_uri=paper.source_uri,
                    market_scope=paper.market_scope,
                    sections=sections,
                )
        elif paper.raw_pdf_bytes:
            pdf_bytes = paper.raw_pdf_bytes
            if pdf_bytes.startswith(b"%PDF-") and len(pdf_bytes) >= 3000:
                val, pdf_chunks = self.pdf_parser.validate_and_chunk(
                    doc_id=paper.doc_id,
                    doc_title=paper.title,
                    source_uri=paper.source_uri,
                    market_scope=paper.market_scope,
                    pdf_bytes=pdf_bytes,
                )
                if val.is_valid:
                    chunks = pdf_chunks

        if not chunks or len(chunks) < 3:
            # Full-text validation requires >= 3 chunks; zero synthetic abstract chunks allowed
            return []

        # Validate that body does not match any negative blocking topic
        sample_body = " ".join(c.verbatim_content for c in chunks[:5])
        relevance = self.relevance_classifier.evaluate(paper.title, paper.abstract, sample_body)
        if not relevance.is_relevant and relevance.matched_negative:
            logger.info(
                "Paper %s failed body relevance check: %s",
                paper.doc_id,
                relevance.matched_negative,
            )
            return []

        records: list[DocChunkRecord] = []
        raw_c = paper.metadata.get("cited_by_count")
        citations = int(str(raw_c)) if raw_c is not None else 0
        authority_weight = 1.0 + min(2.0, (citations / 100.0) * 0.5)

        headers = [self.contextualizer.generate_header(paper.abstract, c) for c in chunks]
        chunk_texts = [
            f"{h}\n\n{c.verbatim_content}" for h, c in zip(headers, chunks, strict=False)
        ]
        embeddings = self.embedder.embed_batch(chunk_texts, is_query=False, batch_size=16)

        for chunk, header, emb in zip(chunks, headers, embeddings, strict=False):
            record = chunk.to_doc_chunk_record(
                domain_id=QUANT_DOMAIN_ID,
                contextual_header=header,
                allowed_scopes=[QUANT_SCOPE],
                embedding=emb,
                authority_weight=authority_weight,
                published_date=paper.published_date,
            )
            records.append(record)

        return records

    def ingest_records(self, records: list[DocChunkRecord], batch_size: int = 50) -> int:
        """Persists document chunks into PostgreSQL 17 doc_chunks table."""
        if not records:
            return 0

        admin_mgr = RLSConnectionManager(self.settings.db_admin_url)
        admin_principal = PrincipalContext(
            user_id="corpus_admin",
            role=RoleType.ENGINEER,
            allowed_scopes=[QUANT_SCOPE, "admin"],
        )

        # Ensure domain pack prerequisite exists
        with admin_mgr.scoped_transaction(
            admin_principal, enable_hnsw_iterative_scan=False
        ) as conn:
            conn.execute(
                """
                INSERT INTO domain_packs (
                    domain_id, display_name, lora_adapter_id,
                    primary_truth_source, freshness_half_life_days, config_json
                )
                VALUES (
                    'quant_research_scout', 'Quant Research Scout', 'gemma4_26b_quant_scout',
                    'document', 180, '{"live_corpus": true}'::jsonb
                )
                ON CONFLICT (domain_id) DO NOTHING;
                """
            )

        mgr = RLSConnectionManager(self.settings.db_app_url)
        app_principal = PrincipalContext(
            user_id="harvester_agent",
            role=RoleType.QUANT_RESEARCHER,
            allowed_scopes=[QUANT_SCOPE],
        )

        total_inserted = 0
        with mgr.scoped_transaction(app_principal, enable_hnsw_iterative_scan=False) as conn:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                self.repo.insert_doc_chunks_batch(conn, batch)
                total_inserted += len(batch)

        logger.info("Successfully persisted %d doc_chunks into PostgreSQL 17", total_inserted)
        return total_inserted

    def run_harvest_pipeline(
        self,
        target_papers_per_market: int = 110,
    ) -> dict[str, object]:
        """Executes end-to-end harvesting, caching, and persistence across all markets."""
        catalog_summary: dict[str, object] = {
            "execution_date": date.today().isoformat(),
            "target_per_market": target_papers_per_market,
            "markets": {},
            "total_papers_harvested": 0,
            "total_chunks_ingested": 0,
        }

        global_seen_ids: set[str] = set()
        total_inserted = 0

        for regime in [
            MarketRegime.INDIA_NSE_BSE,
            MarketRegime.GERMANY_XETRA_DAX,
            MarketRegime.US_NYSE_NASDAQ,
        ]:
            logger.info(
                "Starting candidate harvest loop for %s (target=%d)...",
                regime.value,
                target_papers_per_market,
            )
            verified_papers: list[HarvestedPaper] = []
            market_records: list[DocChunkRecord] = []

            for candidate in self.fetch_market_candidates_stream(regime, seen_ids=global_seen_ids):
                if len(verified_papers) >= target_papers_per_market:
                    break

                # Pre-download title & abstract relevance screening
                verdict = self.relevance_classifier.evaluate(candidate.title, candidate.abstract)
                if not verdict.is_relevant and candidate.source != HarvesterSource.LOCAL_CACHE:
                    logger.debug(
                        "Candidate %s rejected: %s", candidate.doc_id, verdict.rejection_reason
                    )
                    continue

                # Download source (only writes .tex or .pdf if valid)
                cached = self.cache_paper_source(candidate, download_full_text=True)
                if not cached.raw_tex_content and not cached.raw_pdf_bytes:
                    continue

                # Chunk, contextualize, and embed
                records = self.chunk_and_contextualize(cached)
                if len(records) < 3:
                    # Clean up invalid file from disk if it was written
                    sanitized = sanitize_filename(candidate.doc_id)
                    for ext in (".tex", ".pdf"):
                        p = self.content_dir / regime.value / f"{sanitized}{ext}"
                        if p.exists():
                            p.unlink(missing_ok=True)
                    continue

                verified_papers.append(cached)
                market_records.extend(records)
                logger.info(
                    "Verified %s paper %d/%d (%s): %s [%d chunks]",
                    regime.value,
                    len(verified_papers),
                    target_papers_per_market,
                    cached.doc_id,
                    cached.title[:50],
                    len(records),
                )

            # Record metadata to jsonl
            jsonl_path = self.metadata_dir / f"{regime.value}_papers.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as f:
                for paper_item in verified_papers:
                    p_dict = asdict(paper_item)
                    p_dict["published_date"] = paper_item.published_date.isoformat()
                    p_dict.pop("raw_pdf_bytes", None)
                    p_dict.pop("raw_tex_content", None)
                    f.write(json.dumps(p_dict) + "\n")

            # Persist chunks into PostgreSQL 17 for this market regime immediately
            inserted_market = self.ingest_records(market_records)
            total_inserted += inserted_market

            market_stats = {
                "papers_count": len(verified_papers),
                "full_sources_cached": len(verified_papers),
                "chunks_generated": len(market_records),
                "chunks_persisted": inserted_market,
            }
            markets_dict = catalog_summary["markets"]
            if isinstance(markets_dict, dict):
                markets_dict[regime.value] = market_stats

            logger.info(
                "Completed %s: %d verified full-text papers, %d chunks persisted",
                regime.value,
                len(verified_papers),
                inserted_market,
            )

        markets_dict = catalog_summary.get("markets")
        total_papers = 0
        if isinstance(markets_dict, dict):
            for m in markets_dict.values():
                if isinstance(m, dict):
                    total_papers += int(m.get("papers_count", 0))

        catalog_summary["total_papers_harvested"] = total_papers
        catalog_summary["total_chunks_ingested"] = total_inserted

        catalog_path = self.metadata_dir / "harvest_catalog.json"
        catalog_path.write_text(json.dumps(catalog_summary, indent=2), encoding="utf-8")
        return catalog_summary


__all__ = [
    "DEFAULT_PAPERS_DIR",
    "QUANT_DOMAIN_ID",
    "QUANT_SCOPE",
    "CorpusHarvestPipeline",
    "sanitize_filename",
]
