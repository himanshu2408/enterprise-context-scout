"""Scaled quantitative research paper harvester and PostgreSQL 17 ingestion pipeline (M3).

Harvests top quantitative trading and market microstructure research papers across
IN_NSE, DE_XETRA, and US_EQUITY markets, caches artifacts locally in gitignored data/papers/,
parses lossless chunks with situated headers, and persists into doc_chunks.
"""

import json
import logging
import re
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Final

from context_scout.config import ScoutSettings
from context_scout.db.repositories import DocChunkRecord, PostgresRLSRepository
from context_scout.db.session import RLSConnectionManager
from context_scout.ingestion.contextualizer import DeterministicContextualizer
from context_scout.ingestion.doc_parser import LatexDocParser, ParsedSection, PdfDocParser
from context_scout.ingestion.git_merkle_indexer import generate_deterministic_embedding
from context_scout.ingestion.paper_harvester import (
    MARKET_QUERY_PRESETS,
    HarvestedPaper,
    HarvesterSource,
    HttpPaperHarvesterClient,
    MarketHarvestQuery,
    build_arxiv_query_url,
    build_openalex_query_url,
    parse_arxiv_atom_feed,
    parse_openalex_response,
)
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

        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensures required cache directories exist."""
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.content_dir.mkdir(parents=True, exist_ok=True)
        for regime in MarketRegime:
            (self.content_dir / regime.value).mkdir(parents=True, exist_ok=True)

    def fetch_market_candidates(
        self,
        market: MarketRegime,
        target_count: int = 120,
    ) -> list[HarvestedPaper]:
        """Harvests ranked candidate papers from OpenAlex and arXiv for a market regime."""
        query_preset = MARKET_QUERY_PRESETS[market]
        candidates: dict[str, HarvestedPaper] = {}

        # 1. Fetch citation-ranked peer-reviewed works from OpenAlex
        try:
            openalex_query = MarketHarvestQuery(
                market_scope=market,
                keywords=query_preset.keywords,
                arxiv_categories=query_preset.arxiv_categories,
                openalex_concepts=query_preset.openalex_concepts,
                start_year=query_preset.start_year,
                max_results=min(200, target_count + 30),
            )
            oa_url = build_openalex_query_url(openalex_query, sort_by="cited_by_count:desc")
            oa_body = self.client._execute_http_get(oa_url)
            if oa_body:
                data = json.loads(oa_body)
                if isinstance(data, dict):
                    oa_papers = parse_openalex_response(data, market)
                    for p in oa_papers:
                        if p.title and p.doc_id not in candidates:
                            candidates[p.doc_id] = p
            logger.info("Retrieved %d OpenAlex papers for %s", len(candidates), market.value)
        except Exception as err:
            logger.warning("Failed OpenAlex candidate fetch for %s: %s", market.value, err)

        # 2. Fetch arXiv market microstructure & quantitative trading preprints
        try:
            arxiv_query = MarketHarvestQuery(
                market_scope=market,
                keywords=query_preset.keywords,
                arxiv_categories=query_preset.arxiv_categories,
                openalex_concepts=query_preset.openalex_concepts,
                start_year=query_preset.start_year,
                max_results=30,
            )
            arxiv_url = build_arxiv_query_url(arxiv_query)
            arxiv_xml = self.client._execute_http_get(arxiv_url)
            if arxiv_xml:
                arxiv_papers = parse_arxiv_atom_feed(arxiv_xml, market)
                for p in arxiv_papers:
                    if p.title and p.doc_id not in candidates:
                        candidates[p.doc_id] = p
            logger.info("Total candidates after arXiv for %s: %d", market.value, len(candidates))
        except Exception as err:
            logger.warning("Failed arXiv candidate fetch for %s: %s", market.value, err)

        paper_list = list(candidates.values())

        # Sort primarily by citation count, secondarily by publication date
        def paper_rank_key(p: HarvestedPaper) -> tuple[int, str]:
            raw_c = p.metadata.get("cited_by_count")
            citations = int(str(raw_c)) if raw_c is not None else 0
            return (citations, p.published_date.isoformat())

        paper_list.sort(key=paper_rank_key, reverse=True)
        return paper_list[:target_count]

    def cache_paper_source(
        self, paper: HarvestedPaper, download_full_text: bool = True
    ) -> HarvestedPaper:
        """Downloads source if missing from cache, and stores content to local disk."""
        sanitized = sanitize_filename(paper.doc_id)
        market_dir = self.content_dir / paper.market_scope.value
        tex_path = market_dir / f"{sanitized}.tex"
        pdf_path = market_dir / f"{sanitized}.pdf"
        txt_path = market_dir / f"{sanitized}.txt"

        # Check existing disk cache first
        if tex_path.exists():
            content = tex_path.read_text(encoding="utf-8", errors="replace")
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

        # Download full text source if requested
        downloaded = paper
        if download_full_text:
            try:
                downloaded = self.client.download_source(paper)
            except Exception as err:
                logger.warning("Error downloading source for %s: %s", paper.doc_id, err)

        if downloaded.raw_tex_content:
            tex_path.write_text(downloaded.raw_tex_content, encoding="utf-8")
        elif downloaded.raw_pdf_bytes:
            pdf_path.write_bytes(downloaded.raw_pdf_bytes)
        else:
            # Fallback to caching metadata and abstract text
            summary_content = f"# {paper.title}\n\n{paper.abstract}\n"
            txt_path.write_text(summary_content, encoding="utf-8")

        return downloaded

    def chunk_and_contextualize(self, paper: HarvestedPaper) -> list[DocChunkRecord]:
        """Parses paper into chunks, generates situated context headers and embeddings."""
        chunks = []

        if paper.raw_tex_content:
            sections = self.latex_parser.parse_latex(paper.raw_tex_content)
            chunks = self.latex_parser.chunk_paper(
                doc_id=paper.doc_id,
                doc_title=paper.title,
                source_uri=paper.source_uri,
                market_scope=paper.market_scope,
                sections=sections,
            )
        elif paper.raw_pdf_bytes:
            chunks = self.pdf_parser.chunk_pdf(
                doc_id=paper.doc_id,
                doc_title=paper.title,
                source_uri=paper.source_uri,
                market_scope=paper.market_scope,
                pdf_bytes=paper.raw_pdf_bytes,
            )

        if not chunks:
            # Structured abstract fallback
            body = (paper.abstract or paper.title).strip()
            sections = [
                ParsedSection(
                    heading="Abstract & Summary",
                    level=1,
                    content=body,
                    formulas_latex=[],
                    tables_markdown=[],
                )
            ]
            chunks = self.latex_parser.chunk_paper(
                doc_id=paper.doc_id,
                doc_title=paper.title,
                source_uri=paper.source_uri,
                market_scope=paper.market_scope,
                sections=sections,
            )

        records: list[DocChunkRecord] = []
        raw_c = paper.metadata.get("cited_by_count")
        citations = int(str(raw_c)) if raw_c is not None else 0
        authority_weight = 1.0 + min(2.0, (citations / 100.0) * 0.5)

        for chunk in chunks:
            header = self.contextualizer.generate_header(paper.abstract, chunk)
            embedding = generate_deterministic_embedding(chunk.verbatim_content, 768)
            record = chunk.to_doc_chunk_record(
                domain_id=QUANT_DOMAIN_ID,
                contextual_header=header,
                allowed_scopes=[QUANT_SCOPE],
                embedding=embedding,
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
        full_source_downloads_per_market: int = 15,
    ) -> dict[str, object]:
        """Executes end-to-end harvesting, caching, and persistence across all markets."""
        catalog_summary: dict[str, object] = {
            "execution_date": date.today().isoformat(),
            "target_per_market": target_papers_per_market,
            "markets": {},
            "total_papers_harvested": 0,
            "total_chunks_ingested": 0,
        }

        all_records: list[DocChunkRecord] = []

        for regime in [
            MarketRegime.INDIA_NSE_BSE,
            MarketRegime.GERMANY_XETRA_DAX,
            MarketRegime.US_NYSE_NASDAQ,
        ]:
            logger.info("Starting harvesting pipeline for %s...", regime.value)
            papers = self.fetch_market_candidates(regime, target_count=target_papers_per_market)

            # Record metadata to jsonl
            jsonl_path = self.metadata_dir / f"{regime.value}_papers.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as f:
                for p in papers:
                    p_dict = asdict(p)
                    p_dict["published_date"] = p.published_date.isoformat()
                    # Strip binary bytes from metadata jsonl
                    p_dict.pop("raw_pdf_bytes", None)
                    p_dict.pop("raw_tex_content", None)
                    f.write(json.dumps(p_dict) + "\n")

            market_records: list[DocChunkRecord] = []
            full_downloads = 0
            for paper in papers:
                # Download full source for top candidates or arXiv papers
                should_download_full = (
                    full_downloads < full_source_downloads_per_market
                    or paper.source == HarvesterSource.ARXIV
                )
                cached_paper = self.cache_paper_source(
                    paper, download_full_text=should_download_full
                )
                if cached_paper.raw_tex_content or cached_paper.raw_pdf_bytes:
                    full_downloads += 1

                records = self.chunk_and_contextualize(cached_paper)
                market_records.extend(records)

            all_records.extend(market_records)
            market_stats = {
                "papers_count": len(papers),
                "full_sources_cached": full_downloads,
                "chunks_generated": len(market_records),
            }
            markets_dict = catalog_summary["markets"]
            if isinstance(markets_dict, dict):
                markets_dict[regime.value] = market_stats

            logger.info(
                "Completed %s: %d papers, %d chunks",
                regime.value,
                len(papers),
                len(market_records),
            )

        # Ingest all chunks into PostgreSQL 17
        total_inserted = self.ingest_records(all_records)
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
