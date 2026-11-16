"""Audited PostgreSQL 17 doc_chunks hygiene + real open-source foundational doc enrichment.

Usage:
    .venv/bin/python scripts/clean_and_enrich_corpus.py --clean --enrich --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import urllib.parse
import urllib.request
from pathlib import Path
from uuid import NAMESPACE_DNS, UUID, uuid5

import psycopg

from context_scout.config import ScoutSettings
from context_scout.ingestion.doc_parser import (
    DocumentChunk,
    MarkdownTableChunker,
    clean_latex_artifacts,
    extract_sgml_docbook_text,
    is_license_or_stub_text,
)
from context_scout.retrieval.embedder import LocalSemanticEmbedder
from context_scout.schemas.contracts import MarketRegime

logger = logging.getLogger(__name__)

TARGET_CLEANUP_DOMAINS: tuple[str, ...] = ("data_engineering_scout", "quant_research_scout")

WIKIPEDIA_FOUNDATIONAL_PAGES: tuple[tuple[str, str, str], ...] = (
    (
        "ACID",
        "ACID (Atomicity, Consistency, Isolation, Durability)",
        "https://en.wikipedia.org/wiki/ACID",
    ),
    (
        "CAP_theorem",
        "CAP Theorem (Consistency, Availability, Partition Tolerance)",
        "https://en.wikipedia.org/wiki/CAP_theorem",
    ),
    (
        "PACELC_design_principle",
        "PACELC Theorem",
        "https://en.wikipedia.org/wiki/PACELC_design_principle",
    ),
    (
        "Multiversion_concurrency_control",
        "Multiversion Concurrency Control (MVCC)",
        "https://en.wikipedia.org/wiki/Multiversion_concurrency_control",
    ),
    (
        "Write-ahead_logging",
        "Write-Ahead Logging (WAL)",
        "https://en.wikipedia.org/wiki/Write-ahead_logging",
    ),
    (
        "Change_data_capture",
        "Change Data Capture (CDC)",
        "https://en.wikipedia.org/wiki/Change_data_capture",
    ),
    (
        "Database_transaction",
        "Database Transaction",
        "https://en.wikipedia.org/wiki/Database_transaction",
    ),
    (
        "Isolation_(database_systems)",
        "Transaction Isolation Levels",
        "https://en.wikipedia.org/wiki/Isolation_(database_systems)",
    ),
)

RAW_LATEX_REF_RE = re.compile(r"\\(?:label|cref|Cref|autoref|eqref|ref)\{[^}]*\}")


def fetch_wikipedia_plaintext(page_slug: str) -> str:
    """Fetches real CC-BY-SA 4.0 encyclopedic plain text from Wikipedia API."""
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
            "titles": page_slug,
            "format": "json",
        }
    )
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "enterprise-context-scout/1.0 (open-source corpus enrichment)"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        extract = page.get("extract", "")
        if isinstance(extract, str) and extract.strip():
            return _mediawiki_extract_to_markdown(extract)
    return ""


def _mediawiki_extract_to_markdown(extract: str) -> str:
    """Converts MediaWiki == Heading == markers into Markdown headings."""
    lines: list[str] = []
    for line in extract.splitlines():
        m4 = re.match(r"^====\s*(.+?)\s*====$", line)
        m3 = re.match(r"^===\s*(.+?)\s*===$", line)
        m2 = re.match(r"^==\s*(.+?)\s*==$", line)
        if m4:
            lines.append(f"#### {m4.group(1)}")
        elif m3:
            lines.append(f"### {m3.group(1)}")
        elif m2:
            lines.append(f"## {m2.group(1)}")
        else:
            lines.append(line)
    # Drop trailing References / External links / See also sections if present
    text = "\n".join(lines)
    text = re.split(
        r"\n##\s+(?:See also|References|External links|Further reading)\b", text, maxsplit=1
    )[0]
    return text.strip()


def get_domain_counts(conn: psycopg.Connection[tuple[object, ...]]) -> dict[str, int]:
    """Returns read-only doc_chunks counts grouped by domain_id."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT domain_id, count(*) FROM doc_chunks GROUP BY domain_id ORDER BY domain_id;"
        )
        return {str(row[0]): int(str(row[1])) for row in cur.fetchall()}


def clean_doc_chunks(settings: ScoutSettings) -> dict[str, dict[str, int]]:
    """Audits and cleans corrupt stubs, pure-license boilerplate, and raw LaTeX ref macros."""
    summary: dict[str, dict[str, int]] = {}
    with psycopg.connect(settings.db_admin_url) as conn:
        before_counts = get_domain_counts(conn)
        print("=== BEFORE CLEANUP COUNTS ===")
        for dom, cnt in before_counts.items():
            print(f"  {dom}: {cnt}")

        for domain_id in TARGET_CLEANUP_DOMAINS:
            delete_ids: list[UUID] = []
            repair_rows: list[tuple[str, UUID]] = []

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk_id, verbatim_content FROM doc_chunks WHERE domain_id = %s;",
                    (domain_id,),
                )
                for chunk_id_raw, content in cur.fetchall():
                    chunk_id = (
                        chunk_id_raw if isinstance(chunk_id_raw, UUID) else UUID(str(chunk_id_raw))
                    )
                    text = str(content or "")
                    stripped = text.strip()

                    is_tiny_punct_fragment = bool(
                        re.match(r"^[.,;:!?)\]]+", stripped) and len(stripped) < 120
                    )
                    if is_license_or_stub_text(stripped) or is_tiny_punct_fragment:
                        delete_ids.append(chunk_id)
                        continue

                    if RAW_LATEX_REF_RE.search(text):
                        cleaned = clean_latex_artifacts(text)
                        if cleaned != text and not is_license_or_stub_text(cleaned):
                            repair_rows.append((cleaned, chunk_id))

            deleted_count = 0
            repaired_count = 0
            with conn.cursor() as cur:
                if delete_ids:
                    cur.execute(
                        "DELETE FROM doc_chunks WHERE domain_id = %s AND chunk_id = ANY(%s);",
                        (domain_id, delete_ids),
                    )
                    deleted_count = cur.rowcount or 0
                if repair_rows:
                    cur.executemany(
                        "UPDATE doc_chunks SET verbatim_content = %s "
                        "WHERE domain_id = %s AND chunk_id = %s;",
                        [(txt, domain_id, cid) for txt, cid in repair_rows],
                    )
                    repaired_count = len(repair_rows)
            conn.commit()

            summary[domain_id] = {
                "before": before_counts.get(domain_id, 0),
                "deleted": deleted_count,
                "repaired": repaired_count,
                "after": before_counts.get(domain_id, 0) - deleted_count,
            }

        after_counts = get_domain_counts(conn)
        print("=== AFTER CLEANUP COUNTS ===")
        for dom, cnt in after_counts.items():
            print(f"  {dom}: {cnt}")
        print("=== CLEANUP SUMMARY ===", json.dumps(summary, indent=2))
    return summary


def build_foundational_chunks(repo_root: Path) -> list[DocumentChunk]:
    """Parses real upstream Wikipedia CC-BY-SA + local official PostgreSQL SGML docs."""
    chunker = MarkdownTableChunker()
    all_chunks: list[DocumentChunk] = []

    for slug, title, source_uri in WIKIPEDIA_FOUNDATIONAL_PAGES:
        md_text = fetch_wikipedia_plaintext(slug)
        if not md_text:
            logger.warning("Empty extract for Wikipedia page %s", slug)
            continue
        doc_id = f"data_engineering_scout:upstream/wikipedia/{slug}"
        raw_chunks = chunker.parse_and_chunk(
            doc_id=doc_id,
            doc_title=title,
            source_uri=source_uri,
            market_scope=MarketRegime.GLOBAL_MULTI,
            markdown_content=f"# {title}\n\n{md_text}",
            min_chunk_tokens=300,
            max_chunk_tokens=700,
        )
        for idx, ch in enumerate(raw_chunks):
            det_id = uuid5(NAMESPACE_DNS, f"{doc_id}:{idx}")
            all_chunks.append(
                DocumentChunk(
                    chunk_id=det_id,
                    doc_id=doc_id,
                    doc_title=title,
                    source_uri=source_uri,
                    market_scope=MarketRegime.GLOBAL_MULTI,
                    section_heading=ch.section_heading if ch.section_heading != "Main" else title,
                    verbatim_content=ch.verbatim_content,
                    formulas_latex=ch.formulas_latex,
                    tables_markdown=ch.tables_markdown,
                    token_count_approx=ch.token_count_approx,
                    metadata_json={
                        "source_type": "wikipedia_cc_by_sa",
                        "slug": slug,
                        "chunk_index": idx,
                        "total_chunks": len(raw_chunks),
                    },
                )
            )
        logger.info("Parsed %d chunks from %s", len(raw_chunks), source_uri)

    # Re-chunk local PostgreSQL official SGML docs (mvcc.sgml, wal.sgml) cleanly
    pg_sgml_dir = repo_root / "data/data_engineering/train/repos/postgres__postgres/doc/src/sgml"
    for fname, title in (
        ("mvcc.sgml", "PostgreSQL Concurrency Control (MVCC)"),
        ("wal.sgml", "PostgreSQL Write-Ahead Logging (WAL)"),
    ):
        sgml_path = pg_sgml_dir / fname
        if not sgml_path.is_file():
            continue
        md_text = extract_sgml_docbook_text(sgml_path.read_text(encoding="utf-8", errors="replace"))
        doc_id = f"data_engineering_scout:upstream/postgres/{fname}"
        source_uri = (
            f"https://raw.githubusercontent.com/postgres/postgres/master/doc/src/sgml/{fname}"
        )
        raw_chunks = chunker.parse_and_chunk(
            doc_id=doc_id,
            doc_title=title,
            source_uri=source_uri,
            market_scope=MarketRegime.GLOBAL_MULTI,
            markdown_content=md_text,
            min_chunk_tokens=350,
            max_chunk_tokens=750,
        )
        for idx, ch in enumerate(raw_chunks):
            det_id = uuid5(NAMESPACE_DNS, f"{doc_id}:{idx}")
            all_chunks.append(
                DocumentChunk(
                    chunk_id=det_id,
                    doc_id=doc_id,
                    doc_title=title,
                    source_uri=source_uri,
                    market_scope=MarketRegime.GLOBAL_MULTI,
                    section_heading=ch.section_heading if ch.section_heading != "Main" else title,
                    verbatim_content=ch.verbatim_content,
                    formulas_latex=ch.formulas_latex,
                    tables_markdown=ch.tables_markdown,
                    token_count_approx=ch.token_count_approx,
                    metadata_json={
                        "source_type": "postgres_official_sgml",
                        "filename": fname,
                        "chunk_index": idx,
                        "total_chunks": len(raw_chunks),
                    },
                )
            )
        logger.info("Parsed %d clean chunks from %s", len(raw_chunks), fname)

    return all_chunks


def enrich_data_engineering_corpus(settings: ScoutSettings, repo_root: Path) -> int:
    """Embeds real foundational docs with EmbeddingGemma 2 and upserts into PG17."""
    chunks = build_foundational_chunks(repo_root)
    if not chunks:
        return 0

    embedder = LocalSemanticEmbedder(settings=settings)
    texts_to_embed = [
        f"Document: {c.doc_title} | Section: {c.section_heading}\n{c.verbatim_content}"
        for c in chunks
    ]
    embeddings = embedder.embed_batch(texts_to_embed, is_query=False, batch_size=16)

    upsert_sql = """
        INSERT INTO doc_chunks (
            chunk_id, domain_id, doc_id, doc_title, source_uri,
            market_scope, published_date, authority_weight,
            section_heading, contextual_header, verbatim_content,
            metadata_json, allowed_scopes, embedding
        )
        VALUES (
            %s, 'data_engineering_scout', %s, %s, %s,
            %s, NULL, 1.2,
            %s, %s, %s,
            %s::jsonb,
            ARRAY[
                'public', 'public_eng', 'public_quant', 'public_de',
                'domain:data_engineering', 'domain:data_engineering_scout'
            ],
            %s::vector
        )
        ON CONFLICT (chunk_id) DO UPDATE SET
            doc_title = EXCLUDED.doc_title,
            source_uri = EXCLUDED.source_uri,
            authority_weight = EXCLUDED.authority_weight,
            section_heading = EXCLUDED.section_heading,
            contextual_header = EXCLUDED.contextual_header,
            verbatim_content = EXCLUDED.verbatim_content,
            metadata_json = EXCLUDED.metadata_json,
            allowed_scopes = EXCLUDED.allowed_scopes,
            embedding = EXCLUDED.embedding;
    """
    with psycopg.connect(settings.db_admin_url) as conn:
        with conn.cursor() as cur:
            for ch, emb in zip(chunks, embeddings, strict=True):
                ctx_header = (
                    f"Foundational data engineering reference: "
                    f"{ch.doc_title} — {ch.section_heading}."
                )
                emb_literal = "[" + ",".join(f"{v:.7f}" for v in emb) + "]"
                cur.execute(
                    upsert_sql,
                    (
                        ch.chunk_id,
                        ch.doc_id,
                        ch.doc_title,
                        ch.source_uri,
                        ch.market_scope.value,
                        ch.section_heading,
                        ctx_header,
                        ch.verbatim_content,
                        json.dumps(ch.metadata_json),
                        emb_literal,
                    ),
                )
        conn.commit()
    print(f"Upserted {len(chunks)} real foundational chunks into data_engineering_scout.")
    return len(chunks)


def verify_queries(settings: ScoutSettings) -> None:
    """Runs read-only vector + FTS verification queries against data_engineering_scout."""
    queries = [
        "What is ACID",
        "Explain CAP theorem",
        "What is Apache Iceberg?",
        "How does PostgreSQL MVCC work?",
        "What is PACELC?",
    ]
    embedder = LocalSemanticEmbedder(settings=settings)
    q_embs = embedder.embed_batch(queries, is_query=True, batch_size=8)

    with psycopg.connect(settings.db_admin_url) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order';")
        for q, emb in zip(queries, q_embs, strict=True):
            emb_literal = "[" + ",".join(f"{v:.7f}" for v in emb) + "]"
            cur.execute(
                """
                SELECT doc_title, section_heading,
                       1 - (embedding <=> %s::vector) AS cosine_sim,
                       left(
                           regexp_replace(verbatim_content, E'[\\n\\r]+', ' ', 'g'),
                           140
                       ) AS preview
                FROM doc_chunks
                WHERE domain_id = 'data_engineering_scout'
                ORDER BY embedding <=> %s::vector ASC
                LIMIT 3;
                """,
                (emb_literal, emb_literal),
            )
            print(f"\n[VERIFY QUERY] {q!r}")
            for row in cur.fetchall():
                print(f"  cos={row[2]:.3f} | {row[0]} > {row[1]} | {row[3]}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Clean broken doc_chunks & enrich foundational docs."
    )
    parser.add_argument("--clean", action="store_true", help="Run audited PG17 doc_chunks cleanup")
    parser.add_argument(
        "--enrich", action="store_true", help="Ingest real upstream foundational docs"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Run read-only vector retrieval checks"
    )
    args = parser.parse_args()

    settings = ScoutSettings()
    repo_root = Path(__file__).resolve().parent.parent

    if args.clean:
        clean_doc_chunks(settings)
    if args.enrich:
        enrich_data_engineering_corpus(settings, repo_root)
    if args.verify:
        verify_queries(settings)


if __name__ == "__main__":
    main()
