#!/usr/bin/env python3
"""
SEC filing ingestion pipeline.

Downloads 10-K and 10-Q filings for target tickers via edgartools,
splits into section-aware chunks with metadata headers, and saves to data/filings/.

Usage:
    python src/ingest.py --tickers AAPL MSFT NVDA --filing-types 10-K 10-Q
    python src/ingest.py --tickers AAPL --filing-types 10-K --limit 2
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

CHUNK_SIZE = 2000     # characters (~512 tokens)
CHUNK_OVERLAP = 200   # characters (~10% overlap)
DATA_DIR = Path("data/filings")

# Sections to extract per form type — (edgartools key, human-readable name)
_SECTIONS: Dict[str, List[Tuple[str, str]]] = {
    "10-K": [
        ("Item 1A", "Risk Factors"),
        ("Item 7",  "Management Discussion and Analysis"),
        ("Item 8",  "Financial Statements"),
    ],
    "10-Q": [
        ("Part I, Item 1",   "Financial Statements"),
        ("Part I, Item 2",   "Management Discussion and Analysis"),
        ("Part II, Item 1A", "Risk Factors"),
    ],
}


# ── Public API ────────────────────────────────────────────────────────────────

def ingest(
    tickers: List[str],
    filing_types: List[str],
    limit: int = 2,
    output_dir: Path = DATA_DIR,
) -> List[Dict[str, Any]]:
    """
    Download, chunk, and save SEC filings for the given tickers.

    Returns the full list of chunk dicts (also written to output_dir).
    """
    _setup_edgar()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Dict[str, Any]] = []
    for ticker in tickers:
        for form_type in filing_types:
            if form_type not in _SECTIONS:
                print(f"[WARN] Unsupported form type '{form_type}', skipping.")
                continue
            chunks = _ingest_ticker(ticker, form_type, limit, output_dir)
            all_chunks.extend(chunks)

    index_path = output_dir / "index.json"
    index_path.write_text(
        json.dumps({"total_chunks": len(all_chunks), "chunks": all_chunks}, indent=2)
    )
    print(f"\nTotal: {len(all_chunks)} chunks saved to {output_dir}/")
    return all_chunks


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ingest_ticker(
    ticker: str,
    form_type: str,
    limit: int,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    from edgar import Company

    print(f"\n[{ticker}] {form_type} — fetching last {limit} filing(s)")
    company = Company(ticker)
    filings = company.get_filings(form=form_type)

    chunks: List[Dict[str, Any]] = []
    count = 0
    for filing in filings:
        if count >= limit:
            break
        try:
            filing_chunks = _process_filing(ticker, filing, form_type)
            chunks.extend(filing_chunks)
            count += 1
        except Exception as e:
            print(f"  [WARN] {filing.accession_no}: {e}")

    out_path = output_dir / f"{ticker}_{form_type.replace('-', '')}.json"
    out_path.write_text(json.dumps(chunks, indent=2))
    print(f"  → {len(chunks)} chunks saved to {out_path.name}")
    return chunks


def _process_filing(
    ticker: str,
    filing: Any,
    form_type: str,
) -> List[Dict[str, Any]]:
    meta = {
        "ticker": ticker,
        "company": filing.company,
        "form_type": form_type,
        "period": str(filing.period_of_report),
        "filing_date": str(filing.filing_date),
        "accession_no": filing.accession_no,
    }
    print(f"  period={meta['period']}  filed={meta['filing_date']}")

    doc = filing.obj()
    chunks: List[Dict[str, Any]] = []

    for section_key, section_name in _SECTIONS[form_type]:
        text = _extract_section_text(doc, section_key)
        if not text:
            print(f"    [SKIP] {section_key}: not found or empty")
            continue
        section_chunks = _chunk_text(text, section_key, section_name, meta)
        print(f"    {section_key}: {len(section_chunks)} chunks")
        chunks.extend(section_chunks)

    return chunks


def _extract_section_text(doc: Any, section_key: str) -> Optional[str]:
    """
    Pull plain text from a filing section.
    edgartools section objects expose .text, .content, or are directly str-able.
    """
    try:
        section = doc[section_key]
    except (KeyError, TypeError):
        return None

    if section is None:
        return None

    if hasattr(section, "text"):
        raw = section.text
    elif hasattr(section, "content"):
        raw = section.content
    else:
        raw = str(section)

    return raw.strip() or None


def _chunk_text(
    text: str,
    section_key: str,
    section_name: str,
    meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Split text into overlapping chunks; prepend metadata header to each."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(text)

    header = (
        f"[Company: {meta['ticker']} | "
        f"Form: {meta['form_type']} | "
        f"Period: {meta['period']} | "
        f"Section: {section_name}]"
    )

    # Stable, human-readable chunk ID
    section_slug = section_key.replace(" ", "").replace(",", "").lower()
    id_prefix = f"{meta['ticker']}_{meta['form_type'].replace('-','')}_{meta['period']}_{section_slug}"

    return [
        {
            "chunk_id": f"{id_prefix}_{i}",
            **meta,
            "section_key": section_key,
            "section_name": section_name,
            "chunk_index": i,
            "total_chunks_in_section": len(raw_chunks),
            "text": f"{header}\n\n{chunk_text}",
        }
        for i, chunk_text in enumerate(raw_chunks)
    ]


def _setup_edgar() -> None:
    from edgar import set_identity
    identity = os.getenv("SEC_EDGAR_USER_AGENT")
    if not identity:
        raise EnvironmentError(
            "SEC_EDGAR_USER_AGENT not set. "
            'Add to .env: SEC_EDGAR_USER_AGENT="Your Name your@email.com"'
        )
    set_identity(identity)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest SEC filings into chunked JSON for RAG indexing."
    )
    parser.add_argument(
        "--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA"],
        help="Ticker symbols to ingest (default: AAPL MSFT NVDA)",
    )
    parser.add_argument(
        "--filing-types", nargs="+", default=["10-K", "10-Q"],
        help="SEC form types (default: 10-K 10-Q)",
    )
    parser.add_argument(
        "--limit", type=int, default=2,
        help="Max filings per ticker per type (default: 2)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DATA_DIR,
        help=f"Output directory (default: {DATA_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ingest(
        tickers=args.tickers,
        filing_types=args.filing_types,
        limit=args.limit,
        output_dir=args.output_dir,
    )
