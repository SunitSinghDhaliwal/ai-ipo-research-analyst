import argparse
import json
from pathlib import Path
from typing import Optional

import pymupdf
import pdfplumber

from table_normalizer import normalize_table


# --------------------------------------------------
# Default Configuration
# --------------------------------------------------

DEFAULT_PDF_PATH = "data/raw/maharashtra_oil_extractions_drhp.pdf"
DEFAULT_COMPANY = "Maharashtra Oil Extractions Limited"
DEFAULT_DOC_TYPE = "DRHP"
DEFAULT_IPO_ID = "moel_ipo"
DEFAULT_DOC_VERSION = "drhp_v1"
DEFAULT_FILING_DATE = "2024-03-20"
DEFAULT_OUTPUT_DIR = Path("data/processed/maharashtra_oil_extractions")


def ingest_document(
    pdf_path: str = DEFAULT_PDF_PATH,
    company: str = DEFAULT_COMPANY,
    document_type: str = DEFAULT_DOC_TYPE,
    ipo_id: str = DEFAULT_IPO_ID,
    document_version: str = DEFAULT_DOC_VERSION,
    filing_date: str = DEFAULT_FILING_DATE,
    is_latest: bool = True,
    output_dir: Optional[Path] = None
):
    """
    Generalized document ingestion pipeline supporting both DRHP and RHP filings.
    Extracts page text and structured tables with explicit IPO and document metadata.
    """
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_doc_type = document_type.upper()

    # ----------------------------------------------
    # 1. Extract text pages
    # ----------------------------------------------
    print(f"Extracting pages from {pdf_path} ({norm_doc_type})...")
    doc = pymupdf.open(pdf_path)
    pages = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()
        pages.append({
            "company": company,
            "ipo_id": ipo_id,
            "document_type": norm_doc_type,
            "document_version": document_version,
            "filing_date": filing_date,
            "is_latest": is_latest,
            "page": page_number,
            "content_type": "text",
            "text": text
        })

    doc.close()
    print(f"Extracted {len(pages)} pages.")

    pages_path = out_dir / "pages.json"
    with open(pages_path, "w", encoding="utf-8") as file:
        json.dump(pages, file, ensure_ascii=False, indent=2)
    print(f"Saved pages → {pages_path}")

    # ----------------------------------------------
    # 2. Extract tables
    # ----------------------------------------------
    print("\nExtracting tables...")
    all_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table_number, raw_table in enumerate(tables, start=1):
                # Use document-safe and company-safe naming convention
                if ipo_id == "moel_ipo":
                    if norm_doc_type == "DRHP":
                        table_id = f"moe_p{page_number}_t{table_number}"
                    else:
                        table_id = f"moe_{norm_doc_type.lower()}_p{page_number}_t{table_number}"
                else:
                    prefix = ipo_id.split("_")[0]
                    table_id = f"{prefix}_{norm_doc_type.lower()}_p{page_number}_t{table_number}"

                normalized_table = normalize_table(
                    raw_table=raw_table,
                    page_number=page_number,
                    table_id=table_id,
                    company=company,
                    document_type=norm_doc_type,
                    ipo_id=ipo_id,
                    document_version=document_version,
                    filing_date=filing_date,
                    is_latest=is_latest
                )
                all_tables.append(normalized_table)

    print(f"Extracted {len(all_tables)} tables.")

    tables_path = out_dir / "tables.json"
    with open(tables_path, "w", encoding="utf-8") as file:
        json.dump(all_tables, file, ensure_ascii=False, indent=2)
    print(f"Saved tables → {tables_path}")
    print("\nIngestion complete!")
    return pages, all_tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPO Document Ingestion Pipeline")
    parser.add_argument("--pdf", default=DEFAULT_PDF_PATH, help="Path to prospectus PDF")
    parser.add_argument("--company", default=DEFAULT_COMPANY, help="Legal company name")
    parser.add_argument("--doc_type", default=DEFAULT_DOC_TYPE, choices=["DRHP", "RHP"], help="Document type")
    parser.add_argument("--ipo_id", default=DEFAULT_IPO_ID, help="Canonical IPO offering identifier")
    parser.add_argument("--version", default=DEFAULT_DOC_VERSION, help="Filing version tag")
    parser.add_argument("--filing_date", default=DEFAULT_FILING_DATE, help="Filing date (YYYY-MM-DD)")
    parser.add_argument("--is_latest", action=argparse.BooleanOptionalAction, default=True, help="Whether this is the latest filing")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for processed JSONs")

    args = parser.parse_args()
    ingest_document(
        pdf_path=args.pdf,
        company=args.company,
        document_type=args.doc_type,
        ipo_id=args.ipo_id,
        document_version=args.version,
        filing_date=args.filing_date,
        is_latest=args.is_latest,
        output_dir=Path(args.output_dir)
    )