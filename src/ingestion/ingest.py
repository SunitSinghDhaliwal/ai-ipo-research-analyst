import json
from pathlib import Path

import pymupdf
import pdfplumber

from table_normalizer import normalize_table


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PDF_PATH = "data/raw/maharashtra_oil_extractions_drhp.pdf"

COMPANY = "Maharashtra Oil Extractions Limited"
DOCUMENT_TYPE = "DRHP"

OUTPUT_DIR = Path("data/processed/maharashtra_oil_extractions")


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Extract pages
# --------------------------------------------------

print("Extracting pages...")

document = pymupdf.open(PDF_PATH)

pages = []

for page_number, page in enumerate(document, start=1):

    text = page.get_text()

    pages.append({
    "company": COMPANY,
    "document_type": DOCUMENT_TYPE,
    "page": page_number,
    "content_type": "text",
    "text": text
})

document.close()

print(f"Extracted {len(pages)} pages.")


# --------------------------------------------------
# Save pages
# --------------------------------------------------

pages_path = OUTPUT_DIR / "pages.json"

with open(pages_path, "w", encoding="utf-8") as file:
    json.dump(pages, file, ensure_ascii=False, indent=2)

print(f"Saved pages → {pages_path}")


# --------------------------------------------------
# Extract tables
# --------------------------------------------------

print("\nExtracting tables...")

all_tables = []

with pdfplumber.open(PDF_PATH) as pdf:

    for page_number, page in enumerate(pdf.pages, start=1):

        tables = page.extract_tables()

        for table_number, raw_table in enumerate(tables, start=1):

            table_id = f"moe_p{page_number}_t{table_number}"

            normalized_table = normalize_table(
                raw_table=raw_table,
                page_number=page_number,
                table_id=table_id,
                company=COMPANY,
                document_type=DOCUMENT_TYPE
            )

            all_tables.append(normalized_table)


print(f"Extracted {len(all_tables)} tables.")


# --------------------------------------------------
# Save tables
# --------------------------------------------------

tables_path = OUTPUT_DIR / "tables.json"

with open(tables_path, "w", encoding="utf-8") as file:
    json.dump(all_tables, file, ensure_ascii=False, indent=2)

print(f"Saved tables → {tables_path}")


# --------------------------------------------------
# Finished
# --------------------------------------------------

print("\nIngestion complete!")