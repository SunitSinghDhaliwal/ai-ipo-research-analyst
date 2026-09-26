import json
from pathlib import Path

from langchain_core.documents import Document


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TABLES_PATH = Path(
    "data/processed/maharashtra_oil_extractions/tables.json"
)


# --------------------------------------------------
# Convert table rows to searchable text
# --------------------------------------------------

def table_to_text(rows):
    """
    Convert normalized table rows into an explicit, searchable
    text representation with column header identification and
    clean row formatting.
    """
    if not rows:
        return ""

    non_empty_rows = []
    for row in rows:
        cleaned_cells = [
            str(cell).strip() if cell is not None and str(cell).strip() != "" else "-"
            for cell in row
        ]
        if any(c != "-" for c in cleaned_cells):
            non_empty_rows.append(cleaned_cells)

    if not non_empty_rows:
        return ""

    lines = []

    # Format header row and data rows explicitly
    header_candidate = non_empty_rows[0]
    has_headers = len([c for c in header_candidate if c != "-"]) > 1

    start_row_idx = 0
    if has_headers:
        lines.append("Headers: " + " | ".join(header_candidate))
        start_row_idx = 1

    for idx, row in enumerate(non_empty_rows[start_row_idx:], start=1):
        lines.append(f"Row {idx}: " + " | ".join(row))

    return "\n".join(lines)


import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "utils"))
from section_detector import build_document_section_hierarchy

PAGES_PATH = Path(
    "data/processed/maharashtra_oil_extractions/pages.json"
)


# --------------------------------------------------
# Load tables
# --------------------------------------------------

def load_table_documents():

    with open(TABLES_PATH, "r", encoding="utf-8") as file:
        tables = json.load(file)

    section_hierarchy = {}
    if PAGES_PATH.exists():
        with open(PAGES_PATH, "r", encoding="utf-8") as file:
            pages = json.load(file)
            section_hierarchy = build_document_section_hierarchy(pages)

    documents = []
    ids = []

    for table in tables:

        table_text = table_to_text(table["rows"])

        if not table_text.strip():
            continue

        meaningful_cells = [
            cell.strip()
            for row in table["rows"]
            for cell in row
            if cell and cell.strip()
        ]

        if len(meaningful_cells) < 5:
            continue

        page_num = table["page"]
        page_info = section_hierarchy.get(page_num, {
            "section": "GENERAL",
            "chapter": "GENERAL",
            "section_path": "GENERAL"
        })
        sec_path = page_info["section_path"]

        document = Document(
            page_content=table_text,
            metadata={
                "table_id": table["table_id"],
                "company": table["company"],
                "document_type": table["document_type"],
                "page": page_num,
                "content_type": "table",
                "section": page_info["section"],
                "chapter": page_info["chapter"],
                "section_path": sec_path
            }
        )

        documents.append(document)
        ids.append(table["table_id"])

    return documents, ids



# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    documents, ids = load_table_documents()

    print(f"Loaded {len(documents)} tables.")

    if documents:

        print("\nFirst table:")
        print("=" * 80)

        print("ID:", ids[0])
        print("Metadata:", documents[0].metadata)

        print("\nSearchable representation:")
        print(documents[0].page_content)