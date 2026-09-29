import re


def clean_cell(cell):
    """Clean a single PDF-extracted cell."""

    if cell is None:
        return None

    cell = str(cell)

    # Remove newlines and extra spaces
    cell = re.sub(r"\s+", " ", cell).strip()

    # Empty cells
    if cell == "":
        return None

    return cell


def normalize_table(
    raw_table,
    page_number,
    table_id,
    company,
    document_type="DRHP",
    ipo_id="moel_ipo",
    document_version="drhp_v1",
    filing_date="2024-03-20",
    is_latest=True
):
    """
    Normalize a raw pdfplumber table while preserving its structure.
    """

    # Clean every cell
    rows = []

    for row in raw_table:
        cleaned_row = [clean_cell(cell) for cell in row]

        # Skip completely empty rows
        if all(cell is None for cell in cleaned_row):
            continue

        rows.append(cleaned_row)

    return {
        "table_id": table_id,
        "company": company,
        "ipo_id": ipo_id,
        "document_type": document_type,
        "document_version": document_version,
        "filing_date": filing_date,
        "is_latest": is_latest,
        "page": page_number,
        "content_type": "table",
        "rows": rows
    }