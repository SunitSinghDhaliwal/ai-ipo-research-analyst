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


def normalize_table(raw_table, page_number, table_id, company, document_type="DRHP"):
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
    "document_type": document_type,
    "page": page_number,
    "content_type": "table",
    "rows": rows
}