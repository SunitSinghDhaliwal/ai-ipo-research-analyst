"""
Recover financial tables from page text.

Many prospectuses use borderless tables. pdfplumber's line-based table
detection then captures only the ruled header (e.g. "Fiscal | 2026 | 2025 | 2024")
and drops every data row. PyMuPDF's page text still contains the rows, one cell
per line:

    Fiscal
    2026
    2025
    2024
    Revenue from Operations
    1,468,853
    1,282,184
    1,095,581

This module rebuilds those rows as [label, value_2026, value_2025, value_2024]
so the financial agent can map each value to the right fiscal year.
"""

import re
from typing import List, Optional

YEAR_LINE = re.compile(r"^(?:fiscal|fy)?\s*(?:ended\s+)?(?:march\s+31,?\s*)?((?:19|20)\d{2})$", re.IGNORECASE)
NUMBER_LINE = re.compile(r"^\(?-?[\d,]*\d(?:\.\d+)?\)?\s*%?\)?$|^\(?-?[\d,]*\d(?:\.\d+)?\)?%$")
MIN_ROWS = 3


def _is_number(line: str) -> bool:
    return bool(NUMBER_LINE.match(line.replace(" ", "")))


def _is_percent(line: str) -> bool:
    return "%" in line


def _find_year_headers(lines: List[str]) -> List[int]:
    """Indices where a run of >= 2 consecutive year lines starts (e.g. 2026, 2025, 2024)."""
    starts = []
    i = 0
    while i < len(lines):
        if YEAR_LINE.match(lines[i]):
            j = i
            while j < len(lines) and YEAR_LINE.match(lines[j]):
                j += 1
            if j - i >= 2:
                starts.append(i)
            i = j
        else:
            i += 1
    return starts


def extract_text_tables(page_text: str) -> List[List[List[str]]]:
    """
    Return a list of tables; each table is a list of rows, first row is the header
    ["Particulars", "Fiscal 2026", "Fiscal 2025", "Fiscal 2024"].
    Only rows whose amount count equals the number of years are kept.
    """
    if not page_text:
        return []
    lines = [ln.strip() for ln in page_text.split("\n")]
    tables = []
    header_starts = _find_year_headers(lines)

    for h_idx, start in enumerate(header_starts):
        years: List[str] = []
        i = start
        while i < len(lines) and YEAR_LINE.match(lines[i]):
            years.append(YEAR_LINE.match(lines[i]).group(1))
            i += 1
        n = len(years)
        end = header_starts[h_idx + 1] if h_idx + 1 < len(header_starts) else len(lines)

        rows: List[List[str]] = []
        label_parts: List[str] = []
        while i < end:
            line = lines[i]
            if not line:
                label_parts = []  # blank line separates headings from row labels
                i += 1
                continue
            if _is_number(line):
                values: List[str] = []
                while i < end and lines[i] and _is_number(lines[i]):
                    if not _is_percent(lines[i]):
                        values.append(lines[i])
                    i += 1
                label = " ".join(label_parts).strip()
                if label and len(values) == n and re.search(r"[A-Za-z]", label):
                    rows.append([label] + values)
                label_parts = []
                continue
            # Unit / column caption lines, not row labels
            if re.fullmatch(r"\(?(in )?₹ ?(in )?(million|crore|lakh)s?\)?.*|\(% of.*|income\)|particulars", line, re.IGNORECASE):
                i += 1
                continue
            label_parts.append(line)
            if len(label_parts) > 3:  # long prose, not a table label
                label_parts = label_parts[-2:]
            i += 1

        if len(rows) >= MIN_ROWS:
            header = ["Particulars"] + [f"Fiscal {y}" for y in years]
            tables.append([header] + rows)
    return tables


def table_rows_preview(table: List[List[str]], limit: int = 6) -> Optional[str]:
    return "\n".join(" | ".join(r) for r in table[:limit]) if table else None
