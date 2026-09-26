import re

# Generic SEBI DRHP/RHP Section Regex:
# Matches lines like "SECTION I – GENERAL", "SECTION II – RISK FACTORS", etc.
SECTION_REGEX = re.compile(
    r'^\s*(SECTION\s+[IVXLCDM\d]+\s*[\–\-\:\—]?\s*[A-Z\s\,\&\(\)]+)',
    re.MULTILINE
)

# Generic Chapter / Major Heading Regex:
# Matches standalone uppercase lines (4 to 60 characters)
CHAPTER_REGEX = re.compile(
    r'^\s*([A-Z][A-Z0-9\s\,\–\-\&\(\)\'\"]{3,58})\s*$',
    re.MULTILINE
)

NOISE_HEADINGS = {
    "DRAFT RED HERRING PROSPECTUS", "RED HERRING PROSPECTUS", "PROSPECTUS",
    "TABLE OF CONTENTS", "DRHP", "RHP", "CONFIDENTIAL", "DATED", "LIMITED",
    "CIN", "CORPORATE IDENTITY NUMBER"
}


def _clean_heading(text):
    text = re.sub(r'\s*\.\.\.*.*$', '', text)
    text = re.sub(r'\s*\d+$', '', text)
    return text.strip()


def extract_headings_from_page(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    sections = []
    chapters = []

    for line in lines:
        if "........" in line or line.count(".") > 4:
            continue

        # 1. Section Header Check
        sec_match = SECTION_REGEX.match(line)
        if sec_match:
            hdr = _clean_heading(sec_match.group(1))
            if len(hdr) > 5 and not any(nw in hdr for nw in ["Act", "Section 26", "Section 40"]):
                sections.append(hdr)
                continue

        # 2. Structural Uppercase Chapter Heading Check
        chap_match = CHAPTER_REGEX.match(line)
        if chap_match:
            hdr = _clean_heading(chap_match.group(1))
            if hdr in NOISE_HEADINGS or len(hdr) < 5 or hdr.isdigit():
                continue
            if re.match(r'^\d+[\.\,]?\d*$', hdr) or "Rs." in hdr or "₹" in hdr:
                continue
            chapters.append(hdr)

    return sections, chapters


def build_document_section_hierarchy(pages):
    """
    Generic state machine that scans pages from Page 1 to Page N,
    tracks Section and Chapter heading transitions, and returns
    a mapping: page_number -> {"section": ..., "chapter": ..., "section_path": ...}
    """
    current_section = "GENERAL"
    current_chapter = "COVER PAGE"

    hierarchy_map = {}

    for p in pages:
        page_num = p["page"]
        text = p["text"]

        # Skip Table of Contents pages
        if "........" in text and page_num <= 5:
            hierarchy_map[page_num] = {
                "section": current_section,
                "chapter": "TABLE OF CONTENTS",
                "section_path": f"{current_section} > TABLE OF CONTENTS"
            }
            continue

        secs, chaps = extract_headings_from_page(text)

        if secs:
            current_section = secs[-1]

        if chaps:
            valid_chaps = [c for c in chaps if len(c) >= 6 and c not in ["STATEMENT OF", "PARTICULARS", "DESCRIPTION"]]
            if valid_chaps:
                current_chapter = valid_chaps[0]

        section_path = f"{current_section} > {current_chapter}" if current_chapter else current_section

        hierarchy_map[page_num] = {
            "section": current_section,
            "chapter": current_chapter,
            "section_path": section_path
        }

    return hierarchy_map
