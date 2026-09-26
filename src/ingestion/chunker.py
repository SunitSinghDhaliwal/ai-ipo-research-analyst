import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_PATH = Path(
    "data/processed/maharashtra_oil_extractions/pages.json"
)

OUTPUT_PATH = Path(
    "data/processed/maharashtra_oil_extractions/text_chunks.json"
)


# --------------------------------------------------
# Load pages
# --------------------------------------------------

print("Loading pages...")

with open(INPUT_PATH, "r", encoding="utf-8") as file:
    pages = json.load(file)

print(f"Loaded {len(pages)} pages.")


# --------------------------------------------------
# Text splitter
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "utils"))
from section_detector import build_document_section_hierarchy

# --------------------------------------------------
# Compute document section hierarchy
# --------------------------------------------------

print("Detecting section hierarchy...")
section_hierarchy = build_document_section_hierarchy(pages)


# --------------------------------------------------
# Create chunks
# --------------------------------------------------

chunks = []

for page in pages:

    text = page["text"].strip()

    if not text:
        continue

    page_chunks = splitter.split_text(text)
    page_info = section_hierarchy.get(page["page"], {
        "section": "GENERAL",
        "chapter": "GENERAL",
        "section_path": "GENERAL"
    })
    sec_path = page_info["section_path"]

    for chunk_number, chunk_text in enumerate(page_chunks, start=1):

        chunks.append({
            "chunk_id": (
                f"{page['document_type'].lower()}"
                f"_p{page['page']}"
                f"_c{chunk_number}"
            ),
            "company": page["company"],
            "document_type": page["document_type"],
            "page": page["page"],
            "content_type": "text",
            "section": page_info["section"],
            "chapter": page_info["chapter"],
            "section_path": sec_path,
            "text": chunk_text
        })



# --------------------------------------------------
# Save chunks
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(
        chunks,
        file,
        ensure_ascii=False,
        indent=2
    )

print(f"Created {len(chunks)} chunks.")
print(f"Saved chunks → {OUTPUT_PATH}")