import argparse
import json
from pathlib import Path
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(str(Path(__file__).resolve().parent.parent / "utils"))
from section_detector import build_document_section_hierarchy


# --------------------------------------------------
# Default Configuration
# --------------------------------------------------

DEFAULT_INPUT_PATH = Path(
    "data/processed/maharashtra_oil_extractions/pages.json"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/processed/maharashtra_oil_extractions/text_chunks.json"
)


def create_chunks(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    chunk_size: int = 1000,
    chunk_overlap: int = 150
):
    """
    Split extracted pages into overlapping text chunks with hierarchical section metadata
    and document versioning properties (ipo_id, document_type, document_version, filing_date, is_latest).
    """
    in_path = Path(input_path)
    out_path = Path(output_path)

    print(f"Loading pages from {in_path}...")
    with open(in_path, "r", encoding="utf-8") as file:
        pages = json.load(file)
    print(f"Loaded {len(pages)} pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    print("Detecting section hierarchy...")
    section_hierarchy = build_document_section_hierarchy(pages)

    chunks = []
    for page in pages:
        text = page.get("text", "").strip()
        if not text:
            continue

        page_chunks = splitter.split_text(text)
        page_info = section_hierarchy.get(page["page"], {
            "section": "GENERAL",
            "chapter": "GENERAL",
            "section_path": "GENERAL"
        })
        sec_path = page_info["section_path"]

        doc_type = page.get("document_type", "DRHP")
        ipo_id = page.get("ipo_id", "moel_ipo")
        doc_version = page.get("document_version", f"{doc_type.lower()}_v1")
        filing_date = page.get("filing_date", "2024-03-20")
        is_latest = page.get("is_latest", True)

        for chunk_number, chunk_text in enumerate(page_chunks, start=1):
            if ipo_id == "moel_ipo":
                chunk_id = f"{doc_type.lower()}_p{page['page']}_c{chunk_number}"
            else:
                prefix = ipo_id.split("_")[0]
                chunk_id = f"{prefix}_{doc_type.lower()}_p{page['page']}_c{chunk_number}"
            chunks.append({
                "chunk_id": chunk_id,
                "company": page["company"],
                "ipo_id": ipo_id,
                "document_type": doc_type,
                "document_version": doc_version,
                "filing_date": filing_date,
                "is_latest": is_latest,
                "page": page["page"],
                "content_type": "text",
                "section": page_info["section"],
                "chapter": page_info["chapter"],
                "section_path": sec_path,
                "text": chunk_text
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print(f"Created {len(chunks)} chunks.")
    print(f"Saved chunks → {out_path}")
    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Chunker with Version-Aware Metadata")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Path to input pages.json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Path to output text_chunks.json")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size")
    parser.add_argument("--chunk_overlap", type=int, default=150, help="Chunk overlap")

    args = parser.parse_args()
    create_chunks(
        input_path=Path(args.input),
        output_path=Path(args.output),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )