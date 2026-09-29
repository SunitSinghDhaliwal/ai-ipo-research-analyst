"""
Generalized IPO Ingestion & Vector Indexing Pipeline.
Consolidates PDF extraction, page chunking, and Chroma vector indexing into a single reusable CLI.
Supports onboarding new IPOs without creating company-specific scripts.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional

# Ensure project paths are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "ingestion"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "retrieval"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "utils"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from langchain_core.documents import Document
from langchain_chroma import Chroma
from embeddings import get_embedding_model
from table_documents import load_table_documents
from ingest import ingest_document
from chunker import create_chunks

DEFAULT_VECTORSTORE_DIR = "data/vectorstore/ipo_knowledge_base"
DEFAULT_COLLECTION_NAME = "ipo_knowledge_base"


def load_chunks_and_tables(processed_dir: Path) -> tuple[List[Document], List[str]]:
    """
    Loads text chunks and structured tables from a processed document directory
    and converts them into LangChain Documents with complete metadata.
    """
    docs: List[Document] = []
    ids: List[str] = []

    # 1. Load Text Chunks
    chunks_path = processed_dir / "text_chunks.json"
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        for c in chunks:
            meta = {
                "chunk_id": c["chunk_id"],
                "company": c["company"],
                "ipo_id": c.get("ipo_id"),
                "document_type": c.get("document_type"),
                "document_version": c.get("document_version"),
                "filing_date": c.get("filing_date"),
                "is_latest": c.get("is_latest", True),
                "page": c["page"],
                "content_type": c.get("content_type", "text")
            }
            if "section" in c: meta["section"] = c["section"]
            if "chapter" in c: meta["chapter"] = c["chapter"]
            if "section_path" in c: meta["section_path"] = c["section_path"]
            docs.append(Document(page_content=c["text"], metadata=meta))
            ids.append(c["chunk_id"])
        print(f"Loaded {len(chunks)} text chunks from {chunks_path}")

    # 2. Load Structured Tables
    tables_path = processed_dir / "tables.json"
    pages_path = processed_dir / "pages.json"
    if tables_path.exists():
        t_docs, t_ids = load_table_documents(tables_path, pages_path)
        docs.extend(t_docs)
        ids.extend(t_ids)
        print(f"Loaded {len(t_docs)} tables from {tables_path}")

    return docs, ids


def index_into_chroma(
    documents: List[Document],
    ids: List[str],
    vectorstore_dir: str = DEFAULT_VECTORSTORE_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = 500
):
    """
    Indexes document objects and table representations into the shared Chroma collection.
    Appends to existing collection without deleting previously ingested companies.
    """
    if not documents:
        print("No documents to index.")
        return

    print(f"\nInitializing Chroma collection '{collection_name}' at '{vectorstore_dir}'...")
    embedding_model = get_embedding_model()
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(vectorstore_dir)
    )

    total = len(documents)
    print(f"Indexing {total} items in batches of {batch_size}...")
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        vector_store.add_documents(
            documents=documents[i:end],
            ids=ids[i:end]
        )
        print(f"  Indexed items {i + 1} to {end} / {total}")

    print("Vector store indexing complete!")


def run_pipeline(
    pdf_path: Optional[str] = None,
    company: str = "Maharashtra Oil Extractions Limited",
    ipo_id: str = "moel_ipo",
    doc_type: str = "DRHP",
    version: str = "drhp_v1",
    filing_date: str = "2024-03-20",
    is_latest: bool = True,
    processed_dir: Optional[str] = None,
    vectorstore_dir: str = DEFAULT_VECTORSTORE_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    index_only: bool = False
):
    """
    Executes end-to-end ingestion and indexing for an IPO document.
    """
    out_dir = Path(processed_dir) if processed_dir else Path(f"data/processed/{ipo_id}_{doc_type.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not index_only:
        if not pdf_path or not Path(pdf_path).exists():
            raise FileNotFoundError(f"Source PDF not found at {pdf_path}")

        print(f"\n--- 1. Ingesting PDF: {pdf_path} ({company}, {doc_type}) ---")
        ingest_document(
            pdf_path=pdf_path,
            company=company,
            document_type=doc_type,
            ipo_id=ipo_id,
            document_version=version,
            filing_date=filing_date,
            is_latest=is_latest,
            output_dir=out_dir
        )

        print(f"\n--- 2. Generating Chunks with Section Metadata ---")
        create_chunks(
            input_path=out_dir / "pages.json",
            output_path=out_dir / "text_chunks.json"
        )

    print(f"\n--- 3. Indexing into Chroma Knowledge Base ---")
    docs, ids = load_chunks_and_tables(out_dir)
    index_into_chroma(
        documents=docs,
        ids=ids,
        vectorstore_dir=vectorstore_dir,
        collection_name=collection_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generalized IPO Document Ingestion & Indexing CLI")
    parser.add_argument("--pdf", default=None, help="Path to raw prospectus PDF file")
    parser.add_argument("--company", required=True, help="Full legal name of the IPO company")
    parser.add_argument("--ipo_id", required=True, help="Canonical IPO offering identifier (e.g. hdbfs_ipo, moel_ipo)")
    parser.add_argument("--doc_type", default="DRHP", choices=["DRHP", "RHP"], help="Document type (DRHP or RHP)")
    parser.add_argument("--version", default="v1", help="Document version string")
    parser.add_argument("--filing_date", default="2024-01-01", help="Date of filing (YYYY-MM-DD)")
    parser.add_argument("--is_latest", action=argparse.BooleanOptionalAction, default=True, help="Whether this is the latest filing")
    parser.add_argument("--processed_dir", default=None, help="Directory containing pre-processed chunks and tables")
    parser.add_argument("--vectorstore_dir", default=DEFAULT_VECTORSTORE_DIR, help="Directory for Chroma vectorstore")
    parser.add_argument("--collection_name", default=DEFAULT_COLLECTION_NAME, help="Chroma collection name")
    parser.add_argument("--index-only", action="store_true", help="Skip PDF extraction and index existing processed JSONs")

    args = parser.parse_args()
    run_pipeline(
        pdf_path=args.pdf,
        company=args.company,
        ipo_id=args.ipo_id,
        doc_type=args.doc_type,
        version=args.version,
        filing_date=args.filing_date,
        is_latest=args.is_latest,
        processed_dir=args.processed_dir,
        vectorstore_dir=args.vectorstore_dir,
        collection_name=args.collection_name,
        index_only=args.index_only
    )
