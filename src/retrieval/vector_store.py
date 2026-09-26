import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma

from embeddings import get_embedding_model
from table_documents import load_table_documents


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TEXT_CHUNKS_PATH = Path(
    "data/processed/maharashtra_oil_extractions/text_chunks.json"
)

VECTORSTORE_DIR = Path(
    "data/vectorstore/maharashtra_oil_extractions"
)

COLLECTION_NAME = "maharashtra_oil_extractions"


# --------------------------------------------------
# Load text chunks
# --------------------------------------------------

print("Loading text chunks...")

with open(TEXT_CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Loaded {len(chunks)} text chunks.")


# --------------------------------------------------
# Convert text chunks to Documents
# --------------------------------------------------

documents = []
ids = []

for chunk in chunks:

    meta = {
        "chunk_id": chunk["chunk_id"],
        "company": chunk["company"],
        "document_type": chunk["document_type"],
        "page": chunk["page"],
        "content_type": chunk["content_type"]
    }
    if "section" in chunk:
        meta["section"] = chunk["section"]
    if "chapter" in chunk:
        meta["chapter"] = chunk["chapter"]
    if "section_path" in chunk:
        meta["section_path"] = chunk["section_path"]

    document = Document(
        page_content=chunk["text"],
        metadata=meta
    )

    documents.append(document)
    ids.append(chunk["chunk_id"])



# --------------------------------------------------
# Load table documents
# --------------------------------------------------

print("\nLoading table documents...")

table_documents, table_ids = load_table_documents()

print(f"Loaded {len(table_documents)} tables.")


# --------------------------------------------------
# Add tables to documents
# --------------------------------------------------

documents.extend(table_documents)
ids.extend(table_ids)

print(f"\nTotal documents to index: {len(documents)}")


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("\nLoading embedding model...")

embedding_model = get_embedding_model()


import shutil

# --------------------------------------------------
# Create / load Chroma
# --------------------------------------------------

print("\nRebuilding Chroma vector store...")

if VECTORSTORE_DIR.exists():
    shutil.rmtree(VECTORSTORE_DIR, ignore_errors=True)


vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=str(VECTORSTORE_DIR)
)


# --------------------------------------------------
# Add documents
# --------------------------------------------------

print("Adding documents to Chroma...")

vector_store.add_documents(
    documents=documents,
    ids=ids
)


# --------------------------------------------------
# Finished
# --------------------------------------------------

print("\nVector store updated successfully!")
print(f"Indexed {len(documents)} documents.")