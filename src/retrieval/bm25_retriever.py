import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from table_documents import load_table_documents


# --------------------------------------------------
# Paths
# --------------------------------------------------

TEXT_CHUNKS_PATH = Path(
    "data/processed/maharashtra_oil_extractions/text_chunks.json"
)


PROCESSED_DIR = Path("data/processed")


# --------------------------------------------------
# Load text chunks
# --------------------------------------------------

def load_text_documents(chunks_path=None):

    if chunks_path is not None:
        chunk_files = [Path(chunks_path)]
    else:
        chunk_files = sorted(PROCESSED_DIR.glob("*/text_chunks.json"))
        if not chunk_files and TEXT_CHUNKS_PATH.exists():
            chunk_files = [TEXT_CHUNKS_PATH]

    documents = []

    for c_file in chunk_files:
        with open(c_file, "r", encoding="utf-8") as file:
            chunks = json.load(file)

        for chunk in chunks:
            meta = {
                "chunk_id": chunk["chunk_id"],
                "company": chunk["company"],
                "ipo_id": chunk.get("ipo_id", "moel_ipo"),
                "document_type": chunk.get("document_type", "DRHP"),
                "document_version": chunk.get("document_version", "drhp_v1"),
                "filing_date": chunk.get("filing_date", "2024-03-20"),
                "is_latest": chunk.get("is_latest", True),
                "page": chunk["page"],
                "content_type": chunk["content_type"]
            }
            if "section" in chunk:
                meta["section"] = chunk["section"]
            if "chapter" in chunk:
                meta["chapter"] = chunk["chapter"]
            if "section_path" in chunk:
                meta["section_path"] = chunk["section_path"]

            documents.append({
                "id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": meta
            })


    return documents


# --------------------------------------------------
# Load all searchable documents
# --------------------------------------------------

def load_all_documents():

    text_documents = load_text_documents()

    table_documents, table_ids = load_table_documents()

    documents = text_documents

    for document, table_id in zip(
        table_documents,
        table_ids
    ):

        documents.append({
            "id": table_id,
            "text": document.page_content,
            "metadata": document.metadata
        })

    return documents


# --------------------------------------------------
# Build BM25 index
# --------------------------------------------------

def build_bm25_index(documents):

    corpus = []
    for document in documents:
        sec_path = document.get("metadata", {}).get("section_path", "")
        if sec_path:
            full_bm25_text = f"Section: {sec_path}\n{document['text']}"
        else:
            full_bm25_text = document["text"]
        corpus.append(full_bm25_text.lower().split())

    return BM25Okapi(corpus)


# --------------------------------------------------
# BM25 search
# --------------------------------------------------

def bm25_search(
    query,
    bm25,
    documents,
    k=20,
    metadata_filter=None
):

    # ----------------------------------------------
    # Apply metadata filter
    # ----------------------------------------------

    if metadata_filter is not None:

        filtered_documents = []

        for document in documents:
            doc_meta = document.get("metadata", {})
            matches = True
            for key, value in metadata_filter.items():
                doc_val = doc_meta.get(key)
                # If checking is_latest, fallback to True if missing for backward compatibility
                if key == "is_latest" and doc_val is None:
                    doc_val = True
                if doc_val != value:
                    matches = False
                    break

            if matches:
                filtered_documents.append(document)

    else:

        filtered_documents = documents

    if not filtered_documents:
        return []


    # ----------------------------------------------
    # Build BM25 index for filtered documents
    # ----------------------------------------------

    corpus = []
    for document in filtered_documents:
        sec_path = document.get("metadata", {}).get("section_path", "")
        if sec_path:
            full_bm25_text = f"Section: {sec_path}\n{document['text']}"
        else:
            full_bm25_text = document["text"]
        corpus.append(full_bm25_text.lower().split())

    filtered_bm25 = BM25Okapi(corpus)


    # ----------------------------------------------
    # Search
    # ----------------------------------------------

    tokenized_query = query.lower().split()

    scores = filtered_bm25.get_scores(tokenized_query)

    ranked_indices = scores.argsort()[::-1][:k]


    # ----------------------------------------------
    # Return results
    # ----------------------------------------------

    results = []

    for index in ranked_indices:

        document = filtered_documents[index]

        results.append({
            "document": document,
            "score": scores[index]
        })

    return results


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    print("Loading documents...")

    documents = load_all_documents()

    print(
        f"Loaded {len(documents)} "
        "searchable documents."
    )

    print("\nBuilding BM25 index...")

    bm25 = build_bm25_index(documents)

    query = (
        "What was the company's revenue "
        "in the latest fiscal year?"
    )

    metadata_filter = {
        "content_type": "table"
    }

    results = bm25_search(
        query=query,
        bm25=bm25,
        documents=documents,
        k=20,
        metadata_filter=metadata_filter
    )

    print(f"\nQuery: {query}")
    print("=" * 80)

    for rank, result in enumerate(
        results,
        start=1
    ):

        document = result["document"]

        print(f"\nRESULT {rank}")
        print("-" * 80)

        print(f"ID: {document['id']}")
        print(
            f"Page: "
            f"{document['metadata']['page']}"
        )

        print(
            f"Content type: "
            f"{document['metadata']['content_type']}"
        )

        print(
            f"BM25 score: "
            f"{result['score']:.4f}"
        )

        print("\nText:")
        print(document["text"][:500])