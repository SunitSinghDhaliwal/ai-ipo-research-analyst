from langchain_chroma import Chroma

from embeddings import get_embedding_model


# --------------------------------------------------
# Configuration
# --------------------------------------------------

VECTORSTORE_DIR = "data/vectorstore/maharashtra_oil_extractions"
COLLECTION_NAME = "maharashtra_oil_extractions"


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

embedding_model = get_embedding_model()


# --------------------------------------------------
# Load existing Chroma database
# --------------------------------------------------

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=VECTORSTORE_DIR
)


# --------------------------------------------------
# Query
# --------------------------------------------------

query = "What was the company's revenue in the latest fiscal year?"

# Optional metadata filter.
# None = search the entire collection.
metadata_filter = {"content_type": "table"}


# --------------------------------------------------
# Retrieve documents
# --------------------------------------------------

results = vector_store.similarity_search_with_score(
    query,
    k=20,
    filter=metadata_filter
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print(f"\nQuery: {query}\n")
print("=" * 80)

for rank, (document, score) in enumerate(results, start=1):

    print(f"\nRESULT {rank}")
    print("-" * 80)

    content_type = document.metadata["content_type"]

    if content_type == "text":
        identifier = document.metadata["chunk_id"]
    else:
        identifier = document.metadata["table_id"]

    print(f"ID: {identifier}")
    print(f"Page: {document.metadata['page']}")
    print(f"Content type: {content_type}")
    print(f"Chroma distance: {score:.4f}")

    print("\nText:")
    print(document.page_content[:500])