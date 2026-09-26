from bm25_retriever import (
    load_all_documents,
    build_bm25_index,
    bm25_search
)

from langchain_chroma import Chroma

from embeddings import get_embedding_model
from reranker import rerank


# --------------------------------------------------
# Configuration
# --------------------------------------------------

VECTORSTORE_DIR = (
    "data/vectorstore/maharashtra_oil_extractions"
)

COLLECTION_NAME = "maharashtra_oil_extractions"

RRF_K = 60


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

embedding_model = get_embedding_model()


# --------------------------------------------------
# Load Chroma
# --------------------------------------------------

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=VECTORSTORE_DIR
)


# --------------------------------------------------
# Load BM25
# --------------------------------------------------

documents = load_all_documents()

bm25 = build_bm25_index(documents)


# --------------------------------------------------
# Reciprocal Rank Fusion
# --------------------------------------------------

def reciprocal_rank_fusion(
    vector_results,
    bm25_results,
    k=60,
    top_k=10
):

    scores = {}
    document_lookup = {}

    # ----------------------------------------------
    # Chroma results
    # ----------------------------------------------

    for rank, (document, distance) in enumerate(
        vector_results,
        start=1
    ):

        document_id = (
            document.metadata.get("chunk_id")
            or document.metadata.get("table_id")
        )

        scores[document_id] = (
            scores.get(document_id, 0)
            + 1 / (k + rank)
        )

        document_lookup[document_id] = document


    # ----------------------------------------------
    # BM25 results
    # ----------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        document = result["document"]

        document_id = document["id"]

        scores[document_id] = (
            scores.get(document_id, 0)
            + 1 / (k + rank)
        )

        document_lookup[document_id] = document


    # ----------------------------------------------
    # Final ranking
    # ----------------------------------------------

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for document_id, score in ranked[:top_k]:

        results.append({
            "document_id": document_id,
            "document": document_lookup[document_id],
            "rrf_score": score
        })

    return results


# --------------------------------------------------
# Main retrieval interface
# --------------------------------------------------

def hybrid_search(
    query,
    top_k=5,
    use_reranker=False,
    metadata_filter=None,
    vector_top_k=20,
    bm25_top_k=20,
    rrf_candidate_k=20,
    final_top_k=None
):
    """
    Unified hybrid retrieval pipeline interface.

    Pipeline:
      1. Chroma dense retrieval
      2. BM25 sparse retrieval
      3. Reciprocal Rank Fusion (RRF) candidate pool
      4. Optional Cross-Encoder reranking (BAAI/bge-reranker-base)
      5. Return top_k final results
    """
    if final_top_k is not None:
        top_k = final_top_k

    # ----------------------------------------------
    # Candidate pool sizing: ensure candidate pool is larger than top_k
    # ----------------------------------------------
    candidate_k = max(rrf_candidate_k, top_k * 2, 20)
    v_k = max(vector_top_k, candidate_k)
    b_k = max(bm25_top_k, candidate_k)

    # ----------------------------------------------
    # 1. Chroma dense retrieval
    # ----------------------------------------------
    vector_results = vector_store.similarity_search_with_score(
        query,
        k=v_k,
        filter=metadata_filter
    )

    # ----------------------------------------------
    # 2. BM25 sparse retrieval
    # ----------------------------------------------
    bm25_results = bm25_search(
        query=query,
        bm25=bm25,
        documents=documents,
        k=b_k,
        metadata_filter=metadata_filter
    )

    # ----------------------------------------------
    # 3. Reciprocal Rank Fusion (RRF)
    # ----------------------------------------------
    rrf_candidates = reciprocal_rank_fusion(
        vector_results=vector_results,
        bm25_results=bm25_results,
        k=RRF_K,
        top_k=candidate_k
    )

    if not use_reranker:
        return rrf_candidates[:top_k]

    # ----------------------------------------------
    # 4. Optional Cross-Encoder Reranking
    # ----------------------------------------------
    reranked = rerank(
        query=query,
        documents=rrf_candidates,
        top_k=top_k
    )

    results = []
    for item, score in reranked:
        res = dict(item)
        res["rerank_score"] = float(score)
        results.append(res)

    return results


def retrieve(
    query,
    k=10,
    metadata_filter=None,
    use_reranker=True
):
    return hybrid_search(
        query=query,
        top_k=k,
        use_reranker=use_reranker,
        metadata_filter=metadata_filter
    )



# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    query = (
        "What are the major risks "
        "associated with the company?"
    )

    results = retrieve(
        query=query,
        k=10,
        metadata_filter=None
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

        print(
            f"ID: "
            f"{result['document_id']}"
        )

        print(
            f"RRF score: "
            f"{result['rrf_score']:.6f}"
        )

        if "rerank_score" in result:
            print(
                f"Rerank score: "
                f"{result['rerank_score']:.6f}"
            )

        if hasattr(document, "metadata"):

            print(
                f"Page: "
                f"{document.metadata['page']}"
            )

            print(
                f"Content type: "
                f"{document.metadata['content_type']}"
            )

            print("\nText:")
            print(document.page_content[:500])

        else:

            print(
                f"Page: "
                f"{document['metadata']['page']}"
            )

            print(
                f"Content type: "
                f"{document['metadata']['content_type']}"
            )

            print("\nText:")
            print(document["text"][:500])