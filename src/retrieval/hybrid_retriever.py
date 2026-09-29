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

VECTORSTORE_DIR = "data/vectorstore/ipo_knowledge_base"
COLLECTION_NAME = "ipo_knowledge_base"

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
# Document Scope Resolution Helper
# --------------------------------------------------

def resolve_document_filter(document_target="latest", metadata_filter=None, ipo_id=None):
    """
    Combines caller metadata filter with version-aware document scoping and company isolation.
      - 'latest': filters for is_latest=True (authoritative current filing)
      - 'drhp': filters for document_type='DRHP'
      - 'rhp': filters for document_type='RHP'
    """
    merged = dict(metadata_filter or {})
    if ipo_id:
        merged.setdefault("ipo_id", ipo_id)
    target = (document_target or "latest").lower()

    if target == "latest":
        merged.setdefault("is_latest", True)
    elif target == "drhp":
        merged["document_type"] = "DRHP"
    elif target == "rhp":
        merged["document_type"] = "RHP"
    return merged


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
    final_top_k=None,
    document_target="latest",
    ipo_id=None
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

    effective_filter = resolve_document_filter(
        document_target=document_target,
        metadata_filter=metadata_filter,
        ipo_id=ipo_id
    ) if document_target != "none" else metadata_filter

    if ipo_id and effective_filter is not None:
        effective_filter.setdefault("ipo_id", ipo_id)

    # ----------------------------------------------
    # Candidate pool sizing: ensure candidate pool is larger than top_k
    # ----------------------------------------------
    candidate_k = max(rrf_candidate_k, top_k * 2, 20)
    v_k = max(vector_top_k, candidate_k)
    b_k = max(bm25_top_k, candidate_k)

    # ----------------------------------------------
    # 1. Chroma dense retrieval
    # ----------------------------------------------
    try:
        vector_results = vector_store.similarity_search_with_score(
            query,
            k=v_k,
            filter=effective_filter
        )
    except Exception:
        # Fallback if vectorstore encounters filtering incompatibility on empty collection
        vector_results = []

    # ----------------------------------------------
    # 2. BM25 sparse retrieval
    # ----------------------------------------------
    bm25_results = bm25_search(
        query=query,
        bm25=bm25,
        documents=documents,
        k=b_k,
        metadata_filter=effective_filter
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

    if not rrf_candidates:
        return []

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
    use_reranker=True,
    document_target="latest",
    ipo_id=None
):
    """
    Version-aware and company-isolated retrieval dispatcher supporting:
      - 'latest': (default) queries authoritative filing (is_latest=True)
      - 'drhp': queries DRHP only (document_type='DRHP')
      - 'rhp': queries RHP only (document_type='RHP')
      - 'comparative': dual-partition retrieval across DRHP and RHP independently
    """
    target = (document_target or "latest").lower()

    if target == "comparative":
        half_k = max(2, k // 2)

        # If query is asking what changed generally, enrich query so retrieval targets substantive updates
        q_low = query.lower()
        search_query = query
        if any(k in q_low for k in ["what changed", "difference", "between drhp and rhp", "drhp vs rhp", "compare"]):
            search_query = f"{query} revenue profit PAT financial performance restated statements offer details risk factors"

        # DRHP partition
        drhp_filter = dict(metadata_filter or {})
        if ipo_id:
            drhp_filter["ipo_id"] = ipo_id
        drhp_filter["document_type"] = "DRHP"
        drhp_results = hybrid_search(
            query=search_query,
            top_k=half_k,
            use_reranker=use_reranker,
            metadata_filter=drhp_filter,
            document_target="none",
            ipo_id=ipo_id
        )
        for item in drhp_results:
            item["filing_partition"] = "DRHP"

        # RHP partition
        rhp_filter = dict(metadata_filter or {})
        if ipo_id:
            rhp_filter["ipo_id"] = ipo_id
        rhp_filter["document_type"] = "RHP"
        rhp_results = hybrid_search(
            query=search_query,
            top_k=half_k,
            use_reranker=use_reranker,
            metadata_filter=rhp_filter,
            document_target="none",
            ipo_id=ipo_id
        )
        for item in rhp_results:
            item["filing_partition"] = "RHP"

        return drhp_results + rhp_results

    effective_filter = dict(metadata_filter or {})
    if ipo_id:
        effective_filter["ipo_id"] = ipo_id

    return hybrid_search(
        query=query,
        top_k=k,
        use_reranker=use_reranker,
        metadata_filter=effective_filter,
        document_target=target,
        ipo_id=ipo_id
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