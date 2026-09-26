import json
from collections import defaultdict

from bm25_retriever import (
    load_all_documents,
    build_bm25_index,
    bm25_search
)

from hybrid_retriever import hybrid_search

from langchain_chroma import Chroma

from embeddings import get_embedding_model


# --------------------------------------------------
# Configuration
# --------------------------------------------------

VECTORSTORE_DIR = (
    "data/vectorstore/maharashtra_oil_extractions"
)

COLLECTION_NAME = "maharashtra_oil_extractions"

TOP_K = 10


# --------------------------------------------------
# Approved 16-Query Evaluation Dataset
# --------------------------------------------------

EVALUATION_SET = [
    # 1. Risk Factors
    {
        "id": "Q1",
        "category": "Risk Factors",
        "query": "What are the key internal risk factors regarding raw material supply and price volatility?",
        "expected_ids": ["drhp_p30_c1", "drhp_p32_c1"]
    },
    {
        "id": "Q2",
        "category": "Risk Factors",
        "query": "What are the risks related to the geographic concentration of the company's business operations in Maharashtra?",
        "expected_ids": ["drhp_p32_c2", "drhp_p32_c3"]
    },

    # 2. Litigation
    {
        "id": "Q3",
        "category": "Litigation",
        "query": "What outstanding litigation cases and material legal proceedings are pending against the company?",
        "expected_ids": ["drhp_p44_c3", "drhp_p45_c2", "drhp_p476_c3", "drhp_p477_c3"]
    },
    {
        "id": "Q4",
        "category": "Litigation",
        "query": "What are the pending direct tax and indirect tax litigation matters involving the company?",
        "expected_ids": ["drhp_p477_c3", "drhp_p478_c1"]
    },

    # 3. Financial Information
    {
        "id": "Q5",
        "category": "Financial Information",
        "query": "What was the company's revenue from operations and restated profit after tax for the latest fiscal years?",
        "expected_ids": ["drhp_p404_c4", "drhp_p464_c1", "moe_p444_t1", "moe_p86_t1"]
    },
    {
        "id": "Q6",
        "category": "Financial Information",
        "query": "What are the details of the company's total borrowings and financial indebtedness?",
        "expected_ids": ["drhp_p469_c1", "drhp_p470_c1", "moe_p470_t1"]
    },

    # 4. Related-Party Transactions
    {
        "id": "Q7",
        "category": "Related-Party Transactions",
        "query": "What related-party transactions has the company entered into with promoters and group entities?",
        "expected_ids": ["drhp_p71_c3", "drhp_p474_c1", "moe_p474_t1"]
    },
    {
        "id": "Q8",
        "category": "Related-Party Transactions",
        "query": "Who are the promoters and promoter group entities of the company?",
        "expected_ids": ["drhp_p349_c1", "drhp_p349_c2", "moe_p7_t1"]
    },

    # 5. Management
    {
        "id": "Q9",
        "category": "Management",
        "query": "Who are the members of the Board of Directors of the company and their designations?",
        "expected_ids": ["drhp_p330_c1", "drhp_p331_c1", "moe_p330_t1"]
    },
    {
        "id": "Q10",
        "category": "Management",
        "query": "Who are the Key Managerial Personnel (KMP) including the CFO and Company Secretary?",
        "expected_ids": ["drhp_p6_c1", "drhp_p350_c1", "drhp_p350_c2"]
    },

    # 6. IPO / Issue Information
    {
        "id": "Q11",
        "category": "IPO / Issue Information",
        "query": "What are the main objects of the offer and how will the net proceeds be utilized?",
        "expected_ids": ["drhp_p138_c1", "drhp_p138_c2", "moe_p138_t1"]
    },
    {
        "id": "Q12",
        "category": "IPO / Issue Information",
        "query": "What is the breakdown of the Fresh Issue and Offer for Sale in the IPO?",
        "expected_ids": ["drhp_p1_c2", "drhp_p3_c4", "drhp_p111_c1"]
    },

    # 7. Business
    {
        "id": "Q13",
        "category": "Business",
        "query": "What are the primary business activities, products, and solvent extraction operations of the company?",
        "expected_ids": ["drhp_p285_c1", "drhp_p285_c2", "drhp_p438_c4"]
    },
    {
        "id": "Q14",
        "category": "Business",
        "query": "Where are the company's manufacturing facilities located and what are their operational details?",
        "expected_ids": ["drhp_p289_c1", "drhp_p289_c2", "drhp_p153_c3"]
    },

    # 8. General Company Information
    {
        "id": "Q15",
        "category": "General Company Information",
        "query": "Where is the registered and corporate office of the company located?",
        "expected_ids": ["drhp_p101_c1", "moe_p5_t1"]
    },
    {
        "id": "Q16",
        "category": "General Company Information",
        "query": "What is the history of the company's incorporation and name changes?",
        "expected_ids": ["drhp_p3_c2", "drhp_p101_c1"]
    }
]


# --------------------------------------------------
# Load Chroma
# --------------------------------------------------

print("Loading embedding model...")

embedding_model = get_embedding_model()

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=VECTORSTORE_DIR
)


# --------------------------------------------------
# Load BM25
# --------------------------------------------------

print("\nLoading BM25 documents...")

documents = load_all_documents()

print(
    f"Loaded {len(documents)} searchable documents."
)

print("\nBuilding BM25 index...")

bm25 = build_bm25_index(documents)


# --------------------------------------------------
# Helper: get document ID
# --------------------------------------------------

def get_document_id(document):

    if hasattr(document, "metadata"):

        return (
            document.metadata.get("chunk_id")
            or document.metadata.get("table_id")
        )

    return document["id"]


# --------------------------------------------------
# Hit@K
# --------------------------------------------------

def hit_at_k(retrieved_ids, expected_ids, k):

    retrieved_top_k = retrieved_ids[:k]

    return any(
        document_id in expected_ids
        for document_id in retrieved_top_k
    )


# --------------------------------------------------
# Evaluate one query
# --------------------------------------------------

def evaluate_query(query, expected_ids):

    # 1. Chroma
    vector_results = vector_store.similarity_search_with_score(query, k=TOP_K)
    chroma_ids = [get_document_id(doc) for doc, score in vector_results]

    # 2. BM25
    bm25_results = bm25_search(query=query, bm25=bm25, documents=documents, k=TOP_K)
    bm25_ids = [result["document"]["id"] for result in bm25_results]

    # 3. Hybrid RRF (No Reranker)
    hybrid_rrf_results = hybrid_search(
        query=query,
        metadata_filter=None,
        vector_top_k=TOP_K,
        bm25_top_k=TOP_K,
        final_top_k=TOP_K,
        use_reranker=False
    )
    hybrid_rrf_ids = [res["document_id"] for res in hybrid_rrf_results]

    # 4. Hybrid + Reranker
    hybrid_rerank_results = hybrid_search(
        query=query,
        metadata_filter=None,
        vector_top_k=TOP_K,
        bm25_top_k=TOP_K,
        final_top_k=TOP_K,
        use_reranker=True
    )
    hybrid_rerank_ids = [res["document_id"] for res in hybrid_rerank_results]

    results = {
        "chroma": {
            "ids": chroma_ids,
            "hit@5": hit_at_k(chroma_ids, expected_ids, 5),
            "hit@10": hit_at_k(chroma_ids, expected_ids, 10)
        },
        "bm25": {
            "ids": bm25_ids,
            "hit@5": hit_at_k(bm25_ids, expected_ids, 5),
            "hit@10": hit_at_k(bm25_ids, expected_ids, 10)
        },
        "hybrid_rrf": {
            "ids": hybrid_rrf_ids,
            "hit@5": hit_at_k(hybrid_rrf_ids, expected_ids, 5),
            "hit@10": hit_at_k(hybrid_rrf_ids, expected_ids, 10)
        },
        "hybrid_rerank": {
            "ids": hybrid_rerank_ids,
            "hit@5": hit_at_k(hybrid_rerank_ids, expected_ids, 5),
            "hit@10": hit_at_k(hybrid_rerank_ids, expected_ids, 10)
        }
    }

    return results


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION BENCHMARK (16 QUERIES)")
    print("=" * 80)

    methods = ["chroma", "bm25", "hybrid_rrf", "hybrid_rerank"]
    total_queries = len(EVALUATION_SET)

    overall_summary = {m: {"hit@5": 0, "hit@10": 0} for m in methods}
    category_summary = defaultdict(lambda: {m: {"hit@5": 0, "hit@10": 0, "total": 0} for m in methods})

    per_query_details = []

    for item in EVALUATION_SET:
        q_id = item["id"]
        category = item["category"]
        query = item["query"]
        expected_ids = item["expected_ids"]

        eval_res = evaluate_query(query, expected_ids)

        query_record = {
            "id": q_id,
            "category": category,
            "query": query,
            "expected_ids": expected_ids,
            "methods": eval_res
        }
        per_query_details.append(query_record)

        for m in methods:
            hit5 = eval_res[m]["hit@5"]
            hit10 = eval_res[m]["hit@10"]

            if hit5:
                overall_summary[m]["hit@5"] += 1
                category_summary[category][m]["hit@5"] += 1

            if hit10:
                overall_summary[m]["hit@10"] += 1
                category_summary[category][m]["hit@10"] += 1

            category_summary[category][m]["total"] += 1

    # Print Detailed Per-Query Breakdown
    print("\n" + "=" * 80)
    print("PER-QUERY BREAKDOWN")
    print("=" * 80)

    for rec in per_query_details:
        print(f"\n[{rec['id']}] Category: {rec['category']}")
        print(f"Query: {rec['query']}")
        print(f"Relevant Doc IDs: {rec['expected_ids']}")
        
        for m in methods:
            m_res = rec["methods"][m]
            top5 = m_res["ids"][:5]
            top10 = m_res["ids"][:10]
            print(f"  - {m.upper()}:")
            print(f"      Hit@5: {'YES' if m_res['hit@5'] else 'NO'} | Hit@10: {'YES' if m_res['hit@10'] else 'NO'}")
            print(f"      Top-5 IDs : {top5}")
            print(f"      Top-10 IDs: {top10}")

    # Print Final Summary Table
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY RESULTS")
    print("=" * 80)

    print(f"{'METHOD':<15} | {'HIT@5 COUNT':<12} | {'HIT@5 %':<10} | {'HIT@10 COUNT':<12} | {'HIT@10 %':<10}")
    print("-" * 75)
    for m in methods:
        h5 = overall_summary[m]["hit@5"]
        h10 = overall_summary[m]["hit@10"]
        h5_pct = (h5 / total_queries) * 100
        h10_pct = (h10 / total_queries) * 100
        print(f"{m.upper():<15} | {h5}/{total_queries:<10} | {h5_pct:>8.1f}% | {h10}/{total_queries:<10} | {h10_pct:>8.1f}%")

    # Print Category Summary Table
    print("\n" + "=" * 80)
    print("CATEGORY BREAKDOWN (HIT@10)")
    print("=" * 80)

    for cat, m_dict in category_summary.items():
        total_cat = m_dict["chroma"]["total"]
        print(f"\nCategory: {cat} (Total Queries: {total_cat})")
        for m in methods:
            c_h5 = m_dict[m]["hit@5"]
            c_h10 = m_dict[m]["hit@10"]
            print(f"  - {m.upper():<15}: Hit@5 = {c_h5}/{total_cat} | Hit@10 = {c_h10}/{total_cat}")