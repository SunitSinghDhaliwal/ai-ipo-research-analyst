import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root and src/retrieval are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RETRIEVAL_DIR = PROJECT_ROOT / "src" / "retrieval"
GENERATION_DIR = PROJECT_ROOT / "src" / "generation"

for p in [str(PROJECT_ROOT), str(RETRIEVAL_DIR), str(GENERATION_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Set offline huggingface flags so local embedding & reranker don't trigger network retries
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from hybrid_retriever import retrieve
from context_builder import build_context, extract_source_metadata
from llm_generator import generate_answer, get_chat_llm


def answer_query(
    query: str,
    top_k: int = 5,
    use_reranker: bool = True,
    metadata_filter: Optional[Dict[str, Any]] = None,
    document_target: str = "latest",
    llm: Optional[Any] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    End-to-End Query-to-Answer Pipeline:
      1. Retrieval: Hybrid (Chroma dense + BM25 sparse + RRF) with optional Cross-Encoder reranking
      2. Context Building: Format retrieved chunks preserving chunk_id, page, content_type, text
      3. LLM Generation: Strictly grounded answer with mandatory citations and hallucination refusal
    """
    # 1. Retrieve candidates
    retrieved_items = retrieve(
        query=query,
        k=top_k,
        metadata_filter=metadata_filter,
        use_reranker=use_reranker,
        document_target=document_target
    )

    # Safety handling: RHP requested but missing
    if document_target == "rhp" and not retrieved_items:
        return {
            "query": query,
            "answer": "No RHP evidence is available. The Red Herring Prospectus (RHP) has not been loaded into the knowledge base yet.",
            "context": "No relevant RHP context was retrieved.",
            "sources": [],
            "num_sources": 0
        }

    # Safety handling: Comparative query requested but RHP missing
    if document_target == "comparative":
        has_rhp = any(
            (isinstance(item, dict) and item.get("filing_partition") == "RHP")
            or (hasattr(item, "metadata") and item.metadata.get("document_type") == "RHP")
            for item in retrieved_items
        )
        if not has_rhp:
            context_str, sources_metadata = build_context(retrieved_items, is_comparative=True)
            return {
                "query": query,
                "answer": "A comparison between the DRHP and RHP cannot be performed because RHP evidence is not currently loaded in the knowledge base. Only the Draft Red Herring Prospectus (DRHP) is available.",
                "context": context_str,
                "sources": sources_metadata,
                "num_sources": len(sources_metadata)
            }

    # 2. Build structured context
    context_str, sources_metadata = build_context(
        retrieved_items,
        is_comparative=(document_target == "comparative")
    )

    # 3. Generate answer strictly grounded in context
    answer = generate_answer(
        query=query,
        context=context_str,
        llm=llm,
        provider=provider,
        model_name=model_name,
        temperature=temperature
    )

    return {
        "query": query,
        "answer": answer,
        "context": context_str,
        "sources": sources_metadata,
        "num_sources": len(sources_metadata)
    }


def format_qa_output(result: Dict[str, Any]) -> str:
    """
    Formats the complete pipeline result into a readable report.
    """
    output = []
    output.append("=" * 80)
    output.append(f"RESEARCH QUERY: {result['query']}")
    output.append("=" * 80)
    output.append("\nANSWER:")
    output.append("-" * 80)
    output.append(result["answer"])
    output.append("\n" + "-" * 80)
    output.append(f"RETRIEVED CONTEXT SOURCES ({result['num_sources']} chunks):")
    output.append("-" * 80)
    for idx, s in enumerate(result["sources"], start=1):
        score_str = ""
        if s.get("rerank_score") is not None:
            score_str = f" | Rerank Score: {s['rerank_score']:.4f}"
        elif s.get("rrf_score") is not None:
            score_str = f" | RRF Score: {s['rrf_score']:.4f}"
        output.append(
            f"[{idx}] {s['source_id']} | Page: {s['page']} | Type: {s['content_type']} | Section: {s['section_path']}{score_str}"
        )
    output.append("=" * 80)
    return "\n".join(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IPO Research Agent - Day 4 QA Pipeline")
    parser.add_argument(
        "query",
        nargs="?",
        default="Where is the registered and corporate office of the company located?",
        help="Research query about the IPO DRHP"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking")
    parser.add_argument("--filter-type", choices=["text", "table"], default=None, help="Filter by content_type")
    parser.add_argument("--provider", default=None, help="LLM provider (default: groq)")
    parser.add_argument("--model", default=None, help="LLM model name")

    args = parser.parse_args()

    meta_filter = {"content_type": args.filter_type} if args.filter_type else None

    print(f"\nExecuting pipeline for query: '{args.query}' (top_k={args.top_k}, rerank={not args.no_rerank})...\n")

    res = answer_query(
        query=args.query,
        top_k=args.top_k,
        use_reranker=not args.no_rerank,
        metadata_filter=meta_filter,
        provider=args.provider,
        model_name=args.model
    )

    print(format_qa_output(res))
