from typing import Any, Dict, List, Tuple, Union
from langchain_core.documents import Document


def extract_source_metadata(item: Union[Dict[str, Any], Document]) -> Dict[str, Any]:
    """
    Extracts and standardizes chunk metadata and text from various document formats:
      - Hybrid retriever result dicts: {"document_id": ..., "document": ..., "rrf_score": ..., ...}
      - LangChain Document objects
      - Plain dictionary records

    Preserves:
      - chunk_id / table_id
      - page number
      - company / document_type
      - content_type ('text' or 'table')
      - section_path / section / chapter
      - source text
      - scores (rrf_score, rerank_score)
    """
    raw_doc = item
    rrf_score = None
    rerank_score = None

    if isinstance(item, dict):
        if "rrf_score" in item:
            rrf_score = item.get("rrf_score")
        if "rerank_score" in item:
            rerank_score = item.get("rerank_score")
        if "document" in item:
            raw_doc = item["document"]

    # Extract metadata and page content
    if hasattr(raw_doc, "metadata") and hasattr(raw_doc, "page_content"):
        meta = raw_doc.metadata or {}
        text = raw_doc.page_content
    elif isinstance(raw_doc, dict):
        meta = raw_doc.get("metadata", {}) or raw_doc
        text = raw_doc.get("text", "") or raw_doc.get("page_content", "")
    else:
        meta = {}
        text = str(raw_doc)

    source_id = (
        meta.get("chunk_id")
        or meta.get("table_id")
        or (item.get("document_id") if isinstance(item, dict) else None)
        or meta.get("id")
        or "unknown_id"
    )

    page = meta.get("page", "N/A")
    company = meta.get("company", "Maharashtra Oil Extractions Limited")
    doc_type = meta.get("document_type", "DRHP")
    content_type = meta.get("content_type", "text")
    section_path = meta.get("section_path") or meta.get("section") or "GENERAL"

    return {
        "source_id": str(source_id),
        "page": page,
        "company": company,
        "document_type": doc_type,
        "content_type": content_type,
        "section_path": section_path,
        "text": text.strip(),
        "rrf_score": rrf_score,
        "rerank_score": rerank_score,
    }


def format_single_source(source_meta: Dict[str, Any], index: int) -> str:
    """
    Format a single retrieved chunk into a cleanly structured context block.
    """
    header = (
        f"--- [SOURCE {index}] ---\n"
        f"Source ID: {source_meta['source_id']}\n"
        f"Document: {source_meta['document_type']} ({source_meta['company']})\n"
        f"Page: {source_meta['page']}\n"
        f"Content Type: {source_meta['content_type']}\n"
        f"Section: {source_meta['section_path']}\n"
    )

    if source_meta.get("rerank_score") is not None:
        header += f"Relevance Score: {source_meta['rerank_score']:.4f}\n"

    content_header = "Extracted Table:" if source_meta["content_type"] == "table" else "Extracted Text:"
    body = f"\n{content_header}\n{source_meta['text']}\n"
    return header + body


def build_context(
    retrieved_items: List[Union[Dict[str, Any], Document]],
    max_tokens: int = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Builds a unified, structured prompt context string from retrieved chunks,
    preserving chunk_id, page number, document/company, content_type, and source text.

    Returns:
      (context_string, list_of_source_metadata_dicts)
    """
    if not retrieved_items:
        return "No relevant DRHP context was retrieved.", []

    formatted_blocks = []
    sources_metadata = []

    for idx, item in enumerate(retrieved_items, start=1):
        source_meta = extract_source_metadata(item)
        sources_metadata.append(source_meta)
        block = format_single_source(source_meta, index=idx)
        formatted_blocks.append(block)

    full_context = "\n".join(formatted_blocks)
    return full_context, sources_metadata


if __name__ == "__main__":
    # Self-test with mock data
    mock_chunk = {
        "document_id": "drhp_p30_c1",
        "document": {
            "text": "Our business is subject to raw material availability and price fluctuations.",
            "metadata": {
                "chunk_id": "drhp_p30_c1",
                "page": 30,
                "company": "Maharashtra Oil Extractions Limited",
                "document_type": "DRHP",
                "content_type": "text",
                "section_path": "SECTION II – RISK FACTORS > INTERNAL RISK FACTORS"
            }
        },
        "rrf_score": 0.0163,
        "rerank_score": 0.895
    }

    ctx, sources = build_context([mock_chunk])
    print("Test Context Output:\n")
    print(ctx)
    print("\nMetadata output:", sources[0]["source_id"], "Page:", sources[0]["page"])
