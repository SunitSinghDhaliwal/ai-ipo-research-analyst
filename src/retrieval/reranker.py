from sentence_transformers import CrossEncoder


MODEL_NAME = "BAAI/bge-reranker-base"


def get_reranker():
    """
    Load the local reranking model.
    """

    model = CrossEncoder(
        MODEL_NAME,
        device="cpu"
    )

    return model


def _extract_text(document):
    if hasattr(document, "page_content"):
        return document.page_content
    elif isinstance(document, dict):
        if "text" in document:
            return document["text"]
        elif "document" in document:
            return _extract_text(document["document"])
    raise ValueError(f"Unable to extract text from document structure: {document}")


def rerank(query, documents, top_k=5):
    """
    Rerank retrieved documents using a cross-encoder.
    """
    if not documents:
        return []

    model = get_reranker()

    pairs = [
        (query, _extract_text(doc))
        for doc in documents
    ]

    scores = model.predict(pairs)

    ranked_results = sorted(
        zip(documents, scores),
        key=lambda x: float(x[1]),
        reverse=True
    )

    return ranked_results[:top_k]