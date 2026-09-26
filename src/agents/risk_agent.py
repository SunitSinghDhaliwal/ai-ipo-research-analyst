import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

# Ensure project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RETRIEVAL_DIR = PROJECT_ROOT / "src" / "retrieval"
GENERATION_DIR = PROJECT_ROOT / "src" / "generation"
AGENTS_DIR = PROJECT_ROOT / "src" / "agents"

for p in [str(PROJECT_ROOT), str(RETRIEVAL_DIR), str(GENERATION_DIR), str(AGENTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from hybrid_retriever import retrieve
from context_builder import build_context, extract_source_metadata
from llm_generator import get_chat_llm
from state import IPOAgentState


# --------------------------------------------------
# Structured Risk Factor Schemas
# --------------------------------------------------

class RiskItem(BaseModel):
    """
    Structured representation of an extracted IPO Risk Factor from the DRHP.
    """
    title: str = Field(description="Title or concise description of the risk factor.")
    category: str = Field(default="Internal Risk", description="Risk category: 'Internal Risk', 'External Risk', or 'General'.")
    causes: List[str] = Field(default_factory=list, description="Underlying drivers or causes stated in the DRHP.")
    consequences: List[str] = Field(default_factory=list, description="Potential business, financial, or operational impacts.")
    quantified_metrics: Optional[List[str]] = Field(default=None, description="Reported percentages, costs, or numbers.")
    source_id: str = Field(description="Source chunk ID.")
    page: Union[int, str] = Field(description="Page number in the DRHP.")


class RiskExtractionResult(BaseModel):
    """
    Structured container for extracted risks.
    """
    risks: List[RiskItem] = Field(default_factory=list, description="List of identified risk factors.")
    summary: str = Field(default="", description="Executive summary of the disclosed risks.")


# --------------------------------------------------
# Specialized Risk Retrieval
# --------------------------------------------------

def risk_retrieve(query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    """
    Specialized retrieval for IPO Risk Factor queries:
      1. Boosts and enriches queries with domain terms (e.g., 'Risk Factors', 'material adverse effect').
      2. Prioritizes chunks originating from SECTION II – RISK FACTORS.
      3. Performs hybrid retrieval (dense + BM25 + BGE cross-encoder reranking).
      4. Merges and deduplicates while prioritizing primary risk disclosures.
    """
    q_lower = query.lower()
    
    # 1. Enriched search query if needed
    enriched_query = query
    if "risk" not in q_lower:
        enriched_query = f"{query} risk factors internal external material adverse effect"

    # 2. Retrieve primary candidates with hybrid search
    candidates = retrieve(
        query=enriched_query,
        k=top_k + 4,
        metadata_filter=None,
        use_reranker=True
    )

    # 3. Sort/prioritize chunks from Section II - Risk Factors or with high relevance
    def risk_priority_score(item: Dict[str, Any]) -> float:
        meta = extract_source_metadata(item)
        score = float(item.get("rerank_score", 0.0))
        sec_path = (meta.get("section_path") or "").upper()
        # Chunks in Section II (Risk Factors) get a natural boost
        if "RISK" in sec_path or "SECTION II" in sec_path:
            score += 2.0
        # Chunks whose text explicitly discusses risk factors
        text = meta.get("text", "").lower()
        if "risk factor" in text or "internal risk" in text or "external risk" in text:
            score += 1.0
        return score

    sorted_candidates = sorted(candidates, key=risk_priority_score, reverse=True)

    # 4. Deduplicate and return top_k
    seen_ids = set()
    deduped = []
    for item in sorted_candidates:
        doc_id = item["document_id"]
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            deduped.append(item)
        if len(deduped) >= top_k:
            break

    return deduped


# --------------------------------------------------
# Risk System Prompt & Templates
# --------------------------------------------------

RISK_AGENT_SYSTEM_PROMPT = """You are a specialized IPO Risk Analyst evaluating the Draft Red Herring Prospectus (DRHP) for Maharashtra Oil Extractions Limited.

Your objective is to provide precise, structured, and strictly grounded assessments of risk factors based ONLY on the retrieved DRHP context provided.

CRITICAL INSTRUCTIONS:
1. STRICT FACTUAL GROUNDING:
   - Answer exclusively using the risk disclosures in the context.
   - Do NOT assume, infer, extrapolate, or invent risks or potential business hazards.

2. STRUCTURED RISK BREAKDOWN:
   For every relevant risk factor identified in the context, present:
   - **Risk Factor**: Clear title or description of the risk as named in the DRHP.
   - **Category**: Internal Risk vs. External Risk (if stated in the text).
   - **Causes & Drivers**: Specific root causes disclosed (e.g., monsoon dependence, commodity price cycles, supplier dependence, lack of long-term contracts).
   - **Potential Impact & Consequences**: Stated consequences on the company's business, financial condition, cash flows, or results of operations.
   - **Reported Figures & Exposure**: Any exact percentages, revenue shares, or numbers mentioned (e.g., top 5 customers contribution, raw material cost share).
   - **Source Citation**: Explicit citation in the format `[<Source ID>, Page <Page Number>]`.

3. AVOID INVENTED SEVERITY OR RANKINGS:
   - Do NOT invent subjective severity ratings (e.g., do not say "Risk Level: High / 9 out of 10" or "This is the most critical risk facing the company") unless the DRHP itself explicitly ranks or labels the risk in that manner.

4. INSUFFICIENT EVIDENCE & HALLUCINATION REFUSAL:
   - If the user asks about a risk that is not disclosed in the provided DRHP context (e.g., cryptocurrency risks, satellite failures, cybersecurity breaches not in the text, or other unmentioned topics), you MUST refuse:
     "Insufficient evidence in the provided DRHP context to answer this question."
   - Explain clearly that no such risk factor is disclosed in the DRHP excerpts provided. Never fabricate risks.

5. MANDATORY CITATIONS & SOURCES TABLE:
   - Every risk detail must include its bracketed source citation: `[<Source ID>, Page <Page Number>]`.
   - Conclude your response with a structured markdown table:
     ### Sources Cited
     | Source ID | Page | Content Type | Section |
"""

RISK_USER_TEMPLATE = """RESEARCH QUERY:
{query}

RETRIEVED DRHP RISK CONTEXT:
{context}

Please provide your rigorous, cited risk factor analysis following the instructions above.
"""


# --------------------------------------------------
# Execution & LangGraph Node
# --------------------------------------------------

def run_risk_agent(
    query: str,
    top_k: int = 6,
    llm: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Executes the specialized Risk Agent:
      1. Risk-prioritized hybrid retrieval.
      2. Context building preserving chunk metadata and section hierarchy.
      3. Grounded risk evaluation using the specialized risk system prompt.
    """
    retrieved_items = risk_retrieve(query=query, top_k=top_k)
    context_str, sources_metadata = build_context(retrieved_items)

    if llm is None:
        llm = get_chat_llm(temperature=0.0)

    user_prompt = RISK_USER_TEMPLATE.format(
        query=query,
        context=context_str
    )

    messages = [
        ("system", RISK_AGENT_SYSTEM_PROMPT),
        ("user", user_prompt)
    ]

    response = llm.invoke(messages)
    answer = response.content.strip()

    return {
        "query": query,
        "answer": answer,
        "context": context_str,
        "sources": sources_metadata,
        "retrieval_results": retrieved_items
    }


def risk_agent_node(
    state: IPOAgentState,
    top_k: int = 6,
    llm: Optional[Any] = None
) -> Dict[str, Any]:
    """
    LangGraph node implementing the specialized Risk Agent.
    Updates the state with risk evaluation findings, context, sources, and cited answer.
    """
    query = state["query"]
    try:
        agent_res = run_risk_agent(query=query, top_k=top_k, llm=llm)
        return {
            "route": "risk",
            "retrieval_results": agent_res["retrieval_results"],
            "context": agent_res["context"],
            "answer": agent_res["answer"],
            "sources": agent_res["sources"],
            "error": None
        }
    except Exception as e:
        return {
            "route": "risk",
            "answer": f"Risk Agent error: {str(e)}",
            "error": str(e)
        }


if __name__ == "__main__":
    test_q = "What are the major risks associated with raw material prices?"
    print(f"\nRunning Risk Agent on: '{test_q}'\n")
    res = run_risk_agent(test_q)
    print("ANSWER:\n", res["answer"])
    print("\nSOURCES:")
    for s in res["sources"]:
        print(f"  - {s['source_id']} (Page {s['page']}, {s['content_type']})")
