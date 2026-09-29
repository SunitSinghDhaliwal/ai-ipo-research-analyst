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

def risk_retrieve(
    query: str,
    top_k: int = 6,
    document_target: str = "latest",
    ipo_id: Optional[str] = None
) -> List[Dict[str, Any]]:
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

    base_filter = {"ipo_id": ipo_id} if ipo_id else None
    # 2. Retrieve primary candidates with hybrid search
    candidates = retrieve(
        query=enriched_query,
        k=top_k + 4,
        metadata_filter=base_filter,
        use_reranker=True,
        document_target=document_target,
        ipo_id=ipo_id
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

RISK_AGENT_SYSTEM_PROMPT = """You are a specialized IPO Risk Analyst evaluating IPO prospectus filings (DRHP / RHP) for the subject company.

Your objective is to provide an executive, highly readable, and strictly grounded assessment of key risk factors based ONLY on the retrieved prospectus context provided.

CRITICAL PRESENTATION RULES (MINIMAL CLUTTER, NO EXCESSIVE HASHTAGS):
- DO NOT use excessive markdown hashtags (avoid ###, ####) and DO NOT format risks into cramped 7-column spreadsheet tables.
- Present each distinct risk factor using bold section headers:

**1. [Risk Factor Title / Descriptor]**
- **Category**: Internal Risk / External Risk (as stated in prospectus)
- **Underlying Causes & Drivers**: [Direct factual root causes disclosed in the text]
- **Potential Business & Financial Impact**: [Specific consequences on operations, cash flow, or regulatory standing]
- **Disclosed Exposure / Metrics**: [Any reported metrics, percentages, revenue concentration, or debt figures]
- **Filing Citation**: `[Source: <Source ID>, Page <Page Number>]`

CRITICAL GROUNDING RULES:
1. STRICT FACTUAL GROUNDING:
   - Use exclusively the risk disclosures in the context. Never infer or invent risks.
2. AVOID INVENTED SEVERITY OR RANKINGS:
   - Do NOT invent subjective rankings ("High / 9 out of 10") unless explicitly stated in the filing.
3. INSUFFICIENT EVIDENCE & HALLUCINATION REFUSAL:
   - If the user asks about a risk not disclosed in the text, clearly state:
     "Insufficient evidence in the provided prospectus context to answer this question."
4. MANDATORY CITATIONS:
   - Every risk detail must cite its source: `[Source: <Source ID>, Page <Page Number>]`.
   - Do NOT output a raw text table of sources at the end (the UI already renders interactive source cards).
"""

RISK_USER_TEMPLATE = """RESEARCH QUERY:
{query}

RETRIEVED PROSPECTUS RISK CONTEXT:
{context}

Please provide your rigorous, cited risk factor analysis following the instructions above.
"""


# --------------------------------------------------
# Execution & LangGraph Node
# --------------------------------------------------

def run_risk_agent(
    query: str,
    top_k: int = 6,
    llm: Optional[Any] = None,
    document_target: str = "latest",
    ipo_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the specialized Risk Agent:
      1. Risk-prioritized hybrid retrieval with document scope filtering and IPO isolation.
      2. Context building preserving chunk metadata and section hierarchy.
      3. Grounded risk evaluation using the specialized risk system prompt.
    """
    retrieved_items = risk_retrieve(
        query=query,
        top_k=top_k,
        document_target=document_target,
        ipo_id=ipo_id
    )

    # Safe handling: If query specifically asks for RHP and no RHP evidence is loaded
    if document_target == "rhp" and not retrieved_items:
        return {
            "query": query,
            "answer": "No RHP evidence is available. The Red Herring Prospectus (RHP) has not been loaded into the knowledge base yet.",
            "context": "No relevant RHP context was retrieved.",
            "sources": [],
            "retrieval_results": []
        }

    # Safe handling: If query asks for comparative DRHP vs RHP analysis and RHP is missing
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
                "answer": "A comparison between DRHP and RHP risk factors cannot be performed because RHP evidence is not currently loaded in the knowledge base. Only the Draft Red Herring Prospectus (DRHP) is available.",
                "context": context_str,
                "sources": sources_metadata,
                "retrieval_results": retrieved_items
            }

    context_str, sources_metadata = build_context(
        retrieved_items,
        is_comparative=(document_target == "comparative")
    )

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

    try:
        response = llm.invoke(messages)
        answer = response.content.strip()
    except Exception as e:
        answer = f"Risk factor analysis based on retrieved evidence:\n\n{context_str[:600]}...\n\n(Note: LLM generation encountered {e})"

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
    document_target = state.get("document_target", "latest")
    ipo_id = state.get("ipo_id")
    try:
        agent_res = run_risk_agent(
            query=query,
            top_k=top_k,
            llm=llm,
            document_target=document_target,
            ipo_id=ipo_id
        )
        return {
            "ipo_id": ipo_id,
            "route": "risk",
            "document_target": document_target,
            "retrieval_results": agent_res["retrieval_results"],
            "context": agent_res["context"],
            "answer": agent_res["answer"],
            "sources": agent_res["sources"],
            "error": None
        }
    except Exception as e:
        return {
            "ipo_id": ipo_id,
            "route": "risk",
            "document_target": document_target,
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
