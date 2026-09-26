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
# Structured Litigation Schemas
# --------------------------------------------------

class LitigationCase(BaseModel):
    """
    Structured representation of legal proceedings disclosed in the DRHP.
    """
    entity: str = Field(description="Entity involved: 'Company', 'Promoter', 'Director', 'Subsidiary', or 'KMP'.")
    nature_of_action: str = Field(description="'Against' or 'By' the entity.")
    proceeding_type: str = Field(description="Type: 'Tax', 'Criminal', 'Statutory/Regulatory', or 'Material Civil'.")
    number_of_cases: int = Field(default=0, description="Total count of proceedings.")
    aggregate_amount_million: Optional[float] = Field(default=None, description="Aggregate amount involved in ₹ million.")
    brief_description: Optional[str] = Field(default=None, description="Details or forum if specified.")
    source_id: str = Field(description="Source chunk ID or table ID (e.g. 'moe_p44_t2', 'drhp_p476_c1').")
    page: Union[int, str] = Field(description="Page number in the DRHP.")


class LitigationSummary(BaseModel):
    """
    Structured summary of all outstanding litigation and regulatory proceedings.
    """
    cases: List[LitigationCase] = Field(default_factory=list, description="List of structured litigation items.")
    sebi_disciplinary_actions_last_5_years: bool = Field(default=False, description="Whether SEBI/Stock Exchanges took actions against Promoters in last 5 fiscals.")
    summary: str = Field(default="", description="High-level legal and regulatory risk assessment.")


# --------------------------------------------------
# Specialized Litigation Retrieval
# --------------------------------------------------

def litigation_retrieve(query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    """
    Specialized retrieval for IPO Legal and Litigation queries:
      1. Prioritizes the authoritative SEBI ICDR litigation summary tables (moe_p44_t2, moe_p45_t1).
      2. Retrieves detailed proceedings from SECTION VI – OUTSTANDING LITIGATION AND MATERIAL DEVELOPMENTS (pages 476–481).
      3. Performs hybrid retrieval (dense + BM25 + BGE cross-encoder).
      4. Merges and deduplicates while guaranteeing top placement for core litigation tables and disclosures.
    """
    q_lower = query.lower()

    # 1. Enriched query to pull both summary tables and full section disclosures
    enriched_query = f"{query} outstanding litigation criminal tax statutory regulatory proceedings SEBI company promoters directors"

    # 2. Table-specific retrieval for official SEBI ICDR litigation tables
    table_candidates = retrieve(
        query="Name of Individual Entity Criminal Tax Statutory Regulatory Proceedings SEBI stock exchanges Promoters Company",
        k=4,
        metadata_filter={"content_type": "table"},
        use_reranker=True
    )

    # 3. Hybrid candidates for full text disclosures
    hybrid_candidates = retrieve(
        query=enriched_query,
        k=top_k + 4,
        metadata_filter=None,
        use_reranker=True
    )

    # 4. Priority scoring
    def litigation_priority_score(item: Dict[str, Any]) -> float:
        meta = extract_source_metadata(item)
        score = float(item.get("rerank_score", 0.0))
        tid = meta.get("source_id", "")
        # Primary SEBI ICDR summary tables
        if tid in ["moe_p44_t2", "moe_p45_t1"]:
            score += 5.0
        sec_path = (meta.get("section_path") or "").upper()
        if "OUTSTANDING LITIGATION" in sec_path or "LEGAL AND OTHER" in sec_path:
            score += 3.0
        text = meta.get("text", "").lower()
        if "criminal proceedings" in text or "tax proceedings" in text or "statutory or regulatory" in text:
            score += 1.5
        return score

    # Combine candidates
    all_candidates = table_candidates + hybrid_candidates
    sorted_candidates = sorted(all_candidates, key=litigation_priority_score, reverse=True)

    # 5. Deduplicate preserving priority
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
# Litigation System Prompt & Templates
# --------------------------------------------------

LITIGATION_AGENT_SYSTEM_PROMPT = """You are a specialized IPO Legal & Litigation Analyst evaluating the Draft Red Herring Prospectus (DRHP) for Maharashtra Oil Extractions Limited.

Your objective is to provide precise, structured, and strictly grounded assessments of outstanding litigation, material legal proceedings, tax disputes, and regulatory actions based ONLY on the retrieved DRHP context provided.

CRITICAL INSTRUCTIONS:
1. STRICT FACTUAL GROUNDING:
   - Answer exclusively using the exact legal records, case numbers, statutory summaries, and tables in the context.
   - Do NOT assume, infer, speculate, or invent lawsuits, claims, or regulatory penalties.

2. CRITICAL ENTITY DISTINCTION (MANDATORY):
   You MUST explicitly distinguish between proceedings involving:
   a) **The Company**:
      - Distinguish proceedings filed *Against* the Company vs. *By* the Company.
      - Disclose the number of criminal, tax, statutory/regulatory, and material civil proceedings, along with the aggregate amount involved.
   b) **The Promoters**:
      - Note proceedings filed *Against* Promoters vs. *By* Promoters.
   c) **Directors (other than Promoters)**:
      - Detail proceedings filed *Against* vs. *By* Directors.
   d) **Subsidiary & Group Companies & KMPs**:
      - Confirm if any proceedings are pending against Subsidiary, Group Companies, or Key Managerial Personnel.

3. REGULATORY & SEBI ACTIONS:
   - When asked about SEBI or regulatory proceedings, explicitly check the official SEBI ICDR table disclosure.
   - For example, if disciplinary actions by SEBI or stock exchanges against Promoters in the last five Fiscals are "Nil", state this unequivocally with citation.

4. STRUCTURED LEGAL BREAKDOWN:
   Where multiple proceedings exist, report clearly:
   - **Party / Entity**: [Company / Promoter / Director / Subsidiary]
   - **Nature**: [Against or By the Entity]
   - **Category**: [Tax / Criminal / Statutory-Regulatory / Civil]
   - **Number of Cases**: [Count or Nil]
   - **Aggregate Amount Involved**: [₹ in million, or Nil/NA]
   - **Details / Forums**: [As described in the DRHP]
   - **Citation**: `[<Source ID>, Page <Page Number>]`

5. INSUFFICIENT EVIDENCE & HALLUCINATION REFUSAL:
   - If the user asks about a legal proceeding, enforcement action, or dispute not disclosed in the provided DRHP context (e.g. Enforcement Directorate / ED probes, foreign litigation, antitrust suits, or fabricated disputes), you MUST state clearly:
     "Insufficient evidence in the provided DRHP context to answer this question."
   - State clearly that no such litigation or regulatory action is disclosed in the provided DRHP excerpts.

6. MANDATORY CITATIONS & SOURCES TABLE:
   - Every single legal figure, case count, and claim must be cited: `[<Source ID>, Page <Page Number>]`.
   - Conclude your answer with a structured markdown table:
     ### Sources Cited
     | Source ID | Page | Content Type | Section |
"""

LITIGATION_USER_TEMPLATE = """RESEARCH QUERY:
{query}

RETRIEVED DRHP LITIGATION CONTEXT:
{context}

Please provide your rigorous, cited legal and litigation analysis following the instructions above.
"""


# --------------------------------------------------
# Execution & LangGraph Node
# --------------------------------------------------

def run_litigation_agent(
    query: str,
    top_k: int = 6,
    llm: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Executes the specialized Litigation Agent:
      1. Litigation-prioritized hybrid retrieval (summary tables + section disclosures).
      2. Structured context generation preserving legal tables and citations.
      3. Grounded legal evaluation with mandatory entity distinctions.
    """
    retrieved_items = litigation_retrieve(query=query, top_k=top_k)
    context_str, sources_metadata = build_context(retrieved_items)

    if llm is None:
        llm = get_chat_llm(temperature=0.0)

    user_prompt = LITIGATION_USER_TEMPLATE.format(
        query=query,
        context=context_str
    )

    messages = [
        ("system", LITIGATION_AGENT_SYSTEM_PROMPT),
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


def litigation_agent_node(
    state: IPOAgentState,
    top_k: int = 6,
    llm: Optional[Any] = None
) -> Dict[str, Any]:
    """
    LangGraph node implementing the specialized Litigation Agent.
    Updates the state with litigation evaluation findings, context, sources, and cited answer.
    """
    query = state["query"]
    try:
        agent_res = run_litigation_agent(query=query, top_k=top_k, llm=llm)
        return {
            "route": "litigation",
            "retrieval_results": agent_res["retrieval_results"],
            "context": agent_res["context"],
            "answer": agent_res["answer"],
            "sources": agent_res["sources"],
            "error": None
        }
    except Exception as e:
        return {
            "route": "litigation",
            "answer": f"Litigation Agent error: {str(e)}",
            "error": str(e)
        }


if __name__ == "__main__":
    test_q = "What outstanding litigation cases and material legal proceedings are pending?"
    print(f"\nRunning Litigation Agent on: '{test_q}'\n")
    res = run_litigation_agent(test_q)
    print("ANSWER:\n", res["answer"])
    print("\nSOURCES:")
    for s in res["sources"]:
        print(f"  - {s['source_id']} (Page {s['page']}, {s['content_type']})")
