import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project paths are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RETRIEVAL_DIR = PROJECT_ROOT / "src" / "retrieval"
GENERATION_DIR = PROJECT_ROOT / "src" / "generation"
AGENTS_DIR = PROJECT_ROOT / "src" / "agents"

for p in [str(PROJECT_ROOT), str(RETRIEVAL_DIR), str(GENERATION_DIR), str(AGENTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from langgraph.graph import StateGraph, START, END
from state import IPOAgentState
from router import router_node
from hybrid_retriever import retrieve
from context_builder import build_context
from llm_generator import generate_answer, get_chat_llm


# --------------------------------------------------
# Route Dispatcher Nodes (General / Ambiguous / Multi-intent)
# --------------------------------------------------


def general_route_node(state: IPOAgentState) -> Dict[str, Any]:
    """Route handler for general company queries (office location, history, management)."""
    return {"route": "general"}


def ambiguous_route_node(state: IPOAgentState) -> Dict[str, Any]:
    """Route handler for ambiguous or underspecified queries."""
    return {"route": "ambiguous"}


def multi_intent_route_node(state: IPOAgentState) -> Dict[str, Any]:
    """Route handler for multi-intent cross-cutting queries."""
    return {"route": "multi_intent"}


def select_route_edge(state: IPOAgentState) -> str:
    """
    Conditional edge function routing from router to category route nodes.
    """
    route = state.get("route", "general")
    valid_routes = {
        "financial",
        "risk",
        "litigation",
        "related_party",
        "summary",
        "general",
        "ambiguous",
        "multi_intent"
    }
    return route if route in valid_routes else "general"


# --------------------------------------------------
# Shared RAG Pipeline Nodes
# --------------------------------------------------

def retrieval_node(state: IPOAgentState, top_k: int = 5, use_reranker: bool = True) -> Dict[str, Any]:
    """
    Shared retrieval node invoking the hybrid retriever with version-aware document scoping and IPO isolation.
    """
    query = state["query"]
    document_target = state.get("document_target", "latest")
    ipo_id = state.get("ipo_id")
    try:
        retrieved_items = retrieve(
            query=query,
            k=top_k,
            use_reranker=use_reranker,
            document_target=document_target,
            ipo_id=ipo_id
        )
        return {
            "retrieval_results": retrieved_items,
            "error": None
        }
    except Exception as e:
        return {
            "retrieval_results": [],
            "error": f"Retrieval failed: {str(e)}"
        }


def context_builder_node(state: IPOAgentState) -> Dict[str, Any]:
    """
    Shared context builder node invoking src/generation/context_builder.py.
    Preserves chunk_id, page, filing, content_type, and text.
    """
    results = state.get("retrieval_results", [])
    document_target = state.get("document_target", "latest")
    is_comparative = (document_target == "comparative") or state.get("filing_comparison", False)
    context_str, sources_metadata = build_context(results, is_comparative=is_comparative)
    return {
        "context": context_str,
        "sources": sources_metadata
    }


def generation_node(state: IPOAgentState, llm: Optional[Any] = None) -> Dict[str, Any]:
    """
    Shared generation node invoking src/generation/llm_generator.py.
    Ensures strict grounding, hallucination refusal, and page citations.
    """
    query = state["query"]
    context_str = state.get("context", "")
    document_target = state.get("document_target", "latest")
    retrieval_results = state.get("retrieval_results", [])

    if document_target == "rhp" and not retrieval_results:
        return {
            "answer": "No RHP evidence is available. The Red Herring Prospectus (RHP) has not been loaded into the knowledge base yet."
        }

    if document_target == "comparative":
        has_rhp = any(
            (isinstance(item, dict) and item.get("filing_partition") == "RHP")
            or (hasattr(item, "metadata") and item.metadata.get("document_type") == "RHP")
            for item in retrieval_results
        )
        if not has_rhp:
            return {
                "answer": "A comparison between the DRHP and RHP cannot be performed because RHP evidence is not currently loaded in the knowledge base. Only the Draft Red Herring Prospectus (DRHP) is available."
            }

    try:
        answer = generate_answer(query=query, context=context_str, llm=llm)
        return {"answer": answer}
    except Exception as e:
        return {
            "answer": f"Generation error: {str(e)}",
            "error": str(e)
        }


SUMMARY_SYSTEM_PROMPT = """You are a rigorous, highly disciplined senior IPO Research Analyst.
Using ONLY the grounded specialist evidence provided below, produce a concise, structured research summary for the IPO.

REQUIRED OUTPUT STRUCTURE:
# IPO Research Summary: {company_name}

## Company Overview
[Brief overview of the business model, legal background, and core operations based strictly on the evidence]

## Financial Snapshot
* Revenue from Operations: [Disclosed metric with source citation]
* Growth / Trend: [Calculated or disclosed trend with source citation]
* Key Financial Observations: [Any other supported margins/debt figures with source citations]

## Key Risks
* [Risk 1 with causes and source citation]
* [Risk 2 with causes and source citation]
* [Risk 3 with causes and source citation]

## Litigation
[Summary of outstanding proceedings, tax claims, and contingent liabilities with source citations]

## Related Parties
[Summary of transactions, promoter group entities, and loans/advances with source citations]

## DRHP → RHP Changes
[Summarize material updates between DRHP and RHP if comparative evidence is provided (e.g. for HDBFS, summarize PAT updates, full-year numbers, or NPA metrics). If only DRHP is available (e.g. MOEL) or if RHP is not loaded, explicitly state: "No RHP filing available in the knowledge base; comparison not applicable."]

## Key Items Requiring Attention
[Top 2-3 critical risks or exposure items from the evidence requiring investor diligence]

CRITICAL RULES:
- STRICT FACTUAL GROUNDING: Use ONLY the supplied evidence. Do NOT invent facts or financial values.
- NO HALLUCINATED RHP: If RHP evidence is not loaded or missing, explicitly state that no RHP is available.
- MANDATORY INLINE CITATIONS: Every claim, metric, or risk must cite its source (e.g., `[hdbfs_rhp_p80_c1, Page 80]`, `[drhp_p31_c4, Page 31]`).
- NO INVESTMENT RECOMMENDATIONS: Do NOT provide Buy/Sell recommendations, do not score or rank the IPO, and do not make speculative investment conclusions.
- SOURCES SUMMARY: Conclude with a clean markdown table of cited sources.
"""

SUMMARY_PROMPT_TEMPLATE = """RESEARCH QUERY:
{query}

COMPANY: {company_name} (IPO ID: {ipo_id})

GROUNDED SPECIALIST EVIDENCE:
{evidence}

Please produce the structured IPO Research Summary following the instructions above.
"""


def summary_node(state: IPOAgentState, llm: Optional[Any] = None) -> Dict[str, Any]:
    """
    Day 9 Executive IPO Summary:
    Collects grounded evidence from the existing 4 specialist agents
    (Financial, Risk, Litigation, Related Party) and synthesizes a structured
    executive report using ONE summary prompt with zero duplicate retrieval.
    """
    from financial_agent import run_financial_agent
    from risk_agent import run_risk_agent
    from litigation_agent import run_litigation_agent
    from related_party_agent import run_related_party_agent

    query = state["query"]
    ipo_id = state.get("ipo_id") or "moel_ipo"
    doc_target = state.get("document_target", "latest")
    company_name = "HDB Financial Services Limited" if ipo_id == "hdbfs_ipo" else "Maharashtra Oil Extractions Limited"

    # 1. Collect outputs from existing specialist agents with zero duplicate retrieval
    fin_res = run_financial_agent(f"What are the financial metrics and revenue of {company_name}?", top_k=3, document_target=doc_target, ipo_id=ipo_id, llm=llm)
    risk_res = run_risk_agent(f"What are the primary risk factors for {company_name}?", top_k=3, document_target=doc_target, ipo_id=ipo_id, llm=llm)
    lit_res = run_litigation_agent(f"What outstanding litigation is disclosed for {company_name}?", top_k=3, document_target=doc_target, ipo_id=ipo_id, llm=llm)
    rpt_res = run_related_party_agent(f"What related party transactions are disclosed for {company_name}?", top_k=3, document_target=doc_target, ipo_id=ipo_id, llm=llm)

    # When multiple filings are available (HDBFS), collect comparative evidence from existing financial agent
    comp_evidence = ""
    comp_sources = []
    if ipo_id == "hdbfs_ipo":
        comp_res = run_financial_agent(f"Compare financial metrics and updates between DRHP and RHP for {company_name}", top_k=3, document_target="comparative", ipo_id=ipo_id, llm=llm)
        comp_evidence = f"\n=== DRHP vs RHP COMPARATIVE EVIDENCE ===\n{comp_res.get('answer', '')}\n"
        comp_sources = comp_res.get("sources", [])

    # Combine unique sources from specialist agents
    all_sources = []
    seen_sources = set()
    for res in [fin_res, risk_res, lit_res, rpt_res]:
        for s in res.get("sources", []):
            sid = s.get("source_id")
            if sid and sid not in seen_sources:
                seen_sources.add(sid)
                all_sources.append(s)
    for s in comp_sources:
        sid = s.get("source_id")
        if sid and sid not in seen_sources:
            seen_sources.add(sid)
            all_sources.append(s)

    # 2. Compile grounded evidence for the single summary prompt
    summary_context = (
        f"COMPANY: {company_name} (ID: {ipo_id})\n\n"
        f"=== FINANCIAL EVIDENCE & SNAPSHOT ===\n{fin_res.get('answer', '')}\n\n"
        f"=== KEY RISK FACTORS EVIDENCE ===\n{risk_res.get('answer', '')}\n\n"
        f"=== LITIGATION & LEGAL PROCEEDINGS EVIDENCE ===\n{lit_res.get('answer', '')}\n\n"
        f"=== RELATED PARTY TRANSACTIONS EVIDENCE ===\n{rpt_res.get('answer', '')}\n"
        f"{comp_evidence}"
    )

    sys_prompt = SUMMARY_SYSTEM_PROMPT.format(company_name=company_name)
    user_prompt = SUMMARY_PROMPT_TEMPLATE.format(
        query=query,
        company_name=company_name,
        ipo_id=ipo_id,
        evidence=summary_context
    )

    if llm is None:
        llm = get_chat_llm(temperature=0.0)

    messages = [
        ("system", sys_prompt),
        ("user", user_prompt)
    ]

    try:
        response = llm.invoke(messages)
        answer = response.content.strip()
    except Exception as e:
        # Fallback to structured compilation of specialist sections if external LLM unavailable
        answer = (
            f"# IPO Research Summary: {company_name}\n\n"
            f"## Company Overview\nAnalysis synthesized for {company_name} based on retrieved prospectus filings.\n\n"
            f"## Financial Snapshot\n{fin_res.get('answer', 'No financial data available.')}\n\n"
            f"## Key Risks\n{risk_res.get('answer', 'No risk data available.')}\n\n"
            f"## Litigation\n{lit_res.get('answer', 'No litigation data available.')}\n\n"
            f"## Related Parties\n{rpt_res.get('answer', 'No related party data available.')}\n\n"
            f"## DRHP → RHP Changes\n{'Material updates analyzed across filings.' if ipo_id == 'hdbfs_ipo' else 'No RHP filing available in the knowledge base; comparison not applicable.'}\n\n"
            f"## Key Items Requiring Attention\nKey risk and financial exposure items disclosed in the sections above require standard investor diligence.\n\n"
            f"*(Note: Direct LLM synthesis fallback applied: {e})*"
        )

    return {
        "ipo_id": ipo_id,
        "route": "summary",
        "document_target": doc_target,
        "context": summary_context,
        "answer": answer,
        "sources": all_sources,
        "error": None
    }


# --------------------------------------------------
# LangGraph Workflow Construction
# --------------------------------------------------

def build_ipo_graph(top_k: int = 5, use_reranker: bool = True, llm: Optional[Any] = None):
    """
    Constructs the compiled LangGraph workflow:
      START -> router -> [conditional routing] -> category_route -> retrieval -> context_builder -> generation -> END
    """
    workflow = StateGraph(IPOAgentState)

    from financial_agent import financial_agent_node
    from risk_agent import risk_agent_node
    from litigation_agent import litigation_agent_node
    from related_party_agent import related_party_agent_node

    # 1. Add Router Node
    workflow.add_node("router", router_node)

    # 2. Add Route Dispatcher Nodes (Specialized agents for Day 5; summary for Day 9; others generic)
    workflow.add_node("financial", lambda state: financial_agent_node(state, top_k=top_k, llm=llm))
    workflow.add_node("risk", lambda state: risk_agent_node(state, top_k=top_k, llm=llm))
    workflow.add_node("litigation", lambda state: litigation_agent_node(state, top_k=top_k, llm=llm))
    workflow.add_node("related_party", lambda state: related_party_agent_node(state, top_k=top_k, llm=llm))
    workflow.add_node("summary", lambda state: summary_node(state, llm=llm))
    workflow.add_node("general", general_route_node)
    workflow.add_node("ambiguous", ambiguous_route_node)
    workflow.add_node("multi_intent", multi_intent_route_node)

    # 3. Add Retrieval, Context, and Generation Nodes
    workflow.add_node(
        "retrieval",
        lambda state: retrieval_node(state, top_k=top_k, use_reranker=use_reranker)
    )
    workflow.add_node("context_builder", context_builder_node)
    workflow.add_node(
        "generation",
        lambda state: generation_node(state, llm=llm)
    )

    # 4. Wire Edges
    workflow.add_edge(START, "router")

    # Conditional branching from router to category route nodes
    workflow.add_conditional_edges(
        "router",
        select_route_edge,
        {
            "financial": "financial",
            "risk": "risk",
            "litigation": "litigation",
            "related_party": "related_party",
            "summary": "summary",
            "general": "general",
            "ambiguous": "ambiguous",
            "multi_intent": "multi_intent"
        }
    )

    # Specialized agents handle their own pipeline and route to END
    workflow.add_edge("financial", END)
    workflow.add_edge("risk", END)
    workflow.add_edge("litigation", END)
    workflow.add_edge("related_party", END)
    workflow.add_edge("summary", END)

    # Remaining routes converge into existing hybrid retriever
    for r in ["general", "ambiguous", "multi_intent"]:
        workflow.add_edge(r, "retrieval")

    workflow.add_edge("retrieval", "context_builder")
    workflow.add_edge("context_builder", "generation")
    workflow.add_edge("generation", END)

    return workflow.compile()


def run_ipo_agent(
    query: str,
    top_k: int = 5,
    use_reranker: bool = True,
    llm: Optional[Any] = None,
    ipo_id: Optional[str] = None,
    document_target: Optional[str] = None
) -> IPOAgentState:
    """
    Executes the compiled LangGraph agent on a single query.
    """
    graph = build_ipo_graph(top_k=top_k, use_reranker=use_reranker, llm=llm)
    effective_target = document_target or "latest"
    initial_state: IPOAgentState = {
        "query": query,
        "ipo_id": ipo_id,
        "route": "",
        "document_target": effective_target,
        "filing_comparison": effective_target == "comparative",
        "route_confidence": None,
        "route_reasoning": None,
        "retrieval_results": [],
        "context": "",
        "answer": "",
        "sources": [],
        "error": None
    }
    final_state = graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    test_q = "Where is the registered and corporate office of the company located?"
    print(f"\nRunning LangGraph IPO Agent on: '{test_q}'\n")
    state = run_ipo_agent(test_q)
    print("=" * 80)
    print(f"QUERY: {state['query']}")
    print(f"ROUTE: {state['route']} (Confidence: {state['route_confidence']})")
    print(f"REASONING: {state['route_reasoning']}")
    print("-" * 80)
    print("ANSWER:")
    print(state["answer"])
    print("-" * 80)
    print(f"SOURCES ({len(state['sources'])} chunks):")
    for s in state["sources"]:
        print(f"  - {s['source_id']} (Page {s['page']}, {s['content_type']})")
    print("=" * 80)
