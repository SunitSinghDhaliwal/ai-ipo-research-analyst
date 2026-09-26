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
# Route Dispatcher Nodes (Day 4 Scaffold for Day 5)
# --------------------------------------------------

def financial_route_node(state: IPOAgentState) -> Dict[str, Any]:
    """Route handler for financial queries (metrics, revenue, debt, PAT)."""
    return {"route": "financial"}


def risk_route_node(state: IPOAgentState) -> Dict[str, Any]:
    """Route handler for risk queries (raw material volatility, operational hazards)."""
    return {"route": "risk"}


def litigation_route_node(state: IPOAgentState) -> Dict[str, Any]:
    """Route handler for litigation queries (lawsuits, tax proceedings, penalties)."""
    return {"route": "litigation"}


def related_party_route_node(state: IPOAgentState) -> Dict[str, Any]:
    """Route handler for related-party queries (promoter transactions, entity ties)."""
    return {"route": "related_party"}


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
    Shared retrieval node invoking the existing Days 1-3 hybrid retriever.
    All routed categories converge here in Day 4.
    """
    query = state["query"]
    try:
        retrieved_items = retrieve(
            query=query,
            k=top_k,
            use_reranker=use_reranker
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
    Preserves chunk_id, page, content_type, and text.
    """
    results = state.get("retrieval_results", [])
    context_str, sources_metadata = build_context(results)
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

    try:
        answer = generate_answer(query=query, context=context_str, llm=llm)
        return {"answer": answer}
    except Exception as e:
        return {
            "answer": f"Generation error: {str(e)}",
            "error": str(e)
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

    # 2. Add Route Dispatcher Nodes (Specialized agents for Day 5; others generic)
    workflow.add_node("financial", lambda state: financial_agent_node(state, top_k=top_k, llm=llm))
    workflow.add_node("risk", lambda state: risk_agent_node(state, top_k=top_k, llm=llm))
    workflow.add_node("litigation", lambda state: litigation_agent_node(state, top_k=top_k, llm=llm))
    workflow.add_node("related_party", lambda state: related_party_agent_node(state, top_k=top_k, llm=llm))
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
    llm: Optional[Any] = None
) -> IPOAgentState:
    """
    Executes the compiled LangGraph agent on a single query.
    """
    graph = build_ipo_graph(top_k=top_k, use_reranker=use_reranker, llm=llm)
    initial_state: IPOAgentState = {
        "query": query,
        "route": "",
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
