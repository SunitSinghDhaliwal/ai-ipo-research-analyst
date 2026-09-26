from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class IPOAgentState(TypedDict):
    """
    State definition for the IPO Research Agent LangGraph workflow.

    Fields:
      query: The original user research question.
      route: The classified intent (financial, risk, litigation, related_party, general, ambiguous, multi_intent).
      route_confidence: Confidence score of the routing decision (0.0 to 1.0).
      route_reasoning: Brief explanation of why this route was selected.
      retrieval_results: Raw retrieved candidate chunks/tables from the hybrid retriever.
      context: Formatted context string with source IDs and page numbers for the LLM.
      answer: Final generated answer strictly grounded in context with citations.
      sources: Structured metadata list of sources cited in the context.
      error: Error message if any step in the pipeline fails.
    """
    query: str
    route: str
    route_confidence: Optional[float]
    route_reasoning: Optional[str]
    retrieval_results: List[Dict[str, Any]]
    context: str
    answer: str
    sources: List[Dict[str, Any]]
    error: Optional[str]
