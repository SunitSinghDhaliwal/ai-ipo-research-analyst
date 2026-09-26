import os
import sys
from pathlib import Path
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field

# Ensure project paths are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "src" / "generation") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "generation"))

from llm_generator import get_chat_llm
from state import IPOAgentState

RouteType = Literal[
    "financial",
    "risk",
    "litigation",
    "related_party",
    "general",
    "ambiguous",
    "multi_intent"
]


class RouteDecision(BaseModel):
    """
    Structured output schema for the IPO Research Query Router.
    """
    route: RouteType = Field(
        description="The primary classification route of the user's IPO research query."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score in the classification from 0.0 to 1.0."
    )
    reasoning: str = Field(
        description="Brief justification explaining why the query matches this classification."
    )


ROUTER_SYSTEM_PROMPT = """You are an expert IPO Research Query Router analyzing questions regarding the Draft Red Herring Prospectus (DRHP) of Maharashtra Oil Extractions Limited.
All queries about 'the company', 'the issuer', 'promoters', 'management', or IPO details refer to this DRHP. Do not classify queries as ambiguous simply because the company name is omitted.

Classify the user query into exactly ONE of the following categories based on the true semantic intent:

1. "financial":
   - Queries requesting financial performance, revenue from operations, net profit (PAT), EBITDA, margins, financial ratios (such as current ratio, quick ratio, debt-to-equity ratio, return on equity/net worth), historical financial statements, borrowings, debt obligations, working capital, offer proceeds breakdown, capital expenditure, or share capital details.

2. "risk":
   - Queries inquiring about business risks, operational vulnerabilities, raw material price volatility/supply disruptions, geographic or customer concentration risks, competitive threats, regulatory hurdles, or disclosures from the Risk Factors section of the prospectus.
   - Questions examining how external risks, commodity fluctuations, or supply disruptions impact business profitability/margins belong in "risk".

3. "litigation":
   - Queries seeking information about pending legal proceedings, court cases, criminal/civil suits, tax disputes (direct and indirect tax demands), statutory notices, or regulatory enforcement actions against the company, promoters, or directors.

4. "related_party":
   - Queries asking about related-party transactions (RPTs), contracts or commercial dealings with promoters, loans or advances to promoter-affiliated entities, promoter shareholding, or identification of promoters and promoter group entities.

5. "general":
   - Queries about general corporate background, date and state of incorporation, registered and corporate office address, objects of the offer, board of directors composition, Key Managerial Personnel (KMP) names/roles, or general business overview.

6. "multi_intent":
   - Queries that explicitly and substantially span two or more distinct categories equally (e.g., asking for both pending litigation and promoter related-party contracts, or asking for both comprehensive financial statement figures and litigation details in a dual-question manner).

7. "ambiguous":
   - Queries that are excessively vague, underspecified, conversational, or lack sufficient context to determine a specific research intent (e.g., "tell me about the numbers", "what about the company?", "is this good?", "give me details").

CRITICAL CLASSIFICATION GUIDELINES:
- Classify by INTENT, not mere keyword matching.
- Return ONLY the structured classification object.
"""


def get_router_runnable(llm: Optional[Any] = None, provider: Optional[str] = None):
    """
    Creates a runnable chain using the existing Groq LLM with structured output.
    Defaults to openai/gpt-oss-120b on Groq for robust tool/schema generation.
    """
    if llm is None:
        model = os.getenv("GROQ_ROUTER_MODEL", "openai/gpt-oss-120b")
        llm = get_chat_llm(provider=provider, model_name=model, temperature=0.0)

    # Use LangChain structured output with Pydantic schema
    return llm.with_structured_output(RouteDecision)


def classify_query(
    query: str,
    llm: Optional[Any] = None,
    provider: Optional[str] = None
) -> RouteDecision:
    """
    Standalone function to classify a single research query into a RouteDecision.
    """
    runnable = get_router_runnable(llm=llm, provider=provider)
    messages = [
        ("system", ROUTER_SYSTEM_PROMPT),
        ("user", f"CLASSIFY THIS IPO RESEARCH QUERY:\n{query}")
    ]
    try:
        decision = runnable.invoke(messages)
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            fb_llm = get_chat_llm(model_name="openai/gpt-oss-20b")
            fb_runnable = fb_llm.with_structured_output(RouteDecision)
            decision = fb_runnable.invoke(messages)
        else:
            raise e
    return decision


def router_node(state: IPOAgentState) -> Dict[str, Any]:
    """
    LangGraph node function for query routing.
    Takes current state and updates route, route_confidence, and route_reasoning.
    """
    query = state.get("query", "").strip()
    if not query:
        return {
            "route": "ambiguous",
            "route_confidence": 0.0,
            "route_reasoning": "Empty query provided.",
            "error": "Query cannot be empty."
        }

    try:
        decision = classify_query(query)
        return {
            "route": decision.route,
            "route_confidence": decision.confidence,
            "route_reasoning": decision.reasoning,
            "error": None
        }
    except Exception as e:
        # Fallback gracefully to general route on routing exception
        return {
            "route": "general",
            "route_confidence": 0.0,
            "route_reasoning": f"Routing fallback due to error: {str(e)}",
            "error": str(e)
        }


if __name__ == "__main__":
    # Test queries
    test_queries = [
        "What was the company's revenue in FY2026?",
        "What are the major risks related to raw material prices?",
        "What litigation proceedings are pending against the company?",
        "What related party transactions were disclosed?",
        "Where is the company's registered office?",
        "Tell me about the general things in the report",
        "What are the pending litigation cases and what was the net profit in FY2025?"
    ]

    print("Running router node self-test...\n")
    for q in test_queries:
        res = classify_query(q)
        print(f"Query: '{q}'\n  -> Route: {res.route} (Confidence: {res.confidence:.2f})\n  -> Reasoning: {res.reasoning}\n")
