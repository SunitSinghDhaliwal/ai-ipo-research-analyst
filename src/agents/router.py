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
if str(PROJECT_ROOT / "src" / "agents") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "agents"))
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
    "multi_intent",
    "summary"
]

DocumentTargetType = Literal[
    "latest",
    "drhp",
    "rhp",
    "comparative"
]


def detect_company_ipo_id(query: str, default: str = "moel_ipo") -> str:
    """
    Deterministic rule-based company detector for IPO isolation.
    """
    q_low = query.lower()
    if any(h in q_low for h in ["hdb", "hdbfs", "hdb financial"]):
        return "hdbfs_ipo"
    if any(m in q_low for m in ["maharashtra oil", "moel", "extractions"]):
        return "moel_ipo"
    return default


class RouteDecision(BaseModel):
    """
    Structured output schema for the IPO Research Query Router.
    """
    route: RouteType = Field(
        description="The primary classification route of the user's IPO research query."
    )
    ipo_id: Optional[str] = Field(
        default=None,
        description="The IPO identifier: 'hdbfs_ipo' for HDB Financial Services Limited, 'moel_ipo' for Maharashtra Oil Extractions Limited. None if not specified in query."
    )
    document_target: DocumentTargetType = Field(
        default="latest",
        description="The document scope targeted by the user query: 'latest' (default when unspecified or referring to current/latest state), 'drhp' (explicitly asks about DRHP), 'rhp' (explicitly asks about RHP), or 'comparative' (explicitly compares DRHP vs RHP, or asks what changed between filings)."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score in the classification from 0.0 to 1.0."
    )
    reasoning: str = Field(
        description="Brief justification explaining why the query matches this classification and document target."
    )


ROUTER_SYSTEM_PROMPT = """You are an expert IPO Research Query Router analyzing questions regarding IPO prospectuses of companies (such as Maharashtra Oil Extractions Limited and HDB Financial Services Limited).
All queries about 'the company', 'the issuer', 'promoters', 'management', or IPO details refer to the identified issuer. Do not classify queries as ambiguous simply because the company name is omitted.

1. DOMAIN CLASSIFICATION:
Classify the user query into exactly ONE of the following domain categories:
- "financial":
  Queries requesting financial performance, revenue from operations, net profit (PAT), EBITDA, margins, financial ratios (current ratio, quick ratio, debt-to-equity ratio, return on equity/net worth), historical financial statements, borrowings, debt obligations, working capital, offer proceeds breakdown, capital expenditure, or share capital details.
- "risk":
  Queries inquiring about business risks, operational vulnerabilities, raw material price volatility/supply disruptions, geographic or customer concentration risks, competitive threats, regulatory hurdles, or disclosures from the Risk Factors section of the prospectus.
  Questions examining how external risks, commodity fluctuations, or supply disruptions impact business profitability/margins belong in "risk".
- "litigation":
  Queries seeking information about pending legal proceedings, court cases, criminal/civil suits, tax disputes (direct and indirect tax demands), statutory notices, or regulatory enforcement actions against the company, promoters, or directors.
- "related_party":
  Queries asking about related-party transactions (RPTs), contracts or commercial dealings with promoters, loans or advances to promoter-affiliated entities, promoter shareholding, or identification of promoters and promoter group entities.
- "general":
  Queries about general corporate background, date and state of incorporation, registered and corporate office address, objects of the offer, board of directors composition, Key Managerial Personnel (KMP) names/roles, or general business overview.
- "summary":
  Queries requesting a broad, complete, executive, or comprehensive research summary, investment brief, or holistic overview of the IPO (e.g., "Give me a complete research summary of HDB Financial Services", "Analyze this IPO comprehensively", "Give me the key things I should know about this IPO", "Provide an executive summary of this IPO").
- "multi_intent":
  Queries that explicitly and substantially span two or more distinct categories equally (e.g., asking for both pending litigation and promoter related-party contracts, or asking for both comprehensive financial statement figures and litigation details in a dual-question manner).
- "ambiguous":
  Queries that are excessively vague, underspecified, conversational, or lack sufficient context to determine a specific research intent (e.g., "tell me about the numbers", "what about the company?", "is this good?", "give me details").

2. DOCUMENT TARGET CLASSIFICATION:
Classify the targeted document scope into exactly ONE of:
- "latest":
  (DEFAULT) Queries asking general questions without mentioning a specific filing, or queries asking for "current", "latest", "recent", or "now" status.
- "drhp":
  Queries that explicitly specify the "DRHP", "Draft Red Herring Prospectus", "draft filing", or "initial draft" (e.g., "According to the DRHP, what was revenue?").
- "rhp":
  Queries that explicitly specify the "RHP", "Red Herring Prospectus", "final prospectus", or "RoC filing" (e.g., "According to the RHP, what is the revenue?").
- "comparative":
  Queries that explicitly compare, contrast, or ask what changed/differed between the DRHP and RHP (e.g., "What changed between the DRHP and RHP?", "Compare revenue in DRHP vs RHP").

CRITICAL GUIDELINES:
- Classify by INTENT, not mere keyword matching.
- Return ONLY the structured classification object with route, document_target, confidence, and reasoning.
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


def deterministic_query_fallback(query: str, default_ipo_id: str = "moel_ipo") -> RouteDecision:
    """
    Deterministic rule-based classification fallback when LLM router is unavailable or rate-limited.
    """
    q_low = query.lower()
    # 1. Company
    ipo_id = detect_company_ipo_id(query, default=default_ipo_id)

    # 2. Document target
    if any(k in q_low for k in ["compare", "difference", "between drhp and rhp", "drhp vs rhp", "drhp to rhp", "what changed", "changes between"]):
        doc_target = "comparative"
    elif "drhp" in q_low or "draft" in q_low:
        doc_target = "drhp"
    elif "rhp" in q_low or "final prospectus" in q_low:
        doc_target = "rhp"
    else:
        doc_target = "latest"

    # 3. Route
    # Remove known company names from query to avoid company name tokens (like "financial" in "HDB Financial Services") triggering wrong routes
    q_clean = q_low.replace("hdb financial services", "").replace("financial services", "").replace("maharashtra oil extractions", "").strip()

    if any(k in q_clean for k in ["what changed", "difference between drhp and rhp", "changes between", "compare drhp and rhp", "drhp vs rhp", "drhp to rhp"]):
        route = "summary"
        doc_target = "comparative"
    elif any(k in q_clean for k in ["risk", "hazard", "threat", "vulnerability"]):
        route = "risk"
    elif any(k in q_clean for k in ["litigation", "lawsuit", "case", "criminal", "tax dispute", "penalty", "sebi", "proceedings"]):
        route = "litigation"
    elif any(k in q_clean for k in ["related party", "related-party", "promoter group", "facility sharing", "advance", "rpt"]):
        route = "related_party"
    elif any(k in q_clean for k in ["summary", "complete research", "comprehensive analysis", "analyze this ipo", "key things i should know", "executive summary", "overview of the ipo", "full research", "comprehensive research", "research report"]):
        route = "summary"
    elif any(k in q_clean for k in ["revenue", "profit", "pat", "ebitda", "borrowing", "debt", "margin", "ratio", "income", "sales", "growth", "financials", "financial metric", "financial statement", "financial performance"]):
        route = "financial"
    else:
        route = "general"

    return RouteDecision(
        route=route,
        ipo_id=ipo_id,
        document_target=doc_target,
        confidence=0.85,
        reasoning="Deterministic classification based on domain keywords."
    )


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
        decision = deterministic_query_fallback(query)
    return decision


def router_node(state: IPOAgentState) -> Dict[str, Any]:
    """
    LangGraph node function for query routing.
    Takes current state and updates route, document_target, route_confidence, and route_reasoning.
    """
    query = state.get("query", "").strip()
    if not query:
        return {
            "route": "ambiguous",
            "document_target": "latest",
            "filing_comparison": False,
            "route_confidence": 0.0,
            "route_reasoning": "Empty query provided.",
            "error": "Query cannot be empty."
        }

    target_ipo_id = state.get("ipo_id")
    if not target_ipo_id:
        target_ipo_id = detect_company_ipo_id(query, default="moel_ipo")

    try:
        decision = classify_query(query)
        detected_ipo = target_ipo_id
        if not state.get("ipo_id") and getattr(decision, "ipo_id", None):
            detected_ipo = decision.ipo_id
        # Strict override if explicit company keywords exist in query
        if any(h in query.lower() for h in ["hdb", "hdbfs", "hdb financial"]):
            detected_ipo = "hdbfs_ipo"
        elif any(m in query.lower() for m in ["maharashtra oil", "moel"]):
            detected_ipo = "moel_ipo"

        # Respect explicitly set document_target from API (e.g. UI dropdown)
        explicit_target = state.get("document_target")
        final_doc_target = explicit_target if (explicit_target and explicit_target != "latest") else decision.document_target

        return {
            "ipo_id": detected_ipo,
            "route": decision.route,
            "document_target": final_doc_target,
            "filing_comparison": final_doc_target == "comparative",
            "route_confidence": decision.confidence,
            "route_reasoning": decision.reasoning,
            "error": None
        }
    except Exception as e:
        fb = deterministic_query_fallback(query, default_ipo_id=target_ipo_id)
        explicit_target = state.get("document_target")
        final_doc_target = explicit_target if (explicit_target and explicit_target != "latest") else fb.document_target
        return {
            "ipo_id": fb.ipo_id,
            "route": fb.route,
            "document_target": final_doc_target,
            "filing_comparison": final_doc_target == "comparative",
            "route_confidence": fb.confidence,
            "route_reasoning": fb.reasoning,
            "error": None
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
