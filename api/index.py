import os
import sys
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Project root path resolution
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "src" / "tools"))

load_dotenv(BASE_DIR / ".env")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Try importing local financial calculator; provide pure-Python fallback if unavailable
try:
    from financial_calculator import (
        absolute_change,
        percentage_change,
        margin,
        format_currency,
        format_percentage
    )
except ImportError:
    def absolute_change(old: float, new: float) -> float:
        return round(float(new) - float(old), 4)

    def percentage_change(old: float, new: float) -> Optional[float]:
        old_f, new_f = float(old), float(new)
        return round(((new_f - old_f) / abs(old_f)) * 100.0, 4) if old_f != 0 else None

    def margin(part: float, whole: float) -> Optional[float]:
        whole_f = float(whole)
        return round((float(part) / whole_f) * 100.0, 4) if whole_f != 0 else None

    def format_currency(value: float, unit: str = "₹ million") -> str:
        return f"{value:,.2f} {unit}".strip()

    def format_percentage(value: Optional[float], include_sign: bool = True) -> str:
        if value is None:
            return "N/A"
        return f"+{value:.2f}%" if include_sign and value > 0 else f"{value:.2f}%"

app = FastAPI(
    title="IPO Research Agent API (Vercel Serverless)",
    description="Serverless API for Indian IPO Prospectus Research",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Schemas
# --------------------------------------------------

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Research question to analyze")
    ipo_id: Optional[str] = Field(None, description="Target IPO identifier (e.g. hdbfs_ipo, moel_ipo)")
    document_target: Optional[str] = Field("latest", description="Target document scope (latest, drhp, rhp, comparative)")


class ResearchResponse(BaseModel):
    query: str
    ipo_id: Optional[str] = None
    route: str
    document_target: Optional[str] = None
    answer: str
    sources: List[Dict[str, Any]] = []
    route_confidence: Optional[float] = None
    route_reasoning: Optional[str] = None
    calculation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# --------------------------------------------------
# Company Metadata
# --------------------------------------------------

KNOWN_COMPANIES = {
    "hdbfs_ipo": {
        "ipo_id": "hdbfs_ipo",
        "name": "HDB Financial Services Limited",
        "short_name": "HDB Financial Services",
        "available_docs": ["DRHP", "RHP"],
        "latest_doc": "RHP",
        "scopes": [
            {"id": "latest", "label": "Latest", "hint": "RHP", "available": True},
            {"id": "drhp", "label": "DRHP", "hint": "Draft", "available": True},
            {"id": "rhp", "label": "RHP", "hint": "Final Prospectus", "available": True},
            {"id": "comparative", "label": "Comparative", "hint": "DRHP vs RHP", "available": True},
        ],
        "examples": [
            {"category": "Financials", "text": "What was the company's revenue and revenue growth?"},
            {"category": "Risk", "text": "What are the major risks associated with the company?"},
            {"category": "Litigation", "text": "What litigation is currently disclosed?"},
            {"category": "Related Parties", "text": "What related-party transactions are disclosed?"},
            {"category": "DRHP → RHP", "text": "What changed between the DRHP and RHP?"},
            {"category": "Summary", "text": "Give me a complete research summary of HDB Financial Services."}
        ]
    },
    "moel_ipo": {
        "ipo_id": "moel_ipo",
        "name": "Maharashtra Oil Extractions Limited",
        "short_name": "Maharashtra Oil Extractions",
        "available_docs": ["DRHP"],
        "latest_doc": "DRHP",
        "scopes": [
            {"id": "latest", "label": "Latest", "hint": "DRHP", "available": True},
            {"id": "drhp", "label": "DRHP", "hint": "Draft", "available": True},
            {"id": "rhp", "label": "RHP", "hint": "Not available", "available": False},
            {"id": "comparative", "label": "Comparative", "hint": "Not available (no RHP)", "available": False},
        ],
        "examples": [
            {"category": "Financials", "text": "What was the company's revenue and revenue growth?"},
            {"category": "Risk", "text": "What are the major risks associated with the company?"},
            {"category": "Litigation", "text": "What litigation is currently disclosed?"},
            {"category": "Related Parties", "text": "What related-party transactions are disclosed?"},
            {"category": "General", "text": "Where is the registered and corporate office of the company located?"}
        ]
    }
}


# --------------------------------------------------
# In-Memory Chunk Index for Serverless Retrieval
# --------------------------------------------------

_CHUNKS_CACHE: Dict[str, List[Dict[str, Any]]] = {}

def get_processed_chunks() -> List[Dict[str, Any]]:
    global _CHUNKS_CACHE
    if "all" in _CHUNKS_CACHE:
        return _CHUNKS_CACHE["all"]

    chunks: List[Dict[str, Any]] = []
    processed_dir = BASE_DIR / "data" / "processed"

    if processed_dir.exists():
        for item in processed_dir.iterdir():
            if not item.is_dir():
                continue
            c_file = item / "text_chunks.json"
            if c_file.exists():
                try:
                    with open(c_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    chunks.extend(data)
                except Exception as e:
                    print(f"Error loading {c_file}: {e}")

    _CHUNKS_CACHE["all"] = chunks
    return chunks


def filter_and_rank_chunks(query: str, ipo_id: str, document_target: str, top_k: int = 5) -> List[Dict[str, Any]]:
    chunks = get_processed_chunks()
    if not chunks:
        return []

    # 1. Filter by IPO and document scope
    filtered: List[Dict[str, Any]] = []
    for c in chunks:
        c_ipo = c.get("ipo_id")
        if ipo_id and c_ipo != ipo_id:
            continue

        c_doc_type = (c.get("document_type") or "").upper()
        if document_target == "drhp" and c_doc_type != "DRHP":
            continue
        if document_target == "rhp" and c_doc_type != "RHP":
            continue
        filtered.append(c)

    if not filtered:
        return []

    # 2. Simple BM25 or Token Frequency scoring
    query_tokens = set(re.findall(r"\w+", query.lower()))
    scored: List[tuple[float, Dict[str, Any]]] = []

    for c in filtered:
        text = c.get("text", "").lower()
        score = 0.0
        for token in query_tokens:
            if len(token) > 2 and token in text:
                score += text.count(token)
        if score > 0:
            scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@app.get("/api/health")
async def health():
    backend_url = os.getenv("BACKEND_URL")
    return {
        "status": "healthy",
        "service": "IPO Research Agent (Vercel Serverless)",
        "backend_proxy": backend_url if backend_url else "Standalone Serverless Mode",
        "has_groq_key": bool(os.getenv("GROQ_API_KEY"))
    }


@app.get("/api/companies")
async def get_companies():
    return {"companies": list(KNOWN_COMPANIES.values())}


@app.get("/api/ipos")
async def get_ipos():
    return {"companies": list(KNOWN_COMPANIES.values())}


@app.post("/api/research", response_model=ResearchResponse)
async def research(req: ResearchRequest):
    query = req.query.strip()
    ipo_id = req.ipo_id or "hdbfs_ipo"
    target = req.document_target or "latest"

    # Option 1: Proxy to external heavy backend if configured
    backend_url = os.getenv("BACKEND_URL")
    if backend_url:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(
                    f"{backend_url.rstrip('/')}/api/research",
                    json={"query": query, "ipo_id": ipo_id, "document_target": target}
                )
                if resp.status_code == 200:
                    return resp.json()
        except Exception as e:
            print(f"Backend proxy failed ({e}), falling back to serverless execution...")

    # Option 2: Serverless Execution
    # Safe refusal check for MOEL (which has only DRHP filed)
    if ipo_id == "moel_ipo" and target in ["rhp", "comparative"]:
        return ResearchResponse(
            query=query,
            ipo_id=ipo_id,
            route="refusal",
            document_target=target,
            answer="**Refusal Notice:** Insufficient evidence in the provided filing context.\n\n"
                   "Maharashtra Oil Extractions Limited has only filed a **Draft Red Herring Prospectus (DRHP)**. "
                   "No Red Herring Prospectus (RHP) has been filed with SEBI for this company. "
                   "Comparative analysis between DRHP and RHP is unavailable.",
            sources=[],
            route_confidence=1.0,
            route_reasoning="Refused due to requested filing (RHP) being unavailable for MOEL."
        )

    # Classify Route
    q_lower = query.lower()
    if any(k in q_lower for k in ["summary", "overview", "comprehensive", "executive summary", "key things"]):
        route = "summary"
    elif any(k in q_lower for k in ["revenue", "profit", "ebitda", "financial", "growth", "margin", "borrowing", "debt", "pat"]):
        route = "financial"
    elif any(k in q_lower for k in ["risk", "hazard", "threat", "danger", "vulnerability"]):
        route = "risk"
    elif any(k in q_lower for k in ["litigation", "lawsuit", "legal", "court", "tax dispute", "sebi"]):
        route = "litigation"
    elif any(k in q_lower for k in ["related party", "promoter", "guarantee", "advance"]):
        route = "related_party"
    else:
        route = "general"

    # Retrieve candidate chunks
    candidate_chunks = filter_and_rank_chunks(query, ipo_id, target, top_k=5)
    sources = []
    context_text = ""

    for c in candidate_chunks:
        sid = c.get("chunk_id", "source")
        page = c.get("page", 1)
        doc_type = c.get("document_type", "DRHP")
        sec = c.get("section_path") or c.get("section") or "Section"
        text = c.get("text", "")
        sources.append({
            "source_id": sid,
            "page": page,
            "document_type": doc_type,
            "section": sec,
            "content_type": "text"
        })
        context_text += f"\n--- [Source: {sid}, Page: {page}, Type: {doc_type}] ---\n{text}\n"

    # Check for Groq API
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ResearchResponse(
            query=query,
            ipo_id=ipo_id,
            route=route,
            document_target=target,
            answer="**Configuration Required:** Please configure `GROQ_API_KEY` in your Vercel Environment Variables to generate grounded AI responses.",
            sources=sources,
            route_confidence=0.9
        )

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        # Primary model and resilient fallback candidates for Groq
        configured_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        model_candidates = [
            configured_model,
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b"
        ]
        # Deduplicate while preserving order
        unique_models = list(dict.fromkeys([m for m in model_candidates if m]))

        company_info = KNOWN_COMPANIES.get(ipo_id, {})
        company_name = company_info.get("name", ipo_id)

        system_prompt = (
            f"You are an expert SEBI IPO Research Analyst specializing in {company_name}. "
            "Your answers must be grounded strictly in the provided prospectus excerpts. "
            "Always include page citations in the format [source_id, Page X]. "
            "Never hallucinate facts or extrapolate beyond provided evidence. "
            "If evidence is insufficient, state that clearly."
        )

        user_content = (
            f"Question: {query}\n"
            f"Company: {company_name} (Scope: {target.upper()})\n\n"
            f"Prospectus Evidence Context:\n{context_text if context_text else 'No specific excerpts retrieved.'}\n\n"
            "Provide a clear, analytical answer with page citations."
        )

        completion = None
        last_err = None
        used_model = unique_models[0]

        for m_name in unique_models:
            try:
                completion = client.chat.completions.create(
                    model=m_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.1,
                    max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "1500"))
                )
                used_model = m_name
                break
            except Exception as model_err:
                last_err = model_err
                err_str = str(model_err).lower()
                # If model not found or deprecated, try next candidate
                if "model_not_found" in err_str or "does not exist" in err_str or "404" in err_str:
                    continue
                else:
                    raise model_err

        if completion is None and last_err is not None:
            raise last_err

        msg = completion.choices[0].message
        answer = msg.content or getattr(msg, "reasoning", None) or "No answer generated."

        return ResearchResponse(
            query=query,
            ipo_id=ipo_id,
            route=route,
            document_target=target,
            answer=answer,
            sources=sources,
            route_confidence=0.95,
            route_reasoning=f"Handled via {route} pipeline on {company_name} (Model: {used_model})."
        )

    except Exception as e:
        return ResearchResponse(
            query=query,
            ipo_id=ipo_id,
            route=route,
            document_target=target,
            answer=f"Error generating answer: {str(e)}",
            sources=sources,
            error=str(e)
        )


@app.post("/api/upload")
async def upload_document():
    backend_url = os.getenv("BACKEND_URL")
    if backend_url:
        return {"status": "info", "message": f"Upload available via dedicated backend at {backend_url}."}
    return {
        "status": "info",
        "message": "Prospectus PDF extraction and OCR table indexing requires a persistent backend container. "
                   "Run 'python server.py' locally or connect a dedicated container via the BACKEND_URL environment variable."
    }
