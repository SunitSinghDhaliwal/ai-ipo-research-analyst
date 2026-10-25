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
from fastapi.responses import JSONResponse
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
_TABLES_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_BM25_CACHE: Dict[Any, Any] = {}

EXCERPT_CHARS = 1500
TABLE_CONTEXT_CHARS = 3000

STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "and", "or", "in", "of", "to", "for", "with",
    "on", "at", "by", "from", "up", "about", "into", "over", "after", "company", "disclosed",
    "associated", "currently", "between", "according", "give", "tell", "show", "details",
    "hdb", "hdbfs", "financial", "services", "limited", "maharashtra", "oil", "extractions"
}

def _processed_dir() -> Optional[Path]:
    # Check inside api/data/processed first (packaged for Vercel serverless), then project root
    api_dir = Path(__file__).resolve().parent
    for c_dir in [api_dir / "data" / "processed", BASE_DIR / "data" / "processed"]:
        if c_dir.exists() and any(c_dir.iterdir()):
            return c_dir
    return None


def _load_json_files(file_name: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    processed_dir = _processed_dir()
    if not processed_dir:
        return records
    for item in sorted(processed_dir.iterdir()):
        if not item.is_dir():
            continue
        f_path = item / file_name
        if f_path.exists():
            try:
                with open(f_path, "r", encoding="utf-8") as f:
                    records.extend(json.load(f))
            except Exception as e:
                print(f"Error loading {f_path}: {e}")
    return records


def get_processed_chunks() -> List[Dict[str, Any]]:
    if not _CHUNKS_CACHE.get("all"):
        _CHUNKS_CACHE["all"] = _load_json_files("text_chunks.json")
    return _CHUNKS_CACHE["all"]


def table_to_text(rows: List[List[Any]]) -> str:
    """
    Convert extracted table rows into searchable text that keeps the
    header + row structure together (same format as the full pipeline).
    """
    lines: List[str] = []
    cleaned_rows = []
    for row in rows or []:
        cells = [str(c).strip() if c is not None and str(c).strip() else "-" for c in row]
        if any(c != "-" for c in cells):
            cleaned_rows.append(cells)
    if not cleaned_rows:
        return ""
    start = 0
    if len([c for c in cleaned_rows[0] if c != "-"]) > 1:
        lines.append("Headers: " + " | ".join(cleaned_rows[0]))
        start = 1
    for idx, row in enumerate(cleaned_rows[start:], start=1):
        lines.append(f"Row {idx}: " + " | ".join(row))
    return "\n".join(lines)


def get_processed_tables() -> List[Dict[str, Any]]:
    if not _TABLES_CACHE.get("all"):
        tables = []
        for t in _load_json_files("tables.json"):
            text = table_to_text(t.get("rows", []))
            if len(text) < 40:
                continue
            tables.append({
                "chunk_id": t.get("table_id", "table"),
                "ipo_id": t.get("ipo_id"),
                "document_type": t.get("document_type"),
                "is_latest": t.get("is_latest"),
                "page": t.get("page"),
                "content_type": "table",
                "section_path": "TABLE",
                "text": text,
            })
        _TABLES_CACHE["all"] = tables
    return _TABLES_CACHE["all"]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def _query_tokens(query: str) -> List[str]:
    q_words = _tokenize(query)
    tokens = [w for w in q_words if len(w) > 2 and w not in STOPWORDS]
    if not tokens:
        tokens = [w for w in q_words if len(w) > 2]
    return tokens or q_words


def _partition(records: List[Dict[str, Any]], ipo_id: str, doc_scope: str) -> List[Dict[str, Any]]:
    """
    Strict company isolation + document version scoping.
    doc_scope: 'latest' | 'DRHP' | 'RHP'
    """
    out = []
    for c in records:
        if ipo_id and c.get("ipo_id") != ipo_id:
            continue
        d_type = (c.get("document_type") or "").upper()
        if doc_scope == "latest":
            if c.get("is_latest") is False:
                continue
        elif d_type != doc_scope:
            continue
        out.append(c)
    return out


def _bm25_search(kind: str, ipo_id: str, doc_scope: str, query: str, route: str, top_k: int) -> List[Dict[str, Any]]:
    records = get_processed_tables() if kind == "table" else get_processed_chunks()
    key = (kind, ipo_id, doc_scope)
    if key not in _BM25_CACHE:
        docs = _partition(records, ipo_id, doc_scope)
        corpus = [_tokenize(d.get("text", "")) for d in docs]
        try:
            from rank_bm25 import BM25Okapi
            index = BM25Okapi(corpus) if corpus else None
        except Exception:
            index = None
        _BM25_CACHE[key] = (docs, corpus, index)
    docs, corpus, index = _BM25_CACHE[key]
    if not docs:
        return []

    tokens = _query_tokens(query)
    if index is not None:
        scores = list(index.get_scores(tokens))
    else:
        scores = [float(sum(toks.count(t) for t in tokens)) for toks in corpus]

    scored = []
    for i, c in enumerate(docs):
        score = float(scores[i])
        sec = (c.get("section_path") or c.get("section") or "").upper()
        text_lower = c.get("text", "").lower()
        if route == "risk":
            if "RISK" in sec or "SECTION II" in sec:
                score += 5.0
            if "risk factor" in text_lower or "internal risk" in text_lower:
                score += 2.0
        elif route == "litigation":
            if "LITIGATION" in sec or "LEGAL" in sec or "OUTSTANDING" in sec:
                score += 5.0
            if "proceedings" in text_lower or "tax dispute" in text_lower:
                score += 2.0
        elif route == "financial":
            if "FINANCIAL INFORMATION" in sec or "SECTION V" in sec or "RESULTS OF OPERATIONS" in sec:
                score += 5.0
            if "revenue from operations" in text_lower or "restated profit" in text_lower or "profit for the" in text_lower:
                score += 2.0
        elif route == "related_party":
            if "RELATED PARTY" in sec or "PROMOTER" in sec:
                score += 5.0
        elif route == "summary":
            if "SUMMARY" in sec or "OVERVIEW" in sec or "INTRODUCTION" in sec:
                score += 3.0
        if score > 0:
            scored.append((score, i))

    if not scored:
        return docs[:top_k] if kind == "text" else []
    scored.sort(key=lambda x: x[0], reverse=True)
    return [docs[i] for _, i in scored[:top_k]]


def filter_and_rank_chunks(
    query: str,
    ipo_id: str,
    document_target: str,
    route: str = "general",
    top_k: int = 6
) -> List[Dict[str, Any]]:
    """
    Route-aware keyword (BM25) retrieval over text chunks and extracted tables.
      - Financial / summary questions also search the separate table store,
        so numbers stay attached to their headers and years.
      - 'comparative' retrieves evidence from BOTH the DRHP and the RHP.
    """
    if document_target == "comparative":
        scopes = ["DRHP", "RHP"]
    elif document_target in ("drhp", "rhp"):
        scopes = [document_target.upper()]
    else:
        scopes = ["latest"]

    n_tables = 3 if route == "financial" else (2 if route == "summary" else 0)
    per_scope_text = max(2, (top_k - (n_tables if len(scopes) == 1 else 0)) // len(scopes))
    per_scope_tables = n_tables if len(scopes) == 1 else (1 if n_tables else 0)

    results: List[Dict[str, Any]] = []
    for scope in scopes:
        results.extend(_bm25_search("text", ipo_id, scope, query, route, per_scope_text))
        if per_scope_tables:
            results.extend(_bm25_search("table", ipo_id, scope, query, route, per_scope_tables))
    return results


# --------------------------------------------------
# Deterministic Financial Calculations
# --------------------------------------------------

def _period_year(period: str) -> Optional[int]:
    if not period or "month" in period.lower() or "quarter" in period.lower():
        return None  # skip partial periods for YoY
    years = re.findall(r"(?:19|20)\d{2}", period)
    if years:
        return int(years[-1])
    short = re.search(r"FY\s*'?(\d{2})\b", period, re.I)
    return 2000 + int(short.group(1)) if short else None


def extract_financial_figures(client: Any, model: str, query: str, context_text: str, company_name: str = "the company") -> List[Dict[str, Any]]:
    """
    The LLM only EXTRACTS reported figures (with their source + page) as JSON.
    It never computes anything; all arithmetic happens in Python below.
    """
    prompt = (
        "From the prospectus excerpts below, extract the reported annual figures relevant to the question. "
        "Only copy numbers that appear verbatim in the excerpts. Do NOT calculate anything.\n"
        f"Only extract figures that belong to {company_name} itself: ignore peer / industry comparison tables "
        "that list other companies, and ignore percentages, margins, ratios and growth rates (absolute amounts only).\n"
        "Return JSON: {\"figures\": [{\"metric\": str, \"period\": str, \"value\": number, \"unit\": str, "
        "\"source_id\": str, \"page\": int}]}\n"
        "Use a consistent metric name for the same line item across years (e.g. 'Revenue from operations'). "
        "Use the period label as written (e.g. 'Fiscal 2025'). Values must be plain numbers without commas.\n\n"
        f"Question: {query}\n\nExcerpts:\n{context_text}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        figures = data.get("figures", []) if isinstance(data, dict) else []
        clean = []
        for f in figures:
            try:
                clean.append({
                    "metric": str(f.get("metric", "")).strip(),
                    "period": str(f.get("period", "")).strip(),
                    "value": float(str(f.get("value")).replace(",", "")),
                    "unit": str(f.get("unit", "") or "").strip(),
                    "source_id": str(f.get("source_id", "") or "").strip(),
                    "page": f.get("page"),
                })
            except (TypeError, ValueError):
                continue
        return clean
    except Exception as e:
        print(f"Figure extraction skipped: {e}")
        return []


def compute_financial_calculations(figures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Year-over-year growth between consecutive fiscal years, computed in Python."""
    by_metric: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for f in figures:
        year = _period_year(f["period"])
        if year is None or not f["metric"]:
            continue
        # Only grow absolute amounts; skip figures that are already rates (growth %, margins, ratios)
        label = f"{f['metric']} {f['unit']}".lower()
        if "%" in label or re.search(r"\b(growth|margin|ratios?|percent(age)?|return on|yield)\b", label):
            continue
        by_metric.setdefault(f["metric"].lower(), {}).setdefault(year, f)

    calcs = []
    for _, years in by_metric.items():
        ordered = sorted(years.items())
        for (y0, f0), (y1, f1) in zip(ordered, ordered[1:]):
            if y1 - y0 != 1 or f0["unit"].lower() != f1["unit"].lower():
                continue
            pct = percentage_change(f0["value"], f1["value"])
            if pct is None:
                continue
            calcs.append({
                "metric": f1["metric"],
                "from_period": f0["period"],
                "to_period": f1["period"],
                "from_value": f0["value"],
                "to_value": f1["value"],
                "unit": f1["unit"],
                "abs_change": absolute_change(f0["value"], f1["value"]),
                "change_pct": round(pct, 2),
                "formula": f"(({f1['value']:,} - {f0['value']:,}) / {f0['value']:,}) x 100 = {pct:+.2f}%",
                "citations": [
                    {"source_id": f0["source_id"], "page": f0["page"]},
                    {"source_id": f1["source_id"], "page": f1["page"]},
                ],
            })
    return calcs


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
    q_clean = query.lower()
    for comp_token in ["hdb financial services limited", "hdb financial services", "financial services", "maharashtra oil extractions limited", "maharashtra oil extractions"]:
        q_clean = q_clean.replace(comp_token, "").strip()

    if any(k in q_clean for k in ["summary", "overview", "comprehensive", "executive summary", "key things", "analyze this ipo"]):
        route = "summary"
    elif any(k in q_clean for k in ["risk", "hazard", "threat", "danger", "vulnerability"]):
        route = "risk"
    elif any(k in q_clean for k in ["litigation", "lawsuit", "legal", "court", "tax dispute", "sebi", "proceedings"]):
        route = "litigation"
    elif any(k in q_clean for k in ["related party", "related-party", "related parties", "promoter", "guarantee", "advance", "rpt"]):
        route = "related_party"
    elif any(k in q_clean for k in ["revenue", "profit", "ebitda", "financial", "growth", "margin", "borrowing", "debt", "pat"]):
        route = "financial"
    else:
        route = "general"

    # Retrieve candidate chunks with domain section boosting
    candidate_chunks = filter_and_rank_chunks(query, ipo_id, target, route=route, top_k=6)
    sources = []
    context_text = ""

    for c in candidate_chunks:
        sid = c.get("chunk_id", "source")
        page = c.get("page", 1)
        doc_type = c.get("document_type", "DRHP")
        sec = c.get("section_path") or c.get("section") or "Section"
        text = c.get("text", "")
        content_type = c.get("content_type", "text")
        if content_type == "table":
            text = text[:TABLE_CONTEXT_CHARS]
        sources.append({
            "source_id": sid,
            "page": page,
            "document_type": doc_type,
            "section": sec,
            "section_path": sec,
            "content_type": content_type,
            "text": text[:EXCERPT_CHARS]
        })
        context_text += f"\n--- [Source: {sid}, Page: {page}, Type: {doc_type}, Content: {content_type}] ---\n{text}\n"

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
            f"You are an expert SEBI IPO Research Analyst specializing in {company_name}.\n"
            "Your output must be strictly grounded in the provided prospectus excerpts.\n\n"
            "OUTPUT FORMATTING RULES (MANDATORY):\n"
            "1. INLINE CITATIONS: Every factual finding, disclosure, or metric must cite its source in brackets: [source_id, Page X].\n"
            "2. NO RAW HTML TAGS: Do NOT use raw HTML tags like '<br>', '<br/>', or '<span>' in markdown tables or bullet points. Use standard Markdown linebreaks or clean bullet lists.\n"
            "3. CLEAN READABLE ARITHMETIC: Format all financial calculations using clean, standard plain-text notation (e.g., '((140,593.8 - 123,181.2) / 123,181.2) × 100 = +14.1%'). Avoid raw LaTeX display code like '\\[', '\\]', '\\frac{}{}', '\\text{}', or '\\approx'.\n"
            "4. COMPLETE ANALYSIS: Always generate full, completed conclusions and bullet points without cutting off or truncating mid-sentence.\n"
            "5. ZERO HALLUCINATION: If evidence is insufficient, state: 'Insufficient evidence in the provided prospectus context to answer this question.'"
        )

        if target == "comparative":
            system_prompt += (
                "\n6. COMPARATIVE SCOPE: Excerpts come from both the DRHP (draft) and the RHP (final). "
                "Compare them explicitly and say which filing each point comes from. Only describe changes the excerpts show."
            )

        # Financial questions: LLM extracts figures, Python does ALL the arithmetic
        calculation = None
        calc_block = ""
        if route in ("financial", "summary") and context_text:
            figures = extract_financial_figures(client, os.getenv("GROQ_EXTRACT_MODEL", "openai/gpt-oss-20b"), query, context_text, company_name)
            calcs = compute_financial_calculations(figures)
            if calcs:
                calculation = {"method": "deterministic_python", "items": calcs}
                calc_lines = [
                    f"- {c['metric']} {c['from_period']} -> {c['to_period']}: change = {c['abs_change']:,} {c['unit']}; {c['formula']} "
                    f"[{c['citations'][0]['source_id']}, Page {c['citations'][0]['page']}] "
                    f"[{c['citations'][1]['source_id']}, Page {c['citations'][1]['page']}]"
                    for c in calcs
                ]
                calc_block = (
                    "\n\nVERIFIED CALCULATIONS (computed in Python from the cited figures):\n"
                    + "\n".join(calc_lines)
                )
            system_prompt += (
                "\n7. NO ARITHMETIC: Do not compute any differences, growth rates, margins or ratios yourself. "
                "Use only the VERIFIED CALCULATIONS provided, copied exactly. If a calculation is not provided, "
                "report the raw figures with citations and do not derive new percentages."
            )

        user_content = (
            f"Question: {query}\n"
            f"Company: {company_name} (Scope: {target.upper()})\n\n"
            f"Prospectus Evidence Context:\n{context_text if context_text else 'No specific excerpts retrieved.'}"
            f"{calc_block}\n\n"
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
                    max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "4096"))
                )
                used_model = m_name
                break
            except Exception as model_err:
                last_err = model_err
                err_str = str(model_err).lower()
                # Model missing/deprecated, or rate-limited on Groq's free tier: try the next model
                if ("model_not_found" in err_str or "does not exist" in err_str or "404" in err_str
                        or "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str):
                    continue
                else:
                    raise model_err

        if completion is None and last_err is not None:
            raise last_err

        msg = completion.choices[0].message
        answer = re.sub(r"<think>.*?</think>", "", msg.content or "", flags=re.DOTALL).strip() \
            or getattr(msg, "reasoning", None) or "No answer generated."

        return ResearchResponse(
            query=query,
            ipo_id=ipo_id,
            route=route,
            document_target=target,
            answer=answer,
            sources=sources,
            calculation=calculation,
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
        detail = f"Uploads are handled by the dedicated backend at {backend_url}."
    else:
        detail = ("Uploading new prospectuses needs the full version (PDF parsing, embeddings and a persistent index), "
                  "which runs in Docker or locally with 'python server.py'. This live demo comes with two pre-indexed IPOs: "
                  "HDB Financial Services (DRHP + RHP) and Maharashtra Oil Extractions (DRHP).")
    return JSONResponse(status_code=503, content={"status": "unavailable", "detail": detail})
