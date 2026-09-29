import os
import sys
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
for sub in ["src", "src/agents", "src/generation", "src/retrieval", "src/tools", "src/utils", "scripts"]:
    p = str(PROJECT_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

# Load environment
load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Backend imports
from graph import run_ipo_agent
import hybrid_retriever
from bm25_retriever import load_all_documents, build_bm25_index
from ingest_ipo import run_pipeline

app = FastAPI(
    title="IPO Research Agent API",
    description="Agentic RAG for Indian IPO Prospectuses",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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
    error: Optional[str] = None


# --------------------------------------------------
# Helpers
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


def discover_all_companies() -> List[Dict[str, Any]]:
    """
    Discovers all ingested companies from data/processed directory,
    merging known metadata with any newly uploaded/ingested IPOs.
    """
    companies_dict = dict(KNOWN_COMPANIES)
    processed_dir = PROJECT_ROOT / "data" / "processed"

    if processed_dir.exists():
        for item in processed_dir.iterdir():
            if not item.is_dir():
                continue
            chunks_file = item / "text_chunks.json"
            if chunks_file.exists():
                try:
                    with open(chunks_file, "r", encoding="utf-8") as f:
                        chunks = json.load(f)
                    if chunks:
                        first = chunks[0]
                        ipo_id = first.get("ipo_id")
                        company_name = first.get("company")
                        doc_type = first.get("document_type", "DRHP")

                        if ipo_id and ipo_id not in companies_dict:
                            # Dynamic entry for uploaded company
                            companies_dict[ipo_id] = {
                                "ipo_id": ipo_id,
                                "name": company_name,
                                "short_name": company_name.replace("Limited", "").strip(),
                                "available_docs": [doc_type],
                                "latest_doc": doc_type,
                                "scopes": [
                                    {"id": "latest", "label": "Latest", "hint": doc_type, "available": True},
                                    {"id": "drhp", "label": "DRHP", "hint": "Available" if doc_type == "DRHP" else "Not available", "available": doc_type == "DRHP"},
                                    {"id": "rhp", "label": "RHP", "hint": "Available" if doc_type == "RHP" else "Not available", "available": doc_type == "RHP"},
                                    {"id": "comparative", "label": "Comparative", "hint": "Requires both DRHP & RHP", "available": False},
                                ],
                                "examples": [
                                    {"category": "Financials", "text": f"What was {company_name}'s revenue and revenue growth?"},
                                    {"category": "Risk", "text": f"What are the major risks associated with {company_name}?"},
                                    {"category": "Litigation", "text": f"What litigation is currently disclosed for {company_name}?"},
                                    {"category": "Related Parties", "text": f"What related-party transactions are disclosed for {company_name}?"}
                                ]
                            }
                        elif ipo_id and ipo_id in companies_dict:
                            # Update available documents
                            if doc_type not in companies_dict[ipo_id]["available_docs"]:
                                companies_dict[ipo_id]["available_docs"].append(doc_type)
                                if "DRHP" in companies_dict[ipo_id]["available_docs"] and "RHP" in companies_dict[ipo_id]["available_docs"]:
                                    for s in companies_dict[ipo_id]["scopes"]:
                                        s["available"] = True
                                        if s["id"] == "comparative":
                                            s["hint"] = "DRHP vs RHP"
                except Exception as e:
                    print(f"Warning discovering company in {item}: {e}")

    return list(companies_dict.values())


# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@app.get("/")
async def root():
    """Serves the primary UI interface."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI index.html not found.")
    return FileResponse(index_file)


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "service": "IPO Research Agent",
        "vectorstore": "Chroma (ChromaDB)",
        "retriever": "Hybrid (Dense BGE + Sparse BM25 + Cross-Encoder Reranker)"
    }


@app.get("/api/companies")
async def get_companies():
    """Returns list of all available IPO companies and document scopes."""
    companies = discover_all_companies()
    return {"companies": companies}


@app.post("/api/research", response_model=ResearchResponse)
async def research(req: ResearchRequest):
    """
    Executes grounded research query through the LangGraph pipeline.
    Preserves document scoping and cross-IPO isolation.
    """
    try:
        final_state = run_ipo_agent(
            query=req.query.strip(),
            ipo_id=req.ipo_id,
            document_target=req.document_target or "latest"
        )
        return ResearchResponse(
            query=final_state.get("query", req.query),
            ipo_id=final_state.get("ipo_id"),
            route=final_state.get("route", "general"),
            document_target=final_state.get("document_target", req.document_target),
            answer=final_state.get("answer", "No answer generated."),
            sources=final_state.get("sources", []),
            route_confidence=final_state.get("route_confidence"),
            route_reasoning=final_state.get("route_reasoning"),
            error=final_state.get("error")
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ResearchResponse(
            query=req.query,
            ipo_id=req.ipo_id,
            route="error",
            document_target=req.document_target,
            answer=f"Error executing research query: {str(e)}",
            sources=[],
            error=str(e)
        )


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    company: str = Form(...),
    ipo_id: str = Form(...),
    doc_type: str = Form("DRHP"),
    filing_date: str = Form("2024-01-01"),
    is_latest: bool = Form(True)
):
    """
    Uploads and indexes a new IPO prospectus PDF into Chroma and BM25 knowledge base.
    Uses the existing generalized ingestion pipeline without modifying backend architecture.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    clean_ipo_id = ipo_id.strip().lower().replace(" ", "_")
    clean_doc_type = doc_type.strip().upper()
    if clean_doc_type not in ["DRHP", "RHP"]:
        raise HTTPException(status_code=400, detail="Document type must be 'DRHP' or 'RHP'.")

    # 1. Save uploaded PDF
    raw_dir = PROJECT_ROOT / "data" / "raw" / clean_ipo_id / clean_doc_type.lower()
    raw_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = raw_dir / file.filename

    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 2. Run existing ingestion pipeline
        run_pipeline(
            pdf_path=str(pdf_path),
            company=company.strip(),
            ipo_id=clean_ipo_id,
            doc_type=clean_doc_type,
            version=f"{clean_doc_type.lower()}_v1",
            filing_date=filing_date.strip(),
            is_latest=is_latest
        )

        # 3. Reload in-memory BM25 index
        print("\nReloading in-memory BM25 corpus...")
        hybrid_retriever.documents = load_all_documents()
        hybrid_retriever.bm25 = build_bm25_index(hybrid_retriever.documents)
        print(f"BM25 reloaded with {len(hybrid_retriever.documents)} total searchable chunks.")

        return {
            "status": "success",
            "message": f"Successfully ingested and indexed {company} ({clean_doc_type}).",
            "ipo_id": clean_ipo_id,
            "company": company,
            "document_type": clean_doc_type,
            "filename": file.filename,
            "total_indexed_documents": len(hybrid_retriever.documents)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("\n=======================================================")
    print(" IPO RESEARCH AGENT — WEB DEMO SERVER")
    print(" Local URL: http://localhost:8000")
    print("=======================================================\n")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
