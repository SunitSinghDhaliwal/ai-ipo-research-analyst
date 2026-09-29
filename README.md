# IPO Research Agent

[![Tests](https://img.shields.io/badge/tests-62%2F62%20passing-brightgreen)](#running-test-suite)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange)](https://github.com/langchain-ai/langgraph)
[![Deploy on Vercel](https://vercel.com/button)](https://vercel.com/new)

An **Agentic RAG system** designed for Indian IPO prospectuses (Draft Red Herring Prospectus [DRHP] and Red Herring Prospectus [RHP]). Features **page-cited evidence attribution**, **deterministic financial calculation tools**, **domain-specialized LangGraph agents**, **version-aware comparative analysis**, and **100% cross-IPO tenant isolation**.

---

## Key Highlights & Benchmark Results

* **Two-Stage Hybrid Retrieval**: Combines dense semantic search (**ChromaDB** with `BAAI/bge-base-en-v1.5`), sparse lexical search (**BM25Okapi**), **Reciprocal Rank Fusion (RRF)**, and **Cross-Encoder Reranking** (`BAAI/bge-reranker-base`).
  * **Hit@5**: **75.0%** (up from 50.0% for standalone BM25, a **+50% relative lift**).
  * **Hit@10**: **100.0%** across complex prospectus queries.
* **Deterministic Financial Calculations (0% Arithmetic Hallucination)**: YoY growth, EBITDA margins, PAT margins, and debt-to-equity ratios are executed via a deterministic Python arithmetic engine with strict algebraic audit trails.
* **LangGraph Multi-Agent Architecture**: 100.0% classification accuracy across an 18-query routing benchmark spanning Financial, Risk, Litigation, Related-Party, and Executive Summary domains.
* **Comparative Version Scoping (DRHP vs. RHP)**: Dual-partition retrieval identifies balance sheet changes, business risk evolutions, and regulatory disclosures between draft and final filings.
* **100% Cross-IPO Tenant Isolation**: Zero cross-contamination across 10,000+ indexed chunks between distinct IPO entities (e.g. HDB Financial Services vs. Maharashtra Oil Extractions).
* **Safe Refusal Guardrails**: Refuses speculative questions on unfiled prospectus versions (e.g., MOEL RHP).
* **Production Reliability**: **62/62 passing unit and integration tests**.

---

## Project Architecture

```
User Query (via Web UI / REST API)
           │
           ▼
┌───────────────────────────────────────────────┐
│              FastAPI Server                   │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│        LangGraph State Router (100% Acc)      │
└──────┬──────────┬──────────┬──────────┬───────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
┌───────────┐┌──────────┐┌──────────┐┌──────────────┐
│ Financial ││   Risk   ││Litigation││Related-Party │
│   Agent   ││  Agent   ││  Agent   ││    Agent     │
└─────┬─────┘└────┬─────┘└────┬─────┘└──────┬───────┘
      │           │           │             │
      │   ┌───────┴───────────┴─────────┐   │
      │   │  Hybrid Dense-Sparse Engine │   │
      │   │  (BM25 + Chroma + Reranker) │   │
      │   └─────────────────────────────┘   │
      ▼                                     ▼
┌───────────────────────────┐ ┌─────────────────────┐
│  Deterministic Calculator │ │  Page Attribution   │
│   (YoY, Margins, Ratios)  │ │  [Doc ID, Page X]   │
└───────────────────────────┘ └─────────────────────┘
```

---

## Online Deployment

### 1. Deploy Frontend & Serverless API to Vercel

This repository includes a native `vercel.json` and a lightweight serverless handler in [`api/index.py`](api/index.py) designed to run inside Vercel's 250MB serverless limit:

1. Push this repository to your GitHub account (see [GitHub Setup](#github-setup) below).
2. Go to [Vercel Dashboard](https://vercel.com/new) and click **Add New Project** -> **Import Git Repository**.
3. In **Environment Variables**, add:
   * `GROQ_API_KEY`: Your Groq Cloud API key ([console.groq.com/keys](https://console.groq.com/keys))
   * `GROQ_MODEL`: `openai/gpt-oss-120b` or `llama-3.3-70b-versatile`
4. Click **Deploy**. Your IPO Research Agent UI and Serverless API will be live instantly!

*(Optional: To connect the Vercel frontend to a dedicated heavy GPU/CPU container running the full PyTorch + BGE Reranker + ChromaDB pipeline, set the `BACKEND_URL` environment variable in Vercel to your container URL).*

---

### 2. Deploy Container Backend (Render / Railway / Hugging Face Spaces)

A production [`Dockerfile`](Dockerfile) is included for container platforms:

```bash
# Build the container
docker build -t ipo-research-agent .

# Run the container
docker run -p 8000:8000 --env-file .env ipo-research-agent
```

On **Render** or **Railway**:
1. Connect your GitHub repository.
2. Select **Dockerfile** as the build runtime.
3. Add your `GROQ_API_KEY` in the environment settings.
4. Deploy to obtain your public backend URL.

---

## Local Development & Quickstart

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/YOUR_USERNAME/ai-ipo-research-analyst.git
cd ai-ipo-research-analyst

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Credentials

Create a `.env` file from the provided example:

```bash
cp .env.example .env
```

Add your Groq API key:

```bash
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
GROQ_ROUTER_MODEL=openai/gpt-oss-120b
GROQ_MAX_TOKENS=2048
```

### 3. Initialize / Build Vector Database

Rebuild or verify the Chroma vector store from the pre-processed filings:

```bash
python scripts/build_vectorstore.py
```

### 4. Launch Application

```bash
python server.py
```

Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## Running Test Suite

Run the complete test suite:

```bash
python -m unittest discover tests
```

Expected baseline: **62/62 tests passing (100% OK)**.

---

## Example Demo Queries

### HDB Financial Services Limited
* **Financials**: `What was HDB Financial Services' revenue and revenue growth?`
* **Risks**: `What are the major risks associated with HDB Financial Services?`
* **Litigation**: `What litigation is currently disclosed?`
* **Comparative (DRHP vs. RHP)**: `What changed between the DRHP and RHP?`
* **Executive Summary**: `Give me a complete research summary of HDB Financial Services.`

### Maharashtra Oil Extractions Limited
* **Financials**: `What was the company's revenue and revenue growth?`
* **Risks**: `What are the major risks associated with the company?`
* **Litigation**: `What litigation is currently disclosed?`
* **Missing Filing Refusal**: Select scope **RHP** or **Comparative** to verify safe refusal (MOEL has only filed a DRHP).

---

## License

MIT License. See [LICENSE](LICENSE) for details.
