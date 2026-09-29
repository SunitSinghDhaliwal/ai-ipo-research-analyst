import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

# Ensure project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RETRIEVAL_DIR = PROJECT_ROOT / "src" / "retrieval"
GENERATION_DIR = PROJECT_ROOT / "src" / "generation"
TOOLS_DIR = PROJECT_ROOT / "src" / "tools"
AGENTS_DIR = PROJECT_ROOT / "src" / "agents"

for p in [str(PROJECT_ROOT), str(RETRIEVAL_DIR), str(GENERATION_DIR), str(TOOLS_DIR), str(AGENTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from hybrid_retriever import retrieve
from context_builder import build_context, extract_source_metadata
from llm_generator import get_chat_llm
from state import IPOAgentState
from financial_calculator import (
    absolute_change,
    percentage_change,
    margin,
    format_currency,
    format_percentage,
    format_ratio,
    calculate_growth,
    calculate_margin,
    calculate_ratio,
    compare_financial_years,
    calculate_revenue_growth,
    calculate_profit_growth,
    calculate_ebitda_margin,
    calculate_net_profit_margin,
    calculate_debt_to_equity,
    calculate_current_ratio,
    calculate_period_comparison
)


# --------------------------------------------------
# Structured Financial Finding Schema
# --------------------------------------------------

class FinancialFinding(BaseModel):
    """
    Structured representation of a single extracted financial metric.
    """
    metric: str = Field(description="Canonical financial metric name, e.g. 'Revenue from Operations', 'Restated Profit after Tax'.")
    concept: str = Field(default="", description="Canonical concept key, e.g. 'revenue', 'pat', 'ebitda', 'debt', 'equity'.")
    raw_label: str = Field(default="", description="Original verbatim row label or text extracted from document.")
    period: str = Field(description="Fiscal period, e.g. 'FY2026', 'FY2025', 'FY2024'.")
    value: float = Field(description="Numerical value extracted from DRHP context.")
    unit: str = Field(default="₹ million", description="Currency unit or scale (e.g. '₹ million', '%').")
    source_id: str = Field(description="Source chunk ID or table ID (e.g. 'moe_p86_t1', 'drhp_p464_c1').")
    page: int = Field(description="Page number in the DRHP.")
    is_restated: bool = Field(default=True, description="Whether the figure represents restated financials.")
    document_type: str = Field(default="DRHP", description="Filing type: 'DRHP' or 'RHP'.")
    is_latest: bool = Field(default=True, description="Whether this finding comes from the authoritative latest filing.")


class FinancialExtractionResult(BaseModel):
    """
    Structured container for extracted financial findings and detected calculation needs.
    """
    findings: List[FinancialFinding] = Field(default_factory=list, description="Verified financial metrics extracted from context.")
    calculation_needed: bool = Field(default=False, description="True if query requires growth, margin, ratio, or comparison calculation.")
    calculation_type: Optional[str] = Field(default=None, description="Type of calculation: 'growth', 'margin', 'ratio', 'multi_period_trend'.")
    calculation_result: Optional[Dict[str, Any]] = Field(default=None, description="Result produced by deterministic Python calculation tools.")
    missing_data: bool = Field(default=False, description="True if required data for calculation could not be found.")
    missing_reason: Optional[str] = Field(default=None, description="Explanation of missing metric or fiscal period.")


# --------------------------------------------------
# Canonical Financial Concepts & Synonyms Map
# --------------------------------------------------

FINANCIAL_CONCEPT_SYNONYMS: Dict[str, List[str]] = {
    "total_income": [
        "total income c",
        "total income b",
        "total income",
        "total revenue i",
        "total revenue",
    ],
    "revenue": [
        "revenue from operations net",
        "revenue from operations a",
        "revenue from operations 1",
        "revenue from operations",
        "revenue from domestic sales",
        "revenue",
        "sales",
    ],
    "ebitda": [
        "ebitda h d e f g b",
        "ebitda 2",
        "ebitda",
        "ebidta",
        "operating profit",
    ],
    "pbt": [
        "restated profit before tax iii i ii",
        "restated profit before tax",
        "profit before tax",
        "pbt",
    ],
    "pat": [
        "restated profit for the period year v iii iv",
        "restated profit loss for the period fiscal year d",
        "restated profit for the period fiscal year a",
        "restated profit loss for the year a",
        "restated profit after tax 2",
        "restated profit after tax",
        "profit after tax pat 4",
        "profit after tax pat",
        "profit after tax",
        "restated profit",
        "net profit",
        "pat",
    ],
    "total_assets": [
        "total assets",
        "total equity and liabilities",
    ],
    "current_assets": [
        "total current assets a",
        "total current assets",
        "current assets",
    ],
    "current_liabilities": [
        "total current liabilities b",
        "total current liabilities",
        "current liabilities",
    ],
    "total_liabilities": [
        "total liabilities",
        "total non current liabilities",
    ],
    "equity": [
        "total equity attributable to owners of the parent",
        "total equity b",
        "total equity",
        "equity share capital a",
        "equity share capital",
        "adjusted equity",
        "net worth",
    ],
    "debt": [
        "total borrowings a",
        "total borrowings 7",
        "total borrowings",
        "total borrowing a b",
        "total borrowing",
        "borrowings",
        "total debt",
    ],
    "cash": [
        "cash and cash equivalents",
        "cash and bank balances",
    ]
}

CANONICAL_METRIC_NAMES: Dict[str, str] = {
    "revenue": "Revenue from Operations",
    "total_income": "Total Income",
    "ebitda": "EBITDA",
    "pbt": "Profit Before Tax (PBT)",
    "pat": "Profit After Tax (PAT)",
    "total_assets": "Total Assets",
    "current_assets": "Total Current Assets",
    "current_liabilities": "Total Current Liabilities",
    "total_liabilities": "Total Liabilities",
    "equity": "Total Equity",
    "debt": "Total Borrowings (Debt)",
    "cash": "Cash and Cash Equivalents"
}


# --------------------------------------------------
# Financial Normalization Helpers
# --------------------------------------------------

def clean_text(s: str) -> str:
    """Normalize whitespace and punctuation for semantic matching."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", s.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_financial_concept(label: str) -> Optional[str]:
    """
    Map an arbitrary table row label or narrative text string to a canonical financial concept.
    Evaluates exact match first, then substring containment in order of synonym specificity.
    """
    cleaned_label = clean_text(label)
    if not cleaned_label:
        return None

    for concept, syns in FINANCIAL_CONCEPT_SYNONYMS.items():
        for syn in syns:
            clean_syn = clean_text(syn)
            # Exact match or boundary-contained match
            if clean_syn == cleaned_label:
                return concept
            if f" {clean_syn} " in f" {cleaned_label} " or cleaned_label.startswith(clean_syn):
                return concept

    return None


def normalize_fiscal_period(token: str) -> Optional[str]:
    """
    Standardize period references across prospectuses:
    'Fiscal 2026', 'FY26', 'FY2026', '2026', 'March 31, 2026' -> 'FY2026'
    """
    token_str = str(token).strip()
    # Match FY26, FY2026, Fiscal 2026, 2026
    m = re.search(r"\b(?:fiscal\s*|fy\s*)?(202\d|2\d)\b", token_str, re.IGNORECASE)
    if m:
        yr = m.group(1)
        return f"FY20{yr}" if len(yr) == 2 else f"FY{yr}"
    return None


def parse_financial_number(s: str) -> Optional[float]:
    """
    Parse numerical values from prospectus cells, handling commas, currency symbols,
    percentages, and parentheses for negative values.
    """
    if not s or not isinstance(s, (str, int, float)):
        return None
    s_str = str(s).strip()
    if s_str in ["-", "--", "N/A", "NA", "Nil", "[●]", "•", ""]:
        return None

    # Remove currency, commas, percentages, and spaces
    cleaned = s_str.replace(",", "").replace("₹", "").replace("Rs", "").replace("%", "").strip()

    # Negative accounting values in parens: (69.47) -> -69.47
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1].strip()

    try:
        return float(cleaned)
    except ValueError:
        return None


# --------------------------------------------------
# Specialized Financial Retrieval
# --------------------------------------------------

def enrich_financial_query(query: str) -> str:
    """
    Enriches user financial query with domain-specific keywords and accounting terminology
    to ensure authoritative financial tables and restated statements are prioritized.
    """
    q_low = query.lower()
    additions = []

    if any(k in q_low for k in ["debt", "borrowing", "equity", "leverage"]):
        additions.append("total borrowings total equity debt to equity capital indebtedness")
    if any(k in q_low for k in ["current ratio", "working capital", "current asset", "current liabilit"]):
        additions.append("total current assets total current liabilities working capital balance sheet")
    if any(k in q_low for k in ["ebitda", "operating margin", "ebidta"]):
        additions.append("EBITDA margin revenue from operations restated profit")
    if any(k in q_low for k in ["revenue", "sales", "turnover", "income"]):
        additions.append("revenue from operations restated statement profit loss")
    if any(k in q_low for k in ["profit", "pat", "pbt", "net income", "loss"]):
        additions.append("restated profit after tax PAT profit for the year")
    if any(k in q_low for k in ["change", "growth", "compare", "trend", "three years", "between", "from fy", "to fy", "across"]):
        additions.append("Fiscal 2026 Fiscal 2025 Fiscal 2024 restated financial statements")

    if additions:
        return f"{query} " + " ".join(additions)
    return f"{query} restated statement profit loss balance sheet"


def financial_retrieve(
    query: str,
    top_k: int = 6,
    document_target: str = "latest",
    ipo_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Specialized retrieval for IPO financial queries:
      1. Enriches query with relevant accounting concepts and fiscal year tokens.
      2. Prioritizes financial tables (content_type='table') containing structured P&L/Balance Sheet data.
      3. Retrieves standard hybrid candidates (text + tables) with cross-encoder reranking.
      4. Merges and deduplicates while ensuring authoritative financial statements are placed first.
    """
    enriched_query = enrich_financial_query(query)

    table_filter = {"content_type": "table"}
    if ipo_id:
        table_filter["ipo_id"] = ipo_id

    # 1. Retrieve authoritative financial tables
    table_candidates = retrieve(
        query=enriched_query,
        k=max(4, top_k // 2),
        metadata_filter=table_filter,
        use_reranker=True,
        document_target=document_target,
        ipo_id=ipo_id
    )

    base_filter = {"ipo_id": ipo_id} if ipo_id else None
    # 2. Retrieve standard hybrid candidates (text + tables)
    standard_candidates = retrieve(
        query=enriched_query,
        k=top_k,
        metadata_filter=base_filter,
        use_reranker=True,
        document_target=document_target,
        ipo_id=ipo_id
    )

    # 3. Merge candidates, placing high-relevance tables first while deduplicating
    seen_ids = set()
    merged_results = []

    for item in table_candidates:
        doc_id = item["document_id"]
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            merged_results.append(item)

    for item in standard_candidates:
        doc_id = item["document_id"]
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            merged_results.append(item)

    return merged_results[:top_k]


# --------------------------------------------------
# Generalized Financial Data Extraction Engine
# --------------------------------------------------

def extract_financial_findings_from_context(context_str: str) -> List[FinancialFinding]:
    """
    Extracts structured financial findings from retrieved DRHP context without hardcoding.
    Parses both structured markdown/table blocks and narrative text blocks.
    Identifies metrics, fiscal periods, values, units, and source citations.
    """
    findings: List[FinancialFinding] = []
    seen_keys = set()  # (concept, period, source_id)

    # Split by standard context block delimiter
    blocks = context_str.split("--- [SOURCE ")
    if len(blocks) <= 1:
        # If no standard header, treat entire context as a single block
        blocks = ["", context_str]

    for block in blocks[1:]:
        # 1. Extract block metadata
        source_id = "unknown_source"
        page = 1
        doc_type = "DRHP"
        is_latest = True

        sid_match = re.search(r"Source ID:\s*([^\n]+)", block)
        if sid_match:
            source_id = sid_match.group(1).strip()
        page_match = re.search(r"Page:\s*(\d+)", block)
        if page_match:
            page = int(page_match.group(1).strip())
        filing_match = re.search(r"Filing:\s*([^\n]+)", block)
        if filing_match:
            doc_type = filing_match.group(1).strip().upper()

        lines = block.splitlines()

        # 2. Identify period sequence from table header lines
        header_periods: List[str] = []
        for line in lines[:10]:
            if line.startswith("Headers:") or line.startswith("Row 1:"):
                # Search for year tokens in the header line
                matches = re.findall(r"\b(?:fiscal\s*|fy\s*)?(202\d|2\d)\b", line, re.IGNORECASE)
                if len(matches) >= 2:
                    for m in matches:
                        p = f"FY20{m}" if len(m) == 2 else f"FY{m}"
                        if p not in header_periods:
                            header_periods.append(p)
                    if header_periods:
                        break

        # Single period fallback for balance sheet/indebtedness tables (e.g. 'March 31, 2026')
        if not header_periods:
            for line in lines[:5]:
                p = normalize_fiscal_period(line)
                if p and p not in header_periods:
                    header_periods.append(p)
                    break

        # 3. Parse table rows
        for line in lines:
            line_str = line.strip()
            if not line_str.startswith("Row "):
                continue

            parts = line_str.split(":", 1)
            if len(parts) < 2:
                continue

            # Split cells by table pipe delimiter
            cells = [c.strip() for c in parts[1].split("|")]
            # Filter empty or spacer cells
            meaningful_cells = [c for c in cells if c not in ["", "-"]]
            if not meaningful_cells:
                continue

            raw_label = meaningful_cells[0]
            concept = normalize_financial_concept(raw_label)
            if not concept:
                continue

            canonical_name = CANONICAL_METRIC_NAMES.get(concept, raw_label)

            # Extract numeric cells in the row
            num_cells = []
            for c in meaningful_cells[1:]:
                val = parse_financial_number(c)
                if val is not None:
                    num_cells.append(val)

            if not num_cells:
                continue

            # If header periods match the number of numeric cells, map sequentially
            if header_periods:
                pairs = list(zip(header_periods, num_cells))
                for period, val in pairs:
                    key = (concept, period, raw_label, source_id, doc_type)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        findings.append(FinancialFinding(
                            metric=canonical_name,
                            concept=concept,
                            raw_label=raw_label,
                            period=period,
                            value=val,
                            unit="₹ million",
                            source_id=source_id,
                            page=page,
                            is_restated=True,
                            document_type=doc_type,
                            is_latest=is_latest
                        ))
            else:
                # Default to FY2026 if single number found in single-period context
                val = num_cells[0]
                period = "FY2026"
                key = (concept, period, raw_label, source_id, doc_type)
                if key not in seen_keys:
                    seen_keys.add(key)
                    findings.append(FinancialFinding(
                        metric=canonical_name,
                        concept=concept,
                        raw_label=raw_label,
                        period=period,
                        value=val,
                        unit="₹ million",
                        source_id=source_id,
                        page=page,
                        is_restated=True,
                        document_type=doc_type,
                        is_latest=is_latest
                    ))

        # 4. Extract explicit narrative text disclosures if table rows did not capture them
        text_matches = re.finditer(
            r"(revenue from operations|ebitda|profit after tax|total borrowings|total equity)[^.\n]*?(?:increased|decreased|was|to|of)?\s*(?:₹|rs\.?)?\s*([\d,]+(?:\.\d+)?)\s*(?:million|crore)?[^.\n]*?(fiscal\s*202\d|fy\s*202\d|fy2\d|202\d)",
            block,
            re.IGNORECASE
        )
        for tm in text_matches:
            c_label = tm.group(1)
            val_str = tm.group(2)
            p_str = tm.group(3)
            concept = normalize_financial_concept(c_label)
            val = parse_financial_number(val_str)
            period = normalize_fiscal_period(p_str)
            if concept and val is not None and period:
                key = (concept, period, c_label, source_id, doc_type)
                if key not in seen_keys:
                    seen_keys.add(key)
                    canonical_name = CANONICAL_METRIC_NAMES.get(concept, c_label)
                    findings.append(FinancialFinding(
                        metric=canonical_name,
                        concept=concept,
                        raw_label=c_label,
                        period=period,
                        value=val,
                        unit="₹ million",
                        source_id=source_id,
                        page=page,
                        is_restated=True,
                        document_type=doc_type,
                        is_latest=is_latest
                    ))

    return findings


# --------------------------------------------------
# Best Finding Selector
# --------------------------------------------------

def select_best_finding(
    findings_list: List[FinancialFinding],
    concept: str,
    period: Optional[str] = None,
    document_target: str = "latest"
) -> Optional[FinancialFinding]:
    """
    Selects the most authoritative financial finding for a given concept and period,
    prioritizing official aggregate line items over sub-components or breakdowns.
    Also respects document_target ('latest', 'drhp', 'rhp').
    """
    candidates = [f for f in findings_list if f.concept == concept]
    if period:
        period_candidates = [f for f in candidates if f.period == period]
        if period_candidates:
            candidates = period_candidates

    target = (document_target or "latest").lower()
    if target == "drhp":
        target_cands = [f for f in candidates if getattr(f, "document_type", "DRHP") == "DRHP"]
        if target_cands:
            candidates = target_cands
    elif target == "rhp":
        target_cands = [f for f in candidates if getattr(f, "document_type", "DRHP") == "RHP"]
        candidates = target_cands
    elif target == "latest":
        latest_cands = [f for f in candidates if getattr(f, "is_latest", True)]
        if latest_cands:
            candidates = latest_cands

    if not candidates:
        return None

    # Priority 1: Aggregate totals for Balance Sheet and Debt/Equity
    if concept in ["debt", "equity", "current_assets", "current_liabilities", "total_assets", "total_liabilities"]:
        totals = [
            f for f in candidates
            if "total" in f.raw_label.lower()
            and "other" not in f.raw_label.lower()
            and "non-current" not in f.raw_label.lower()
            and "share" not in f.raw_label.lower()
        ]
        if totals:
            return totals[0]

    # Priority 2: For revenue, prioritize "Revenue from operations" over domestic/export breakdowns
    if concept == "revenue":
        ops = [
            f for f in candidates
            if "operations" in f.raw_label.lower()
            and "total" not in f.raw_label.lower()
            and "domestic" not in f.raw_label.lower()
            and "export" not in f.raw_label.lower()
            and "b2b" not in f.raw_label.lower()
            and "b2c" not in f.raw_label.lower()
        ]
        if ops:
            return ops[0]

    # Priority 3: For PAT, prioritize profit for the period/year over other comprehensive income
    if concept == "pat":
        pat_items = [
            f for f in candidates
            if "comprehensive" not in f.raw_label.lower()
            and "share" not in f.raw_label.lower()
        ]
        if pat_items:
            return pat_items[0]

    return candidates[0]


# --------------------------------------------------
# Deterministic Financial Calculation Dispatcher
# --------------------------------------------------

def detect_and_execute_financial_calculation(
    query: str,
    findings: List[FinancialFinding],
    document_target: str = "latest"
) -> FinancialExtractionResult:
    """
    Inspects user query intent and maps it to the appropriate deterministic calculation tool.
    Retrieves operands from verified findings and executes calculations in pure Python.
    Safely flags missing data to prevent hallucination.
    """
    q_low = query.lower()
    res = FinancialExtractionResult(findings=findings)

    def get_finding(c: str, p: Optional[str] = None) -> Optional[FinancialFinding]:
        return select_best_finding(findings, c, p, document_target=document_target)

    # 1. Determine Target Concept
    target_concept = None
    if "ebitda" in q_low:
        target_concept = "ebitda"
    elif any(k in q_low for k in ["profit", "pat", "net income"]):
        target_concept = "pat"
    elif any(k in q_low for k in ["revenue", "sales", "turnover"]):
        target_concept = "revenue"
    elif any(k in q_low for k in ["debt", "borrowing"]):
        target_concept = "debt"
    elif "equity" in q_low:
        target_concept = "equity"
    elif "current ratio" in q_low:
        target_concept = "current_ratio"
    elif "debt-to-equity" in q_low or "debt to equity" in q_low:
        target_concept = "debt_to_equity"

    # 2. Extract Requested Fiscal Periods from Query
    periods_found = []
    for yr in ["2024", "2025", "2026", "2027"]:
        if yr in q_low or f"fy{yr}" in q_low or f"fy{yr[-2:]}" in q_low:
            p = f"FY{yr}"
            if p not in periods_found:
                periods_found.append(p)

    # --------------------------------------------------
    # Calculation Case A: Margin Calculation (EBITDA Margin, PAT Margin)
    # --------------------------------------------------
    if "margin" in q_low or ("ebitda" in q_low and "margin" in q_low):
        res.calculation_needed = True
        res.calculation_type = "margin"
        target_period = periods_found[0] if periods_found else "FY2026"

        if target_concept == "pat" or "pat" in q_low or "profit" in q_low:
            # PAT Margin
            num_item = select_best_finding(findings, "pat", target_period)
            den_item = select_best_finding(findings, "revenue", target_period) or select_best_finding(findings, "total_income", target_period)
            margin_name = "PAT Margin"
            num_name = "Profit After Tax (PAT)"
        else:
            # EBITDA Margin (Default)
            num_item = select_best_finding(findings, "ebitda", target_period)
            den_item = select_best_finding(findings, "revenue", target_period)
            margin_name = "EBITDA Margin"
            num_name = "EBITDA"

        if num_item and den_item:
            calc = calculate_margin(
                numerator=num_item.value,
                denominator=den_item.value,
                margin_name=margin_name,
                numerator_name=num_name,
                denominator_name="Revenue from Operations",
                period=target_period,
                unit=num_item.unit
            )
            calc["numerator_source"] = f"{num_item.source_id}, Page {num_item.page}"
            calc["denominator_source"] = f"{den_item.source_id}, Page {den_item.page}"
            res.calculation_result = calc
        else:
            missing_terms = []
            if not num_item:
                missing_terms.append(f"{num_name} for {target_period}")
            if not den_item:
                missing_terms.append(f"Revenue for {target_period}")
            res.missing_data = True
            res.missing_reason = (
                f"I could not calculate the {margin_name} reliably because the required "
                f"{' and '.join(missing_terms)} values were not found in the retrieved prospectus evidence."
            )
        return res

    # --------------------------------------------------
    # Calculation Case B: Debt-to-Equity Ratio
    # --------------------------------------------------
    if "debt-to-equity" in q_low or "debt to equity" in q_low:
        res.calculation_needed = True
        res.calculation_type = "ratio"
        target_period = periods_found[0] if periods_found else "FY2026"

        d_item = select_best_finding(findings, "debt", target_period) or select_best_finding(findings, "debt")
        e_item = select_best_finding(findings, "equity", target_period) or select_best_finding(findings, "equity")

        if d_item and e_item:
            calc = calculate_debt_to_equity(
                debt=d_item.value,
                equity=e_item.value,
                period=target_period,
                unit=d_item.unit
            )
            calc["numerator_source"] = f"{d_item.source_id}, Page {d_item.page}"
            calc["denominator_source"] = f"{e_item.source_id}, Page {e_item.page}"
            res.calculation_result = calc
        else:
            res.missing_data = True
            res.missing_reason = (
                "I could not calculate the debt-to-equity ratio reliably because the required "
                "total debt/borrowings and total equity values were not found in the retrieved prospectus evidence."
            )
        return res

    # --------------------------------------------------
    # Calculation Case C: Current Ratio
    # --------------------------------------------------
    if "current ratio" in q_low:
        res.calculation_needed = True
        res.calculation_type = "ratio"
        target_period = periods_found[0] if periods_found else "FY2026"

        ca_item = select_best_finding(findings, "current_assets", target_period) or select_best_finding(findings, "current_assets")
        cl_item = select_best_finding(findings, "current_liabilities", target_period) or select_best_finding(findings, "current_liabilities")

        if ca_item and cl_item:
            calc = calculate_current_ratio(
                current_assets=ca_item.value,
                current_liabilities=cl_item.value,
                period=target_period,
                unit=ca_item.unit
            )
            calc["numerator_source"] = f"{ca_item.source_id}, Page {ca_item.page}"
            calc["denominator_source"] = f"{cl_item.source_id}, Page {cl_item.page}"
            res.calculation_result = calc
        else:
            res.missing_data = True
            res.missing_reason = (
                "I could not calculate the current ratio reliably because the required "
                "current assets and current liabilities values were not found in the retrieved prospectus evidence."
            )
        return res

    # --------------------------------------------------
    # Calculation Case D: Multi-Period Trend / YoY Comparison (3+ years or 'three years')
    # --------------------------------------------------
    is_multi_period = (
        len(periods_found) >= 3
        or any(k in q_low for k in ["last three years", "three years", "over fy2024", "across financial years"])
    )
    if is_multi_period:
        res.calculation_needed = True
        res.calculation_type = "multi_period_trend"
        if target_concept is None and not any(k in q_low for k in ["revenue", "profit", "ebitda", "sales", "income"]):
            res.missing_data = True
            res.missing_reason = (
                "I could not calculate the multi-period comparison reliably because the requested "
                "metric was not found in the retrieved prospectus evidence."
            )
            return res

        metric_concept = target_concept or "revenue"
        canonical_name = CANONICAL_METRIC_NAMES.get(metric_concept, "Financial Metric")

        target_periods = periods_found if len(periods_found) >= 3 else ["FY2024", "FY2025", "FY2026"]
        available_data = {}
        citations = {}
        for p in target_periods:
            best_f = select_best_finding(findings, metric_concept, p)
            if best_f:
                available_data[p] = best_f.value
                citations[p] = f"{best_f.source_id}, Page {best_f.page}"

        if len(available_data) >= 2:
            calc = compare_financial_years(
                values=available_data,
                metric=canonical_name,
                unit="₹ million"
            )
            calc["citations"] = citations
            res.calculation_result = calc
        else:
            res.missing_data = True
            res.missing_reason = (
                f"I could not calculate the multi-period comparison for {canonical_name} reliably "
                f"because the required values across periods ({', '.join(target_periods)}) were not found in the retrieved evidence."
            )
        return res

    # --------------------------------------------------
    # Calculation Case E: Two-Period Growth / Change
    # --------------------------------------------------
    is_growth_query = any(k in q_low for k in ["growth", "change", "increased", "decreased", "from fy", "to fy", "between"])
    if is_growth_query and len(periods_found) >= 2:
        res.calculation_needed = True
        res.calculation_type = "growth"

        if target_concept is None:
            res.missing_data = True
            res.missing_reason = (
                "I could not calculate growth reliably because the required metric "
                "was not found in the retrieved prospectus evidence."
            )
            return res

        metric_concept = target_concept
        canonical_name = CANONICAL_METRIC_NAMES.get(metric_concept, "Financial Metric")

        base_p = periods_found[0]
        target_p = periods_found[1]

        b_item = select_best_finding(findings, metric_concept, base_p)
        t_item = select_best_finding(findings, metric_concept, target_p)

        if b_item and t_item:
            calc = calculate_growth(
                current_value=t_item.value,
                previous_value=b_item.value,
                metric=canonical_name,
                current_period=target_p,
                previous_period=base_p,
                unit=b_item.unit
            )
            calc["base_source"] = f"{b_item.source_id}, Page {b_item.page}"
            calc["target_source"] = f"{t_item.source_id}, Page {t_item.page}"
            res.calculation_result = calc
        else:
            missing_periods = []
            if not b_item:
                missing_periods.append(base_p)
            if not t_item:
                missing_periods.append(target_p)
            res.missing_data = True
            res.missing_reason = (
                f"I could not calculate the {canonical_name} growth reliably because the required "
                f"values for {' and '.join(missing_periods)} were not found in the retrieved prospectus evidence."
            )
        return res

    # --------------------------------------------------
    # Calculation Case F: Single Metric Inquiry
    # --------------------------------------------------
    if target_concept and not res.calculation_needed:
        target_p = periods_found[0] if periods_found else "FY2026"
        best_f = select_best_finding(findings, target_concept, target_p)
        if best_f:
            res.calculation_type = "single_metric"
            res.calculation_result = {
                "tool": "single_metric",
                "metric": best_f.metric,
                "period": best_f.period,
                "value": best_f.value,
                "unit": best_f.unit,
                "source": f"{best_f.source_id}, Page {best_f.page}",
                "summary": f"{best_f.metric} in {best_f.period} was {best_f.value:,.2f} {best_f.unit} [{best_f.source_id}, Page {best_f.page}]."
            }

    return res


def detect_and_execute_comparison(query: str, context: str) -> Optional[Dict[str, Any]]:
    """
    Backward-compatible wrapper executing general extraction and deterministic calculation.
    """
    findings = extract_financial_findings_from_context(context)
    res = detect_and_execute_financial_calculation(query, findings)
    return res.calculation_result


# --------------------------------------------------
# Financial System Prompt & Generation
# --------------------------------------------------

FINANCIAL_AGENT_SYSTEM_PROMPT = """You are a Senior Equity Research Analyst specializing in Indian IPO prospectus filings.

Your objective is to provide an executive, insightful, and strictly grounded financial analysis based ONLY on the retrieved prospectus context provided.

CRITICAL INSTRUCTIONS:
1. STRICT FACTUAL GROUNDING & OPERATIONAL REVENUE:
   - In Indian IPO prospectuses (DRHPs/RHPs), company "Revenue" refers specifically to "Revenue from Operations" (the core operational top-line), NOT "Total Revenue" / "Total Income" (which includes non-operating other income). Always report Revenue from Operations when asked for revenue or revenue growth.
   - Answer exclusively using the exact financial numbers, tables, restated statements, and deterministic calculation blocks provided.
   - Never extrapolate, assume, interpolate, or invent financial figures.

2. GROUNDED CALCULATIONS VS REPORTED FACTS:
   - Always clearly report:
     A) Retrieved Facts: Reported source figures from the prospectus with exact page citations.
     B) Calculated Values: Deterministic calculations computed in pure Python (growth %, margins, ratios).
   - When a DETERMINISTIC PYTHON CALCULATION or VERIFIED PROSPECTUS METRIC block is provided, you MUST report the exact metric name, underlying values, and computed results from that block as your definitive primary answer. Do not substitute other lines from the context.
   - Never perform mental arithmetic or produce conflicting numbers.
   - Do NOT output python code blocks or raw code snippets. Present the calculated metrics naturally as clean text.

3. EXECUTIVE FINANCIAL REPORTING FORMAT (NO EXCESSIVE HASHTAGS):
   Structure your response cleanly using bold section headings:

   **Executive Summary**
   State the primary finding, metric value, and period in one clear, concise sentence with explicit citation: [Source: <Source ID>, Page <Page Number>].

   **Financial Performance & Metrics**
   - **Target Period**: <Metric Name> of <Value with unit> in <Period> [Source: <Source ID>, Page <Page Number>]
   - **Comparison / Base Period**: <Metric Name> of <Value with unit> in <Period> [Source: <Source ID>, Page <Page Number>]
   - **Growth / Change**: <Growth percentage with sign>% (Absolute change: <Value with unit>)

   **Analyst Assessment**
   Provide 1-2 professional analyst sentences explaining what this trend indicates based strictly on the text.

4. INSUFFICIENT EVIDENCE & HALLUCINATION REFUSAL:
   - If a requested financial metric or fiscal period is missing from the provided context:
     State clearly: "I could not calculate <metric/ratio> reliably because the required <values/periods> were not found in the retrieved prospectus evidence."
   - Do NOT guess, approximate, or substitute unrelated numbers.

5. MANDATORY INLINE CITATIONS:
   - Every reported figure MUST be immediately followed by its bracket citation: `[Source: <Source ID>, Page <Page Number>]`.
   - Do NOT output a raw text table of sources at the end (the UI already renders interactive source cards).
"""

FINANCIAL_USER_TEMPLATE = """RESEARCH QUERY:
{query}

{calculation_injection}

RETRIEVED DRHP FINANCIAL CONTEXT:
{context}

Please provide your rigorous, cited financial analysis following the instructions above.
"""


def run_financial_agent(
    query: str,
    top_k: int = 6,
    llm: Optional[Any] = None,
    document_target: str = "latest",
    ipo_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the specialized Financial Agent:
      1. Financial-prioritized hybrid retrieval with document scope filtering and IPO isolation.
      2. Context building preserving table structure, filing metadata, and comparative partitioning.
      3. Generalized financial data extraction from context.
      4. Deterministic Python calculation dispatch.
      5. Grounded answer generation using the specialized financial system prompt.
    """
    # 1. Specialized retrieval
    retrieved_items = financial_retrieve(
        query=query,
        top_k=top_k,
        document_target=document_target,
        ipo_id=ipo_id
    )

    # Safe handling: If query specifically asks for RHP and no RHP evidence is loaded
    if document_target == "rhp" and not retrieved_items:
        return {
            "query": query,
            "answer": "No RHP evidence is available. The Red Herring Prospectus (RHP) has not been loaded into the knowledge base yet.",
            "context": "No relevant RHP context was retrieved.",
            "sources": [],
            "findings": [],
            "calculation": None,
            "extraction_result": {},
            "retrieval_results": []
        }

    # Safe handling: If query asks for comparative DRHP vs RHP analysis and RHP is missing
    if document_target == "comparative":
        has_rhp = any(
            (isinstance(item, dict) and item.get("filing_partition") == "RHP")
            or (hasattr(item, "metadata") and item.metadata.get("document_type") == "RHP")
            for item in retrieved_items
        )
        if not has_rhp:
            context_str, sources_metadata = build_context(retrieved_items, is_comparative=True)
            return {
                "query": query,
                "answer": "A comparison between the DRHP and RHP cannot be performed because RHP evidence is not currently loaded in the knowledge base. Only the Draft Red Herring Prospectus (DRHP) is available.",
                "context": context_str,
                "sources": sources_metadata,
                "findings": [],
                "calculation": None,
                "extraction_result": {},
                "retrieval_results": retrieved_items
            }

    # 2. Build structured context
    context_str, sources_metadata = build_context(
        retrieved_items,
        is_comparative=(document_target == "comparative")
    )

    # 3. Generalized Financial Extraction
    findings = extract_financial_findings_from_context(context_str)

    # 4. Deterministic Calculation Execution
    extraction_res = detect_and_execute_financial_calculation(
        query,
        findings,
        document_target=document_target
    )
    calc_info = extraction_res.calculation_result

    # 5. Build Calculation Prompt Injection
    calc_injection = ""
    if extraction_res.missing_data and extraction_res.missing_reason:
        calc_injection = (
            "DATA AVAILABILITY STATUS:\n"
            f"{extraction_res.missing_reason}\n"
            "Instruct: Refuse to invent or hallucinate the missing numbers. Report this limitation directly.\n"
        )
    elif calc_info:
        tool_type = calc_info.get("tool")
        if tool_type == "calculate_growth":
            calc_injection = (
                "DETERMINISTIC PYTHON CALCULATION (Use these exact computed figures):\n"
                f"- Metric: {calc_info['metric']}\n"
                f"- {calc_info['previous_period']}: {calc_info['previous_value']:,.2f} {calc_info['unit']} [{calc_info.get('base_source', 'DRHP')}]\n"
                f"- {calc_info['current_period']}: {calc_info['current_value']:,.2f} {calc_info['unit']} [{calc_info.get('target_source', 'DRHP')}]\n"
                f"- Absolute Change: {calc_info['absolute_change']:+,.2f} {calc_info['unit']}\n"
                f"- Percentage Growth: {calc_info['formatted_growth']}\n"
                f"- Formula: {calc_info['formula']}\n"
                "State explicitly: 'The calculation is derived from the cited DRHP values.'\n"
            )
        elif tool_type == "calculate_margin":
            calc_injection = (
                "DETERMINISTIC PYTHON CALCULATION (Use these exact computed figures):\n"
                f"- Metric: {calc_info['margin_name']}\n"
                f"- Period: {calc_info.get('period', 'N/A')}\n"
                f"- {calc_info['numerator_name']}: {calc_info['numerator']:,.2f} {calc_info['unit']} [{calc_info.get('numerator_source', 'DRHP')}]\n"
                f"- {calc_info['denominator_name']}: {calc_info['denominator']:,.2f} {calc_info['unit']} [{calc_info.get('denominator_source', 'DRHP')}]\n"
                f"- Calculated Margin: {calc_info['formatted_margin']}\n"
                f"- Formula: {calc_info['formula']}\n"
                "State explicitly: 'The calculation is derived from the cited DRHP values.'\n"
            )
        elif tool_type == "calculate_ratio":
            calc_injection = (
                "DETERMINISTIC PYTHON CALCULATION (Use these exact computed figures):\n"
                f"- Ratio: {calc_info['ratio_name']}\n"
                f"- Period: {calc_info.get('period', 'N/A')}\n"
                f"- Underlying Figures:\n"
                f"  - {calc_info['numerator_name']}: {calc_info['numerator']:,.2f} {calc_info['unit']} [{calc_info.get('numerator_source', 'DRHP')}]\n"
                f"  - {calc_info['denominator_name']}: {calc_info['denominator']:,.2f} {calc_info['unit']} [{calc_info.get('denominator_source', 'DRHP')}]\n"
                f"- Calculated Ratio: {calc_info['formatted_ratio']}\n"
                f"- Formula: {calc_info['formula']}\n"
                "State explicitly: 'The calculation is derived from the cited DRHP values.'\n"
            )
        elif tool_type == "compare_financial_years":
            calc_injection = (
                "DETERMINISTIC PYTHON CALCULATION (Use these exact computed figures):\n"
                f"- Metric: {calc_info['metric']}\n"
                f"{calc_info['summary']}\n"
                "State explicitly: 'The calculation is derived from the cited DRHP values.'\n"
            )
        elif tool_type == "single_metric":
            calc_injection = (
                "VERIFIED PROSPECTUS METRIC:\n"
                f"- Metric: {calc_info['metric']}\n"
                f"- Period: {calc_info['period']}\n"
                f"- Reported Value: {calc_info['value']:,.2f} {calc_info['unit']} [{calc_info['source']}]\n"
                "State this verified metric clearly with the exact source citation.\n"
            )

    # 6. Generate Grounded Answer
    if llm is None:
        llm = get_chat_llm(temperature=0.0)

    user_prompt = FINANCIAL_USER_TEMPLATE.format(
        query=query,
        context=context_str,
        calculation_injection=calc_injection
    )

    messages = [
        ("system", FINANCIAL_AGENT_SYSTEM_PROMPT),
        ("user", user_prompt)
    ]

    try:
        response = llm.invoke(messages)
        answer = response.content.strip()
    except Exception as e:
        import traceback
        print("FINANCIAL AGENT EXCEPTION:", e)
        traceback.print_exc()
        answer = f"Financial analysis generated from verified calculations: {calc_injection if calc_injection else str(e)}"

    return {
        "query": query,
        "answer": answer,
        "context": context_str,
        "sources": sources_metadata,
        "findings": [f.model_dump() for f in findings],
        "calculation": calc_info,
        "extraction_result": extraction_res.model_dump(),
        "retrieval_results": retrieved_items
    }


def financial_agent_node(state: IPOAgentState, top_k: int = 6, llm: Optional[Any] = None) -> Dict[str, Any]:
    """
    LangGraph node implementing the specialized Financial Agent.
    Updates the state with financial findings, retrieval results, context, sources, and cited answer.
    """
    query = state["query"]
    document_target = state.get("document_target", "latest")
    ipo_id = state.get("ipo_id")
    try:
        agent_res = run_financial_agent(
            query=query,
            top_k=top_k,
            llm=llm,
            document_target=document_target,
            ipo_id=ipo_id
        )
        return {
            "ipo_id": ipo_id,
            "route": "financial",
            "document_target": document_target,
            "retrieval_results": agent_res["retrieval_results"],
            "context": agent_res["context"],
            "answer": agent_res["answer"],
            "sources": agent_res["sources"],
            "error": None
        }
    except Exception as e:
        return {
            "ipo_id": ipo_id,
            "route": "financial",
            "document_target": document_target,
            "answer": f"Financial Agent error: {str(e)}",
            "error": str(e)
        }


if __name__ == "__main__":
    test_q = "What was the company's revenue in FY2026?"
    print(f"\nRunning Financial Agent on: '{test_q}'\n")
    res = run_financial_agent(test_q)
    print("ANSWER:\n", res["answer"])
    print("\nFINDINGS EXTRACTED:", len(res["findings"]))
    for f in res["findings"][:5]:
        print(f"  - {f['metric']} ({f['period']}): {f['value']} [{f['source_id']}, Page {f['page']}]")
