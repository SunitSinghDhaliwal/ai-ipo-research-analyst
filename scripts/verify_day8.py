import os
import sys
from pathlib import Path
import json
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "retrieval"))
sys.path.insert(0, str(ROOT / "src" / "generation"))
sys.path.insert(0, str(ROOT / "src" / "tools"))
sys.path.insert(0, str(ROOT / "src" / "agents"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from graph import run_ipo_agent
from financial_agent import run_financial_agent
from risk_agent import run_risk_agent
from litigation_agent import run_litigation_agent
from related_party_agent import run_related_party_agent
from router import router_node, classify_query

def print_separator(title):
    time.sleep(3)
    print("\n" + "=" * 80)
    print(f" {title} ")
    print("=" * 80)

def test_queries():
    results = {}

    # ----------------------------------------------------
    # STEP 7: BASIC HDB DOCUMENT SCOPING
    # ----------------------------------------------------
    print_separator("STEP 7: BASIC HDB DOCUMENT SCOPING")
    
    q1 = "Where is the registered office of HDB Financial Services Limited located?"
    print(f"\nQuery 1: {q1}")
    r1 = run_ipo_agent(q1, ipo_id="hdbfs_ipo")
    print(f"ROUTE: {r1['route']}, TARGET: {r1['document_target']}")
    print(f"ANSWER:\n{r1['answer']}")
    print(f"SOURCES: {[(s['source_id'], s['page'], s.get('document_type')) for s in r1['sources'][:3]]}")
    results["q1"] = {"answer": r1["answer"], "sources": r1["sources"][:3], "route": r1["route"], "target": r1["document_target"]}
    time.sleep(3)

    q2 = "According to the DRHP, what are the primary business segments of HDB Financial Services?"
    print(f"\nQuery 2: {q2}")
    r2 = run_ipo_agent(q2, ipo_id="hdbfs_ipo")
    print(f"ROUTE: {r2['route']}, TARGET: {r2['document_target']}")
    print(f"ANSWER:\n{r2['answer']}")
    print(f"SOURCES: {[(s['source_id'], s['page'], s.get('document_type')) for s in r2['sources'][:3]]}")
    results["q2"] = {"answer": r2["answer"], "sources": r2["sources"][:3], "route": r2["route"], "target": r2["document_target"]}
    time.sleep(3)

    q3 = "According to the RHP, what are the primary business segments of HDB Financial Services?"
    print(f"\nQuery 3: {q3}")
    r3 = run_ipo_agent(q3, ipo_id="hdbfs_ipo")
    print(f"ROUTE: {r3['route']}, TARGET: {r3['document_target']}")
    print(f"ANSWER:\n{r3['answer']}")
    print(f"SOURCES: {[(s['source_id'], s['page'], s.get('document_type')) for s in r3['sources'][:3]]}")
    results["q3"] = {"answer": r3["answer"], "sources": r3["sources"][:3], "route": r3["route"], "target": r3["document_target"]}
    time.sleep(3)

    q4 = "What are the main risk factors disclosed in the DRHP for HDB Financial Services?"
    print(f"\nQuery 4: {q4}")
    r4 = run_risk_agent(q4, document_target="drhp", ipo_id="hdbfs_ipo")
    print(f"ANSWER:\n{r4['answer']}")
    print(f"SOURCES: {[(s['source_id'], s['page'], s.get('document_type')) for s in r4['sources'][:3]]}")
    results["q4"] = {"answer": r4["answer"], "sources": r4["sources"][:3]}
    time.sleep(3)

    q5 = "What are the main risk factors disclosed in the RHP for HDB Financial Services?"
    print(f"\nQuery 5: {q5}")
    r5 = run_risk_agent(q5, document_target="rhp", ipo_id="hdbfs_ipo")
    print(f"ANSWER:\n{r5['answer']}")
    print(f"SOURCES: {[(s['source_id'], s['page'], s.get('document_type')) for s in r5['sources'][:3]]}")
    results["q5"] = {"answer": r5["answer"], "sources": r5["sources"][:3]}
    time.sleep(3)

    # ----------------------------------------------------
    # STEP 8: COMPARATIVE DRHP VS RHP RETRIEVAL (HDB)
    # ----------------------------------------------------
    print_separator("STEP 8: COMPARATIVE DRHP VS RHP RETRIEVAL (HDB)")

    q6 = "Compare the revenue or total income between the DRHP and RHP for HDB Financial Services. What changed?"
    print(f"\nQuery 6: {q6}")
    r6 = run_financial_agent(q6, document_target="comparative", ipo_id="hdbfs_ipo")
    print(f"ANSWER:\n{r6['answer']}")
    print(f"SOURCES: {[(s['source_id'], s['page'], s.get('document_type')) for s in r6['sources'][:4]]}")
    results["q6"] = {"answer": r6["answer"], "sources": r6["sources"][:4]}
    time.sleep(3)

    q7 = "Compare the risk factors between the DRHP and RHP for HDB Financial Services."
    print(f"\nQuery 7: {q7}")
    r7 = run_risk_agent(q7, document_target="comparative", ipo_id="hdbfs_ipo")
    print(f"ANSWER:\n{r7['answer']}")
    print(f"SOURCES: {[(s['source_id'], s['page'], s.get('document_type')) for s in r7['sources'][:4]]}")
    results["q7"] = {"answer": r7["answer"], "sources": r7["sources"][:4]}
    time.sleep(3)

    q8 = "Compare outstanding litigation between DRHP and RHP for HDB Financial Services."
    print(f"\nQuery 8: {q8}")
    r8 = run_litigation_agent(q8, document_target="comparative", ipo_id="hdbfs_ipo")
    print(f"ANSWER:\n{r8['answer']}")
    print(f"SOURCES: {[(s['source_id'], s['page'], s.get('document_type')) for s in r8['sources'][:4]]}")
    results["q8"] = {"answer": r8["answer"], "sources": r8["sources"][:4]}
    time.sleep(3)

    q9 = "Compare related party transactions between DRHP and RHP for HDB Financial Services."
    print(f"\nQuery 9: {q9}")
    r9 = run_related_party_agent(q9, document_target="comparative", ipo_id="hdbfs_ipo")
    print(f"ANSWER:\n{r9['answer']}")
    print(f"SOURCES: {[(s['source_id'], s['page'], s.get('document_type')) for s in r9['sources'][:4]]}")
    results["q9"] = {"answer": r9["answer"], "sources": r9["sources"][:4]}
    time.sleep(3)

    # ----------------------------------------------------
    # STEP 9: CROSS-IPO ISOLATION
    # ----------------------------------------------------
    print_separator("STEP 9: CROSS-IPO ISOLATION")

    q11 = "What are the major risks associated with raw material prices?"
    print(f"\nQuery 11 (MOEL Raw Material): {q11}")
    r11 = run_ipo_agent(q11, ipo_id="moel_ipo")
    print(f"IPO ID: {r11.get('ipo_id')}, ROUTE: {r11['route']}")
    print(f"ANSWER:\n{r11['answer']}")
    moel_sources = [s['source_id'] for s in r11['sources']]
    print(f"SOURCES: {moel_sources}")
    has_hdb_in_moel = any("hdb" in sid for sid in moel_sources)
    print(f"Any HDB in MOEL? {has_hdb_in_moel}")
    results["q11"] = {"answer": r11["answer"], "sources": r11["sources"], "has_hdb_in_moel": has_hdb_in_moel}
    time.sleep(3)

    q12 = "What are the major credit, interest rate, and lending risks?"
    print(f"\nQuery 12 (HDB Lending Risk): {q12}")
    r12 = run_ipo_agent(q12, ipo_id="hdbfs_ipo")
    print(f"IPO ID: {r12.get('ipo_id')}, ROUTE: {r12['route']}")
    print(f"ANSWER:\n{r12['answer']}")
    hdb_sources = [s['source_id'] for s in r12['sources']]
    print(f"SOURCES: {hdb_sources}")
    has_moel_in_hdb = any("moe" in sid or "drhp_p" in sid for sid in hdb_sources)
    print(f"Any MOEL in HDB? {has_moel_in_hdb}")
    results["q12"] = {"answer": r12["answer"], "sources": r12["sources"], "has_moel_in_hdb": has_moel_in_hdb}
    time.sleep(3)

    # ----------------------------------------------------
    # STEP 10: FINANCIAL ANALYSIS TOOLS ON HDB
    # ----------------------------------------------------
    print_separator("STEP 10: FINANCIAL ANALYSIS TOOLS ON HDB")

    q13 = "What was HDB Financial Services revenue from operations in FY2024?"
    print(f"\nQuery 13: {q13}")
    r13 = run_financial_agent(q13, ipo_id="hdbfs_ipo")
    print(f"ANSWER:\n{r13['answer']}")
    print(f"CALCULATION: {r13.get('calculation')}")
    results["q13"] = {"answer": r13["answer"], "calc": r13.get("calculation")}
    time.sleep(3)

    q14 = "What was HDB Financial Services revenue growth between FY2023 and FY2024?"
    print(f"\nQuery 14: {q14}")
    r14 = run_financial_agent(q14, ipo_id="hdbfs_ipo")
    print(f"ANSWER:\n{r14['answer']}")
    print(f"CALCULATION: {r14.get('calculation')}")
    results["q14"] = {"answer": r14["answer"], "calc": r14.get("calculation")}
    time.sleep(3)

    # ----------------------------------------------------
    # STEP 12: REFUSAL & FAILURE CASES
    # ----------------------------------------------------
    print_separator("STEP 12: REFUSAL & FAILURE CASES")

    q16 = "According to the RHP of Maharashtra Oil Extractions Limited, what was the revenue?"
    print(f"\nQuery 16: {q16}")
    r16 = run_ipo_agent(q16, ipo_id="moel_ipo")
    print(f"ANSWER:\n{r16['answer']}")
    results["q16"] = {"answer": r16["answer"]}
    time.sleep(3)

    q17 = "Compare the DRHP vs RHP risk factors for Maharashtra Oil Extractions Limited"
    print(f"\nQuery 17: {q17}")
    r17 = run_ipo_agent(q17, ipo_id="moel_ipo")
    print(f"ANSWER:\n{r17['answer']}")
    results["q17"] = {"answer": r17["answer"]}

    # Save results to json
    out_path = ROOT / "data" / "day8_validation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved validation results to {out_path}")

if __name__ == "__main__":
    test_queries()
