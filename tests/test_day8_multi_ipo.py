import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "retrieval"))
sys.path.insert(0, str(ROOT / "src" / "generation"))
sys.path.insert(0, str(ROOT / "src" / "tools"))
sys.path.insert(0, str(ROOT / "src" / "agents"))

from hybrid_retriever import resolve_document_filter, retrieve
from bm25_retriever import load_all_documents, bm25_search, build_bm25_index
from context_builder import extract_source_metadata, build_context
from router import detect_company_ipo_id, RouteDecision
from financial_agent import (
    FinancialFinding,
    select_best_finding,
    extract_financial_findings_from_context,
    run_financial_agent
)
from risk_agent import run_risk_agent
from litigation_agent import run_litigation_agent
from related_party_agent import run_related_party_agent
from llm_generator import get_chat_llm


class TestDay8MultiIPOArchitecture(unittest.TestCase):

    def test_company_detector(self):
        """Test rule-based IPO company identifier detection."""
        self.assertEqual(detect_company_ipo_id("What is HDB Financial Services revenue?"), "hdbfs_ipo")
        self.assertEqual(detect_company_ipo_id("What are HDBFS risk factors?"), "hdbfs_ipo")
        self.assertEqual(detect_company_ipo_id("What is Maharashtra Oil Extractions profit?"), "moel_ipo")
        self.assertEqual(detect_company_ipo_id("What are MOEL promoter holdings?"), "moel_ipo")
        self.assertEqual(detect_company_ipo_id("What is the company revenue?", default="moel_ipo"), "moel_ipo")

    def test_multi_document_bm25_corpus(self):
        """Verify that the BM25 corpus loads documents from MOEL, HDB DRHP, and HDB RHP."""
        docs = load_all_documents()
        self.assertGreater(len(docs), 10000, "BM25 corpus should contain > 10,000 documents across MOEL + HDB")

        ipo_ids = {d.get("metadata", {}).get("ipo_id") for d in docs}
        self.assertIn("moel_ipo", ipo_ids)
        self.assertIn("hdbfs_ipo", ipo_ids)

        doc_types = {d.get("metadata", {}).get("document_type") for d in docs}
        self.assertIn("DRHP", doc_types)
        self.assertIn("RHP", doc_types)

    def test_cross_ipo_isolation_bm25(self):
        """Verify strict IPO isolation: HDB query with ipo_id='hdbfs_ipo' never returns MOEL documents."""
        docs = load_all_documents()
        bm25_idx = build_bm25_index(docs)

        # Search with HDB filter
        hdb_results = bm25_search("revenue operations profit", bm25=bm25_idx, documents=docs, k=10, metadata_filter={"ipo_id": "hdbfs_ipo"})
        self.assertGreater(len(hdb_results), 0)
        for r in hdb_results:
            meta = r["document"].get("metadata", {})
            self.assertEqual(meta.get("ipo_id"), "hdbfs_ipo")
            self.assertNotEqual(meta.get("company_name"), "Maharashtra Oil Extractions Limited")

        # Search with MOEL filter
        moel_results = bm25_search("revenue operations profit", bm25=bm25_idx, documents=docs, k=10, metadata_filter={"ipo_id": "moel_ipo"})
        self.assertGreater(len(moel_results), 0)
        for r in moel_results:
            meta = r["document"].get("metadata", {})
            self.assertEqual(meta.get("ipo_id"), "moel_ipo")
            self.assertNotEqual(meta.get("company_name"), "HDB Financial Services Limited")

    def test_hdb_document_version_scoping(self):
        """Verify HDB DRHP vs RHP document version retrieval scoping."""
        # 1. HDB DRHP scoping
        drhp_results = retrieve("total assets borrowings", k=5, document_target="drhp", ipo_id="hdbfs_ipo", use_reranker=False)
        self.assertGreater(len(drhp_results), 0)
        for r in drhp_results:
            meta = extract_source_metadata(r)
            self.assertEqual(meta.get("ipo_id"), "hdbfs_ipo")
            self.assertEqual(meta.get("document_type"), "DRHP")

        # 2. HDB RHP scoping
        rhp_results = retrieve("total assets borrowings", k=5, document_target="rhp", ipo_id="hdbfs_ipo", use_reranker=False)
        self.assertGreater(len(rhp_results), 0)
        for r in rhp_results:
            meta = extract_source_metadata(r)
            self.assertEqual(meta.get("ipo_id"), "hdbfs_ipo")
            self.assertEqual(meta.get("document_type"), "RHP")

        # 3. HDB Latest scoping (should retrieve RHP as is_latest=True)
        latest_results = retrieve("total assets borrowings", k=5, document_target="latest", ipo_id="hdbfs_ipo", use_reranker=False)
        self.assertGreater(len(latest_results), 0)
        for r in latest_results:
            meta = extract_source_metadata(r)
            self.assertEqual(meta.get("ipo_id"), "hdbfs_ipo")
            self.assertTrue(meta.get("is_latest"))
            self.assertEqual(meta.get("document_type"), "RHP")

    def test_hdb_comparative_retrieval(self):
        """Verify dual-partition comparative retrieval for HDB returns both DRHP and RHP partitions."""
        comp_results = retrieve("interest income loan assets", k=6, document_target="comparative", ipo_id="hdbfs_ipo", use_reranker=False)
        self.assertGreater(len(comp_results), 0)

        partitions = {r.get("filing_partition") for r in comp_results}
        self.assertIn("DRHP", partitions, "Comparative retrieval must return DRHP partition")
        self.assertIn("RHP", partitions, "Comparative retrieval must return RHP partition")

        for r in comp_results:
            meta = extract_source_metadata(r)
            self.assertEqual(meta.get("ipo_id"), "hdbfs_ipo")

    def test_moel_rhp_safe_refusal(self):
        """Verify that attempting to retrieve RHP for MOEL safely yields empty results and proper agent refusal."""
        mock_llm = get_chat_llm(provider="mock")
        res = run_financial_agent(
            query="What is the RHP revenue?",
            top_k=2,
            llm=mock_llm,
            document_target="rhp",
            ipo_id="moel_ipo"
        )
        self.assertIn("No RHP evidence is available", res["answer"])
        self.assertEqual(len(res["sources"]), 0)

    def test_moel_comparative_safe_refusal(self):
        """Verify that comparative DRHP vs RHP query on MOEL safely refuses due to missing RHP."""
        mock_llm = get_chat_llm(provider="mock")
        res = run_risk_agent(
            query="Compare DRHP vs RHP risk factors",
            top_k=2,
            llm=mock_llm,
            document_target="comparative",
            ipo_id="moel_ipo"
        )
        self.assertIn("cannot be performed because RHP evidence is not currently loaded", res["answer"])

    def test_hdb_citation_formatting(self):
        """Verify context builder outputs company name, document type, and page in citations for HDB."""
        results = retrieve("capital adequacy ratio", k=2, document_target="latest", ipo_id="hdbfs_ipo", use_reranker=False)
        self.assertGreater(len(results), 0)
        ctx, sources = build_context(results)

        self.assertIn("HDB Financial Services Limited", ctx)
        self.assertIn("RHP", ctx)
        for s in sources:
            self.assertEqual(s.get("ipo_id"), "hdbfs_ipo")
            self.assertIn("hdbfs_", s.get("source_id"))


if __name__ == "__main__":
    unittest.main()
