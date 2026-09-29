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
from context_builder import extract_source_metadata, format_single_source, build_context
from router import RouteDecision
from state import IPOAgentState
from financial_agent import (
    FinancialFinding,
    select_best_finding,
    detect_and_execute_financial_calculation,
    run_financial_agent
)
from llm_generator import get_chat_llm


class TestMultiDocumentArchitecture(unittest.TestCase):

    def test_resolve_document_filter(self):
        """Verify filter resolution across all document target modes."""
        f_latest = resolve_document_filter("latest", {"content_type": "table"})
        self.assertEqual(f_latest["is_latest"], True)
        self.assertEqual(f_latest["content_type"], "table")

        f_drhp = resolve_document_filter("drhp")
        self.assertEqual(f_drhp["document_type"], "DRHP")

        f_rhp = resolve_document_filter("rhp")
        self.assertEqual(f_rhp["document_type"], "RHP")

    def test_retrieve_rhp_unavailable_safety(self):
        """Verify that searching for RHP returns empty results safely with no crash when RHP is not available for MOEL."""
        rhp_results = retrieve("revenue", k=5, document_target="rhp", use_reranker=False, ipo_id="moel_ipo")
        self.assertEqual(len(rhp_results), 0)

    def test_retrieve_comparative_dual_partition(self):
        """Verify dual-partition retrieval returns DRHP partition and flags partitions for MOEL."""
        comp_results = retrieve("revenue", k=4, document_target="comparative", use_reranker=False, ipo_id="moel_ipo")
        self.assertGreater(len(comp_results), 0)
        for item in comp_results:
            self.assertEqual(item.get("filing_partition"), "DRHP")

    def test_context_builder_comparative_sections(self):
        """Verify context builder outputs distinct DRHP and RHP evidence sections."""
        mock_item = {
            "document_id": "drhp_p10_c1",
            "document": {
                "text": "DRHP revenue text",
                "metadata": {
                    "chunk_id": "drhp_p10_c1",
                    "company": "Maharashtra Oil Extractions Limited",
                    "document_type": "DRHP",
                    "page": 10,
                    "content_type": "text"
                }
            },
            "filing_partition": "DRHP"
        }
        ctx, meta = build_context([mock_item], is_comparative=True)
        self.assertIn("=== DRHP EVIDENCE ===", ctx)
        self.assertIn("=== RHP EVIDENCE ===", ctx)
        self.assertIn("Filing: DRHP", ctx)
        self.assertIn("No RHP evidence available", ctx)

    def test_financial_finding_schema_with_document_metadata(self):
        """Verify FinancialFinding captures document_type and is_latest."""
        finding = FinancialFinding(
            metric="Revenue from Operations",
            concept="revenue",
            raw_label="Revenue from operations",
            period="FY2026",
            value=22115.37,
            source_id="moe_p86_t1",
            page=86,
            document_type="DRHP",
            is_latest=True
        )
        self.assertEqual(finding.document_type, "DRHP")
        self.assertTrue(finding.is_latest)

    def test_financial_agent_rhp_missing_safe_response(self):
        """Verify financial agent returns safe unavailable response when RHP is targeted for MOEL."""
        mock_llm = get_chat_llm(provider="mock")
        res = run_financial_agent(
            query="According to the RHP, what was revenue?",
            top_k=2,
            llm=mock_llm,
            document_target="rhp",
            ipo_id="moel_ipo"
        )
        self.assertIn("No RHP evidence is available", res["answer"])
        self.assertEqual(len(res["sources"]), 0)

    def test_financial_agent_comparative_missing_safe_response(self):
        """Verify financial agent returns safe response for DRHP vs RHP comparative query on MOEL."""
        mock_llm = get_chat_llm(provider="mock")
        res = run_financial_agent(
            query="What changed in revenue between DRHP and RHP?",
            top_k=2,
            llm=mock_llm,
            document_target="comparative",
            ipo_id="moel_ipo"
        )
        self.assertIn("cannot be performed because RHP evidence is not currently loaded", res["answer"])

    def test_route_decision_document_target_field(self):
        """Verify RouteDecision supports document_target."""
        dec = RouteDecision(
            route="financial",
            document_target="drhp",
            confidence=0.99,
            reasoning="Explicit DRHP mention"
        )
        self.assertEqual(dec.document_target, "drhp")


if __name__ == "__main__":
    unittest.main()
