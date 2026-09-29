import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "retrieval"))
sys.path.insert(0, str(ROOT / "src" / "generation"))
sys.path.insert(0, str(ROOT / "src" / "tools"))
sys.path.insert(0, str(ROOT / "src" / "agents"))

from router import classify_query, deterministic_query_fallback
from graph import build_ipo_graph, run_ipo_agent, select_route_edge
from llm_generator import get_chat_llm


class TestDay9SummaryFlow(unittest.TestCase):

    def test_summary_query_routing(self):
        """Verify that broad summary queries route to 'summary'."""
        queries = [
            "Give me a complete research summary of HDB Financial Services.",
            "Analyze this IPO comprehensively.",
            "Give me the key things I should know about this IPO.",
            "Provide an executive summary of Maharashtra Oil Extractions Limited."
        ]
        for q in queries:
            dec = deterministic_query_fallback(q)
            self.assertEqual(dec.route, "summary", f"Query '{q}' should route to 'summary', got '{dec.route}'")

    def test_focused_query_regression(self):
        """Verify focused queries continue routing to their specialized domains."""
        self.assertEqual(deterministic_query_fallback("What was revenue in FY2024?").route, "financial")
        self.assertEqual(deterministic_query_fallback("What are the major risks?").route, "risk")
        self.assertEqual(deterministic_query_fallback("What litigation is disclosed?").route, "litigation")
        self.assertEqual(deterministic_query_fallback("What related party transactions were disclosed?").route, "related_party")

    def test_graph_topology_contains_summary(self):
        """Verify compiled LangGraph contains summary node and edge mapping."""
        graph = build_ipo_graph(top_k=2, use_reranker=False)
        self.assertIn("summary", graph.nodes)
        self.assertEqual(select_route_edge({"route": "summary"}), "summary")

    def test_summary_execution_moel(self):
        """Verify summary generation on MOEL enforces MOEL isolation and proper sectioning."""
        mock_llm = get_chat_llm(provider="mock")
        res = run_ipo_agent(
            query="Give me a complete research summary of Maharashtra Oil Extractions Limited.",
            top_k=2,
            use_reranker=False,
            llm=mock_llm,
            ipo_id="moel_ipo"
        )
        self.assertEqual(res["route"], "summary")
        self.assertEqual(res.get("ipo_id"), "moel_ipo")
        self.assertGreater(len(res["sources"]), 0)
        # Verify 0 HDB sources in MOEL summary
        source_ids = [s["source_id"] for s in res["sources"]]
        self.assertFalse(any("hdb" in sid for sid in source_ids), "MOEL summary must not contain HDB sources")
        self.assertIn("MOCK ANSWER", res["answer"])

    def test_summary_execution_hdbfs(self):
        """Verify summary generation on HDBFS enforces HDB isolation and proper sectioning."""
        mock_llm = get_chat_llm(provider="mock")
        res = run_ipo_agent(
            query="Give me a complete research summary of HDB Financial Services.",
            top_k=2,
            use_reranker=False,
            llm=mock_llm,
            ipo_id="hdbfs_ipo"
        )
        self.assertEqual(res["route"], "summary")
        self.assertEqual(res.get("ipo_id"), "hdbfs_ipo")
        # Verify 0 MOEL sources in HDBFS summary
        source_ids = [s["source_id"] for s in res["sources"]]
        self.assertTrue(all(s.get("ipo_id") == "hdbfs_ipo" for s in res["sources"]), "All sources must belong to hdbfs_ipo")
        self.assertFalse(any(sid.startswith("moe_") or sid.startswith("drhp_p") for sid in source_ids), "HDBFS summary must not contain MOEL sources")
        self.assertIn("MOCK ANSWER", res["answer"])


if __name__ == "__main__":
    unittest.main()
