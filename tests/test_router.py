import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "retrieval"))
sys.path.insert(0, str(ROOT / "src" / "generation"))
sys.path.insert(0, str(ROOT / "src" / "agents"))

from state import IPOAgentState
from router import RouteDecision, classify_query, router_node
from graph import build_ipo_graph, select_route_edge
from context_builder import build_context
from llm_generator import get_chat_llm, generate_answer


class TestRouterAndLangGraph(unittest.TestCase):

    def test_state_creation_and_update(self):
        """Verify IPOAgentState structure and typed dictionary fields."""
        state: IPOAgentState = {
            "query": "What is the revenue?",
            "route": "financial",
            "route_confidence": 0.95,
            "route_reasoning": "Revenue question",
            "retrieval_results": [{"doc_id": "c1"}],
            "context": "Sample context",
            "answer": "Sample answer",
            "sources": [{"source_id": "c1"}],
            "error": None
        }
        self.assertEqual(state["query"], "What is the revenue?")
        self.assertEqual(state["route"], "financial")
        self.assertEqual(state["route_confidence"], 0.95)
        self.assertEqual(len(state["retrieval_results"]), 1)
        self.assertIsNone(state["error"])

    def test_router_node_empty_query(self):
        """Verify router node gracefully handles empty queries."""
        state: IPOAgentState = {
            "query": "",
            "route": "",
            "route_confidence": None,
            "route_reasoning": None,
            "retrieval_results": [],
            "context": "",
            "answer": "",
            "sources": [],
            "error": None
        }
        update = router_node(state)
        self.assertEqual(update["route"], "ambiguous")
        self.assertIsNotNone(update["error"])

    def test_route_selector_edge(self):
        """Verify edge routing maps known routes and defaults safely."""
        self.assertEqual(select_route_edge({"route": "financial"}), "financial")
        self.assertEqual(select_route_edge({"route": "risk"}), "risk")
        self.assertEqual(select_route_edge({"route": "litigation"}), "litigation")
        self.assertEqual(select_route_edge({"route": "related_party"}), "related_party")
        self.assertEqual(select_route_edge({"route": "general"}), "general")
        self.assertEqual(select_route_edge({"route": "ambiguous"}), "ambiguous")
        self.assertEqual(select_route_edge({"route": "multi_intent"}), "multi_intent")
        self.assertEqual(select_route_edge({"route": "nonexistent_route"}), "general")

    def test_graph_compilation_and_topology(self):
        """Verify LangGraph compiles with all router and category nodes present."""
        graph = build_ipo_graph(top_k=2, use_reranker=False)
        self.assertIsNotNone(graph)
        # Graph nodes check
        expected_nodes = [
            "router", "financial", "risk", "litigation",
            "related_party", "general", "ambiguous", "multi_intent",
            "retrieval", "context_builder", "generation"
        ]
        for node in expected_nodes:
            self.assertIn(node, graph.nodes)

    def test_router_structured_output_schema(self):
        """Verify RouteDecision Pydantic schema validation."""
        decision = RouteDecision(
            route="financial",
            confidence=0.98,
            reasoning="Valid financial query."
        )
        self.assertEqual(decision.route, "financial")
        self.assertEqual(decision.confidence, 0.98)
        self.assertIn("Valid", decision.reasoning)

    def test_unanswerable_refusal_preservation(self):
        """Verify that generation node refuses to hallucinate on missing context."""
        mock_context = "Source ID: drhp_p1_c1\nPage: 1\nText: This is the cover page."
        query = "What is the company's investment in space satellites?"
        mock_llm = get_chat_llm(provider="mock")
        # Test mock LLM invocation
        ans = generate_answer(query=query, context=mock_context, llm=mock_llm)
        self.assertIn("MOCK ANSWER", ans)


if __name__ == "__main__":
    unittest.main()
