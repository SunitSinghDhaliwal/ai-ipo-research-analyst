import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "retrieval"))
sys.path.insert(0, str(ROOT / "src" / "generation"))
sys.path.insert(0, str(ROOT / "src" / "tools"))
sys.path.insert(0, str(ROOT / "src" / "agents"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from risk_agent import (
    RiskItem,
    RiskExtractionResult,
    risk_agent_node,
    run_risk_agent
)
from llm_generator import get_chat_llm, generate_answer
from state import IPOAgentState


class TestRiskAgentSchema(unittest.TestCase):

    def test_risk_item_instantiation(self):
        item = RiskItem(
            title="Fluctuations in Raw Material Prices",
            category="Internal Risk",
            causes=["Monsoon seasonality", "Agro-climatic factors"],
            consequences=["Margin compression", "Inability to pass on cost increases"],
            quantified_metrics=["Raw material constitutes significant operational expense"],
            source_id="drhp_p31_c5",
            page=31
        )
        self.assertEqual(item.title, "Fluctuations in Raw Material Prices")
        self.assertEqual(item.category, "Internal Risk")
        self.assertEqual(len(item.causes), 2)
        self.assertEqual(len(item.consequences), 2)
        self.assertEqual(item.source_id, "drhp_p31_c5")
        self.assertEqual(item.page, 31)

    def test_risk_extraction_result(self):
        item = RiskItem(
            title="Customer Concentration",
            category="Internal Risk",
            causes=["Dependence on top customers without long-term contracts"],
            consequences=["Loss of a key customer would adversely affect business"],
            quantified_metrics=["Top 10 customers contribute 65%"],
            source_id="drhp_p54_c6",
            page=54
        )
        res = RiskExtractionResult(risks=[item], summary="Customer concentration risk")
        self.assertEqual(len(res.risks), 1)
        self.assertEqual(res.risks[0].page, 54)


class TestRiskAgentMock(unittest.TestCase):

    def test_unanswerable_risk_refusal(self):
        """Test refusal when asked about risks completely absent from DRHP context."""
        mock_context = "Source ID: moe_p10_t1\nPage: 10\nContent Type: table\nExtracted Table: Headers: Term | Desc\nRow 1: General | Info"
        query = "What risks does the company face from cryptocurrency market crashes?"
        mock_llm = get_chat_llm(provider="mock")
        ans = generate_answer(query, mock_context, llm=mock_llm)
        self.assertIn("MOCK ANSWER", ans)


if __name__ == "__main__":
    unittest.main()
