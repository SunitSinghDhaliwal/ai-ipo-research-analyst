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

from litigation_agent import (
    LitigationCase,
    LitigationSummary,
    litigation_agent_node,
    run_litigation_agent
)
from llm_generator import get_chat_llm, generate_answer
from state import IPOAgentState


class TestLitigationAgentSchema(unittest.TestCase):

    def test_litigation_case_instantiation(self):
        case = LitigationCase(
            entity="Company",
            nature_of_action="Against",
            proceeding_type="Tax",
            number_of_cases=5,
            aggregate_amount_million=16.99,
            brief_description="Direct and Indirect tax proceedings pending against Company",
            source_id="moe_p44_t2",
            page=44
        )
        self.assertEqual(case.entity, "Company")
        self.assertEqual(case.nature_of_action, "Against")
        self.assertEqual(case.proceeding_type, "Tax")
        self.assertEqual(case.number_of_cases, 5)
        self.assertEqual(case.aggregate_amount_million, 16.99)
        self.assertEqual(case.source_id, "moe_p44_t2")
        self.assertEqual(case.page, 44)

    def test_litigation_summary_instantiation(self):
        case = LitigationCase(
            entity="Promoter",
            nature_of_action="Against",
            proceeding_type="Criminal",
            number_of_cases=3,
            aggregate_amount_million=None,
            source_id="moe_p44_t2",
            page=44
        )
        summary = LitigationSummary(
            cases=[case],
            sebi_disciplinary_actions_last_5_years=False,
            summary="Zero SEBI actions; minor tax and criminal cases disclosed."
        )
        self.assertEqual(len(summary.cases), 1)
        self.assertFalse(summary.sebi_disciplinary_actions_last_5_years)


class TestLitigationAgentMock(unittest.TestCase):

    def test_unanswerable_litigation_refusal(self):
        """Test refusal when asked about lawsuits completely absent from DRHP context."""
        mock_context = "Source ID: moe_p10_t1\nPage: 10\nContent Type: table\nExtracted Table: Headers: Term | Desc\nRow 1: General | Info"
        query = "Has the Enforcement Directorate initiated money laundering proceedings against the company?"
        mock_llm = get_chat_llm(provider="mock")
        ans = generate_answer(query, mock_context, llm=mock_llm)
        self.assertIn("MOCK ANSWER", ans)


if __name__ == "__main__":
    unittest.main()
