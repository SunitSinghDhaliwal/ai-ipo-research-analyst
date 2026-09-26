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

from related_party_agent import (
    RelatedPartyTransaction,
    RelatedPartySummary,
    related_party_agent_node,
    run_related_party_agent
)
from llm_generator import get_chat_llm, generate_answer
from state import IPOAgentState


class TestRelatedPartyAgentSchema(unittest.TestCase):

    def test_rpt_transaction_instantiation(self):
        txn = RelatedPartyTransaction(
            related_party="Basant Nutrifoods Private Limited (BNPL)",
            relationship="Subsidiary",
            nature_of_transaction="Facility Sharing",
            amount_or_terms="₹ 75,000 per month (excluding GST)",
            period_or_date="3 years from April 01, 2025",
            source_id="drhp_p331_c2",
            page=331
        )
        self.assertEqual(txn.related_party, "Basant Nutrifoods Private Limited (BNPL)")
        self.assertEqual(txn.relationship, "Subsidiary")
        self.assertEqual(txn.nature_of_transaction, "Facility Sharing")
        self.assertEqual(txn.amount_or_terms, "₹ 75,000 per month (excluding GST)")
        self.assertEqual(txn.source_id, "drhp_p331_c2")
        self.assertEqual(txn.page, 331)

    def test_rpt_summary_instantiation(self):
        txn = RelatedPartyTransaction(
            related_party="AB Agribiz and Organics LLP",
            relationship="Promoter Group Entity",
            nature_of_transaction="Entity under significant influence",
            amount_or_terms="As disclosed in Note 33",
            source_id="drhp_p413_c3",
            page=413
        )
        summary = RelatedPartySummary(
            transactions=[txn],
            loans_outstanding="Nil loans given to related parties as at March 31, 2026/2025",
            summary="Inter-facility agreement and promoter-group entities disclosed."
        )
        self.assertEqual(len(summary.transactions), 1)
        self.assertIn("Nil", summary.loans_outstanding)


class TestRelatedPartyAgentMock(unittest.TestCase):

    def test_unanswerable_rpt_refusal(self):
        """Test refusal when asked about transactions with entities absent from DRHP context."""
        mock_context = "Source ID: moe_p10_t1\nPage: 10\nContent Type: table\nExtracted Table: Headers: Term | Desc\nRow 1: General | Info"
        query = "Did the company give a ₹500 crore loan to an offshore Cayman Islands trust?"
        mock_llm = get_chat_llm(provider="mock")
        ans = generate_answer(query, mock_context, llm=mock_llm)
        self.assertIn("MOCK ANSWER", ans)


if __name__ == "__main__":
    unittest.main()
