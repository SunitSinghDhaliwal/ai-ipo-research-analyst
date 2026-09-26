import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "retrieval"))
sys.path.insert(0, str(ROOT / "src" / "generation"))

from context_builder import build_context, extract_source_metadata
from llm_generator import generate_answer, get_chat_llm


class TestDay4Generation(unittest.TestCase):

    def test_extract_source_metadata_text(self):
        mock_item = {
            "document_id": "drhp_p10_c1",
            "document": {
                "text": "Sample text chunk content.",
                "metadata": {
                    "chunk_id": "drhp_p10_c1",
                    "company": "Maharashtra Oil Extractions Limited",
                    "document_type": "DRHP",
                    "page": 10,
                    "content_type": "text",
                    "section_path": "SECTION I > DEFINITIONS"
                }
            },
            "rrf_score": 0.02,
            "rerank_score": 0.95
        }

        meta = extract_source_metadata(mock_item)
        self.assertEqual(meta["source_id"], "drhp_p10_c1")
        self.assertEqual(meta["page"], 10)
        self.assertEqual(meta["content_type"], "text")
        self.assertEqual(meta["text"], "Sample text chunk content.")
        self.assertEqual(meta["rerank_score"], 0.95)

    def test_extract_source_metadata_table(self):
        mock_item = {
            "document_id": "moe_p272_t1",
            "document": {
                "text": "Headers: Col1 | Col2\nRow 1: Val1 | Val2",
                "metadata": {
                    "table_id": "moe_p272_t1",
                    "company": "Maharashtra Oil Extractions Limited",
                    "document_type": "DRHP",
                    "page": 272,
                    "content_type": "table",
                    "section_path": "SECTION IV > FINANCIALS"
                }
            },
            "rrf_score": 0.03
        }

        meta = extract_source_metadata(mock_item)
        self.assertEqual(meta["source_id"], "moe_p272_t1")
        self.assertEqual(meta["page"], 272)
        self.assertEqual(meta["content_type"], "table")
        self.assertIn("Headers: Col1", meta["text"])

    def test_build_context_formatting(self):
        items = [
            {
                "document_id": "drhp_p1_c1",
                "document": {
                    "text": "Company registered office text.",
                    "metadata": {
                        "chunk_id": "drhp_p1_c1",
                        "company": "Maharashtra Oil Extractions Limited",
                        "document_type": "DRHP",
                        "page": 1,
                        "content_type": "text",
                        "section_path": "GENERAL"
                    }
                }
            }
        ]

        ctx, meta = build_context(items)
        self.assertIn("--- [SOURCE 1] ---", ctx)
        self.assertIn("Source ID: drhp_p1_c1", ctx)
        self.assertIn("Page: 1", ctx)
        self.assertIn("Company registered office text.", ctx)
        self.assertEqual(len(meta), 1)

    def test_modular_mock_llm(self):
        mock_llm = get_chat_llm(provider="mock")
        ans = generate_answer(
            query="What is the revenue?",
            context="Sample context",
            llm=mock_llm
        )
        self.assertIn("MOCK ANSWER", ans)


if __name__ == "__main__":
    unittest.main()
