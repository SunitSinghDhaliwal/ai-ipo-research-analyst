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
from financial_agent import (
    FinancialFinding,
    FinancialExtractionResult,
    normalize_financial_concept,
    normalize_fiscal_period,
    parse_financial_number,
    extract_financial_findings_from_context,
    detect_and_execute_financial_calculation,
    detect_and_execute_comparison,
    run_financial_agent
)
from llm_generator import get_chat_llm, generate_answer


class TestFinancialCalculator(unittest.TestCase):

    def test_absolute_change(self):
        diff = absolute_change(17449.01, 22115.37)
        self.assertAlmostEqual(diff, 4666.36, places=2)

    def test_percentage_change(self):
        pct = percentage_change(17449.01, 22115.37)
        self.assertIsNotNone(pct)
        self.assertAlmostEqual(pct, 26.74, places=1)

    def test_percentage_change_zero_division(self):
        self.assertIsNone(percentage_change(0.0, 100.0))

    def test_margin(self):
        m = margin(862.22, 22170.94)
        self.assertAlmostEqual(m, 3.89, places=1)

    def test_margin_zero_division(self):
        self.assertIsNone(margin(100.0, 0.0))

    def test_calculate_growth(self):
        # Day 6 Test 2: FY2025 (18,832.99) to FY2026 (22,115.37)
        res = calculate_growth(
            current_value=22115.37,
            previous_value=18832.99,
            metric="Revenue from Operations",
            current_period="FY2026",
            previous_period="FY2025"
        )
        self.assertEqual(res["tool"], "calculate_growth")
        self.assertEqual(res["metric"], "Revenue from Operations")
        self.assertAlmostEqual(res["absolute_change"], 3282.38, places=2)
        self.assertAlmostEqual(res["percentage_growth"], 17.43, places=1)
        self.assertIn("+17.43%", res["formatted_growth"])
        self.assertIn("Revenue from Operations increased", res["summary"])

    def test_calculate_margin(self):
        # Day 6 Test 3: EBITDA (1,321.35) / Revenue (22,115.37)
        res = calculate_margin(
            numerator=1321.35,
            denominator=22115.37,
            margin_name="EBITDA Margin",
            numerator_name="EBITDA",
            denominator_name="Revenue from Operations",
            period="FY2026"
        )
        self.assertEqual(res["tool"], "calculate_margin")
        self.assertAlmostEqual(res["margin_percentage"], 5.97, places=2)
        self.assertEqual(res["formatted_margin"], "5.97%")
        self.assertIn("5.97%", res["summary"])

    def test_calculate_ratio(self):
        # Debt (2,853.88) / Equity (4,659.05) -> 0.61x
        res = calculate_ratio(
            numerator=2853.88,
            denominator=4659.05,
            ratio_name="Debt-to-Equity Ratio",
            numerator_name="Total Borrowings",
            denominator_name="Total Equity"
        )
        self.assertEqual(res["tool"], "calculate_ratio")
        self.assertAlmostEqual(res["ratio"], 0.6125, places=2)
        self.assertEqual(res["formatted_ratio"], "0.61x")

        # Current Assets (6,620.28) / Current Liabilities (3,253.02) -> 2.04x
        cr = calculate_current_ratio(6620.28, 3253.02)
        self.assertAlmostEqual(cr["ratio"], 2.0351, places=2)
        self.assertEqual(cr["formatted_ratio"], "2.04x")

    def test_compare_financial_years(self):
        # Day 6 Test 4: Multi-year comparison
        rev_trend = {
            "FY2024": 17449.01,
            "FY2025": 18832.99,
            "FY2026": 22115.37
        }
        res = compare_financial_years(rev_trend, metric="Revenue from Operations")
        self.assertEqual(res["tool"], "compare_financial_years")
        self.assertEqual(len(res["yoy_changes"]), 2)
        # FY2024 to FY2025: (18832.99 - 17449.01) / 17449.01 * 100 = 7.93%
        self.assertAlmostEqual(res["yoy_changes"][0]["percentage_growth"], 7.93, places=1)
        # FY2025 to FY2026: (22115.37 - 18832.99) / 18832.99 * 100 = 17.43%
        self.assertAlmostEqual(res["yoy_changes"][1]["percentage_growth"], 17.43, places=1)
        # Overall FY2024 to FY2026: 26.74%
        self.assertAlmostEqual(res["overall_percentage"], 26.74, places=1)

    def test_convenience_wrappers(self):
        rev_g = calculate_revenue_growth(22115.37, 18832.99)
        self.assertAlmostEqual(rev_g["percentage_growth"], 17.43, places=1)

        pat_g = calculate_profit_growth(862.22, 870.81)
        self.assertAlmostEqual(pat_g["percentage_growth"], -0.99, places=1)

        ebitda_m = calculate_ebitda_margin(1321.35, 22115.37)
        self.assertAlmostEqual(ebitda_m["margin_percentage"], 5.97, places=2)

        pat_m = calculate_net_profit_margin(862.22, 22170.94)
        self.assertAlmostEqual(pat_m["margin_percentage"], 3.89, places=2)

        de_r = calculate_debt_to_equity(2853.88, 4659.05)
        self.assertAlmostEqual(de_r["ratio"], 0.61, places=2)

    def test_calculate_period_comparison(self):
        res = calculate_period_comparison(
            metric="Revenue from Operations",
            base_period="FY2024",
            base_value=17449.01,
            target_period="FY2026",
            target_value=22115.37
        )
        self.assertEqual(res["metric"], "Revenue from Operations")
        self.assertAlmostEqual(res["absolute_change"], 4666.36, places=2)
        self.assertAlmostEqual(res["percentage_change"], 26.74, places=1)
        self.assertIn("Revenue from Operations increased", res["explanation"])


class TestFinancialExtraction(unittest.TestCase):

    def test_normalize_financial_concept(self):
        # Revenue synonyms
        self.assertEqual(normalize_financial_concept("Revenue from Operations (net)"), "revenue")
        self.assertEqual(normalize_financial_concept("Revenue from operations"), "revenue")
        self.assertEqual(normalize_financial_concept("Total Revenue (I)"), "total_income")
        self.assertEqual(normalize_financial_concept("Total Income (C)"), "total_income")

        # EBITDA synonyms
        self.assertEqual(normalize_financial_concept("EBIDTA"), "ebitda")
        self.assertEqual(normalize_financial_concept("EBITDA [H= (D+E+F+G-B)]"), "ebitda")

        # PAT synonyms
        self.assertEqual(normalize_financial_concept("Restated profit for the period/year (V) (III-IV)"), "pat")
        self.assertEqual(normalize_financial_concept("Restated profit after tax"), "pat")
        self.assertEqual(normalize_financial_concept("Profit after tax (PAT)"), "pat")

        # Balance sheet synonyms
        self.assertEqual(normalize_financial_concept("Total current assets"), "current_assets")
        self.assertEqual(normalize_financial_concept("Total Current Liabilities"), "current_liabilities")
        self.assertEqual(normalize_financial_concept("Total borrowings (A)"), "debt")
        self.assertEqual(normalize_financial_concept("Total Equity (B)"), "equity")
        self.assertEqual(normalize_financial_concept("Cash and cash equivalents"), "cash")

    def test_normalize_fiscal_period(self):
        self.assertEqual(normalize_fiscal_period("Fiscal 2026"), "FY2026")
        self.assertEqual(normalize_fiscal_period("FY26"), "FY2026")
        self.assertEqual(normalize_fiscal_period("2025"), "FY2025")
        self.assertEqual(normalize_fiscal_period("March 31, 2024"), "FY2024")

    def test_parse_financial_number(self):
        self.assertEqual(parse_financial_number("22,115.37"), 22115.37)
        self.assertEqual(parse_financial_number("₹ 1,321.35"), 1321.35)
        self.assertEqual(parse_financial_number("5.97%"), 5.97)
        self.assertEqual(parse_financial_number("(69.47)"), -69.47)
        self.assertIsNone(parse_financial_number("-"))
        self.assertIsNone(parse_financial_number("[●]"))
        self.assertIsNone(parse_financial_number("Nil"))

    def test_extract_findings_from_mock_context(self):
        mock_context = (
            "--- [SOURCE 1] ---\n"
            "Source ID: moe_p86_t1\n"
            "Page: 86\n"
            "Content Type: table\n"
            "Extracted Table:\n"
            "Headers: Particulars | Fiscal 2026 | Fiscal 2025 | Fiscal 2024\n"
            "Row 1: Revenue from operations (net) | 22,115.37 | 18,832.99 | 17,449.01\n"
            "Row 2: Restated profit for the period/year | 862.22 | 870.81 | 173.52\n"
        )
        findings = extract_financial_findings_from_context(mock_context)
        self.assertEqual(len(findings), 6)

        rev_2026 = [f for f in findings if f.concept == "revenue" and f.period == "FY2026"]
        self.assertEqual(len(rev_2026), 1)
        self.assertEqual(rev_2026[0].value, 22115.37)
        self.assertEqual(rev_2026[0].page, 86)
        self.assertEqual(rev_2026[0].source_id, "moe_p86_t1")

        pat_2024 = [f for f in findings if f.concept == "pat" and f.period == "FY2024"]
        self.assertEqual(len(pat_2024), 1)
        self.assertEqual(pat_2024[0].value, 173.52)


class TestFinancialCalculationDispatch(unittest.TestCase):

    def setUp(self):
        # Standard verified findings across statements
        self.findings = [
            FinancialFinding(metric="Revenue from Operations", concept="revenue", period="FY2024", value=17449.01, source_id="moe_p86_t1", page=86),
            FinancialFinding(metric="Revenue from Operations", concept="revenue", period="FY2025", value=18832.99, source_id="moe_p86_t1", page=86),
            FinancialFinding(metric="Revenue from Operations", concept="revenue", period="FY2026", value=22115.37, source_id="moe_p86_t1", page=86),
            FinancialFinding(metric="EBITDA", concept="ebitda", period="FY2026", value=1321.35, source_id="moe_p444_t1", page=444),
            FinancialFinding(metric="Profit After Tax (PAT)", concept="pat", period="FY2024", value=173.52, source_id="moe_p86_t1", page=86),
            FinancialFinding(metric="Profit After Tax (PAT)", concept="pat", period="FY2025", value=870.81, source_id="moe_p86_t1", page=86),
            FinancialFinding(metric="Profit After Tax (PAT)", concept="pat", period="FY2026", value=862.22, source_id="moe_p86_t1", page=86),
            FinancialFinding(metric="Total Borrowings (Debt)", concept="debt", period="FY2026", value=2853.88, source_id="moe_p473_t1", page=473),
            FinancialFinding(metric="Total Equity", concept="equity", period="FY2026", value=4659.05, source_id="moe_p473_t1", page=473),
            FinancialFinding(metric="Total Current Assets", concept="current_assets", period="FY2026", value=6620.28, source_id="moe_p85_t1", page=85),
            FinancialFinding(metric="Total Current Liabilities", concept="current_liabilities", period="FY2026", value=3253.02, source_id="moe_p85_t1", page=85),
        ]

    def test_dispatch_revenue_growth_fy25_fy26(self):
        # Day 6 Test 2
        q = "What was the revenue growth from FY2025 to FY2026?"
        res = detect_and_execute_financial_calculation(q, self.findings)
        self.assertTrue(res.calculation_needed)
        self.assertFalse(res.missing_data)
        calc = res.calculation_result
        self.assertEqual(calc["tool"], "calculate_growth")
        self.assertAlmostEqual(calc["previous_value"], 18832.99, places=2)
        self.assertAlmostEqual(calc["current_value"], 22115.37, places=2)
        self.assertAlmostEqual(calc["percentage_growth"], 17.43, places=1)

    def test_dispatch_ebitda_margin_fy26(self):
        # Day 6 Test 3
        q = "What was the EBITDA margin in FY2026?"
        res = detect_and_execute_financial_calculation(q, self.findings)
        self.assertTrue(res.calculation_needed)
        self.assertFalse(res.missing_data)
        calc = res.calculation_result
        self.assertEqual(calc["tool"], "calculate_margin")
        self.assertAlmostEqual(calc["numerator"], 1321.35, places=2)
        self.assertAlmostEqual(calc["denominator"], 22115.37, places=2)
        self.assertAlmostEqual(calc["margin_percentage"], 5.97, places=2)

    def test_dispatch_multi_year_trend(self):
        # Day 6 Test 4
        q = "How did revenue change over FY2024, FY2025 and FY2026?"
        res = detect_and_execute_financial_calculation(q, self.findings)
        self.assertTrue(res.calculation_needed)
        calc = res.calculation_result
        self.assertEqual(calc["tool"], "compare_financial_years")
        self.assertEqual(len(calc["yoy_changes"]), 2)
        self.assertAlmostEqual(calc["overall_percentage"], 26.74, places=1)

    def test_dispatch_debt_to_equity(self):
        q = "What is the company's debt-to-equity ratio?"
        res = detect_and_execute_financial_calculation(q, self.findings)
        self.assertTrue(res.calculation_needed)
        calc = res.calculation_result
        self.assertEqual(calc["tool"], "calculate_ratio")
        self.assertAlmostEqual(calc["ratio"], 0.61, places=2)

    def test_dispatch_current_ratio(self):
        q = "What is the current ratio?"
        res = detect_and_execute_financial_calculation(q, self.findings)
        self.assertTrue(res.calculation_needed)
        calc = res.calculation_result
        self.assertEqual(calc["tool"], "calculate_ratio")
        self.assertAlmostEqual(calc["ratio"], 2.04, places=2)

    def test_missing_data_safe_refusal(self):
        # Day 6 Test 5: Missing metric
        q = "What was the research and development expense growth between FY2025 and FY2026?"
        res = detect_and_execute_financial_calculation(q, self.findings)
        # Since r&d is missing from findings, it should flag missing_data=True
        self.assertTrue(res.missing_data)
        self.assertIn("could not calculate", res.missing_reason)


class TestFinancialAgentMock(unittest.TestCase):

    def test_unanswerable_refusal(self):
        """Test 5: Refusal on missing financial metric without hallucination."""
        mock_context = "Source ID: moe_p10_t1\nPage: 10\nContent Type: table\nExtracted Table: Headers: Term | Desc\nRow 1: General | Info"
        query = "What was the company's research and development expense for space technology in FY2026?"
        mock_llm = get_chat_llm(provider="mock")
        ans = generate_answer(query, mock_context, llm=mock_llm)
        self.assertIn("MOCK ANSWER", ans)


if __name__ == "__main__":
    unittest.main()
