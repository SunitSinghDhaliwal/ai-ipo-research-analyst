"""
Deterministic Financial Calculation Utility for IPO Prospectus Analysis.
Provides pure Python arithmetic functions to eliminate LLM calculation hallucinations.
All calculations are deterministic, typed, and provide full audit trails.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# --------------------------------------------------
# Primitive Arithmetic & Formatting Helpers
# --------------------------------------------------

def absolute_change(old: float, new: float) -> float:
    """
    Calculate absolute difference between two financial periods.
    Formula: new - old
    """
    return round(float(new) - float(old), 4)


def percentage_change(old: float, new: float) -> Optional[float]:
    """
    Calculate percentage growth/change between two financial periods.
    Formula: ((new - old) / old) * 100.0
    Returns None if old is zero.
    """
    old_f = float(old)
    new_f = float(new)
    if old_f == 0.0:
        return None
    pct = ((new_f - old_f) / abs(old_f)) * 100.0
    return round(pct, 4)


def margin(part: float, whole: float) -> Optional[float]:
    """
    Calculate percentage margin (e.g., PAT Margin, EBITDA Margin).
    Formula: (part / whole) * 100.0
    Returns None if whole is zero.
    """
    whole_f = float(whole)
    if whole_f == 0.0:
        return None
    m = (float(part) / whole_f) * 100.0
    return round(m, 4)


def format_currency(value: float, unit: str = "₹ million") -> str:
    """Format numerical values with standard comma separation."""
    return f"{value:,.2f} {unit}".strip()


def format_percentage(value: Optional[float], include_sign: bool = True) -> str:
    """Format floating point percentages cleanly."""
    if value is None:
        return "N/A (zero base)"
    if include_sign and value > 0:
        return f"+{value:.2f}%"
    return f"{value:.2f}%"


def format_ratio(value: Optional[float]) -> str:
    """Format numerical ratios cleanly (e.g. 0.61x)."""
    if value is None:
        return "N/A (zero denominator)"
    return f"{value:.2f}x"


# --------------------------------------------------
# Core Deterministic Financial Tools (Day 6)
# --------------------------------------------------

def calculate_growth(
    current_value: float,
    previous_value: float,
    metric: str = "Financial Metric",
    current_period: str = "Current Period",
    previous_period: str = "Previous Period",
    unit: str = "₹ million"
) -> Dict[str, Any]:
    """
    Calculate deterministic growth between two financial periods.
    Formula: growth = (current - previous) / previous * 100
    """
    cur_f = float(current_value)
    prev_f = float(previous_value)
    abs_chg = round(cur_f - prev_f, 4)
    pct_chg = percentage_change(prev_f, cur_f)

    direction = "increased" if abs_chg > 0 else "decreased" if abs_chg < 0 else "remained flat"

    summary = (
        f"{metric} {direction} by {abs(abs_chg):,.2f} {unit} ({format_percentage(pct_chg)}) "
        f"from {prev_f:,.2f} {unit} in {previous_period} to {cur_f:,.2f} {unit} in {current_period}."
    )
    formula = f"(({cur_f:,.2f} - {prev_f:,.2f}) / {prev_f:,.2f}) × 100 = {format_percentage(pct_chg)}"

    return {
        "tool": "calculate_growth",
        "metric": metric,
        "previous_period": previous_period,
        "previous_value": prev_f,
        "current_period": current_period,
        "current_value": cur_f,
        "absolute_change": abs_chg,
        "percentage_growth": pct_chg,
        "unit": unit,
        "formatted_growth": format_percentage(pct_chg),
        "formula": formula,
        "summary": summary
    }


def calculate_margin(
    numerator: float,
    denominator: float,
    margin_name: str = "Margin",
    numerator_name: str = "Numerator",
    denominator_name: str = "Revenue",
    period: Optional[str] = None,
    unit: str = "₹ million"
) -> Dict[str, Any]:
    """
    Calculate deterministic margin percentage.
    Formula: margin = numerator / denominator * 100
    """
    num_f = float(numerator)
    den_f = float(denominator)
    m = margin(num_f, den_f)

    period_str = f" in {period}" if period else ""
    summary = (
        f"{margin_name}{period_str} was {format_percentage(m, include_sign=False)} "
        f"based on {numerator_name} of {num_f:,.2f} {unit} and {denominator_name} of {den_f:,.2f} {unit}."
    )
    formula = f"({num_f:,.2f} / {den_f:,.2f}) × 100 = {format_percentage(m, include_sign=False)}"

    return {
        "tool": "calculate_margin",
        "margin_name": margin_name,
        "numerator_name": numerator_name,
        "numerator": num_f,
        "denominator_name": denominator_name,
        "denominator": den_f,
        "period": period,
        "margin_percentage": m,
        "formatted_margin": format_percentage(m, include_sign=False),
        "unit": unit,
        "formula": formula,
        "summary": summary
    }


def calculate_ratio(
    numerator: float,
    denominator: float,
    ratio_name: str = "Ratio",
    numerator_name: str = "Numerator",
    denominator_name: str = "Denominator",
    period: Optional[str] = None,
    unit: str = "₹ million"
) -> Dict[str, Any]:
    """
    Calculate deterministic financial ratio.
    Formula: ratio = numerator / denominator
    """
    num_f = float(numerator)
    den_f = float(denominator)
    r = round(num_f / den_f, 4) if den_f != 0.0 else None

    period_str = f" in {period}" if period else ""
    summary = (
        f"{ratio_name}{period_str} is {format_ratio(r)} "
        f"({numerator_name}: {num_f:,.2f} {unit}, {denominator_name}: {den_f:,.2f} {unit})."
    )
    formula = f"{num_f:,.2f} / {den_f:,.2f} = {format_ratio(r)}"

    return {
        "tool": "calculate_ratio",
        "ratio_name": ratio_name,
        "numerator_name": numerator_name,
        "numerator": num_f,
        "denominator_name": denominator_name,
        "denominator": den_f,
        "period": period,
        "ratio": r,
        "formatted_ratio": format_ratio(r),
        "unit": unit,
        "formula": formula,
        "summary": summary
    }


def compare_financial_years(
    values: Union[Dict[str, float], Sequence[Tuple[str, float]]],
    metric: str = "Financial Metric",
    unit: str = "₹ million"
) -> Dict[str, Any]:
    """
    Perform generic year-over-year comparison across multiple fiscal periods.
    Accepts a dictionary of {period: value} or list of (period, value) tuples.
    Calculates consecutive YoY growth and overall change.
    """
    if isinstance(values, dict):
        items = list(values.items())
    else:
        items = list(values)

    if not items:
        return {
            "tool": "compare_financial_years",
            "metric": metric,
            "periods": [],
            "yearly_data": [],
            "yoy_changes": [],
            "overall_change": None,
            "overall_percentage": None,
            "summary": "No data provided for comparison."
        }

    # Helper to sort periods chronologically if they contain years
    def sort_key(item: Tuple[str, float]) -> str:
        return item[0]

    # Preserve order if already sorted, or sort by period string
    periods_data = [
        {"period": p, "value": float(v), "formatted": f"{float(v):,.2f} {unit}"}
        for p, v in items
    ]

    yoy_changes = []
    for i in range(1, len(periods_data)):
        prev = periods_data[i - 1]
        curr = periods_data[i]
        diff = round(curr["value"] - prev["value"], 4)
        growth_pct = percentage_change(prev["value"], curr["value"])
        yoy_changes.append({
            "from_period": prev["period"],
            "to_period": curr["period"],
            "base_value": prev["value"],
            "target_value": curr["value"],
            "absolute_change": diff,
            "percentage_growth": growth_pct,
            "formatted_growth": format_percentage(growth_pct),
            "formula": f"(({curr['value']:,.2f} - {prev['value']:,.2f}) / {prev['value']:,.2f}) × 100 = {format_percentage(growth_pct)}"
        })

    # Overall change
    first = periods_data[0]
    last = periods_data[-1]
    overall_diff = round(last["value"] - first["value"], 4)
    overall_pct = percentage_change(first["value"], last["value"])

    summary_lines = [f"{metric} Multi-Period Trend:"]
    for d in periods_data:
        summary_lines.append(f"  - {d['period']}: {d['formatted']}")
    if yoy_changes:
        summary_lines.append("Year-over-Year Growth:")
        for y in yoy_changes:
            summary_lines.append(
                f"  - {y['from_period']} to {y['to_period']}: {y['absolute_change']:+,.2f} {unit} ({y['formatted_growth']})"
            )
        summary_lines.append(
            f"Overall Change ({first['period']} to {last['period']}): {overall_diff:+,.2f} {unit} ({format_percentage(overall_pct)})"
        )

    return {
        "tool": "compare_financial_years",
        "metric": metric,
        "periods": [d["period"] for d in periods_data],
        "yearly_data": periods_data,
        "yoy_changes": yoy_changes,
        "overall_change": overall_diff,
        "overall_percentage": overall_pct,
        "formatted_overall_growth": format_percentage(overall_pct),
        "unit": unit,
        "summary": "\n".join(summary_lines)
    }


# --------------------------------------------------
# Named Convenience Wrappers (At Minimum Day 6 Requirements)
# --------------------------------------------------

def calculate_revenue_growth(
    current_revenue: float,
    previous_revenue: float,
    current_period: str = "FY2026",
    previous_period: str = "FY2025",
    unit: str = "₹ million"
) -> Dict[str, Any]:
    """Convenience wrapper for Revenue Growth."""
    return calculate_growth(
        current_value=current_revenue,
        previous_value=previous_revenue,
        metric="Revenue from Operations",
        current_period=current_period,
        previous_period=previous_period,
        unit=unit
    )


def calculate_profit_growth(
    current_profit: float,
    previous_profit: float,
    current_period: str = "FY2026",
    previous_period: str = "FY2025",
    unit: str = "₹ million"
) -> Dict[str, Any]:
    """Convenience wrapper for Net Profit (PAT) Growth."""
    return calculate_growth(
        current_value=current_profit,
        previous_value=previous_profit,
        metric="Profit After Tax (PAT)",
        current_period=current_period,
        previous_period=previous_period,
        unit=unit
    )


def calculate_ebitda_margin(
    ebitda: float,
    revenue: float,
    period: Optional[str] = "FY2026",
    unit: str = "₹ million"
) -> Dict[str, Any]:
    """Convenience wrapper for EBITDA Margin."""
    return calculate_margin(
        numerator=ebitda,
        denominator=revenue,
        margin_name="EBITDA Margin",
        numerator_name="EBITDA",
        denominator_name="Revenue from Operations",
        period=period,
        unit=unit
    )


def calculate_net_profit_margin(
    pat: float,
    revenue: float,
    period: Optional[str] = "FY2026",
    unit: str = "₹ million"
) -> Dict[str, Any]:
    """Convenience wrapper for Net Profit (PAT) Margin."""
    return calculate_margin(
        numerator=pat,
        denominator=revenue,
        margin_name="PAT Margin",
        numerator_name="Profit After Tax (PAT)",
        denominator_name="Revenue from Operations",
        period=period,
        unit=unit
    )


def calculate_debt_to_equity(
    debt: float,
    equity: float,
    period: Optional[str] = None,
    unit: str = "₹ million"
) -> Dict[str, Any]:
    """Convenience wrapper for Debt-to-Equity Ratio."""
    return calculate_ratio(
        numerator=debt,
        denominator=equity,
        ratio_name="Debt-to-Equity Ratio",
        numerator_name="Total Borrowings (Debt)",
        denominator_name="Total Equity",
        period=period,
        unit=unit
    )


def calculate_current_ratio(
    current_assets: float,
    current_liabilities: float,
    period: Optional[str] = None,
    unit: str = "₹ million"
) -> Dict[str, Any]:
    """Convenience wrapper for Current Ratio."""
    return calculate_ratio(
        numerator=current_assets,
        denominator=current_liabilities,
        ratio_name="Current Ratio",
        numerator_name="Total Current Assets",
        denominator_name="Total Current Liabilities",
        period=period,
        unit=unit
    )


def calculate_period_comparison(
    metric: str,
    base_period: str,
    base_value: float,
    target_period: str,
    target_value: float,
    unit: str = "₹ million"
) -> Dict[str, Any]:
    """
    Backward-compatible period comparison helper.
    """
    res = calculate_growth(
        current_value=target_value,
        previous_value=base_value,
        metric=metric,
        current_period=target_period,
        previous_period=base_period,
        unit=unit
    )
    explanation = (
        f"{res['summary']}\n\n"
        f"Calculation Details:\n"
        f"  Absolute Change  : {target_value:,.2f} - {base_value:,.2f} = {res['absolute_change']:,.2f} {unit}\n"
        f"  Percentage Change: {res['formula']}"
    )
    return {
        "metric": metric,
        "base_period": base_period,
        "base_value": base_value,
        "target_period": target_period,
        "target_value": target_value,
        "absolute_change": res["absolute_change"],
        "percentage_change": res["percentage_growth"],
        "unit": unit,
        "explanation": explanation
    }


if __name__ == "__main__":
    # Self-test all tools
    print("Testing calculate_growth:")
    print(calculate_revenue_growth(22115.37, 18832.99)["summary"])

    print("\nTesting calculate_margin:")
    print(calculate_ebitda_margin(1321.35, 22115.37)["summary"])

    print("\nTesting calculate_ratio:")
    print(calculate_debt_to_equity(2853.88, 4659.05)["summary"])
    print(calculate_current_ratio(6620.28, 3253.02)["summary"])

    print("\nTesting compare_financial_years:")
    rev_trend = {
        "FY2024": 17449.01,
        "FY2025": 18832.99,
        "FY2026": 22115.37
    }
    print(compare_financial_years(rev_trend, metric="Revenue from Operations")["summary"])
