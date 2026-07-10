from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from pricing_engine.reports.model_comparison import (
    _time_to_expiry_years,
    build_comparison_table,
    write_report,
)


def test_comparison_table_has_expected_columns() -> None:
    chain = pd.DataFrame(
        {
            "strike": [95.0, 100.0, 105.0],
            "T": [1.0, 1.0, 1.0],
            "market_price": [12.0, 8.5, 5.8],
            "option_type": ["call", "call", "call"],
        }
    )
    table = build_comparison_table(chain, S=100.0, r=0.05)

    expected_cols = {
        "strike", "market_price", "bs_price", "heston_price", "sabr_price",
        "bs_error_pct", "heston_error_pct", "sabr_error_pct",
    }
    assert expected_cols.issubset(set(table.columns))
    assert len(table) == 3


def test_comparison_table_errors_are_finite() -> None:
    chain = pd.DataFrame(
        {
            "strike": [100.0],
            "T": [1.0],
            "market_price": [8.5],
            "option_type": ["call"],
        }
    )
    table = build_comparison_table(chain, S=100.0, r=0.05)
    assert table["bs_error_pct"].notna().all()


def test_write_report_writes_csv_with_expected_rows(tmp_path) -> None:
    chain = pd.DataFrame(
        {
            "strike": [95.0, 100.0],
            "T": [1.0, 1.0],
            "market_price": [12.0, 8.5],
            "option_type": ["call", "call"],
        }
    )
    out_path = tmp_path / "model_comparison.csv"

    write_report(chain, S=100.0, r=0.05, out_path=str(out_path))

    assert out_path.exists()
    written = pd.read_csv(out_path)
    assert len(written) == 2
    assert "heston_error_pct" in written.columns


def test_time_to_expiry_years_computes_positive_fraction() -> None:
    as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)
    expiry = pd.Timestamp(as_of + timedelta(days=365))
    years = _time_to_expiry_years(expiry, as_of)
    assert years == 1.0


def test_time_to_expiry_years_floors_past_expiry_at_zero() -> None:
    as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)
    expiry = pd.Timestamp(as_of - timedelta(days=30))
    years = _time_to_expiry_years(expiry, as_of)
    assert years == 0.0
