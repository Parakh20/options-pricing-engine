"""Validate Black-Scholes model prices against real AAPL market option prices.

For each option in a live AAPL options chain, computes the Black-Scholes
price using the 30-day realized volatility as the sigma input, compares it
to the observed market mid price, and reports aggregate error statistics
(MAE, RMSE, mean percent error). Also recovers the per-option implied
volatility and assembles a volatility smile/surface.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from core.black_scholes import BlackScholes
    from core.implied_vol import ImpliedVolCalculator
    from data.market_data import get_historical_vol, get_options_chain, get_risk_free_rate, get_spot_price
except ImportError:  # pragma: no cover - fallback for direct script execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.black_scholes import BlackScholes
    from core.implied_vol import ImpliedVolCalculator
    from data.market_data import get_historical_vol, get_options_chain, get_risk_free_rate, get_spot_price

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
TICKER = "AAPL"


def _time_to_expiry_years(expiry: pd.Timestamp, as_of: datetime) -> float:
    """Convert an expiry timestamp to years-to-expiry from ``as_of``."""
    expiry_naive = expiry.to_pydatetime().replace(tzinfo=None)
    as_of_naive = as_of.replace(tzinfo=None)
    return max((expiry_naive - as_of_naive).days, 0) / 365.0


def build_validation_table(
    ticker: str = TICKER,
    n_expiries: int = 3,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fetch a live options chain and compare Black-Scholes prices to market prices.

    Returns:
        A tuple of (comparison DataFrame, summary stats dict). The
        DataFrame has one row per valid (T > 0, mid_price available)
        option, with columns: expiry, option_type, strike, T, spot,
        market_price, model_price, error, pct_error, implied_vol.
    """
    spot = get_spot_price(ticker)
    r = get_risk_free_rate()
    realized_vol = get_historical_vol(ticker, window=30)
    chain = get_options_chain(ticker, n_expiries=n_expiries)

    as_of = datetime.now(timezone.utc)
    chain = chain.copy()
    chain["T"] = chain["expiry"].apply(lambda exp: _time_to_expiry_years(exp, as_of))
    chain = chain[(chain["T"] > 0) & chain["mid_price"].notna() & (chain["mid_price"] > 0)].reset_index(drop=True)

    if chain.empty:
        raise ValueError(f"No valid priced options remain for {ticker} after filtering T>0 and mid_price>0.")

    iv_calc = ImpliedVolCalculator()

    def _row_model_and_iv(row: pd.Series) -> pd.Series:
        model_price = BlackScholes(
            S=spot, K=row["strike"], T=row["T"], r=r, sigma=realized_vol, option_type=row["option_type"]
        ).price()
        implied_vol = iv_calc.solve(
            market_price=row["mid_price"], S=spot, K=row["strike"], T=row["T"], r=r, option_type=row["option_type"]
        )
        return pd.Series({"model_price": model_price, "implied_vol": implied_vol})

    computed = chain.apply(_row_model_and_iv, axis=1)
    result = pd.concat([chain, computed], axis=1)

    result["spot"] = spot
    result["error"] = result["model_price"] - result["mid_price"]
    result["pct_error"] = (result["error"] / result["mid_price"]) * 100.0

    output_columns = [
        "expiry",
        "option_type",
        "strike",
        "T",
        "spot",
        "mid_price",
        "model_price",
        "error",
        "pct_error",
        "implied_vol",
    ]
    comparison = result[output_columns].rename(columns={"mid_price": "market_price"})

    mae = float(np.mean(np.abs(comparison["error"])))
    rmse = float(np.sqrt(np.mean(comparison["error"] ** 2)))
    mean_pct_error = float(comparison["pct_error"].mean())
    mean_iv = float(comparison["implied_vol"].dropna().mean())

    summary = {
        "n_options": len(comparison),
        "spot": spot,
        "risk_free_rate": r,
        "realized_vol": realized_vol,
        "mae": mae,
        "rmse": rmse,
        "mean_pct_error": mean_pct_error,
        "mean_implied_vol": mean_iv,
        "min_implied_vol": float(comparison["implied_vol"].dropna().min()),
        "max_implied_vol": float(comparison["implied_vol"].dropna().max()),
    }

    return comparison, summary


def build_vol_smile(comparison: pd.DataFrame) -> pd.DataFrame:
    """Pivot the comparison table into a strike x expiry implied-vol surface."""
    return comparison.pivot_table(index="strike", columns="T", values="implied_vol")


if __name__ == "__main__":
    print(f"=== Market validation smoke test ({TICKER}) ===")

    try:
        comparison_df, summary_stats = build_validation_table(TICKER, n_expiries=3)
    except (ConnectionError, TimeoutError, OSError) as network_error:
        print(f"\nNetwork access appears to be unavailable: {network_error}")
        print("Market validation requires live data and could not be exercised in this run.")
        raise SystemExit(0)

    print(f"\nOptions analyzed: {summary_stats['n_options']}")
    print(f"Spot: {summary_stats['spot']:.2f}  |  r: {summary_stats['risk_free_rate']:.4%}  |  "
          f"30d realized vol: {summary_stats['realized_vol']:.4%}")
    print(f"MAE: {summary_stats['mae']:.4f}  |  RMSE: {summary_stats['rmse']:.4f}  |  "
          f"Mean % error: {summary_stats['mean_pct_error']:.2f}%")
    print(f"Mean IV: {summary_stats['mean_implied_vol']:.4%}  "
          f"(range {summary_stats['min_implied_vol']:.4%} - {summary_stats['max_implied_vol']:.4%})")

    bias_note = "overprices" if summary_stats["mean_pct_error"] > 0 else "underprices"
    if abs(summary_stats["mean_pct_error"]) > 5.0:
        print(f"FLAG: model systematically {bias_note} options relative to market "
              f"(mean % error {summary_stats['mean_pct_error']:.2f}% exceeds +/-5% threshold).")
    else:
        print(f"No strong systematic bias detected (model {bias_note} slightly, within +/-5%).")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "market_validation.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison table to {csv_path}")

    smile = build_vol_smile(comparison_df)
    print("\nImplied volatility smile (rows=strike, cols=T):")
    print(smile.round(4))
