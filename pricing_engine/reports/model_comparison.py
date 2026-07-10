"""Compares Black-Scholes, Heston, and SABR prices against a market option
chain, after calibrating Heston and SABR to that same chain. LocalVol is
omitted from the per-row comparison (it prices off a full IV surface, not a
single implied vol) and is reported separately when a full surface is
available (phase 5's vol_surface module).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pricing_engine.calibration.engine import CalibrationEngine
from pricing_engine.calibration.market_quote import MarketQuote
from pricing_engine.core.implied_vol import ImpliedVolCalculator
from pricing_engine.models.black_scholes_model import BlackScholesModel
from pricing_engine.models.heston import HestonModel
from pricing_engine.models.sabr import SABRModel


def build_comparison_table(chain: pd.DataFrame, S: float, r: float) -> pd.DataFrame:
    quotes = [
        MarketQuote(
            S=S, K=row.strike, T=row.T, r=r, option_type=row.option_type,
            market_price=row.market_price,
        )
        for row in chain.itertuples()
    ]

    # "lm" (scipy least_squares(method="lm")) requires at least as many
    # residuals (quotes) as free parameters, which real market chains
    # satisfy but small/edge-case chains may not (Heston has 5 free
    # params, SABR has 4). "lbfgsb" minimizes a scalar sum-of-squares
    # objective instead, so it has no such lower bound on the number of
    # quotes and is used here for robustness across chain sizes.
    heston_guess = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma_v": 0.3, "rho": -0.5}
    heston_params = CalibrationEngine().calibrate(HestonModel, quotes, heston_guess, method="lbfgsb")

    sabr_guess = {"alpha": 0.2, "beta": 0.5, "rho": -0.3, "nu": 0.4}
    sabr_params = CalibrationEngine().calibrate(SABRModel, quotes, sabr_guess, method="lbfgsb")

    rows = []
    for quote in quotes:
        implied = ImpliedVolCalculator().solve(
            market_price=quote.market_price, S=S, K=quote.K, T=quote.T, r=r,
            option_type=quote.option_type,
        )

        bs_price = BlackScholesModel(
            S=S, K=quote.K, T=quote.T, r=r, sigma=implied, option_type=quote.option_type,
        ).price()
        heston_price = HestonModel(
            S=S, K=quote.K, T=quote.T, r=r, option_type=quote.option_type, **heston_params,
        ).price()
        sabr_price = SABRModel(
            F=S, K=quote.K, T=quote.T, r=r, option_type=quote.option_type, **sabr_params,
        ).price()

        def pct_error(model_price: float) -> float:
            return float((model_price - quote.market_price) / quote.market_price * 100.0)

        rows.append(
            {
                "strike": quote.K,
                "market_price": quote.market_price,
                "bs_price": bs_price,
                "heston_price": heston_price,
                "sabr_price": sabr_price,
                "bs_error_pct": pct_error(bs_price),
                "heston_error_pct": pct_error(heston_price),
                "sabr_error_pct": pct_error(sabr_price),
            }
        )

    return pd.DataFrame(rows)


def write_report(chain: pd.DataFrame, S: float, r: float, out_path: str = "results/model_comparison.csv") -> None:
    table = build_comparison_table(chain, S, r)
    table.to_csv(out_path, index=False)
    print(f"Wrote {len(table)}-row model comparison to {out_path}")
    for model in ("bs", "heston", "sabr"):
        mae = np.abs(table[f"{model}_error_pct"]).mean()
        print(f"  {model.upper()} mean abs %% error vs market: {mae:.2f}%")


def _time_to_expiry_years(expiry: pd.Timestamp, as_of: datetime) -> float:
    """Convert an expiry timestamp to years-to-expiry from ``as_of``."""
    expiry_naive = expiry.to_pydatetime().replace(tzinfo=None)
    as_of_naive = as_of.replace(tzinfo=None)
    return max((expiry_naive - as_of_naive).days, 0) / 365.0


if __name__ == "__main__":
    from pricing_engine.market.market_data import get_options_chain, get_risk_free_rate, get_spot_price

    ticker = "AAPL"
    spot = get_spot_price(ticker)
    rate = get_risk_free_rate()
    raw_chain = get_options_chain(ticker)

    # get_options_chain() returns columns: expiry, option_type, strike, bid,
    # ask, last, volume, openInterest, mid_price, mid_price_source. It does
    # NOT return `T` or `market_price` directly, so both are derived here:
    # `T` from `expiry` (years from now), `market_price` from `mid_price`
    # (the bid/ask mid, falling back to lastPrice when bid/ask are
    # unusable — see market_data.get_options_chain docstring).
    as_of = datetime.now(timezone.utc)
    raw_chain = raw_chain.copy()
    raw_chain["T"] = raw_chain["expiry"].apply(lambda exp: _time_to_expiry_years(exp, as_of))
    raw_chain["market_price"] = raw_chain["mid_price"]
    raw_chain = raw_chain[
        (raw_chain["T"] > 0) & raw_chain["market_price"].notna() & (raw_chain["market_price"] > 0)
    ].reset_index(drop=True)

    write_report(raw_chain, S=spot, r=rate)
