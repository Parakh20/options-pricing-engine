"""Live market data fetchers backed by yfinance.

This module pulls real inputs for the pricing engine: spot prices, options
chains, a risk-free rate proxy, and historical volatility. All network calls
go through ``yfinance`` (tested against version 1.3.0).

Design notes / choices made while implementing this module:

* ``get_spot_price`` uses ``Ticker.fast_info["lastPrice"]`` as the primary
  source. ``fast_info`` is a lightweight, fast-resolving accessor backed by
  the same underlying quote data as ``history(period="1d")``; in manual
  testing against AAPL and ``^NSEI`` it matched the most recent daily close
  exactly. ``Ticker.info`` is avoided here because it is a much heavier,
  slower endpoint (it scrapes a large quote-summary payload) and is more
  prone to rate limiting. If ``fast_info`` is unavailable or empty, we fall
  back to the last close from ``history(period="1d")``.
* ``get_risk_free_rate`` uses ``^IRX`` (13-week / 3-month T-bill discount
  rate). yfinance reports this ticker's close in *percentage points*
  (e.g. a close of ``3.63`` means 3.63%), so it is divided by 100 to yield a
  decimal annualized rate (e.g. ``0.0363``) suitable for use as ``r`` in the
  Black-Scholes model.
* ``get_options_chain`` falls back to ``lastPrice`` for ``mid_price`` when
  bid/ask are missing, NaN, or both zero (illiquid/far OTM contracts
  frequently report zero bid/ask on Yahoo Finance). This is documented
  explicitly in the function docstring and via an added ``mid_price_source``
  column so downstream consumers can tell which rows used a fallback.
* The NIFTY 50 index ticker on yfinance is ``^NSEI`` (verified live), not
  ``.NSEI``.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Project root is the parent of this file's parent directory (data/..).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "results" / "cached_chains"
CACHE_STALENESS_SECONDS = 3600  # 1 hour
TRADING_DAYS_PER_YEAR = 252


def get_spot_price(ticker: str) -> float:
    """Fetch the current/last spot price for ``ticker``.

    Primary source is ``Ticker.fast_info["lastPrice"]``, which is a
    lightweight accessor that resolves quickly and matches the most recent
    daily close in practice. If ``fast_info`` does not yield a usable price,
    falls back to the last close from ``history(period="1d")``.

    Args:
        ticker: yfinance ticker symbol, e.g. ``"AAPL"`` or ``"^NSEI"``.

    Returns:
        The last/current price as a float.

    Raises:
        ValueError: If no price data could be retrieved for ``ticker``,
            indicating an invalid ticker or no available market data.
    """
    if not ticker or not ticker.strip():
        raise ValueError("ticker must be a non-empty string")

    t = yf.Ticker(ticker)

    try:
        fast_info = t.fast_info
        price = fast_info.get("lastPrice") if hasattr(fast_info, "get") else getattr(fast_info, "last_price", None)
        if price is not None and not math.isnan(price):
            return float(price)
    except Exception:
        # fast_info can raise on bad tickers or transient API issues; fall
        # through to the history-based fallback rather than swallowing this.
        pass

    history = t.history(period="1d")
    if history.empty or "Close" not in history.columns:
        raise ValueError(
            f"Could not retrieve spot price for ticker '{ticker}': "
            "yfinance returned no fast_info price and no daily history. "
            "Verify the ticker symbol is correct and markets data is available."
        )

    return float(history["Close"].iloc[-1])


def _compute_mid_price(row: pd.Series) -> tuple[float, str]:
    """Compute a mid price for a single option-chain row.

    Uses ``(bid + ask) / 2`` when both bid and ask are present and strictly
    positive. Falls back to ``lastPrice`` when bid/ask are missing, NaN, or
    either is zero (a common signal of an illiquid/stale quote on Yahoo
    Finance options data).

    Args:
        row: A row from a calls/puts DataFrame with ``bid``, ``ask``, and
            ``last`` (renamed from yfinance's ``lastPrice``) columns.

    Returns:
        A tuple of ``(mid_price, source)`` where ``source`` is either
        ``"bid_ask"`` or ``"last_price_fallback"``.
    """
    bid = row.get("bid")
    ask = row.get("ask")

    bid_valid = bid is not None and not pd.isna(bid) and bid > 0
    ask_valid = ask is not None and not pd.isna(ask) and ask > 0

    if bid_valid and ask_valid:
        return (bid + ask) / 2.0, "bid_ask"

    last_price = row.get("last")
    if last_price is None or pd.isna(last_price):
        return float("nan"), "last_price_fallback"
    return float(last_price), "last_price_fallback"


def _latest_cache_file(ticker: str) -> Path | None:
    """Return the most recently modified cache file for ``ticker``, if any."""
    if not CACHE_DIR.exists():
        return None
    safe_ticker = ticker.replace("^", "").replace("/", "_")
    candidates = sorted(
        CACHE_DIR.glob(f"{safe_ticker}_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _is_cache_fresh(cache_file: Path) -> bool:
    """Check whether ``cache_file`` was modified within the staleness window."""
    age_seconds = datetime.now().timestamp() - cache_file.stat().st_mtime
    return age_seconds < CACHE_STALENESS_SECONDS


def get_options_chain(ticker: str, n_expiries: int = 3) -> pd.DataFrame:
    """Fetch the options chain (calls and puts) for the next N expiries.

    Results are cached to ``results/cached_chains/{ticker}_{timestamp}.csv``.
    If a cached file for ``ticker`` exists and is less than one hour old, it
    is loaded and returned instead of hitting the network again.

    Mid-price handling: ``mid_price = (bid + ask) / 2`` is used when both
    bid and ask are present and positive. When bid/ask is missing, NaN, or
    zero (common for illiquid or far out-of-the-money contracts), the row
    falls back to ``lastPrice`` instead, and the ``mid_price_source`` column
    records which method was used (``"bid_ask"`` or ``"last_price_fallback"``).

    Args:
        ticker: yfinance ticker symbol, e.g. ``"AAPL"``.
        n_expiries: Number of upcoming expiry dates to fetch (default 3).

    Returns:
        A single concatenated DataFrame with columns: ``expiry``,
        ``option_type``, ``strike``, ``bid``, ``ask``, ``last`` (lastPrice),
        ``volume``, ``openInterest``, ``mid_price``, ``mid_price_source``.

    Raises:
        ValueError: If ``n_expiries`` is not positive, or if yfinance
            returns no expiries / no chain data for ``ticker``.
    """
    if n_expiries <= 0:
        raise ValueError(f"n_expiries must be positive, got {n_expiries}")

    cached_file = _latest_cache_file(ticker)
    if cached_file is not None and _is_cache_fresh(cached_file):
        return pd.read_csv(cached_file, parse_dates=["expiry"])

    t = yf.Ticker(ticker)
    available_expiries = t.options
    if not available_expiries:
        raise ValueError(
            f"yfinance returned no available option expiries for ticker '{ticker}'. "
            "Verify the ticker symbol is correct and that it has listed options."
        )

    expiries_to_fetch = available_expiries[:n_expiries]
    frames: list[pd.DataFrame] = []

    for expiry in expiries_to_fetch:
        chain = t.option_chain(expiry)

        for option_type, raw_df in (("call", chain.calls), ("put", chain.puts)):
            if raw_df.empty:
                continue

            df = raw_df[["strike", "bid", "ask", "lastPrice", "volume", "openInterest"]].copy()
            df = df.rename(columns={"lastPrice": "last"})
            df["option_type"] = option_type
            df["expiry"] = expiry

            mid_results = df.apply(_compute_mid_price, axis=1, result_type="expand")
            df["mid_price"] = mid_results[0]
            df["mid_price_source"] = mid_results[1]

            frames.append(df)

    if not frames:
        raise ValueError(
            f"yfinance returned no options chain data for ticker '{ticker}' "
            f"across expiries {list(expiries_to_fetch)}."
        )

    result = pd.concat(frames, ignore_index=True)
    result["expiry"] = pd.to_datetime(result["expiry"])

    column_order = [
        "expiry",
        "option_type",
        "strike",
        "bid",
        "ask",
        "last",
        "volume",
        "openInterest",
        "mid_price",
        "mid_price_source",
    ]
    result = result[column_order]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_ticker = ticker.replace("^", "").replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_path = CACHE_DIR / f"{safe_ticker}_{timestamp}.csv"
    result.to_csv(cache_path, index=False)

    return result


def get_risk_free_rate() -> float:
    """Fetch a risk-free rate proxy from the 3-month US T-bill yield.

    Uses the ``^IRX`` ticker (CBOE 13-week T-bill rate). yfinance quotes
    this value in percentage points (e.g. a close of ``3.63`` means 3.63%),
    so the returned value is divided by 100 to produce a decimal annualized
    rate (e.g. ``0.0363``) suitable for direct use as ``r`` in Black-Scholes.

    Returns:
        The most recent decimal annualized risk-free rate.

    Raises:
        ValueError: If yfinance returns no historical data for ``^IRX``.
    """
    t = yf.Ticker("^IRX")
    history = t.history(period="5d")

    if history.empty or "Close" not in history.columns:
        raise ValueError(
            "Could not retrieve risk-free rate: yfinance returned no recent "
            "history for '^IRX' (3-month T-bill proxy)."
        )

    latest_close_percent = float(history["Close"].iloc[-1])
    return latest_close_percent / 100.0


def get_historical_vol(ticker: str, window: int = 30) -> float:
    """Compute annualized historical (realized) volatility from daily returns.

    Fetches enough daily price history to cover ``window`` trading days of
    log returns, computes the standard deviation of the most recent
    ``window`` daily log returns, and annualizes by multiplying by
    ``sqrt(252)``.

    Args:
        ticker: yfinance ticker symbol, e.g. ``"AAPL"``.
        window: Number of most recent daily returns to use (default 30).

    Returns:
        Annualized historical volatility as a decimal (e.g. ``0.25`` for 25%).

    Raises:
        ValueError: If ``window`` is not positive, or if yfinance returns
            insufficient price history for ``ticker``.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")

    # Fetch extra calendar days as a buffer for weekends/holidays so that
    # we reliably end up with at least `window` trading-day returns.
    lookback_days = window * 3 + 30
    t = yf.Ticker(ticker)
    history = t.history(period=f"{lookback_days}d")

    if history.empty or "Close" not in history.columns:
        raise ValueError(
            f"Could not retrieve price history for ticker '{ticker}' to compute "
            "historical volatility. Verify the ticker symbol is correct."
        )

    closes = history["Close"].dropna()
    if len(closes) < window + 1:
        raise ValueError(
            f"Insufficient price history for ticker '{ticker}': need at least "
            f"{window + 1} closing prices to compute a {window}-day return window, "
            f"got {len(closes)}."
        )

    log_returns = np.log(closes / closes.shift(1)).dropna()
    recent_returns = log_returns.iloc[-window:]

    daily_std = float(recent_returns.std())
    return daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)


if __name__ == "__main__":
    DEFAULT_EQUITY_TICKER = "AAPL"
    DEFAULT_INDEX_TICKER = "^NSEI"  # NIFTY 50 on yfinance (not ".NSEI")

    print("=== Market data smoke test ===")

    try:
        spot = get_spot_price(DEFAULT_EQUITY_TICKER)
        print(f"\n[{DEFAULT_EQUITY_TICKER}] Spot price: {spot:.2f}")

        chain = get_options_chain(DEFAULT_EQUITY_TICKER, n_expiries=3)
        print(f"\n[{DEFAULT_EQUITY_TICKER}] Options chain preview ({len(chain)} rows):")
        print(chain.head())

        rfr = get_risk_free_rate()
        print(f"\nRisk-free rate (^IRX proxy): {rfr:.4%}")

        hist_vol = get_historical_vol(DEFAULT_EQUITY_TICKER, window=30)
        print(f"\n[{DEFAULT_EQUITY_TICKER}] 30-day annualized historical vol: {hist_vol:.4%}")

        index_spot = get_spot_price(DEFAULT_INDEX_TICKER)
        print(f"\n[{DEFAULT_INDEX_TICKER}] Spot price: {index_spot:.2f}")

    except (ConnectionError, TimeoutError, OSError) as network_error:
        print(
            "\nNetwork access appears to be unavailable, so live data could not "
            f"be fetched: {network_error}"
        )
        print(
            "Code structure was not exercised against live data in this run. "
            "Run `python3 -c \"import ast; ast.parse(open('data/market_data.py').read())\"` "
            "to confirm the module is syntactically valid, or re-run this script "
            "with network access to perform a full smoke test."
        )
    except ValueError as data_error:
        print(f"\nData error during smoke test: {data_error}")
        raise
