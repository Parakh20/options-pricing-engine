"""Market quote container used as calibration input across models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketQuote:
    S: float
    K: float
    T: float
    r: float
    option_type: str
    market_price: float
