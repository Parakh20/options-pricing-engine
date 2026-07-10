"""PricingModel adapter around the existing closed-form Black-Scholes pricer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pricing_engine.core.black_scholes import BlackScholes
from pricing_engine.models.base import PricingModel


@dataclass
class BlackScholesModel(PricingModel):
    S: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: str = "call"

    def _pricer(self) -> BlackScholes:
        return BlackScholes(self.S, self.K, self.T, self.r, self.sigma, self.option_type)

    def price(self) -> float:
        return self._pricer().price()

    def greeks(self) -> dict[str, float]:
        return self._pricer().all_greeks()

    @classmethod
    def calibrate(
        cls, market_data: Any, initial_guess: dict[str, float]
    ) -> "BlackScholesModel":
        """Black-Scholes has one free parameter (sigma) — calibration here is
        just implied-vol inversion against a single market price, reusing
        ImpliedVolCalculator. market_data must expose S, K, T, r, option_type,
        market_price attributes.
        """
        from pricing_engine.core.implied_vol import ImpliedVolCalculator

        calc = ImpliedVolCalculator()
        sigma = calc.solve(
            market_price=market_data.market_price,
            S=market_data.S,
            K=market_data.K,
            T=market_data.T,
            r=market_data.r,
            option_type=market_data.option_type,
        )
        return cls(
            S=market_data.S,
            K=market_data.K,
            T=market_data.T,
            r=market_data.r,
            sigma=sigma,
            option_type=market_data.option_type,
        )
