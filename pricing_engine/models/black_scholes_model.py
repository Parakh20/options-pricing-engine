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
        raw_greeks = self._pricer().all_greeks()
        # BlackScholes.all_greeks() may return numpy.float64 values from scipy.stats
        # (norm.cdf, norm.pdf). Cast to float for consistent return type.
        return {k: float(v) for k, v in raw_greeks.items()}

    @classmethod
    def calibrate(
        cls, market_data: Any, initial_guess: dict[str, float]
    ) -> "BlackScholesModel":
        """Black-Scholes has one free parameter (sigma) — calibration here is
        just implied-vol inversion against a single market price, reusing
        ImpliedVolCalculator. market_data must expose S, K, T, r, option_type,
        market_price attributes.

        Raises:
            ValueError: If the implied volatility solver does not converge,
                e.g. when the market price is below intrinsic value or
                otherwise unreachable within the solver's bracket.
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
        if sigma is None:
            raise ValueError(
                "Could not calibrate BlackScholesModel: implied vol solver did not converge. "
                "This typically occurs when the market price is below intrinsic value or "
                "otherwise unreachable within the solver bracket."
            )
        return cls(
            S=market_data.S,
            K=market_data.K,
            T=market_data.T,
            r=market_data.r,
            sigma=sigma,
            option_type=market_data.option_type,
        )
