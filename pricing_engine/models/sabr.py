"""SABR stochastic volatility model, priced via the Hagan et al. (2002)
lognormal implied-volatility approximation fed into Black-76/Black-Scholes.

    dF_t = alpha_t * F_t^beta * dW1_t
    d(alpha_t) = nu * alpha_t * dW2_t,   corr(dW1, dW2) = rho
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from pricing_engine.core.black_scholes import BlackScholes
from pricing_engine.models.base import PricingModel


@dataclass
class SABRModel(PricingModel):
    F: float  # forward price
    K: float
    T: float
    alpha: float  # initial vol
    beta: float  # CEV exponent, in [0, 1]
    rho: float  # correlation, in [-1, 1]
    nu: float  # vol-of-vol
    r: float
    option_type: str = "call"

    def __post_init__(self) -> None:
        if not (0.0 <= self.beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {self.beta}")
        if not (-1.0 <= self.rho <= 1.0):
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
        if self.alpha <= 0 or self.nu < 0:
            raise ValueError("alpha must be positive and nu non-negative")

    def implied_vol(self) -> float:
        """Hagan's lognormal SABR implied volatility approximation."""
        F, K, T = self.F, self.K, self.T
        alpha, beta, rho, nu = self.alpha, self.beta, self.rho, self.nu

        if math.isclose(F, K, rel_tol=1e-9):
            fk_beta = F ** (1.0 - beta)
            term1 = alpha / fk_beta
            term2 = 1.0 + (
                ((1 - beta) ** 2 / 24.0) * (alpha**2 / fk_beta**2)
                + 0.25 * rho * beta * nu * alpha / fk_beta
                + ((2 - 3 * rho**2) / 24.0) * nu**2
            ) * T
            return term1 * term2

        fk = (F * K) ** ((1 - beta) / 2.0)
        log_fk = math.log(F / K)
        z = (nu / alpha) * fk * log_fk
        x_z = math.log((math.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

        term1 = alpha / (
            fk
            * (
                1
                + ((1 - beta) ** 2 / 24.0) * log_fk**2
                + ((1 - beta) ** 4 / 1920.0) * log_fk**4
            )
        )
        term2 = z / x_z
        term3 = 1.0 + (
            ((1 - beta) ** 2 / 24.0) * (alpha**2 / fk**2)
            + 0.25 * rho * beta * nu * alpha / fk
            + ((2 - 3 * rho**2) / 24.0) * nu**2
        ) * T
        return term1 * term2 * term3

    def price(self) -> float:
        sigma = self.implied_vol()
        return BlackScholes(self.F, self.K, self.T, self.r, sigma, self.option_type).price()

    def greeks(self) -> dict[str, float]:
        sigma = self.implied_vol()
        raw_greeks = BlackScholes(
            self.F, self.K, self.T, self.r, sigma, self.option_type
        ).all_greeks()
        # BlackScholes.all_greeks() may return numpy.float64 values from scipy.stats
        # (norm.cdf, norm.pdf). Cast to float for consistent return type.
        return {k: float(v) for k, v in raw_greeks.items()}

    @classmethod
    def calibrate(  # type: ignore[override]
        cls, market_data: Any, initial_guess: dict[str, float]
    ) -> dict[str, float]:
        """Returns the calibrated parameter dict (not a model instance) —
        see CalibrationEngine.calibrate for why: market_data here is a list
        of quotes across strikes, not a single contract, so there is no
        single model instance to construct. Callers build a model per-quote
        from the result. This intentionally narrows PricingModel.calibrate's
        declared return type; see base.PricingModel.calibrate docstring.
        """
        from pricing_engine.calibration.engine import CalibrationEngine

        return CalibrationEngine().calibrate(cls, market_data, initial_guess)
