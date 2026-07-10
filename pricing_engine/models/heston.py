"""Heston (1993) stochastic volatility model, priced via the semi-analytic
characteristic-function formula (Gatheral / Albrecher "Little Trap" form to
avoid branch-cut discontinuities in the complex logarithm).

    dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW1_t
    dv_t = kappa * (theta - v_t) * dt + sigma_v * sqrt(v_t) * dW2_t
    corr(dW1, dW2) = rho
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import quad

from pricing_engine.models.base import PricingModel


@dataclass
class HestonModel(PricingModel):
    S: float
    K: float
    T: float
    r: float
    v0: float  # initial variance
    kappa: float  # mean-reversion speed
    theta: float  # long-run variance
    sigma_v: float  # vol of vol
    rho: float  # correlation
    option_type: str = "call"

    def __post_init__(self) -> None:
        if self.v0 <= 0 or self.kappa <= 0 or self.theta <= 0 or self.sigma_v < 0:
            raise ValueError("v0, kappa, theta must be positive; sigma_v non-negative")
        if not (-1.0 <= self.rho <= 1.0):
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")

    def _char_func(self, phi: complex, j: int) -> complex:
        """Heston characteristic function, "Little Trap" formulation.

        j=1 for the P1 (delta-adjacent) probability, j=2 for P2 (risk-neutral).
        """
        S, T, r = self.S, self.T, self.r
        kappa, theta, sigma_v, rho, v0 = self.kappa, self.theta, self.sigma_v, self.rho, self.v0

        u = 0.5 if j == 1 else -0.5
        b = kappa - rho * sigma_v if j == 1 else kappa

        a = kappa * theta
        x = cmath.log(S)

        d = cmath.sqrt((rho * sigma_v * phi * 1j - b) ** 2 - sigma_v**2 * (2 * u * phi * 1j - phi**2))
        # "Little Trap" (Albrecher, Mayer, Schoutens, Tistaert 2007): use the
        # reciprocal of the classical Heston g, i.e. flip the sign on d in
        # every (b - rho*sigma_v*phi*i (+/-) d) term. The naive +d form is
        # numerically unstable (g blows up to ~1e8+ and the quad integrand
        # diverges/overflows) because it crosses a branch cut of the complex
        # logarithm; the -d/reciprocal-g form stays bounded for all phi.
        g = (b - rho * sigma_v * phi * 1j - d) / (b - rho * sigma_v * phi * 1j + d)

        exp_dt = cmath.exp(-d * T)
        c = r * phi * 1j * T + (a / sigma_v**2) * (
            (b - rho * sigma_v * phi * 1j - d) * T - 2 * cmath.log((1 - g * exp_dt) / (1 - g))
        )
        D = (b - rho * sigma_v * phi * 1j - d) / sigma_v**2 * ((1 - exp_dt) / (1 - g * exp_dt))

        return cmath.exp(c + D * v0 + 1j * phi * x)

    def _probability(self, j: int) -> float:
        def integrand(phi: float) -> float:
            if phi == 0.0:
                return 0.0
            numerator = cmath.exp(-1j * phi * np.log(self.K)) * self._char_func(phi, j)
            return (numerator / (1j * phi)).real

        integral, _ = quad(integrand, 1e-10, 100.0, limit=200)
        return 0.5 + integral / np.pi

    def price(self) -> float:
        S, K, T, r = self.S, self.K, self.T, self.r
        P1 = self._probability(1)
        P2 = self._probability(2)
        call = S * P1 - K * np.exp(-r * T) * P2
        if self.option_type == "call":
            return float(call)
        # put via put-call parity
        return float(call - S + K * np.exp(-r * T))

    def greeks(self) -> dict[str, float]:
        """Finite-difference delta and vega (v0 bump) — analytic Greeks for
        Heston require differentiating the characteristic function integral,
        deferred to the AD-Greeks work in phase 2.
        """
        bump_s = self.S * 1e-4
        up = HestonModel(**{**self.__dict__, "S": self.S + bump_s})
        down = HestonModel(**{**self.__dict__, "S": self.S - bump_s})
        delta = (up.price() - down.price()) / (2 * bump_s)

        bump_v = self.v0 * 1e-2
        up_v = HestonModel(**{**self.__dict__, "v0": self.v0 + bump_v})
        down_v = HestonModel(**{**self.__dict__, "v0": self.v0 - bump_v})
        vega = (up_v.price() - down_v.price()) / (2 * bump_v) * 0.01

        return {"delta": delta, "vega": vega}

    @classmethod
    def calibrate(
        cls, market_data: Any, initial_guess: dict[str, float]
    ) -> "HestonModel":
        from pricing_engine.calibration.engine import CalibrationEngine

        return CalibrationEngine().calibrate(cls, market_data, initial_guess)
