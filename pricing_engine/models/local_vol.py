"""Dupire local volatility: derives a local vol surface sigma_LV(K, T) from
an implied vol surface via finite differences of the Dupire PDE, then prices
via Monte Carlo simulation on that local vol surface.

Dupire's formula (in terms of call price C(K, T)):

    sigma_LV(K, T)^2 = 2 * (dC/dT + r*K*dC/dK) / (K^2 * d2C/dK2)

We compute call prices from the input IV surface via Black-Scholes and take
finite differences numerically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from pricing_engine.core.black_scholes import BlackScholes
from pricing_engine.models.base import PricingModel

IVSurface = Callable[[float, float], float]


def build_local_vol_surface(
    iv_surface: IVSurface, S: float, r: float, dK: float = 0.5, dT: float = 1e-3
) -> Callable[[float, float], float]:
    """Return a function (K, T) -> local volatility, via finite-differenced
    Dupire formula applied to Black-Scholes call prices implied by iv_surface.
    """

    def call_price(K: float, T: float) -> float:
        T_eff = max(T, 1e-6)
        sigma = iv_surface(K, T_eff)
        return BlackScholes(S, K, T_eff, r, sigma, "call").price()

    def local_vol(K: float, T: float) -> float:
        T = max(T, dT * 2)

        dC_dT = (call_price(K, T + dT) - call_price(K, T - dT)) / (2 * dT)
        dC_dK = (call_price(K + dK, T) - call_price(K - dK, T)) / (2 * dK)
        d2C_dK2 = (
            call_price(K + dK, T) - 2 * call_price(K, T) + call_price(K - dK, T)
        ) / (dK**2)

        numerator = dC_dT + r * K * dC_dK
        denominator = 0.5 * K**2 * d2C_dK2

        if denominator <= 1e-12 or numerator <= 0:
            # Numerically degenerate region (deep wings / near-zero maturity):
            # fall back to the input surface's own IV at that point.
            return iv_surface(K, T)

        return float(np.sqrt(numerator / denominator))

    return local_vol


@dataclass
class LocalVolModel(PricingModel):
    # NOTE: _simulate_terminal_prices is O(n_paths * n_steps) with a scalar
    # local_vol() call per path per step, so the 50_000-path default can be
    # slow (multi-second) — see Phase-2 follow-up for vectorizing local_vol.
    S: float
    K: float
    T: float
    r: float
    iv_surface: IVSurface
    option_type: str = "call"
    n_steps: int = 100
    n_paths: int = 50_000
    seed: int | None = None

    def _simulate_terminal_prices(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        local_vol = build_local_vol_surface(self.iv_surface, self.S, self.r)
        dt = self.T / self.n_steps
        S_t = np.full(self.n_paths, self.S, dtype=float)

        for step in range(self.n_steps):
            t = step * dt
            # Bucket paths into strike bands to avoid an O(n_paths) local_vol
            # call per path per step; a coarse per-path evaluation is used
            # here for correctness first (perf backends land in phase 2).
            sigmas = np.array([local_vol(float(s), t) for s in S_t])
            Z = rng.standard_normal(self.n_paths)
            S_t = S_t * np.exp(
                (self.r - 0.5 * sigmas**2) * dt + sigmas * np.sqrt(dt) * Z
            )
        return S_t

    def price(self) -> float:
        terminal = self._simulate_terminal_prices()
        if self.option_type == "call":
            payoff = np.maximum(terminal - self.K, 0.0)
        else:
            payoff = np.maximum(self.K - terminal, 0.0)
        return float(np.exp(-self.r * self.T) * payoff.mean())

    def greeks(self) -> dict[str, float]:
        bump = self.S * 1e-3
        up = LocalVolModel(**{**self.__dict__, "S": self.S + bump, "seed": self.seed})
        down = LocalVolModel(**{**self.__dict__, "S": self.S - bump, "seed": self.seed})
        delta = (up.price() - down.price()) / (2 * bump)
        return {"delta": delta, "vega": float("nan")}

    @classmethod
    def calibrate(
        cls, market_data: Any, initial_guess: dict[str, float]
    ) -> "LocalVolModel":
        raise NotImplementedError(
            "LocalVolModel is calibrated by fitting the IV surface directly "
            "(see market.vol_surface, phase 5), not via parametric calibration."
        )
