"""Closed-form Black-Scholes-Merton option pricing and analytic Greeks.

The Black-Scholes-Merton model assumes the underlying follows geometric
Brownian motion under the risk-neutral measure:

    dS_t = r * S_t * dt + sigma * S_t * dW_t

with constant risk-free rate ``r`` and volatility ``sigma``. Under this
assumption, European option prices and their sensitivities (Greeks) have
closed-form expressions in terms of the standard normal CDF/PDF.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm


@dataclass
class BlackScholes:
    """Black-Scholes-Merton closed-form pricer for European options.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiry, in years.
        r: Continuously compounded risk-free rate (annualized, decimal).
        sigma: Annualized volatility of the underlying (decimal).
        option_type: Either ``"call"`` or ``"put"``.
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: str = "call"

    def __post_init__(self) -> None:
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {self.option_type!r}")
        if self.S <= 0 or self.K <= 0:
            raise ValueError("S and K must be positive")
        if self.T < 0:
            raise ValueError("T must be non-negative")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")

    def _d1_d2(self) -> tuple[float, float]:
        """Compute d1 and d2.

        d1 = [ln(S/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        """
        if self.T <= 0 or self.sigma <= 0:
            raise ValueError("T and sigma must be strictly positive to compute d1/d2")
        sqrt_t = math.sqrt(self.T)
        d1 = (
            math.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T
        ) / (self.sigma * sqrt_t)
        d2 = d1 - self.sigma * sqrt_t
        return d1, d2

    def price(self) -> float:
        """Closed-form Black-Scholes price.

        Call: ``S * N(d1) - K * exp(-rT) * N(d2)``
        Put:  ``K * exp(-rT) * N(-d2) - S * N(-d1)``
        """
        if self.T == 0:
            if self.option_type == "call":
                return max(self.S - self.K, 0.0)
            return max(self.K - self.S, 0.0)

        d1, d2 = self._d1_d2()
        discount = math.exp(-self.r * self.T)

        if self.option_type == "call":
            return self.S * norm.cdf(d1) - self.K * discount * norm.cdf(d2)
        return self.K * discount * norm.cdf(-d2) - self.S * norm.cdf(-d1)

    def delta(self) -> float:
        """dV/dS. Call: N(d1). Put: N(d1) - 1."""
        d1, _ = self._d1_d2()
        if self.option_type == "call":
            return norm.cdf(d1)
        return norm.cdf(d1) - 1.0

    def gamma(self) -> float:
        """d^2V/dS^2 = phi(d1) / (S * sigma * sqrt(T)). Identical for calls and puts."""
        d1, _ = self._d1_d2()
        return norm.pdf(d1) / (self.S * self.sigma * math.sqrt(self.T))

    def vega(self) -> float:
        """dV/dsigma, scaled per 1% (0.01) move in volatility.

        Raw vega = S * phi(d1) * sqrt(T). Identical for calls and puts.
        """
        d1, _ = self._d1_d2()
        raw_vega = self.S * norm.pdf(d1) * math.sqrt(self.T)
        return raw_vega * 0.01

    def theta(self) -> float:
        """dV/dt (time decay), scaled per calendar day (raw annual theta / 365)."""
        d1, d2 = self._d1_d2()
        discount = math.exp(-self.r * self.T)
        term1 = -(self.S * norm.pdf(d1) * self.sigma) / (2 * math.sqrt(self.T))

        if self.option_type == "call":
            term2 = -self.r * self.K * discount * norm.cdf(d2)
        else:
            term2 = self.r * self.K * discount * norm.cdf(-d2)

        raw_theta = term1 + term2
        return raw_theta / 365.0

    def rho(self) -> float:
        """dV/dr, scaled per 1% (0.01) move in the risk-free rate."""
        _, d2 = self._d1_d2()
        discount = math.exp(-self.r * self.T)

        if self.option_type == "call":
            raw_rho = self.K * self.T * discount * norm.cdf(d2)
        else:
            raw_rho = -self.K * self.T * discount * norm.cdf(-d2)

        return raw_rho * 0.01

    def all_greeks(self) -> dict[str, float]:
        """Return all five Greeks as a dict: delta, gamma, vega, theta, rho."""
        return {
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "theta": self.theta(),
            "rho": self.rho(),
        }


def check_put_call_parity(S: float, K: float, T: float, r: float, sigma: float, tol: float = 1e-6) -> bool:
    """Verify put-call parity: C - P = S - K * exp(-rT), within tolerance.

    Raises AssertionError if parity is violated beyond ``tol``.
    """
    call_price = BlackScholes(S, K, T, r, sigma, "call").price()
    put_price = BlackScholes(S, K, T, r, sigma, "put").price()
    lhs = call_price - put_price
    rhs = S - K * math.exp(-r * T)
    diff = abs(lhs - rhs)
    assert diff < tol, f"Put-call parity violated: |{lhs} - {rhs}| = {diff} >= {tol}"
    return True


if __name__ == "__main__":
    bs_call = BlackScholes(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
    bs_put = BlackScholes(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put")

    print("=== Black-Scholes smoke test ===")
    print(f"Call price: {bs_call.price():.4f}")
    print(f"Put price:  {bs_put.price():.4f}")
    print(f"Call Greeks: {bs_call.all_greeks()}")
    print(f"Put Greeks:  {bs_put.all_greeks()}")

    parity_ok = check_put_call_parity(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    print(f"Put-call parity check passed: {parity_ok}")
