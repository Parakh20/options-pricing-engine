"""Implied volatility recovery from observed market option prices.

Given a market-observed option price, the implied volatility ``sigma_impl``
is the value of ``sigma`` that makes the Black-Scholes model price equal to
the market price:

    BlackScholes(S, K, T, r, sigma_impl, option_type).price() == market_price

There is no closed-form inverse of the Black-Scholes formula for ``sigma``,
so it must be solved numerically. This module implements two root-finding
methods:

1. **Newton-Raphson** (primary): fast quadratic convergence using the
   analytic vega as the derivative of price with respect to sigma.
2. **Bisection** (fallback): slower but guaranteed to converge given a
   bracketing interval, used when Newton-Raphson fails or diverges.

Important scaling note
-----------------------
``BlackScholes.vega()`` in ``core/black_scholes.py`` returns vega scaled
*per 1% (0.01) move in volatility* (i.e. ``raw_vega * 0.01``), matching
trading-desk Greek conventions. The Newton-Raphson update step,

    sigma_new = sigma_old - (price(sigma_old) - market_price) / raw_vega(sigma_old)

requires the *raw* (unscaled) vega -- the true derivative ``dPrice/dSigma``
where sigma is expressed in decimal form (e.g. 0.20, not 20). Using the
0.01-scaled vega directly in the Newton step would make the step 100x too
large and the iteration would diverge immediately. This module therefore
multiplies ``BlackScholes(...).vega()`` by 100 to recover raw vega before
using it as the Newton-Raphson derivative.
"""

from __future__ import annotations

import math

import pandas as pd

try:
    from core.black_scholes import BlackScholes
except ImportError:  # pragma: no cover - fallback when run as a standalone script
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.black_scholes import BlackScholes


class ImpliedVolCalculator:
    """Solves for Black-Scholes implied volatility from a market price.

    Provides a Newton-Raphson root-finder as the primary method, with an
    automatic bisection fallback for cases where Newton-Raphson fails to
    converge (e.g. poor initial guess, near-zero vega regions deep
    in/out-of-the-money).
    """

    def __init__(
        self,
        max_iter_newton: int = 100,
        max_iter_bisection: int = 200,
        tol: float = 1e-6,
        sigma_lower: float = 0.001,
        sigma_upper: float = 5.0,
    ) -> None:
        """Configure solver iteration limits, tolerance, and bisection bracket.

        Args:
            max_iter_newton: Maximum Newton-Raphson iterations.
            max_iter_bisection: Maximum bisection iterations.
            tol: Convergence tolerance on the price difference (absolute,
                in the same units as the option price).
            sigma_lower: Lower bound of the bisection sigma bracket.
            sigma_upper: Upper bound of the bisection sigma bracket.
        """
        self.max_iter_newton = max_iter_newton
        self.max_iter_bisection = max_iter_bisection
        self.tol = tol
        self.sigma_lower = sigma_lower
        self.sigma_upper = sigma_upper

    @staticmethod
    def _intrinsic_value(S: float, K: float, r: float, T: float, option_type: str) -> float:
        """Discounted intrinsic value, used as a floor sanity check for market_price."""
        discount = math.exp(-r * T)
        if option_type == "call":
            return max(S - K * discount, 0.0)
        return max(K * discount - S, 0.0)

    def newton_raphson(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        sigma_init: float = 0.2,
    ) -> float | None:
        """Solve for implied volatility via Newton-Raphson iteration.

        Update rule (derivative is the *raw*, unscaled vega):

            sigma_new = sigma_old - (price(sigma_old) - market_price) / raw_vega(sigma_old)

        where ``raw_vega = BlackScholes(...).vega() * 100`` because
        ``BlackScholes.vega()`` is scaled per 1% vol move (see module
        docstring for the full explanation of this scaling).

        Args:
            market_price: Observed market price of the option.
            S: Spot price of the underlying.
            K: Strike price.
            T: Time to expiry, in years.
            r: Risk-free rate (annualized, decimal).
            option_type: ``"call"`` or ``"put"``.
            sigma_init: Initial volatility guess (decimal).

        Returns:
            The implied volatility (decimal) if Newton-Raphson converges
            within ``max_iter_newton`` iterations and the result is a
            sane, positive volatility. Returns ``None`` if the market
            price is below intrinsic value (no valid IV exists), if a
            zero/near-zero vega is encountered (stalled iteration), if
            sigma wanders non-positive or to an absurd magnitude, or if
            the loop fails to converge within the iteration budget.
        """
        if T <= 0 or S <= 0 or K <= 0:
            return None

        intrinsic = self._intrinsic_value(S, K, r, T, option_type)
        if market_price < intrinsic - self.tol:
            return None

        sigma = sigma_init
        for _ in range(self.max_iter_newton):
            try:
                model = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type)
                model_price = model.price()
                raw_vega = model.vega() * 100.0
            except ValueError:
                return None

            diff = model_price - market_price
            if abs(diff) < self.tol:
                return sigma

            if raw_vega < 1e-10:
                # Vega has collapsed (deep ITM/OTM); Newton step is unreliable.
                return None

            sigma_new = sigma - diff / raw_vega

            if not math.isfinite(sigma_new) or sigma_new <= 0 or sigma_new > 10.0:
                # Diverged outside a sane volatility range.
                return None

            sigma = sigma_new

        return None

    def bisection(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
    ) -> float | None:
        """Solve for implied volatility via bisection on ``[sigma_lower, sigma_upper]``.

        Standard bisection on ``f(sigma) = price(sigma) - market_price``,
        halving the bracket until ``|f(sigma_mid)| < tol`` or the iteration
        budget is exhausted.

        Args:
            market_price: Observed market price of the option.
            S: Spot price of the underlying.
            K: Strike price.
            T: Time to expiry, in years.
            r: Risk-free rate (annualized, decimal).
            option_type: ``"call"`` or ``"put"``.

        Returns:
            The implied volatility (decimal) if found within tolerance,
            or ``None`` if the market price is not bracketed by
            ``[price(sigma_lower), price(sigma_upper)]`` (i.e. the market
            price is unreachable within the bracket) or T/S/K are invalid.
        """
        if T <= 0 or S <= 0 or K <= 0:
            return None

        def f(sigma: float) -> float:
            return BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type).price() - market_price

        lo, hi = self.sigma_lower, self.sigma_upper
        f_lo, f_hi = f(lo), f(hi)

        if f_lo == 0.0:
            return lo
        if f_hi == 0.0:
            return hi
        if f_lo * f_hi > 0:
            # market_price is not bracketed: no root in [sigma_lower, sigma_upper].
            return None

        for _ in range(self.max_iter_bisection):
            mid = (lo + hi) / 2.0
            f_mid = f(mid)

            if abs(f_mid) < self.tol:
                return mid

            if f_lo * f_mid < 0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid

        return (lo + hi) / 2.0

    def solve(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        sigma_init: float = 0.2,
    ) -> float | None:
        """Solve for implied volatility, trying Newton-Raphson first then bisection.

        Args:
            market_price: Observed market price of the option.
            S: Spot price of the underlying.
            K: Strike price.
            T: Time to expiry, in years.
            r: Risk-free rate (annualized, decimal).
            option_type: ``"call"`` or ``"put"``.
            sigma_init: Initial volatility guess for Newton-Raphson (decimal).

        Returns:
            The implied volatility (decimal), or ``None`` if neither method
            converges (e.g. market price is not attainable for any sigma in
            the bisection bracket, such as a price below intrinsic value).
        """
        iv = self.newton_raphson(market_price, S, K, T, r, option_type, sigma_init)
        if iv is not None:
            return iv
        return self.bisection(market_price, S, K, T, r, option_type)

    def build_iv_surface(
        self,
        chain: pd.DataFrame,
        S: float | None = None,
        r: float | None = None,
    ) -> pd.DataFrame:
        """Compute implied vols for an options chain and pivot into a vol surface.

        Args:
            chain: DataFrame with one row per option quote. Required
                columns: ``strike`` (float), ``T`` (time to expiry in
                years, float), ``market_price`` (float),
                ``option_type`` (``"call"``/``"put"``). Optionally,
                ``S`` and ``r`` may be supplied as per-row columns
                instead of (or in addition to) the scalar ``S``/``r``
                arguments -- per-row columns take precedence when both
                are present, since they let a single chain mix
                multiple as-of snapshots or rate curves.
            S: Scalar spot price applied to every row, used only when
                ``chain`` has no ``S`` column.
            r: Scalar risk-free rate applied to every row, used only when
                ``chain`` has no ``r`` column.

        Returns:
            A pivot table with strikes as the row index, expiries (``T``)
            as columns, and implied volatility as cell values. Rows/cells
            where no IV could be recovered contain ``NaN``.

        Raises:
            ValueError: If ``S`` is not resolvable (no ``S`` column and no
                scalar ``S`` given), or likewise for ``r``.
        """
        working = chain.copy()

        if "S" not in working.columns:
            if S is None:
                raise ValueError("S must be provided as a column or scalar argument")
            working["S"] = S
        if "r" not in working.columns:
            if r is None:
                raise ValueError("r must be provided as a column or scalar argument")
            working["r"] = r

        def _solve_row(row: pd.Series) -> float | None:
            return self.solve(
                market_price=row["market_price"],
                S=row["S"],
                K=row["strike"],
                T=row["T"],
                r=row["r"],
                option_type=row["option_type"],
            )

        working["implied_vol"] = working.apply(_solve_row, axis=1)

        return working.pivot_table(index="strike", columns="T", values="implied_vol")


if __name__ == "__main__":
    calc = ImpliedVolCalculator()

    print("=== Implied volatility smoke test ===")

    # 1. Newton-Raphson recovery.
    true_sigma = 0.20
    S0, K0, T0, r0 = 100.0, 100.0, 1.0, 0.05
    synthetic_price = BlackScholes(S=S0, K=K0, T=T0, r=r0, sigma=true_sigma, option_type="call").price()
    print(f"Synthetic market price (sigma={true_sigma}): {synthetic_price:.6f}")

    nr_sigma = calc.newton_raphson(synthetic_price, S0, K0, T0, r0, option_type="call")
    print(f"Newton-Raphson recovered sigma: {nr_sigma:.6f}")
    assert nr_sigma is not None and abs(nr_sigma - true_sigma) < 1e-4, "Newton-Raphson failed to recover sigma"
    print("Newton-Raphson smoke test passed.")

    # 2. Bisection recovery (called directly to exercise the fallback path).
    bisect_sigma = calc.bisection(synthetic_price, S0, K0, T0, r0, option_type="call")
    print(f"Bisection recovered sigma:      {bisect_sigma:.6f}")
    assert bisect_sigma is not None and abs(bisect_sigma - true_sigma) < 1e-4, "Bisection failed to recover sigma"
    print("Bisection smoke test passed.")

    # 3. Build a synthetic IV surface with a simple smile/skew baked in.
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    expiries = [0.25, 0.5, 1.0]

    def synthetic_sigma(strike: float, expiry: float) -> float:
        """Toy smile: vol rises away from ATM, and decays slightly with tenor."""
        moneyness = (strike - 100.0) / 100.0
        smile = 0.20 + 0.5 * moneyness**2
        term_decay = 0.02 * expiry
        return smile - term_decay

    rows = []
    for K in strikes:
        for T in expiries:
            sigma_true = synthetic_sigma(K, T)
            price = BlackScholes(S=100.0, K=K, T=T, r=0.05, sigma=sigma_true, option_type="call").price()
            rows.append({"strike": K, "T": T, "market_price": price, "option_type": "call"})

    chain = pd.DataFrame(rows)
    surface = calc.build_iv_surface(chain, S=100.0, r=0.05)

    print("\nSynthetic options chain (head):")
    print(chain.head())

    print("\nRecovered implied volatility surface (rows=strike, cols=T):")
    print(surface.round(4))
