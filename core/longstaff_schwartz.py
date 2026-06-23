"""Longstaff-Schwartz Method (LSM) for pricing American-style options.

American options may be exercised at any time up to expiry, so their value
must account for the optimal stopping decision at every point along the
underlying's path. There is generally no closed-form solution once early
exercise is allowed (except for special cases), so we resort to a Monte
Carlo simulation combined with a regression-based approximation of the
continuation value: the Longstaff-Schwartz (2001) method.

Algorithm summary
------------------
1. Simulate ``n_paths`` trajectories of geometric Brownian motion (GBM)
   over ``n_steps`` discrete time steps under the risk-neutral measure:

       dS_t = r * S_t * dt + sigma * S_t * dW_t

   which has the exact (in log-space) discretization:

       S_{t+dt} = S_t * exp((r - sigma^2/2) * dt + sigma * sqrt(dt) * Z),  Z ~ N(0, 1)

2. At the final time step (expiry), the option value on every path is its
   intrinsic value: ``max(K - S_T, 0)`` for a put.

3. Step backward through time. At each interior time step ``t``:
   a. Identify in-the-money (ITM) paths only -- only paths where immediate
      exercise has positive value are candidates for early exercise, and
      regressing on ITM paths only avoids extrapolation bias from deep
      out-of-the-money paths (Longstaff & Schwartz, 2001).
   b. Regress the discounted future cash flow (the "continuation value")
      on a set of basis functions of the current spot price ``S_t``, using
      least squares over the ITM paths only.
   c. For each ITM path, compare the immediate exercise value to the
      fitted continuation value. If exercise is more valuable, mark the
      path as exercised at time ``t``: its cash flow becomes the intrinsic
      value at ``t``, and all of its future (later) cash flows are zeroed
      out (an option can only be exercised once).
   d. Record the exercise boundary at this time step: the highest spot
      price among ITM paths that chose to exercise (the critical price
      below which immediate exercise is optimal).

4. Each path now has exactly one (discounted-from-its-own-time) cash flow.
   Discount every path's realized cash flow back to t=0 using
   ``exp(-r * t_exercise)`` and average across paths to obtain the price
   estimate.

This module prices American **put** options. Calls on non-dividend-paying
stocks are never optimal to exercise early, so the put is the canonical
test case for early-exercise premium.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial import laguerre as np_laguerre

try:
    from core.black_scholes import BlackScholes
except ImportError:  # pragma: no cover - fallback for running as a standalone script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.black_scholes import BlackScholes


@dataclass
class LongstaffSchwartz:
    """Longstaff-Schwartz Monte Carlo pricer for American put options.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiry, in years.
        r: Continuously compounded risk-free rate (annualized, decimal).
        sigma: Annualized volatility of the underlying (decimal).
        n_paths: Number of simulated GBM paths.
        n_steps: Number of discrete time steps between 0 and T.
        basis: Basis function family for the continuation-value regression,
            either ``"polynomial"`` (1, S, S^2, ...) or ``"laguerre"``
            (Laguerre polynomials, as used in the original Longstaff &
            Schwartz (2001) paper).
        degree: Highest degree/order of the basis function expansion.
        seed: Optional seed for the random number generator, for
            reproducibility.
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    n_paths: int = 10_000
    n_steps: int = 50
    basis: str = "laguerre"
    degree: int = 3
    seed: int | None = None

    _exercise_boundary: list[tuple[float, float]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.S <= 0 or self.K <= 0:
            raise ValueError("S and K must be positive")
        if self.T <= 0:
            raise ValueError("T must be strictly positive")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        if self.n_paths < 1 or self.n_steps < 1:
            raise ValueError("n_paths and n_steps must be positive integers")
        if self.basis not in ("polynomial", "laguerre"):
            raise ValueError(f"basis must be 'polynomial' or 'laguerre', got {self.basis!r}")
        if self.degree < 1:
            raise ValueError("degree must be at least 1")

    def _simulate_paths(self) -> np.ndarray:
        """Simulate GBM paths under the risk-neutral measure.

        Returns an array of shape ``(n_paths, n_steps + 1)`` where column 0
        is the initial spot price ``S`` (identical across all paths) and
        column ``j`` holds the simulated spot price at time ``j * dt``.

        Uses the exact log-Euler discretization of GBM, vectorized across
        all paths and all time steps simultaneously (no per-path loop):

            S_{t+dt} = S_t * exp((r - sigma^2 / 2) * dt + sigma * sqrt(dt) * Z)
        """
        rng = np.random.default_rng(self.seed)
        dt = self.T / self.n_steps
        drift = (self.r - 0.5 * self.sigma**2) * dt
        vol = self.sigma * math.sqrt(dt)

        z = rng.standard_normal(size=(self.n_paths, self.n_steps))
        log_increments = drift + vol * z
        log_paths = np.cumsum(log_increments, axis=1)

        paths = np.empty((self.n_paths, self.n_steps + 1))
        paths[:, 0] = self.S
        paths[:, 1:] = self.S * np.exp(log_paths)
        return paths

    def _basis_matrix(self, x: np.ndarray) -> np.ndarray:
        """Build the design matrix of basis functions evaluated at ``x``.

        For ``basis="polynomial"``, columns are ``[1, x, x^2, ..., x^degree]``.

        For ``basis="laguerre"``, columns are the first ``degree + 1``
        (physicists'/probabilists') Laguerre polynomials ``L_0(x), ..., L_degree(x)``,
        evaluated via ``numpy.polynomial.laguerre``. Laguerre polynomials are
        the basis originally proposed by Longstaff & Schwartz (2001) since
        they are a natural fit for functions defined on [0, inf).
        """
        if self.basis == "polynomial":
            return np.vstack([x**p for p in range(self.degree + 1)]).T

        # Laguerre: evaluate each L_p(x) individually via its coefficient vector.
        columns = []
        for p in range(self.degree + 1):
            coeffs = np.zeros(p + 1)
            coeffs[p] = 1.0
            columns.append(np_laguerre.lagval(x, coeffs))
        return np.vstack(columns).T

    def price(self) -> tuple[float, pd.DataFrame]:
        """Run the LSM algorithm and return the American put price.

        Returns:
            A tuple ``(price, boundary_df)`` where ``price`` is the
            Monte Carlo estimate of the American put value, and
            ``boundary_df`` is a DataFrame with columns ``["time",
            "critical_price"]`` describing the estimated early-exercise
            boundary at each time step (rows are omitted for time steps
            with no in-the-money exercises).
        """
        dt = self.T / self.n_steps
        paths = self._simulate_paths()

        # Intrinsic (exercise) value of a put at every time step, for every path.
        intrinsic = np.maximum(self.K - paths, 0.0)

        # cash_flow[i] = the (un-discounted-to-zero, time-of-payment-tagged) payoff
        # path i ultimately realizes. exercise_time[i] tracks *when* (in step index)
        # that payoff occurs, so we can discount each path back from its own
        # exercise time at the end.
        cash_flow = intrinsic[:, -1].copy()
        exercise_time = np.full(self.n_paths, self.n_steps, dtype=int)

        self._exercise_boundary = []

        # Backward induction from the second-to-last step down to step 1.
        # (Step 0 is t=0, never optimal to exercise immediately at inception
        # for a freshly priced option, so we stop at step 1.)
        for t in range(self.n_steps - 1, 0, -1):
            spot_t = paths[:, t]
            itm_mask = spot_t < self.K
            if not np.any(itm_mask):
                continue

            exercise_value = intrinsic[:, t]

            # Discount each path's currently-recorded future cash flow back
            # to time t (not all the way to 0) so the regression target is
            # the continuation value as seen from time t.
            time_to_cf = (exercise_time - t) * dt
            discounted_cf = cash_flow * np.exp(-self.r * time_to_cf)

            x_itm = spot_t[itm_mask]
            y_itm = discounted_cf[itm_mask]

            design = self._basis_matrix(x_itm)
            coeffs, *_ = np.linalg.lstsq(design, y_itm, rcond=None)
            continuation_value = design @ coeffs

            exercise_now = exercise_value[itm_mask] > continuation_value
            itm_indices = np.flatnonzero(itm_mask)
            exercised_indices = itm_indices[exercise_now]

            if exercised_indices.size > 0:
                cash_flow[exercised_indices] = exercise_value[exercised_indices]
                exercise_time[exercised_indices] = t

                critical_price = spot_t[exercised_indices].max()
                self._exercise_boundary.append((t * dt, float(critical_price)))

        # Discount every path's realized payoff back to t=0 from its own
        # (possibly early) exercise time, then average across paths.
        discount_factors = np.exp(-self.r * exercise_time * dt)
        price = float(np.mean(cash_flow * discount_factors))

        boundary_df = pd.DataFrame(
            sorted(self._exercise_boundary, key=lambda pair: pair[0]),
            columns=["time", "critical_price"],
        )
        return price, boundary_df


if __name__ == "__main__":
    print("=== Longstaff-Schwartz American put smoke test ===")

    lsm_atm = LongstaffSchwartz(
        S=100, K=100, T=1.0, r=0.05, sigma=0.20,
        n_paths=50_000, n_steps=50, basis="laguerre", degree=3, seed=42,
    )
    atm_price, atm_boundary = lsm_atm.price()
    bs_atm_put = BlackScholes(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put").price()

    print(f"ATM American put price (LSM): {atm_price:.4f}")
    print(f"ATM European put price (BS):  {bs_atm_put:.4f}")
    print(f"Early exercise premium:       {atm_price - bs_atm_put:.4f}")
    print(f"Exercise boundary (first 5 rows):\n{atm_boundary.head()}")

    sane_range_ok = 5.0 <= atm_price <= 7.0
    print(f"[{'PASS' if sane_range_ok else 'FAIL'}] ATM price in sane range [5.0, 7.0]: {atm_price:.4f}")
    assert sane_range_ok, f"ATM American put price {atm_price:.4f} is outside the expected sane range"

    # --- Deep out-of-the-money sanity check ---------------------------------
    # S=150, K=100: the put is far out of the money and essentially never
    # crosses into ITM territory, so there is negligible early-exercise
    # value. American price should be close to the European price.
    print("\n--- Deep OTM sanity check (S=150, K=100) ---")
    lsm_otm = LongstaffSchwartz(
        S=150, K=100, T=1.0, r=0.05, sigma=0.20,
        n_paths=50_000, n_steps=50, basis="laguerre", degree=3, seed=42,
    )
    otm_price, _ = lsm_otm.price()
    bs_otm_put = BlackScholes(S=150, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put").price()
    otm_diff = abs(otm_price - bs_otm_put)

    print(f"American price: {otm_price:.4f}  |  European price: {bs_otm_put:.4f}  |  diff: {otm_diff:.4f}")
    otm_ok = otm_diff < 0.50
    print(f"[{'PASS' if otm_ok else 'FAIL'}] Deep OTM: American ~= European (diff < 0.50): {otm_diff:.4f}")
    assert otm_ok, f"Deep OTM American/European price diff too large: {otm_diff:.4f}"

    # --- Deep in-the-money sanity check -------------------------------------
    # S=60, K=100: the put is deep ITM with substantial early-exercise
    # value, so American price should exceed the European price (a
    # positive early-exercise premium), within Monte Carlo noise.
    print("\n--- Deep ITM sanity check (S=60, K=100) ---")
    lsm_itm = LongstaffSchwartz(
        S=60, K=100, T=1.0, r=0.05, sigma=0.20,
        n_paths=50_000, n_steps=50, basis="laguerre", degree=3, seed=42,
    )
    itm_price, _ = lsm_itm.price()
    bs_itm_put = BlackScholes(S=60, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put").price()
    itm_tolerance = 0.05  # allow tiny MC noise around the boundary

    print(f"American price: {itm_price:.4f}  |  European price: {bs_itm_put:.4f}")
    itm_ok = itm_price >= bs_itm_put - itm_tolerance
    print(f"[{'PASS' if itm_ok else 'FAIL'}] Deep ITM: American >= European (within MC noise): "
          f"{itm_price:.4f} >= {bs_itm_put:.4f} - {itm_tolerance}")
    assert itm_ok, f"Deep ITM American price {itm_price:.4f} is below European price {bs_itm_put:.4f}"

    print("\nAll Longstaff-Schwartz smoke tests passed.")
