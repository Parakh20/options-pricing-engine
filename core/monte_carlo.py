"""Monte Carlo pricing for European and Asian options under GBM.

The underlying is simulated under the risk-neutral measure as geometric
Brownian motion (GBM):

    dS_t = r * S_t * dt + sigma * S_t * dW_t

For a single terminal draw this has the exact (no-discretization-bias)
solution:

    S_T = S_0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z),  Z ~ N(0, 1)

For path-dependent (Asian) payoffs, the same exact-update scheme is applied
recursively over M equally spaced time steps of size dt = T / M:

    S_{t+dt} = S_t * exp((r - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z_t)

Two variance-reduction techniques are used for the European pricer:

1. **Antithetic variates** -- each standard normal draw Z is paired with -Z.
   Since GBM terminal value is a monotonic function of Z, this induces
   negative correlation between paired payoffs and reduces estimator
   variance without introducing bias.

2. **Control variates** -- the simulated terminal price S_T is used as a
   control variate. Its risk-neutral expectation is known exactly in closed
   form, E[S_T] = S_0 * exp(r * T), so deviations of the simulated sample
   mean from this known mean can be used to correct the option-payoff
   estimator. If Y is the discounted MC payoff and X = exp(-rT) * S_T is the
   discounted control with known mean E[X] = S_0 (since
   exp(-rT) * S_0 * exp(rT) = S_0), the control-variate estimator is

       Y_cv = Y - beta * (X - E[X])

   with beta chosen to minimize Var(Y_cv): beta* = Cov(X, Y) / Var(X). Since
   the option payoff is a (correlated, monotonic-ish) function of the same
   S_T used to build X, this control captures a large share of the Monte
   Carlo sampling error without needing the answer itself. The
   Black-Scholes closed-form price is used only as an independent
   *reference* to measure error in the convergence analysis, not folded
   into the estimator (which would make the correction circular).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from core.black_scholes import BlackScholes
except ImportError:  # pragma: no cover - fallback for `python3 core/monte_carlo.py`
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.black_scholes import BlackScholes


@dataclass
class MonteCarloResult:
    """Result of a Monte Carlo price estimate.

    Attributes:
        price: Estimated option price (discounted, sample-mean payoff).
        std_error: Standard error of the price estimate
            (sample std / sqrt(n_simulations)).
        ci_lower: Lower bound of the 95% confidence interval.
        ci_upper: Upper bound of the 95% confidence interval.
        n_simulations: Number of simulated paths used (after antithetic
            pairing, this is the *effective* total path count).
    """

    price: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n_simulations: int

    def as_dict(self) -> dict[str, float]:
        """Return the result as a plain dict."""
        return {
            "price": self.price,
            "std_error": self.std_error,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n_simulations": self.n_simulations,
        }


@dataclass
class MonteCarloPricer:
    """Monte Carlo pricer for European and arithmetic Asian options under GBM.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option (used for European pricing and
            fixed-strike Asian pricing).
        T: Time to expiry, in years.
        r: Continuously compounded risk-free rate (annualized, decimal).
        sigma: Annualized volatility of the underlying (decimal).
        option_type: Either ``"call"`` or ``"put"``.
        seed: Optional seed for the NumPy random generator, for
            reproducibility. ``None`` draws fresh entropy.
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: str = "call"
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {self.option_type!r}")
        if self.S <= 0 or self.K <= 0:
            raise ValueError("S and K must be positive")
        if self.T <= 0:
            raise ValueError("T must be strictly positive")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        self._rng = np.random.default_rng(self.seed)

    def _draw_antithetic_normals(self, n_simulations: int) -> np.ndarray:
        """Draw ``n_simulations`` standard normal variates using antithetic pairing.

        Half of ``n_simulations`` independent draws Z are made, and each is
        paired with its antithetic partner -Z, giving ``n_simulations`` total
        draws (rounded up to an even number). Pairing Z with -Z preserves the
        N(0, 1) marginal distribution (since -Z ~ N(0, 1) too) while inducing
        negative correlation between paired outcomes, which lowers the
        variance of sample averages over monotonic functions of Z.

        Returns:
            1-D array of length ``2 * ceil(n_simulations / 2)`` of standard
            normal draws, arranged as [Z_1..Z_h, -Z_1..-Z_h].
        """
        half = math.ceil(n_simulations / 2)
        z = self._rng.standard_normal(half)
        return np.concatenate([z, -z])

    def _simulate_terminal_prices(self, n_simulations: int) -> np.ndarray:
        """Simulate terminal prices S_T via the exact GBM solution.

        Vectorized over all paths simultaneously (no per-path Python loop):

            S_T = S_0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)

        Uses antithetic-paired normal draws for variance reduction.
        """
        z = self._draw_antithetic_normals(n_simulations)
        drift = (self.r - 0.5 * self.sigma**2) * self.T
        diffusion = self.sigma * math.sqrt(self.T) * z
        return self.S * np.exp(drift + diffusion)

    def _payoff(self, terminal_prices: np.ndarray) -> np.ndarray:
        """Vectorized European payoff at maturity, undiscounted."""
        if self.option_type == "call":
            return np.maximum(terminal_prices - self.K, 0.0)
        return np.maximum(self.K - terminal_prices, 0.0)

    def price_european(self, n_simulations: int = 100_000, use_control_variate: bool = True) -> MonteCarloResult:
        """Price a European option via Monte Carlo with antithetic + control variates.

        Steps:
            1. Simulate terminal prices S_T with antithetic-paired normals.
            2. Compute discounted payoffs Y_i = exp(-rT) * payoff(S_T,i).
            3. If ``use_control_variate`` is True, apply the control-variate
               correction using the discounted terminal price
               X_i = exp(-rT) * S_T,i as the control. Under the Black-Scholes
               GBM model, X is a martingale with known closed-form mean
               E[X] = S_0:

                   beta_hat = Cov(X, Y) / Var(X)
                   Y_cv,i = Y_i - beta_hat * (X_i - E[X])

               X is correlated with the option payoff Y (both are functions
               of the same simulated S_T) but has a *known* population mean,
               so the realized sampling error in X (X_bar - S_0) can be used
               to correct Y without referencing the option's own
               closed-form price. This reduces variance to
               Var(Y) * (1 - corr(X, Y)^2) in the limit of an exact beta.
            4. Standard error = sample_std / sqrt(n), 95% CI = price +/- 1.96 * SE.

        Args:
            n_simulations: Target number of simulated paths (rounded up to
                an even number for antithetic pairing).
            use_control_variate: Whether to apply the discounted-terminal-
                price control variate correction (tied to the Black-Scholes
                GBM model's known martingale property).

        Returns:
            A :class:`MonteCarloResult` with price, standard error, and 95%
            confidence interval.
        """
        terminal_prices = self._simulate_terminal_prices(n_simulations)
        discount = math.exp(-self.r * self.T)
        payoffs = discount * self._payoff(terminal_prices)

        if use_control_variate:
            # Control variate: discounted terminal price X = exp(-rT) * S_T.
            # Under the risk-neutral GBM (Black-Scholes) model this has a
            # known closed-form mean E[X] = S_0 (martingale property), since
            # E[S_T] = S_0 * exp(rT). Using the realized sampling error in X
            # to correct the option-payoff estimator Y is the classic Boyle
            # (1977) control-variate scheme tied to the Black-Scholes model
            # of the underlying.
            control = discount * terminal_prices
            control_mean = self.S
            control_var = np.var(control, ddof=1)
            if control_var > 0:
                beta_hat = np.cov(control, payoffs, ddof=1)[0, 1] / control_var
            else:
                beta_hat = 0.0
            adjusted = payoffs - beta_hat * (control - control_mean)
        else:
            adjusted = payoffs

        n_eff = adjusted.size
        price = float(np.mean(adjusted))
        std_error = float(np.std(adjusted, ddof=1) / math.sqrt(n_eff))
        ci_half_width = 1.96 * std_error

        return MonteCarloResult(
            price=price,
            std_error=std_error,
            ci_lower=price - ci_half_width,
            ci_upper=price + ci_half_width,
            n_simulations=n_eff,
        )

    def _simulate_paths(self, n_simulations: int, n_steps: int) -> np.ndarray:
        """Simulate full GBM paths over ``n_steps`` time increments.

        Vectorized across all paths and all time steps at once via a single
        cumulative-sum over log-increments (the only loop-like structure is
        implicit in ``np.cumsum`` along the time axis, not a Python loop
        over paths or steps):

            log S_{t_k} = log S_0 + sum_{j=1}^{k} [(r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_j]

        Antithetic pairing is applied to the underlying normal draws.

        Args:
            n_simulations: Target number of simulated paths.
            n_steps: Number of time steps M (dt = T / M).

        Returns:
            Array of shape (n_effective_paths, n_steps + 1) including the
            initial price S_0 in column 0.
        """
        half = math.ceil(n_simulations / 2)
        z = self._rng.standard_normal((half, n_steps))
        z = np.concatenate([z, -z], axis=0)  # antithetic pairing across paths

        dt = self.T / n_steps
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * math.sqrt(dt) * z

        log_increments = drift + diffusion
        log_paths = np.cumsum(log_increments, axis=1)
        log_s0 = math.log(self.S)

        n_eff = z.shape[0]
        full_log_paths = np.empty((n_eff, n_steps + 1))
        full_log_paths[:, 0] = log_s0
        full_log_paths[:, 1:] = log_s0 + log_paths
        return np.exp(full_log_paths)

    def price_asian(
        self,
        n_simulations: int = 100_000,
        n_steps: int = 252,
        strike_type: str = "fixed",
    ) -> MonteCarloResult:
        """Price an arithmetic-average Asian option via Monte Carlo.

        The path is simulated over ``n_steps`` equal increments of dt = T/M,
        and the payoff is based on the arithmetic average of the simulated
        prices along the path (excluding S_0, i.e. averaging over the M
        observation points t_1, ..., t_M).

        Fixed-strike payoff:
            Call: max(avg(S) - K, 0)
            Put:  max(K - avg(S), 0)

        Floating-strike payoff:
            Call: max(S_T - avg(S), 0)
            Put:  max(avg(S) - S_T, 0)

        Args:
            n_simulations: Target number of simulated paths.
            n_steps: Number of discrete time steps M used to build the
                averaging path (loop over time steps is fine; it is O(M),
                not O(N) over paths -- the path array itself is built and
                averaged via vectorized NumPy operations).
            strike_type: Either ``"fixed"`` or ``"floating"``.

        Returns:
            A :class:`MonteCarloResult` with price, standard error, and 95%
            confidence interval.
        """
        if strike_type not in ("fixed", "floating"):
            raise ValueError(f"strike_type must be 'fixed' or 'floating', got {strike_type!r}")

        paths = self._simulate_paths(n_simulations, n_steps)
        average_price = np.mean(paths[:, 1:], axis=1)  # exclude S_0
        terminal_price = paths[:, -1]

        if strike_type == "fixed":
            if self.option_type == "call":
                payoffs = np.maximum(average_price - self.K, 0.0)
            else:
                payoffs = np.maximum(self.K - average_price, 0.0)
        else:
            if self.option_type == "call":
                payoffs = np.maximum(terminal_price - average_price, 0.0)
            else:
                payoffs = np.maximum(average_price - terminal_price, 0.0)

        discount = math.exp(-self.r * self.T)
        discounted_payoffs = discount * payoffs

        n_eff = discounted_payoffs.size
        price = float(np.mean(discounted_payoffs))
        std_error = float(np.std(discounted_payoffs, ddof=1) / math.sqrt(n_eff))
        ci_half_width = 1.96 * std_error

        return MonteCarloResult(
            price=price,
            std_error=std_error,
            ci_lower=price - ci_half_width,
            ci_upper=price + ci_half_width,
            n_simulations=n_eff,
        )

    def convergence_analysis(
        self,
        simulation_sizes: list[int] | None = None,
        use_control_variate: bool = True,
    ) -> pd.DataFrame:
        """Run the European pricer across a range of simulation sizes.

        For each ``n`` in ``simulation_sizes``, computes the Monte Carlo
        price, its standard error, and its absolute error against the exact
        closed-form Black-Scholes price. Demonstrates Monte Carlo
        convergence (error should shrink roughly as O(1/sqrt(n))).

        Args:
            simulation_sizes: List of path counts to evaluate. Defaults to
                ``[1000, 5000, 10000, 50000, 100000]``.
            use_control_variate: Whether the European pricer should use the
                Black-Scholes control variate at each simulation size.

        Returns:
            A pandas DataFrame with columns: ``n_simulations``, ``price``,
            ``std_error``, ``abs_error_vs_bs``.
        """
        if simulation_sizes is None:
            simulation_sizes = [1_000, 5_000, 10_000, 50_000, 100_000]

        bs_price = BlackScholes(self.S, self.K, self.T, self.r, self.sigma, self.option_type).price()

        rows: list[dict[str, float]] = []
        for n in simulation_sizes:
            result = self.price_european(n_simulations=n, use_control_variate=use_control_variate)
            rows.append(
                {
                    "n_simulations": result.n_simulations,
                    "price": result.price,
                    "std_error": result.std_error,
                    "abs_error_vs_bs": abs(result.price - bs_price),
                }
            )

        return pd.DataFrame(rows, columns=["n_simulations", "price", "std_error", "abs_error_vs_bs"])


if __name__ == "__main__":
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    print("=== Monte Carlo smoke test ===")

    bs_call_price = BlackScholes(S, K, T, r, sigma, "call").price()
    print(f"Black-Scholes call price (exact): {bs_call_price:.4f}")

    mc_call = MonteCarloPricer(S, K, T, r, sigma, option_type="call", seed=42)

    mc_result_cv = mc_call.price_european(n_simulations=100_000, use_control_variate=True)
    print(
        f"MC call price (antithetic + control variate, n={mc_result_cv.n_simulations}): "
        f"{mc_result_cv.price:.4f}  (SE={mc_result_cv.std_error:.5f}, "
        f"95% CI=[{mc_result_cv.ci_lower:.4f}, {mc_result_cv.ci_upper:.4f}])"
    )

    mc_result_no_cv = mc_call.price_european(n_simulations=100_000, use_control_variate=False)
    print(
        f"MC call price (antithetic only,        n={mc_result_no_cv.n_simulations}): "
        f"{mc_result_no_cv.price:.4f}  (SE={mc_result_no_cv.std_error:.5f}, "
        f"95% CI=[{mc_result_no_cv.ci_lower:.4f}, {mc_result_no_cv.ci_upper:.4f}])"
    )
    print(
        "Variance reduction from control variate: "
        f"{(1 - (mc_result_cv.std_error / mc_result_no_cv.std_error) ** 2) * 100:.1f}% lower SE^2"
    )

    asian_fixed = mc_call.price_asian(n_simulations=100_000, n_steps=252, strike_type="fixed")
    print(
        f"Asian fixed-strike call price (n={asian_fixed.n_simulations}, steps=252): "
        f"{asian_fixed.price:.4f}  (SE={asian_fixed.std_error:.5f})"
    )

    asian_floating = mc_call.price_asian(n_simulations=100_000, n_steps=252, strike_type="floating")
    print(
        f"Asian floating-strike call price (n={asian_floating.n_simulations}, steps=252): "
        f"{asian_floating.price:.4f}  (SE={asian_floating.std_error:.5f})"
    )

    print("\n=== Convergence analysis (European call, with control variate) ===")
    convergence_df = mc_call.convergence_analysis()
    print(convergence_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
