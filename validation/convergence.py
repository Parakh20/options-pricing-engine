"""Monte Carlo convergence validation against closed-form Black-Scholes prices.

Demonstrates that the Monte Carlo estimator's error shrinks as the number of
simulated paths grows (the classic O(1/sqrt(n)) Monte Carlo convergence
rate), and quantifies how much antithetic + control variates reduce that
error relative to a naive (no variance reduction) estimator at the same
sample sizes.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

try:
    from core.black_scholes import BlackScholes
    from core.monte_carlo import MonteCarloPricer
except ImportError:  # pragma: no cover - fallback for direct script execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.black_scholes import BlackScholes
    from core.monte_carlo import MonteCarloPricer

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SIMULATION_SIZES = [1_000, 5_000, 10_000, 50_000, 100_000]


def run_convergence_analysis(
    S: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.20,
    option_type: str = "call",
    seed: int = 42,
) -> pd.DataFrame:
    """Run MC convergence for both naive (antithetic-only) and control-variate pricing.

    Returns:
        DataFrame with columns: n_simulations, price_naive, abs_error_naive,
        price_cv, abs_error_cv, std_error_naive, std_error_cv.
    """
    bs_price = BlackScholes(S, K, T, r, sigma, option_type).price()
    pricer = MonteCarloPricer(S, K, T, r, sigma, option_type=option_type, seed=seed)

    rows: list[dict[str, float]] = []
    for n in SIMULATION_SIZES:
        naive = pricer.price_european(n_simulations=n, use_control_variate=False)
        cv = pricer.price_european(n_simulations=n, use_control_variate=True)
        rows.append(
            {
                "n_simulations": naive.n_simulations,
                "price_naive": naive.price,
                "std_error_naive": naive.std_error,
                "abs_error_naive": abs(naive.price - bs_price),
                "price_cv": cv.price,
                "std_error_cv": cv.std_error,
                "abs_error_cv": abs(cv.price - bs_price),
            }
        )

    return pd.DataFrame(rows)


def plot_convergence(df: pd.DataFrame, bs_price: float, save_path: str) -> None:
    """Plot absolute error vs n_simulations (log-log) for naive vs control-variate MC."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(df["n_simulations"], df["abs_error_naive"], "o-", label="Antithetic only (naive)", color="tab:orange")
    ax.loglog(df["n_simulations"], df["abs_error_cv"], "s-", label="Antithetic + control variate", color="tab:blue")

    ax.set_xlabel("Number of simulations (log scale)")
    ax.set_ylabel("Absolute error vs Black-Scholes price (log scale)")
    ax.set_title(f"Monte Carlo Convergence to Black-Scholes Price (BS = {bs_price:.4f})")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    print("=== Convergence smoke test (European call) ===")
    bs_call_price = BlackScholes(S, K, T, r, sigma, "call").price()
    call_df = run_convergence_analysis(S, K, T, r, sigma, "call")
    print(call_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\n=== Convergence smoke test (European put) ===")
    bs_put_price = BlackScholes(S, K, T, r, sigma, "put").price()
    put_df = run_convergence_analysis(S, K, T, r, sigma, "put")
    print(put_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    save_path = os.path.join(RESULTS_DIR, "convergence.png")
    plot_convergence(call_df, bs_call_price, save_path)
    print(f"\nSaved convergence plot to {save_path}")

    final_naive_err = call_df["abs_error_naive"].iloc[-1]
    final_cv_err = call_df["abs_error_cv"].iloc[-1]
    print(
        f"\nAt n=100000: naive abs error = {final_naive_err:.5f}, "
        f"control-variate abs error = {final_cv_err:.5f}"
    )
