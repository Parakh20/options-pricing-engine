"""Visualisations for the options pricing engine: Greeks surfaces, vol smile,
Monte Carlo convergence, early exercise boundary, and American vs European
premium. All plots are saved as PNG files under ``results/``.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

try:
    from core.black_scholes import BlackScholes
    from core.longstaff_schwartz import LongstaffSchwartz
    from core.implied_vol import ImpliedVolCalculator
except ImportError:  # pragma: no cover - fallback for direct script execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.black_scholes import BlackScholes
    from core.longstaff_schwartz import LongstaffSchwartz
    from core.implied_vol import ImpliedVolCalculator

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
GREEK_NAMES = ["price", "delta", "gamma", "vega", "theta", "rho"]


def _save(fig: plt.Figure, filename: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_greeks_surfaces(
    K: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.20,
    spot_range: tuple[float, float] = (60.0, 140.0),
    expiry_range: tuple[float, float] = (0.05, 2.0),
    n_points: int = 40,
) -> dict[str, str]:
    """Plot price + 5 Greeks as 3D surfaces (x=spot, y=time to expiry).

    Produces two figures (call, put), each a 2x3 grid of surfaces.

    Returns:
        Dict mapping ``"call"``/``"put"`` to the saved file path.
    """
    spots = np.linspace(*spot_range, n_points)
    expiries = np.linspace(*expiry_range, n_points)
    S_grid, T_grid = np.meshgrid(spots, expiries)

    saved_paths: dict[str, str] = {}

    for option_type in ("call", "put"):
        surfaces = {name: np.zeros_like(S_grid) for name in GREEK_NAMES}

        for i in range(S_grid.shape[0]):
            for j in range(S_grid.shape[1]):
                bs = BlackScholes(S=S_grid[i, j], K=K, T=T_grid[i, j], r=r, sigma=sigma, option_type=option_type)
                surfaces["price"][i, j] = bs.price()
                greeks = bs.all_greeks()
                for name in GREEK_NAMES[1:]:
                    surfaces[name][i, j] = greeks[name]

        fig = plt.figure(figsize=(16, 10))
        for idx, name in enumerate(GREEK_NAMES):
            ax = fig.add_subplot(2, 3, idx + 1, projection="3d")
            ax.plot_surface(S_grid, T_grid, surfaces[name], cmap="viridis", linewidth=0, antialiased=True)
            ax.set_xlabel("Spot price")
            ax.set_ylabel("Time to expiry (yrs)")
            ax.set_zlabel(name.capitalize())
            ax.set_title(f"{option_type.capitalize()} {name.capitalize()}")

        fig.suptitle(f"Black-Scholes Greeks Surfaces ({option_type.capitalize()}, K={K})", fontsize=14)
        saved_paths[option_type] = _save(fig, f"greeks_surface_{option_type}.png")

    return saved_paths


def plot_vol_smile(
    S: float = 100.0,
    r: float = 0.05,
    strikes: list[float] | None = None,
    expiries: list[float] | None = None,
    realized_vol: float = 0.20,
) -> str:
    """Plot a synthetic implied volatility smile across strikes, one line per expiry.

    A synthetic smile (vol rising away from ATM, mild term decay) is used so
    this plot is runnable without live market data. Overlays a flat
    horizontal line at ``realized_vol`` for comparison against the
    flat-vol (pure Black-Scholes) assumption.
    """
    if strikes is None:
        strikes = list(np.linspace(70.0, 130.0, 13))
    if expiries is None:
        expiries = [0.25, 0.5, 1.0]

    def synthetic_sigma(strike: float, expiry: float) -> float:
        moneyness = (strike - S) / S
        smile = 0.20 + 0.5 * moneyness**2
        term_decay = 0.02 * expiry
        return smile - term_decay

    iv_calc = ImpliedVolCalculator()
    fig, ax = plt.subplots(figsize=(9, 6))

    for T in expiries:
        ivs = []
        for K in strikes:
            true_sigma = synthetic_sigma(K, T)
            price = BlackScholes(S=S, K=K, T=T, r=r, sigma=true_sigma, option_type="call").price()
            iv = iv_calc.solve(market_price=price, S=S, K=K, T=T, r=r, option_type="call")
            ivs.append(iv)
        ax.plot(strikes, ivs, "o-", label=f"T={T:.2f}y")

    ax.axhline(realized_vol, color="gray", linestyle="--", label=f"Flat vol assumption ({realized_vol:.0%})")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility")
    ax.set_title("Implied Volatility Smile (synthetic)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    return _save(fig, "vol_smile.png")


def plot_mc_convergence(call_df: pd.DataFrame, bs_price: float) -> str:
    """Plot MC absolute error vs n_simulations (log-log), naive vs control variate.

    Args:
        call_df: DataFrame as returned by
            ``validation.convergence.run_convergence_analysis``, with
            columns ``n_simulations``, ``abs_error_naive``, ``abs_error_cv``.
        bs_price: The exact Black-Scholes price used as the convergence target.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(call_df["n_simulations"], call_df["abs_error_naive"], "o-", label="Antithetic only (naive)")
    ax.loglog(call_df["n_simulations"], call_df["abs_error_cv"], "s-", label="Antithetic + control variate")
    ax.set_xlabel("Number of simulations (log scale)")
    ax.set_ylabel("Absolute error vs Black-Scholes price (log scale)")
    ax.set_title(f"Monte Carlo Convergence (BS = {bs_price:.4f})")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    return _save(fig, "mc_convergence.png")


def plot_early_exercise_boundary(
    S: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.20,
    n_paths: int = 50_000,
    n_steps: int = 50,
    seed: int = 42,
) -> str:
    """Plot the LSM early-exercise boundary: critical stock price vs time to expiry."""
    lsm = LongstaffSchwartz(S=S, K=K, T=T, r=r, sigma=sigma, n_paths=n_paths, n_steps=n_steps, seed=seed)
    _, boundary_df = lsm.price()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(boundary_df["time"], boundary_df["critical_price"], "o-", color="tab:red")
    ax.set_xlabel("Time to expiry (years)")
    ax.set_ylabel("Critical stock price (exercise if S below this)")
    ax.set_title(f"American Put Early Exercise Boundary (K={K}, sigma={sigma:.0%})")
    ax.grid(True, linestyle="--", alpha=0.5)
    return _save(fig, "early_exercise_boundary.png")


def plot_american_vs_european_premium(
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.20,
    spot_range: tuple[float, float] = (60.0, 140.0),
    n_points: int = 15,
    n_paths: int = 20_000,
    n_steps: int = 50,
    seed: int = 42,
) -> str:
    """Plot the American-European put premium spread as spot price (moneyness) varies."""
    spots = np.linspace(*spot_range, n_points)
    european_prices = []
    american_prices = []

    for S in spots:
        european_prices.append(BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma, option_type="put").price())
        lsm = LongstaffSchwartz(S=S, K=K, T=T, r=r, sigma=sigma, n_paths=n_paths, n_steps=n_steps, seed=seed)
        price, _ = lsm.price()
        american_prices.append(price)

    european_prices = np.array(european_prices)
    american_prices = np.array(american_prices)
    premium = american_prices - european_prices

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(spots, european_prices, "o-", label="European put (BS)")
    ax1.plot(spots, american_prices, "s-", label="American put (LSM)")
    ax1.axvline(K, color="gray", linestyle="--", alpha=0.5, label=f"Strike K={K}")
    ax1.set_xlabel("Spot price")
    ax1.set_ylabel("Put price")
    ax1.set_title("American vs European Put Price")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(spots, premium, "o-", color="tab:green")
    ax2.axvline(K, color="gray", linestyle="--", alpha=0.5, label=f"Strike K={K}")
    ax2.set_xlabel("Spot price")
    ax2.set_ylabel("Early exercise premium (American - European)")
    ax2.set_title("Early Exercise Premium vs Moneyness")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    return _save(fig, "american_vs_european_premium.png")


if __name__ == "__main__":
    print("=== Visualisation smoke test ===")

    print("Plotting Greeks surfaces...")
    greek_paths = plot_greeks_surfaces(n_points=20)
    print(f"  Saved: {greek_paths}")

    print("Plotting volatility smile...")
    smile_path = plot_vol_smile()
    print(f"  Saved: {smile_path}")

    print("Plotting early exercise boundary...")
    boundary_path = plot_early_exercise_boundary(n_paths=20_000)
    print(f"  Saved: {boundary_path}")

    print("Plotting American vs European premium...")
    premium_path = plot_american_vs_european_premium(n_paths=10_000, n_points=10)
    print(f"  Saved: {premium_path}")

    print("\nAll visualisation smoke tests completed.")
