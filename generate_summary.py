"""Generate results/summary.txt: a consolidated benchmark report across all
pricing models (Black-Scholes, Monte Carlo, Longstaff-Schwartz) plus a real
AAPL market validation. Run from the project root: `python3 generate_summary.py`.
"""

from __future__ import annotations

import os

from core.black_scholes import BlackScholes
from core.longstaff_schwartz import LongstaffSchwartz
from core.monte_carlo import MonteCarloPricer
from validation.market_validation import build_validation_table

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


def main() -> None:
    lines: list[str] = []

    lines.append("BENCHMARK PARAMETERS:")
    lines.append(f"  S={S:.0f}, K={K:.0f}, T={T:.1f}, r={r:.2f}, sigma={sigma:.2f}")
    lines.append("")

    bs_call = BlackScholes(S, K, T, r, sigma, "call")
    bs_put = BlackScholes(S, K, T, r, sigma, "put")
    bs_call_price = bs_call.price()

    mc_pricer = MonteCarloPricer(S, K, T, r, sigma, option_type="call", seed=42)
    mc_result = mc_pricer.price_european(n_simulations=100_000, use_control_variate=True)
    mc_error_pct = abs(mc_result.price - bs_call_price) / bs_call_price * 100.0

    lines.append("EUROPEAN CALL PRICING COMPARISON:")
    lines.append(f"  Black-Scholes:      ${bs_call_price:.2f}")
    lines.append(f"  Monte Carlo (100k): ${mc_result.price:.2f} +/- ${1.96 * mc_result.std_error:.2f} (95% CI)")
    lines.append(f"  MC error vs BS:     {mc_error_pct:.2f}%")
    lines.append("")

    greeks = bs_call.all_greeks()
    lines.append("GREEKS AT THE MONEY:")
    lines.append(
        f"  Delta: {greeks['delta']:.2f}  Gamma: {greeks['gamma']:.2f}  "
        f"Vega: {greeks['vega']:.2f}  Theta: {greeks['theta']:.2f}  Rho: {greeks['rho']:.2f}"
    )
    lines.append("")

    european_put_price = bs_put.price()
    lsm = LongstaffSchwartz(S, K, T, r, sigma, n_paths=50_000, n_steps=50, basis="laguerre", degree=3, seed=42)
    american_put_price, _ = lsm.price()
    premium = american_put_price - european_put_price
    premium_pct = premium / european_put_price * 100.0

    lines.append("AMERICAN vs EUROPEAN PUT:")
    lines.append(f"  European: ${european_put_price:.2f}")
    lines.append(f"  American: ${american_put_price:.2f}")
    lines.append(f"  Early exercise premium: ${premium:.2f} ({premium_pct:.2f}%)")
    lines.append("")

    lines.append("MARKET VALIDATION (AAPL):")
    try:
        comparison_df, summary_stats = build_validation_table("AAPL", n_expiries=3)
        lines.append(f"  Options analyzed: {summary_stats['n_options']}")
        lines.append(f"  MAE vs market: ${summary_stats['mae']:.2f}")
        lines.append(f"  Mean IV: {summary_stats['mean_implied_vol']:.2%}")
        lines.append(
            f"  IV smile: [{summary_stats['min_implied_vol']:.2%}] to "
            f"[{summary_stats['max_implied_vol']:.2%}] across strikes"
        )
        bias_note = "overprices" if summary_stats["mean_pct_error"] > 0 else "underprices"
        if abs(summary_stats["mean_pct_error"]) > 5.0:
            lines.append(
                f"  FLAG: model systematically {bias_note} relative to market "
                f"(mean % error {summary_stats['mean_pct_error']:.2f}%)"
            )
    except (ConnectionError, TimeoutError, OSError) as network_error:
        lines.append(f"  Market validation skipped (no network access): {network_error}")

    report = "\n".join(lines) + "\n"
    print(report)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(report)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
