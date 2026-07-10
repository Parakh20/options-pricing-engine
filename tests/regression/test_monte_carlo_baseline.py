from __future__ import annotations

import math

from pricing_engine.core.black_scholes import BlackScholes
from pricing_engine.core.monte_carlo import MonteCarloPricer


def test_european_call_converges_to_black_scholes(benchmark_params: dict[str, float]) -> None:
    bs_price = BlackScholes(**benchmark_params, option_type="call").price()
    pricer = MonteCarloPricer(**benchmark_params, option_type="call", seed=42)
    result = pricer.price_european(n_simulations=100_000)
    assert math.isclose(result.price, bs_price, rel_tol=0.01)
