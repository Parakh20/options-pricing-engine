from __future__ import annotations

import math

from core.black_scholes import BlackScholes
from core.implied_vol import ImpliedVolCalculator


def test_recovers_input_volatility(benchmark_params: dict[str, float]) -> None:
    true_sigma = benchmark_params["sigma"]
    price = BlackScholes(**benchmark_params, option_type="call").price()
    calc = ImpliedVolCalculator()
    implied = calc.solve(
        market_price=price,
        S=benchmark_params["S"],
        K=benchmark_params["K"],
        T=benchmark_params["T"],
        r=benchmark_params["r"],
        option_type="call",
    )
    assert implied is not None
    assert math.isclose(implied, true_sigma, abs_tol=1e-4)
