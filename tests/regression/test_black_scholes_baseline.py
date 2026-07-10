from __future__ import annotations

import math

from core.black_scholes import BlackScholes, check_put_call_parity


def test_call_price_matches_known_benchmark(benchmark_params: dict[str, float]) -> None:
    bs = BlackScholes(**benchmark_params, option_type="call")
    assert math.isclose(bs.price(), 10.4506, rel_tol=1e-3)


def test_put_price_matches_known_benchmark(benchmark_params: dict[str, float]) -> None:
    bs = BlackScholes(**benchmark_params, option_type="put")
    assert math.isclose(bs.price(), 5.5735, rel_tol=1e-3)


def test_greeks_match_known_benchmark(benchmark_params: dict[str, float]) -> None:
    bs = BlackScholes(**benchmark_params, option_type="call")
    greeks = bs.all_greeks()
    assert math.isclose(greeks["delta"], 0.6368, abs_tol=1e-3)
    assert math.isclose(greeks["gamma"], 0.0188, abs_tol=1e-3)
    assert math.isclose(greeks["vega"], 0.3752, abs_tol=1e-3)


def test_put_call_parity_holds(benchmark_params: dict[str, float]) -> None:
    assert check_put_call_parity(**benchmark_params)
