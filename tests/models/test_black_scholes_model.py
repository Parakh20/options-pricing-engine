"""Test BlackScholesModel adapter."""

from __future__ import annotations

import math

from pricing_engine.models.black_scholes_model import BlackScholesModel


def test_price_matches_core_black_scholes(benchmark_params: dict[str, float]) -> None:
    model = BlackScholesModel(**benchmark_params, option_type="call")
    assert math.isclose(model.price(), 10.4506, rel_tol=1e-3)


def test_greeks_returns_delta_and_vega(benchmark_params: dict[str, float]) -> None:
    model = BlackScholesModel(**benchmark_params, option_type="call")
    g = model.greeks()
    assert "delta" in g and "vega" in g
