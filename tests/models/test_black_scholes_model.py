"""Test BlackScholesModel adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from pricing_engine.models.black_scholes_model import BlackScholesModel


def test_price_matches_core_black_scholes(benchmark_params: dict[str, float]) -> None:
    model = BlackScholesModel(**benchmark_params, option_type="call")
    assert math.isclose(model.price(), 10.4506, rel_tol=1e-3)


def test_greeks_returns_delta_and_vega(benchmark_params: dict[str, float]) -> None:
    model = BlackScholesModel(**benchmark_params, option_type="call")
    g = model.greeks()
    assert "delta" in g and "vega" in g


def test_calibrate_raises_on_non_convergence() -> None:
    """Verify calibrate() raises ValueError when solver can't find implied vol.

    This occurs when market_price is below intrinsic value, making it
    impossible to match the price with any sigma in the solver bracket.

    Arrange: Create market_data with a price below intrinsic value
    Act: Call calibrate() with impossible market price
    Assert: ValueError is raised with clear message
    """

    @dataclass
    class MockMarketData:
        """Minimal market data with price below intrinsic value."""

        S: float = 100.0
        K: float = 100.0
        T: float = 1.0
        r: float = 0.05
        option_type: str = "call"
        market_price: float = 1.0  # Below intrinsic value (~4.88 for these params)

    market_data = MockMarketData()

    # The solver should fail to converge because market_price < intrinsic value
    with pytest.raises(ValueError, match="Could not calibrate BlackScholesModel"):
        BlackScholesModel.calibrate(market_data, initial_guess={})
