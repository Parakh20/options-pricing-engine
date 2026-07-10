# tests/models/test_heston.py
from __future__ import annotations

import math

from pricing_engine.core.black_scholes import BlackScholes
from pricing_engine.models.heston import HestonModel


def test_heston_reduces_to_black_scholes_as_vol_of_vol_vanishes() -> None:
    """With sigma_v -> 0 and v0 == theta, Heston collapses to constant-vol
    GBM, so the price should converge to Black-Scholes with sigma=sqrt(v0).
    """
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    v0 = theta = 0.04  # sigma = 0.20
    heston = HestonModel(
        S=S, K=K, T=T, r=r, v0=v0, kappa=2.0, theta=theta, sigma_v=1e-4, rho=0.0, option_type="call"
    )
    bs = BlackScholes(S, K, T, r, math.sqrt(v0), "call")
    assert math.isclose(heston.price(), bs.price(), rel_tol=1e-2)


def test_price_is_positive_and_finite() -> None:
    heston = HestonModel(
        S=100.0, K=100.0, T=1.0, r=0.05, v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7,
        option_type="call",
    )
    price = heston.price()
    assert 0.0 < price < 100.0


def test_greeks_contains_delta_and_vega() -> None:
    heston = HestonModel(
        S=100.0, K=100.0, T=1.0, r=0.05, v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7,
        option_type="call",
    )
    g = heston.greeks()
    assert "delta" in g and "vega" in g
