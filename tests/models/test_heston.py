# tests/models/test_heston.py
from __future__ import annotations

import math

import pytest

from pricing_engine.calibration.market_quote import MarketQuote
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


def test_put_price_matches_put_call_parity() -> None:
    """Heston put is derived via put-call parity: put = call - S + K*exp(-rT)."""
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    params = dict(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
    call = HestonModel(S=S, K=K, T=T, r=r, option_type="call", **params)
    put = HestonModel(S=S, K=K, T=T, r=r, option_type="put", **params)
    expected_put = call.price() - S + K * math.exp(-r * T)
    assert math.isclose(put.price(), expected_put, rel_tol=1e-6)


def test_post_init_rejects_non_positive_v0_kappa_theta() -> None:
    with pytest.raises(ValueError, match="v0, kappa, theta must be positive"):
        HestonModel(S=100.0, K=100.0, T=1.0, r=0.05, v0=0.0, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
    with pytest.raises(ValueError, match="v0, kappa, theta must be positive"):
        HestonModel(S=100.0, K=100.0, T=1.0, r=0.05, v0=0.04, kappa=0.0, theta=0.04, sigma_v=0.3, rho=-0.7)


def test_post_init_rejects_rho_outside_minus_one_one() -> None:
    with pytest.raises(ValueError, match="rho must be in"):
        HestonModel(S=100.0, K=100.0, T=1.0, r=0.05, v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=1.5)


def test_calibrate_classmethod_returns_finite_params() -> None:
    """HestonModel.calibrate() delegates to CalibrationEngine (default "lm").
    LM requires at least as many quotes as free params (5), so a small
    strike ladder is used, matching tests/calibration/test_engine.py.
    """
    true_params = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma_v": 0.3, "rho": -0.6}
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0, 115.0]
    quotes = [
        MarketQuote(
            S=100.0, K=k, T=1.0, r=0.05, option_type="call",
            market_price=HestonModel(S=100.0, K=k, T=1.0, r=0.05, option_type="call", **true_params).price(),
        )
        for k in strikes
    ]
    initial_guess = {"v0": 0.05, "kappa": 1.5, "theta": 0.05, "sigma_v": 0.4, "rho": -0.3}

    calibrated = HestonModel.calibrate(quotes, initial_guess)
    assert set(calibrated.keys()) == set(initial_guess.keys())
    for value in calibrated.values():
        assert math.isfinite(value)
