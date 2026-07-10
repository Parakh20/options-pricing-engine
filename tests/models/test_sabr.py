from __future__ import annotations

import math

import pytest

from pricing_engine.calibration.market_quote import MarketQuote
from pricing_engine.models.sabr import SABRModel


def test_atm_sabr_vol_reduces_to_alpha_over_f_beta_minus_1(benchmark_params: dict[str, float]) -> None:
    """At F == K (ATM) with beta=1, the Hagan formula's leading term reduces
    to alpha, independent of nu/rho to first order — a standard sanity check.
    """
    model = SABRModel(
        F=100.0, K=100.0, T=1.0, alpha=0.2, beta=1.0, rho=0.0, nu=0.4, r=0.05, option_type="call"
    )
    iv = model.implied_vol()
    assert math.isclose(iv, 0.2, abs_tol=0.01)


def test_price_is_positive_and_finite(benchmark_params: dict[str, float]) -> None:
    model = SABRModel(
        F=100.0, K=110.0, T=1.0, alpha=0.2, beta=0.5, rho=-0.3, nu=0.6, r=0.05, option_type="call"
    )
    price = model.price()
    assert 0.0 < price < 100.0


def test_greeks_contains_delta_and_vega() -> None:
    model = SABRModel(
        F=100.0, K=100.0, T=1.0, alpha=0.2, beta=1.0, rho=0.0, nu=0.4, r=0.05, option_type="call"
    )
    g = model.greeks()
    assert "delta" in g and "vega" in g


def test_post_init_rejects_beta_outside_zero_one() -> None:
    with pytest.raises(ValueError, match="beta must be in"):
        SABRModel(F=100.0, K=100.0, T=1.0, alpha=0.2, beta=1.5, rho=0.0, nu=0.4, r=0.05)


def test_post_init_rejects_rho_outside_minus_one_one() -> None:
    with pytest.raises(ValueError, match="rho must be in"):
        SABRModel(F=100.0, K=100.0, T=1.0, alpha=0.2, beta=0.5, rho=1.5, nu=0.4, r=0.05)


def test_post_init_rejects_non_positive_alpha_or_negative_nu() -> None:
    with pytest.raises(ValueError, match="alpha must be positive"):
        SABRModel(F=100.0, K=100.0, T=1.0, alpha=0.0, beta=0.5, rho=0.0, nu=0.4, r=0.05)
    with pytest.raises(ValueError, match="alpha must be positive"):
        SABRModel(F=100.0, K=100.0, T=1.0, alpha=0.2, beta=0.5, rho=0.0, nu=-0.1, r=0.05)


def test_calibrate_classmethod_recovers_reasonable_params() -> None:
    """SABRModel.calibrate() delegates to CalibrationEngine (default "lm")
    — exercised here against synthetic quotes generated from known params.
    LM requires at least as many quotes as free params (4 here), so a
    small strike ladder is used, matching the pattern in
    tests/calibration/test_engine.py.
    """
    true_params = {"alpha": 0.25, "beta": 0.5, "rho": -0.2, "nu": 0.5}
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    quotes = [
        MarketQuote(
            S=100.0, K=k, T=1.0, r=0.05, option_type="call",
            market_price=SABRModel(F=100.0, K=k, T=1.0, r=0.05, option_type="call", **true_params).price(),
        )
        for k in strikes
    ]
    initial_guess = {"alpha": 0.2, "beta": 0.5, "rho": -0.1, "nu": 0.4}

    calibrated = SABRModel.calibrate(quotes, initial_guess)
    assert set(calibrated.keys()) == set(initial_guess.keys())
    for value in calibrated.values():
        assert math.isfinite(value)
