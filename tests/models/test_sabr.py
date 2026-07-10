from __future__ import annotations

import math

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
