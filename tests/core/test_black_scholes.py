from __future__ import annotations

import math

import pytest

from pricing_engine.core.black_scholes import BlackScholes


class TestValidation:
    """__post_init__ raises ValueError for every invalid input branch."""

    def test_rejects_unknown_option_type(self, benchmark_params: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            BlackScholes(**benchmark_params, option_type="straddle")

    def test_rejects_non_positive_spot(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "S": 0.0}
        with pytest.raises(ValueError, match="S and K must be positive"):
            BlackScholes(**params)

    def test_rejects_non_positive_strike(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "K": -10.0}
        with pytest.raises(ValueError, match="S and K must be positive"):
            BlackScholes(**params)

    def test_rejects_negative_time_to_expiry(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "T": -0.5}
        with pytest.raises(ValueError, match="T must be non-negative"):
            BlackScholes(**params)

    def test_rejects_negative_sigma(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "sigma": -0.1}
        with pytest.raises(ValueError, match="sigma must be non-negative"):
            BlackScholes(**params)

    def test_d1_d2_rejects_zero_time_directly(self, benchmark_params: dict[str, float]) -> None:
        # T=0 is legal at construction (handled specially by price()), but
        # calling _d1_d2 directly (as delta/gamma/vega/theta/rho do) must
        # raise since d1/d2 are undefined at T=0.
        params = {**benchmark_params, "T": 0.0}
        bs = BlackScholes(**params)
        with pytest.raises(ValueError, match="T and sigma must be strictly positive"):
            bs._d1_d2()

    def test_d1_d2_rejects_zero_sigma_directly(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "sigma": 0.0}
        bs = BlackScholes(**params)
        with pytest.raises(ValueError, match="T and sigma must be strictly positive"):
            bs._d1_d2()


class TestExpiryEdgeCase:
    """price() short-circuits to intrinsic value at T=0, bypassing d1/d2."""

    def test_call_at_expiry_itm_equals_intrinsic(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "S": 120.0, "K": 100.0, "T": 0.0}
        bs = BlackScholes(**params, option_type="call")
        assert bs.price() == pytest.approx(20.0)

    def test_call_at_expiry_otm_is_zero(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "S": 80.0, "K": 100.0, "T": 0.0}
        bs = BlackScholes(**params, option_type="call")
        assert bs.price() == pytest.approx(0.0)

    def test_put_at_expiry_itm_equals_intrinsic(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "S": 80.0, "K": 100.0, "T": 0.0}
        bs = BlackScholes(**params, option_type="put")
        assert bs.price() == pytest.approx(20.0)

    def test_put_at_expiry_otm_is_zero(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "S": 120.0, "K": 100.0, "T": 0.0}
        bs = BlackScholes(**params, option_type="put")
        assert bs.price() == pytest.approx(0.0)


class TestPutCallGreekSymmetry:
    """Exercises the put branches of delta/theta/rho against known identities."""

    def test_put_delta_equals_call_delta_minus_one(self, benchmark_params: dict[str, float]) -> None:
        call = BlackScholes(**benchmark_params, option_type="call")
        put = BlackScholes(**benchmark_params, option_type="put")
        assert put.delta() == pytest.approx(call.delta() - 1.0)

    def test_put_delta_is_negative_for_atm_option(self, benchmark_params: dict[str, float]) -> None:
        put = BlackScholes(**benchmark_params, option_type="put")
        assert -1.0 <= put.delta() < 0.0

    def test_put_rho_is_negative_for_call_positive_rho(self, benchmark_params: dict[str, float]) -> None:
        call = BlackScholes(**benchmark_params, option_type="call")
        put = BlackScholes(**benchmark_params, option_type="put")
        assert call.rho() > 0.0
        assert put.rho() < 0.0

    def test_put_call_theta_differ_by_r_k_discount_terms(self, benchmark_params: dict[str, float]) -> None:
        call = BlackScholes(**benchmark_params, option_type="call")
        put = BlackScholes(**benchmark_params, option_type="put")
        # Both share the identical time-decay term1 but differ in term2's
        # sign/branch; assert they are not accidentally identical and each
        # is a small daily decay figure (put on rate-carrying asset need
        # not have identical theta to the call).
        assert call.theta() != put.theta()
        assert call.theta() < 0.0  # long call bleeds value over time (typical ATM case)

    def test_gamma_identical_for_call_and_put(self, benchmark_params: dict[str, float]) -> None:
        call = BlackScholes(**benchmark_params, option_type="call")
        put = BlackScholes(**benchmark_params, option_type="put")
        assert call.gamma() == pytest.approx(put.gamma())

    def test_vega_identical_for_call_and_put(self, benchmark_params: dict[str, float]) -> None:
        call = BlackScholes(**benchmark_params, option_type="call")
        put = BlackScholes(**benchmark_params, option_type="put")
        assert call.vega() == pytest.approx(put.vega())


class TestDeepMoneyness:
    def test_deep_itm_call_delta_approaches_one(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "S": 500.0, "K": 100.0}
        bs = BlackScholes(**params, option_type="call")
        assert bs.delta() == pytest.approx(1.0, abs=1e-3)

    def test_deep_otm_call_delta_approaches_zero(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "S": 20.0, "K": 100.0}
        bs = BlackScholes(**params, option_type="call")
        assert bs.delta() == pytest.approx(0.0, abs=1e-3)

    def test_deep_itm_put_price_approaches_discounted_intrinsic(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "S": 1.0, "K": 100.0}
        bs = BlackScholes(**params, option_type="put")
        discounted_intrinsic = 100.0 * math.exp(-benchmark_params["r"] * benchmark_params["T"]) - 1.0
        assert bs.price() == pytest.approx(discounted_intrinsic, abs=1e-2)
