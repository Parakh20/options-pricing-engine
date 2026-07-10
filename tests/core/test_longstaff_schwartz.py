from __future__ import annotations

import pytest

from pricing_engine.core.black_scholes import BlackScholes
from pricing_engine.core.longstaff_schwartz import LongstaffSchwartz


class TestValidation:
    """__post_init__ raises ValueError for every invalid input branch."""

    def test_rejects_non_positive_spot(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "S": 0.0}
        with pytest.raises(ValueError, match="S and K must be positive"):
            LongstaffSchwartz(**params)

    def test_rejects_non_positive_strike(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "K": -5.0}
        with pytest.raises(ValueError, match="S and K must be positive"):
            LongstaffSchwartz(**params)

    def test_rejects_non_positive_time(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "T": 0.0}
        with pytest.raises(ValueError, match="T must be strictly positive"):
            LongstaffSchwartz(**params)

    def test_rejects_negative_sigma(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "sigma": -0.2}
        with pytest.raises(ValueError, match="sigma must be non-negative"):
            LongstaffSchwartz(**params)

    def test_rejects_non_positive_n_paths(self, benchmark_params: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="n_paths and n_steps must be positive integers"):
            LongstaffSchwartz(**benchmark_params, n_paths=0)

    def test_rejects_non_positive_n_steps(self, benchmark_params: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="n_paths and n_steps must be positive integers"):
            LongstaffSchwartz(**benchmark_params, n_steps=0)

    def test_rejects_unknown_basis(self, benchmark_params: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="basis must be 'polynomial' or 'laguerre'"):
            LongstaffSchwartz(**benchmark_params, basis="chebyshev")

    def test_rejects_degree_below_one(self, benchmark_params: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="degree must be at least 1"):
            LongstaffSchwartz(**benchmark_params, degree=0)


class TestBasisFunctions:
    def test_polynomial_basis_matrix_matches_monomial_powers(self, benchmark_params: dict[str, float]) -> None:
        import numpy as np

        lsm = LongstaffSchwartz(**benchmark_params, basis="polynomial", degree=2, n_paths=100, n_steps=5, seed=1)
        x = np.array([1.0, 2.0, 3.0])
        design = lsm._basis_matrix(x)
        expected = np.vstack([x**0, x**1, x**2]).T
        assert np.allclose(design, expected)

    def test_polynomial_basis_prices_without_error(self, benchmark_params: dict[str, float]) -> None:
        lsm = LongstaffSchwartz(**benchmark_params, basis="polynomial", degree=3, n_paths=2_000, n_steps=25, seed=7)
        price, boundary_df = lsm.price()
        assert price > 0.0
        assert set(boundary_df.columns) == {"time", "critical_price"}

    def test_laguerre_and_polynomial_bases_agree_reasonably(self, benchmark_params: dict[str, float]) -> None:
        # Both basis families should recover roughly the same American put
        # price for identical paths/seed -- not identical (different
        # regression fits) but within a sane band of each other.
        laguerre = LongstaffSchwartz(**benchmark_params, basis="laguerre", degree=3, n_paths=5_000, n_steps=50, seed=99)
        polynomial = LongstaffSchwartz(**benchmark_params, basis="polynomial", degree=3, n_paths=5_000, n_steps=50, seed=99)
        laguerre_price, _ = laguerre.price()
        polynomial_price, _ = polynomial.price()
        assert laguerre_price == pytest.approx(polynomial_price, abs=0.5)


class TestNoInTheMoneyPathsAtATimeStep:
    def test_deep_otm_put_skips_time_steps_with_no_itm_paths(self, benchmark_params: dict[str, float]) -> None:
        # S far above K: the put almost never crosses into ITM territory at
        # early time steps, exercising the "no ITM paths -> continue" branch
        # for at least some steps in the backward induction.
        params = {**benchmark_params, "S": 500.0, "K": 100.0}
        lsm = LongstaffSchwartz(**params, n_paths=1_000, n_steps=20, seed=3)
        price, boundary_df = lsm.price()
        european_put = BlackScholes(**params, option_type="put").price()
        # Negligible early-exercise value: American ~= European, and the
        # exercise boundary should have far fewer than n_steps rows because
        # most early steps had no ITM paths at all.
        assert price == pytest.approx(european_put, abs=0.5)
        assert len(boundary_df) < 20


class TestAmericanPutProperties:
    def test_deep_itm_put_exceeds_european_price(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "S": 60.0, "K": 100.0}
        lsm = LongstaffSchwartz(**params, n_paths=5_000, n_steps=50, seed=42)
        american_price, _ = lsm.price()
        european_price = BlackScholes(**params, option_type="put").price()
        assert american_price >= european_price - 0.05

    def test_exercise_boundary_is_below_strike(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "S": 100.0, "K": 100.0}
        lsm = LongstaffSchwartz(**params, n_paths=5_000, n_steps=50, seed=42)
        _, boundary_df = lsm.price()
        assert not boundary_df.empty
        assert (boundary_df["critical_price"] < params["K"]).all()
