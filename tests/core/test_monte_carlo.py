from __future__ import annotations

import math

import pytest

from pricing_engine.core.black_scholes import BlackScholes
from pricing_engine.core.monte_carlo import MonteCarloPricer, MonteCarloResult


class TestValidation:
    """__post_init__ raises ValueError for every invalid input branch."""

    def test_rejects_unknown_option_type(self, benchmark_params: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            MonteCarloPricer(**benchmark_params, option_type="strangle")

    def test_rejects_non_positive_spot(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "S": 0.0}
        with pytest.raises(ValueError, match="S and K must be positive"):
            MonteCarloPricer(**params)

    def test_rejects_non_positive_strike(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "K": -1.0}
        with pytest.raises(ValueError, match="S and K must be positive"):
            MonteCarloPricer(**params)

    def test_rejects_non_positive_time(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "T": 0.0}
        with pytest.raises(ValueError, match="T must be strictly positive"):
            MonteCarloPricer(**params)

    def test_rejects_negative_sigma(self, benchmark_params: dict[str, float]) -> None:
        params = {**benchmark_params, "sigma": -0.01}
        with pytest.raises(ValueError, match="sigma must be non-negative"):
            MonteCarloPricer(**params)


class TestMonteCarloResult:
    def test_as_dict_returns_all_fields(self) -> None:
        result = MonteCarloResult(price=1.0, std_error=0.1, ci_lower=0.8, ci_upper=1.2, n_simulations=1000)
        assert result.as_dict() == {
            "price": 1.0,
            "std_error": 0.1,
            "ci_lower": 0.8,
            "ci_upper": 1.2,
            "n_simulations": 1000,
        }


class TestPriceEuropeanPutPayoff:
    def test_put_payoff_matches_black_scholes_within_ci(self, benchmark_params: dict[str, float]) -> None:
        # Exercises the put branch of _payoff (call branch is covered by
        # the existing regression baseline test).
        mc = MonteCarloPricer(**benchmark_params, option_type="put", seed=123)
        result = mc.price_european(n_simulations=50_000, use_control_variate=True)
        bs_price = BlackScholes(**benchmark_params, option_type="put").price()
        assert result.ci_lower - 3 * result.std_error <= bs_price <= result.ci_upper + 3 * result.std_error

    def test_control_variate_reduces_standard_error(self, benchmark_params: dict[str, float]) -> None:
        mc = MonteCarloPricer(**benchmark_params, option_type="call", seed=7)
        with_cv = mc.price_european(n_simulations=50_000, use_control_variate=True)
        mc_no_cv = MonteCarloPricer(**benchmark_params, option_type="call", seed=7)
        without_cv = mc_no_cv.price_european(n_simulations=50_000, use_control_variate=False)
        assert with_cv.std_error < without_cv.std_error

    def test_zero_variance_control_falls_back_to_zero_beta(self) -> None:
        # T -> effectively deterministic terminal price requires sigma=0,
        # which drives Var(control)=0 and exercises the beta_hat=0.0
        # fallback branch instead of dividing by zero.
        params = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.0}
        mc = MonteCarloPricer(**params, option_type="call", seed=1)
        result = mc.price_european(n_simulations=1_000, use_control_variate=True)
        # With zero volatility, the terminal price is deterministic and the
        # payoff is deterministic too, so std_error should be ~0 and price
        # should equal the discounted deterministic payoff.
        forward = 100.0 * math.exp(0.05 * 1.0)
        expected_price = math.exp(-0.05 * 1.0) * max(forward - 100.0, 0.0)
        assert result.price == pytest.approx(expected_price, abs=1e-6)
        assert result.std_error == pytest.approx(0.0, abs=1e-9)


class TestPriceAsian:
    def test_rejects_unknown_strike_type(self, benchmark_params: dict[str, float]) -> None:
        mc = MonteCarloPricer(**benchmark_params, option_type="call", seed=1)
        with pytest.raises(ValueError, match="strike_type must be 'fixed' or 'floating'"):
            mc.price_asian(n_simulations=100, n_steps=10, strike_type="lookback")

    def test_fixed_strike_call_payoff_non_negative(self, benchmark_params: dict[str, float]) -> None:
        mc = MonteCarloPricer(**benchmark_params, option_type="call", seed=1)
        result = mc.price_asian(n_simulations=2_000, n_steps=20, strike_type="fixed")
        assert result.price >= 0.0

    def test_fixed_strike_put_payoff_non_negative(self, benchmark_params: dict[str, float]) -> None:
        mc = MonteCarloPricer(**benchmark_params, option_type="put", seed=1)
        result = mc.price_asian(n_simulations=2_000, n_steps=20, strike_type="fixed")
        assert result.price >= 0.0

    def test_floating_strike_call_payoff_non_negative(self, benchmark_params: dict[str, float]) -> None:
        mc = MonteCarloPricer(**benchmark_params, option_type="call", seed=1)
        result = mc.price_asian(n_simulations=2_000, n_steps=20, strike_type="floating")
        assert result.price >= 0.0

    def test_floating_strike_put_payoff_non_negative(self, benchmark_params: dict[str, float]) -> None:
        mc = MonteCarloPricer(**benchmark_params, option_type="put", seed=1)
        result = mc.price_asian(n_simulations=2_000, n_steps=20, strike_type="floating")
        assert result.price >= 0.0

    def test_asian_call_price_below_european_call_price(self, benchmark_params: dict[str, float]) -> None:
        # Averaging reduces variance of the terminal-like payoff, so a
        # fixed-strike arithmetic Asian call should be cheaper than the
        # equivalent European call (well-known inequality).
        mc = MonteCarloPricer(**benchmark_params, option_type="call", seed=55)
        asian = mc.price_asian(n_simulations=20_000, n_steps=50, strike_type="fixed")
        european = BlackScholes(**benchmark_params, option_type="call").price()
        assert asian.price < european

    def test_asian_result_shape(self, benchmark_params: dict[str, float]) -> None:
        mc = MonteCarloPricer(**benchmark_params, option_type="call", seed=1)
        result = mc.price_asian(n_simulations=1_000, n_steps=10, strike_type="fixed")
        assert result.n_simulations >= 1_000
        assert result.ci_lower <= result.price <= result.ci_upper


class TestConvergenceAnalysis:
    def test_convergence_dataframe_has_expected_columns_and_shrinking_error(
        self, benchmark_params: dict[str, float]
    ) -> None:
        mc = MonteCarloPricer(**benchmark_params, option_type="call", seed=17)
        df = mc.convergence_analysis(simulation_sizes=[500, 50_000])
        assert list(df.columns) == ["n_simulations", "price", "std_error", "abs_error_vs_bs"]
        assert len(df) == 2
        # Larger sample size should generally produce a smaller standard
        # error (variance shrinks as 1/n).
        assert df.iloc[1]["std_error"] < df.iloc[0]["std_error"]

    def test_convergence_uses_default_simulation_sizes(self, benchmark_params: dict[str, float]) -> None:
        mc = MonteCarloPricer(**benchmark_params, option_type="call", seed=17)
        df = mc.convergence_analysis(simulation_sizes=None, use_control_variate=False)
        assert len(df) == 5
