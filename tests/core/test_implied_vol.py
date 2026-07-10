from __future__ import annotations

import math

import pandas as pd
import pytest

from pricing_engine.core.black_scholes import BlackScholes
from pricing_engine.core.implied_vol import ImpliedVolCalculator


class TestNewtonRaphsonEdgeCases:
    def test_returns_none_for_non_positive_time(self, benchmark_params: dict[str, float]) -> None:
        calc = ImpliedVolCalculator()
        assert calc.newton_raphson(10.0, benchmark_params["S"], benchmark_params["K"], 0.0, 0.05, "call") is None

    def test_returns_none_for_non_positive_spot(self) -> None:
        calc = ImpliedVolCalculator()
        assert calc.newton_raphson(10.0, 0.0, 100.0, 1.0, 0.05, "call") is None

    def test_returns_none_for_non_positive_strike(self) -> None:
        calc = ImpliedVolCalculator()
        assert calc.newton_raphson(10.0, 100.0, 0.0, 1.0, 0.05, "call") is None

    def test_returns_none_when_price_below_intrinsic(self) -> None:
        # A near-zero market price for a deep ITM call is below the
        # discounted intrinsic value floor -- no sigma can produce it.
        calc = ImpliedVolCalculator()
        assert calc.newton_raphson(0.0001, S=100.0, K=50.0, T=1.0, r=0.05, option_type="call") is None

    def test_returns_none_when_bs_construction_raises(self) -> None:
        # An invalid sigma_init causes BlackScholes.__post_init__ to raise
        # ValueError on the first iteration; the except clause should
        # convert that into a clean None return rather than propagating.
        calc = ImpliedVolCalculator()
        result = calc.newton_raphson(10.0, 100.0, 100.0, 1.0, 0.05, "call", sigma_init=-0.1)
        assert result is None

    def test_returns_none_when_vega_collapses(self) -> None:
        # Deep OTM, short-dated call: vega is essentially zero everywhere
        # near the initial guess, so the Newton step is unreliable.
        calc = ImpliedVolCalculator()
        result = calc.newton_raphson(1e-6, S=100.0, K=300.0, T=0.05, r=0.05, option_type="call", sigma_init=0.05)
        assert result is None

    def test_returns_none_when_iteration_budget_exhausted(self) -> None:
        # A single allowed iteration starting far from the true sigma
        # cannot converge within tol, so the loop falls through to None.
        calc = ImpliedVolCalculator(max_iter_newton=1)
        price = BlackScholes(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.5, option_type="call").price()
        result = calc.newton_raphson(price, 100.0, 100.0, 1.0, 0.05, "call", sigma_init=0.01)
        assert result is None

    def test_recovers_put_volatility(self, benchmark_params: dict[str, float]) -> None:
        true_sigma = benchmark_params["sigma"]
        price = BlackScholes(**benchmark_params, option_type="put").price()
        calc = ImpliedVolCalculator()
        implied = calc.newton_raphson(
            price, benchmark_params["S"], benchmark_params["K"], benchmark_params["T"],
            benchmark_params["r"], "put",
        )
        assert implied is not None
        assert math.isclose(implied, true_sigma, abs_tol=1e-4)


class TestBisectionEdgeCases:
    def test_returns_none_for_non_positive_time(self) -> None:
        calc = ImpliedVolCalculator()
        assert calc.bisection(10.0, 100.0, 100.0, 0.0, 0.05, "call") is None

    def test_returns_lower_bound_when_price_matches_exactly(self) -> None:
        calc = ImpliedVolCalculator()
        lo_price = BlackScholes(S=100.0, K=100.0, T=1.0, r=0.05, sigma=calc.sigma_lower, option_type="call").price()
        assert calc.bisection(lo_price, 100.0, 100.0, 1.0, 0.05, "call") == calc.sigma_lower

    def test_returns_upper_bound_when_price_matches_exactly(self) -> None:
        calc = ImpliedVolCalculator()
        hi_price = BlackScholes(S=100.0, K=100.0, T=1.0, r=0.05, sigma=calc.sigma_upper, option_type="call").price()
        assert calc.bisection(hi_price, 100.0, 100.0, 1.0, 0.05, "call") == calc.sigma_upper

    def test_returns_none_when_price_unreachable_in_bracket(self) -> None:
        # A market price above what even sigma_upper can produce is not
        # bracketed -- f_lo and f_hi have the same sign.
        calc = ImpliedVolCalculator()
        huge_price = 10_000.0
        assert calc.bisection(huge_price, 100.0, 100.0, 1.0, 0.05, "call") is None

    def test_recovers_known_sigma_via_bisection_loop(self, benchmark_params: dict[str, float]) -> None:
        true_sigma = benchmark_params["sigma"]
        price = BlackScholes(**benchmark_params, option_type="call").price()
        calc = ImpliedVolCalculator()
        implied = calc.bisection(
            price, benchmark_params["S"], benchmark_params["K"], benchmark_params["T"],
            benchmark_params["r"], "call",
        )
        assert implied is not None
        assert math.isclose(implied, true_sigma, abs_tol=1e-4)


class TestSolveFallback:
    def test_solve_returns_none_when_both_methods_fail(self) -> None:
        calc = ImpliedVolCalculator()
        # Below intrinsic value: newton_raphson short-circuits to None, and
        # bisection cannot bracket it either.
        assert calc.solve(0.0001, S=100.0, K=50.0, T=1.0, r=0.05, option_type="call") is None


class TestBuildIvSurface:
    def test_uses_scalar_s_and_r_when_columns_absent(self) -> None:
        calc = ImpliedVolCalculator()
        strikes = [90.0, 100.0, 110.0]
        rows = [
            {
                "strike": k,
                "T": 1.0,
                "market_price": BlackScholes(S=100.0, K=k, T=1.0, r=0.05, sigma=0.20, option_type="call").price(),
                "option_type": "call",
            }
            for k in strikes
        ]
        chain = pd.DataFrame(rows)
        surface = calc.build_iv_surface(chain, S=100.0, r=0.05)

        assert list(surface.index) == strikes
        assert list(surface.columns) == [1.0]
        for k in strikes:
            assert surface.loc[k, 1.0] == pytest.approx(0.20, abs=1e-3)

    def test_per_row_s_and_r_columns_take_precedence(self) -> None:
        calc = ImpliedVolCalculator()
        rows = [
            {
                "strike": 100.0,
                "T": 1.0,
                "S": 100.0,
                "r": 0.05,
                "market_price": BlackScholes(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, option_type="call").price(),
                "option_type": "call",
            },
            {
                "strike": 100.0,
                "T": 2.0,
                "S": 110.0,
                "r": 0.03,
                "market_price": BlackScholes(S=110.0, K=100.0, T=2.0, r=0.03, sigma=0.15, option_type="call").price(),
                "option_type": "call",
            },
        ]
        chain = pd.DataFrame(rows)
        # Scalar S/r passed here should be ignored since the chain already
        # carries its own S/r columns.
        surface = calc.build_iv_surface(chain, S=999.0, r=999.0)

        assert surface.loc[100.0, 1.0] == pytest.approx(0.25, abs=1e-3)
        assert surface.loc[100.0, 2.0] == pytest.approx(0.15, abs=1e-3)

    def test_raises_when_s_not_resolvable(self) -> None:
        calc = ImpliedVolCalculator()
        chain = pd.DataFrame([{"strike": 100.0, "T": 1.0, "market_price": 10.0, "option_type": "call"}])
        with pytest.raises(ValueError, match="S must be provided"):
            calc.build_iv_surface(chain, S=None, r=0.05)

    def test_raises_when_r_not_resolvable(self) -> None:
        calc = ImpliedVolCalculator()
        chain = pd.DataFrame([{"strike": 100.0, "T": 1.0, "market_price": 10.0, "option_type": "call"}])
        with pytest.raises(ValueError, match="r must be provided"):
            calc.build_iv_surface(chain, S=100.0, r=None)

    def test_unrecoverable_rows_produce_nan_cells(self) -> None:
        calc = ImpliedVolCalculator()
        # A negative market_price is below intrinsic value (which floors at
        # 0), so neither Newton-Raphson nor bisection can bracket/solve it.
        # The row should fail gracefully and leave NaN, not raise.
        chain = pd.DataFrame(
            [{"strike": 300.0, "T": 0.01, "market_price": -1.0, "option_type": "call"}]
        )
        surface = calc.build_iv_surface(chain, S=100.0, r=0.05)
        assert surface.isna().all().all()
