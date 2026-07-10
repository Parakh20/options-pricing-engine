from __future__ import annotations

from pricing_engine.core.black_scholes import BlackScholes
from pricing_engine.core.longstaff_schwartz import LongstaffSchwartz


def test_american_put_premium_over_european(benchmark_params: dict[str, float]) -> None:
    european_put = BlackScholes(**benchmark_params, option_type="put").price()
    lsm = LongstaffSchwartz(**benchmark_params)
    american_price, _ = lsm.price()
    assert american_price >= european_put - 1e-6
