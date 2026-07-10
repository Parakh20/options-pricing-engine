from __future__ import annotations

import math

import pytest

from pricing_engine.calibration.engine import CalibrationEngine
from pricing_engine.calibration.market_quote import MarketQuote
from pricing_engine.models.heston import HestonModel
from pricing_engine.models.sabr import SABRModel


def _synthetic_quotes() -> list[MarketQuote]:
    """Generate quotes from a known Heston model so calibration has a
    ground truth to recover.
    """
    true_params = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma_v": 0.3, "rho": -0.6}
    # LM (Levenberg-Marquardt via scipy least_squares) requires the number of
    # residuals to be >= the number of free parameters (5 here), or scipy
    # raises a ValueError before it can even attempt an iteration. Real
    # Heston calibration to a market chain always uses many more quotes than
    # free parameters (a handful of strikes is not representative), so we use
    # a strike ladder spanning ITM/ATM/OTM rather than a bare 3-quote set —
    # this both satisfies the LM shape requirement and matches how
    # calibration is actually done in practice.
    strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
    quotes = []
    for K in strikes:
        model = HestonModel(S=100.0, K=K, T=1.0, r=0.05, option_type="call", **true_params)
        quotes.append(
            MarketQuote(S=100.0, K=K, T=1.0, r=0.05, option_type="call", market_price=model.price())
        )
    return quotes


def test_lm_calibration_recovers_prices_within_tolerance() -> None:
    quotes = _synthetic_quotes()
    initial_guess = {"v0": 0.05, "kappa": 1.5, "theta": 0.05, "sigma_v": 0.4, "rho": -0.3}

    calibrated = CalibrationEngine().calibrate(
        HestonModel, quotes, initial_guess, method="lm"
    )

    for quote in quotes:
        model = HestonModel(
            S=quote.S, K=quote.K, T=quote.T, r=quote.r, option_type=quote.option_type,
            **{k: v for k, v in calibrated.items() if k in ("v0", "kappa", "theta", "sigma_v", "rho")},
        )
        assert math.isclose(model.price(), quote.market_price, rel_tol=0.05)


def _synthetic_sabr_quotes() -> list[MarketQuote]:
    """SABR pricing is closed-form (Hagan approximation), so this stays fast
    even under "lbfgsb"/"de" global-ish searches, unlike Heston's
    quadrature-based pricer.
    """
    true_params = {"alpha": 0.25, "beta": 0.5, "rho": -0.2, "nu": 0.5}
    strikes = [90.0, 100.0, 110.0]
    quotes = []
    for K in strikes:
        model = SABRModel(F=100.0, K=K, T=1.0, r=0.05, option_type="call", **true_params)
        quotes.append(MarketQuote(S=100.0, K=K, T=1.0, r=0.05, option_type="call", market_price=model.price()))
    return quotes


def test_lbfgsb_calibration_returns_bounded_finite_params() -> None:
    """"lbfgsb" (bounded L-BFGS-B on a scalar sum-of-squares objective) has
    no minimum-residuals-vs-params requirement, unlike "lm" — verified here
    with fewer quotes (3) than SABR's free params (4).
    """
    quotes = _synthetic_sabr_quotes()[:1]
    initial_guess = {"alpha": 0.2, "beta": 0.5, "rho": -0.1, "nu": 0.4}

    calibrated = CalibrationEngine().calibrate(SABRModel, quotes, initial_guess, method="lbfgsb")

    assert 1e-4 <= calibrated["alpha"] <= 2.0
    assert 0.0 <= calibrated["beta"] <= 1.0
    assert -0.999 <= calibrated["rho"] <= 0.999
    assert 1e-4 <= calibrated["nu"] <= 3.0


def test_de_calibration_recovers_prices_within_tolerance() -> None:
    """"de" (differential_evolution) is a global bounded search; sanity
    checked on the fast-pricing SABR model with a small strike ladder.
    """
    quotes = _synthetic_sabr_quotes()
    initial_guess = {"alpha": 0.2, "beta": 0.5, "rho": -0.1, "nu": 0.4}

    calibrated = CalibrationEngine().calibrate(SABRModel, quotes, initial_guess, method="de")

    for quote in quotes:
        model = SABRModel(
            F=quote.S, K=quote.K, T=quote.T, r=quote.r, option_type=quote.option_type, **calibrated,
        )
        assert math.isclose(model.price(), quote.market_price, rel_tol=0.1)


def test_calibrate_raises_on_unknown_method() -> None:
    quotes = _synthetic_sabr_quotes()
    initial_guess = {"alpha": 0.2, "beta": 0.5, "rho": -0.1, "nu": 0.4}

    with pytest.raises(ValueError, match="Unknown calibration method"):
        CalibrationEngine().calibrate(SABRModel, quotes, initial_guess, method="nelder-mead")
