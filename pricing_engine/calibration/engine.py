"""Model-agnostic calibration engine: fits a PricingModel subclass's free
parameters to a set of market quotes by minimizing squared price error.

Supports three optimizers:
  - "lm":      scipy.optimize.least_squares (Levenberg-Marquardt) — fast,
               local, default choice when the initial guess is reasonable.
  - "lbfgsb":  scipy.optimize.minimize(method="L-BFGS-B") — bounded local
               search, useful when parameters have hard bounds.
  - "de":      scipy.optimize.differential_evolution — global search, slower,
               used when the initial guess may be far from the optimum.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, least_squares, minimize

from pricing_engine.calibration.market_quote import MarketQuote

# Reasonable bounds per free parameter, used by "lbfgsb" and "de".
_DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "v0": (1e-4, 2.0),
    "kappa": (1e-3, 10.0),
    "theta": (1e-4, 2.0),
    "sigma_v": (1e-4, 3.0),
    "rho": (-0.999, 0.999),
    "alpha": (1e-4, 2.0),
    "beta": (0.0, 1.0),
    "nu": (1e-4, 3.0),
}


class CalibrationEngine:
    def _residuals(
        self, param_values: np.ndarray, param_names: list[str], model_cls: type,
        quotes: list[MarketQuote],
    ) -> np.ndarray:
        params = dict(zip(param_names, param_values))
        # Most models take a spot price as `S`, but SABRModel takes a
        # forward price as `F` instead. Inspect the constructor so this
        # engine can calibrate either without a model-specific branch;
        # MarketQuote only carries `S`, which is used as the forward price
        # for models with no `S` parameter (consistent with how callers
        # price SABR off spot-as-forward elsewhere, e.g. the model
        # comparison report).
        spot_kwarg = "S" if "S" in inspect.signature(model_cls).parameters else "F"
        errors = []
        for quote in quotes:
            try:
                model = model_cls(
                    **{spot_kwarg: quote.S}, K=quote.K, T=quote.T, r=quote.r,
                    option_type=quote.option_type, **params,
                )
                errors.append(model.price() - quote.market_price)
            except ValueError:
                # "lm" (scipy least_squares(method="lm")) has no bounds
                # support, so the objective must stay defined everywhere in
                # R^n — unconstrained trial steps can land on parameter
                # combinations a model rejects as invalid (e.g. negative
                # Heston variance). Return a large penalty instead of letting
                # the ValueError propagate, so LM is steered back toward the
                # feasible region rather than crashing.
                errors.append(1e6)
        return np.array(errors)

    def calibrate(
        self,
        model_cls: type,
        quotes: list[MarketQuote],
        initial_guess: dict[str, float],
        method: str = "lm",
    ) -> dict[str, float]:
        """Returns the calibrated parameter dict (not a model instance, since
        market_data here is a list of quotes across strikes, not a single
        contract) — callers construct a model per-quote from the result.
        """
        param_names = list(initial_guess.keys())
        x0 = np.array([initial_guess[name] for name in param_names])
        bounds_lower = [_DEFAULT_BOUNDS.get(name, (-np.inf, np.inf))[0] for name in param_names]
        bounds_upper = [_DEFAULT_BOUNDS.get(name, (-np.inf, np.inf))[1] for name in param_names]

        if method == "lm":
            result = least_squares(
                self._residuals, x0, args=(param_names, model_cls, quotes), method="lm",
            )
            fitted = result.x
        elif method == "lbfgsb":
            def objective(x: np.ndarray) -> float:
                return float(np.sum(self._residuals(x, param_names, model_cls, quotes) ** 2))

            result = minimize(
                objective, x0, method="L-BFGS-B", bounds=list(zip(bounds_lower, bounds_upper)),
            )
            fitted = result.x
        elif method == "de":
            def objective(x: np.ndarray) -> float:
                return float(np.sum(self._residuals(x, param_names, model_cls, quotes) ** 2))

            result = differential_evolution(objective, bounds=list(zip(bounds_lower, bounds_upper)))
            fitted = result.x
        else:
            raise ValueError(f"Unknown calibration method: {method!r}")

        return dict(zip(param_names, fitted))
