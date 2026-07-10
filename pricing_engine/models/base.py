"""Abstract base class every pricing model (BS, Heston, SABR, LocalVol) implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PricingModel(ABC):
    """Common interface for all pricing models in the platform.

    Subclasses represent a fully-specified model (parameters + contract
    terms) and can price themselves, compute Greeks, and be constructed
    from a calibration fit to market data.
    """

    @abstractmethod
    def price(self) -> float:
        """Return the model price of the option this instance represents."""

    @abstractmethod
    def greeks(self) -> dict[str, float]:
        """Return a dict of Greeks (at minimum: delta, vega)."""

    @classmethod
    @abstractmethod
    def calibrate(
        cls, market_data: Any, initial_guess: dict[str, float]
    ) -> "PricingModel":
        """Fit model parameters to market_data starting from initial_guess."""
