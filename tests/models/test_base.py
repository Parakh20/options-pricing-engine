"""Test PricingModel abstract base class."""

from __future__ import annotations

import pytest

from pricing_engine.models.base import PricingModel


def test_pricing_model_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        PricingModel()  # type: ignore[abstract]
