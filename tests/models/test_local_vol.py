# tests/models/test_local_vol.py
from __future__ import annotations

import math

import numpy as np

from pricing_engine.core.black_scholes import BlackScholes
from pricing_engine.models.local_vol import LocalVolModel, build_local_vol_surface


def _flat_iv_surface(sigma: float):
    """A constant-IV surface: local vol should reduce to that constant."""

    def iv(K: float, T: float) -> float:
        return sigma

    return iv


def test_local_vol_matches_dupire_formula_for_flat_surface() -> None:
    iv_surface = _flat_iv_surface(0.20)
    local_vol = build_local_vol_surface(iv_surface, S=100.0, r=0.05)
    # A flat IV surface implies a flat local vol surface at the same level.
    assert math.isclose(local_vol(100.0, 1.0), 0.20, abs_tol=0.03)


def test_local_vol_mc_price_converges_to_black_scholes_for_flat_surface() -> None:
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    iv_surface = _flat_iv_surface(sigma)
    model = LocalVolModel(
        S=S, K=K, T=T, r=r, iv_surface=iv_surface, option_type="call",
        n_steps=100, n_paths=5_000, seed=42,
    )
    bs_price = BlackScholes(S, K, T, r, sigma, "call").price()
    assert math.isclose(model.price(), bs_price, rel_tol=0.08)
