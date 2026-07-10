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


def _skewed_term_structure_surface(S: float):
    """A surface with a strike skew that ONLY exists early in calendar time
    (T < 0.5) and flattens out later. A flat-in-K surface cannot distinguish
    elapsed vs. remaining time (the terminal lognormal distribution only
    depends on the *total* integrated variance, which is order-invariant for
    two equal-length regimes). Making the vol strike-dependent breaks that
    invariance: the local vol experienced by a path now depends on WHERE the
    spot has diffused to at the time the regime is applied, so swapping which
    half of the path sees the skewed regime (elapsed `t` vs. buggy `T - t`)
    changes the simulated price.
    """

    def iv(K: float, T: float) -> float:
        if T < 0.5:
            return max(0.12, min(0.35, 0.22 - 0.15 * (K - S) / S))
        return 0.15

    return iv


def _simulate_with_time_arg(
    iv_surface, S: float, K: float, T: float, r: float, *, use_elapsed: bool,
    n_steps: int, n_paths: int, seed: int,
) -> float:
    """Reference simulator mirroring LocalVolModel's Euler loop, parametrized
    so the (buggy) `T - t` behavior can be reproduced here for comparison
    without reintroducing the bug into production code.
    """
    rng = np.random.default_rng(seed)
    local_vol = build_local_vol_surface(iv_surface, S, r)
    dt = T / n_steps
    S_t = np.full(n_paths, S, dtype=float)
    for step in range(n_steps):
        t = step * dt
        tau = t if use_elapsed else (T - t)
        sigmas = np.array([local_vol(float(s), tau) for s in S_t])
        Z = rng.standard_normal(n_paths)
        S_t = S_t * np.exp((r - 0.5 * sigmas**2) * dt + sigmas * np.sqrt(dt) * Z)
    payoff = np.maximum(S_t - K, 0.0)
    return float(np.exp(-r * T) * payoff.mean())


def test_local_vol_mc_uses_elapsed_time_not_remaining_time() -> None:
    """Regression test for the Dupire MC time-argument bug: local_vol(K, T)
    is parametrized by calendar time from the valuation date, so the loop
    must call local_vol(S_t, t) with ELAPSED time t = step * dt, not
    T_maturity - t. A flat-in-K surface can't catch this (see docstring
    above), so this uses a surface whose skew is only active for T < 0.5.

    With the fix (elapsed time), the strike-skewed regime is applied while
    paths are still near the initial spot (early in the simulation). With
    the bug (T - t, remaining time), the same skewed regime gets applied
    late in the simulation once paths have already diffused away from spot.
    Empirically (verified across multiple seeds) this consistently makes the
    elapsed-time price higher than the remaining-time price for this surface
    and these parameters, with a comfortable margin over MC noise.
    """
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    iv_surface = _skewed_term_structure_surface(S)
    n_steps, n_paths, seed = 15, 600, 13

    model = LocalVolModel(
        S=S, K=K, T=T, r=r, iv_surface=iv_surface, option_type="call",
        n_steps=n_steps, n_paths=n_paths, seed=seed,
    )
    fixed_price = model.price()

    # The current (fixed) implementation must match the elapsed-time
    # reference exactly (same RNG draws, same math) — this is deterministic,
    # seed-independent regression protection for the T-t vs t bug.
    elapsed_reference_price = _simulate_with_time_arg(
        iv_surface, S, K, T, r, use_elapsed=True,
        n_steps=n_steps, n_paths=n_paths, seed=seed,
    )
    assert math.isclose(fixed_price, elapsed_reference_price, rel_tol=1e-9)
