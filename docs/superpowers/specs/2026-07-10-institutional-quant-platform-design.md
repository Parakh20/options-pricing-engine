# Institutional Quantitative Derivatives Pricing & Risk Platform — Design

## Goal

Evolve the existing standalone Options Pricing Engine (BS, MC, LSM, implied vol,
single-source market data) into a modular, tested, benchmarked quant derivatives
platform suitable for demonstrating research-grade (not just implementation-grade)
skill to quant/ML-systems interviewers (Jane Street, Citadel, HRT, Tower, IMC,
Optiver, Two Sigma, DE Shaw class firms).

## Non-goals

- Not a production trading system (no live order routing, no real capital).
- Not attempting every item on the original wishlist superficially — depth over
  breadth. PINNs / Neural SDE pricing are explicitly out of scope unless a later
  phase's users asks for them.

## Architecture

```
pricing_engine/
├── core/            # BS, MC, LSM, implied vol (migrated from top-level core/)
├── models/          # PricingModel ABC → BlackScholes, Heston, SABR, LocalVol,
│                    #   Merton/Kou/VarianceGamma jump models, HullWhite/Vasicek/CIR
├── market/          # data providers (yfinance + pluggable Polygon/IEX/Alpaca),
│                    #   vol surface builder, caching layer
├── calibration/     # optimizers (Levenberg-Marquardt, L-BFGS, differential
│                    #   evolution) + CalibrationEngine
├── simulation/       # MC engine: NumPy/Numba/CuPy backends, QMC (Sobol/Halton/
│                    #   LHS), exotics payoffs (barrier, lookback, basket,
│                    #   digital, Bermudan, cliquet), Brownian bridge
├── risk/             # Greeks (analytic/FD/AD/pathwise), portfolio risk,
│                    #   VaR/CVaR, stress testing, hedging simulator
├── portfolio/        # position/portfolio objects, multi-asset correlated GBM
│                    #   (Cholesky), basket/spread/worst-of options
├── api/              # FastAPI backend for the live dashboard
├── visualization/    # existing plots + vol surface 3D/heatmap/animation
tests/                # pytest, 80%+ coverage target
benchmarks/           # QuantLib/SciPy accuracy + latency + calibration-time
                      #   comparisons, MC convergence, CI regression gates
dashboard/            # React frontend, talks to pricing_engine/api over
                      #   REST + WebSocket
```

Core abstraction: a `PricingModel` ABC (`price`, `calibrate`, `greeks`) that
`BlackScholes`, `Heston`, `SABR`, `LocalVol`, and the jump-diffusion models all
implement, so the calibration engine and comparison reports are model-agnostic.
Config-driven runs (YAML → pricing/calibration parameters) replace hand-edited
scripts.

## Phased roadmap

Each phase is its own implementation cycle (research/reuse → TDD → review →
commit) executed via subagent-driven development, using Sonnet/Haiku only.

1. **Foundation + Vol Models** (this cycle's target)
   - Restructure repo into `pricing_engine/` package; migrate existing
     `core/`, `data/`, `validation/`, `visualisation/` code with tests added
     retroactively.
   - Add pytest suite + coverage config + GitHub Actions CI (pytest, coverage,
     lint, mypy).
   - Implement `PricingModel` ABC.
   - Implement Dupire Local Volatility, Heston (semi-analytic + MC), SABR
     (Hagan approximation).
   - Implement `CalibrationEngine` with Levenberg-Marquardt (primary),
     L-BFGS-B and differential-evolution (fallback/global) optimizers,
     calibrating each model to the cached AAPL option chain.
   - Produce a BS vs Heston vs LocalVol vs SABR comparison report
     (price/IV error vs market) reusing `results/`-style output.

2. **Monte Carlo Engine Upgrade**
   - QMC (Sobol, Halton, Latin Hypercube) with convergence comparison vs
     standard MC.
   - Numba JIT backend; CuPy/PyTorch GPU backend, gated behind availability
     checks (falls back to NumPy if no GPU/Numba present) — CPU-vs-GPU
     benchmark at 100k/1M/10M paths.
   - Pathwise and AD Greeks (PyTorch) compared against finite-difference and
     analytic Greeks for accuracy/speed/memory.
   - Exotics on the existing MC framework: barrier, lookback, basket,
     digital, chooser, Bermudan, cliquet. Brownian bridge for path
     construction. Carr-Madan FFT / COS method for the models that support
     characteristic functions (Heston).

3. **Rates & Jump Models**
   - Vasicek, CIR, Hull-White short-rate models; bond options / simple
     swaption pricing.
   - Merton jump-diffusion, Kou, Variance Gamma; BS-vs-jump-model fit
     comparison on the AAPL chain (esp. short-dated skew, per the current
     README's known BS limitation).
   - Multi-asset: Cholesky-correlated GBM, basket/spread/worst-of options.

4. **Risk & Portfolio Layer**
   - Portfolio-level Greeks aggregation, VaR/CVaR (historical + parametric +
     Monte Carlo), stress testing / scenario analysis.
   - Delta-hedging simulator: simulate a market path, dynamically hedge,
     report PnL and hedging error.
   - Parallel pricing engine: ProcessPoolExecutor and (optionally) Ray/Dask,
     benchmarked against single-threaded.

5. **Market Data + Dashboard**
   - Pluggable market data provider interface (yfinance today; Polygon/IEX/
     Alpaca adapters), async fetch, retry, rate limiting, parquet caching.
   - Vol surface builder (strike × expiry → IV surface, smile/term-structure
     interpolation) with 3D/heatmap visualization.
   - FastAPI backend exposing pricing/calibration/risk; React frontend with
     WebSocket live updates for Greeks/IV/smile/calibration/risk.

6. **Benchmarking + Explainability**
   - Accuracy/latency/calibration-time benchmarks vs QuantLib and SciPy
     references; MC convergence plots; CI regression gate that fails on
     benchmark regression beyond a tolerance.
   - Auto-generated PDF report: market data → calibration → pricing → risk →
     plots, for a given ticker/date.

## Testing strategy

- Unit tests per model/module (closed-form checks, put-call parity, known
  limiting cases e.g. Heston → BS as vol-of-vol → 0).
- Property-based tests (Hypothesis) for invariants (monotonicity in strike,
  no-arbitrage bounds).
- Regression tests pinning current BS/MC/LSM outputs so the restructure in
  phase 1 provably doesn't change behavior.
- Numerical stability tests (extreme strikes/maturities, near-zero vol).
- Target: 80%+ coverage, enforced in CI.

## Risks / open questions

- GPU backend (CuPy/PyTorch-CUDA) can't be validated without GPU hardware in
  this environment — code will be written with a CPU fallback and unit-tested
  on CPU; GPU benchmarks documented as "run on CUDA-capable hardware."
- QuantLib is a heavy dependency; benchmarks phase will add it as a
  dev/benchmark-only extra, not a runtime dependency.
- Scope is intentionally large — phases 2-6 are sequenced but only phase 1 is
  planned in detail right now. Later phases get their own planning pass
  before implementation, per phase, so plans don't rot waiting on earlier
  phases.
