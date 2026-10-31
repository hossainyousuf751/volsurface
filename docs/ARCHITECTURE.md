# Architecture

How volsurface is put together, the conventions every stage shares, and the invariants a
change must preserve. For the development workflow, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Data flow

```
raw quotes (T, strike, option_type, bid, ask)
   │  chain.clean_quotes            drop zero bids, crossed/wide markets, T < 7 days
   ▼
   │  chain.implied_forward_and_discount   per expiry: regress C − P on K ⇒ F, DF
   │  chain.build_iv_table          OTM only; iv_bid / iv_mid / iv_ask via black.implied_vols
   ▼
IV table (T, strike, k, w, iv_*, forward, discount, vega)
   │  svi.fit_svi                   one smile per expiry, vega-weighted
   │  θ(T) = SVI w(0), forced non-decreasing
   │  ssvi.fit_ssvi                 global (ρ, η, γ) across all quotes
   ▼
surface.ForwardCurve + SVISliceSurface + SSVISurface
   │  arbitrage.check_surface       butterfly (g ≥ 0) and calendar (∂T w ≥ 0)
   │  VolSurface.local_vol          Dupire in total variance, finite differences
   ▼
pipeline.SurfaceFit                 everything above in one object
pipeline.fit_errors                 RMSE and in-spread % per expiry
```

`pipeline.fit_surface(quotes, spot)` is the single entry point. Everything else is a
building block that can be used and tested on its own.

## Modules and dependencies

Dependencies point one way only. A module may import from the modules above it in this
list, never from those below:

| Module | Responsibility | Imports from |
|---|---|---|
| `black` | Black-76 price, vega, and implied vol on (F, DF) | (none) |
| `svi` | `SVIParams`, analytic derivatives, `fit_svi` | (none) |
| `ssvi` | `SSVIParams`, `fit_ssvi` | (none) |
| `chain` | Cleaning, parity forwards, IV table | `black` |
| `synthetic` | Option chains from a known SSVI truth | `black`, `ssvi` |
| `surface` | `ForwardCurve`, `VolSurface` ABC and implementations, local vol | `svi`, `ssvi` |
| `arbitrage` | `svi_g`, `check_surface`, `ArbitrageReport` | `surface`, `svi` |
| `pipeline` | `fit_surface`, `SurfaceFit`, `fit_errors` | all of the above |
| `plotting` | Diagnostic figures (lazy `matplotlib`) | `pipeline` |
| `data` | Yahoo Finance loader (lazy `yfinance`) | (none) |

Keep model modules (`svi`, `ssvi`) free of pandas and of each other. They work on arrays only.

## Conventions shared by every stage

| Quantity | Symbol | Unit / definition |
|---|---|---|
| Maturity | `T` | Years (ACT/365.25 in `data.py`) |
| Log-moneyness | `k` | ln(K / F(T)), always against the **forward**, never spot |
| Total variance | `w` | σ²·T. All fitting and interpolation is done in `w`, not σ |
| ATM total variance | `theta` | w(0, T) |
| Discount factor | `DF`, `df` | Implied from parity, not from a rate curve |
| Vol | `iv_*`, `vol` | Decimals (0.20). Only plots and `fit_errors` use vol points (×100) |

Quotes DataFrame columns: `T, strike, option_type ('C'/'P'), bid, ask` (required) plus
optional extras such as `volume` and `open_interest`.

## A new surface type

Subclass `VolSurface` and implement one method:

```python
def total_variance(self, k, T) -> np.ndarray: ...
```

`implied_vol`, `local_vol`, `check_surface`, `fit_errors` and the plots then work
unchanged. `total_variance` must broadcast `k` against `T` and accept scalars and arrays.

## Invariants

A change that breaks one of these is a bug, even if the tests still pass.

1. **Forwards come from the market.** No stage assumes a rate or dividend yield. F and DF
   per expiry come from put-call parity only.
2. **OTM quotes only** enter the IV table: calls with K ≥ F, puts with K < F.
3. **θ(T) is non-decreasing and C¹.** The node values are made monotone in `fit_surface` and
   again in `SSVISurface.__init__`. Between nodes, the monotone PCHIP interpolation keeps θ(T)
   non-decreasing, which rules out calendar arbitrage in SSVI, and continuously
   differentiable, which removes local-vol jumps at the expiries. Any replacement
   interpolator must keep both properties.
4. **SSVI candidates are always admissible.** η = s·2/(1+|ρ|), s ∈ (0,1], γ ∈ (0, ½].
   Don't replace this reparameterisation with a penalty.
5. **SVI slices are only penalised, not guaranteed.** Their arbitrage status must be
   *reported* (`fit.svi_arbitrage`), never assumed.
6. **Local vol returns NaN, never a made-up number,** where ∂T w ≤ 0 or the Dupire
   denominator is ≤ 0.
7. **The synthetic generator stays the ground truth.** Every new stage needs a way to be
   checked against `synthetic_chain`'s `truth` dict.

## Extension points (roadmap mapping)

| Roadmap item | Where it lands |
|---|---|
| M2 smooth θ(T) | **Done:** PCHIP in `SSVISurface.theta` |
| M3 real-data study | A new `examples/` script over `data/*.csv`. No library change needed |
| M4 eSSVI ρ(θ) | New params class in `ssvi.py` (or `essvi.py`) plus a new `VolSurface` subclass |
| M5 local-vol MC | Consumes `VolSurface.local_vol`. Lives in the pricer repo, not here |
