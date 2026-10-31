# Contributing to volsurface

How to set up, change, test, and ship this project. Read
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first for how the pipeline fits together
and which mathematical invariants must never break.

## 1. Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on Linux/macOS)
pip install -e ".[dev]"           # numpy, scipy, pandas, pytest, matplotlib
pip install -e ".[data]"          # only if you need live Yahoo Finance chains
pytest                            # must be green before you start
```

Supported Python: 3.10, 3.11, 3.12 (the CI matrix). Do not use syntax newer than 3.10.

## 2. The development loop

Every change follows the same order:

1. **State the change in one sentence.** If you can't, split it.
2. **Write or update a test first** that fails without the change (see §4 for what kind).
3. **Make the smallest change that passes it.**
4. **Lint and run the full suite:** `ruff check . && pytest -q`.
5. **Run the synthetic demo:** `python examples/synthetic_demo.py`. Check that the fitted
   SSVI parameters still match the truth (ρ ≈ −0.70, η ≈ 1.00, γ ≈ 0.40), that both
   arbitrage reports say `True`, and that the plots in `docs/` still look right.
6. **Update the README** if you changed a public function, a method choice, a validation
   number, or completed a roadmap item.
7. **Commit**, one logical change per commit (see §6).

If a change moves the numbers in the README's *Validation* table, update the table in the
same commit and give the reason in the commit message.

## 3. Code conventions

These are the existing patterns. Match them; don't introduce new ones.

### Module layout

- One module per pipeline stage (`black`, `chain`, `svi`, `ssvi`, `surface`, `arbitrage`,
  `pipeline`). New stages get a new module; don't add a second job to an existing one.
- Every module starts with a docstring that says what it does and, for model modules, gives
  the formula in plain text (see `svi.py`, `ssvi.py`).
- Re-export public names from `src/volsurface/__init__.py` and keep `__all__` sorted.
- Optional dependencies (`matplotlib`, `yfinance`) are imported **inside the function** that
  uses them, so the core package installs with numpy/scipy/pandas only.
- Start each module with `from __future__ import annotations`.

### Numerics

- Functions take array-likes and coerce once at the top: `k = np.asarray(k, dtype=float)`.
  Broadcast with `np.broadcast_arrays` when two inputs must share a shape.
- Write vectorised numpy. Python loops are fine across expiries or optimiser starts, not
  across strikes. (`implied_vols` is the one known exception; see the improvement list.)
- Return `NaN` for "no valid answer at this point" (an implied vol outside the bounds, local
  vol under arbitrage). Raise `ValueError` when the *whole call* cannot proceed (missing
  columns, too few quotes, an implausible discount factor). Don't return `None`.
- Wrap expected divide-by-zero in `np.errstate(...)` scoped to the lines that need it, not
  a global setting.
- Every model parameter set is a `@dataclass(frozen=True)` with methods for its own
  evaluation (`total_variance`, derivatives) and a validity property (`is_valid`,
  `is_arbitrage_free`).
- Fitters use `scipy.optimize.least_squares` with bounds, a small multi-start grid, and keep
  the lowest cost. Weights are normalised by their mean and passed as `sqrt(weights)`.

### Naming: follow the maths

- `k` log-moneyness ln(K/F), `w` total implied variance σ²T, `theta` ATM total variance,
  `T` maturity in years, `F` forward, `DF`/`df` discount factor, `K` strike.
- Upper-case single letters are fine for financial quantities (`F`, `K`, `T`); everything
  else is `snake_case`.
- DataFrame columns are `snake_case` and match the names in `chain.py`'s docstring.

### Style

- Line length 110, enforced by `ruff check .` (rules E, W, F; configured in `pyproject.toml`).
- Comments explain *why* (a modelling choice, a paper reference), never *what*.
- Cite the source paper or remark in the docstring when you implement a formula from one.

## 4. Testing

Tests live in `tests/`, one file per area. Use plain `pytest` functions and
`pytest.approx`/`np.testing.assert_allclose`. No fixtures framework, no mocks of numpy.

The project's core testing idea: **generate data from a known truth, then recover it.**

| Kind of change | Required test |
| --- | --- |
| New pricing / inversion code | Round trip: price → invert → original input, to ~1e-8 |
| New fitter or model | Parameter recovery on noiseless data from `synthetic_chain` or the model itself |
| New arbitrage check | A known-bad case is flagged *and* a known-good case passes (cf. Vogt example) |
| New surface / interpolation | An analytic limit (e.g. flat vol ⇒ local vol = implied vol) |
| Pipeline change | `test_end_to_end_on_noisy_synthetic_chain` still passes; add assertions for new outputs |
| Bug fix | A test that reproduces the bug, written before the fix |

Rules:

- Tests must be deterministic: always pass `seed=` to `synthetic_chain`.
- Tests must run offline. Nothing in `tests/` may call Yahoo Finance. To test against real
  data, commit a small CSV snapshot and load it.
- Tolerances must be justified by the setup: tight (1e-6 to 1e-8) for noiseless round
  trips, loose (vol points) for noisy fits. Never loosen a tolerance just to get green.
  Find out why it moved.
- Keep the suite fast (currently about 6 s). Mark anything slower than a few seconds with
  `@pytest.mark.slow`.

## 5. Changing the maths

Model changes (new parameterisation, new interpolation in T, new constraint) carry the
most risk. Before merging:

1. Write down the formula and its source in the module docstring.
2. State which no-arbitrage condition it preserves or breaks, and how that is enforced:
   by construction (reparameterisation, as in `fit_ssvi`), by penalty (as in `fit_svi`),
   or only diagnosed (`check_surface`).
3. Show `check_surface(...).is_free` on the synthetic chain, before and after.
4. Compare `fit_errors` before and after on the noisy synthetic chain and, ideally, on a
   saved real snapshot in `data/`.
5. Regenerate `docs/*.png` with `python examples/synthetic_demo.py` and commit them with
   the change.

## 6. Git and CI

- Branch from `main`: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`.
- Commit messages: imperative subject ≤ 72 chars (`Add monotone spline for theta(T)`),
  body explaining *why* and any change in validation numbers.
- CI (`.github/workflows/ci.yml`) runs `ruff check`, `pytest`, and the synthetic demo on
  3.10 to 3.12 for every push and PR. A PR merges only when CI is green.
- Never commit `data/` (git-ignored: live snapshots may be large or licence-restricted),
  virtualenvs, or `__pycache__`. Do commit the regenerated `docs/*.png` used by the README.

## 7. Working with live data

- For real data at any hour, `examples/snapshot.py` saves CBOE's closing ^SPX / ^NDX chains to
  `data/`. Run it once per trading day; `examples/real_data_study.py` fits them all.
- Use `examples/live_fit.py <ticker>` (Yahoo) during US market hours. Outside them bids are often zero.
- Prefer European index options (`^SPX`). For American options (ETFs, single stocks), expect
  a small early-exercise bias.
- Every live run writes its raw chain to `data/<TICKER>_<YYYYMMDD>.csv`. Keep the snapshots
  you draw conclusions from so the result can be reproduced.
- Treat a live fit as suspect if `fit_errors` shows `ssvi_in_spread_pct` well below 50% or
  either arbitrage report is `False`. Look at `plot_smiles` before trusting the numbers.

## 8. Definition of done

- [ ] `ruff check .` and `pytest -q` green locally, and CI green on all three Python versions
- [ ] New behaviour has a test of the right kind (§4)
- [ ] `python examples/synthetic_demo.py` recovers the true parameters; arbitrage reports `True`
- [ ] Public API changes are re-exported in `__init__.py` and reflected in the README
- [ ] README *Validation* and *Roadmap* are updated if affected
- [ ] No new runtime dependency unless agreed; optional ones go in an extra
