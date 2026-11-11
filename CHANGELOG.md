# Changelog

## 0.2.1 (2026-11-11)

First release on PyPI, as `volsurface-lab` (`pip install volsurface-lab`): the name `volsurface` is
taken on PyPI by an unrelated project. The import name is unchanged (`import volsurface`). No code
changes; every result is identical to 0.2.0.
- Licence metadata (MIT, `LICENSE` already in the repo) and the repository URL.
- `publish` workflow: a version bump merged to main tags, releases and uploads to PyPI.
- README links and images are absolute, so they work on PyPI.

## 0.2.0 (2026-11-01)

First tagged release. It also contains milestones M2–M6, merged after 0.1.0 without a release:
- smooth θ(T)
- the real-data study
- eSSVI
- the local-vol Monte Carlo
- arbitrage-free SVI

### Added
- `VolSurface.local_vol(..., floor=, cap=, extrapolate="flat")`: safe evaluation for simulation.
  - Local vol is held flat in k beyond the fitted range.
  - NaN falls back to implied vol, and the result is clipped to [floor, cap].
  - Changed evaluations are counted in `n_clamped` / `n_local_vol`.
  - Without the new keywords, the output is unchanged.
- `VolSurface.k_range`, which `fit_surface` sets from the quotes' log-moneyness range, and which
  `SurfaceFit.local_vol_surface()` carries over.
- `ForwardCurve.to_dict` / `from_dict` and `SSVISurface.to_dict` / `from_dict` (SSVI and eSSVI): JSON-safe
  and exact.
- `VolSurface.shifted(parallel, skew)` and `ShiftedSurface`: scenario surfaces that raise `ValueError` on
  static arbitrage.
- `examples/real_data_study.py` prints RMSE by standard-deviation band, raw quote counts and calendar-violation
  counts, so every number in the README's real-data section now comes from a script.

### Removed
- `examples/autocallable_skew.py` and the `pricer` extra. The study moved to quant-stack (S1), which
  reproduces its table, so volsurface no longer depends on sppricer anywhere.

### Fixed
- README: arbitrage-free SVI band RMSE is "0.16–0.42", not "0.15–0.42". The script prints 0.16.

### Tests
- Regression pins for the README numbers (`tests/test_regression.py`). The real-data pins are marked `slow`
  and skip without the local snapshots. The default run excludes `slow`; run those with `pytest -m slow`.
