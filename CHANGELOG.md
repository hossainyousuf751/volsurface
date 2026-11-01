# Changelog

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
- `VolSurface.k_range`, which `fit_surface` sets from the quotes' log-moneyness range.
- `ForwardCurve.to_dict` / `from_dict` and `SSVISurface.to_dict` / `from_dict` (SSVI and eSSVI): JSON-safe
  and exact.
- `VolSurface.shifted(parallel, skew)` and `ShiftedSurface`: scenario surfaces that raise `ValueError` on
  static arbitrage.
- `examples/real_data_study.py` prints RMSE by standard-deviation band, raw quote counts and calendar-violation
  counts, so every number in the README's real-data section now comes from a script.

### Fixed
- README: arbitrage-free SVI band RMSE is "0.16–0.42", not "0.15–0.42". The script prints 0.16.

### Tests
- Regression pins for the README numbers (`tests/test_regression.py`). The real-data pins are marked `slow`
  and skip without the local snapshots. The default run excludes `slow`; run those with `pytest -m slow`.
