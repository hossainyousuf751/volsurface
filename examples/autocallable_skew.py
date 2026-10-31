"""M5: how much does skew lower the fair value of a worst-of autocallable?

Fits eSSVI surfaces to the latest saved SPX and NDX snapshots (python examples/snapshot.py),
simulates both indices under Dupire local vol, and prices sppricer's worst-of Phoenix on
those paths. The flat-vol price uses each index's ATM implied vol at maturity on the same
forward curves and the same random numbers, so the skew effect has a small standard error.
Requires the pricer: pip install -e ".[pricer]"
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sppricer import Market, PhoenixAutocallable, Underlying

from volsurface import fit_surface
from volsurface.montecarlo import simulate_local_vol

N_PATHS, SEED, MATURITY, CORRELATION = 200_000, 42, 3.0, 0.9  # SPX/NDX: typical daily-return correlation

data_dir = Path(__file__).resolve().parent.parent / "data"
surfaces, spots, flat_vols = [], [], []
for ticker in ("SPX", "NDX"):
    path = sorted(data_dir.glob(f"{ticker}_20*.csv"))[-1]
    quotes = pd.read_csv(path)
    spot = float(quotes["spot"].iloc[0])
    surf = fit_surface(quotes, spot, max_rel_spread=0.3).essvi_surface
    surfaces.append(surf)
    spots.append(spot)
    flat_vols.append(float(surf.implied_vol(surf.curve.forward(MATURITY), MATURITY)))
    print(f"{ticker} {path.stem}: spot {spot:.2f}, ATM vol at {MATURITY:.0f}y {flat_vols[-1]:.2%}")

# sppricer discounts at a flat rate; take it from the SPX parity discount factor at maturity.
rate = -np.log(float(surfaces[0].curve.discount(MATURITY))) / MATURITY
market = Market(tuple(Underlying(n, s, v) for n, s, v in zip(("SPX", "NDX"), spots, flat_vols)), rate,
                np.array([[1.0, CORRELATION], [CORRELATION, 1.0]]))
corr = market.correlation


def note(protection=0.60, coupon_barrier=0.70):
    """The sppricer README note: 3y quarterly, 8% p.a. memory coupon, European protection."""
    return PhoenixAutocallable(initial_levels=tuple(spots),
                               observation_times=tuple(0.25 * k for k in range(1, 13)),
                               coupon_rate=0.02, coupon_barrier=coupon_barrier, autocall_barrier=1.00,
                               protection_barrier=protection, memory=True, first_autocall_index=3)


times = np.asarray(note().observation_times)
paths = {"local vol": simulate_local_vol(surfaces, times, N_PATHS, corr, seed=SEED),
         "flat ATM vol": simulate_local_vol(surfaces, times, N_PATHS, corr, seed=SEED, flat_vols=flat_vols)}
print(f"\nrate {rate:.3%} (SPX parity), correlation {CORRELATION}, {N_PATHS:,} paths, daily steps\n")


def paired(pv):
    """Average antithetic pairs, as sppricer.price_mc does, so standard errors are honest."""
    return 0.5 * (pv[: len(pv) // 2] + pv[len(pv) // 2:])


rows = []
for protection, coupon_barrier in ((0.60, 0.70), (0.70, 0.80), (0.50, 0.60)):
    product = note(protection, coupon_barrier)
    pv = {name: paired(product.discounted_payoff(p, market)) for name, p in paths.items()}
    diff = pv["local vol"] - pv["flat ATM vol"]
    worst = {name: (p[:, -1] / np.asarray(spots)).min(axis=1) for name, p in paths.items()}
    rows.append({"protection / coupon barrier": f"{protection:.0%} / {coupon_barrier:.0%}",
                 "flat ATM vol": pv["flat ATM vol"].mean(), "local vol": pv["local vol"].mean(),
                 "skew effect": diff.mean(), "± (95%)": 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)),
                 "P(worst < barrier), flat": (worst["flat ATM vol"] < protection).mean(),
                 "P(worst < barrier), LV": (worst["local vol"] < protection).mean()})
with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
    print(pd.DataFrame(rows).to_string(index=False))
