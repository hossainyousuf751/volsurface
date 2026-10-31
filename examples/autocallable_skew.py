"""M5/M6: how much does skew lower the fair value of a worst-of autocallable?

Fits the latest saved SPX and NDX snapshots (python examples/snapshot.py), simulates both
indices under Dupire local vol and prices sppricer's worst-of Phoenix on those paths. Three
legs share the same forward curves and random numbers, so their differences have small
standard errors:

    flat ATM vol      each index's ATM implied vol at maturity (what a flat-vol pricer uses)
    eSSVI LV          local vol from eSSVI, which under-states the SPX deep put wing
    arb-free SVI LV   local vol from arbitrage-free SVI on pillar expiries (M6)

Requires the pricer: pip install -e ".[pricer]"
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sppricer import Market, PhoenixAutocallable, Underlying

from volsurface import fit_surface
from volsurface.montecarlo import simulate_local_vol

N_PATHS, SEED, MATURITY, CORRELATION = 200_000, 42, 3.0, 0.9  # SPX/NDX: typical daily-return correlation
# NDX 2y and 3y quotes are wide (median spread ~1/3 of mid). At 0.3 they are dropped, NDX stops at 1.13y
# and holding its smile fixed beyond that overstates the skew effect by ~0.8 on the 60% note.
MAX_REL_SPREAD = {"SPX": 0.3, "NDX": 0.5}

data_dir = Path(__file__).resolve().parent.parent / "data"
fits, spots = [], []
for ticker in ("SPX", "NDX"):
    path = sorted(data_dir.glob(f"{ticker}_20*.csv"))[-1]
    quotes = pd.read_csv(path)
    spots.append(float(quotes["spot"].iloc[0]))
    fits.append(fit_surface(quotes, spots[-1], max_rel_spread=MAX_REL_SPREAD[ticker]))
    print(f"{ticker} {path.stem}: spot {spots[-1]:.2f}, last expiry {fits[-1].expiries[-1]:.2f}y")

surfaces = {"eSSVI LV": [f.essvi_surface for f in fits],
            "arb-free SVI LV": [f.local_vol_surface() for f in fits]}
af = surfaces["arb-free SVI LV"]
flat_vols = [float(s.implied_vol(s.curve.forward(MATURITY), MATURITY)) for s in af]
print("ATM vol at maturity:", ", ".join(f"{v:.2%}" for v in flat_vols))

# sppricer discounts at a flat rate; take it from the SPX parity discount factor at maturity.
rate = -np.log(float(fits[0].curve.discount(MATURITY))) / MATURITY
market = Market(tuple(Underlying(n, s, v) for n, s, v in zip(("SPX", "NDX"), spots, flat_vols)), rate,
                np.array([[1.0, CORRELATION], [CORRELATION, 1.0]]))


def note(protection=0.60, coupon_barrier=0.70):
    """The sppricer README note: 3y quarterly, 8% p.a. memory coupon, European protection."""
    return PhoenixAutocallable(initial_levels=tuple(spots),
                               observation_times=tuple(0.25 * k for k in range(1, 13)),
                               coupon_rate=0.02, coupon_barrier=coupon_barrier, autocall_barrier=1.00,
                               protection_barrier=protection, memory=True, first_autocall_index=3)


times = np.asarray(note().observation_times)
kw = dict(times=times, n_paths=N_PATHS, correlation=market.correlation, seed=SEED)
paths = {"flat ATM vol": simulate_local_vol(af, flat_vols=flat_vols, **kw)}
paths.update({name: simulate_local_vol(s, **kw) for name, s in surfaces.items()})
print(f"rate {rate:.3%} (SPX parity), correlation {CORRELATION}, {N_PATHS:,} paths, daily steps\n")


def paired(pv):
    """Average antithetic pairs, as sppricer.price_mc does, so standard errors are honest."""
    return 0.5 * (pv[: len(pv) // 2] + pv[len(pv) // 2:])


def diff(a, b):
    d = a - b
    return f"{d.mean():+.2f} ± {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.2f}"


rows = []
for protection, coupon_barrier in ((0.60, 0.70), (0.70, 0.80), (0.50, 0.60)):
    product = note(protection, coupon_barrier)
    pv = {name: paired(product.discounted_payoff(p, market)) for name, p in paths.items()}
    worst = {name: (p[:, -1] / np.asarray(spots)).min(axis=1) for name, p in paths.items()}
    knock_in = {name: (w < protection).mean() for name, w in worst.items()}
    rows.append({"barrier / coupon": f"{protection:.0%} / {coupon_barrier:.0%}",
                 **{name: v.mean() for name, v in pv.items()},
                 "skew effect": diff(pv["arb-free SVI LV"], pv["flat ATM vol"]),
                 "wing fix (vs eSSVI)": diff(pv["arb-free SVI LV"], pv["eSSVI LV"]),
                 "P(worst<barrier) flat/LV":
                     f"{knock_in['flat ATM vol']:.1%} / {knock_in['arb-free SVI LV']:.1%}"})
with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 200):
    print(pd.DataFrame(rows).to_string(index=False))
