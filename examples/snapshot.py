"""Save today's delayed CBOE index chains to data/ for the real-data study.

Works any time: CBOE keeps the last session's closing quotes. Run once per
trading day (e.g. from Task Scheduler or cron after 16:15 New York time).
Usage: python examples/snapshot.py [^SPX ^NDX ...]
"""

import sys
from pathlib import Path

from volsurface.data import load_cboe_chain

data_dir = Path(__file__).resolve().parent.parent / "data"
data_dir.mkdir(exist_ok=True)

for ticker in sys.argv[1:] or ["^SPX", "^NDX"]:
    quotes, spot, now = load_cboe_chain(ticker)
    path = data_dir / f"{ticker.strip('^')}_{now:%Y%m%d}.csv"
    quotes.assign(spot=spot, quote_time=now.isoformat()).to_csv(path, index=False)
    print(f"{ticker}: {len(quotes)} quotes, spot {spot:.2f}, as of {now} -> {path.name}")
