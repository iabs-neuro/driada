"""Exact interval ends of every variant of the calibration (s2/4_calibration.py, s2/11_s3_with_s7.py),
from the bootstrap draws (the summary tables round to 0.01, which leaves ties such as 3.75 that do not
show whether the end is 3.7 or 3.8 at one decimal); the 3.7-5.4% of Supplementary Section S2 is the
variant s2+s7|neuron.
Output: <[s2] out_dir>/data/s3_intervals_exact.csv (expected false multi-selective per 100: 2.5 and
97.5 percentiles; observed minus expected: the same).
usage: python s2/15_exact_intervals.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

DATA = P.get("s2", "out_dir") / "data"
rows = []
for f in ("s3_bootstrap.csv", "s3_bootstrap_s7.csv"):
    b = pd.read_csv(DATA / f)
    for v, g in b.groupby("variant", sort=False):
        lo, hi = np.percentile(g.expected_per100, [2.5, 97.5])
        dlo, dhi = np.percentile(g.observed_per100 - g.expected_per100, [2.5, 97.5])
        rows.append({"file": f, "variant": v, "replicates": len(g), "expected_lo": round(lo, 4),
                     "expected_hi": round(hi, 4), "width": round(hi - lo, 4),
                     "obs_minus_exp_lo": round(dlo, 4), "obs_minus_exp_hi": round(dhi, 4)})
out = pd.DataFrame(rows)
out.to_csv(DATA / "s3_intervals_exact.csv", index=False)
print(out.to_string(index=False))
