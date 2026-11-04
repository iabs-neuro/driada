"""Summarise the null runs: false-positive rate per null, delay mode and feature type.

Reads results/*.csv written by calib_null.py (and results_diag/*.csv of
diag_delay.py when present) and writes

  calibration_table.csv   rates with 95 % intervals and the expected number of
                          false pairs in the paper cohort
  calibration_table.md    the same as a markdown table
  diag_table.csv          rates of diag_delay.py under the three null rules

Two intervals are given: Clopper-Pearson (pairs independent) and a bootstrap
over recording units (sessions or synthetic experiments), which accounts for
pairs of one session sharing the same behaviour.

Usage: python analyze.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

HERE = P.CALIBRATION_DIR   # reads results/, writes the tables next to it
ALPHA = 0.001
MI_EXCESS_THR = 0.04
N_BOOT = 2000
SEED = 42

# Paper cohort: 30 105 neuron-sessions, by_type features of NOF.
COHORT_NEURONS = 30105
COHORT_FEATURES = {"place": 1, "speed": 1, "direction": 2, "discrete": 13}
COHORT_FEATURES["all"] = sum(COHORT_FEATURES.values())
TYPES = ["place", "speed", "direction", "discrete", "all"]


def cp(k, n, level=0.95):
    a = (1 - level) / 2
    lo = beta.ppf(a, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a, k + 1, n - k) if k < n else 1.0
    return lo, hi


def boot(df, col, rng):
    per = df.groupby("unit")[col].agg(["sum", "count"])
    k, n = per["sum"].to_numpy(), per["count"].to_numpy()
    idx = rng.integers(0, len(per), size=(N_BOOT, len(per)))
    r = k[idx].sum(1) / n[idx].sum(1)
    return np.percentile(r, [2.5, 97.5])


def load():
    files = sorted((HERE / "results").glob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["sig04"] = df["sig"] & (df["me_excess"] > MI_EXCESS_THR)
    return df


def summarise(df):
    rng = np.random.default_rng(SEED)
    rows = []
    for (null, delays), g in df.groupby(["null", "delays"]):
        for t in TYPES:
            s = g if t == "all" else g[g.ftype == t]
            n = len(s)
            row = dict(null=null, delays=delays, ftype=t, n_units=s.unit.nunique(),
                       n_pairs=n, stage1_rate=s.stage1.mean())
            for col, tag in (("sig", ""), ("sig04", "_mi04")):
                k = int(s[col].sum())
                lo, hi = cp(k, n)
                blo, bhi = boot(s, col, rng)
                rate = k / n
                m = COHORT_NEURONS * COHORT_FEATURES[t]
                row.update({
                    f"k{tag}": k, f"rate{tag}": rate, f"cp_lo{tag}": lo, f"cp_hi{tag}": hi,
                    f"boot_lo{tag}": blo, f"boot_hi{tag}": bhi,
                    f"ratio_to_nominal{tag}": rate / ALPHA,
                    f"cohort_expected{tag}": rate * m,
                    f"cohort_lo{tag}": blo * m, f"cohort_hi{tag}": bhi * m,
                })
            row["cohort_pairs"] = COHORT_NEURONS * COHORT_FEATURES[t]
            rows.append(row)
    return pd.DataFrame(rows)


def fmt_rate(r, lo, hi):
    return f"{r * 1000:.2f} [{lo * 1000:.2f}-{hi * 1000:.2f}]"


def to_md(tab):
    lines = ["| null | delay | type | pairs | false | rate x10^-3 [95% bootstrap] | x nominal | expected in the cohort | false, 0.04 bits | rate, 0.04 bits | expected, 0.04 bits |",
             "|---|---|---|---:|---:|---|---:|---|---:|---|---|"]
    for _, r in tab.iterrows():
        lines.append(
            f"| {r.null} | {'yes' if r.delays else 'no'} | {r.ftype} | {r.n_pairs} | {r.k} | "
            f"{fmt_rate(r.rate, r.boot_lo, r.boot_hi)} | {r.ratio_to_nominal:.1f} | "
            f"{r.cohort_expected:.0f} [{r.cohort_lo:.0f}-{r.cohort_hi:.0f}] | {r.k_mi04} | "
            f"{fmt_rate(r.rate_mi04, r.boot_lo_mi04, r.boot_hi_mi04)} | "
            f"{r.cohort_expected_mi04:.0f} [{r.cohort_lo_mi04:.0f}-{r.cohort_hi_mi04:.0f}] |")
    return "\n".join(lines)


def diag():
    files = sorted((HERE / "results_diag").glob("*.csv"))
    if not files:
        return None
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    rows = []
    for t in TYPES:
        s = d if t == "all" else d[d.ftype == t]
        for c in ("p_nodelay", "p_delay", "p_maxnull"):
            k = int((s[c] < ALPHA).sum())
            lo, hi = cp(k, len(s))
            rows.append(dict(ftype=t, rule=c, n_units=s.unit.nunique(), n_pairs=len(s),
                             k=k, rate=k / len(s), cp_lo=lo, cp_hi=hi))
    return pd.DataFrame(rows)


def main():
    df = load()
    tab = summarise(df)
    tab.to_csv(HERE / "calibration_table.csv", index=False)
    (HERE / "calibration_table.md").write_text(to_md(tab), encoding="utf-8")
    print(to_md(tab))
    fp = df[df.sig & df.delays]
    fp[["null", "unit", "neuron", "feature", "ftype", "me", "me_excess", "pval",
        "delay_frames"]].to_csv(HERE / "false_pairs_delay.csv", index=False)
    print("\nselected delays of false pairs (frames at 20 fps):")
    print(fp.delay_frames.value_counts().sort_index().to_string())
    dg = diag()
    if dg is not None:
        dg.to_csv(HERE / "diag_table.csv", index=False)
        print("\n", dg.to_string())


if __name__ == "__main__":
    main()
