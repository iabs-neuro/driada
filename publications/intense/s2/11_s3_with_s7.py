"""The calibration of s2/4_calibration.py with the synthetic neurons at CA1 strength (s7/), and with the
synthetic neurons resampled by session: the expected 4.5% and its 95% interval (Results; Supplementary
Section S2) against the observed 11.2%.

The strength calibration of s2/4_calibration.py (main variant: CA1 counting rule, strength of the
true label) refit on two training sets:
  s2      synthetic neurons of the S2 categories only (as s2/4_calibration.py)
  s2+s7   the same plus the single-variable cells of s7/ (place_in, place_out, bodydirection at CA1
          strength); the model of the paper
and two resampling schemes for the interval (CA1 always by mouse-session, 64 groups):
  neuron  synthetic neurons within category (as s2/4_calibration.py); the interval of the paper
  session synthetic sessions (16, with all their neurons), so that the interval also carries how much
          the synthetic error changes from one recorded behaviour to another
1000 replicates each. Also the model-free check on s7/ alone: CA1 place and body-direction
neuron-sessions inside the strength range of s7/, expected = sum over strength bins of CA1 count x
rate of s7/.
Outputs (<[s2] out_dir>/data/): s3_bootstrap_s7.csv, s3_summary_s7.csv, s3_percat_s2s7.csv (s2+s7
point model per category: n, observed, expected; read by s2/5_composition.py).
s2/16_significance.py and the script of Supplementary Fig. 8 import this file for s7_cells.
usage: python s2/11_s3_with_s7.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402

HERE = P.get("s2", "out_dir")
DATA = HERE / "data"
_spec = importlib.util.spec_from_file_location("cal", Path(__file__).resolve().parent / "4_calibration.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)
RNG = np.random.default_rng(1102)
B = 1000


def s7_cells():
    s7 = pd.read_csv(DATA / "s7_neurons.csv")
    s7 = s7[(s7.category != "place+bodydirection") & s7.found_1].copy()
    s7["cat"] = s7.category.replace({"place_in": "place", "place_out": "place"})
    s7["x"], s7["y"] = np.log10(s7.excess_1), s7.multi
    s7["session"] = "s7_" + s7.session      # sessions of s7/ are separate draws of the same behaviour
    return s7[["session", "neuron", "category", "cat", "snr", "x", "y"]]


def main():
    ca1 = pd.read_csv(DATA / "ca1_neuron_sessions.csv")
    ca1["x"] = np.log10(ca1.max_excess)
    ca1["grp"] = ca1.mouse + "_" + ca1.session
    an = ca1[ca1.category.isin(C.CATS)].reset_index(drop=True)
    obs, n_an = int(an.multi_after.sum()), len(an)
    groups = an.grp.unique()
    gidx = {g: np.flatnonzero(an.grp.values == g) for g in groups}
    s2 = C.s2_training("main")
    s7 = s7_cells()
    print("S7 single-variable cells with the true label found:", s7.groupby("cat").y.agg(["sum", "size"]).to_string())
    rows, boots = [], []
    for tname, t in (("s2", s2), ("s2+s7", pd.concat([s2, s7], ignore_index=True))):
        models = C.fit(t)
        p = C.predict(models, an.category.values, an.x.values)
        exp_n = p.sum()
        if tname == "s2+s7":
            per = an.assign(p=p).groupby("category").agg(n=("p", "size"), observed=("multi_after", "sum"),
                                                         expected=("p", "sum")).round(1)
            print(per.to_string())
            per.to_csv(DATA / "s3_percat_s2s7.csv")  # per category, read by 5_composition.py
        tcats = {c: np.flatnonzero(t.cat.values == c) for c in C.CATS}
        # session key: sessions of s2/ and s7/ of the same recording are the same behaviour
        skey = t.session.str.replace("s7_", "", regex=False).values
        sess = np.unique(skey)
        sidx = {s: np.flatnonzero(skey == s) for s in sess}
        for scheme in ("neuron", "session"):
            bt, bo = np.empty(B), np.empty(B)
            for b in range(B):
                if scheme == "neuron":
                    ti = np.concatenate([RNG.choice(ix, size=len(ix), replace=True) for ix in tcats.values()])
                else:
                    ss = RNG.choice(sess, size=len(sess), replace=True)
                    ti = np.concatenate([sidx[s] for s in ss])
                mb = C.fit(t.iloc[ti])
                gs = RNG.choice(groups, size=len(groups), replace=True)
                ci = np.concatenate([gidx[g] for g in gs])
                bt[b] = 100 * C.predict(mb, an.category.values[ci], an.x.values[ci]).sum() / len(ci)
                bo[b] = 100 * an.multi_after.values[ci].sum() / len(ci)
            lo, hi = np.percentile(bt, [2.5, 97.5])
            dlo, dhi = np.percentile(bo - bt, [2.5, 97.5])
            rows.append({"training": tname, "resampling": scheme, "expected_n": round(exp_n, 1),
                         "expected_per100": round(100 * exp_n / n_an, 2), "lo": round(lo, 2), "hi": round(hi, 2),
                         "width": round(hi - lo, 2), "observed_per100": round(100 * obs / n_an, 2),
                         "obs_minus_exp_lo": round(dlo, 2), "obs_minus_exp_hi": round(dhi, 2)})
            boots.append(pd.DataFrame({"variant": f"{tname}|{scheme}", "expected_per100": bt, "observed_per100": bo}))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(DATA / "s3_summary_s7.csv", index=False)
    pd.concat(boots).to_csv(DATA / "s3_bootstrap_s7.csv", index=False)

    # model-free on s7/ alone
    edges = np.array([0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.13, 0.17, 0.25])
    for c in ("place", "bodydirection"):
        d, a = s7[s7.cat == c], an[an.category == c]
        bd, ba = np.digitize(10 ** d.x, edges), np.digitize(a.max_excess, edges)
        e = o = n = 0
        for k in range(1, len(edges)):
            if (bd == k).sum() >= 20:
                e += (ba == k).sum() * d.y[bd == k].mean()
                o += a.multi_after[ba == k].sum()
                n += (ba == k).sum()
        print(f"[S7 alone, bins with >= 20 S7 cells] {c}: CA1 covered {n} of {len(a)}, observed {o} "
              f"({100 * o / max(n, 1):.1f} per 100), expected false {e:.1f} ({100 * e / max(n, 1):.1f} per 100)")


if __name__ == "__main__":
    main()
