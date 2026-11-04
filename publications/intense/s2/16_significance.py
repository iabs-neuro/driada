"""Significance of the CA1 excess over the expected disentanglement error (Supplementary Section S2:
no draw of 2,000 reached 11.2%, p < 0.001; observed at least expected in all 16 mice, one-sided
Wilcoxon signed-rank test, p = 1.5e-5; panel c of Supplementary Fig. 8).

H0: every selective CA1 neuron-session codes a single variable. Then a neuron-session stays multi-selective
after disentanglement only through the procedure's error, with the probability that the model of
s2/11_s3_with_s7.py fitted on the synthetic neurons of the S2 categories and of s7/ gives for its category
and strength (the expected 4.5 per 100).
Observed: 868 of 7,745 neuron-sessions with a synthetic counterpart = 11.21 per 100 (s2/11_s3_with_s7.py).

Test A, main (resampled as for the interval s2+s7|neuron of s2/11_s3_with_s7.py):
  per draw: synthetic single-variable cells resampled within category and the model refitted; CA1 resampled
  by mouse-session (64 groups); null share = Bernoulli draws with the refitted probabilities on the resampled
  CA1 set (the chance variation of a count, which the expectation interval leaves out).
  P = (1 + #{null share >= observed share}) / (B + 1), one-sided.
  Also from the same draws: P_diff = (1 + #{resampled observed - expected <= 0}) / (B + 1).
Test B, conservative: synthetic resampled by recorded session (16 behaviours, the cells of s2/ and s7/ of a
  session together) and CA1 resampled by mouse (16 groups, all sessions of a mouse together). Same P.
Test C, per animal (16 mice, the unit that is independent): point model; per mouse observed vs expected
  multi-selective count; one-sided Wilcoxon signed-rank test and sign test on observed - expected.
B draws per test, from the command line; the default is 10,000, the paper used 2,000:
  python s2/16_significance.py 2000

Outputs (<[s2] out_dir>/data/): s8_significance.csv (one row per test), s8_significance_permouse.csv,
         s8_significance_draws.csv (read by the script of Supplementary Fig. 8)
usage: python s2/16_significance.py [B]

With the second argument B, test B alone is run with B draws and another seed, so that its tail is
estimated rather than bounded; it writes data/s8_significance_B_<draws>.csv and the matching draws
file. The paper does not use it.
usage: python s2/16_significance.py 10000 B
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

HERE = P.get("s2", "out_dir")
DATA = HERE / "data"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("cal", "4_calibration.py")
S = _load("s3s7", "11_s3_with_s7.py")
B = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
ONLY_B = len(sys.argv) > 2 and sys.argv[2] == "B"
RNG = np.random.default_rng(202611041 if ONLY_B else 20261104)


def main():
    ca1 = pd.read_csv(DATA / "ca1_neuron_sessions.csv")
    ca1["x"] = np.log10(ca1.max_excess)
    ca1["grp"] = ca1.mouse + "_" + ca1.session
    an = ca1[ca1.category.isin(C.CATS)].reset_index(drop=True)
    y_obs = an.multi_after.values.astype(int)
    n_an, obs = len(an), int(y_obs.sum())
    obs_share = 100 * obs / n_an
    cats, xs = an.category.values, an.x.values

    t = pd.concat([C.s2_training("main"), S.s7_cells()], ignore_index=True)
    models = C.fit(t)
    p_point = C.predict(models, cats, xs)
    print(f"observed {obs} of {n_an} = {obs_share:.2f} per 100; expected (point) {p_point.sum():.1f} "
          f"= {100 * p_point.sum() / n_an:.2f} per 100", flush=True)

    tcats = {c: np.flatnonzero(t.cat.values == c) for c in C.CATS}
    skey = t.session.str.replace("s7_", "", regex=False).values
    sess = np.unique(skey)
    sidx = {s: np.flatnonzero(skey == s) for s in sess}

    rows, draws = [], []
    tests = (("A_main", "neuron", "mouse-session"), ("B_conservative", "session", "mouse"))
    for name, syn_unit, ca1_unit in tests[1:] if ONLY_B else tests:
        key = an.grp.values if ca1_unit == "mouse-session" else an.mouse.values
        groups = np.unique(key)
        gidx = {g: np.flatnonzero(key == g) for g in groups}
        null, exp_b, obs_b = np.empty(B), np.empty(B), np.empty(B)
        t0 = time.time()
        for b in range(B):
            if syn_unit == "neuron":
                ti = np.concatenate([RNG.choice(ix, size=len(ix), replace=True) for ix in tcats.values()])
            else:
                ss = RNG.choice(sess, size=len(sess), replace=True)
                ti = np.concatenate([sidx[s] for s in ss])
            mb = C.fit(t.iloc[ti])
            gs = RNG.choice(groups, size=len(groups), replace=True)
            ci = np.concatenate([gidx[g] for g in gs])
            pb = C.predict(mb, cats[ci], xs[ci])
            null[b] = 100 * (RNG.random(len(ci)) < pb).sum() / len(ci)
            exp_b[b] = 100 * pb.sum() / len(ci)
            obs_b[b] = 100 * y_obs[ci].sum() / len(ci)
            if b == 99:
                print(f"  {name}: 100 draws in {time.time() - t0:.1f} s", flush=True)
        k_null = int((null >= obs_share).sum())
        k_diff = int((obs_b - exp_b <= 0).sum())
        rows.append({"test": name, "synthetic_resampled_by": syn_unit, "ca1_resampled_by": ca1_unit, "draws": B,
                     "observed_per100": round(obs_share, 2),
                     "null_median": round(float(np.median(null)), 2),
                     "null_q975": round(float(np.percentile(null, 97.5)), 2),
                     "null_q999": round(float(np.percentile(null, 99.9)), 2),
                     "null_max": round(float(null.max()), 2),
                     "null_ge_observed": k_null, "P_null": (1 + k_null) / (B + 1),
                     "diff_lo": round(float(np.percentile(obs_b - exp_b, 2.5)), 2),
                     "diff_hi": round(float(np.percentile(obs_b - exp_b, 97.5)), 2),
                     "diff_min": round(float((obs_b - exp_b).min()), 2),
                     "diff_le_0": k_diff, "P_diff": (1 + k_diff) / (B + 1)})
        draws.append(pd.DataFrame({"test": name, "null_per100": null, "expected_per100": exp_b,
                                   "observed_per100": obs_b}))
        print(rows[-1], flush=True)

    if ONLY_B:
        pd.DataFrame(rows).to_csv(DATA / f"s8_significance_B_{B}.csv", index=False)
        pd.concat(draws).round(4).to_csv(DATA / f"s8_significance_B_{B}_draws.csv", index=False)
        return

    # Test C: per mouse, point model
    pm = an.assign(p=p_point).groupby("mouse").agg(n=("p", "size"), observed=("multi_after", "sum"),
                                                     expected=("p", "sum"))
    pm["observed_per100"] = 100 * pm.observed / pm.n
    pm["expected_per100"] = 100 * pm.expected / pm.n
    pm["diff"] = pm.observed - pm.expected
    w = stats.wilcoxon(pm["diff"], alternative="greater")
    k_pos = int((pm["diff"] > 0).sum())
    sign_p = stats.binomtest(k_pos, len(pm), 0.5, alternative="greater").pvalue
    rows.append({"test": "C_per_mouse", "synthetic_resampled_by": "", "ca1_resampled_by": "", "draws": "",
                 "observed_per100": round(obs_share, 2), "mice": len(pm), "mice_observed_gt_expected": k_pos,
                 "wilcoxon_P": w.pvalue, "sign_test_P": sign_p})
    print(pm.round(2).to_string())
    print(rows[-1])

    pd.DataFrame(rows).to_csv(DATA / "s8_significance.csv", index=False)
    pm.round(3).to_csv(DATA / "s8_significance_permouse.csv")
    pd.concat(draws).round(4).to_csv(DATA / "s8_significance_draws.csv", index=False)


if __name__ == "__main__":
    main()
