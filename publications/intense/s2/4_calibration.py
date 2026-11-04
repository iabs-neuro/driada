"""Expected number of false multi-selective CA1 neuron-sessions if every CA1 neuron coded one variable,
from the synthetic neurons of the S2 categories (files s3_* of data/).

Model on the synthetic neurons (pure categories, place_in and place_out pooled as 'place'):
  P(multi after disentanglement | category, strength), logistic in log10(strength), one fit per
  category (sklearn, C = 100, i.e. almost no penalty).
Main variant: synthetic neurons whose true label passed the criterion before
  disentanglement; strength = excess MI of the true label; outcome = multi under the CA1 counting
  rule (data/s2_neurons_ca1rule.csv, multi_ca1). CA1: strength = excess MI of the strongest label
  before disentanglement, category = its category (data/ca1_neuron_sessions.csv).
Variants:
  paper_extra   outcome = 'extra' of synthetic/score.py (merged place-Z counted as two labels), as in
                the scores of score.py; NOT the CA1 definition of multi-selective
  observable    synthetic neurons with >= 1 label before disentanglement; category and strength of
                the strongest label (the same observable as in CA1)
CA1 strengths outside the synthetic range of a category are clamped to the range; the share is reported.
Interval: bootstrap, 2000 replicates; synthetic neurons resampled within category (refit), CA1 resampled
by mouse-session (64 groups). Neuron-sessions whose strongest label is rear or freezing (no synthetic
counterpart) are left out of both the expected and the observed counts and reported apart.
s2/11_s3_with_s7.py, s2/16_significance.py and the script of Supplementary Fig. 8 import this file for
CATS, s2_training, fit and predict.
Outputs (<[s2] out_dir>/): data/s3_calibration.csv (per category), data/s3_bootstrap.csv,
         data/s3_strength_overlap.csv, data/s3_fitcheck.csv, data/s3_summary.csv,
         plots/s3_calibration.png
usage: python s2/4_calibration.py
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402

HERE = P.get("s2", "out_dir")
DATA, PLOTS = HERE / "data", HERE / "plots"
S2T = HERE / "s2" / "tables"
CATS = ["place", "object", "zone", "bodydirection", "headdirection", "locomotion", "speed"]
CAT_OF_LABEL = {"place": "place", "object1": "object", "object2": "object", "object3": "object",
                "object4": "object", "objects": "object", "bodydirection_2d": "bodydirection",
                "headdirection_2d": "headdirection", "locomotion": "locomotion", "walk": "locomotion",
                "rest": "locomotion", "speed": "speed", "corners": "zone", "walls": "zone", "center": "zone",
                "rear": "none", "freezing": "none"}
B = 2000
RNG = np.random.default_rng(20261102)


def s2_training(variant):
    s2 = pd.read_csv(DATA / "s2_neurons_ca1rule.csv")
    s2["category"] = s2.category.fillna("null")   # read_csv turns the category 'null' into NaN
    s2 = s2[~s2.category.str.contains(r"\+") & (s2.category != "null")].copy()
    s2["cat"] = s2.category.replace({"place_in": "place", "place_out": "place"})
    if variant == "main":
        t = s2[s2.true_found].copy()
        t["x"], t["y"] = np.log10(t.true_excess), t.multi_ca1
    elif variant == "paper_extra":
        neu = pd.read_csv(S2T / "neurons.csv")
        neu = neu[neu.order == "thrfirst"][["session", "neuron", "outcome"]]
        t = s2[s2.true_found].merge(neu, on=["session", "neuron"])
        t["x"], t["y"] = np.log10(t.true_excess), t.outcome == "extra"
    else:  # observable
        t = s2[s2.n_before >= 1].copy()
        t["cat"] = t.max_label.map(CAT_OF_LABEL)
        t = t[t.cat.isin(CATS)]
        t["x"], t["y"] = np.log10(t.max_excess), t.multi_ca1
    return t[["session", "neuron", "category", "cat", "snr", "x", "y"]].reset_index(drop=True)


def fit(t):
    models = {}
    for c in CATS:
        d = t[t.cat == c]
        y = d.y.values.astype(int)
        if y.min() == y.max():
            models[c] = ("const", float(y.mean()), d.x.min(), d.x.max())
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LogisticRegression(C=100.0, max_iter=1000).fit(d[["x"]].values, y)
        models[c] = (m, None, d.x.min(), d.x.max())
    return models


def predict(models, cats, x):
    p = np.zeros(len(x))
    for c in CATS:
        sel = cats == c
        if not sel.any():
            continue
        m, const, lo, hi = models[c]
        xc = np.clip(x[sel], lo, hi)
        p[sel] = const if m == "const" else m.predict_proba(xc[:, None])[:, 1]
    return p


def main():
    ca1 = pd.read_csv(DATA / "ca1_neuron_sessions.csv")
    ca1["x"] = np.log10(ca1.max_excess)
    ca1["grp"] = ca1.mouse + "_" + ca1.session
    an = ca1[ca1.category.isin(CATS)].reset_index(drop=True)
    none = ca1[~ca1.category.isin(CATS)]
    print(f"CA1 selective {len(ca1)}, with an S2 analogue {len(an)}, rear/freezing strongest {len(none)} "
          f"(multi after: {int(none.multi_after.sum())})")
    obs, n_an = int(an.multi_after.sum()), len(an)
    print(f"observed multi after (analogue subset): {obs} of {n_an} = {100 * obs / n_an:.2f} per 100")

    # strength overlap
    t_main = s2_training("main")
    ov = []
    for c in CATS:
        s, a = 10 ** t_main[t_main.cat == c].x, an[an.category == c].max_excess
        lo, hi = s.min(), s.max()
        ov.append({"category": c, "s2_n": len(s), "s2_q25": s.quantile(.25), "s2_median": s.median(),
                   "s2_q75": s.quantile(.75), "s2_min": lo, "s2_max": hi,
                   "ca1_n": len(a), "ca1_q25": a.quantile(.25), "ca1_median": a.median(), "ca1_q75": a.quantile(.75),
                   "ca1_below_s2_min": int((a < lo).sum()), "ca1_above_s2_max": int((a > hi).sum())})
    ov = pd.DataFrame(ov).round(4)
    ov.to_csv(DATA / "s3_strength_overlap.csv", index=False)
    print("\n[S3] strength (excess MI, bits): S2 true label of detected neurons vs CA1 strongest label")
    print(ov.to_string(index=False))

    results, boot_rows, per_cat = {}, [], []
    groups = an.grp.unique()
    gidx = {g: np.flatnonzero(an.grp.values == g) for g in groups}
    for variant in ["main", "paper_extra", "observable"]:
        t = s2_training(variant)
        models = fit(t)
        p = predict(models, an.category.values, an.x.values)
        exp_n = p.sum()
        # fit check: observed vs predicted on S2 by quintile of strength within category
        if variant == "main":
            chk = []
            for c in CATS:
                d = t[t.cat == c].copy()
                d["p"] = predict(models, np.array([c] * len(d)), d.x.values)
                d["q"] = pd.qcut(d.x, 5, labels=False, duplicates="drop")
                g = d.groupby("q").agg(n=("y", "size"), observed=("y", "sum"), predicted=("p", "sum"))
                g["category"] = c
                chk.append(g.reset_index())
            chk = pd.concat(chk).round(2)
            chk.to_csv(DATA / "s3_fitcheck.csv", index=False)
            print("\n[S3] fit check (S2, main): observed vs predicted multi per strength quintile")
            print(chk.groupby("category")[["n", "observed", "predicted"]].sum().to_string())
            for c in CATS:
                sel = an.category == c
                per_cat.append({"category": c, "ca1_n": int(sel.sum()), "ca1_multi_after": int(an.multi_after[sel].sum()),
                                "expected_false_multi": round(float(p[sel].sum()), 1),
                                "s2_n": int((t.cat == c).sum()), "s2_multi": int(t.y[t.cat == c].sum())})
        # bootstrap
        boot = np.empty(B)
        boot_obs = np.empty(B)
        tcats = {c: np.flatnonzero(t.cat.values == c) for c in CATS}
        for b in range(B):
            ti = np.concatenate([RNG.choice(ix, size=len(ix), replace=True) for ix in tcats.values()])
            mb = fit(t.iloc[ti])
            gs = RNG.choice(groups, size=len(groups), replace=True)
            ci = np.concatenate([gidx[g] for g in gs])
            pb = predict(mb, an.category.values[ci], an.x.values[ci])
            boot[b] = 100 * pb.sum() / len(ci)
            boot_obs[b] = 100 * an.multi_after.values[ci].sum() / len(ci)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        dlo, dhi = np.percentile(boot_obs - boot, [2.5, 97.5])
        results[variant] = dict(expected_n=exp_n, expected_per100=100 * exp_n / n_an, lo=lo, hi=hi,
                                width=hi - lo, obs_minus_exp_lo=dlo, obs_minus_exp_hi=dhi)
        boot_rows.append(pd.DataFrame({"variant": variant, "expected_per100": boot, "observed_per100": boot_obs}))
        print(f"\n[S3] {variant}: expected false multi {exp_n:.0f} of {n_an} = {100 * exp_n / n_an:.2f} per 100, "
              f"95% interval {lo:.2f}-{hi:.2f} (width {hi - lo:.2f}); observed {100 * obs / n_an:.2f}; "
              f"observed - expected 95% interval {dlo:.2f}..{dhi:.2f}")
    pc = pd.DataFrame(per_cat)
    pc["ca1_multi_per100"] = (100 * pc.ca1_multi_after / pc.ca1_n).round(1)
    pc["expected_per100"] = (100 * pc.expected_false_multi / pc.ca1_n).round(1)
    pc["s2_multi_per100"] = (100 * pc.s2_multi / pc.s2_n).round(1)
    pc.to_csv(DATA / "s3_calibration.csv", index=False)
    print("\n[S3] per category (main):")
    print(pc.to_string(index=False))
    pd.DataFrame(results).T.round(3).to_csv(DATA / "s3_summary.csv")
    pd.concat(boot_rows).to_csv(DATA / "s3_bootstrap.csv", index=False)
    print(pd.DataFrame(results).T.round(2).to_string())

    # figure: per category, S2 multi rate vs strength (binned) and model; CA1 strength histogram
    t = s2_training("main")
    models = fit(t)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True)
    bins = np.linspace(np.log10(0.04), np.log10(1.5), 12)
    for ax, c in zip(axes.flat, CATS + ["all"]):
        if c == "all":
            ax.axis("off")
            r = results["main"]
            ax.text(0, 0.5, f"CA1 observed: {100 * obs / n_an:.1f} per 100\n"
                            f"expected false (S2 model):\n{r['expected_per100']:.1f} per 100 "
                            f"[{r['lo']:.1f}-{r['hi']:.1f}]", fontsize=11, va="center")
            continue
        d = t[t.cat == c]
        a = an[an.category == c]
        ax2 = ax.twinx()
        ax2.hist(a.x, bins=bins, color="0.85", density=True)
        ax2.set_yticks([])
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        idx = np.digitize(d.x, bins)
        for k in np.unique(idx):
            s = d.y[idx == k]
            if len(s) >= 5:
                ax.plot((bins[min(k, len(bins) - 1)] + bins[k - 1]) / 2, s.mean(), "o", color="C0")
        xs = np.linspace(d.x.min(), d.x.max(), 50)
        ax.plot(xs, predict(models, np.array([c] * 50), xs), color="C0")
        ax.set_ylim(-0.02, 1)
        ax.set_title(f"{c}: CA1 {int(a.multi_after.sum())}/{len(a)} multi")
        ax.set_xticks(np.log10([0.05, 0.1, 0.2, 0.5, 1]))
        ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1"])
    for ax in axes[1]:
        ax.set_xlabel("excess MI of the label, bits (log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("P(multi after), S2 synthetic")
    fig.suptitle("False multi-selectivity of single-variable synthetic neurons vs strength (dots, line); "
                 "gray: CA1 strength distribution", fontsize=10)
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "s3_calibration.png", dpi=130)


if __name__ == "__main__":
    main()
