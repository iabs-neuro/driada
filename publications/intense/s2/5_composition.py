"""Which label combinations make up the CA1 neurons that stay multiply selective after disentanglement,
and which of them the synthetic neurons produce falsely (panel b of Supplementary Fig. 8; the 544 of 868
selective for position and body direction of Supplementary Section S2).

CA1: the 868 neuron-sessions with two or more labels after disentanglement (CA1 counting rule),
data/ca1_neuron_sessions.csv. Synthetic: pure neurons of the S2 categories multi under the same rule,
data/s2_neurons_ca1rule.csv, and the single-variable cells of s7/, data/s7_neurons.csv.
Labels are grouped into types: place; bodydirection; headdirection; zone (corners, walls, center);
object (object1-4, objects); loco (locomotion, walk, rest); speed; rear; freezing.
Expected false count per combination (column ca1_expected_false): for every category c of the strongest
CA1 label, the expected false multi count of the model of s2/11_s3_with_s7.py (synthetic neurons of the
S2 categories and of s7/; data/s3_percat_s2s7.csv), split by the combinations of the false multi of
those synthetic neurons of category c (shares within c). Column ca1_expected_false_s2only: the same with
the model of s2/4_calibration.py (data/s3_calibration.csv) and the combinations of the S2 categories only.
Also a model-free check of the calibration at matched strength: within each category, strength bins with
at least 30 synthetic neurons; CA1 expected = sum over bins of (CA1 neurons in the bin x synthetic rate
in the bin).
Outputs (<[s2] out_dir>/): data/s4_composition.csv (read by the script of Supplementary Fig. 8),
         data/s3_matched_bins.csv, plots/s4_composition.png
usage: python s2/5_composition.py
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

HERE = P.get("s2", "out_dir")
DATA, PLOTS = HERE / "data", HERE / "plots"
TYPE = {"place": "place", "bodydirection_2d": "BD", "headdirection_2d": "HD", "corners": "zone", "walls": "zone",
        "center": "zone", "object1": "object", "object2": "object", "object3": "object", "object4": "object",
        "objects": "object", "locomotion": "loco", "walk": "loco", "rest": "loco", "speed": "speed",
        "rear": "rear", "freezing": "freezing"}
CATS = ["place", "object", "zone", "bodydirection", "headdirection", "locomotion", "speed"]


def combo(labels):
    ts = sorted({TYPE.get(l, l) for l in str(labels).split(";") if l})
    return "+".join(ts)


def main():
    ca1 = pd.read_csv(DATA / "ca1_neuron_sessions.csv")
    s2 = pd.read_csv(DATA / "s2_neurons_ca1rule.csv")
    s2["category"] = s2.category.fillna("null")
    s2 = s2[~s2.category.str.contains(r"\+") & (s2.category != "null")].copy()
    s2["cat"] = s2.category.replace({"place_in": "place", "place_out": "place"})
    s2t = s2[s2.true_found]

    cm = ca1[ca1.multi_after].copy()
    cm["combo"] = cm.after.apply(combo)
    obs = cm.combo.value_counts()
    print(f"CA1 multi after: {len(cm)}")
    sm = s2t[s2t.multi_ca1].copy()
    sm["combo"] = sm.after_ca1.apply(combo)
    print(f"S2 false multi (true label found): {len(sm)}")

    # expected false by combination: S2 model and S2 combinations (4_calibration.py)
    cal = pd.read_csv(DATA / "s3_calibration.csv").set_index("category")
    exp_s2 = Counter()
    for c in CATS:
        g = sm[sm.cat == c].combo.value_counts(normalize=True)
        for k, v in g.items():
            exp_s2[k] += cal.loc[c, "expected_false_multi"] * v
    # main: model of 11_s3_with_s7.py (cells of s2/ and s7/) and the combinations of their false multi
    s7 = pd.read_csv(DATA / "s7_neurons.csv")
    s7 = s7[(s7.category != "place+bodydirection") & s7.found_1 & s7.multi].copy()
    s7["cat"] = s7.category.replace({"place_in": "place", "place_out": "place"})
    s7["combo"] = s7.after_ca1.apply(combo)
    sm7 = pd.concat([sm[["cat", "combo"]], s7[["cat", "combo"]]], ignore_index=True)
    per = pd.read_csv(DATA / "s3_percat_s2s7.csv").set_index("category")
    exp = Counter()
    for c in CATS:
        g = sm7[sm7.cat == c].combo.value_counts(normalize=True)
        for k, v in g.items():
            exp[k] += per.loc[c, "expected"] * v
    print(f"expected false, S2 + S7 model: {sum(exp.values()):.1f}; S2 model: {sum(exp_s2.values()):.1f}")
    s2c = sm.combo.value_counts()
    s7c = s7.combo.value_counts()
    keys = sorted(set(obs.index) | set(exp) | set(exp_s2), key=lambda k: -obs.get(k, 0))
    tab = pd.DataFrame({"combo": keys,
                        "ca1_observed": [int(obs.get(k, 0)) for k in keys],
                        "ca1_expected_false": [round(exp.get(k, 0.0), 1) for k in keys],
                        "ca1_expected_false_s2only": [round(exp_s2.get(k, 0.0), 1) for k in keys],
                        "s2_false_multi": [int(s2c.get(k, 0)) for k in keys],
                        "s7_false_multi": [int(s7c.get(k, 0)) for k in keys]})
    tab["ca1_minus_expected"] = (tab.ca1_observed - tab.ca1_expected_false).round(1)
    tab.to_csv(DATA / "s4_composition.csv", index=False)
    print(tab.head(25).to_string(index=False))
    pdir = tab[tab.combo.isin(["BD+place", "HD+place", "BD+HD+place"])]
    print("\nplace+direction combos: observed", int(pdir.ca1_observed.sum()), "expected false",
          round(pdir.ca1_expected_false.sum(), 1))
    print("S2 false multi by category and combo:")
    print(sm.groupby(["cat", "combo"]).size().to_string())

    # model-free matched bins
    edges = np.array([0.04, 0.06, 0.08, 0.1, 0.13, 0.17, 0.22, 0.3, 0.4, 0.7])
    rows = []
    an = ca1[ca1.category.isin(CATS)]
    for c in CATS:
        d, a = s2t[s2t.cat == c], an[an.category == c]
        bd = np.digitize(d.true_excess, edges)
        ba = np.digitize(a.max_excess, edges)
        for k in range(1, len(edges)):
            n2, na = int((bd == k).sum()), int((ba == k).sum())
            rows.append({"category": c, "bin_lo": edges[k - 1], "bin_hi": edges[k], "s2_n": n2,
                         "s2_multi": int(d.multi_ca1[bd == k].sum()), "ca1_n": na,
                         "ca1_multi": int(a.multi_after[ba == k].sum())})
    mb = pd.DataFrame(rows)
    mb.to_csv(DATA / "s3_matched_bins.csv", index=False)
    ok = mb[mb.s2_n >= 30]
    e = (ok.ca1_n * ok.s2_multi / ok.s2_n).sum()
    print(f"\n[S3 matched bins, S2 n >= 30] CA1 neurons covered {int(ok.ca1_n.sum())} of {len(an)}; "
          f"observed multi {int(ok.ca1_multi.sum())} = {100 * ok.ca1_multi.sum() / ok.ca1_n.sum():.2f} per 100; "
          f"expected false {e:.0f} = {100 * e / ok.ca1_n.sum():.2f} per 100")
    for c in CATS:
        o = ok[ok.category == c]
        if len(o):
            ee = (o.ca1_n * o.s2_multi / o.s2_n).sum()
            print(f"  {c}: covered {int(o.ca1_n.sum())}, observed {int(o.ca1_multi.sum())}, expected {ee:.1f}")

    # figure: observed vs expected for the top combinations
    top = tab.head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(top))
    ax.bar(xs - 0.2, top.ca1_observed, 0.4, label="CA1 observed after disentanglement")
    ax.bar(xs + 0.2, top.ca1_expected_false, 0.4, label="expected false if every neuron coded one variable")
    ax.set_xticks(xs)
    ax.set_xticklabels(top.combo, rotation=40, ha="right")
    ax.set_ylabel("neuron-sessions")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("CA1 multi-selective neuron-sessions by label combination", fontsize=10)
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "s4_composition.png", dpi=130)


if __name__ == "__main__":
    main()
