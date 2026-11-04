"""Synthetic neurons of the S2 categories re-scored with the counting rule of the CA1 numbers
(numbers/recount_v8.py).

CA1 counts a neuron-session as multi-selective when, in NOF_post_disent.csv, two or more features
pass the paper criterion (p < 0.001, excess MI > 0.04 bits, activity ratio > 1 or NaN). That file is
built by tools/neuron_database of DRIADA 1.2.2, which renames a merged label 'place-Z' (Z one of
walls, corners, center, object1-4, objects) to 'Z' and gives it the zone's own statistics
(pretransform_merge_composite_place in database.py, component_stats of the merged entry). A merged
label is therefore ONE feature in CA1, while synthetic/score.py expands it into two.

This script applies the CA1 rule to the final labels of every synthetic neuron (order thrfirst), and
records the covariates of the strength calibration:
  true_excess   excess MI of the true variable at the INTENSE stage (bits), NaN if not computed
  max_label     label with the largest excess MI among labels that pass the criterion before
                disentanglement (plain CA1 names), max_excess its value
Input: <[s2] out_dir>/s2 (truth, results, tables of s2/s2_synth_fps_16.py).
Output: <[s2] out_dir>/data/s2_neurons_ca1rule.csv (one row per neuron).
usage: python s2/2_rescore_ca1rule.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402

S2 = P.get("s2", "out_dir") / "s2"
OUT = P.get("s2", "out_dir") / "data"
PVAL_THR, MI_THR = 0.001, 0.04
DISCRETE_PLACE = ["walls", "corners", "center", "object1", "object2", "object3", "object4", "objects"]
PLAIN = {"place_quad": "place", "speed_quad": "speed", "bodydirection_harm2": "bodydirection_2d",
         "headdirection_harm2": "headdirection_2d"}
TRUTH_REP = {"place": "place_quad", "speed": "speed_quad", "bodydirection": "bodydirection_harm2",
             "headdirection": "headdirection_harm2"}


def passes(s):
    if not s:
        return False
    sr, me_excess, pval = s.get("signal_ratio"), s.get("me_excess"), s.get("pval")
    sr_ok = sr is None or (isinstance(sr, float) and np.isnan(sr)) or sr > 1
    return pval is not None and pval < PVAL_THR and me_excess is not None and me_excess > MI_THR and sr_ok


def ca1_labels(final_stats):
    """Final labels as the CA1 database counts them: plain names, merged place-Z -> Z (zone stats)."""
    out = {}
    for f, s in final_stats.items():
        if f.startswith("place-") and f[6:] in DISCRETE_PLACE:
            zone = f[6:]
            comp = (s.get("component_stats") or {}).get(zone)
            st = dict(s)
            if isinstance(comp, dict):
                for k in ("me", "me_excess", "pval", "opt_delay", "signal_ratio"):
                    st[k] = comp.get(k)
            if passes(st):
                out[zone] = st.get("me_excess")
        else:
            if passes(s):
                out[PLAIN.get(f, f)] = s.get("me_excess")
    return out


def main():
    rows = []
    for tpath in sorted((S2 / "truth").glob("*.json")):
        truth = json.loads(tpath.read_text())
        with open(S2 / "results" / f"{tpath.stem}.pkl", "rb") as f:
            res = pickle.load(f)
        fin = res["orders"]["thrfirst"]["final_stats"]
        for tr in truth["neurons"]:
            nid = tr["neuron"]
            st = res["stats"].get(nid, {})
            before = {PLAIN.get(f, f): st[f].get("me_excess") for f in res["significant"].get(nid, [])
                      if passes(st.get(f))}
            after = ca1_labels(fin.get(nid, {}))
            T = tr["truth"]
            if len(T) == 1:
                rep = TRUTH_REP.get(T[0], T[0])
                true_excess = st.get(rep, {}).get("me_excess", np.nan)
                true_found = rep in res["significant"].get(nid, []) and passes(st.get(rep))
            else:
                true_excess, true_found = np.nan, False
            mx = max(before.items(), key=lambda kv: kv[1]) if before else (None, np.nan)
            rows.append({"session": tpath.stem, "neuron": nid, "category": tr["category"], "snr": tr["snr"],
                         "truth": "+".join(T), "true_excess": true_excess, "true_found": true_found,
                         "n_before": len(before), "before": ";".join(sorted(before)),
                         "max_label": mx[0], "max_excess": mx[1],
                         "n_after_ca1": len(after), "after_ca1": ";".join(sorted(after)),
                         "selective_ca1": len(after) >= 1, "multi_ca1": len(after) >= 2})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "s2_neurons_ca1rule.csv", index=False)

    # cross-check with the paper's scoring: neurons.csv thrfirst
    neu = pd.read_csv(S2 / "tables" / "neurons.csv")
    neu = neu[neu.order == "thrfirst"].set_index(["session", "neuron"])
    m = df.set_index(["session", "neuron"]).join(neu[["outcome", "before", "after"]], rsuffix="_paper")
    print("[check] true_found == outcome != missed:",
          int(((m.true_found) == (m.outcome != "missed"))[~m.category.str.contains(r"\+|null")].sum()), "of",
          int((~m.category.str.contains(r"\+|null")).sum()))
    print("[check] n labels before equal (paper vs here):",
          int((m.before_paper.fillna("").str.split(";").apply(lambda l: len([x for x in l if x])) == m.n_before).sum()),
          "of", len(m))

    pure = df[~df.category.str.contains(r"\+") & (df.category != "null")]
    t = pure.groupby("category").agg(found=("true_found", "sum"), selective=("selective_ca1", "sum"),
                                     multi=("multi_ca1", "sum"))
    t["multi_of_found_true"] = pure[pure.true_found].groupby("category").multi_ca1.sum()
    print("\n[S2 CA1-rule] pure categories: multi after disentanglement (CA1 counting)")
    print(t.to_string())
    po = pure[pure.category.isin(["place_in", "place_out", "object"])]
    print("\nplace+object: multi among true-found:", int(po[po.true_found].multi_ca1.sum()), "of",
          int(po.true_found.sum()), "| multi among selective:", int(po.multi_ca1.sum()), "of",
          int(po.selective_ca1.sum()))
    print(po[po.true_found].groupby("snr").multi_ca1.agg(["sum", "count"]).to_string())
    m2 = m[m.category.isin(["place_in", "place_out", "object"]) & (m.outcome == "extra")]
    print("\n[S2] the extras under the CA1 rule: multi", int(m2.multi_ca1.sum()), "of", len(m2))
    print(pd.crosstab(m2.category, m2.multi_ca1).to_string())
    nul = df[df.category == "null"]
    print("\nnull: selective", int(nul.selective_ca1.sum()), "of", len(nul))


if __name__ == "__main__":
    main()
