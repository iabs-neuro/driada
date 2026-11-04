"""Supplementary Table 6 by one rule, and the same count for every category of the synthetic neurons.

Rule (as the CA1 numbers): the final labels of a neuron (order thrfirst) are counted as the CA1
database counts them (2_rescore_ca1rule.py): a label merged with a zone or an object, place-Z, is one
label Z with the zone's statistics. A neuron whose true label passed the selection criterion before
disentanglement then either keeps one label (the true one, or the merge of the true variable with the
zone or object at its place) or the true label together with others; the outcome of synthetic/score.py
(s2/tables/neurons.csv) tells whether a single label is the true variable or another one.

Table 6 rows: place field inside a zone, place field outside zones, object.
  passed      true label passed the criterion before disentanglement (true_found)
  alone       passed and one label after (CA1 counting)
  and_others  passed and two or more labels after (CA1 counting)
  one_label_other_variable  passed and one label after that is not the true variable (score.py
              outcome 'rewritten'): the column "another label" of the table
  several_labels_true_lost, no_label  passed and two or more labels after without the true
              variable; passed and no label after. With one_label_other_variable they count the
              true labels that disentanglement reassigned or removed.
The rule of score.py (merged place-Z counted as two labels unless the field centre lies in Z) is
given beside it (columns v11_alone, v11_and_others).

All categories: the same columns for the single-variable categories; for the mixed ones the outcomes of
score.py (both, one, rewritten, lost) and the CA1 count of labels; for null, how many got any label.

usage: python s2/table6.py [--s2 DIR] [--rescore CSV] [--out PREFIX]
       defaults: <[s2] out_dir>/s2, <[s2] out_dir>/data/s2_neurons_ca1rule.csv,
       <[s2] out_dir>/data/table6 (-> table6.csv, table6_all_categories.csv)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402

HERE = P.get("s2", "out_dir")
ROWS = [("place_in", "Place field inside a zone"), ("place_out", "Place field outside zones"),
        ("object", "Object")]
SINGLE = ["place_in", "place_out", "object", "zone", "speed", "locomotion", "headdirection", "bodydirection"]
MIXED = ["speed+locomotion", "place+zone", "zone+object", "headdirection+bodydirection"]


def load(s2, rescore):
    rs = pd.read_csv(rescore)
    rs["category"] = rs.category.fillna("null")   # read_csv reads the category 'null' as NaN
    neu = pd.read_csv(Path(s2) / "tables" / "neurons.csv")
    neu["category"] = neu.category.fillna("null")
    neu = neu[neu.order == "thrfirst"][["session", "neuron", "outcome", "after"]]
    m = rs.merge(neu, on=["session", "neuron"], how="left", validate="one_to_one")
    assert m.outcome.notna().all() and len(m) == len(rs)
    return m


def single_counts(d):
    f = d[d.true_found]
    keep = f.outcome.isin(["exact", "exact_merged", "extra"])
    one, many = f.n_after_ca1 == 1, f.n_after_ca1 >= 2
    out = {"generated": len(d), "passed": len(f),
           "alone": int((one & keep).sum()), "and_others": int((many & keep).sum()),
           "one_label_other_variable": int((one & ~keep).sum()),
           "several_labels_true_lost": int((many & ~keep).sum()),
           "no_label": int((f.n_after_ca1 == 0).sum()),
           "v11_alone": int(f.outcome.isin(["exact", "exact_merged"]).sum()),
           "v11_and_others": int((f.outcome == "extra").sum()),
           "check_found_vs_outcome": int((f.outcome == "missed").sum())}
    out["check_sum"] = out["passed"] - (out["alone"] + out["and_others"] + out["one_label_other_variable"]
                                        + out["several_labels_true_lost"] + out["no_label"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", default=str(HERE / "s2"))
    ap.add_argument("--rescore", default=str(HERE / "data" / "s2_neurons_ca1rule.csv"))
    ap.add_argument("--out", default=str(HERE / "data" / "table6"))
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    m = load(a.s2, a.rescore)
    print(f"sessions {m.session.nunique()}, neurons {len(m)}")

    t6 = pd.DataFrame([{"row": label, "category": c, **single_counts(m[m.category == c])} for c, label in ROWS])
    t6.to_csv(f"{a.out}.csv", index=False)
    print("\n[Table 6, one rule]")
    print(t6[["row", "generated", "passed", "alone", "and_others", "one_label_other_variable",
              "several_labels_true_lost", "no_label", "v11_alone", "v11_and_others", "check_sum"]].to_string(index=False))

    rows = [{"category": c, **single_counts(m[m.category == c])} for c in SINGLE]
    for c in MIXED:
        d = m[m.category == c]
        oc = d.outcome.value_counts()
        found = d[d.outcome != "missed"]
        rows.append({"category": c, "generated": len(d), "passed": len(found),
                     "both": int(oc.get("both", 0)), "one": int(oc.get("one", 0)),
                     "rewritten": int(oc.get("rewritten", 0)), "lost": int(oc.get("lost", 0)),
                     "ca1_two_or_more_labels": int((found.n_after_ca1 >= 2).sum()),
                     "ca1_one_label": int((found.n_after_ca1 == 1).sum())})
    d = m[m.category == "null"]
    rows.append({"category": "null", "generated": len(d), "null_any_label": int((d.n_after_ca1 >= 1).sum())})
    allc = pd.DataFrame(rows)
    allc.to_csv(f"{a.out}_all_categories.csv", index=False)
    print("\n[all categories]")
    print(allc.to_string(index=False))


if __name__ == "__main__":
    main()
