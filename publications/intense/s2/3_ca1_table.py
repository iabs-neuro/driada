"""CA1 neuron-sessions of the paper run, one row each: strongest label before disentanglement and the
labels after it, counted exactly as numbers/recount_v8.py (DRIADA 1.2.2 with conditional MI at the
delays of the pair, [v11] cmi_delays = "aligned").

Input: [results] db_dir: NOF_pre_disent.csv, NOF_post_disent.csv (compute/build_db_fps.py
--round-boundary on the results of compute/run_disent_pair_delays.py).
Paper criterion: p < 0.001, signal_ratio NaN or > 1, excess MI > 0.04 bits; 'any place' and
'any object' excluded.
Category of the strongest label: place -> place; object1-4, objects -> object;
bodydirection_2d -> bodydirection; headdirection_2d -> headdirection; locomotion, walk, rest ->
locomotion; speed -> speed; corners, walls, center -> zone; rear, freezing -> none (no synthetic
counterpart).
Output: <[s2] out_dir>/data/ca1_neuron_sessions.csv
usage: python s2/3_ca1_table.py
"""
import sys
from pathlib import Path

import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402

OUT = P.get("s2", "out_dir") / "data"
KEY = ["mouse", "session", "neuron_idx"]
CAT = {"place": "place", "object1": "object", "object2": "object", "object3": "object", "object4": "object",
       "objects": "object", "bodydirection_2d": "bodydirection", "headdirection_2d": "headdirection",
       "locomotion": "locomotion", "walk": "locomotion", "rest": "locomotion", "speed": "speed",
       "corners": "zone", "walls": "zone", "center": "zone", "rear": "none", "freezing": "none"}


def paper(d):
    d = d[~d.feature.isin(["any place", "any object"])]
    d = d[(d.pval < 0.001) & (d.signal_ratio.isna() | (d.signal_ratio > 1))]
    return d[d.me_excess > 0.04]


def main():
    pre = paper(pd.read_csv(P.DB_PRE))
    post = paper(pd.read_csv(P.DB_POST))
    top = pre.sort_values("me_excess", ascending=False).drop_duplicates(KEY)
    top = top[KEY + ["feature", "me_excess"]].rename(columns={"feature": "max_label", "me_excess": "max_excess"})
    gpre = pre.groupby(KEY).feature.apply(lambda s: ";".join(sorted(set(s)))).rename("before")
    gpost = post.groupby(KEY).feature.apply(lambda s: ";".join(sorted(set(s)))).rename("after")
    df = top.set_index(KEY).join(gpre).join(gpost).reset_index()
    df["n_before"] = df.before.str.split(";").str.len()
    df["n_after"] = df.after.fillna("").apply(lambda s: len([x for x in s.split(";") if x]))
    df["multi_before"] = df.n_before >= 2
    df["multi_after"] = df.n_after >= 2
    df["category"] = df.max_label.map(CAT)
    # strongest label after disentanglement too (for the composition step)
    topp = post.sort_values("me_excess", ascending=False).drop_duplicates(KEY)
    df = df.merge(topp[KEY + ["feature", "me_excess"]].rename(columns={"feature": "max_label_after",
                                                                         "me_excess": "max_excess_after"}),
                  on=KEY, how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "ca1_neuron_sessions.csv", index=False)
    print("selective (pre):", len(df), "| multi pre:", int(df.multi_before.sum()),
          "| selective post:", int((df.n_after >= 1).sum()), "| multi post:", int(df.multi_after.sum()))
    print("sessions:", df.groupby(["mouse", "session"]).ngroups)
    print(df.category.value_counts(dropna=False).to_string())
    print(df.groupby("category").multi_after.agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    main()
