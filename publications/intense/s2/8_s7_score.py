"""The synthetic neurons of s7/ (s2/7_s7_make.py, s2/7_s7_run.py) scored with the CA1 counting rule
(as s2/2_rescore_ca1rule.py).

Single-variable cells (place_in, place_out, bodydirection): multi after disentanglement = false
multi-selectivity, against the excess MI of the true label (CA1 strength range).
Mixed place + body direction cells: sensitivity = share of selective cells (>= 1 label after) that
end with >= 2 labels; both_kept = share keeping place and body direction among those whose two labels
passed the criterion before disentanglement (the 330 of 330 of Supplementary Section S2).
Output: <[s2] out_dir>/data/s7_neurons.csv; printed tables.
usage: python s2/8_s7_score.py
"""
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402

HERE = P.get("s2", "out_dir")
S7 = HERE / "s7"
_spec = importlib.util.spec_from_file_location("rescore", Path(__file__).resolve().parent / "2_rescore_ca1rule.py")
RS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RS)


def main():
    rows = []
    for tpath in sorted((S7 / "truth").glob("*.json")):
        rpath = S7 / "results" / f"{tpath.stem}.pkl"
        if not rpath.exists():
            continue
        truth = json.loads(tpath.read_text())
        with open(rpath, "rb") as f:
            res = pickle.load(f)
        fin = res["orders"]["thrfirst"]["final_stats"]
        for tr in truth["neurons"]:
            nid = tr["neuron"]
            st = res["stats"].get(nid, {})
            sig = res["significant"].get(nid, [])
            before = {RS.PLAIN.get(f, f): st[f].get("me_excess") for f in sig if RS.passes(st.get(f))}
            after = RS.ca1_labels(fin.get(nid, {}))
            T = tr["truth"]
            reps = [RS.TRUTH_REP.get(t, t) for t in T]
            exc = [st.get(r, {}).get("me_excess", np.nan) for r in reps]
            found = [r in sig and RS.passes(st.get(r)) for r in reps]
            plain_after = set(after)
            rows.append({"session": tpath.stem, "neuron": nid, "category": tr["category"], "snr": tr["snr"],
                         "truth": "+".join(T), "excess_1": exc[0], "excess_2": exc[1] if len(exc) > 1 else np.nan,
                         "found_1": found[0], "found_2": found[1] if len(found) > 1 else np.nan,
                         "n_before": len(before), "before": ";".join(sorted(before)),
                         "n_after_ca1": len(after), "after_ca1": ";".join(sorted(after)),
                         "selective": len(after) >= 1, "multi": len(after) >= 2,
                         "has_place_after": bool(plain_after & {"place", "walls", "corners", "center", "object1",
                                                                "object2", "object3", "object4", "objects"}),
                         "has_bd_after": "bodydirection_2d" in plain_after})
    df = pd.DataFrame(rows)
    (HERE / "data").mkdir(parents=True, exist_ok=True)
    df.to_csv(HERE / "data" / "s7_neurons.csv", index=False)
    print("sessions scored:", df.session.nunique(), "neurons:", len(df))
    pure = df[df.category != "place+bodydirection"]
    print("\n[S7] single-variable cells, true label found: multi after (false) by category x SNR")
    f = pure[pure.found_1]
    print(f.groupby(["category", "snr"]).agg(found=("multi", "size"), multi=("multi", "sum"),
                                             median_excess=("excess_1", "median")).round(3).to_string())
    print("\ntotal:", f.groupby("category").multi.agg(["sum", "count"]).to_string())
    print("place (in+out):", int(f[f.category != "bodydirection"].multi.sum()), "of",
          int((f.category != "bodydirection").sum()))
    print("found / generated:", pure.groupby("category").found_1.agg(["sum", "size"]).to_string())
    print("\nfalse multi combinations:", f[f.multi].after_ca1.value_counts().head(10).to_string())
    mx = df[df.category == "place+bodydirection"]
    sel = mx[mx.selective]
    both_found = mx[(mx.found_1 == True) & (mx.found_2 == True)]  # noqa: E712
    print(f"\n[S7] place + body direction cells: {len(mx)}, selective {len(sel)}, multi {int(sel.multi.sum())} "
          f"(sensitivity {100 * sel.multi.mean():.1f} per 100 selective); both labels before {len(both_found)}, "
          f"both kept {int((both_found.has_place_after & both_found.has_bd_after).sum())}")
    print(sel.groupby("snr").agg(selective=("multi", "size"), multi=("multi", "sum"),
                                 place_exc=("excess_1", "median"), bd_exc=("excess_2", "median")).round(3).to_string())


if __name__ == "__main__":
    main()
