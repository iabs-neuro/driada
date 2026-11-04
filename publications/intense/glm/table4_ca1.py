"""Columns of Supplementary Table 4 (CA1: GLM, INTENSE and both, per feature) from saved INTENSE results.

INTENSE side: every *_results.npz of a results folder (layout of save_intense_results of the
paper's tool), pairs accepted by the criterion of the paper (ca1_common.paper_accepts) for the
three features of the table (position, head direction, speed in their type-based
representations). GLM side: a per-neuron CSV with columns mouse, session, neuron_idx and
glm_position, glm_head_direction, glm_speed: the forward-selection GLM of Section S1.4 by
default ([v11] glm_ca1_per_neuron), or the per-feature GLM (glm/ca1_glm_numbers.py).

On the results of the paper run ([v11] stage2_results, with delay search) it gives the published
table: position 371 / 5,566 / 298, head direction 22 / 532 / 10, speed 86 / 170 / 14.

usage: python glm/table4_ca1.py --results DIR --out JSON [--glm CSV] [--pairs-csv CSV] [--label NAME]
       (glm/ca1_numbers.py and glm/ca1_glm_numbers.py call it)
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import paths_v11 as V  # noqa: E402
import ca1_common as C  # noqa: E402

import pandas as pd  # noqa: E402

KEY = ["mouse", "session", "neuron_idx"]
FEATS = ["position", "head_direction", "speed"]


def intense_pairs(results_dir):
    rows = []
    files = sorted(Path(results_dir).glob("NOF_*_results.npz"))
    for f in files:
        _, mouse, day = f.name[: -len("_results.npz")].split("_")
        stats, sig, _ = C.load_results_npz(f)
        for cell, feats in stats.items():
            for feat, table_name in C.CA1_TABLE4.items():
                st = feats.get(feat)
                if st is None:
                    continue
                sg = sig[cell][feat]
                rows.append(dict(mouse=mouse, session=day, neuron_idx=int(cell), feature=table_name,
                                 me=st.get("me"), me_excess=st.get("me_excess"), pval=st.get("pval"),
                                 opt_delay=st.get("opt_delay"), signal_ratio=st.get("signal_ratio"),
                                 stage1=bool(sg.get("stage1", False)), stage2=bool(sg.get("stage2", False)),
                                 significant=C.significant_only(st, sg), accepted=C.paper_accepts(st, sg)))
    return pd.DataFrame(rows), len(files)


def table(pairs, glm):
    piv = (pairs[pairs.accepted].assign(x=1)
           .pivot_table(index=KEY, columns="feature", values="x", aggfunc="max", fill_value=0))
    for c in FEATS:
        if c not in piv:
            piv[c] = 0
    m = glm.merge(piv.reset_index(), on=KEY, how="left").fillna({c: 0 for c in FEATS})
    out = {}
    for c in FEATS:
        i = m[c].astype(int)
        g = m[f"glm_{c}"].astype(int)
        both = int((i & g).sum())
        out[c] = dict(glm=int(g.sum()), intense=int(i.sum()), both=both,
                      intense_over_glm=round(i.sum() / g.sum(), 1) if g.sum() else None,
                      share_glm_found_by_intense=round(both / g.sum(), 3) if g.sum() else None)
    unmatched = len(piv) - len(piv.reset_index().merge(glm[KEY], on=KEY))
    return out, int(unmatched), int(len(m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--glm", type=Path, default=V.get("glm_ca1_per_neuron"))
    ap.add_argument("--pairs-csv", type=Path, default=None)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    pairs, n_files = intense_pairs(a.results)
    glm = pd.read_csv(a.glm)
    glm["session"] = glm["session"].astype(str)
    res, unmatched, n_neurons = table(pairs, glm)
    acc = pairs[pairs.accepted]
    res = dict(label=a.label, results=str(a.results), glm=str(a.glm), sessions=n_files,
               neurons_in_glm_table=n_neurons, accepted_pairs_without_glm_row=unmatched,
               table4=res,
               accepted_pairs_by_feature=acc.groupby("feature").size().to_dict(),
               accepted_with_zero_delay_by_feature=acc[acc.opt_delay == 0].groupby("feature").size().to_dict(),
               significant_pairs_by_feature=pairs[pairs.significant].groupby("feature").size().to_dict())
    C.write_json(a.out, res)
    if a.pairs_csv:
        pairs.to_csv(a.pairs_csv, index=False)
    print(res)


if __name__ == "__main__":
    main()
