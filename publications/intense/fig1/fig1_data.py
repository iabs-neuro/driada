"""Neuron-feature pairs of Figure 1E: Stage-2 p-value, excess MI and acceptance flags.

Reads the INTENSE results of the 64 NOF sessions (before disentanglement) and
writes <cache>/fig1e_real_pairs.npz and <cache>/fig1_numbers.json.

usage: python fig1/fig1_data.py
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

PVAL_THR, MI_THR = P.PVAL_THR, P.MI_EXCESS_THR


def main():
    rows = []
    n_pairs_total = n_stage1_fail = 0
    files = sorted(glob.glob(str(P.INTENSE_RESULTS / "NOF_*_results.npz")))
    assert len(files) == 64, f"{len(files)} result files in {P.INTENSE_RESULTS}"
    for f in files:
        z = np.load(f, allow_pickle=True)
        stats = json.loads(str(z["_stats_json"][0]))
        sig = json.loads(str(z["_significance_json"][0]))
        sess = os.path.basename(f).replace("_results.npz", "")
        P.check_representation(next(iter(stats.values())).keys())
        for n, feats in stats.items():
            for feat, st in feats.items():
                n_pairs_total += 1
                sg = sig.get(n, {}).get(feat, {})
                if not sg.get("stage1"):
                    n_stage1_fail += 1
                    continue
                sr = st.get("signal_ratio")
                rows.append((sess, int(n), feat, st["pval"], st["me"], st["me_excess"],
                             np.nan if sr is None else sr, bool(sg.get("stage2"))))

    sess, neur, feat, pval, me, mex, sr, st2 = map(np.array, zip(*rows))
    pval = pval.astype(float); mex = mex.astype(float); sr = sr.astype(float)
    anti_ok = np.isnan(sr) | (sr > 1)
    accepted = st2 & (pval < PVAL_THR) & (mex > MI_THR) & anti_ok
    anti = (pval < PVAL_THR) & ~anti_ok
    np.savez_compressed(P.cache_path("fig1e_real_pairs.npz"), session=sess, neuron=neur,
                        feature=feat, pval=pval, me=me.astype(float), me_excess=mex,
                        signal_ratio=sr, stage2=st2, accepted=accepted, anti_selective=anti)
    nums = {
        "source": str(P.INTENSE_RESULTS),
        "representation": P.REPRESENTATION,
        "sessions": int(len(set(sess))),
        "pairs_total": int(n_pairs_total),
        "pairs_failed_stage1_not_plotted": int(n_stage1_fail),
        "pairs_tested_in_stage2_plotted": int(len(pval)),
        "stage2_pass_flag": int(st2.sum()),
        "pval_below_0.001": int((pval < PVAL_THR).sum()),
        "accepted_green": int(accepted.sum()),
        "anti_selective_p_below_0.001_grey": int(anti.sum()),
        "rejected_blue": int((~accepted & ~anti).sum()),
        "x_clipped_at_20": int((-np.log10(pval) > 20).sum()),
        "accepted_neuron_sessions": int(len(set(zip(sess[accepted], neur[accepted])))),
        "definitions": {
            "plotted": "pairs that passed stage 1 and therefore have a stage-2 p-value",
            "accepted": "stage-2 significant (p<0.001 and rank criterion) and me_excess>0.04 bits "
                        "and signal_ratio NaN or >1 (anti-selectivity filter)",
            "anti_selective": "stage-2 p<0.001 but signal_ratio<=1 (removed by the anti-selectivity filter)",
            "x": "-log10 stage-2 p", "y": "excess MI above chance (bits)",
        },
    }
    with open(P.cache_path("fig1_numbers.json"), "w") as fh:
        json.dump(nums, fh, indent=1)
    print(json.dumps(nums, indent=1))


if __name__ == "__main__":
    main()
