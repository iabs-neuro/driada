"""Summaries of the mixture analysis of Section S1.2 (no method is run).

From the outputs of glm/mixtures_breakdown.py and glm/mixtures_components.py:

1. Speed detection in pure and mixed Table 3 cells binned by the speed component's own modulation
   (component_strength_cells.csv), with Wilson 95% intervals: the one feature type whose pure and
   mixed cells overlap in modulation.
2. Effective-SNR ratios of snr_effective.csv restricted to mixture detection rates between 0.1 and
   0.9, where the inverse interpolation of the pure-cell curve is not at its floor or ceiling
   (snr_effective_ratio_range.csv: a mixed cell's head direction or event is found as often as that
   of a pure cell at this many times lower SNR).
3. Mixture at SNR 2s against pure cell at s (snr_matched_counts.csv), Supplementary Fig. 2 read at
   the SNR of Table 3 (snr20_halving_share.csv), mixed cells with both features found at SNR 128
   (snr128_both_found.csv, snr_mixed_extra_feature.csv) and which component of a Table 3 mixture is
   the stronger (component_which_stronger.csv).

Reads and writes [glm] mixtures_dir; the SNR sweep is read from <[glm] synth_dir>/parts.

usage: python glm/mixtures_summary.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.dont_write_bytecode = True
PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402

OUT = P.get("glm", "mixtures_dir")


def wilson(k, n, a=0.05):
    if n == 0:
        return np.nan, np.nan
    z = norm.ppf(1 - a / 2)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main():
    c = pd.read_csv(OUT / "component_strength_cells.csv")
    c = c[c.feature == "speed"].copy()
    c["kind"] = np.where(c.weight < 1, "mixture", "pure")
    c["bin"] = pd.cut(c.mod_sd, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7])
    rows = []
    for (kind, b), g in c.groupby(["kind", "bin"], observed=True):
        for col, name in (("found_GLM", "GLM_alpha10"), ("found_INTENSE", "INTENSE_p001")):
            k, n = int(g[col].sum()), len(g)
            lo, hi = wilson(k, n)
            rows.append({"kind": kind, "speed_mod_sd_bin": str(b), "method": name, "found": k, "n": n,
                         "rate": k / n, "ci_lo": lo, "ci_hi": hi})
    sb = pd.DataFrame(rows)
    sb.to_csv(OUT / "speed_by_own_modulation_ci.csv", index=False)

    e = pd.read_csv(OUT / "snr_effective.csv")
    e = e[(e.mix_rate >= 0.1) & (e.mix_rate <= 0.9) & e.snr_ratio.notna()]
    e = e[e.variant.isin(["INTENSE_holm_filter", "V_alpha10"])]
    rng = (e.assign(kind=np.where(e.feature == "speed", "speed", "HD_or_event"))
           .groupby(["variant", "kind"]).snr_ratio.agg(["min", "max", "count"]).reset_index())
    rng.to_csv(OUT / "snr_effective_ratio_range.csv", index=False)

    # mixture at 2s against pure at s: points where at least one of the two rates is off 0 and 1
    m = pd.read_csv(OUT / "snr_matched.csv")
    m = m[m.variant.isin(["INTENSE_holm_filter", "V_alpha10"])]
    m = m[((m.mix_rate > 0) & (m.mix_rate < 1)) | ((m.pure_rate > 0) & (m.pure_rate < 1))]
    cnt = (m.groupby("variant")
           .apply(lambda g: pd.Series({"points": len(g), "mixture_lower": int((g["diff"] < 0).sum()),
                                       "mixture_higher": int((g["diff"] > 0).sum()),
                                       "ci_below_zero": int((g.diff_hi < 0).sum()),
                                       "ci_above_zero": int((g.diff_lo > 0).sum()),
                                       "largest_drop": float(g["diff"].min())}), include_groups=False)
           .reset_index())
    tot = cnt.drop(columns=["variant", "largest_drop"]).sum().to_dict()
    tot.update({"variant": "both", "largest_drop": float(m["diff"].min())})
    cnt = pd.concat([cnt, pd.DataFrame([tot])], ignore_index=True)
    cnt.to_csv(OUT / "snr_matched_counts.csv", index=False)

    # Fig. 2 read at the SNR of Table 3: pure cell at 20, pure cell at (20 + 1) / 2 = 10.5 (the halved
    # modulation alone), mixed cell at 20; log2 interpolation between the sweep levels
    cur = pd.read_csv(OUT / "snr_pure_curves.csv")
    det = load_sweep()
    x = np.log2([2, 4, 8, 16, 32, 64, 128])
    half = []
    for v in ["INTENSE_holm_filter", "V_alpha10"]:
        for grp, feats in (("mixed_hd_speed_lin", ("HD", "speed")), ("mixed_speed_event_lin", ("speed", "event"))):
            for f in feats:
                c = cur[(cur.variant == v) & (cur.feature == f)].sort_values("snr").rate.to_numpy()
                mix = [float(det[(det.variant == v) & (det.group == grp) & (det.snr == s)]
                             .det.apply(lambda z: FULL[f] in z).mean()) for s in [2, 4, 8, 16, 32, 64, 128]]
                p20, p105 = np.interp(np.log2(20), x, c), np.interp(np.log2(10.5), x, c)
                m20 = np.interp(np.log2(20), x, mix)
                half.append({"variant": v, "group": grp, "feature": f, "pure_at_20": p20,
                             "pure_at_10.5_halving_only": p105, "mixed_at_20": m20,
                             "share_of_loss_from_halving": (p20 - p105) / (p20 - m20) if p20 > m20 else np.nan})
    half = pd.DataFrame(half)
    half.to_csv(OUT / "snr20_halving_share.csv", index=False)

    # Fig. 2 at SNR 128: how many mixed cells have both features found
    both = []
    for v in ["INTENSE_holm_filter", "V_alpha10", "V0_base", "V_max"]:
        for grp, (a, b) in (("mixed_hd_speed_lin", ("head_direction", "speed")),
                            ("mixed_speed_event_lin", ("speed", "event_0"))):
            sub = det[(det.variant == v) & (det.group == grp) & (det.snr == 128)]
            k = int(sub.det.apply(lambda z: a in z and b in z).sum())
            extra = int(sub.det.apply(lambda z: len(z - {a, b}) > 0).sum())
            both.append({"variant": v, "group": grp, "snr": 128, "both_found": k, "n": len(sub),
                         "with_extra_feature": extra})
    both = pd.DataFrame(both)
    both.to_csv(OUT / "snr128_both_found.csv", index=False)
    ex = det[det.group.isin(["mixed_hd_speed_lin", "mixed_speed_event_lin"])].copy()
    true = {"mixed_hd_speed_lin": {"head_direction", "speed"}, "mixed_speed_event_lin": {"speed", "event_0"}}
    ex["extra"] = [len(d - true[g]) > 0 for d, g in zip(ex.det, ex.group)]
    extra_all = ex.groupby("variant").extra.agg(["sum", "size"]).reset_index()
    extra_all.to_csv(OUT / "snr_mixed_extra_feature.csv", index=False)

    # which component is stronger, cell by cell, in the mixtures of Table 3
    cc = pd.read_csv(OUT / "component_strength_cells.csv")
    order = []
    for grp, (a, b) in (("mixed_hd_speed_lin", ("HD", "speed")), ("mixed_speed_event_lin", ("speed", "event")),
                        ("mixed_hd_event_lin", ("HD", "event"))):
        piv = cc[cc.group == grp].pivot_table(index=["seed", "neuron"], columns="feature", values="mod_sd")
        order.append({"group": grp, "first": a, "second": b, "cells": len(piv),
                      "first_stronger": int((piv[a] > piv[b]).sum())})
    order = pd.DataFrame(order)
    order.to_csv(OUT / "component_which_stronger.csv", index=False)
    pd.set_option("display.width", 200)
    print(sb.round(2).to_string(index=False))
    print()
    print(e.round(2).to_string(index=False))
    print()
    print(rng.round(2).to_string(index=False))
    print()
    print(cnt.round(2).to_string(index=False))
    print()
    print(half.round(3).to_string(index=False))
    print()
    print(both.to_string(index=False))
    print()
    print(extra_all.to_string(index=False))
    print()
    print(order.to_string(index=False))


FULL = {"HD": "head_direction", "speed": "speed", "event": "event_0"}


def load_sweep():
    parts = P.get("glm", "synth_dir") / "parts"
    seeds = [42, 123, 7, 11, 19, 23, 31, 44, 57, 64]
    d = pd.concat([pd.read_csv(parts / f"snr_snr{s}_{seed}.csv") for s in [2, 4, 8, 16, 32, 64, 128]
                   for seed in seeds], ignore_index=True)
    d["det"] = d.detected.apply(lambda s: set() if (not isinstance(s, str) or s == "") else set(s.split("|")))
    return d


if __name__ == "__main__":
    main()
