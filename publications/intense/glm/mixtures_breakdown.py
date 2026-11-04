"""Which true feature of a linear-mixture cell each method finds (Section S1.2).

Reads only the saved per-cell outcomes of glm/synth_compute.py (<[glm] synth_dir>/parts) and
tables/table3.csv of glm/synth_numbers.py; no method is run.

Table 3 part (parts/t3_t3f1_<seed>.csv, 20 seeds, 900 s, SNR 20, calcium noise 0.05): for every
linear-mixture cell, whether both, only the first, only the second or neither of its two true features
was detected, and whether a feature outside the true pair was added; per-feature detection rates in
mixtures and in pure cells of the same recordings. INTENSE is read at p < 0.001 per pair without the
removal of anti-selective pairs (INTENSE_p001, the reading of Table 3); the GLM variants are the
three of Table 3. The mean F1 per group is recomputed and compared with tables/table3.csv.

SNR-sweep part (parts/snr_snr<S>_<seed>.csv, Supplementary Fig. 2: 10 seeds x 7 SNR levels, calcium
noise 0.1): the generator sets baseline 0.1 and peak 0.1*S, so a pure cell is modulated by
0.1*(S - 1) and each feature of an equal-weight two-feature sum by 0.05*(S - 1). A mixture at SNR S
therefore gives each feature the modulation of a pure cell at SNR (S + 1) / 2; on the doubling grid
of the sweep the mixture at 2s is set against the pure cell at s (per-feature modulation 0.1*(s - 0.5)
against 0.1*(s - 1)). Both come from the same recordings, so the difference is bootstrapped over
seeds. INTENSE is read at p < 0.05 with Holm correction and the removal of anti-selective pairs
(INTENSE_holm_filter, the reading of Fig. 2).

Outputs in [glm] mixtures_dir, among them the sources of the numbers of Section S1.2:
t3_mixture_outcomes.csv (rate_first, rate_second: how often each method finds head direction and the
event in mixed cells), t3_pure_detection.csv (the same in pure cells), snr_pure_snr50.csv (the SNR at
which a method finds the feature of half of the pure cells, and its ratio to INTENSE) and
snr_effective.csv (read by glm/mixtures_summary.py).

usage: python glm/mixtures_breakdown.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402

S1 = P.get("glm", "synth_dir")
PARTS = S1 / "parts"
OUT = P.get("glm", "mixtures_dir")
OUT.mkdir(parents=True, exist_ok=True)

SEEDS20 = [42, 123, 7, 11, 19, 23, 31, 44, 57, 64, 73, 88, 91, 101, 113, 127, 139, 151, 163, 177]
SNRS = [2, 4, 8, 16, 32, 64, 128]

MIX = {  # group -> (first feature, second feature), in the order of the group name
    "mixed_hd_speed_lin": ("head_direction", "speed"),
    "mixed_speed_event_lin": ("speed", "event_0"),
    "mixed_hd_event_lin": ("head_direction", "event_0"),
}
PURE = {"head_direction": "hd_cells", "speed": "speed_cells", "event_0": "event_cells"}
SHORT = {"head_direction": "HD", "speed": "speed", "event_0": "event"}
T3_VARIANTS = ["INTENSE_p001", "V_alpha10", "V0_base", "V_max"]
SNR_VARIANTS = ["INTENSE_holm_filter", "V_alpha10", "V0_base", "V_max"]
N_BOOT = 10000


def det_set(s):
    return set() if (not isinstance(s, str) or s == "") else set(s.split("|"))


def f1(det, true):
    if not det and not true:
        return 1.0
    tp = len(det & true)
    fp = len(det - true)
    fn = len(true - det)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def load(pattern, seeds, extra=None):
    frames = []
    for s in seeds:
        if extra is None:
            frames.append(pd.read_csv(PARTS / pattern.format(seed=s)))
        else:
            for e in extra:
                frames.append(pd.read_csv(PARTS / pattern.format(seed=s, x=e)))
    d = pd.concat(frames, ignore_index=True)
    d["det"] = d["detected"].apply(det_set)
    return d


def outcomes(d, group, variant):
    a, b = MIX[group]
    sub = d[(d.group == group) & (d.variant == variant)]
    has_a = sub.det.apply(lambda x: a in x)
    has_b = sub.det.apply(lambda x: b in x)
    extra = sub.det.apply(lambda x: len(x - {a, b}) > 0)
    f1_cells = sub.det.apply(lambda x: f1(x, {a, b}))
    return {
        "group": group, "first": SHORT[a], "second": SHORT[b], "variant": variant, "n_cells": len(sub),
        "both": int((has_a & has_b).sum()), "only_first": int((has_a & ~has_b).sum()),
        "only_second": int((~has_a & has_b).sum()), "neither": int((~has_a & ~has_b).sum()),
        "with_extra_feature": int(extra.sum()),
        "rate_first": float(has_a.mean()), "rate_second": float(has_b.mean()),
        "mean_f1_cells": float(f1_cells.mean()),
        "mean_f1_seeds": float(sub.assign(f=f1_cells).groupby("seed").f.mean().mean()),
    }


def table3():
    d = load("t3_t3f1_{seed}.csv", SEEDS20)
    rows = [outcomes(d, g, v) for g in MIX for v in T3_VARIANTS]
    out = pd.DataFrame(rows)
    # pure cells of the same recordings: how often the method finds the cell's one feature
    pure = []
    for feat, g in PURE.items():
        for v in T3_VARIANTS:
            sub = d[(d.group == g) & (d.variant == v)]
            hit = sub.det.apply(lambda x: feat in x)
            extra = sub.det.apply(lambda x: len(x - {feat}) > 0)
            pure.append({"group": g, "feature": SHORT[feat], "variant": v, "n_cells": len(sub),
                         "detected": int(hit.sum()), "rate": float(hit.mean()),
                         "with_extra_feature": int(extra.sum())})
    pure = pd.DataFrame(pure)
    # agreement with the printed table
    ref = pd.read_csv(S1 / "tables" / "table3.csv").set_index("group")
    chk = []
    for _, r in out.iterrows():
        chk.append({"group": r.group, "variant": r.variant, "mean_f1_here": r.mean_f1_seeds,
                    "table3": float(ref.loc[r.group, r.variant]),
                    "abs_diff": abs(r.mean_f1_seeds - float(ref.loc[r.group, r.variant]))})
    # the GLM detections of the Fig. 1 files must be the same calls as in the Table 3 files
    f = load("f1_t3f1_{seed}.csv", SEEDS20)
    same = {}
    for v in ["V0_base", "V_alpha10", "V_max"]:
        a = d[d.variant == v].set_index(["seed", "neuron"]).detected.fillna("")
        b = f[f.variant == v].set_index(["seed", "neuron"]).detected.fillna("")
        same[v] = bool((a.sort_index() == b.sort_index()).all())
    return out, pure, pd.DataFrame(chk), same, d


def boot_diff(mix_hits, pure_hits, rng):
    """Seed-level bootstrap of rate(mixture) - rate(pure); inputs are per-seed (hits, n) arrays."""
    seeds = sorted(mix_hits)
    mh = np.array([mix_hits[s][0] for s in seeds], float)
    mn = np.array([mix_hits[s][1] for s in seeds], float)
    ph = np.array([pure_hits[s][0] for s in seeds], float)
    pn = np.array([pure_hits[s][1] for s in seeds], float)
    idx = rng.integers(0, len(seeds), size=(N_BOOT, len(seeds)))
    diff = mh[idx].sum(1) / mn[idx].sum(1) - ph[idx].sum(1) / pn[idx].sum(1)
    return float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))


def per_seed_hits(sub, feat):
    g = sub.assign(h=sub.det.apply(lambda x: feat in x)).groupby("seed").h
    return {int(s): (int(v.sum()), int(v.size)) for s, v in g}


def snr_sweep():
    d = load("snr_snr{x}_{seed}.csv", SEEDS20[:10], extra=SNRS)
    rng = np.random.default_rng(0)
    rows = []
    for v in SNR_VARIANTS:
        for group, (a, b) in MIX.items():
            if group not in set(d.group):
                continue
            for S in SNRS[1:]:
                s = S // 2
                mix = d[(d.group == group) & (d.variant == v) & (d.snr == S)]
                for feat in (a, b):
                    pure = d[(d.group == PURE[feat]) & (d.variant == v) & (d.snr == s)]
                    pure_same = d[(d.group == PURE[feat]) & (d.variant == v) & (d.snr == S)]
                    mh, ph = per_seed_hits(mix, feat), per_seed_hits(pure, feat)
                    lo, hi = boot_diff(mh, ph, rng)
                    k_m = sum(x[0] for x in mh.values())
                    n_m = sum(x[1] for x in mh.values())
                    k_p = sum(x[0] for x in ph.values())
                    n_p = sum(x[1] for x in ph.values())
                    k_ps = int(pure_same.det.apply(lambda x: feat in x).sum())
                    rows.append({"variant": v, "group": group, "feature": SHORT[feat], "snr_mixture": S,
                                 "snr_pure_matched": s, "mix_detected": k_m, "mix_n": n_m,
                                 "mix_rate": k_m / n_m, "pure_detected": k_p, "pure_n": n_p,
                                 "pure_rate": k_p / n_p, "diff": k_m / n_m - k_p / n_p,
                                 "diff_lo": lo, "diff_hi": hi,
                                 "pure_rate_same_snr": k_ps / len(pure_same)})
    # predicted mixture F1 if each feature were found independently at the matched pure-cell rate
    pred = []
    for v in SNR_VARIANTS:
        for group, (a, b) in MIX.items():
            if group not in set(d.group):
                continue
            for S in SNRS[1:]:
                s = S // 2
                pa = d[(d.group == PURE[a]) & (d.variant == v) & (d.snr == s)].det.apply(lambda x: a in x).mean()
                pb = d[(d.group == PURE[b]) & (d.variant == v) & (d.snr == s)].det.apply(lambda x: b in x).mean()
                mix = d[(d.group == group) & (d.variant == v) & (d.snr == S)]
                obs = mix.det.apply(lambda x: f1(x, {a, b})).mean()
                pred_f1 = pa * pb * 1.0 + (pa * (1 - pb) + pb * (1 - pa)) * (2.0 / 3.0)
                pred.append({"variant": v, "group": group, "snr_mixture": S, "snr_pure_matched": s,
                             "p_first_pure": pa, "p_second_pure": pb, "f1_predicted": pred_f1,
                             "f1_observed": obs})
    return pd.DataFrame(rows), pd.DataFrame(pred), d


def pure_curves(d):
    """Share of pure cells whose one feature is detected, per SNR; the SNR where it crosses one half."""
    rows, s50 = [], []
    for v in SNR_VARIANTS:
        for feat, g in PURE.items():
            rates = []
            for S in SNRS:
                sub = d[(d.group == g) & (d.variant == v) & (d.snr == S)]
                k = int(sub.det.apply(lambda x: feat in x).sum())
                rows.append({"variant": v, "feature": SHORT[feat], "snr": S, "detected": k, "n": len(sub),
                             "rate": k / len(sub)})
                rates.append(k / len(sub))
            env = np.maximum.accumulate(np.array(rates))
            x = np.log2(SNRS)
            snr50 = float(2 ** np.interp(0.5, env, x)) if env[0] < 0.5 < env[-1] else float("nan")
            s50.append({"variant": v, "feature": SHORT[feat], "snr50": snr50})
    s50 = pd.DataFrame(s50)
    piv = s50.pivot(index="feature", columns="variant", values="snr50")
    for v in ["V_alpha10", "V0_base", "V_max"]:
        piv[f"ratio_{v}_to_INTENSE"] = piv[v] / piv["INTENSE_holm_filter"]
    return pd.DataFrame(rows), piv.reset_index()


def effective_snr(d, curves):
    """The SNR at which a pure cell's feature is detected as often as the same feature of a mixed cell."""
    rows = []
    x = np.log2(SNRS)
    for v in SNR_VARIANTS:
        for group, (a, b) in MIX.items():
            for feat in (a, b):
                c = curves[(curves.variant == v) & (curves.feature == SHORT[feat])].sort_values("snr")
                env = np.maximum.accumulate(c.rate.to_numpy())
                for S in SNRS:
                    sub = d[(d.group == group) & (d.variant == v) & (d.snr == S)]
                    m = float(sub.det.apply(lambda z: feat in z).mean())
                    ok = env[0] < m < env[-1] and 0.0 < m < 1.0
                    s_eff = float(2 ** np.interp(m, env, x)) if ok else float("nan")
                    rows.append({"variant": v, "group": group, "feature": SHORT[feat], "snr_mixture": S,
                                 "mix_rate": m, "snr_pure_equivalent": s_eff,
                                 "snr_ratio": S / s_eff if ok else float("nan")})
    return pd.DataFrame(rows)


def sweep_at_20(d):
    """GLM per-feature rates of the sweep interpolated (log SNR) to SNR 20, beside Table 3 at SNR 20.

    The sweep differs from Table 3 in calcium noise (0.1 against 0.05); INTENSE is left out because
    the two use different thresholds (Holm p < 0.05 against p < 0.001 per pair).
    """
    t3 = load("t3_t3f1_{seed}.csv", SEEDS20)
    w = (np.log2(20) - 4.0) / 1.0
    rows = []
    for v in ["V_alpha10", "V0_base", "V_max"]:
        for group, feats in list(MIX.items()) + [(g, (f,)) for f, g in PURE.items()]:
            for feat in feats:
                r = {}
                for S in (16, 32):
                    sub = d[(d.group == group) & (d.variant == v) & (d.snr == S)]
                    r[S] = float(sub.det.apply(lambda z: feat in z).mean()) if len(sub) else float("nan")
                sub3 = t3[(t3.group == group) & (t3.variant == v)]
                rows.append({"variant": v, "group": group, "feature": SHORT[feat], "sweep_16": r[16],
                             "sweep_32": r[32], "sweep_interp_20": r[16] + w * (r[32] - r[16]),
                             "table3_snr20": float(sub3.det.apply(lambda z: feat in z).mean())})
    return pd.DataFrame(rows)


def conditional_t3(d):
    """GLM and INTENSE: is the second feature found more often when the first one was found?"""
    from scipy.stats import fisher_exact
    rows = []
    for v in ["INTENSE_p001", "V_alpha10"]:
        for group, (a, b) in MIX.items():
            sub = d[(d.group == group) & (d.variant == v)]
            ha = sub.det.apply(lambda z: a in z)
            hb = sub.det.apply(lambda z: b in z)
            # the weaker of the two in pure cells is conditioned on the stronger one
            weak, strong = (a, b) if a == "speed" else (b, a)
            hw = sub.det.apply(lambda z: weak in z)
            hs = sub.det.apply(lambda z: strong in z)
            tab = [[int((hw & hs).sum()), int((~hw & hs).sum())], [int((hw & ~hs).sum()), int((~hw & ~hs).sum())]]
            p = float(fisher_exact(tab)[1])
            rows.append({"variant": v, "group": group, "weak": SHORT[weak], "strong": SHORT[strong],
                         "weak_found_given_strong_found": f"{tab[0][0]}/{tab[0][0] + tab[0][1]}",
                         "weak_found_given_strong_missed": f"{tab[1][0]}/{tab[1][0] + tab[1][1]}",
                         "fisher_p": p, "both": int((ha & hb).sum())})
    return pd.DataFrame(rows)


def main():
    out, pure, chk, same, t3 = table3()
    out.to_csv(OUT / "t3_mixture_outcomes.csv", index=False)
    pure.to_csv(OUT / "t3_pure_detection.csv", index=False)
    chk.to_csv(OUT / "t3_f1_check.csv", index=False)
    snr, pred, dsnr = snr_sweep()
    snr.to_csv(OUT / "snr_matched.csv", index=False)
    pred.to_csv(OUT / "snr_predicted_f1.csv", index=False)
    curves, s50 = pure_curves(dsnr)
    curves.to_csv(OUT / "snr_pure_curves.csv", index=False)
    s50.to_csv(OUT / "snr_pure_snr50.csv", index=False)
    eff = effective_snr(dsnr, curves)
    eff.to_csv(OUT / "snr_effective.csv", index=False)
    at20 = sweep_at_20(dsnr)
    at20.to_csv(OUT / "sweep_vs_table3_at_snr20.csv", index=False)
    cond = conditional_t3(t3)
    cond.to_csv(OUT / "t3_conditional.csv", index=False)
    summary = {"max_abs_diff_vs_table3": float(chk.abs_diff.max()),
               "glm_calls_same_in_t3_and_f1_files": same}
    (OUT / "mixtures_breakdown_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(out.drop(columns=["mean_f1_cells"]).round(3).to_string(index=False))
    print()
    print(pure.round(3).to_string(index=False))
    print()
    print(summary)
    print()
    cols = ["variant", "group", "feature", "snr_mixture", "snr_pure_matched", "mix_detected", "mix_n",
            "pure_detected", "pure_n", "diff", "diff_lo", "diff_hi", "pure_rate_same_snr"]
    print(snr[cols].round(2).to_string(index=False))
    print()
    print(pred.round(2).to_string(index=False))
    print()
    print(curves.pivot_table(index=["variant", "feature"], columns="snr", values="rate").round(2).to_string())
    print()
    print(s50.round(2).to_string(index=False))
    print()
    e = eff[eff.variant.isin(["INTENSE_holm_filter", "V_alpha10"])].dropna()
    print(e.round(2).to_string(index=False))
    print()
    print(at20.round(2).to_string(index=False))
    print()
    print(cond.to_string(index=False))


if __name__ == "__main__":
    main()
