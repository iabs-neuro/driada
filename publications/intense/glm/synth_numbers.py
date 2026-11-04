"""Numbers and tables of the synthetic benchmark of Supplementary Section S1 (parts of glm/synth_compute.py).

F1 per cell type: mean over seeds of the per-seed mean F1 of its cells, with the 95% bootstrap
interval over seeds (10,000 resamples, generator seed 0). Paired contrast of Supplementary Table 3:
INTENSE minus the strongest of its three GLM variants for that cell type (chosen by mean F1, which
favours the GLM), with its bootstrap interval and the two-sided Wilcoxon signed-rank test over
seeds (zero_method 'wilcox').

Computes what the parts written so far allow and states the number of seeds of each part.
Reads <[glm] synth_dir>/parts and done/; writes <[glm] synth_dir>/tables:
  table3.csv          Supplementary Table 3 with intervals and exact p (INTENSE_p001: INTENSE at
                      p < 0.001 per pair; best_glm: the strongest of the three GLM variants)
  table3_si.csv       Supplementary Table 3 as printed; table3_rows.tex: its LaTeX rows
  fig1_ci.csv         Supplementary Fig. 1: eight GLM variants and INTENSE per cell type
  fig2_ci.csv         Supplementary Fig. 2: three GLM variants and INTENSE per SNR level
  t2_controls.csv     Section S1.3: GLM with interaction blocks and with four bases for speed
  t31.csv             three-feature mixtures (INTENSE with type-based and untransformed features)
  spread_repeat.csv   seeds generated twice: how much the generator alone moves F1
  timing.csv          seconds per unit of work, from done/
  numbers.json        the numbers of the text under the keys table3, fig1, t2, fig2, t31,
                      excess_true_pairs, spread, timing

usage: python glm/synth_numbers.py
"""
import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys  # noqa: E402

sys.dont_write_bytecode = True
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402

OUTD = P.get("glm", "synth_dir")
PARTS = OUTD / "parts"
TABLES = OUTD / "tables"


def boot_ci(vals, n_boot=10000, alpha=0.05):
    """Mean of ``vals`` and its ``1 - alpha`` bootstrap interval (``n_boot`` resamples, generator seed 0)."""
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(0)
    means = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(axis=1)
    return float(vals.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


T3_ORDER = ["hd_cells", "event_cells", "speed_cells", "mixed_hd_speed_lin", "mixed_speed_event_lin",
            "mixed_hd_event_lin", "mixed_hd_speed_and", "mixed_speed_event_and", "nonselective"]
T3_LABEL = {"hd_cells": "Pure head direction", "event_cells": "Pure event", "speed_cells": "Pure speed",
            "mixed_hd_speed_lin": "Linear mixed (HD+speed)", "mixed_speed_event_lin": "Linear mixed (speed+event)",
            "mixed_hd_event_lin": "Linear mixed (HD+event)", "mixed_hd_speed_and": "AND mixed (HD+speed)",
            "mixed_speed_event_and": "AND mixed (speed+event)", "nonselective": "Nonselective"}
TABLE_GLM = ["V0_base", "V_max", "V_alpha10"]
GLM8 = ["V0_base", "V_ttest", "V_fdr", "V_alpha10", "V_folds20", "V_poisson", "V_max", "V_max_perfeat"]
SNR_GROUPS = ["hd_cells", "event_cells", "speed_cells", "mixed_hd_speed_lin", "mixed_speed_event_lin", "nonselective"]
T31_ORDER = ["hd_cells", "speed_cells", "event_cells", "mixed2_lin", "mixed3_lin", "mixed3_and", "nonselective"]


def load(pattern):
    files = sorted(PARTS.glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f, keep_default_na=False, na_values=[""]) for f in files], ignore_index=True)


def per_seed(d, keys=("group", "variant", "seed")):
    return d.groupby(list(keys)).f1.mean().reset_index()


def p_label(p):
    if p >= 0.05:
        return "n.s."
    if p >= 0.01:
        return f"$p{{=}}{p:.2g}$"
    for k in range(3, 8):
        if p >= 10.0 ** -k:
            return f"$p{{<}}10^{{-{k - 1}}}$"
    return "$p{<}10^{-7}$"


def ci_rows(ps, groups, variants, extra=None):
    rows = []
    for g in groups:
        for v in variants:
            vals = ps[(ps.group == g) & (ps.variant == v)].f1.values
            if len(vals):
                m, lo, hi = boot_ci(vals)
                r = {"group": g, "variant": v, "mean_f1": m, "ci_lo": lo, "ci_hi": hi, "n_seeds": len(vals)}
                if extra:
                    r.update(extra)
                rows.append(r)
    return rows


def contrast(wide, a, b):
    x, y = wide[a].values, wide[b].values
    d = x - y
    m, lo, hi = boot_ci(d)
    try:
        p = float(wilcoxon(x, y, zero_method="wilcox").pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    return m, lo, hi, p


def table3(res):
    d = load("t3_t3f1_*.csv")
    if d.empty:
        return
    ps = per_seed(d)
    wide = ps.pivot_table(index=["group", "seed"], columns="variant", values="f1").reset_index()
    rows = []
    for g in T3_ORDER:
        sub = wide[wide.group == g]
        if not len(sub):
            continue
        r = {"group": g, "label": T3_LABEL[g], "n_seeds": int(len(sub))}
        for v in TABLE_GLM + ["INTENSE_p001", "INTENSE_p001_filter"]:
            m, lo, hi = boot_ci(sub[v].values)
            r[v], r[f"{v}_lo"], r[f"{v}_hi"] = m, lo, hi
        best = max(TABLE_GLM, key=lambda v: r[v])
        m, lo, hi, p = contrast(sub, "INTENSE_p001", best)
        r.update(best_glm=best, dF1=m, dF1_lo=lo, dF1_hi=hi, wilcoxon_p=p, p_label=p_label(p),
                 ratio=r["INTENSE_p001"] / r[best] if r[best] else np.nan)
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "table3.csv", index=False)
    # Supplementary Table 3 as printed: two decimals, the strongest of the three GLM variants in bold
    si, tex = [], []
    for _, r in t[t.group != "nonselective"].iterrows():
        cells = {v: f"{r[v]:.2f}" for v in TABLE_GLM + ["INTENSE_p001"]}
        diff = f"{r.dF1:+.2f} [{r.dF1_lo:.2f}, {r.dF1_hi:.2f}]"
        si.append({"Cell type": r.label, "Baseline": cells["V0_base"], "Combined": cells["V_max"],
                   "alpha=0.10": cells["V_alpha10"], "INTENSE": cells["INTENSE_p001"],
                   "strongest GLM": {"V0_base": "Baseline", "V_max": "Combined", "V_alpha10": "alpha=0.10"}[r.best_glm],
                   "dF1 (95% CI)": diff, "Wilcoxon p": f"{r.wilcoxon_p:.3g}",
                   "p as printed": r.p_label.replace("$", "").replace("{", "").replace("}", "")})
        b = {v: (f"\\textbf{{{cells[v]}}}" if v == r.best_glm else cells[v]) for v in TABLE_GLM}
        tex.append(f"{r.label:<27}& {b['V0_base']} & {b['V_max']} & {b['V_alpha10']} & {cells['INTENSE_p001']} & "
                   f"$+{r.dF1:.2f}\\;[{r.dF1_lo:.2f},{r.dF1_hi:.2f}]$, {r.p_label} \\\\")
    pd.DataFrame(si).to_csv(TABLES / "table3_si.csv", index=False)
    (TABLES / "table3_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    non = d[d.group == "nonselective"].assign(fp=lambda x: x.detected.fillna("").astype(str).str.len() > 0)
    res["table3"] = {"seeds": sorted(int(s) for s in d.seed.unique()),
                     "rows": t.round(4).to_dict(orient="records"),
                     "nonselective_cells": int(non.groupby("variant").size().iloc[0]),
                     "nonselective_cells_with_a_detection": non.groupby("variant").fp.sum().astype(int).to_dict(),
                     "filter_changes_p001": int((d[d.variant == "INTENSE_p001"].f1.values
                                                 != d[d.variant == "INTENSE_p001_filter"].f1.values).sum())}


def fig1(res):
    d = load("f1_t3f1_*.csv")
    if d.empty:
        return
    ps = per_seed(d)
    variants = GLM8 + ["INTENSE_holm_filter", "INTENSE_holm"]
    rows = ci_rows(ps, T3_ORDER, variants)
    ci = pd.DataFrame(rows)
    ci.to_csv(TABLES / "fig1_ci.csv", index=False)
    piv = ci.pivot(index="group", columns="variant", values="mean_f1")
    lin = ["mixed_hd_speed_lin", "mixed_speed_event_lin"]
    res["fig1"] = {
        "seeds": sorted(int(s) for s in d.seed.unique()),
        "mean_f1": piv.round(4).to_dict(orient="index"),
        "max_perfeat_vs_max": {g: [round(piv.loc[g, "V_max_perfeat"], 4), round(piv.loc[g, "V_max"], 4)]
                               for g in ["speed_cells"] + lin},
        "best_of_8_glm": {g: [piv.loc[g, GLM8].idxmax(), round(piv.loc[g, GLM8].max(), 4),
                              round(piv.loc[g, "INTENSE_holm_filter"], 4)] for g in piv.index},
        "filter_changes_holm": int((d[d.variant == "INTENSE_holm"].f1.values
                                    != d[d.variant == "INTENSE_holm_filter"].f1.values).sum()),
    }


def t2(res):
    d = load("t2_t3f1_*.csv")
    f = load("f1_t3f1_*.csv")
    if d.empty:
        return
    seeds = sorted(int(s) for s in d.seed.unique())
    inter = per_seed(d[d.method == "GLM_interaction"].assign(variant="GLM_interaction"))
    base = per_seed(f[(f.variant == "V0_base") & f.seed.isin(seeds) & (f.group != "mixed_hd_event_lin")])
    rows = ci_rows(pd.concat([inter, base]), [g for g in T3_ORDER if g != "mixed_hd_event_lin"],
                   ["GLM_interaction", "V0_base"])
    basis = d[d.method == "GLM_speed_basis"]
    for b in ["quantile10", "linear10", "quantile20", "spline"]:
        rows += ci_rows(per_seed(basis[basis.basis == b].assign(variant=f"basis_{b}")),
                        ["speed_cells", "mixed_hd_speed_lin", "mixed_speed_event_lin"], [f"basis_{b}"])
    ci = pd.DataFrame(rows)
    ci.to_csv(TABLES / "t2_controls.csv", index=False)
    piv = ci.pivot(index="group", columns="variant", values="mean_f1")
    res["t2"] = {"seeds": seeds, "mean_f1": piv.round(4).to_dict(orient="index")}


def fig2(res):
    d = load("snr_snr*.csv")
    if d.empty:
        return
    ps = d.groupby(["snr", "group", "variant", "seed"]).f1.mean().reset_index()
    rows = []
    for snr in sorted(ps.snr.unique()):
        rows += ci_rows(ps[ps.snr == snr], SNR_GROUPS, ["V0_base", "V_alpha10", "V_max", "INTENSE_holm_filter",
                                                        "INTENSE_holm"], extra={"snr": int(snr)})
    ci = pd.DataFrame(rows)
    ci.to_csv(TABLES / "fig2_ci.csv", index=False)
    out = {"n_recordings": int(d.groupby(["snr", "seed"]).ngroups), "groups": {}}
    for g in SNR_GROUPS[:-1]:
        sub = ci[ci.group == g].pivot(index="snr", columns="variant", values="mean_f1")
        glm_best = sub[["V0_base", "V_alpha10", "V_max"]].max(axis=1)
        intense = sub["INTENSE_holm_filter"]
        first = lambda s, thr: next((int(i) for i, v in s.items() if v >= thr), None)  # noqa: E731
        out["groups"][g] = {"intense": intense.round(3).to_dict(), "best_glm": glm_best.round(3).to_dict(),
                            "gap": (intense - glm_best).round(3).to_dict(),
                            "first_snr_f1_ge_0.5": [first(intense, 0.5), first(glm_best, 0.5)],
                            "first_snr_f1_ge_0.9": [first(intense, 0.9), first(glm_best, 0.9)]}
    out["filter_changes_holm"] = int((d[d.variant == "INTENSE_holm"].f1.values
                                      != d[d.variant == "INTENSE_holm_filter"].f1.values).sum())
    res["fig2"] = out


def t31(res):
    d = load("t31_t31_*.csv")
    if d.empty:
        return
    ps = per_seed(d)
    variants = ["V0_base", "V_alpha10", "V_max", "INTENSE_holm_filter", "INTENSE_holm",
                "INTENSE_raw_holm_filter", "INTENSE_raw_holm"]
    ci = pd.DataFrame(ci_rows(ps, T31_ORDER, variants))
    ci.to_csv(TABLES / "t31.csv", index=False)
    piv = ci.pivot(index="group", columns="variant", values="mean_f1")
    ratios = {}
    for arm in ["INTENSE_holm_filter", "INTENSE_raw_holm_filter"]:
        for g in ["mixed2_lin", "mixed3_lin", "mixed3_and"]:
            best = piv.loc[g, ["V0_base", "V_alpha10", "V_max"]]
            ratios[f"{arm}:{g}"] = {"intense": round(piv.loc[g, arm], 4), "best_glm": best.idxmax(),
                                    "best_glm_f1": round(best.max(), 4),
                                    "ratio": round(piv.loc[g, arm] / best.max(), 3) if best.max() else None}
    res["t31"] = {"seeds": sorted(int(s) for s in d.seed.unique()), "mean_f1": piv.round(4).to_dict(orient="index"),
                  "ratios": ratios}


def excess(res):
    d = load("pairs_t3f1_*.csv")
    if d.empty:
        return
    tp = d[(d.cfg == "p001") & (d["true"].astype(str) == "True")]
    det = tp[tp.stage2.astype(str) == "True"]
    me = det.me_excess.astype(float)
    reached = tp[tp.me_excess.notna()].me_excess.astype(float)
    res["excess_true_pairs"] = {
        "detected_true_pairs": int(len(det)), "true_pairs": int(len(tp)),
        "detected_min_max": [round(me.min(), 4), round(me.max(), 4)],
        "detected_p5_p95": [round(me.quantile(0.05), 4), round(me.quantile(0.95), 4)],
        "detected_share_above_0.04": round(float((me > 0.04).mean()), 4),
        "detected_above_0.04": int((me > 0.04).sum()),
        "stage2_true_pairs_min_max": [round(reached.min(), 4), round(reached.max(), 4)],
        "by_group_median": det.assign(me_excess=me).groupby("group").me_excess.median().round(4).to_dict(),
    }


def spread(res):
    rep = pd.concat([load("t3_rep_*.csv"), load("f1_rep_*.csv")], ignore_index=True)
    if rep.empty:
        return
    main = pd.concat([load("t3_t3f1_*.csv"), load("f1_t3f1_*.csv")], ignore_index=True)
    seeds = sorted(int(s) for s in rep.seed.unique())
    a = per_seed(main[main.seed.isin(seeds)]).rename(columns={"f1": "f1_main"})
    b = per_seed(rep).rename(columns={"f1": "f1_rep"})
    m = a.merge(b, on=["group", "variant", "seed"])
    m["diff"] = m.f1_rep - m.f1_main
    m.to_csv(TABLES / "spread_repeat.csv", index=False)
    key = ["INTENSE_p001", "INTENSE_holm_filter", "V0_base", "V_alpha10", "V_max"]
    mk = m[m.variant.isin(key)]
    res["spread"] = {"seeds": seeds,
                     "abs_diff_per_seed_group_mean": round(float(mk["diff"].abs().mean()), 4),
                     "abs_diff_per_seed_group_max": round(float(mk["diff"].abs().max()), 4),
                     "by_variant_mean_abs": mk.groupby("variant")["diff"].apply(lambda x: x.abs().mean()).round(4).to_dict(),
                     "share_unchanged": round(float((mk["diff"] == 0).mean()), 3),
                     "glm8_abs_diff_mean": round(float(m[m.variant.isin(GLM8)]["diff"].abs().mean()), 4)}


def timing(res):
    rows = []
    for f in sorted((OUTD / "done").glob("*.ok")):
        j = json.loads(f.read_text(encoding="utf-8"))
        rows.append({"unit": f.stem, "seconds": j["seconds"], **j["parts"]})
    if rows:
        t = pd.DataFrame(rows)
        t.to_csv(TABLES / "timing.csv", index=False)
        res["timing"] = {"units": len(t), "core_seconds": round(float(t.seconds.sum())),
                         "by_kind_mean_s": t.assign(kind=t.unit.str.extract(r"^([a-z0-9]+?)(?:\d*_)")[0])
                         .groupby("kind").seconds.mean().round(0).to_dict()}


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    res = {}
    for fn in (table3, fig1, t2, fig2, t31, excess, spread, timing):
        fn(res)
    (TABLES / "numbers.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8")
    print(json.dumps({k: (v.get("seeds", v.get("n_recordings")) if isinstance(v, dict) else v)
                      for k, v in res.items()}, default=str))


if __name__ == "__main__":
    main()
