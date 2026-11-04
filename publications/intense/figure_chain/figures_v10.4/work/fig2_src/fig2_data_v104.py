"""Data for Figure 2 C, E, F on NOF with the frame rate of each session.

* INTENSE side: [v11] stage2_results/*.npz (feature place_quad); cross-check against
  <[results] db_dir>/NOF_pre_disent.csv.
* Classic side, --classic A: the MATLAB file of [data] classic_mat, not deposited and not used by the paper;
  --classic B, the paper: z of the Python re-implementation on the true clock, event onset, shuffle seed 0
  (<[v11] classic_dir>/classic/classic_NOF_<session>_true_s0.npz);
  --classic Bold: the same re-implementation on the old clock, for the numbers only (no figure).
  For all, label z > 3 and p = 1 - Phi(z), as in v10.2.
* Map similarity (2E): caches of ../ssim_v104.py (synchronized files, events at the session rate).
* Outputs carry the variant in their names; a few counts the manuscript quotes are added
  (classic place cells at z > 4 and z > 3).

INTENSE side: feature place_quad, before disentanglement. Place cell = Stage-2 significant,
p < 0.001 and excess MI > 0.04 bits. Continuous INTENSE statistic for every cell: Stage-1 z,
(observed MI - mean of the 100 Stage-1 shuffles) / their s.d., from info.me_total1 (index 0 =
observed value). INTENSE p for every cell: Stage-2 p if the pair reached Stage 2, else the
Stage-1 p reconstructed from the same shuffles (gamma_zi fit, as IntenseResults.populate_stage1_pvals).

usage: python fig2_data_v104.py --classic A|B  ->  ../cache/fig2_v104_data_<X>.npz, ../cache/fig2_v104_cells_<X>.csv,
                                                  ../fig2_numbers_<X>.json
"""
import argparse
import json

import _paths  # noqa: F401
import numpy as np
import pandas as pd
import scipy.stats as st

from driada.intense import load_results
from driada.intense.stats import reconstruct_stage1_pvals
from data_loader import SESSION_LIST, load_classic_data

FEATURE = "place_quad"
P_THR, MI_THR, Z_THR = 1e-3, 0.04, 3.0
SEED = 42
HERE = _paths.HERE
WORK = HERE.parent
ART = _paths._F.D
RESULTS_DIR = _paths._F.V.get("stage2_results")
DB_NEW = _paths._F.P.DB_PRE
CLASSIC_B = _paths._F.V.get("classic_dir") / "classic"
SSIM_DIR = _paths._F.V.get("fig2_ssim_cache")
OUT_CACHE = _paths._F.out_dir(WORK) / "cache"


def load_intense_place(session):
    res = load_results(RESULTS_DIR / f"NOF_{session}_results.npz")
    feats = list(res.intense_params["feat_bunch"].values()) if isinstance(
        res.intense_params["feat_bunch"], dict) else list(res.intense_params["feat_bunch"])
    neurons = list(res.intense_params["neurons"].values()) if isinstance(
        res.intense_params["neurons"], dict) else list(res.intense_params["neurons"])
    j = feats.index(FEATURE)
    m1 = np.asarray(res.info["me_total1"][:, j:j + 1, :], dtype=np.float64)
    pre_p, _ = reconstruct_stage1_pvals(m1, metric_distr_type=res.intense_params.get(
        "metric_distr_type", "gamma_zi"))
    obs, sh = m1[:, 0, 0], m1[:, 0, 1:]
    z1 = (obs - sh.mean(axis=1)) / sh.std(axis=1)
    n = len(neurons)
    out = dict(z1=z1, p=pre_p[:, 0].copy(), stage2=np.zeros(n, bool), me_excess=np.full(n, np.nan),
               pc=np.zeros(n, bool))
    for i, nm in enumerate(neurons):
        s = res.stats.get(str(nm), {}).get(FEATURE, {})
        g = res.significance.get(str(nm), {}).get(FEATURE, {})
        if s.get("pval") is not None:
            out["p"][i] = s["pval"]
            out["me_excess"][i] = s.get("me_excess", np.nan)
        out["stage2"][i] = bool(g.get("stage2", False))
        out["pc"][i] = out["stage2"][i] and out["p"][i] < P_THR and out["me_excess"][i] > MI_THR
    assert neurons == list(range(n)), f"{session}: neuron names are not 0..n-1"
    return out


def load_classic(variant):
    """z of every cell of every session for classic variant A (MATLAB file) or B (re-implementation)."""
    if variant == "A":
        return {s: np.asarray(v["z_score"], float) for s, v in load_classic_data(zthr=Z_THR).items()}
    # Bold: the same re-implementation on the old clock (20 frames/s everywhere); a check only,
    # it separates the effect of the clock from that of the implementation
    clock = "old" if variant == "Bold" else "true"
    out = {}
    for s in SESSION_LIST:
        d = np.load(CLASSIC_B / f"classic_NOF_{s}_{clock}_s0.npz")
        assert str(d["clock"]) == clock and int(d["seed"]) == 0, s
        assert json.loads(str(d["settings"]))["event_time"] == "run_start", s
        out[s] = np.asarray(d["z"], float)
    return out


def topn_jaccard(a, b, fracs):
    ra, rb = np.argsort(-a), np.argsort(-b)
    out = []
    for f in fracs:
        k = max(1, int(round(f * len(a))))
        A, B = set(ra[:k]), set(rb[:k])
        out.append(len(A & B) / len(A | B))
    return np.array(out)


def baseline_jaccard(rho_s, n, fracs, rng, reps=20):
    r = 2 * np.sin(np.pi * rho_s / 6)  # Pearson r giving Spearman rho_s for a bivariate normal
    cov = [[1, r], [r, 1]]
    return np.mean([topn_jaccard(*rng.multivariate_normal([0, 0], cov, n).T, fracs)
                    for _ in range(reps)], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classic", choices=["A", "B", "Bold"], required=True)
    variant = ap.parse_args().classic
    classic = load_classic(variant)
    rows = []
    for s in SESSION_LIST:
        it = load_intense_place(s)
        zc = classic[s]
        n = len(it["z1"])
        assert len(zc) == n, f"{s}: classic {len(zc)} vs INTENSE {n}"
        ssim = np.load(SSIM_DIR / f"fig2e_{s}_metrics_true.npz")["ssim_vals"]
        assert len(ssim) == n, s
        rows.append(pd.DataFrame(dict(session=s, neuron=np.arange(n), z_int=it["z1"], p_int=it["p"],
                                      stage2=it["stage2"], me_excess=it["me_excess"], pc_int=it["pc"],
                                      z_cl=zc, mse=ssim[:, 0], psnr=ssim[:, 1], ssim=ssim[:, 2])))
    d = pd.concat(rows, ignore_index=True)
    d["p_cl"] = st.norm.sf(d.z_cl)
    d["pc_cl"] = d.z_cl > Z_THR
    OUT_CACHE.mkdir(exist_ok=True)
    d.to_csv(OUT_CACHE / f"fig2_v104_cells_{variant}.csv", index=False)
    N = len(d)

    # Alignment check: rank correlation of the two statistics at index shifts 0 and +-1
    def rho_shift(k):
        vals = []
        for _, g in d.groupby("session", sort=False):
            a, b = g.z_int.values, g.z_cl.values
            if k > 0:
                a, b = a[k:], b[:-k]
            elif k < 0:
                a, b = a[:k], b[-k:]
            vals.append(pd.DataFrame(dict(a=a, b=b)))
        x = pd.concat(vals)
        return float(st.spearmanr(x.a, x.b).statistic)
    align = {str(k): rho_shift(k) for k in (-1, 0, 1)}

    # Cross-check with the tidy pre-disentanglement database of the recompute
    db = pd.read_csv(DB_NEW)
    db_pc = db[(db.feature == "place") & (db.pval < P_THR) & (db.me_excess > MI_THR)]
    n_db_pc = int(db_pc.groupby(["mouse", "session", "neuron_idx"]).ngroups)

    # 2C: Venn on all cells
    a, b = d.pc_int.values, d.pc_cl.values
    venn = dict(intense_only=int((a & ~b).sum()), both=int((a & b).sum()), classic_only=int((~a & b).sum()),
                n_intense=int(a.sum()), n_classic=int(b.sum()), n_cells=N)
    venn["jaccard"] = venn["both"] / (venn["n_intense"] + venn["n_classic"] - venn["both"])
    venn["frac_intense_confirmed"] = venn["both"] / venn["n_intense"]
    venn["frac_classic_confirmed"] = venn["both"] / venn["n_classic"]

    rho = float(st.spearmanr(d.z_int, d.z_cl).statistic)
    s2 = d[d.stage2]
    rho_z1_vs_p2 = float(st.spearmanr(s2.z_int, -np.log10(s2.p_int)).statistic)

    # 2F left: cumulative conditional confirmation
    lp_int, lp_cl = -np.log10(np.clip(d.p_int, 1e-300, 1)), -np.log10(np.clip(d.p_cl, 1e-300, 1))
    grid = np.round(np.arange(0, 12.01, 0.25), 2)
    conf_int, n_int, conf_cl, n_cl = [], [], [], []
    for x in grid:
        m = lp_int >= x
        n_int.append(int(m.sum())); conf_int.append(float(d.pc_cl[m].mean()) if m.sum() else np.nan)
        m = lp_cl >= x
        n_cl.append(int(m.sum())); conf_cl.append(float(d.pc_int[m].mean()) if m.sum() else np.nan)

    def conf_at_int(x):
        m = lp_int >= x
        return dict(level=f"-log10 p_INTENSE >= {x}", n=int(m.sum()), frac_classic_z_gt3=float(d.pc_cl[m].mean()),
                    frac_classic_z_gt6=float((d.z_cl[m] > 6).mean()))

    def conf_at_z(zl):
        m = d.z_cl >= zl
        return dict(level=f"classic z >= {zl}", n=int(m.sum()), frac_intense_pc=float(d.pc_int[m].mean()))

    # 2F right: top-N Jaccard vs baselines
    fracs = np.unique(np.round(np.logspace(np.log10(0.005), np.log10(0.5), 25), 4))
    j_obs = topn_jaccard(d.z_int.values, d.z_cl.values, fracs)
    rng = np.random.default_rng(SEED)
    base = {f"{r:.2f}": baseline_jaccard(r, N, fracs, rng) for r in (rho, 0.5, 0.7, 0.9)}
    i10 = int(np.argmin(np.abs(fracs - 0.10)))

    # 2E: map similarity among cells passing p < threshold in both methods (definition kept from v7)
    prange, xvals_e = np.logspace(-1, -7, 50), np.linspace(1, 7, 50)
    av, err, ne = [], [], []
    sims = d[["mse", "psnr", "ssim"]].values
    for p in prange:
        g = (d.p_int.values < p) & (d.p_cl.values < p)
        s = sims[g]
        s = s[np.all(np.isfinite(s), axis=1)]
        ne.append(len(s))
        av.append(s.mean(0) if len(s) else np.full(3, np.nan))
        err.append(s.std(0) / np.sqrt(len(s)) if len(s) else np.full(3, np.nan))
    av, err = np.array(av), np.array(err)
    # same metrics on the place-cell labels of 2C (whole sets, no p sweep)
    lab = {}
    for name, m in [("both", a & b), ("intense_only", a & ~b), ("classic_only", ~a & b)]:
        s = sims[m]; s = s[np.all(np.isfinite(s), axis=1)]
        lab[name] = dict(n=len(s), mean_ssim=float(s[:, 2].mean()))

    out_npz = OUT_CACHE / f"fig2_v104_data_{variant}.npz"
    np.savez(out_npz, grid=grid, conf_int=conf_int, n_int=n_int, conf_cl=conf_cl,
             n_cl=n_cl, fracs=fracs, j_obs=j_obs, **{f"base_{k}": v for k, v in base.items()},
             base_keys=np.array(list(base)), rho=rho, xvals_e=xvals_e, av_mse=av[:, 0], err_mse=err[:, 0],
             av_psnr=av[:, 1], err_psnr=err[:, 1], av_ssim=av[:, 2], err_ssim=err[:, 2], n_e=ne,
             **{f"venn_{k}": v for k, v in venn.items()})

    src_cl = ("zscoreThr4_binsize4_center.mat (colleague's MATLAB, as in v10.2)" if variant == "A" else
              f"<[v11] classic_dir>/classic/classic_NOF_<session>_{'old' if variant == 'Bold' else 'true'}_s0.npz "
              f"(Python re-implementation, {'20 frames/s in every session' if variant == 'Bold' else 'true frame rate'}, "
              "event onset, shuffle seed 0)")
    numbers = {
        "_source": "[v11] stage2_results/*.npz (DRIADA 1.2.2, frame rate of the session, "
                   f"before disentanglement, feature place_quad) + {src_cl}; map similarity from "
                   "[v11] fig2_ssim_cache/fig2e_<session>_metrics_true.npz; script figures_v10.4/work/fig2_src/fig2_data_v104.py "
                   f"--classic {variant}",
        "definitions": {
            "intense_place_cell": "Stage-2 significant AND p < 0.001 AND me_excess > 0.04 bits (place_quad)",
            "classic_place_cell": "z > 3",
            "intense_continuous_stat": "Stage-1 z = (observed MI - mean of 100 Stage-1 circular-shift MI) / s.d.; "
                                       "observed value = me_total1[..., 0]",
            "intense_p_all_cells": "Stage-2 p where the pair reached Stage 2, else Stage-1 p reconstructed from the "
                                   "100 shuffles (reconstruct_stage1_pvals, gamma_zi)",
            "classic_p": "one-sided normal p = 1 - Phi(z)",
        },
        "alignment_check": {
            "n_cells_total": N,
            "neurons_per_session_match_classic_and_ssim_caches": True,
            "spearman_zint_zcl_by_index_shift": align,
            "pc_count_from_npz": venn["n_intense"],
            "pc_count_from_tidy_pre_disent_db": n_db_pc,
        },
        "classic_counts": {"z_gt_4": int((d.z_cl > 4).sum()), "z_gt_3": int((d.z_cl > 3).sum()),
                           "z_eq_0": int((d.z_cl == 0).sum())},
        "fig2C_venn_all_cells": venn,
        "spearman_intense_z_vs_classic_z_all_cells": rho,
        "spearman_stage1_z_vs_stage2_minuslog10p_on_stage2_cells": {"rho": rho_z1_vs_p2, "n": int(len(s2))},
        "fig2F_left_conditional_confirmation": {
            "intense_to_classic": [conf_at_int(x) for x in (3, 5, 7, 10)],
            "classic_to_intense": [conf_at_z(z) for z in (3, 6, 10)],
            "curve_definition": "cumulative: among all cells at or above the level of the conditioning method",
        },
        "fig2F_right_topN_jaccard": {
            "top_fraction": fracs.tolist(),
            "observed": j_obs.tolist(),
            "baseline_bivariate_normal_same_spearman": base[f"{rho:.2f}"].tolist(),
            "baselines_rho_0.5_0.7_0.9": {k: base[k].tolist() for k in ("0.50", "0.70", "0.90")},
            "at_top_10pct": {"top_fraction": float(fracs[i10]), "observed": float(j_obs[i10]),
                             "baseline_same_rho": float(base[f'{rho:.2f}'][i10])},
            "baseline_method": f"bivariate normal, Pearson r = 2 sin(pi rho/6), n = {N}, 20 draws, seed {SEED}",
        },
        "fig2E_map_similarity": {
            "definition": "mean +- SEM of MSE/PSNR/SSIM (calcium map vs event map, 8x8, smoothing 1 bin, events at "
                          "the session rate) over cells with p_INTENSE < p AND p_classic < p, p from 1e-1 to 1e-7",
            "n_cells_at_p": {f"1e-{k}": int(ne[int(np.argmin(np.abs(xvals_e - k)))]) for k in (1, 3, 5, 7)},
            "ssim_at_p": {f"1e-{k}": float(av[int(np.argmin(np.abs(xvals_e - k))), 2]) for k in (1, 3, 5, 7)},
            "mean_ssim_by_venn_group": lab,
        },
    }
    with open(_paths._F.out_dir(WORK) / f"fig2_numbers_{variant}.json", "w", encoding="utf-8") as f:
        json.dump(numbers, f, indent=1)
    print(json.dumps({k: numbers[k] for k in ("alignment_check", "classic_counts", "fig2C_venn_all_cells",
                                               "spearman_intense_z_vs_classic_z_all_cells",
                                               "fig2F_left_conditional_confirmation")}, indent=1))
    print("top10", numbers["fig2F_right_topN_jaccard"]["at_top_10pct"])
    print("2E", json.dumps(numbers["fig2E_map_similarity"], indent=1)[:1500])


if __name__ == "__main__":
    main()
