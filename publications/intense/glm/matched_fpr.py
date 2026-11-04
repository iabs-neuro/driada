# -*- coding: utf-8 -*-
"""Comparison of INTENSE and the GLM at matched false-positive rates (Section S1.3, Supplementary Fig. 3).

Every method emits one p-value per neuron-feature pair, once; the decision threshold is then swept,
which traces for each method the F1 on each cell type against the false-positive rate on the
nonselective cells (the share of them with at least one detected feature). The methods are compared
at equal false-positive rates instead of at thresholds of their own.

Recordings: the standard population of synth_common (45 cells), 900 s at 20 Hz, SNR 20, without spike
reconstruction, on the first 10 seeds of synth_common.SEEDS20. On each recording:
  M_cv_wilcox      GLM testing each feature on its own: cross-validated R^2 over 10 folds and a
                   Wilcoxon test of the fold R^2 against zero;
  M_ho_shift       GLM testing each feature on its own: R^2 on the last fifth of the recording against
                   INTENSE's circular-shift null (200 shifts);
  INTENSE_delay    INTENSE with delay search;
  INTENSE_nodelay  INTENSE at zero lag, like the GLM.
Both GLM arms fit the rank-Gaussian transform of the fluorescence at zero lag. INTENSE scores every
pair at full second-stage depth (mode 'stage2': 10,000 shuffles for every pair, ds = 5, no
correction, since the sweep does the calibration) on untransformed features (representation 'raw'),
with the library's default seed.

Outputs in [glm] matched_fpr_dir: pvals.csv (one p-value per seed, method, neuron and feature, written
again after each seed), curves.csv (the sweep: false-positive rate and F1 per cell type at each
threshold; f1_mixed_lin is the mean of the two linear mixtures) and matched.csv (F1 on linear mixed
cells at the threshold whose false-positive rate is closest to 0, 0.05, 0.10 and 0.20).
glm/matched_fpr_numbers.py reads the numbers of the text from them, glm/matched_fpr_figure.py draws
Supplementary Fig. 3.

usage: python glm/matched_fpr.py
"""
import sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(HERE))
import intense_paths as P  # noqa: E402

driada = P.use_driada(tools=False)

import numpy as np, pandas as pd  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from driada.experiment.synthetic import generate_tuned_selectivity_exp  # noqa: E402
from synth_common import SEEDS20, POP_STD, TRUE_STD, group_map, prf1, CANON  # noqa: E402
from glm_variants import build_blocks, rank_gaussian, cv_r2, _wilcoxon_p  # noqa: E402

OUT = P.get("glm", "matched_fpr_dir")
SEEDS = SEEDS20[:10]
GLM_METHODS = ["M_cv_wilcox", "M_ho_shift"]
ORDER = ["hd_cells", "speed_cells", "event_cells", "mixed_hd_speed_lin",
         "mixed_speed_event_lin", "mixed_hd_speed_and", "mixed_speed_event_and", "nonselective"]
ALPHAS = np.concatenate([np.logspace(-4, -1, 25), np.linspace(0.12, 0.5, 10)])
INTENSE_ARMS = {"INTENSE_delay": True, "INTENSE_nodelay": False}
ALL = GLM_METHODS + list(INTENSE_ARMS)


def gen(seed):
    return generate_tuned_selectivity_exp(
        population=POP_STD, duration=900.0, fps=20.0, seed=seed,
        n_discrete_features=1, baseline_rate=0.1, peak_rate=2.0,
        decay_time=2.0, calcium_noise=0.05, verbose=False, reconstruct_spikes=None,
    )


def _heldout_r2(X, y, split):
    if X.shape[1] == 0:
        return 0.0
    m = Ridge(alpha=1.0).fit(X[:split], y[:split])
    pred = m.predict(X[split:])
    ss_res = float(((y[split:] - pred) ** 2).sum())
    ss_tot = float(((y[split:] - y[:split].mean()) ** 2).sum())
    return 1.0 - ss_res / (ss_tot + 1e-12)


def pvals_cv_wilcox(blocks, y, seed):
    """Marginal per-feature: CV-R2 over 10 folds, Wilcoxon(fold R2 > 0)."""
    out = {}
    for f, X in blocks.items():
        r2 = cv_r2(X.astype(np.float32), y, 10, 1.0, seed)
        out[f] = _wilcoxon_p(r2) if np.mean(r2) > 0 else 1.0
    return out


def pvals_ho_shift(blocks, y, rng, n_shuffle=200):
    """Marginal per-feature: held-out R2 with INTENSE's circular-shift null."""
    T = len(y); split = int(0.8 * T); out = {}
    for f, X in blocks.items():
        Xd = X.astype(np.float64)
        obs = _heldout_r2(Xd, y, split)
        if obs <= 0:
            out[f] = 1.0; continue
        null = np.empty(n_shuffle)
        for k in range(n_shuffle):
            sh = int(rng.integers(T // 10, T - T // 10))
            null[k] = _heldout_r2(Xd, np.roll(y, sh), split)
        out[f] = (1 + int((null >= obs).sum())) / (n_shuffle + 1)
    return out


def pvals_intense(exp, delays):
    """Raw INTENSE p-value of every neuron-feature pair; a pair without one gets p = 1.

    Mode 'stage2' rather than 'two_stage': a first-stage screen would leave the rejected pairs
    without a second-stage p-value, which would cut INTENSE's threshold sweep short.
    """
    feats = list(exp.dynamic_features)
    # head direction is present as its 1-D components and as the 2-D multifeature that INTENSE
    # scores; only the 2-D one is read, since reading the other would compute it off schedule
    if "head_direction_2d" in feats:
        feats = [f for f in feats if f != "head_direction"]
    driada.compute_cell_feat_significance(
        exp, mode="stage2", n_shuffles_stage1=100, n_shuffles_stage2=10000,
        verbose=False, with_disentanglement=False,
        multifeature_map=driada.intense.DEFAULT_MULTIFEATURE_MAP,
        pval_thr=0.05, multicomp_correction=None, find_optimal_delays=delays, ds=5,
        representation="raw", force_update=True, use_precomputed_stats=False,
    )
    out = {}
    for ni in range(exp.n_cells):
        d = {}
        for feat in feats:
            name = "head_direction" if feat == "head_direction_2d" else feat
            if name not in CANON:
                continue
            try:
                p = exp.get_neuron_feature_pair_stats(ni, feat).get("pval", None)
                d[name] = float(p) if p is not None and np.isfinite(p) else 1.0
            except Exception:
                d[name] = 1.0
        out[ni] = d
    return out


def sweep(df, methods):
    curve = []
    gm = group_map(POP_STD)
    for meth in methods:
        sub = df[df.method == meth]
        for a in ALPHAS:
            f1_seed = {g: [] for g in ORDER}; fpr_seed = []
            for s in SEEDS:
                ss = sub[sub.seed == s]
                f1s = {g: [] for g in ORDER}; fps = []
                for ni, grp in ss.groupby("neuron"):
                    dd = set(grp[grp.pval < a].feature); g = gm[ni]
                    f1s[g].append(prf1(dd, TRUE_STD[g]))
                    if g == "nonselective":
                        fps.append(1.0 if dd else 0.0)
                for g in ORDER:
                    if f1s[g]:
                        f1_seed[g].append(float(np.mean(f1s[g])))
                fpr_seed.append(float(np.mean(fps)) if fps else np.nan)
            rec = {"method": meth, "alpha": float(a), "fpr_nonsel": float(np.nanmean(fpr_seed))}
            for g in ORDER:
                rec[f"f1_{g}"] = float(np.mean(f1_seed[g])) if f1_seed[g] else np.nan
            rec["f1_mixed_lin"] = float(np.nanmean([rec["f1_mixed_hd_speed_lin"], rec["f1_mixed_speed_event_lin"]]))
            curve.append(rec)
    return pd.DataFrame(curve)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in SEEDS:
        t0 = time.time()
        exp = gen(s); groups = group_map(POP_STD)
        blocks = build_blocks(exp); rng = np.random.default_rng(s)
        ip = {}
        for arm, delays in INTENSE_ARMS.items():
            try:
                ip[arm] = pvals_intense(exp, delays)
            except Exception:
                print(f"[FAILED] {arm} seed={s}:\n{traceback.format_exc()}", flush=True); ip[arm] = {}
        for ni in range(exp.n_cells):
            y = rank_gaussian(exp.calcium.data[ni].astype(np.float64))
            pv = {"M_cv_wilcox": pvals_cv_wilcox(blocks, y, s),
                  "M_ho_shift": pvals_ho_shift(blocks, y, rng)}
            for arm in INTENSE_ARMS:
                pv[arm] = ip[arm].get(ni, {})
            for meth, d in pv.items():
                for f in CANON:
                    rows.append({"seed": s, "method": meth, "neuron": ni, "group": groups[ni],
                                 "feature": f, "pval": float(d.get(f, 1.0))})
        pd.DataFrame(rows).to_csv(OUT / "pvals.csv", index=False)
        print(f"  seed={s} {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    cv = sweep(df, ALL); cv.to_csv(OUT / "curves.csv", index=False)
    out_rows = []
    for target in [0.0, 0.05, 0.10, 0.20]:
        line = {"target_fpr": target}
        for meth in ALL:
            sub = cv[cv.method == meth].copy()
            sub["d"] = (sub.fpr_nonsel - target).abs()
            best = sub.sort_values(["d", "alpha"]).iloc[0]
            line[meth] = round(float(best.f1_mixed_lin), 3)
            line[f"{meth}_actual_fpr"] = round(float(best.fpr_nonsel), 3)
        out_rows.append(line)
    matched = pd.DataFrame(out_rows); matched.to_csv(OUT / "matched.csv", index=False)
    print(matched.to_string(index=False), flush=True)


if __name__ == "__main__":
    t = time.time(); main(); print(f"\nTOTAL {time.time()-t:.0f}s", flush=True)
