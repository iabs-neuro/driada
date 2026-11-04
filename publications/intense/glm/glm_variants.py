# -*- coding: utf-8 -*-
"""The GLM of Supplementary Section S1 and its variants (Supplementary Table 2).

Each variant takes a synthetic experiment, a neuron index and a seed (which orders the
cross-validation folds) and returns the features it selects. VARIANTS maps the labels of the
scripts to the rows of Supplementary Table 2 and the bars of Supplementary Fig. 1:

    V0_base        GLM baseline
    V_ttest        + t-test
    V_fdr          + FDR
    V_alpha10      + alpha = 0.10
    V_folds20      + 20 folds
    V_poisson      Poisson/spikes
    V_max          GLM max-strength (the combined variant)
    V_max_perfeat  max, per-feature

The building blocks (rank_gaussian, cv_r2, _forward, _per_feat, _wilcoxon_p) also make the GLM
of the CA1 recordings (ca1_glm.py), the GLM-only controls of Section S1.3 (synth_compute.py) and
the GLM arms of the comparison at matched false-positive rates (matched_fpr.py).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.stats import wilcoxon, ttest_rel, rankdata, norm
from sklearn.linear_model import Ridge, PoissonRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

CANDIDATE_FEATURES = ("head_direction", "speed", "event_0")


def _onehot(idx, n, T):
    out = np.zeros((T, n), dtype=np.float32); out[np.arange(T), idx] = 1.0; return out


def _hd_bins(v, n=24):
    th = np.mod(v, 2 * np.pi); e = np.linspace(0, 2 * np.pi, n + 1)
    return _onehot(np.clip(np.searchsorted(e, th, side="right") - 1, 0, n - 1), n, len(v))


def _speed_bins(v, n=10, mode="quantile"):
    if mode == "quantile":
        e = np.quantile(v, np.linspace(0, 1, n + 1))
        e = np.maximum.accumulate(e + np.arange(len(e)) * 1e-9)
    else:
        e = np.linspace(v.min(), v.max() + 1e-9, n + 1)
    return _onehot(np.clip(np.searchsorted(e, v, side="right") - 1, 0, n - 1), n, len(v))


def _binary_bins(v):
    return _onehot((v > 0.5).astype(int), 2, len(v))


def build_blocks(exp, hd_n=24, speed_n=10, speed_mode="quantile"):
    b = {}
    if "head_direction" in exp.dynamic_features:
        b["head_direction"] = _hd_bins(exp.dynamic_features["head_direction"].data, hd_n)
    if "speed" in exp.dynamic_features:
        b["speed"] = _speed_bins(exp.dynamic_features["speed"].data, speed_n, speed_mode)
    if "event_0" in exp.dynamic_features:
        b["event_0"] = _binary_bins(exp.dynamic_features["event_0"].data.astype(float))
    return b


def rank_gaussian(y):
    return norm.ppf(rankdata(y, method="average") / (len(y) + 1.0))


def _smoothness_D_single(n, circular):
    if circular:
        return np.eye(n) - np.roll(np.eye(n), -1, axis=1)
    return np.eye(n)[1:] - np.eye(n)[:-1]


def make_block_D(specs):
    blks = [_smoothness_D_single(n, c) for (n, c) in specs]
    rows = sum(b.shape[0] for b in blks); cols = sum(b.shape[1] for b in blks)
    D = np.zeros((rows, cols)); r0 = c0 = 0
    for b in blks:
        D[r0:r0 + b.shape[0], c0:c0 + b.shape[1]] = b
        r0 += b.shape[0]; c0 += b.shape[1]
    return D


def block_cv(T, n_folds, seed):
    edges = np.linspace(0, T, n_folds + 1).astype(int)
    order = np.random.default_rng(seed).permutation(n_folds)
    for b in order:
        m = np.zeros(T, dtype=bool); m[edges[b]:edges[b + 1]] = True
        yield (~m).nonzero()[0], m.nonzero()[0]


def cv_r2(X, y, n_folds=10, alpha=1.0, seed=0, D=None, lam_sm=0.0):
    if X.shape[1] == 0:
        return np.zeros(n_folds)
    r2 = np.empty(n_folds)
    for k, (tr, te) in enumerate(block_cv(len(y), n_folds, seed)):
        if D is None or lam_sm == 0:
            m = Ridge(alpha=alpha, fit_intercept=True)
            m.fit(X[tr], y[tr])
            pred = m.predict(X[te])
        else:
            X_tr = X[tr].astype(np.float64); y_tr_c = y[tr] - y[tr].mean()
            n_f = X_tr.shape[1]
            A = X_tr.T @ X_tr + alpha * np.eye(n_f) + lam_sm * (D.T @ D)
            b = X_tr.T @ y_tr_c
            w = np.linalg.solve(A, b)
            pred = X[te] @ w + y[tr].mean()
        ss_res = np.sum((y[te] - pred) ** 2)
        ss_tot = np.sum((y[te] - y[tr].mean()) ** 2)
        r2[k] = 1.0 - ss_res / (ss_tot + 1e-12)
    return r2


def cv_poisson_ll(X, y_counts, n_folds=10, alpha=1.0, seed=0):
    """Per-fold held-out Poisson log-likelihood. Properly handles intercept-only."""
    if y_counts.sum() < 5:
        return np.full(n_folds, -1e9)
    ll = np.empty(n_folds)
    for k, (tr, te) in enumerate(block_cv(len(y_counts), n_folds, seed)):
        if X.shape[1] == 0:
            mu = max(float(y_counts[tr].mean()), 1e-9)
            ll[k] = float(np.sum(y_counts[te] * np.log(mu) - mu))
            continue
        try:
            m = PoissonRegressor(alpha=alpha, fit_intercept=True, max_iter=300)
            m.fit(X[tr], y_counts[tr])
            mu = np.clip(m.predict(X[te]), 1e-12, None)
            ll[k] = float(np.sum(y_counts[te] * np.log(mu) - mu))
        except Exception:
            ll[k] = -1e9
    return ll


# ---------------------------------------------------------------------------
# Significance tests
# ---------------------------------------------------------------------------

def _wilcoxon_p(d):
    if np.all(np.abs(d) < 1e-12):
        return 1.0
    try:
        return float(wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def _ttest_p(d):
    if np.all(np.abs(d) < 1e-12) or d.std() == 0:
        return 1.0
    try:
        t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
        # one-sided greater
        from scipy.stats import t as tdist
        return float(1.0 - tdist.cdf(t, df=len(d) - 1))
    except Exception:
        return 1.0


def fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg. Returns boolean reject array same length as pvals."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    threshold = alpha * (np.arange(1, n + 1) / n)
    below = sorted_p <= threshold
    if not below.any():
        return np.zeros(n, dtype=bool)
    cutoff = np.where(below)[0].max()
    reject = np.zeros(n, dtype=bool)
    reject[order[: cutoff + 1]] = True
    return reject


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

def _forward(score_fn, candidates, test_fn, alpha=0.05, mc="bonferroni"):
    selected = []
    current = score_fn(())
    cand = list(candidates)
    while cand:
        if mc == "bonferroni":
            alpha_local = alpha / len(cand)
        else:
            alpha_local = alpha  # FDR applied across the candidate set per-step
        scores_d, pvals, names, new_scores = [], [], [], []
        for f in cand:
            new = score_fn(tuple(selected) + (f,))
            d = new - current
            pv = test_fn(d)
            scores_d.append(float(np.mean(d)))
            pvals.append(pv); names.append(f); new_scores.append(new)
        order = np.argmin(pvals)
        if mc == "fdr":
            # accept the lowest p-value if it passes FDR among current candidates
            reject = fdr_bh(pvals, alpha=alpha)
            if not reject[order] or scores_d[order] <= 0:
                break
        else:
            if pvals[order] >= alpha_local or scores_d[order] <= 0:
                break
        selected.append(names[order])
        current = new_scores[order]
        cand.remove(names[order])
    return {"selected": selected, "final_score": float(np.mean(current))}


def _per_feat(scorer, candidates, test_fn, alpha=0.05, mc="bonferroni"):
    pvals = []; deltas = []
    baseline = scorer(None)
    for f in candidates:
        sc = scorer(f)
        d = sc - baseline
        pvals.append(test_fn(d))
        deltas.append(float(np.mean(d)))
    if mc == "bonferroni":
        threshold = alpha / len(candidates)
        keep = [(pv < threshold) and (de > 0) for pv, de in zip(pvals, deltas)]
    else:
        reject = fdr_bh(pvals, alpha=alpha)
        keep = [r and (de > 0) for r, de in zip(reject, deltas)]
    selected = [f for f, k in zip(candidates, keep) if k]
    return {"selected": selected, "final_score": 0.0}


# Variant builders

def make_baseline():
    def v(exp, ni, seed=0):
        blocks = build_blocks(exp)
        y = exp.calcium.data[ni].astype(np.float64)
        def sc(sel):
            X = np.hstack([blocks[f] for f in sel]).astype(np.float32) if sel \
                else np.zeros((len(y), 0), np.float32)
            return cv_r2(X, y, 10, 1.0, seed)
        return _forward(sc, list(blocks.keys()), _wilcoxon_p, alpha=0.05, mc="bonferroni")
    return v


def make_ttest():
    """V0 + paired t-test instead of Wilcoxon (more sensitive on small samples)."""
    def v(exp, ni, seed=0):
        blocks = build_blocks(exp)
        y = exp.calcium.data[ni].astype(np.float64)
        def sc(sel):
            X = np.hstack([blocks[f] for f in sel]).astype(np.float32) if sel \
                else np.zeros((len(y), 0), np.float32)
            return cv_r2(X, y, 10, 1.0, seed)
        return _forward(sc, list(blocks.keys()), _ttest_p, alpha=0.05, mc="bonferroni")
    return v


def make_fdr():
    """V0 + FDR (Benjamini-Hochberg) instead of Bonferroni."""
    def v(exp, ni, seed=0):
        blocks = build_blocks(exp)
        y = exp.calcium.data[ni].astype(np.float64)
        def sc(sel):
            X = np.hstack([blocks[f] for f in sel]).astype(np.float32) if sel \
                else np.zeros((len(y), 0), np.float32)
            return cv_r2(X, y, 10, 1.0, seed)
        return _forward(sc, list(blocks.keys()), _wilcoxon_p, alpha=0.05, mc="fdr")
    return v


def make_alpha10():
    """V0 + α=0.10 (more lenient)."""
    def v(exp, ni, seed=0):
        blocks = build_blocks(exp)
        y = exp.calcium.data[ni].astype(np.float64)
        def sc(sel):
            X = np.hstack([blocks[f] for f in sel]).astype(np.float32) if sel \
                else np.zeros((len(y), 0), np.float32)
            return cv_r2(X, y, 10, 1.0, seed)
        return _forward(sc, list(blocks.keys()), _wilcoxon_p, alpha=0.10, mc="bonferroni")
    return v


def make_folds20():
    """V0 + 20 CV folds (more power for the test)."""
    def v(exp, ni, seed=0):
        blocks = build_blocks(exp)
        y = exp.calcium.data[ni].astype(np.float64)
        def sc(sel):
            X = np.hstack([blocks[f] for f in sel]).astype(np.float32) if sel \
                else np.zeros((len(y), 0), np.float32)
            return cv_r2(X, y, 20, 1.0, seed)
        return _forward(sc, list(blocks.keys()), _wilcoxon_p, alpha=0.05, mc="bonferroni")
    return v


def make_poisson():
    """V0 design + Poisson on reconstructed spikes (Hardcastle 2017 exact model)."""
    def v(exp, ni, seed=0):
        blocks = build_blocks(exp)
        s = exp.spikes.data[ni] if (exp.spikes is not None and hasattr(exp.spikes, "data")) else None
        if s is None or np.sum(s) < 5:
            # fallback: threshold on rank-Gaussian calcium
            s = (rank_gaussian(exp.calcium.data[ni].astype(float)) > 1.5).astype(int)
        spike_counts = s.astype(int)
        def sc(sel):
            X = np.hstack([blocks[f] for f in sel]).astype(np.float32) if sel \
                else np.zeros((len(spike_counts), 0), np.float32)
            return cv_poisson_ll(X, spike_counts, 10, 1.0, seed)
        return _forward(sc, list(blocks.keys()), _wilcoxon_p, alpha=0.05, mc="bonferroni")
    return v


def make_max():
    """Throw everything: rank-Gaussian, linear speed bins, smoothness prior,
    t-test, FDR, 20 folds, α=0.10. The maximum-strength Gaussian GLM."""
    def v(exp, ni, seed=0):
        blocks = build_blocks(exp, hd_n=24, speed_n=10, speed_mode="linear")
        y = rank_gaussian(exp.calcium.data[ni].astype(np.float64))
        D_specs = {"head_direction": (24, True), "speed": (10, False), "event_0": (2, False)}
        def sc(sel):
            if not sel:
                return cv_r2(np.zeros((len(y), 0), np.float32), y, 20, 1.0, seed)
            X = np.hstack([blocks[f] for f in sel]).astype(np.float64)
            D = make_block_D([D_specs[f] for f in sel])
            return cv_r2(X, y, 20, 1.0, seed, D=D, lam_sm=2.0)
        return _forward(sc, list(blocks.keys()), _ttest_p, alpha=0.10, mc="fdr")
    return v


def make_max_perfeat():
    """V_max but per-feature univariate test instead of forward selection.
    Lets each feature get a fair shot independent of greedy ordering."""
    def v(exp, ni, seed=0):
        blocks = build_blocks(exp, hd_n=24, speed_n=10, speed_mode="linear")
        y = rank_gaussian(exp.calcium.data[ni].astype(np.float64))
        D_specs = {"head_direction": (24, True), "speed": (10, False), "event_0": (2, False)}
        def scorer(feat):
            if feat is None:
                return cv_r2(np.zeros((len(y), 0), np.float32), y, 20, 1.0, seed)
            D = make_block_D([D_specs[feat]])
            return cv_r2(blocks[feat].astype(np.float64), y, 20, 1.0, seed, D=D, lam_sm=2.0)
        return _per_feat(scorer, list(blocks.keys()), _ttest_p, alpha=0.10, mc="fdr")
    return v


VARIANTS = {
    "V0_base":          make_baseline(),
    "V_ttest":          make_ttest(),
    "V_fdr":            make_fdr(),
    "V_alpha10":        make_alpha10(),
    "V_folds20":        make_folds20(),
    "V_poisson":        make_poisson(),
    "V_max":            make_max(),
    "V_max_perfeat":    make_max_perfeat(),
}


def selected_to_features(selected):
    return {s for s in selected if s in CANDIDATE_FEATURES}
