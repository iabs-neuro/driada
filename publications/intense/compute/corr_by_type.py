"""INTENSE with a linear dependence measure on the by_type representation (NOF).

Answers whether the MI-over-correlation gain of the paper comes from the measure
or from the representation. The Pearson run of the paper (compute/run_pearson.py)
gives the correlation the raw one-dimensional features. Here the same run
(tools/run_intense_analysis.py of DRIADA 1.2.2, --metric fast_pearsonr, the same
seed, shuffles, delay search, p-value and anti-selectivity rule, no
disentanglement) receives the by_type features of the MI run instead: speed ->
[speed, (speed - median)^2], directions -> [cos, sin, cos 2th, sin 2th];
discrete features are unchanged.

The measure is the multiple correlation R = sqrt(R^2) of the linear regression
of the neuron on the components of the feature. For a one-dimensional feature
R = |r|, which is what fast_pearsonr returns, so the measure reduces to the
Pearson statistic of the paper. DRIADA 1.2.2 has no such metric; it is added
here at run time by wrapping the FFT dispatch of driada.intense (get_fft_type,
_extract_fft_data, _FFT_COMPUTE) and get_sim, without changing the library.

Variants (--variant):
  rmult       R on the original scale (main result).
  rmult_rank  R on copula-normalised data of neuron and continuous features
              (normal scores, a Spearman-like analogue). Up to the monotone map
              MI = -1/2 log2(1 - R^2) this is the GCMI statistic itself.
  check1d     raw representation, every pair routed through the R code: must
              reproduce the Pearson run of the paper.

Parallelisation is off (one process per run): loky workers would import an
unpatched library.

usage: python corr_by_type.py --variant rmult --out DIR [npz ...]   (no npz -> all 64)
"""
import argparse
import runpy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

RMULT = "pearson_rmult"
REG = 1e-10


def rmult_fft(z, X, shifts):
    """Multiple correlation of z on the rows of X at circular shifts of X.

    Parameters
    ----------
    z : ndarray of shape (n,)
        Neural signal.
    X : ndarray of shape (d, n) or (n,)
        Feature components.
    shifts : ndarray of int
        Shifts, in the convention of compute_pearson_batch_fft and
        compute_mi_mts_fft (cross_corr[s] = sum_j z[j] x[j + s]).

    Returns
    -------
    ndarray of shape (len(shifts),)
        R in [0, 1]; equals |r| of compute_pearson_batch_fft when d = 1.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = len(z)
    z = z - z.mean()
    X = X - X.mean(axis=1, keepdims=True)
    var_z = np.dot(z, z) / (n - 1)
    if var_z < REG:
        return np.zeros(len(shifts))
    keep = X.var(axis=1) > REG
    if not keep.any():
        return np.zeros(len(shifts))
    X = X[keep]
    cov_xx = np.atleast_2d(np.cov(X))
    cov_xx = cov_xx + np.eye(len(X)) * REG * np.trace(cov_xx)
    s = np.asarray(shifts).astype(int) % n
    fz = np.fft.rfft(z)
    C = np.empty((len(X), len(s)))
    for i in range(len(X)):
        C[i] = np.fft.irfft(fz * np.conj(np.fft.rfft(X[i])), n=n)[s] / (n - 1)
    r2 = np.sum(C * np.linalg.solve(cov_xx, C), axis=0) / var_z
    return np.sqrt(np.clip(r2, 0.0, 1.0))


def install(variant):
    """Wrap the FFT dispatch and get_sim of the configured DRIADA release."""
    import driada
    from driada.information.info_base import MultiTimeSeries, TimeSeries
    import driada.information.info_base as ib
    import driada.intense.delay as dl
    import driada.intense.fft as ff
    import driada.intense.intense_base as base
    from driada.intense.representations import substitute_by_type

    rank = variant == "rmult_rank"
    orig_type, orig_extract, orig_sim = ff.get_fft_type, ff._extract_fft_data, ib.get_sim

    def is_mts(ts):
        return isinstance(ts, MultiTimeSeries)

    def routed(ts1, ts2, metric):
        if metric != "fast_pearsonr" or is_mts(ts1):
            return False
        if is_mts(ts2):
            return not ts2.discrete
        return variant in ("check1d", "rmult_rank") and isinstance(ts2, TimeSeries)

    def arrays(ts1, ts2, ds):
        z = ts1.copula_normal_data if rank else ts1.data
        if ts2.discrete:
            X = np.asarray(ts2.data, dtype=float)
        else:
            X = ts2.copula_normal_data if rank else ts2.data
        return np.asarray(z, dtype=float)[::ds], np.atleast_2d(X)[:, ::ds]

    def get_fft_type(ts1, ts2, metric, mi_estimator, count, engine, for_delays=False):
        if routed(ts1, ts2, metric):
            return RMULT
        return orig_type(ts1, ts2, metric, mi_estimator, count, engine, for_delays=for_delays)

    def extract(ts1, ts2, fft_type, ds):
        if fft_type == RMULT:
            return arrays(ts1, ts2, ds)
        return orig_extract(ts1, ts2, fft_type, ds)

    def get_sim(x, y, metric, shift=0, ds=1, **kw):
        if routed(x, y, metric):
            z, X = arrays(x, y, ds)
            # get_sim rolls y forward by `shift`
            return float(rmult_fft(z, np.roll(X, shift, axis=1), np.array([0]))[0])
        return orig_sim(x, y, metric, shift=shift, ds=ds, **kw)

    ff._FFT_COMPUTE[RMULT] = rmult_fft
    for mod in (ff, base, dl):
        mod.get_fft_type = get_fft_type
        mod._extract_fft_data = extract
    for mod in (base, dl):
        mod.get_sim = get_sim

    orig_ccfs = driada.compute_cell_feat_significance

    def ccfs(exp, feat_bunch=None, **kw):
        kw["enable_parallelization"] = False
        if variant != "check1d":
            feat_bunch, subs = substitute_by_type(list(feat_bunch), exp)
            print(f"by_type substitutions: {subs}", flush=True)
        return orig_ccfs(exp, feat_bunch=feat_bunch, **kw)

    driada.compute_cell_feat_significance = ccfs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["rmult", "rmult_rank", "check1d"])
    ap.add_argument("--out", required=True)
    ap.add_argument("npz", nargs="*")
    a = ap.parse_args()
    driada = P.use_driada()
    print("driada", driada.__version__, driada.__file__, "variant", a.variant, flush=True)
    install(a.variant)
    sys.argv = [str(P.TOOLS_DIR / "run_intense_analysis.py"),
                *(a.npz or ["--dir", str(P.NOF_ALIGNED)]),
                "--output-dir", a.out, "--metric", "fast_pearsonr", "--representation", "raw",
                "--no-disentanglement", "--seed", str(P.SEED), "--skip-computed"]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
