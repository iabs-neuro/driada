"""P-values of the methods without shuffles for every Figure 4 configuration.

run_benchmark.py stores only detections of these methods at fixed thresholds
(p < 1e-50 for correlation and MI, 0.01 / 10 000 for the average), so the
methods cannot be compared at one common threshold afterwards. This script
regenerates each dataset exactly as run_benchmark.py does and saves the per-pair
p-values, so any threshold can be applied without recomputation.

Output: results/full/naive_p/<type>/<stem>.npz with arrays of shape
(neurons, features): p_<method> for each method without shuffles, and gt.
"""
import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist
from scipy.stats import t as t_dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_benchmark as rb  # noqa: E402  (also pins the DRIADA 1.2.2 snapshot)


def p_corr(calcium, feats, ds):
    ca, ft = calcium[:, ::ds], np.asarray(feats)[:, ::ds]
    n = ca.shape[1]
    r = rb._zscore_rows(ca) @ rb._zscore_rows(ft).T / n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return np.nan_to_num(2 * t_dist.sf(np.abs(t), df=n - 2), nan=1.0)


def p_average(calcium, feats, ds):
    ca = calcium[:, ::ds]
    p = np.ones((ca.shape[0], len(feats)))
    for j, f in enumerate(feats):
        on = np.asarray(f)[::ds] == 1
        n1, n0 = on.sum(), (~on).sum()
        if n1 < 2 or n0 < 2:
            continue
        a, b = ca[:, on], ca[:, ~on]
        v1, v0 = a.var(axis=1, ddof=1) / n1, b.var(axis=1, ddof=1) / n0
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (a.mean(axis=1) - b.mean(axis=1)) / np.sqrt(v1 + v0)
            dof = (v1 + v0) ** 2 / (v1 ** 2 / (n1 - 1) + v0 ** 2 / (n0 - 1))
        p[:, j] = np.nan_to_num(2 * t_dist.sf(np.abs(t), df=dof), nan=1.0)
    return p


def p_mi(exp, ftype, calcium, feats, ds):
    # Same tests as rb.naive_mi, returning p instead of a detection mask.
    names = rb.feature_names(ftype)
    n_neurons = calcium.shape[0]
    if ftype == "c":
        ca, ft = calcium[:, ::ds], np.asarray(feats)[:, ::ds]
        n = ca.shape[1]
        signs = np.sign(rb._zscore_rows(ca) @ rb._zscore_rows(ft).T)
        mi = np.zeros((n_neurons, len(names)))
        for i in range(n_neurons):
            for j, name in enumerate(names):
                mi[i, j] = rb.get_mi(exp.neurons[i].ca, exp.dynamic_features[name], ds=ds)
        rho = signs * np.sqrt(np.clip(1 - 2.0 ** (-2 * mi), 0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
        # run_benchmark.py tests one-sided, which never detects negatively tuned
        # pairs; both are kept so the figure can use the two-sided test like the
        # other methods.
        p1 = np.nan_to_num(t_dist.sf(t, df=n - 2), nan=1.0)
        p2 = np.nan_to_num(2 * t_dist.sf(np.abs(t), df=n - 2), nan=1.0)
        p1[np.abs(rho) >= 1.0] = 0.0
        p2[np.abs(rho) >= 1.0] = 0.0
        return p2, p1
    ca = calcium[:, ::ds]
    n = ca.shape[1]
    n_bins = int(np.ceil(np.log2(n)) + 1)
    q = np.linspace(0, 1, n_bins + 1)[1:-1]
    binned = np.stack([np.digitize(row, np.quantile(row, q)) for row in ca])
    p = np.ones((n_neurons, len(feats)))
    for j, f in enumerate(feats):
        fv = np.asarray(f)[::ds]
        vals = np.unique(fv)
        if len(vals) < 2:
            continue
        for i in range(n_neurons):
            ct = np.stack([np.bincount(binned[i][fv == v], minlength=n_bins) for v in vals])
            ct = ct[:, ct.sum(axis=0) > 0]
            ct = ct[ct.sum(axis=1) > 0]
            if min(ct.shape) < 2:
                continue
            exp_ct = ct.sum(axis=1, keepdims=True) * ct.sum(axis=0, keepdims=True) / ct.sum()
            with np.errstate(divide="ignore", invalid="ignore"):
                g = 2 * np.sum(np.where(ct > 0, ct * np.log(ct / exp_ct), 0.0))
            p[i, j] = chi2_dist.sf(g, (ct.shape[0] - 1) * (ct.shape[1] - 1))
    return p


def one(args):
    ftype, snr, skip, r, out_dir = args
    out = Path(out_dir) / f"{rb.config_stem(snr, skip, r)}.npz"
    if out.exists():
        return 0.0
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats, calcium, gt, _ = rb.make_data(ftype, snr, skip, r)
        exp = rb.make_exp(ftype, feats, calcium)
        ds = rb.PAPER["ds"]
        p = p_mi(exp, ftype, calcium, feats, ds)
        arrs = {"gt": gt}
        if ftype == "c":
            arrs["p_mi_naive"], arrs["p_mi_naive_onesided"] = p
            arrs["p_corr"] = p_corr(calcium, feats, ds)
        else:
            arrs["p_mi_naive"] = p
            arrs["p_average"] = p_average(calcium, feats, ds)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrs)
    tmp.replace(out)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="run only the first N configs (timing)")
    a = ap.parse_args()
    tasks = []
    for ftype in ["d", "c"]:
        out_dir = rb.V.get("synthetic_dir") / "full" / "naive_p" / ftype
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in range(rb.N_REALIZATIONS):
            for snr in rb.SNR_VALUES:
                for skip in rb.SKIP_VALUES:
                    tasks.append((ftype, snr, skip, r, str(out_dir)))
    if a.limit:
        tasks = tasks[:a.limit]
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(one, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            sec = fut.result()
            done += 1
            print(f"[{done}/{len(tasks)}] {t[0]} SNR={t[1]} skip={t[2]} r={t[3]}: {sec:.0f} s", flush=True)
    print("[OK]", flush=True)


if __name__ == "__main__":
    main()
