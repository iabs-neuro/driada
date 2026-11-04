"""GLM of Section S1.4 on the CA1 recordings: forward selection and each feature tested on its own.

The synchronized sessions of [v11] nof_v1, data downsampled fivefold, frames with finite behaviour,
position as 7 x 7 bins, head direction as 24 bins, speed as 10 quantile bins, the rank-Gaussian
transform of the fluorescence, ridge regression (alpha = 1), cross-validated R^2 over 10 blocks of
the recording (fold order seeded by the neuron index), one-sided Wilcoxon signed-rank test across the
folds at alpha = 0.05 with Bonferroni correction over the candidates (glm_variants.py). No DRIADA
code is involved: the GLM reads the arrays of the session files directly.

Two selections from one set of cross-validated scores per neuron (a score is computed once per set
of features and reused):
  forward     forward selection: the GLM of Section S1.4 and of Supplementary Table 4;
  perfeature  each feature against the model without features (_per_feat of glm_variants.py with
              the same test and correction): a feature is selected if it improves the prediction
              on its own, whatever the other features add (the column "GLM, each feature").

glm/ca1_glm_numbers.py joins the parts and writes the per-neuron selections, among them
[v11] glm_ca1_per_neuron.

usage: python glm/ca1_glm.py [--shard k/N]  ->  <[glm] ca1_glm_dir>/parts/glm_<k>of<N>.csv
       one row per neuron; the sessions of a part already in its file are skipped
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import intense_paths as P  # noqa: E402
import paths_v11 as V  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from glm_variants import _forward, _per_feat, _wilcoxon_p, cv_r2, rank_gaussian  # noqa: E402

NOF_V1 = V.get("nof_v1")

DS = 5
CANON = ["position", "head_direction", "speed"]
MICE = ["H01", "H02", "H03", "H06", "H07", "H08", "H09", "H14", "H23", "H26", "H27", "H31", "H32", "H33",
        "H36", "H39"]


def bin_position(x, y, n=7):
    xn = (x - x.min()) / (x.max() - x.min() + 1e-12)
    yn = (y - y.min()) / (y.max() - y.min() + 1e-12)
    xi = np.clip((xn * n).astype(int), 0, n - 1)
    yi = np.clip((yn * n).astype(int), 0, n - 1)
    flat = yi * n + xi
    X = np.zeros((len(x), n * n), np.float32)
    X[np.arange(len(x)), flat] = 1.0
    return X


def bin_hd(hd, n=24):
    th = np.mod(hd, 2 * np.pi)
    e = np.linspace(0, 2 * np.pi, n + 1)
    idx = np.clip(np.searchsorted(e, th, side="right") - 1, 0, n - 1)
    X = np.zeros((len(hd), n), np.float32)
    X[np.arange(len(hd)), idx] = 1.0
    return X


def bin_speed(v, n=10):
    e = np.quantile(v, np.linspace(0, 1, n + 1))
    e = np.maximum.accumulate(e + np.arange(len(e)) * 1e-9)
    idx = np.clip(np.searchsorted(e, v, side="right") - 1, 0, n - 1)
    X = np.zeros((len(v), n), np.float32)
    X[np.arange(len(v)), idx] = 1.0
    return X


def run_session(mouse, day):
    """Both selections for every neuron of one session; the cross-validated scores are shared by them."""
    d = np.load(NOF_V1 / f"NOF_{mouse}_{day}_aligned.npz", allow_pickle=True)
    cal = d["Calcium"][:, ::DS].astype(np.float64)
    x, y, hd, sp = d["x"][::DS], d["y"][::DS], d["headdirection"][::DS], d["speed"][::DS]
    beh = np.isfinite(x) & np.isfinite(y) & np.isfinite(hd) & np.isfinite(sp)
    x, y, hd, sp, cal = x[beh], y[beh], hd[beh], sp[beh], cal[:, beh]
    blocks = {"position": bin_position(x, y), "head_direction": bin_hd(hd), "speed": bin_speed(sp)}
    rows = []
    for ni in range(cal.shape[0]):
        yv = rank_gaussian(np.nan_to_num(cal[ni], nan=0.0))
        cache = {}

        def score(sel):
            key = tuple(sel)
            if key not in cache:
                X = (np.hstack([blocks[f] for f in sel]).astype(np.float32) if sel
                     else np.zeros((len(yv), 0), np.float32))
                cache[key] = cv_r2(X, yv, n_folds=10, alpha=1.0, seed=ni)
            return cache[key]

        fwd = set(_forward(score, list(blocks), _wilcoxon_p, alpha=0.05, mc="bonferroni")["selected"])
        pf = set(_per_feat(lambda f: score(()) if f is None else score((f,)), list(blocks), _wilcoxon_p,
                           alpha=0.05, mc="bonferroni")["selected"])
        rec = {"mouse": mouse, "session": day, "neuron_idx": ni}
        for c in CANON:
            d_r2 = score((c,)) - score(())
            rec.update({f"fwd_{c}": int(c in fwd), f"pf_{c}": int(c in pf),
                        f"pf_p_{c}": _wilcoxon_p(d_r2), f"pf_dr2_{c}": float(np.mean(d_r2))})
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    k, n = (int(v) for v in a.shard.split("/"))
    out = P.get("glm", "ca1_glm_dir") / "parts"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"glm_{k}of{n}.csv"
    sessions = [(m, f"{dd}D") for m in MICE for dd in range(1, 5)][k::n]
    done = set()
    if path.exists():
        prev = pd.read_csv(path)
        done = set(zip(prev.mouse, prev.session.astype(str)))
    print(f"part {k}/{n}: {len(sessions)} sessions, {len(done)} done", flush=True)
    t_all = time.time()
    for i, (m, dd) in enumerate(sessions):
        if (m, dd) in done:
            continue
        t0 = time.time()
        df = run_session(m, dd)
        df.to_csv(path, mode="a", header=not path.exists(), index=False)
        print(f"  [{i}] {m}_{dd}: n={len(df)} forward {df.fwd_position.sum()}/{df.fwd_head_direction.sum()}/"
              f"{df.fwd_speed.sum()} per-feature {df.pf_position.sum()}/{df.pf_head_direction.sum()}/"
              f"{df.pf_speed.sum()} ({time.time() - t0:.0f} s; total {time.time() - t_all:.0f} s)", flush=True)
    print(f"[OK] part {k}/{n} finished", flush=True)


if __name__ == "__main__":
    main()
