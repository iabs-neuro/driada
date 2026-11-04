"""Does the zero-inflated gamma model of DRIADA describe the stored stage-2 shuffles of each measure?
And the comparison of MI with correlation when the same gamma model is applied to R^2, the quantity
that plays the role of MI for a linear measure (for Gaussian data MI = -1/2 log2(1 - R^2) ~ R^2).

Reads results/NOF_*_results.npz of the frame-rate recompute (stage-2 values: column 0 the observed value,
columns 1.. the shuffles, stored as float16) for MI (by_type), multiple correlation on the same
representations (rmult) and Pearson correlation on the raw variables (pearson). Nothing is recomputed.

Gamma fit: maximum likelihood with location 0 on the values above 1e-10 (Minka's start + Newton steps),
zero mass = share of values at or below 1e-10, p = (1 - zero mass) * gamma tail, as in
driada.intense.stats.get_gamma_zi_p. Checked against the stored p-values of MI.
Criterion "paper": at most 4 shuffles at or above the observed value (DRIADA criterion2, topk 5) and
p < 0.001; anti-selectivity filter; position (place, x, y) left out, as in the paper's comparison.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.special import digamma, gammaincc, polygamma

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402
import paths_v11 as V  # noqa: E402

RUNS = {"mi": V.get("stage2_results"), "rmult": V.get("rmult_dir") / "results",
        "pearson": P.PEARSON_DIR / "results"}
LEFT_OUT = {"place", "x", "y"}
CIRC = {"headdirection", "bodydirection"}


def base(f):
    return re.sub(r"(_harm2|_quad|_2d)$", "", f)


def zi_gamma_p(sh, val):
    """sh: (P, n) null values, val: (P,) observed -> (P,) zero-inflated gamma tail probabilities."""
    sh = sh.astype(np.float64)
    nz = sh > 1e-10
    pi = 1.0 - nz.mean(1)
    x = np.where(nz, sh, np.nan)
    m = np.nanmean(x, 1)
    lm = np.nanmean(np.log(x), 1)
    s = np.log(m) - lm
    k = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
    for _ in range(6):
        k = k - (np.log(k) - digamma(k) - s) / (1 / k - polygamma(1, k))
    theta = m / k
    p = (1 - pi) * gammaincc(k, np.maximum(val, 0) / theta)
    p = np.where(val <= 1e-10, 1.0, p)
    return np.clip(p, 0, 1)


def load(path):
    z = np.load(path, allow_pickle=True)
    st = json.loads(str(z["_stats_json"][0] if z["_stats_json"].shape != () else z["_stats_json"].item()))
    idx = z["info_me_total2_indices"].astype(int)
    data = z["info_me_total2_data"]
    return st, idx, data


def main():
  res = {}
  for run, folder in RUNS.items():
      tail = {"all": [], "continuous": [], "circular": [], "discrete": []}
      tail998 = []
      pairs_paper, pairs_sq, pairs_rank = set(), set(), set()
      agree = []
      for p in sorted(folder.glob("NOF_*_results.npz")):
          sess = p.name[4:-len("_results.npz")]
          st, idx, data = load(p)
          neurons = list(st.keys())
          feats = list(st[neurons[0]].keys())
          # rows of pairs that did not reach stage 2 are stored as zeros; keep the computed ones
          keep = np.array([st[neurons[i]][feats[j]].get("rval") is not None for i, j in idx])
          idx, data = idx[keep], data[keep]
          obs = data[:, 0].astype(np.float64)
          sh = data[:, 1:]
          n = sh.shape[1]
          # mapping check: the stored observed value equals the statistic of the pair
          me = np.array([st[neurons[i]][feats[j]].get("me", np.nan) for i, j in idx], dtype=float)
          ok_map = np.allclose(obs, me, rtol=3e-3, atol=1e-4, equal_nan=True)
          if not ok_map:
              sys.exit(f"mapping mismatch in {run} {sess}")
          # tail check on the shuffles themselves: fitted tail at the empirical 1-in-100 point
          q99 = np.quantile(sh.astype(np.float64), 0.99, axis=1)
          q998 = np.quantile(sh.astype(np.float64), 0.998, axis=1)
          p99 = zi_gamma_p(sh, q99)
          p998 = zi_gamma_p(sh, q998)
          kinds = []
          for i, j in idx:
              f = base(feats[j])
              kinds.append("circular" if f in CIRC else ("continuous" if f in ("speed",) else "discrete"))
          for kind, a in zip(kinds, p99):
              tail[kind].append(a)
          tail["all"].extend(p99.tolist())
          tail998.extend(p998.tolist())
          # observed value against the null: stored p, refit p on the measure, refit p on its square
          above = (sh.astype(np.float64) >= obs[:, None]).sum(1)
          p_fit = zi_gamma_p(sh, obs)
          p_sq = zi_gamma_p(sh.astype(np.float64) ** 2, obs ** 2)
          for r, (i, j) in enumerate(idx):
              s = st[neurons[i]][feats[j]]
              f = base(feats[j])
              if f in LEFT_OUT or s.get("rval") is None:
                  continue
              sr = s.get("signal_ratio")
              if not (sr is None or sr > 1):
                  continue
              key = (sess, neurons[i], f)
              stored = s.get("pval")
              if stored is not None and stored > 0:
                  agree.append((stored < 0.001) == (p_fit[r] < 0.001))
              if above[r] <= 4 and stored is not None and stored < 0.001:
                  pairs_paper.add(key)
              if above[r] <= 4 and p_sq[r] < 0.001:
                  pairs_sq.add(key)
              if (above[r] + 1) / (n + 1) < 0.001:
                  pairs_rank.add(key)
      res[run] = {
          "tail_at_empirical_1pct": {k: [round(float(np.median(v)), 4), round(float(np.percentile(v, 25)), 4),
                                          round(float(np.percentile(v, 75)), 4), len(v)] for k, v in tail.items() if v},
          "tail_at_empirical_0.2pct_median": round(float(np.median(tail998)), 5),
          "refit_reproduces_stored_decision_share": round(float(np.mean(agree)), 4),
          "pairs_paper_criterion_stored": len(pairs_paper),
          "neurons_paper_criterion_stored": len({k[:2] for k in pairs_paper}),
          "pairs_gamma_on_square": len(pairs_sq),
          "neurons_gamma_on_square": len({k[:2] for k in pairs_sq}),
          "pairs_rank": len(pairs_rank),
          "neurons_rank": len({k[:2] for k in pairs_rank}),
          "by_feature_gamma_on_square": {},
      }
      by = {}
      for k in pairs_sq:
          by[k[2]] = by.get(k[2], 0) + 1
      res[run]["by_feature_gamma_on_square"] = dict(sorted(by.items()))
      by = {}
      for k in pairs_paper:
          by[k[2]] = by.get(k[2], 0) + 1
      res[run]["by_feature_paper_stored"] = dict(sorted(by.items()))
      print(run, json.dumps({k: v for k, v in res[run].items() if not k.startswith("by_feature")}), flush=True)

  out = P.NUMBERS_DIR / "gamma_tail_mi_corr.json"
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(res, indent=1), encoding="utf-8")
  print("written", out)


if __name__ == "__main__":
    main()
