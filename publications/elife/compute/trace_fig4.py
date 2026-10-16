# -*- coding: utf-8 -*-
"""Figure 4 text numbers: example session (panel A/B cache) and its row in the cohort table."""
import os
import sys

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import elife_paths

COHORT = os.path.join(elife_paths.CACHE_DIR, "fig4_multisession_sessions.csv")
c = np.load(os.path.join(elife_paths.CACHE_DIR, "fig4_ab_cache_nn40.npz"))
keys = [str(k) for k in c["metric_keys"]]
print("n_neurons", int(c["n_neurons"]), "filtered", len(c["filt_idx"]), "random subsets", len(c["m_rand"]),
      "shuffles", c["n_shuffles"].tolist(), "seed", int(c["seed"]))
for k in ("r2cv", "dcorr"):
    i = keys.index(k)
    print(f"{k}: all {c['m_all'][i]:.4f} filt {c['m_filt'][i]:.4f} rand {c['m_rand'][:, i].mean():.4f} "
          f"+- {c['m_rand'][:, i].std(ddof=1):.4f} (ddof=0 {c['m_rand'][:, i].std():.4f}) "
          f"shuf_all {c['m_shuf_all'][:, i].mean():.4f} shuf_filt {c['m_shuf_filt'][:, i].mean():.4f}")
print("Dim 1:", len(c["comp_sel_0"]), "Dim 2:", len(c["comp_sel_1"]), "informative raw:", len(c["inf_raw"]),
      "Holm:", len(c["inf_holm"]), "frames lost all/filt:", int(c["x_ds"].size - c["kept_all"].size), int(c["x_ds"].size - c["kept_filt"].size))
s = pd.read_csv(COHORT)
r = s[s.session == "FOF_F28_1D"].iloc[0]
print("cohort row FOF_F28_1D:", {k: round(float(r[k]), 4) for k in ["n_filtered", "r2cv_all", "r2cv_filt", "r2cv_rand_mean", "r2cv_rand_sd",
      "r2cv_shuf_all", "r2cv_shuf_filt", "dcorr_all", "dcorr_filt", "dcorr_rand_mean", "dcorr_shuf_all", "dcorr_shuf_filt", "n_inf_raw"]})
ok = s[s.status == "ok"]
print("cohort: ok sessions", len(ok), "of", len(s), "mice", ok.mouse.nunique())
