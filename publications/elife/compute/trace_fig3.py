# -*- coding: utf-8 -*-
"""Figure 3 text numbers from the INTENSE run (type-based representations,
zone rule place_field).

Reads the significance tables after disentanglement with the loader of
fig3/panel_e.py. The merged labels place-walls and place-centertrue are
features of their own: they add neither to place nor to the zone, while the
neuron stays in the count of selective neurons. Procrustes numbers are printed
when the embedding cache of Figure 3A exists.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import procrustes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import elife_paths
sys.path.insert(0, os.path.join(HERE, "..", "fig3"))
from panel_e import load_selectivity_table  # noqa: E402

NEW_FEATURES = ["place_quad", "headdirection_harm2", "bodydirection_harm2", "speed_quad", "locomotion", "walk",
                "rest", "freezing", "rear", "walls", "centertrue"]
# the cohort table of the multisession analysis gives the neuron count of every session
sess = pd.read_csv(os.path.join(elife_paths.CACHE_DIR, "fig4_multisession_sessions.csv"))
N_TOTAL = int(sess.n_neurons.sum())


def report(df, features, tag):
    print(f"\n===== {tag} =====")
    print("rows", len(df), "features", sorted(df["feature"].unique()))
    print("mice", df["mouse"].nunique(), "sessions", df.groupby(["mouse", "session"]).ngroups)
    sel = df.groupby(["mouse", "session"])["neuron_idx"].nunique()
    cnt = df.groupby(["mouse", "session", "feature"]).size().unstack(fill_value=0).reindex(columns=features, fill_value=0)
    pct = 100 * cnt.div(sel, axis=0)
    pm = pct.groupby("mouse").mean()
    print("selective neurons per session, per-mouse mean", sel.groupby("mouse").mean().round(1).to_dict())
    print("pairs per feature (all sessions)", cnt.sum().to_dict())
    nz = cnt.groupby("mouse").sum() > 0
    out = pd.DataFrame({"median": pm.median(), "min": pm.min(), "max": pm.max(), "n_mice_nonzero": nz.sum()})
    print(out.sort_values("median", ascending=False).round(2))
    print("mice lacking a feature:", {m: [f for f in features if not nz.loc[m, f]] for m in nz.index if not nz.loc[m].all()})
    ranks = pm.rank(axis=1)
    m, n = ranks.shape
    S = ((ranks.sum(0) - m * (n + 1) / 2) ** 2).sum()
    W = 12 * S / (m ** 2 * (n ** 3 - n))
    T = sum(((np.unique(r, return_counts=True)[1]) ** 3 - np.unique(r, return_counts=True)[1]).sum() for r in ranks.values)
    Wt = 12 * S / (m ** 2 * (n ** 3 - n) - m * T)
    for name, w in [("no tie correction", W), ("tie-corrected", Wt)]:
        chi = m * (n - 1) * w
        print(f"Kendall W {name}: {w:.4f} chi2 {chi:.2f} df {n - 1} p {stats.chi2.sf(chi, n - 1):.2e}")
    fpn = df.groupby(["mouse", "session", "neuron_idx"])["feature"].nunique().reset_index(name="k")
    g = fpn.groupby(["mouse", "session"])["k"]
    st = pd.DataFrame({"p1": g.apply(lambda s: 100 * (s == 1).mean()), "p2": g.apply(lambda s: 100 * (s == 2).mean()),
                       "p3": g.apply(lambda s: 100 * (s >= 3).mean())})
    t = stats.t.ppf(0.975, len(st) - 1)
    for col in st:
        mu, h = st[col].mean(), t * st[col].sem()
        print(f"{col}: mean {mu:.2f} CI [{mu - h:.2f}, {mu + h:.2f}] n={len(st)}")
    print("pooled: 1 feat %.2f, 2 %.2f, 3+ %.2f, >=2 %.2f" % tuple(
        100 * x for x in [(fpn.k == 1).mean(), (fpn.k == 2).mean(), (fpn.k >= 3).mean(), (fpn.k >= 2).mean()]))
    print("multi (>=2) session-mean %.2f" % (st.p2 + st.p3).mean())
    print("selective fraction: %d / %d = %.2f%%" % (sel.sum(), N_TOTAL, 100 * sel.sum() / N_TOTAL))


new = load_selectivity_table()
report(new, NEW_FEATURES, "tables after disentanglement")
raw_new = load_selectivity_table(elife_paths.RAW_TABLES_DIR)
print("\nraw significance tables: pairs", len(raw_new), "selective neurons",
      raw_new.groupby(["mouse", "session"])["neuron_idx"].nunique().sum(),
      "| after disentanglement: pairs", len(new), "selective neurons",
      new.groupby(["mouse", "session"])["neuron_idx"].nunique().sum())
# ---- merged labels place-<zone>: how many, and what the other two ways of counting them would give ----
merged = new[new.feature.str.startswith("place-")]
print("merged labels:", merged.feature.value_counts().to_dict(), "| neurons", len(merged),
      "| sessions with any", merged.groupby(["mouse", "session"]).ngroups, "| mice", merged.mouse.nunique())
for tag, fold in [("merged label counted as the zone", lambda f: f[6:]), ("merged label counted as place", lambda f: "place_quad")]:
    alt = new.copy()
    alt["feature"] = [fold(f) if f.startswith("place-") else f for f in alt.feature]
    report(alt, NEW_FEATURES, "ALTERNATIVE, not used in the text: " + tag)
print()
print("sessions", len(sess), "neurons total", N_TOTAL, "mean", round(sess.n_neurons.mean(), 1), "range",
      sess.n_neurons.min(), sess.n_neurons.max())

# ---- Procrustes / PROTEST (recipe of compute_procrustes_protest.py) ----
z = np.load(os.path.join(elife_paths.SYNC_DIR, "FOF_F28_1D_aligned.npz"))
pos = np.column_stack([z["x"].ravel()[::5], z["y"].ravel()[::5]])
for tag, path in [("embedding cache", os.path.join(elife_paths.CACHE_DIR, "FOF_F28_1D_umap_cache.npz"))]:
    if not os.path.exists(path):
        print(f"\n[{tag}] not computed yet: {path}")
        continue
    c = np.load(path)
    d_real = procrustes(pos, c["coords_real"].T)[2]
    d_shuf = procrustes(pos, c["coords_shuf"].T)[2]
    rng = np.random.RandomState(42)
    null = np.array([procrustes(pos[rng.permutation(len(pos))], c["coords_real"].T)[2] for _ in range(999)])
    p = (np.sum(null <= d_real) + 1) / 1000
    print(f"\n[{tag}] samples {len(pos)}: disparity real {d_real:.4f} shuffled {d_shuf:.4f} "
          f"null {null.mean():.5f} +/- {null.std():.5f} p {p:.4f}")
