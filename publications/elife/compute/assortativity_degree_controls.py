# -*- coding: utf-8 -*-
"""Everything that can be measured on the cached real / shuffled matrices of Fig. 4C:
edge counts, degree against activity, and assortativity by selectivity label under
three nulls (plain label permutation, permutation within groups of similar degree,
degree-preserving rewiring)."""
import os
import sys

import numpy as np
from scipy.stats import spearmanr, mannwhitneyu, skew

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assortativity_common import *

rng = np.random.default_rng(42)
labels = load_labels()
sel = labels != "non-sel."
print("population labels:", dict(zip(*np.unique(labels, return_counts=True))), "selective frac %.3f" % sel.mean())

cal = np.load(SYNC)["Calcium"]
T = cal.shape[1]
iu = np.triu_indices(N, 1)
mats = {t: load_cached(t) for t in ("real", "shuffled")}

print("\n=== Q1: edge counts, p-value calibration ===")
for t, (sim, sig, pv) in mats.items():
    e = (sig[iu] > 0)
    p = pv[iu]
    print(f"[{t}] pairs {len(p)}, edges {e.sum()} ({e.mean():.5f}); expected at p<0.001 if calibrated: {len(p)*0.001:.0f}")
    print(f"   symmetric sig: {np.allclose(sig, sig.T)}; pval finite {np.isfinite(p).mean():.3f}; "
          f"pval==1 {np.mean(p==1):.3f}; p<0.05 {np.mean(p<0.05):.4f}; p<0.01 {np.mean(p<0.01):.4f}; p<0.001 {np.mean(p<0.001):.5f}")
    print(f"   sim of edges: median {np.median(sim[iu][e]):.4f}; sim of non-edges median {np.median(sim[iu][~e]):.5f}")
er, es = mats["real"][1][iu] > 0, mats["shuffled"][1][iu] > 0
both = (er & es).sum()
print(f"edges in both: {both}; expected by chance {er.sum()*es.sum()/len(er):.1f}; "
      f"share of shuffled edges that are real edges {both/es.sum():.3f} vs real density {er.mean():.4f}")
print("spearman(real sim, shuffled sim) over pairs: %.3f" % spearmanr(mats["real"][0][iu], mats["shuffled"][0][iu])[0])

# per-neuron activity descriptors from the raw traces
z = (cal - cal.mean(1, keepdims=True)) / cal.std(1, keepdims=True)
desc = {
    "variance": cal.var(1),
    "skewness": skew(cal, axis=1),
    "frac_active(>2sd)": (z > 2).mean(1),
    "n_events(upcross 2sd)": ((z[:, 1:] > 2) & (z[:, :-1] <= 2)).sum(1),
    "acf_lag150": np.array([np.corrcoef(r[:-150], r[150:])[0, 1] for r in cal]),
    "halves_mean_diff": np.abs(z[:, :T // 2].mean(1) - z[:, T // 2:].mean(1)),
}
np.savez(elife_paths.out_path("neuron_descriptors.npz"),
         labels=labels, **{k.split("(")[0]: v for k, v in desc.items()})

for t, (sim, sig, pv) in mats.items():
    A = sig > 0
    deg = A.sum(1)
    G = giant(A)
    nodes = np.array(sorted(G.nodes))
    edges = np.array(G.edges)
    print(f"\n=========== [{t}] giant {len(nodes)} nodes, {len(edges)} edges; all edges {A.sum()//2}; isolates {np.sum(deg==0)}")

    print("--- Q2: selective vs non-selective")
    print(f"   selective fraction: population {sel.mean():.3f}; giant {sel[nodes].mean():.3f}; non-isolated {sel[deg>0].mean():.3f}")
    print(f"   mean degree (all 695): selective {deg[sel].mean():.2f}, non-sel {deg[~sel].mean():.2f}; "
          f"MWU p={mannwhitneyu(deg[sel], deg[~sel]).pvalue:.2g}")
    for lab in np.unique(labels):
        m = labels == lab
        print(f"      {lab:10s} n={m.sum():3d} in giant {np.isin(np.where(m)[0], nodes).mean():.2f} mean deg {deg[m].mean():.2f}")

    print("--- Q3: degree vs activity (spearman over 695 neurons); selective vs non-sel medians")
    for k, v in desc.items():
        r, p = spearmanr(deg, v)
        print(f"   {k:24s} rho(deg)={r:+.3f} p={p:.1e} | median sel {np.median(v[sel]):.4g} non-sel {np.median(v[~sel]):.4g} "
              f"MWU p={mannwhitneyu(v[sel], v[~sel]).pvalue:.1e}")

    print("--- assortativity under different nulls")
    strata = degree_strata(deg, nodes)
    print("   degree strata sizes:", np.bincount(strata))
    for name, lab in (("8 labels", labels), ("binary sel/non-sel", np.where(sel, "sel", "non"))):
        codes, k = encode(lab)
        obs = assort(edges, codes, k)
        n1 = perm_null(edges, codes, k, nodes, rng, 10000)
        n2 = perm_null(edges, codes, k, nodes, rng, 10000, strata=strata)
        n3 = rewire_null(G, codes, k, rng, 1000 if t == 'shuffled' else 300)
        print(f"   {name:20s} r={obs:.3f} | plain perm: null {n1.mean():+.3f} sd {n1.std():.3f} p={pval(n1,obs):.4f}"
              f" | degree-stratified perm: null {n2.mean():+.3f} sd {n2.std():.3f} p={pval(n2,obs):.4f}"
              f" | rewiring: null {n3.mean():+.3f} sd {n3.std():.3f} p={pval(n3,obs):.4f}")
    # selective-only subgraph
    Gs = G.subgraph([i for i in nodes if sel[i]]).copy()
    es_ = np.array(Gs.edges)
    ns = np.array(sorted(Gs.nodes))
    codes, k = encode(labels)
    obs = assort(es_, codes, k)
    degs = np.zeros(N, int)
    for u, d in Gs.degree:
        degs[u] = d
    n1 = perm_null(es_, codes, k, ns, rng, 10000)
    n2 = perm_null(es_, codes, k, ns, rng, 10000, strata=degree_strata(degs, ns, 4))
    Gs0 = Gs.copy(); Gs0.remove_nodes_from(list(nx.isolates(Gs0)))
    n3 = rewire_null(Gs0, codes, k, rng, 1000) if Gs0.number_of_edges() > 3 else np.array([np.nan])
    print(f"   selective-only ({len(ns)} nodes, {len(es_)} edges) r={obs:.3f} | plain p={pval(n1,obs):.4f} (null {n1.mean():+.3f} sd {n1.std():.3f})"
          f" | stratified p={pval(n2,obs):.4f} | rewiring p={pval(n3,obs):.4f} (null {n3.mean():+.3f})")
    # which label pairs carry the excess
    codes, k = encode(labels)
    names = np.unique(labels)
    e = np.zeros((k, k)); np.add.at(e, (codes[edges[:, 0]], codes[edges[:, 1]]), 1); e = e + e.T
    a = e.sum(0) / e.sum()
    print("   same-label edges observed / expected from label degrees (config-model expectation):")
    for i, nm in enumerate(names):
        print(f"      {nm:10s} obs {e[i,i]/2:6.0f}  exp {a[i]*a[i]*e.sum()/2:7.1f}")
