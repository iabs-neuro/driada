# -*- coding: utf-8 -*-
"""Shared helpers: labels, graphs, assortativity with several nulls."""
import json
import os
import sys

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import elife_paths

RES = os.path.join(elife_paths.INTENSE_DIR, "FOF_F28_1D_results.npz")
SYNC = os.path.join(elife_paths.SYNC_DIR, "FOF_F28_1D_aligned.npz")
N = 695

FEATURE_GROUPS = {
    "place": ["place_quad", "place"], "HD": ["bodydirection_harm2", "headdirection_harm2"],
    "speed": ["speed_quad", "speed"], "locomotion": ["locomotion", "walk"],
    "rest": ["rest", "freezing"], "rear": ["rear"],
    "spatial": ["walls", "centertrue"],
}


def _is_sig(e):
    return e.get("stage2", e.get("stage1", False))


def primary_selectivity(i, sig, stats):
    """Same rule as multisession_fig4_controls.primary_selectivity."""
    c = str(i)
    if c not in sig:
        return "non-sel."
    best, best_mi = None, -np.inf
    for g, feats in FEATURE_GROUPS.items():
        for f in feats:
            if f in sig[c] and _is_sig(sig[c][f]) and stats[c][f]["me"] > best_mi:
                best_mi, best = stats[c][f]["me"], g
    return best or "non-sel."


def load_labels():
    res = np.load(RES, allow_pickle=True)
    sig = json.loads(str(res["_significance_json"][0]))
    stats = json.loads(str(res["_stats_json"][0]))
    return np.array([primary_selectivity(i, sig, stats) for i in range(N)])


def load_cached(tag):
    d = np.load(os.path.join(elife_paths.CACHE_DIR, f"network_{tag}_FOF_F28_1D.npz"))
    return d["sim_matrix"], d["sig_matrix"], d["pval_matrix"]


def giant(A):
    G = nx.from_numpy_array((A > 0).astype(int))
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def assort(edges, codes, k):
    """Attribute assortativity (Newman) from an edge array (m,2) and integer codes."""
    e = np.zeros((k, k))
    np.add.at(e, (codes[edges[:, 0]], codes[edges[:, 1]]), 1)
    e = e + e.T
    e /= e.sum()
    a = e.sum(0)
    s = (a * a).sum()
    return (np.trace(e) - s) / (1 - s)


def encode(labels):
    u, codes = np.unique(labels, return_inverse=True)
    return codes, len(u)


def perm_null(edges, codes, k, nodes, rng, n_perm, strata=None):
    """Label permutation over `nodes`; with `strata`, labels move only within a stratum."""
    out = np.empty(n_perm)
    c = codes.copy()
    groups = [nodes] if strata is None else [nodes[strata == s] for s in np.unique(strata)]
    for i in range(n_perm):
        for g in groups:
            c[g] = codes[rng.permutation(g)]
        out[i] = assort(edges, c, k)
    return out


def rewire_null(G, codes, k, rng, n_rep, swaps_per_edge=5):
    """Degree-preserving rewiring (double edge swaps without self-loops or multi-edges), labels fixed."""
    E = [tuple(e) for e in G.edges]
    m = len(E)
    S = set(frozenset(e) for e in E)
    out = np.empty(n_rep)
    for i in range(n_rep):
        idx = rng.integers(0, m, size=(swaps_per_edge * m, 2))
        flip = rng.random(swaps_per_edge * m) < 0.5
        for (x, y), f in zip(idx, flip):
            a, b = E[x]
            c, d = E[y]
            if f:
                c, d = d, c
            if len({a, b, c, d}) < 4:
                continue
            n1, n2 = frozenset((a, d)), frozenset((c, b))
            if n1 in S or n2 in S:
                continue
            S.discard(frozenset((a, b))); S.discard(frozenset((c, d)))
            S.add(n1); S.add(n2)
            E[x], E[y] = (a, d), (c, b)
        out[i] = assort(np.array(E), codes, k)
    return out


def pval(null, obs):
    return (1 + (null >= obs).sum()) / (1 + len(null))


def degree_strata(deg, nodes, n_bins=6):
    """Quantile bins of degree over the given nodes (ties kept together)."""
    d = deg[nodes]
    qs = np.unique(np.quantile(d, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.searchsorted(qs, d, side="left")
