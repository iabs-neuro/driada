# -*- coding: utf-8 -*-
"""Does primary selectivity type predict functional-network links (Fig. 4C)?

Attribute assortativity of the real and shuffled cell-cell significance networks
(FOF_F28_1D) by primary INTENSE selectivity label, with a label-permutation null
(labels permuted over the nodes of the giant component, graph fixed).
"""
import json
import os
import sys

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import elife_paths
from multisession_fig4_controls import primary_selectivity  # by_type feature names, same labelling as the reverse analysis

RES = os.path.join(elife_paths.INTENSE_DIR, "FOF_F28_1D_results.npz")
N_PERM = 10000

res = np.load(RES, allow_pickle=True)
sig = json.loads(str(res["_significance_json"][0]))
stats = json.loads(str(res["_stats_json"][0]))
rng = np.random.default_rng(42)

for tag in ("real", "shuffled"):
    A = np.load(os.path.join(elife_paths.CACHE_DIR, f"network_{tag}_FOF_F28_1D.npz"))["sig_matrix"]
    G = nx.from_numpy_array((A > 0).astype(int))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    nodes = list(G.nodes)
    labels = {i: primary_selectivity(i, sig, stats) for i in nodes}
    nx.set_node_attributes(G, labels, "lab")
    obs = nx.attribute_assortativity_coefficient(G, "lab")
    # same-label edge fraction versus expectation
    same = np.mean([labels[u] == labels[v] for u, v in G.edges])
    vals = list(labels.values())
    null = np.empty(N_PERM)
    null_same = np.empty(N_PERM)
    for k in range(N_PERM):
        perm = dict(zip(nodes, rng.permutation(vals)))
        nx.set_node_attributes(G, perm, "lab")
        null[k] = nx.attribute_assortativity_coefficient(G, "lab")
        null_same[k] = np.mean([perm[u] == perm[v] for u, v in G.edges])
    p = (1 + (null >= obs).sum()) / (1 + N_PERM)
    # the same test with non-selective neurons removed (they dominate the label counts)
    Gs = G.subgraph([i for i in nodes if labels[i] != "non-sel."]).copy()
    nx.set_node_attributes(Gs, {i: labels[i] for i in Gs}, "lab")
    obs_s = nx.attribute_assortativity_coefficient(Gs, "lab") if Gs.number_of_edges() else np.nan
    vs = [labels[i] for i in Gs]
    null_s = []
    for k in range(2000):
        nx.set_node_attributes(Gs, dict(zip(Gs.nodes, rng.permutation(vs))), "lab")
        null_s.append(nx.attribute_assortativity_coefficient(Gs, "lab"))
    p_s = (1 + (np.array(null_s) >= obs_s).sum()) / (1 + len(null_s))
    counts = {l: vals.count(l) for l in sorted(set(vals))}
    print(f"[{tag}] giant component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges; labels {counts}")
    print(f"  assortativity r = {obs:.3f}; null mean {null.mean():.3f} sd {null.std():.3f}; p = {p:.4f}")
    print(f"  same-label edges {same:.3f} vs null {null_same.mean():.3f}")
    print(f"  selective-only subgraph: {Gs.number_of_nodes()} nodes, {Gs.number_of_edges()} edges; "
          f"r = {obs_s:.3f}, null mean {np.nanmean(null_s):.3f}, p = {p_s:.4f}")
