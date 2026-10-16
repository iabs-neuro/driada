"""
Fig 1 subfigure — Clustered network with 3 communities.
Spring layout, colored communities, subtle edges.

Usage:
    python fig1_network.py          # saves fig1_network.pdf + .png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import elife_paths

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

rng = np.random.default_rng(7)

# ── Community sizes ───────────────────────────────────────────────
sizes = [80, 50, 30]  # 3 communities, different sizes

# ── Build stochastic block model ──────────────────────────────────
# High intra-community, low inter-community edge probability
p_intra = 0.35
p_inter = 0.015
probs = [
    [p_intra, p_inter, p_inter],
    [p_inter, p_intra, p_inter],
    [p_inter, p_inter, p_intra],
]

G = nx.stochastic_block_model(sizes, probs, seed=7)

# ── Community assignments ─────────────────────────────────────────
community = []
for i, s in enumerate(sizes):
    community.extend([i] * s)

COLORS = ["#E53935", "#1E88E5", "#43A047"]  # red, blue, green
node_colors = [COLORS[community[n]] for n in G.nodes()]

# Node sizes proportional to degree, with community base
degrees = dict(G.degree())
node_sizes = [20 + 3 * degrees[n] for n in G.nodes()]

# ── Spring layout with stronger attraction within communities ─────
pos = nx.spring_layout(G, k=0.12, iterations=120, seed=42)

# ── Figure ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

# Draw edges first (very subtle)
nx.draw_networkx_edges(G, pos, ax=ax,
                       edge_color="#999999", width=0.3, alpha=0.5)

# Draw nodes
nx.draw_networkx_nodes(G, pos, ax=ax,
                       node_color=node_colors, node_size=node_sizes,
                       edgecolors="white", linewidths=0.4, alpha=0.9)

ax.set_axis_off()
plt.tight_layout(pad=0.5)

# ── Save ──────────────────────────────────────────────────────────
fig.savefig(elife_paths.out_path("fig1_network.pdf"), bbox_inches="tight", dpi=300)
fig.savefig(elife_paths.out_path("fig1_network.png"), bbox_inches="tight", dpi=300)
print("Saved fig1_network.pdf and .png")
