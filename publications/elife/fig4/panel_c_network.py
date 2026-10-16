"""
Panel C — Functional networks (real vs shuffled) and spectral density.

Provides:
  - load_network_data() -> dict with graphs, positions, spectra, selectivity
  - draw_real_network(ax, data) -> draws selectivity-colored network
  - draw_shuffled_network(ax, data) -> draws single-color network
  - draw_spectrum(ax, spectrum, color, title, style, panel_size_cm) -> histogram
  - draw_network_legend(fig, ...) -> selectivity legend at top of network section
"""

import os
import sys
import json
import time

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import elife_paths

from driada.network import Network

SESSION = "FOF_F28_1D"
INTENSE_DIR = elife_paths.INTENSE_DIR

# Panel C colors
COLOR_REAL = "#4C72B0"
COLOR_SHUF = "#DD8452"

# Selectivity groups and colors (for real network coloring)
FEATURE_GROUPS = {
    "place":      ["place_quad", "place"],
    "HD":         ["bodydirection_harm2", "headdirection_harm2"],
    "speed":      ["speed_quad", "speed"],
    "locomotion": ["locomotion", "walk"],
    "rest":       ["rest", "freezing"],
    "rear":       ["rear"],
    "spatial":    ["walls", "centertrue"],
}

SEL_COLORS = {
    "place":      "#E24A33",   # red
    "HD":         "#348ABD",   # blue
    "speed":      "#FF7F0E",   # orange
    "locomotion": "#FBC15E",   # amber
    "rest":       "#17BECF",   # teal
    "rear":       "#E377C2",   # pink
    "spatial":    "#2CA02C",   # green
    "non-sel.":   "#BBBBBB",   # gray
}

# Draw order: background first, then selective on top
SEL_DRAW_ORDER = ["non-sel.", "spatial", "rear", "rest",
                  "locomotion", "speed", "HD", "place"]

# Cell-cell significance matrices (compute/compute_cell_cell_network.py) and the layout cache
_DATA_DIR = elife_paths.CACHE_DIR


def _get_primary_selectivity(neuron_idx, sig_json, stats_json):
    """Return the selectivity group with highest MI among significant features."""
    cidx_str = str(neuron_idx)
    if cidx_str not in sig_json:
        return "non-sel."
    best_group = None
    best_mi = -np.inf
    for group_name, feat_names in FEATURE_GROUPS.items():
        for fname in feat_names:
            if fname not in sig_json[cidx_str]:
                continue
            entry = sig_json[cidx_str][fname]
            is_sig = entry.get("stage2", entry.get("stage1", False))
            if not is_sig:
                continue
            mi = stats_json[cidx_str][fname]["me"]
            if mi > best_mi:
                best_mi = mi
                best_group = group_name
    return best_group if best_group is not None else "non-sel."


def load_network_data():
    """Load pre-computed networks, build graphs, compute spectra and layouts.

    Returns
    -------
    dict with keys:
        G_real, G_shuf : networkx.Graph (trimmed)
        pos_real, pos_shuf : dict of node positions
        spec_real, spec_shuf : ndarray of eigenvalues
        node_selectivity : dict node -> selectivity group (real network only)
        net_real, net_shuf : Network objects
    """
    print("  [Panel C] Loading pre-computed cell-cell networks...")
    t0 = time.perf_counter()

    real = np.load(os.path.join(_DATA_DIR, f"network_real_{SESSION}.npz"))
    shuf = np.load(os.path.join(_DATA_DIR, f"network_shuffled_{SESSION}.npz"))

    net_real = Network(adj=csr_matrix(real["sig_matrix"]),
                       preprocessing="giant_cc", name="real", verbose=False)
    net_shuf = Network(adj=csr_matrix(shuf["sig_matrix"]),
                       preprocessing="giant_cc", name="shuffled", verbose=False)

    # Compute spectra
    net_real.diagonalize(mode="nlap")
    spec_real = net_real.get_spectrum("nlap")
    net_shuf.diagonalize(mode="nlap")
    spec_shuf = net_shuf.get_spectrum("nlap")

    # Build graphs and compute layouts
    G_real = nx.Graph(net_real.graph)
    G_shuf = nx.Graph(net_shuf.graph)

    # Trim degree-1 nodes from real network
    deg1 = [n for n, d in G_real.degree() if d <= 1]
    G_real.remove_nodes_from(deg1)

    # Community-aware weighted spring layout for both (cached)
    LAYOUT_K = 16.0
    LAYOUT_RES = 1.0
    layout_cache = os.path.join(_DATA_DIR, f"network_layout_{SESSION}_k{LAYOUT_K:.0f}.npz")

    if os.path.exists(layout_cache):
        lc = np.load(layout_cache, allow_pickle=True)
        pos_real = lc["pos_real"].item()
        pos_shuf = lc["pos_shuf"].item()
    else:
        def _community_layout(G):
            comms = nx.community.louvain_communities(G, seed=42, resolution=1.0)
            node_to_comm = {}
            for cid, members in enumerate(comms):
                for node in members:
                    node_to_comm[node] = cid

            G_w = G.copy()
            for u, v in G_w.edges():
                G_w[u][v]["weight"] = 1.0 if node_to_comm[u] == node_to_comm[v] else 0.01

            pos = nx.spring_layout(
                G_w, seed=42,
                k=LAYOUT_K / np.sqrt(G.number_of_nodes()),
                iterations=200, weight="weight",
            )
            return pos

        pos_real = _community_layout(G_real)
        pos_shuf = _community_layout(G_shuf)
        np.savez_compressed(layout_cache,
                            pos_real=pos_real, pos_shuf=pos_shuf)

    # Load selectivity for real network nodes
    results = np.load(os.path.join(INTENSE_DIR, f"{SESSION}_results.npz"),
                      allow_pickle=True)
    sig_json = json.loads(str(results["_significance_json"][0]))
    stats_json = json.loads(str(results["_stats_json"][0]))

    node_selectivity = {n: _get_primary_selectivity(n, sig_json, stats_json)
                        for n in G_real.nodes()}

    print(f"  [Panel C] Real: {G_real.number_of_nodes()} nodes, "
          f"{G_real.number_of_edges()} edges | "
          f"Shuffled: {G_shuf.number_of_nodes()} nodes, "
          f"{G_shuf.number_of_edges()} edges | "
          f"{time.perf_counter() - t0:.1f}s")

    return {
        "G_real": G_real, "G_shuf": G_shuf,
        "pos_real": pos_real, "pos_shuf": pos_shuf,
        "spec_real": spec_real, "spec_shuf": spec_shuf,
        "node_selectivity": node_selectivity,
        "net_real": net_real, "net_shuf": net_shuf,
    }


def draw_real_network(ax, data):
    """Draw the real network colored by selectivity type."""
    G = data["G_real"]
    pos = data["pos_real"]
    node_sel = data["node_selectivity"]
    node_list = list(G.nodes())

    nx.draw_networkx_edges(G, pos, ax=ax,
                           alpha=0.08, width=0.4, edge_color="gray")

    for sel_type in SEL_DRAW_ORDER:
        nodes = [n for n in node_list if node_sel.get(n) == sel_type]
        if not nodes:
            continue
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes, ax=ax,
            node_size=14 if sel_type != "non-sel." else 10,
            node_color=SEL_COLORS[sel_type],
            alpha=0.9 if sel_type != "non-sel." else 0.5,
            linewidths=0,
        )

    # Tighten axes limits
    pos_arr = np.array(list(pos.values()))
    margin = 0.05
    x_range = pos_arr[:, 0].max() - pos_arr[:, 0].min()
    y_range = pos_arr[:, 1].max() - pos_arr[:, 1].min()
    ax.set_xlim(pos_arr[:, 0].min() - margin * x_range,
                pos_arr[:, 0].max() + margin * x_range)
    ax.set_ylim(pos_arr[:, 1].min() - margin * y_range,
                pos_arr[:, 1].max() + margin * y_range)

    ax.axis("off")


def draw_shuffled_network(ax, data):
    """Draw the shuffled network in a single color."""
    G = data["G_shuf"]
    pos = data["pos_shuf"]

    nx.draw_networkx_edges(G, pos, ax=ax,
                           alpha=0.15, width=0.4, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_size=12, node_color=COLOR_SHUF,
                           alpha=0.85, linewidths=0)

    ax.axis("off")


def draw_spectrum(ax, spectrum, color, title, style, panel_size_cm):
    """Draw normalized Laplacian spectral density histogram."""
    bins = np.linspace(0, 2, 50)
    ax.hist(spectrum, bins=bins, color=color, alpha=0.8,
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.set_xlim(0, 2)
    style.apply_to_axes(ax, panel_size_cm, "cm")


def draw_network_legend(fig, bbox_anchor=(0.5, 0.67), loc="upper center",
                        fontsize=6, ncol=4):
    """Add selectivity legend to figure.

    Shared between the embedding pies (B) and the networks (C).

    Parameters
    ----------
    fig : Figure
    bbox_anchor : tuple
        (x, y) in figure coordinates for legend placement.
    loc : str
        Legend location anchor point.
    fontsize : float
        Legend text size; handle sizes scale with it.
    ncol : int
        Number of legend columns.
    """
    from matplotlib.patches import Patch
    # Displayed labels: "locomotion" group is shown as "run"
    label_map = {"locomotion": "run"}
    legend_handles = [Patch(facecolor=SEL_COLORS[g], label=label_map.get(g, g))
                      for g in ["place", "HD", "speed", "locomotion",
                                "rest", "rear", "spatial", "non-sel."]]
    fig.legend(handles=legend_handles, fontsize=fontsize, ncol=ncol,
               loc=loc, bbox_to_anchor=bbox_anchor,
               framealpha=0.85, handlelength=1.5, handleheight=1.1,
               columnspacing=1.3, labelspacing=0.5, borderpad=0.6,
               handletextpad=0.5)
