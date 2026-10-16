"""
Fig 2 — Synthetic Validation (5 panels: A–E).

Layout:
  Row 1:  A  (example neurons: event trace + density + HD polar)
  Row 2:  B  (selectivity heatmap, narrow) |  C  (Isomap manifold, large)
  Row 3:  D  (network graph)               |  E  (adjacency matrix)

Usage:
    python publications/elife/fig2/fig2_synthetic.py

Reads fig2_*.npz from the cache directory (written by fig2_data.py)
Outputs: fig2.pdf + .png in the output directory (see elife_paths.py)
"""
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from mpl_toolkits.axes_grid1 import make_axes_locatable

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import elife_paths

from driada.utils.publication import StylePreset

CACHE_DIR = elife_paths.CACHE_DIR
DPI = 600

# ── Layout constants (cm) ──────────────────────────────────
CANVAS_W = 18.0     # eLife full-width
ROW1_H = 5.5        # A: example neurons
ROW2_H = 7.0        # B + C side by side
ROW3_H = 7.4        # D + E side by side
VGAP = 0.8          # vertical gap between rows

# Community colors (tab10-based, consistent across panels)
COMM_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def load_cache():
    """Load all cached NPZ files."""
    data = {}
    for name in ["population", "intense", "isomap", "network",
                  "metrics", "example_neurons"]:
        path = os.path.join(CACHE_DIR, f"fig2_{name}.npz")
        data[name] = dict(np.load(path, allow_pickle=True))
    return data


# ── Panel A: Example Selective Neurons ──────────────────────
def create_panel_a(subfig, style, panel_size, data):
    """Panel A: event-selective neuron (trace+density) + HD neuron (polar plot)."""
    gs = subfig.add_gridspec(1, 3, width_ratios=[0.42, 0.20, 0.38],
                              wspace=0.4, left=0.08, right=0.92,
                              top=0.85, bottom=0.18)

    neurons = data["example_neurons"]
    fps = float(neurons["fps"])
    lw = style.get_line_width(panel_size, "cm") * 0.5

    # ── A-left: event neuron trace ──
    ax_trace = subfig.add_subplot(gs[0, 0])
    calcium = neurons["event_calcium"]
    feature = neurons["event_feature"]
    t = np.arange(len(calcium)) / fps
    n_show = int(60 * fps)
    ax_trace.plot(t[:n_show], calcium[:n_show], lw=lw * 2, color="0.3", alpha=0.8)

    active = feature[:n_show] > 0.5
    diff = np.diff(active.astype(int))
    starts = np.where(diff == 1)[0]
    stops = np.where(diff == -1)[0]
    if active[0]:
        starts = np.concatenate([[0], starts])
    if active[-1]:
        stops = np.concatenate([stops, [n_show - 1]])
    for s, e in zip(starts, stops):
        ax_trace.axvspan(t[s], t[e], color="#2ca02c", alpha=0.45)

    ax_trace.set_xlabel("Time (s)")
    ax_trace.set_ylabel("ΔF/F")
    ax_trace.set_title("Event-selective neuron", fontsize=style.base_title_size)
    style.apply_to_axes(ax_trace, panel_size, "cm")

    # ── A-middle: density comparison ──
    ax_dens = subfig.add_subplot(gs[0, 1])
    ca_on = calcium[feature > 0.5]
    ca_off = calcium[feature <= 0.5]
    bins = np.linspace(0, np.percentile(calcium, 99), 40)
    ax_dens.hist(ca_off, bins=bins, orientation="horizontal", density=True,
                 alpha=0.5, color="0.6", label="off")
    ax_dens.hist(ca_on, bins=bins, orientation="horizontal", density=True,
                 alpha=0.6, color="#2ca02c", label="on")
    ax_dens.set_xlabel("Density")
    ax_dens.legend(fontsize=style.base_legend_fontsize, frameon=False)
    ax_dens.set_yticklabels([])
    style.apply_to_axes(ax_dens, panel_size, "cm")

    # ── A-right: HD polar plot ──
    ax_polar = subfig.add_subplot(gs[0, 2], projection="polar")
    angles = neurons["hd_angles"]
    hd_calcium = neurons["hd_calcium"]
    n_bins = 36
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    means = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
        if np.sum(mask) > 0:
            means[i] = np.mean(hd_calcium[mask[:len(hd_calcium)]])

    ax_polar.bar(bin_centers, means, width=2 * np.pi / n_bins,
                 color="#DD8452", alpha=0.7, edgecolor="k", linewidth=0.3)
    ax_polar.set_title("HD-selective neuron", fontsize=style.base_title_size, pad=12)
    ax_polar.set_yticklabels([])
    ax_polar.set_xticks([0, np.pi, 3*np.pi/2])  # drop 90° (top) to avoid title clash
    ax_polar.set_xticklabels(["0°", "180°", "270°"])
    ax_polar.tick_params(labelsize=style.base_tick_labelsize)


# ── Panel B: Selectivity Heatmap ────────────────────────────
def create_panel_b(ax, style, panel_size, data, parent_fig):
    """Panel B: MI heatmap with significant pairs marked."""
    intense = data["intense"]
    mi_mat = intense["mi_mat"]
    sig_mat = intense["sig_mat"]
    feat_ids = list(intense["feat_ids"])

    im = ax.imshow(mi_mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_anchor("N")

    sig_r, sig_c = np.where(sig_mat > 0.5)
    ms = style.get_marker_size(panel_size, "cm") * 0.3
    ax.scatter(sig_c, sig_r, marker=".", s=ms, color="black", alpha=0.6)

    ax.set_xlabel("Feature")
    ax.set_ylabel("Neuron")
    short_labels = [f"E{i+1}" for i in range(len(feat_ids))]
    ax.set_xticks(range(len(feat_ids)))
    ax.set_xticklabels(short_labels, rotation=0, ha="center")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="6%", pad=0.05)
    cbar = parent_fig.colorbar(im, cax=cax)
    cbar.set_label("MI", fontsize=style.base_label_size)
    cbar.ax.tick_params(labelsize=style.base_tick_labelsize)

    style.apply_to_axes(ax, panel_size, "cm")


# ── Panel C: Isomap Circular Manifold ───────────────────────
def create_panel_c(ax, style, panel_size, data, parent_fig):
    """Panel C: Isomap 2D embedding colored by head direction angle."""
    isomap = data["isomap"]
    coords = isomap["coords"]
    hd = isomap["hd_downsampled"]

    ms = style.get_marker_size(panel_size, "cm") * 2.0
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=hd, cmap="hsv",
                    s=ms, alpha=0.6, edgecolors="none", rasterized=True)

    ax.set_xlabel("Isomap 1")
    ax.set_ylabel("Isomap 2")
    # Force square: equal limits on both axes
    all_vals = np.concatenate([coords[:, 0], coords[:, 1]])
    lo, hi = all_vals.min(), all_vals.max()
    margin = (hi - lo) * 0.05
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("N")  # anchor to top so C aligns with B

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = parent_fig.colorbar(sc, cax=cax)
    cbar.set_label("HD angle (rad)", fontsize=style.base_label_size)
    cbar.ax.tick_params(labelsize=style.base_tick_labelsize)

    style.apply_to_axes(ax, panel_size, "cm")


# ── Panel D: Network Graph ──────────────────────────────────
GT_COLOR_MAP = {
    "event_0_cells": "#4a90ff",   # bright blue
    "event_1_cells": "#ffaa00",   # bright orange
    "event_2_cells": "#33cc33",   # bright green
    "event_0_or_1":  "#bb66ff",   # bright purple  (bridge E1–E2)
    "event_0_or_2":  "#00dddd",   # bright cyan    (bridge E1–E3)
    "event_1_or_2":  "#ff4444",   # bright red     (bridge E2–E3)
    "nonselective":  "#cccccc",   # grey
}
GT_SHORT = {
    "event_0_cells": "E1",
    "event_1_cells": "E2",
    "event_2_cells": "E3",
    "event_0_or_1":  "E1\u222aE2",   # E1∪E2
    "event_0_or_2":  "E1\u222aE3",   # E1∪E3
    "event_1_or_2":  "E2\u222aE3",   # E2∪E3
    "nonselective":  "n.s.",
}


def create_panel_d(ax, style, panel_size, data):
    """Panel D: spring-layout network graph colored by ground-truth group."""
    network = data["network"]
    population = data["population"]
    pos_array = network["graph_pos"]
    node_ids = network["graph_nodes"]
    edges = network["graph_edges"]
    group_names = list(population["group_names"])

    node_colors = [GT_COLOR_MAP[group_names[nid]] for nid in node_ids]

    # Draw edges
    edge_lines = []
    for e in edges:
        i_pos = np.where(node_ids == e[0])[0]
        j_pos = np.where(node_ids == e[1])[0]
        if len(i_pos) > 0 and len(j_pos) > 0:
            edge_lines.append([pos_array[i_pos[0]], pos_array[j_pos[0]]])

    if edge_lines:
        lc = mcoll.LineCollection(edge_lines, colors="0.8", linewidths=0.15,
                                  alpha=0.15, zorder=1)
        ax.add_collection(lc)

    ms_net = style.get_marker_size(panel_size, "cm") * 5
    ax.scatter(pos_array[:, 0], pos_array[:, 1], c=node_colors,
               s=ms_net, edgecolors="k", linewidths=0.2, zorder=2,
               alpha=0.85)

    # Legend: pure groups first (E1, E2, E3), then mixed, 2 rows
    from matplotlib.lines import Line2D
    legend_order = [
        "event_0_cells", "event_1_cells", "event_2_cells",
        "event_0_or_1", "event_0_or_2", "event_1_or_2",
    ]
    present_groups = set(group_names[nid] for nid in node_ids)
    handles = []
    for g in legend_order:
        if g in present_groups:
            handles.append(Line2D([0], [0], marker="o", color="none",
                                  markerfacecolor=GT_COLOR_MAP[g],
                                  markeredgecolor="k", markeredgewidth=0.3,
                                  markersize=5, label=GT_SHORT[g]))
    # Reorder for column-major legend layout (matplotlib fills columns first)
    # Input order: E1,E2,E3,E1∪E2,E1∪E3,E2∪E3 (indices 0-5)
    # Column-major 2rows x 3cols maps position [r,c] to index r + c*nrows
    # Want: row0=[E1,E2,E3], row1=[E1∪E2,E1∪E3,E2∪E3]
    # So position [0,0]=E1(0), [1,0]=E1∪E2(3), [0,1]=E2(1), [1,1]=E1∪E3(4), [0,2]=E3(2), [1,2]=E2∪E3(5)
    if len(handles) == 6:
        handles = [handles[0], handles[3], handles[1],
                   handles[4], handles[2], handles[5]]
    ax.legend(handles=handles, fontsize=style.base_legend_fontsize,
              frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, -0.15), ncol=3)

    ax.set_axis_off()

    pad = 0.08
    xmin, xmax = pos_array[:, 0].min(), pos_array[:, 0].max()
    ymin, ymax = pos_array[:, 1].min(), pos_array[:, 1].max()
    dx, dy = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
    ax.set_ylim(ymin - pad * dy, ymax + pad * dy)


# ── Panel E: Community-sorted adjacency matrix ──────────────
def create_panel_e(ax, style, panel_size, data):
    """Panel E: adjacency matrix sorted by detected communities.

    Within each community, pure neurons come first, mixed neurons last
    (at the boundary), so bridging off-diagonal connections are visible.
    """
    network = data["network"]
    population = data["population"]
    sig_mat = network["sig_mat"]
    detected = network["detected_labels"]
    group_names = list(population["group_names"])

    from matplotlib.colors import to_rgba
    from collections import Counter

    n = sig_mat.shape[0]
    labels = detected[:n].copy()

    # Map communities to E1/E2/E3 order by dominant GT group
    gt_order = {"event_0_cells": 0, "event_1_cells": 1, "event_2_cells": 2}
    comm_to_rank = {}
    for c in sorted(set(labels)):
        if c < 0:
            comm_to_rank[c] = 999  # n.s. last
        else:
            mask = labels == c
            groups = Counter([group_names[i] for i in range(n) if mask[i]])
            dominant = groups.most_common(1)[0][0]
            comm_to_rank[c] = gt_order.get(dominant, 99)

    # Sort: by remapped community rank, then pure before mixed
    is_mixed = np.array(["_or_" in group_names[i] for i in range(n)], dtype=int)
    rank_key = np.array([comm_to_rank[labels[i]] for i in range(n)])
    det_order = np.lexsort((is_mixed, rank_key))
    sorted_adj = sig_mat[np.ix_(det_order, det_order)]
    sorted_ranks = rank_key[det_order]

    # Block boundaries
    boundaries = [0]
    for i in range(1, len(sorted_ranks)):
        if sorted_ranks[i] != sorted_ranks[i - 1]:
            boundaries.append(i)
    boundaries.append(n)

    # Build block index for each neuron (which block it belongs to)
    block_of = np.zeros(n, dtype=int)
    for b_idx in range(len(boundaries) - 1):
        block_of[boundaries[b_idx]:boundaries[b_idx + 1]] = b_idx

    # Color map for blocks: E1=blue, E2=orange, E3=green
    block_colors = {}
    sorted_groups = [group_names[det_order[i]] for i in range(n)]
    for b_idx in range(len(boundaries) - 1):
        start, end = boundaries[b_idx], boundaries[b_idx + 1]
        groups = Counter(sorted_groups[start:end])
        dominant = groups.most_common(1)[0][0]
        block_colors[b_idx] = GT_COLOR_MAP[dominant]

    mixed_color = to_rgba("#888888")  # grey for cross-block connections

    # Build RGBA image
    rgba = np.ones((n, n, 4))  # white background
    for i in range(n):
        for j in range(n):
            if sorted_adj[i, j] > 0.5:
                if block_of[i] == block_of[j]:
                    rgba[i, j, :] = to_rgba(block_colors[block_of[i]])
                else:
                    rgba[i, j, :] = mixed_color

    ax.imshow(rgba, aspect="equal", interpolation="nearest")
    ax.set_anchor("N")
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Neuron")

    # Draw boundary lines and block labels
    for i in range(1, len(sorted_ranks)):
        if sorted_ranks[i] != sorted_ranks[i - 1]:
            b = i - 0.5
            ax.axhline(b, color="tab:red", lw=0.8, alpha=0.8)
            ax.axvline(b, color="tab:red", lw=0.8, alpha=0.8)

    for b_idx in range(len(boundaries) - 1):
        start, end = boundaries[b_idx], boundaries[b_idx + 1]
        mid = (start + end - 1) / 2
        groups = Counter(sorted_groups[start:end])
        dominant = groups.most_common(1)[0][0]
        label = GT_SHORT[dominant]
        ax.text(mid, mid, label, ha="center", va="center",
                fontsize=style.base_label_size * 0.8, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))

    style.apply_to_axes(ax, panel_size, "cm")


# ── Master Figure Composition ───────────────────────────────
def main():
    """Compose all 5 panels into a single figure."""
    data = load_cache()
    style = StylePreset.publication_default()

    total_h = ROW1_H + VGAP + ROW2_H + VGAP + ROW3_H
    fig = plt.figure(
        figsize=(CANVAS_W / 2.54, total_h / 2.54),
        dpi=DPI, facecolor="white",
    )

    # 3 content rows + 2 gap rows
    gs_rows = fig.add_gridspec(
        5, 1,
        height_ratios=[ROW1_H, VGAP, ROW2_H, VGAP, ROW3_H],
        hspace=0,
    )

    # ── Row 1: A ──
    subfig_a = fig.add_subfigure(gs_rows[0, 0])
    create_panel_a(subfig_a, style, (CANVAS_W, ROW1_H), data)

    # ── Row 2: B (50%) | C (50%) ──
    subfig_bc = fig.add_subfigure(gs_rows[2, 0])
    gs_bc = subfig_bc.add_gridspec(
        1, 2, width_ratios=[0.50, 0.50],
        wspace=0.55, left=0.08, right=0.95, top=0.92, bottom=0.14,
    )
    ax_b = subfig_bc.add_subplot(gs_bc[0, 0])
    ax_c = subfig_bc.add_subplot(gs_bc[0, 1])

    create_panel_b(ax_b, style, (CANVAS_W * 0.5, ROW2_H), data, subfig_bc)
    create_panel_c(ax_c, style, (CANVAS_W * 0.5, ROW2_H), data, subfig_bc)

    # ── Row 3: D (50%) | E (50%) — match row 2 margins ──
    subfig_de = fig.add_subfigure(gs_rows[4, 0])
    gs_de = subfig_de.add_gridspec(
        1, 2, width_ratios=[0.50, 0.50],
        wspace=0.55, left=0.08, right=0.95, top=0.95, bottom=0.14,
    )
    ax_d = subfig_de.add_subplot(gs_de[0, 0])
    ax_e = subfig_de.add_subplot(gs_de[0, 1])

    create_panel_d(ax_d, style, (CANVAS_W * 0.5, ROW3_H), data)
    create_panel_e(ax_e, style, (CANVAS_W * 0.5, ROW3_H), data)

    # ── Panel labels ──
    label_fs = 12
    kw = dict(fontsize=label_fs, fontweight="bold", va="top", ha="left")

    subfig_a.text(0.01, 0.97, "A", **kw)
    subfig_bc.text(0.01, 1.0, "B", **kw)
    subfig_bc.text(0.52, 1.0, "C", **kw)
    subfig_de.text(0.01, 1.0, "D", **kw)
    subfig_de.text(0.52, 1.0, "E", **kw)

    # ── Save ──
    out_base = elife_paths.out_path("fig2")
    fig.savefig(out_base + ".png", dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_base + ".pdf", dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved {out_base}.pdf and .png")


if __name__ == "__main__":
    main()
