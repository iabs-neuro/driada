"""
Fig 5 -- Advanced Capabilities on a 200-unit toroidal CANN (4 panels: A-D).

Layout (single master GridSpec, 13 rows x 5 columns):
  Rows 0-6:   A  (raster, col 0)  |  B  (torus + embeddings, cols 2,4)
  Row 8:      A cbar              |  B cbars
  Row 10:     C  (RSA RDM, col 0) |  D  (network + theta, cols 2,4)
  Row 12:     C cbar              |  D cbar

All alignments enforced structurally — no SubplotSpec needed.

Usage:
    python publications/elife/fig5/fig5_advanced.py

Reads fig5_*.npz from the cache directory: fig5_data.py writes all of them
except fig5_ae_sweep.npz, which is written by supplementary/dci_sweep.py
Outputs: fig5.pdf + .png in the output directory (see elife_paths.py)
"""
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import elife_paths

from driada.utils.publication import StylePreset
import colorstamps

CACHE_DIR = elife_paths.CACHE_DIR
DPI = 600

# -- Layout constants --
W_CM = 18.0
H_CM = 22.0
LEFT = 0.08
RIGHT = 0.98
TOP = 0.96
BOTTOM = 0.02

# 5 columns: A/C | main_gap | B/D_left | inner_gap | B/D_right
COL_RATIOS = [3, 0.8, 3.4, 0.6, 3.4]

# 13 rows: all B and D panels are individual master grid rows
ROW_RATIOS = [
    2.0,   # r0:  torus / raster top
    0.9,   # r1:  gap (torus -> PCA)
    2.5,   # r2:  PCA / raster cont'd
    0.9,   # r3:  gap (PCA -> AE)
    2.5,   # r4:  AE / raster cont'd
    0.9,   # r5:  gap (AE -> AE+corr)
    2.5,   # r6:  AE+corr / raster cont'd
    0.8,   # r7:  gap (content -> cbar)
    0.2,   # r8:  cbar row (A + B)
    1.4,   # r9:  gap (Row1 -> Row2)
    3.981, # r10: C / D content (r10_prev * 0.9; panels use aspect=equal)
    0.45,  # r11: gap (C/D content -> cbar)
    0.2,   # r12: C/D cbar
]

# Colormap for circular variables
from matplotlib.colors import LinearSegmentedColormap
CIRC_CMAP1 = LinearSegmentedColormap.from_list("sat_cyclic1", [
    (0.85, 0.15, 0.15), (0.95, 0.65, 0.10), (0.15, 0.75, 0.15),
    (0.15, 0.55, 0.85), (0.60, 0.15, 0.75), (0.85, 0.15, 0.15),
], N=256)

CIRC_CMAP2 = LinearSegmentedColormap.from_list("ocean_fire", [
    (0.0, 0.2, 0.4), (0.0, 0.6, 0.8), (0.95, 0.95, 0.8),
    (0.9, 0.4, 0.1), (0.5, 0.0, 0.1), (0.0, 0.2, 0.4),
], N=256)


def load_cache():
    """Load all cached NPZ files."""
    data = {}
    for name in ["rnn", "intense", "vae", "rsa", "non", "ae_sweep"]:
        path = os.path.join(CACHE_DIR, f"fig5_{name}.npz")
        if not os.path.isfile(path):
            print(f"ERROR: cache file not found: {path}")
            print("Run fig5_data.py (and supplementary/dci_sweep.py for fig5_ae_sweep.npz) first.")
            sys.exit(1)
        data[name] = dict(np.load(path, allow_pickle=True))
    return data


# ── Panel A: RNN Activity Raster ──────────────────────────────────────────
def create_panel_a(ax, cbar_ax, fig, style, data):
    """Panel A: RNN activation raster (first 50s, sorted by mean rate)."""
    rnn = data["rnn"]
    activations = rnn["activations"]
    fps = float(rnn["fps"])

    show_sec = 50
    show_frames = int(show_sec * fps)
    act_slice = activations[:, :show_frames]
    order = np.argsort(act_slice.mean(axis=1))
    act_sorted = act_slice[order]

    im = ax.imshow(
        act_sorted, aspect="auto", cmap="inferno",
        extent=[0, show_sec, 0, activations.shape[0]],
        interpolation="nearest", rasterized=True,
    )
    ax.set_ylabel("Unit (sorted)", labelpad=2)
    ax.set_xticks([0, 50])
    ax.set_xlabel("Time (s)", labelpad=-7)
    ax.tick_params(pad=2)
    style.apply_to_axes(ax, (4.5, 14.0), "cm")  # AFTER labels exist

    # Horizontal colorbar below raster
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=style.base_tick_labelsize - 1, pad=1)
    cbar.set_ticks([act_sorted.min(), act_sorted.max()])
    cbar.set_ticklabels(["min", "max"])


# ── Panel B: Torus + Embeddings (individual axes) ─────────────────────────
def create_panel_b(fig, b_axes, style, data):
    """Panel B: 2-column layout with pre-created axes.

    Parameters
    ----------
    b_axes : dict with keys:
        'torus': [ax_0, ax_1]  (3D projection axes)
        'emb': [[ax_r0_c0, ax_r0_c1], [ax_r1_c0, ax_r1_c1], [ax_r2_c0, ax_r2_c1]]
        'cbar': [ax_cb0, ax_cb1]
    """
    vae = data["vae"]
    rnn = data["rnn"]

    pca_coords = vae["pca_coords"]
    pca_D = float(vae.get("pca_disentanglement", 0))

    # Each autoencoder is shown at its best initialization, the one whose
    # D is quoted in the text; columns of the sweep: plain AE, AE+corr
    sweep = data["ae_sweep"]
    best = np.argmax(sweep["dci_matrix"], axis=0)
    ae_recon_coords = sweep["coords"][best[0], 0]
    ae_coords = sweep["coords"][best[1], 1]
    ae_recon_D = float(sweep["dci_matrix"][best[0], 0])
    ae_D = float(sweep["dci_matrix"][best[1], 1])

    theta1 = rnn["theta1"]
    theta2 = rnn["theta2"]

    ds = 5
    t1_ds, t2_ds = theta1[::ds], theta2[::ds]
    pca_ds = pca_coords[:, ::ds]
    ae_recon_ds = ae_recon_coords[:, ::ds]
    ae_ds = ae_coords[:, ::ds]

    scatter_kw = dict(s=10, alpha=0.6, edgecolors="none", rasterized=True)
    gt_kw = dict(s=20, alpha=0.85, edgecolors="0.3", linewidths=0.2,
                 depthshade=False, rasterized=True)

    # Torus geometry
    R_maj, r_min = 1.0, 0.55
    phi1, phi2 = rnn["phi1"], rnn["phi2"]
    x_tor = (R_maj + r_min * np.cos(phi2)) * np.cos(phi1)
    y_tor = (R_maj + r_min * np.cos(phi2)) * np.sin(phi1)
    z_tor = r_min * np.sin(phi2)

    u_wf = np.linspace(0, 2 * np.pi, 40)
    v_wf = np.linspace(0, 2 * np.pi, 20)
    U, V = np.meshgrid(u_wf, v_wf)
    X_wf = (R_maj + r_min * np.cos(V)) * np.cos(U)
    Y_wf = (R_maj + r_min * np.cos(V)) * np.sin(U)
    Z_wf = r_min * np.sin(V)

    col_dim_pairs = [(0, 1), (1, 2)]
    col_cmaps = [CIRC_CMAP1, CIRC_CMAP2]
    var_cols = [
        ("$\\theta_1$", t1_ds, "theta1"),
        ("$\\theta_2$", t2_ds, "theta2"),
    ]

    emb_configs = [
        (pca_ds, "PC", pca_D, "PCA"),
        (ae_recon_ds, "AE dim", ae_recon_D, "AE"),
        (ae_ds, "AE+corr dim", ae_D, "AE+corr"),
    ]

    for col_i, (col_title, beh, var_key) in enumerate(var_cols):
        xd, yd = col_dim_pairs[col_i]
        ccmap = col_cmaps[col_i]
        order = np.argsort(beh)

        # Torus
        ax_gt = b_axes["torus"][col_i]
        ax_gt.set_proj_type("ortho")
        gt_color = phi1 if var_key == "theta1" else phi2
        gt_order = np.argsort(gt_color)
        ax_gt.scatter(x_tor[gt_order], y_tor[gt_order], z_tor[gt_order],
                      c=gt_color[gt_order], cmap=ccmap, vmin=0,
                      vmax=2 * np.pi, **gt_kw)
        ax_gt.plot_wireframe(X_wf, Y_wf, Z_wf, color="0.7", lw=0.4,
                             alpha=0.5, rasterized=True)
        ax_gt.set_axis_off()
        if var_key == "theta2":
            ax_gt.view_init(elev=55, azim=-60)
        else:
            ax_gt.view_init(elev=25, azim=-60)
        lim = 1.2
        ax_gt.set_xlim(-lim, lim)
        ax_gt.set_ylim(-lim, lim)
        ax_gt.set_zlim(-0.55, 0.55)
        ax_gt.set_box_aspect([1, 1, 0.4])

        # Embedding rows
        last_sc = None
        for row_i, (coords, dim_prefix, D_val, method_name) in enumerate(emb_configs):
            ax = b_axes["emb"][row_i][col_i]
            last_sc = ax.scatter(coords[xd, order], coords[yd, order],
                                 c=beh[order], cmap=ccmap, vmin=0,
                                 vmax=2 * np.pi, **scatter_kw)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"{dim_prefix} {xd+1}",
                          fontsize=style.base_label_size - 1, labelpad=4)
            ax.set_ylabel(f"{dim_prefix} {yd+1}",
                          fontsize=style.base_label_size - 1, labelpad=4)
            ax.tick_params(pad=3)
            style.apply_to_axes(ax, (5.0, 3.0), "cm")  # AFTER labels exist
            if col_i == 0:
                ax.set_title(f"{method_name}, $D = {D_val:.2f}$",
                             fontsize=style.base_label_size - 1, pad=2,
                             loc="left")

        # Colorbar
        ax_cb = b_axes["cbar"][col_i]
        cb = fig.colorbar(last_sc, cax=ax_cb, orientation="horizontal")
        cb.ax.tick_params(labelsize=style.base_tick_labelsize - 1)
        cb.set_ticks([0, np.pi, 2 * np.pi])
        cb.set_ticklabels(["0", "$\\pi$", "$2\\pi$"])


# ── Panel C: RSA RDM ──────────────────────────────────────────────────────
def create_panel_c(ax, cbar_ax, fig, style, data):
    """Panel C: RSA RDM reordered alphabetically."""
    rsa = data["rsa"]
    rdm = rsa["rdm"]
    rdm_labels = rsa["rdm_labels"]

    order = np.argsort(rdm_labels)
    rdm_ordered = rdm[np.ix_(order, order)]
    labels_ordered = rdm_labels[order]

    im = ax.imshow(rdm_ordered, cmap="RdBu_r", vmin=0, interpolation="nearest")
    ax.set_xticks([])  # colorbar replaces x labels
    ax.set_yticks(range(len(labels_ordered)))
    ax.set_yticklabels(labels_ordered,
                       fontsize=style.base_tick_labelsize * 0.8)
    ax.set_anchor("N")  # pin square heatmap to top of taller cell
    style.apply_to_axes(ax, (4.5, 5.0), "cm")

    # Horizontal colorbar below RDM
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=style.base_tick_labelsize - 1, pad=1)
    cbar.set_ticks([rdm_ordered.min(), rdm_ordered.max()])
    cbar.set_ticklabels(["similar", "dissimilar"])


# ── Panel D: NoN Network Graph ────────────────────────────────────────────
def _reshape_positions(pos, power=0.35):
    """Radial transform: spread the dense core, compress outliers."""
    center = np.median(pos, axis=0)
    shifted = pos - center
    r = np.sqrt((shifted ** 2).sum(axis=1))
    r_max = r.max()
    if r_max < 1e-12:
        return pos.copy()
    r_norm = r / r_max
    r_new = r_norm ** power * r_max
    scale = np.where(r > 1e-12, r_new / r, 1.0)
    return center + shifted * scale[:, None]


def create_panel_d(fig, d_axes, style, data):
    """Panel D: Left = NoN spring layout, Right = theta-plane + HSV inset.

    Parameters
    ----------
    d_axes : dict with keys:
        'spring': ax_left
        'theta': ax_right
        'cbar': ax_cbar_l
    """
    non = data["non"]
    rnn = data["rnn"]
    graph_nodes = non["graph_nodes"].astype(int)
    graph_pos = _reshape_positions(non["graph_pos"])
    # Rotate 45 degrees to better fill the panel
    cos45, sin45 = np.cos(np.pi / 4), np.sin(np.pi / 4)
    rot = np.array([[cos45, -sin45], [sin45, cos45]])
    center = graph_pos.mean(axis=0)
    graph_pos = (graph_pos - center) @ rot.T + center
    graph_edges = non["graph_edges"]

    phi1, phi2 = rnn["phi1"], rnn["phi2"]
    node_phi1 = phi1[graph_nodes]
    node_phi2 = phi2[graph_nodes]

    # Bivariate colors
    hsv_cmap = colorstamps.get_cmap("hsv")
    biv_colors, _ = colorstamps.apply_stamp(
        node_phi1, node_phi2, hsv_cmap,
        vmin_0=0, vmax_0=2 * np.pi, vmin_1=0, vmax_1=2 * np.pi,
    )
    biv_colors[np.isnan(biv_colors).any(axis=1)] = 0.7

    node_idx = {nid: i for i, nid in enumerate(graph_nodes)}
    ms = style.get_marker_size((10, 6), "cm") * 4.0

    ax_left = d_axes["spring"]
    ax_right = d_axes["theta"]
    ax_cbar_l = d_axes["cbar"]

    # -- Left: spring layout colored by theta1 --
    edge_lines = []
    for e in graph_edges:
        i_idx = node_idx.get(int(e[0]))
        j_idx = node_idx.get(int(e[1]))
        if i_idx is not None and j_idx is not None:
            edge_lines.append([graph_pos[i_idx], graph_pos[j_idx]])

    if edge_lines:
        lc = mcoll.LineCollection(edge_lines, colors="0.85", linewidths=0.15,
                                  alpha=0.4, rasterized=True)
        ax_left.add_collection(lc)

    sc = ax_left.scatter(graph_pos[:, 0], graph_pos[:, 1], c=node_phi1,
                         cmap=CIRC_CMAP1, vmin=0, vmax=2*np.pi, s=ms,
                         edgecolors="0.3", linewidths=0.3, zorder=3,
                         rasterized=True)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    for sp in ax_left.spines.values():
        sp.set_visible(False)
    ax_left.set_box_aspect(1)
    ax_left.set_anchor("NW")  # pin to top-left for column alignment
    ax_left.set_title("$\\phi_1$", fontsize=style.base_label_size, pad=2)

    # Colorbar below left
    cb = fig.colorbar(sc, cax=ax_cbar_l, orientation="horizontal")
    cb.ax.tick_params(labelsize=style.base_tick_labelsize - 1)
    cb.set_ticks([0, np.pi, 2 * np.pi])
    cb.set_ticklabels(["0", "$\\pi$", "$2\\pi$"])

    # -- Right: theta-plane with bivariate HSV --
    edge_lines_2d = []
    for e in graph_edges:
        i = node_idx.get(int(e[0]))
        j = node_idx.get(int(e[1]))
        if i is None or j is None:
            continue
        p1 = [node_phi1[i], node_phi2[i]]
        p2 = [node_phi1[j], node_phi2[j]]
        if abs(p1[0] - p2[0]) > np.pi or abs(p1[1] - p2[1]) > np.pi:
            continue
        edge_lines_2d.append([p1, p2])

    if edge_lines_2d:
        lc2 = mcoll.LineCollection(edge_lines_2d, colors="0.85", linewidths=0.15,
                                   alpha=0.3, rasterized=True)
        ax_right.add_collection(lc2)

    ax_right.scatter(node_phi1, node_phi2, c=biv_colors, s=ms,
                     edgecolors="0.3", linewidths=0.3, zorder=3,
                     rasterized=True)
    ax_right.set_xlim(-0.2, 2 * np.pi + 0.2)
    ax_right.set_ylim(-0.2, 2 * np.pi + 0.2)
    ax_right.set_xticks([0, np.pi, 2 * np.pi])
    ax_right.set_xticklabels(["0", "$\\pi$", "$2\\pi$"])
    ax_right.set_yticks([0, np.pi, 2 * np.pi])
    ax_right.set_yticklabels(["0", "$\\pi$", "$2\\pi$"])
    ax_right.tick_params(direction="in", pad=2,
                         labelsize=style.base_tick_labelsize - 1)
    ax_right.set_box_aspect(1)
    ax_right.set_anchor("NW")  # pin to top-left for column alignment
    ax_right.set_xlabel("$\\phi_1$", fontsize=style.base_label_size - 1,
                        labelpad=1)
    ax_right.set_ylabel("$\\phi_2$", fontsize=style.base_label_size - 1,
                        labelpad=1)
    ax_right.set_title("$(\\phi_1, \\phi_2)$ plane",
                       fontsize=style.base_label_size, pad=2)
    style.apply_to_axes(ax_right, (5, 5), "cm")

    # HSV colorplane inset — scatter grid colored by same apply_stamp
    _ng = 50
    _g1 = np.linspace(0, 2 * np.pi, _ng)
    _g2 = np.linspace(0, 2 * np.pi, _ng)
    _gg1, _gg2 = np.meshgrid(_g1, _g2)
    _grid_colors, _ = colorstamps.apply_stamp(
        _gg1.ravel(), _gg2.ravel(), hsv_cmap,
        vmin_0=0, vmax_0=2 * np.pi, vmin_1=0, vmax_1=2 * np.pi,
    )
    ax_inset = inset_axes(ax_right, width="20%", height="20%", loc="center",
                          borderpad=0.2)
    ax_inset.scatter(_gg1.ravel(), _gg2.ravel(), c=_grid_colors,
                     s=1.5, edgecolors="none", rasterized=True)
    ax_inset.set_xlim(0, 2 * np.pi)
    ax_inset.set_ylim(0, 2 * np.pi)
    ax_inset.set_aspect("equal")
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    for sp in ax_inset.spines.values():
        sp.set_linewidth(0.5)
    ax_inset.set_label("D_hsv")


# ── Verification ──────────────────────────────────────────────────────────
def verify_figure(fig, expected=None, tolerance=0.005):
    """Verify axis alignment, squareness, and overlap."""
    fig.canvas.draw()
    fig_w, fig_h = fig.get_size_inches()
    issues = []

    axes_info = {}
    for ax in fig.axes:
        name = ax.get_label()
        if not name or name.startswith("_"):
            continue
        pos = ax.get_position()
        w_inch = pos.width * fig_w
        h_inch = pos.height * fig_h
        axes_info[name] = {
            "x0": pos.x0, "y0": pos.y0,
            "x1": pos.x0 + pos.width, "y1": pos.y0 + pos.height,
            "w_cm": w_inch * 2.54, "h_cm": h_inch * 2.54,
            "aspect": h_inch / w_inch if w_inch > 0 else 0,
        }

    print("\nAxes positions:")
    for name, info in sorted(axes_info.items()):
        sq = " [SQUARE]" if abs(info["aspect"] - 1.0) < 0.05 else ""
        print(f"  {name:20s}: ({info['w_cm']:.1f} x {info['h_cm']:.1f} cm) "
              f"aspect={info['aspect']:.3f}{sq}")

    if expected is None:
        return axes_info

    # Check squareness
    for name in expected.get("square", []):
        if name in axes_info:
            a = axes_info[name]["aspect"]
            if abs(a - 1.0) > 0.02:
                issues.append(f"SQUARE FAIL: {name} aspect={a:.3f} (need 1.0)")

    # Check row alignment
    for row in expected.get("rows", []):
        present = [n for n in row if n in axes_info]
        if len(present) < 2:
            continue
        bottoms = [axes_info[n]["y0"] for n in present]
        tops = [axes_info[n]["y1"] for n in present]
        spread_b = max(bottoms) - min(bottoms)
        spread_t = max(tops) - min(tops)
        if spread_b > tolerance:
            issues.append(f"ROW BASELINE MISALIGNED: {present} spread={spread_b:.4f}")
        if spread_t > tolerance:
            issues.append(f"ROW TOP MISALIGNED: {present} spread={spread_t:.4f}")

    # Check column alignment
    for col in expected.get("cols", []):
        present = [n for n in col if n in axes_info]
        if len(present) < 2:
            continue
        lefts = [axes_info[n]["x0"] for n in present]
        spread = max(lefts) - min(lefts)
        if spread > tolerance:
            issues.append(f"COL LEFT MISALIGNED: {present} spread={spread:.4f}")

    if issues:
        print("\nVERIFICATION FAILED:")
        for issue in issues:
            print(f"  ! {issue}")
    else:
        print("\nVERIFICATION PASSED")

    return axes_info


# ── Panel Labels ──────────────────────────────────────────────────────────
def add_panel_labels(fig, label_map, fontsize=14, x_offset=-0.03, y_offset=0.01):
    """Add panel labels aligned by grid row and column."""
    fig.canvas.draw()

    info = {}
    for label, ax_ref in label_map.items():
        if isinstance(ax_ref, str):
            ax = next(a for a in fig.axes if a.get_label() == ax_ref)
        else:
            ax = ax_ref
        pos = ax.get_position()
        info[label] = {
            "x": pos.x0 + x_offset,
            "y": pos.y0 + pos.height + y_offset,
            "row_top": round(pos.y0 + pos.height, 2),
            "col_left": round(pos.x0, 2),
        }

    # Enforce row alignment
    row_groups = defaultdict(list)
    for label, d in info.items():
        row_groups[d["row_top"]].append(label)
    for members in row_groups.values():
        if len(members) > 1:
            shared_y = max(info[l]["y"] for l in members)
            for l in members:
                info[l]["y"] = shared_y

    # Enforce column alignment
    col_groups = defaultdict(list)
    for label, d in info.items():
        col_groups[d["col_left"]].append(label)
    for members in col_groups.values():
        if len(members) > 1:
            shared_x = min(info[l]["x"] for l in members)
            for l in members:
                info[l]["x"] = shared_x

    for label, d in info.items():
        fig.text(d["x"], d["y"], label,
                 fontsize=fontsize, fontweight="bold",
                 va="bottom", ha="left", transform=fig.transFigure)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    data = load_cache()
    style = StylePreset.publication_default()

    fig = plt.figure(
        figsize=(W_CM / 2.54, H_CM / 2.54),
        dpi=DPI, facecolor="white",
    )

    gs = GridSpec(
        len(ROW_RATIOS), len(COL_RATIOS), figure=fig,
        width_ratios=COL_RATIOS,
        height_ratios=ROW_RATIOS,
        left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM,
        wspace=0, hspace=0,
    )

    # ── Panel A: raster (col 0, rows 0-6) + cbar (row 8) ──
    ax_a = fig.add_subplot(gs[0:7, 0], label="A")
    ax_a_cbar = fig.add_subplot(gs[8, 0], label="A_cbar")
    create_panel_a(ax_a, ax_a_cbar, fig, style, data)

    # ── Panel B: individual master grid cells ──
    b_axes = {
        "torus": [
            fig.add_subplot(gs[0, 2], projection="3d", label="B_torus_0"),
            fig.add_subplot(gs[0, 4], projection="3d", label="B_torus_1"),
        ],
        "emb": [
            [fig.add_subplot(gs[2, 2], label="B_PCA_0"),
             fig.add_subplot(gs[2, 4], label="B_PCA_1")],
            [fig.add_subplot(gs[4, 2], label="B_AE_0"),
             fig.add_subplot(gs[4, 4], label="B_AE_1")],
            [fig.add_subplot(gs[6, 2], label="B_AE+corr_0"),
             fig.add_subplot(gs[6, 4], label="B_AE+corr_1")],
        ],
        "cbar": [
            fig.add_subplot(gs[8, 2], label="B_cbar_0"),
            fig.add_subplot(gs[8, 4], label="B_cbar_1"),
        ],
    }
    create_panel_b(fig, b_axes, style, data)

    # ── Panel C: RSA RDM (col 0, row 10) + cbar (row 12) ──
    ax_c = fig.add_subplot(gs[10, 0], label="C")
    ax_c_cbar = fig.add_subplot(gs[12, 0], label="C_cbar")
    create_panel_c(ax_c, ax_c_cbar, fig, style, data)

    # ── Panel D: individual master grid cells ──
    d_axes = {
        "spring": fig.add_subplot(gs[10, 2], label="D_spring"),
        "theta": fig.add_subplot(gs[10, 4], label="D_theta"),
        "cbar": fig.add_subplot(gs[12, 2], label="D_cbar_l"),
    }
    create_panel_d(fig, d_axes, style, data)

    # Fix torus 3D whitespace: 3D axes auto-square to min(w,h).
    # The projected torus (aspect ~1.45) fills ~69% of the square vertically.
    # Expand to a LARGER square so the content fills the visible cell height.
    # Overflow goes into surrounding gap rows/columns (invisible).
    fig.canvas.draw()
    for ax in fig.axes:
        name = ax.get_label()
        if name.startswith("B_torus_"):
            pos = ax.get_position()  # auto-squared position
            cx = pos.x0 + pos.width / 2
            cy = pos.y0 + pos.height / 2
            # Scale up by 1/content_frac; content fills ~69% of height
            target = pos.width * 1.42
            ax.set_position([cx - target / 2, cy - target / 2,
                             target, target])

    # "Ground truth" label centered above both tori (inside TOP margin)
    fig.canvas.draw()
    t0 = next(a for a in fig.axes if a.get_label() == "B_torus_0").get_position()
    t1 = next(a for a in fig.axes if a.get_label() == "B_torus_1").get_position()
    cx = (t0.x0 + t1.x0 + t1.width) / 2
    fig.text(cx, TOP, "ground truth", fontsize=style.base_label_size,
             ha="center", va="bottom")

    # Panel labels (placed last, from grid geometry)
    fig.canvas.draw()

    add_panel_labels(fig, {
        "A": "A",
        "B": "B_PCA_0",
        "C": "C",
        "D": "D_spring",
    }, fontsize=12, x_offset=-0.02, y_offset=0.005)

    # Force row alignment: A+B share top y, C+D share top y
    label_texts = {t.get_text(): t for t in fig.texts
                   if t.get_text() in ("A", "B", "C", "D")}
    if "A" in label_texts and "B" in label_texts:
        ay = label_texts["A"].get_position()[1]
        bx, _ = label_texts["B"].get_position()
        label_texts["B"].set_position((bx, ay))
    if "C" in label_texts and "D" in label_texts:
        cy = label_texts["C"].get_position()[1]
        dx, _ = label_texts["D"].get_position()
        label_texts["D"].set_position((dx, cy))

    # Programmatic verification
    verify_figure(fig, expected={
        "square": ["D_spring", "D_theta"],
        "rows": [
            ["A_cbar", "B_cbar_0", "B_cbar_1"],
            ["C_cbar", "D_cbar_l"],
        ],
        "cols": [
            ["A", "C", "A_cbar", "C_cbar"],
            ["B_PCA_0", "B_AE_0", "B_AE+corr_0", "B_cbar_0", "D_spring", "D_cbar_l"],
            ["B_PCA_1", "B_AE_1", "B_AE+corr_1", "B_cbar_1", "D_theta"],
        ],
    })

    # Save (no bbox_inches='tight')
    out_base = elife_paths.out_path("fig5")
    fig.savefig(out_base + ".png", dpi=DPI, facecolor="white", edgecolor="none")
    fig.savefig(out_base + ".pdf", dpi=DPI, facecolor="white", edgecolor="none")
    print(f"\nSaved {out_base}.pdf and .png")


if __name__ == "__main__":
    main()
