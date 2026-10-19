"""
Panel B — Example selective neurons: place cell, HD cell, locomotion cell.

Delegated panel: receives fig + SubplotSpec, creates internal grid.
Layout: top row = [place heatmap | HD polar], bottom row = [loco trace | density].
"""

import os
import sys
import json
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter, binary_closing

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import elife_paths
elife_paths.add_tools_to_path()

import matplotlib.pyplot as plt

from driada.experiment import load_exp_from_aligned_data
from driada.intense.visual import plot_neuron_feature_pair

SYNC_DIR = elife_paths.SYNC_DIR
INTENSE_DIR = elife_paths.INTENSE_DIR
SESSION = "FOF_F28_1D"
FPS = 30  # nominal rate of the experiment, as in the INTENSE run
SEED = 42
TRACE_START = 16000  # first frame of the displayed trace
CLOSING_FRAMES = 100  # gaps in the run label shorter than this are merged for display
ACQ_FPS = 29.78  # acquisition rate from the miniscope timestamps

# Hardcoded neuron choices
PLACE_IDX = 614
HD_IDX = 660
DISCRETE_IDX = 158
DISCRETE_FEAT = "locomotion"
DISCRETE_LABEL = "run"
DISCRETE_NON_LABEL = "no run"
DISCRETE_COLOR = "darkorange"
OPT_DELAY_DISCRETE = 35

FORCE_CONTINUOUS = ["x", "y", "speed", "bodydirection", "headdirection"]
STATIC_FEATURES = {"fps": FPS, "t_rise_sec": 0.05, "t_off_sec": 1.5}


def create_panel_B(fig, subplot_spec, style, panel_size_cm):
    """Create the example neurons panel.

    Parameters
    ----------
    fig : Figure
    subplot_spec : SubplotSpec
        Region in the master GridSpec assigned to this panel.
    style : StylePreset
    panel_size_cm : tuple of (width_cm, height_cm)
        Physical size of the full panel region.
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    print("  [Panel B] Loading data...")

    # --- Load raw aligned data ---
    aligned = np.load(os.path.join(SYNC_DIR, f"{SESSION}_aligned.npz"))
    calcium = aligned["Calcium"]
    x_pos = aligned["x"].ravel()
    y_pos = aligned["y"].ravel()
    head_dir = aligned["headdirection"].ravel()

    # --- Load Experiment for locomotion panel ---
    data_dict = {key: aligned[key] for key in aligned.keys()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp = load_exp_from_aligned_data(
            data_source="FOF",
            exp_params={"track": "FOF", "animal_id": "F28", "session": "1D"},
            data=data_dict,
            force_continuous=FORCE_CONTINUOUS,
            static_features=STATIC_FEATURES,
            verbose=False,
            seed=SEED,
        )

    # Binarize discrete feature + merge gaps
    feat_disc = getattr(exp, DISCRETE_FEAT)
    disc_binary = feat_disc.data > 0
    disc_binary = binary_closing(disc_binary, structure=np.ones(CLOSING_FRAMES))
    feat_disc.data = disc_binary.astype(float)
    feat_disc.scdata = feat_disc.data.copy()

    # Apply optimal delay
    if OPT_DELAY_DISCRETE != 0:
        exp.neurons[DISCRETE_IDX].ca.scdata = np.roll(
            exp.neurons[DISCRETE_IDX].ca.scdata, -OPT_DELAY_DISCRETE
        )

    # --- Load MI values ---
    results = np.load(os.path.join(INTENSE_DIR, f"{SESSION}_results.npz"),
                      allow_pickle=True)
    stats = json.loads(str(results["_stats_json"][0]))
    place_mi = stats[str(PLACE_IDX)]["place_quad"]["me"]
    hd_mi = stats[str(HD_IDX)]["headdirection_harm2"]["me"]
    disc_mi = stats[str(DISCRETE_IDX)][DISCRETE_FEAT]["me"]

    # --- Create inner grid ---
    # Top row: place (left) + HD polar (right)
    # Bottom row: discrete trace (left, wider) + density (right)
    # height_ratios and hspace MUST match panel_a for cross-panel row alignment
    inner = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=subplot_spec,
        height_ratios=[1, 0.8],
        width_ratios=[0.55, 0.45],
        wspace=0.50, hspace=0.45,
    )

    top_size = (panel_size_cm[0] / 2, panel_size_cm[1] * 0.55)
    bot_size = (panel_size_cm[0] * 0.55, panel_size_cm[1] * 0.45)

    # --- B1: Place cell heatmap (top-left) ---
    ax_place = fig.add_subplot(inner[0, 0], label="B_place")

    ca_place = calcium[PLACE_IDX]
    n_bins = 25
    min_visits = 5
    weights, xedges, yedges = np.histogram2d(x_pos, y_pos, bins=n_bins,
                                              weights=ca_place)
    counts, _, _ = np.histogram2d(x_pos, y_pos, bins=n_bins)
    with np.errstate(invalid="ignore"):
        mean_ca = weights / counts
    mean_ca[counts < min_visits] = np.nan
    mean_ca_smooth = gaussian_filter(np.nan_to_num(mean_ca, nan=0.0), sigma=1.5)
    mean_ca_smooth[counts < min_visits] = np.nan
    # Relative to the map maximum: the panel shows the shape of the field, not its amplitude
    mean_ca_smooth = mean_ca_smooth / np.nanmax(mean_ca_smooth)

    im = ax_place.imshow(mean_ca_smooth.T, origin="lower", aspect="equal",
                         cmap="jet",
                         extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax_place)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Mean activity (norm.)", fontsize=style.base_tick_labelsize)
    cbar.ax.tick_params(labelsize=style.base_tick_labelsize - 1)
    ax_place.set_xlabel("x (cm)")
    ax_place.set_ylabel("y (cm)")
    ax_place.set_title("Place cell")
    style.apply_to_axes(ax_place, top_size, "cm")

    # --- B2: HD polar plot (top-right) ---
    ax_hd = fig.add_subplot(inner[0, 1], label="B_hd", projection="polar")

    ca_hd = calcium[HD_IDX]
    n_dir_bins = 36
    bin_edges = np.linspace(0, 2 * np.pi, n_dir_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    hd_rad = np.deg2rad(head_dir) if head_dir.max() > 2 * np.pi else head_dir.copy()
    hd_rad = hd_rad % (2 * np.pi)
    bin_idx = np.digitize(hd_rad, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_dir_bins - 1)

    mean_ca_hd = np.array([ca_hd[bin_idx == i].mean() if np.sum(bin_idx == i) > 0
                            else 0.0 for i in range(n_dir_bins)])

    norm = plt.Normalize(0, 2 * np.pi)
    hd_colors = plt.cm.hsv(norm(bin_centers))
    width = 2 * np.pi / n_dir_bins

    ax_hd.bar(bin_centers, mean_ca_hd, width=width, color=hd_colors, alpha=0.85,
              edgecolor="k", linewidth=0.3)
    angles_closed = np.append(bin_centers, bin_centers[0])
    vals_closed = np.append(mean_ca_hd, mean_ca_hd[0])
    ax_hd.plot(angles_closed, vals_closed, color="k", linewidth=1.2)

    ax_hd.set_title("HD cell",
                    fontsize=style.base_title_size, pad=4)
    ax_hd.set_yticklabels([])  # remove radial numbers
    ax_hd.set_thetagrids([0, 45, 135, 180, 225, 270, 315],
                          fontsize=style.base_tick_labelsize)
    ax_hd.tick_params(labelsize=style.base_tick_labelsize,
                      pad=style.base_tick_pad)

    # --- B3: Locomotion trace + density (bottom row) ---
    # Create sub-grid for trace (wider) + density (narrower)
    inner_bot = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=inner[1, :],
        width_ratios=[0.75, 0.25], wspace=0.20,
    )
    ax_trace = fig.add_subplot(inner_bot[0, 0], label="B_disc")
    ax_density = fig.add_subplot(inner_bot[0, 1], label="B_density")

    n_frames = exp.calcium.data.shape[1]
    ind2 = n_frames - max(OPT_DELAY_DISCRETE, 0)

    plot_neuron_feature_pair(
        exp,
        cell_id=DISCRETE_IDX,
        featname=DISCRETE_FEAT,
        ind1=0,
        ind2=ind2,
        ds=5,
        add_density_plot=True,
        axs=(ax_trace, ax_density),
        skip_tight_layout=True,
        bcolor=DISCRETE_COLOR,
        neuron_label="neuron",
        feature_label=DISCRETE_LABEL,
        non_feature_label=DISCRETE_NON_LABEL,
    )

    style.apply_to_axes(ax_trace, bot_size, "cm")
    style.apply_to_axes(ax_density, (panel_size_cm[0] * 0.2, bot_size[1]), "cm")

    print(f"  [Panel B] Place MI={place_mi:.4f}, HD MI={hd_mi:.4f}, "
          f"Loco MI={disc_mi:.4f}")

    return {"B_place": ax_place, "B_hd": ax_hd,
            "B_loco": ax_trace, "B_density": ax_density}


def plot_on_axes(ax_place, ax_hd, ax_loco, ax_density, fig, style, sub_sizes):
    """Plot panel B content on pre-created axes (master grid mode).

    Parameters
    ----------
    ax_place : Axes  — place cell heatmap
    ax_hd : Axes     — HD polar plot (must have projection='polar')
    ax_loco : Axes   — locomotion trace
    ax_density : Axes — locomotion density
    fig : Figure
    style : StylePreset
    sub_sizes : dict mapping axes name to (w_cm, h_cm)
    """
    print("  [Panel B] Loading data...")

    # --- Load raw aligned data ---
    aligned = np.load(os.path.join(SYNC_DIR, f"{SESSION}_aligned.npz"))
    calcium = aligned["Calcium"]
    x_pos = aligned["x"].ravel()
    y_pos = aligned["y"].ravel()
    head_dir = aligned["headdirection"].ravel()

    # --- Load Experiment for locomotion panel ---
    data_dict = {key: aligned[key] for key in aligned.keys()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp = load_exp_from_aligned_data(
            data_source="FOF",
            exp_params={"track": "FOF", "animal_id": "F28", "session": "1D"},
            data=data_dict,
            force_continuous=FORCE_CONTINUOUS,
            static_features=STATIC_FEATURES,
            verbose=False,
            seed=SEED,
        )

    # Binarize discrete feature + merge gaps
    feat_disc = getattr(exp, DISCRETE_FEAT)
    disc_binary = feat_disc.data > 0
    disc_binary = binary_closing(disc_binary, structure=np.ones(CLOSING_FRAMES))
    feat_disc.data = disc_binary.astype(float)
    feat_disc.scdata = feat_disc.data.copy()

    # Apply optimal delay
    if OPT_DELAY_DISCRETE != 0:
        exp.neurons[DISCRETE_IDX].ca.scdata = np.roll(
            exp.neurons[DISCRETE_IDX].ca.scdata, -OPT_DELAY_DISCRETE
        )

    # --- Load MI values ---
    results = np.load(os.path.join(INTENSE_DIR, f"{SESSION}_results.npz"),
                      allow_pickle=True)
    stats = json.loads(str(results["_stats_json"][0]))
    place_mi = stats[str(PLACE_IDX)]["place_quad"]["me"]
    hd_mi = stats[str(HD_IDX)]["headdirection_harm2"]["me"]
    disc_mi = stats[str(DISCRETE_IDX)][DISCRETE_FEAT]["me"]

    # --- B1: Place cell heatmap ---
    ca_place = calcium[PLACE_IDX]
    n_bins = 25
    min_visits = 5
    weights, xedges, yedges = np.histogram2d(x_pos, y_pos, bins=n_bins,
                                              weights=ca_place)
    counts, _, _ = np.histogram2d(x_pos, y_pos, bins=n_bins)
    with np.errstate(invalid="ignore"):
        mean_ca = weights / counts
    mean_ca[counts < min_visits] = np.nan
    mean_ca_smooth = gaussian_filter(np.nan_to_num(mean_ca, nan=0.0), sigma=1.5)
    mean_ca_smooth[counts < min_visits] = np.nan
    # Relative to the map maximum: the panel shows the shape of the field, not its amplitude
    mean_ca_smooth = mean_ca_smooth / np.nanmax(mean_ca_smooth)

    im = ax_place.imshow(mean_ca_smooth.T, origin="lower", aspect="equal",
                         cmap="jet",
                         extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset_axes
    cax = _inset_axes(ax_place, width="100%", height="4%", loc="lower center",
                      bbox_to_anchor=(0, -0.38, 1, 1),
                      bbox_transform=ax_place.transAxes, borderpad=0)
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Mean activity (norm.)", fontsize=max(style.base_label_size - 2, 4))
    cbar.ax.tick_params(labelsize=max(style.base_tick_labelsize - 2, 4))
    ax_place.set_xlabel("x (cm)")
    ax_place.set_ylabel("y (cm)")
    ax_place.set_title("Place cell")
    style.apply_to_axes(ax_place, sub_sizes["B_place"], "cm")

    # --- B2: HD polar plot ---
    ca_hd = calcium[HD_IDX]
    n_dir_bins = 36
    bin_edges = np.linspace(0, 2 * np.pi, n_dir_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    hd_rad = np.deg2rad(head_dir) if head_dir.max() > 2 * np.pi else head_dir.copy()
    hd_rad = hd_rad % (2 * np.pi)
    bin_idx = np.digitize(hd_rad, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_dir_bins - 1)

    mean_ca_hd = np.array([ca_hd[bin_idx == i].mean() if np.sum(bin_idx == i) > 0
                            else 0.0 for i in range(n_dir_bins)])

    norm = plt.Normalize(0, 2 * np.pi)
    hd_colors = plt.cm.hsv(norm(bin_centers))
    width = 2 * np.pi / n_dir_bins

    ax_hd.bar(bin_centers, mean_ca_hd, width=width, color=hd_colors, alpha=0.85,
              edgecolor="k", linewidth=0.3)
    angles_closed = np.append(bin_centers, bin_centers[0])
    vals_closed = np.append(mean_ca_hd, mean_ca_hd[0])
    ax_hd.plot(angles_closed, vals_closed, color="k", linewidth=1.2)

    ax_hd.set_title("HD cell",
                    fontsize=style.base_title_size, pad=4)
    ax_hd.set_yticklabels([])  # remove radial numbers
    ax_hd.set_thetagrids([0, 45, 135, 180, 225, 270, 315],
                          fontsize=style.base_tick_labelsize)
    ax_hd.tick_params(labelsize=style.base_tick_labelsize,
                      pad=style.base_tick_pad)

    # --- B3: Discrete feature trace + density ---
    n_frames = exp.calcium.data.shape[1]
    ind2 = n_frames - max(OPT_DELAY_DISCRETE, 0)

    ind1 = TRACE_START

    plot_neuron_feature_pair(
        exp,
        cell_id=DISCRETE_IDX,
        featname=DISCRETE_FEAT,
        ind1=ind1,
        ind2=ind2,
        ds=5,
        add_density_plot=True,
        axs=(ax_loco, ax_density),
        skip_tight_layout=True,
        bcolor=DISCRETE_COLOR,
        neuron_label="neuron",
        feature_label=DISCRETE_LABEL,
        non_feature_label=DISCRETE_NON_LABEL,
    )

    ax_loco.set_title("Run cell", fontsize=style.base_title_size)

    # plot_neuron_feature_pair draws the trace against frames / 20 whatever the
    # experiment rate is; label the axis in acquisition time.
    _real_ticks = np.array([550, 600, 650, 700])
    ax_loco.set_xticks(_real_ticks * ACQ_FPS / 20.0)
    ax_loco.set_xticklabels([str(t) for t in _real_ticks])

    # Halve all linewidths set by plot_neuron_feature_pair
    for line in ax_loco.get_lines():
        line.set_linewidth(line.get_linewidth() / 2)
    for line in ax_density.get_lines():
        line.set_linewidth(line.get_linewidth() / 2)

    style.apply_to_axes(ax_loco, sub_sizes["B_disc"], "cm")
    style.apply_to_axes(ax_density, sub_sizes["B_density"], "cm")

    # Collect handles/labels from BOTH axes, then remove both legends.
    leg_trace = ax_loco.get_legend()
    leg_dens = ax_density.get_legend()

    handles, labels = [], []
    if leg_trace is not None:
        handles.extend(leg_trace.legend_handles)
        labels.extend(t.get_text() for t in leg_trace.get_texts())
        leg_trace.remove()
    if leg_dens is not None:
        for h, t in zip(leg_dens.legend_handles, leg_dens.get_texts()):
            if t.get_text() not in labels:
                handles.append(h)
                labels.append(t.get_text())
        leg_dens.remove()

    # Single combined legend below the trace
    if handles:
        ax_loco.legend(handles, labels,
                       fontsize=style.base_legend_fontsize,
                       loc="upper center", bbox_to_anchor=(0.5, -0.35),
                       ncol=len(handles), frameon=False,
                       handlelength=1.5, handletextpad=0.4,
                       labelspacing=0.3, borderpad=0.0,
                       columnspacing=1.0)

    print(f"  [Panel B] Place MI={place_mi:.4f}, HD MI={hd_mi:.4f}, "
          f"{DISCRETE_FEAT} MI={disc_mi:.4f}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.gridspec import GridSpec
    from driada.utils.publication.style import StylePreset

    W, H = 9, 9
    fig = plt.figure(figsize=(W / 2.54, H / 2.54))
    gs = GridSpec(1, 1, figure=fig)
    style = StylePreset.publication_default().copy(
        base_spine_width=1.5, base_tick_width=1.5, base_tick_length=4,
        base_tick_pad=5, base_tick_labelsize=9, base_label_size=10,
        base_title_size=11, tight_layout=False,
    )
    create_panel_B(fig, gs[0], style, (W, H))
    fig.savefig(elife_paths.out_path("panel_b_standalone.png"), dpi=300,
                facecolor="white", edgecolor="none")
    print("Saved standalone panel_b")
