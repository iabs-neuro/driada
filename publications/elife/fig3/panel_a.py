"""
Panel A — Dataset overview: trajectory, occupancy, UMAP real, UMAP shuffled.

Delegated panel: receives fig + SubplotSpec, creates internal 2x2 grid.
"""

import os
import sys
import json

import numpy as np
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import elife_paths
elife_paths.add_tools_to_path()

SYNC_DIR = elife_paths.SYNC_DIR
INTENSE_DIR = elife_paths.INTENSE_DIR
SESSION = "FOF_F28_1D"
FPS = 30
DS = 5
SEED = 42

# The UMAP embedding is cached: it takes minutes and is not seeded
_DATA_DIR = elife_paths.CACHE_DIR


def _load_experiment():
    """Build the experiment as the multisession analysis does (frame rate 30, seed 42)."""
    import warnings
    from driada.experiment import load_exp_from_aligned_data

    data = dict(np.load(os.path.join(SYNC_DIR, f"{SESSION}_aligned.npz")))
    track, animal, day = SESSION.split("_")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_exp_from_aligned_data(
            data_source="FOF", exp_params={"track": track, "animal_id": animal, "session": day},
            data=data, force_continuous=["x", "y", "speed", "bodydirection", "headdirection"],
            static_features={"fps": FPS, "t_rise_sec": 0.05, "t_off_sec": 1.5},
            verbose=False, seed=SEED)


def create_panel_A(fig, subplot_spec, style, panel_size_cm):
    """Create the dataset overview panel (trajectory + occupancy + 2 UMAPs).

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
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import Circle as mplCircle
    import colorstamps

    print("  [Panel A] Loading data...")

    # --- Load experiment ---
    from driada.experiment import load_exp_from_pickle
    from load_synchronized_experiments import load_experiment_from_npz
    from driada.experiment import save_exp_to_pickle
    from pathlib import Path

    exp = _load_experiment()

    calcium = exp.calcium.scdata
    x = exp.dynamic_features["x"].data
    y = exp.dynamic_features["y"].data
    n_neurons, n_frames = calcium.shape

    # Filter to place + spatial-selective neurons
    results_intense = np.load(
        os.path.join(INTENSE_DIR, f"{SESSION}_results.npz"), allow_pickle=True,
    )
    sig_json = json.loads(str(results_intense["_significance_json"][0]))
    SPATIAL_FEATURES = ["place_quad", "place", "walls", "centertrue"]
    place_spatial_idx = []
    for i in range(n_neurons):
        cidx_str = str(i)
        if cidx_str not in sig_json:
            continue
        for fname in SPATIAL_FEATURES:
            if fname not in sig_json[cidx_str]:
                continue
            entry = sig_json[cidx_str][fname]
            if entry.get("stage2", entry.get("stage1", False)):
                place_spatial_idx.append(i)
                break

    calcium = calcium[place_spatial_idx]
    n_neurons = len(place_spatial_idx)

    # --- UMAP embeddings ---
    from driada.dim_reduction.manifold_metrics import procrustes_analysis

    UMAP_CACHE = os.path.join(_DATA_DIR, f"{SESSION}_umap_cache.npz")
    if os.path.exists(UMAP_CACHE):
        cache = np.load(UMAP_CACHE)
        coords_real = cache["coords_real"]
        coords_shuf = cache["coords_shuf"]
    else:
        from driada.information.info_base import MultiTimeSeries
        calcium_ds = calcium[:, ::DS]
        mts = MultiTimeSeries(calcium_ds, discrete=False)
        emb_real = mts.get_embedding(method="umap", dim=2, n_neighbors=100,
                                     min_dist=0.9)
        coords_real = emb_real.coords

        rng = np.random.default_rng(42)
        calcium_shuffled = calcium.copy()
        for i in range(calcium_shuffled.shape[0]):
            shift = rng.integers(1000, calcium_shuffled.shape[1] - 1000)
            calcium_shuffled[i] = np.roll(calcium_shuffled[i], shift)
        calcium_shuf_ds = calcium_shuffled[:, ::DS]
        mts_shuf = MultiTimeSeries(calcium_shuf_ds, discrete=False)
        emb_shuf = mts_shuf.get_embedding(method="umap", dim=2, n_neighbors=100,
                                           min_dist=0.9)
        coords_shuf = emb_shuf.coords
        np.savez_compressed(UMAP_CACHE, coords_real=coords_real,
                            coords_shuf=coords_shuf)

    x_ds = x[::DS]
    y_ds = y[::DS]
    positions = np.column_stack([x_ds, y_ds])

    aligned_real, disp_real, _ = procrustes_analysis(
        positions, coords_real.T, scaling=True, reflection=True,
    )
    aligned_shuf, disp_shuf, _ = procrustes_analysis(
        positions, coords_shuf.T, scaling=True, reflection=True,
    )

    # --- Bivariate coloring ---
    cstamp = colorstamps.get_cmap("hsv")
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    _cx, _cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    _r = min(x_max - x_min, y_max - y_min) / 2

    colors_ds, stamp = colorstamps.apply_stamp(
        x_ds, y_ds, cstamp,
        vmin_0=x_min, vmax_0=x_max, vmin_1=y_min, vmax_1=y_max,
    )
    colors_full, _ = colorstamps.apply_stamp(
        x, y, cstamp,
        vmin_0=x_min, vmax_0=x_max, vmin_1=y_min, vmax_1=y_max,
    )

    # --- Create inner 2x2 grid ---
    # height_ratios and hspace MUST match panel_b for cross-panel row alignment
    inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=subplot_spec,
                                    height_ratios=[1, 0.8],
                                    wspace=0.45, hspace=0.45)

    sub_w = panel_size_cm[0] / 2
    sub_h = panel_size_cm[1] / 2
    sub_size = (sub_w, sub_h)

    arena_xlim = [x_min - 3, x_max + 3]
    arena_ylim = [y_min - 3, y_max + 3]

    # --- A1: Trajectory (top-left) ---
    ax_traj = fig.add_subplot(inner[0, 0], label="A_traj")
    ax_traj.plot(x, y, lw=0.8, color="0.75", alpha=0.5, rasterized=True, zorder=1)
    ax_traj.scatter(x, y, c=colors_full, s=8, alpha=0.7, edgecolors="none",
                    rasterized=True, zorder=2)
    ax_traj.set_xlim(arena_xlim)
    ax_traj.set_ylim(arena_ylim)
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("x (cm)")
    ax_traj.set_ylabel("y (cm)")
    ax_traj.set_title("Trajectory")
    style.apply_to_axes(ax_traj, sub_size, "cm")

    # --- A2: Occupancy heatmap (top-right) ---
    ax_occ = fig.add_subplot(inner[0, 1], label="A_occ")
    nbins = 30
    valid = np.isfinite(x) & np.isfinite(y)
    H, xedges, yedges = np.histogram2d(x[valid], y[valid], bins=nbins)
    H_smooth = gaussian_filter(H.T, sigma=1)
    H_smooth_sec = H_smooth / FPS

    xc_bins = (xedges[:-1] + xedges[1:]) / 2
    yc_bins = (yedges[:-1] + yedges[1:]) / 2
    xxb, yyb = np.meshgrid(xc_bins, yc_bins)
    dist_from_center = np.sqrt((xxb - _cx)**2 + (yyb - _cy)**2)
    H_smooth_sec[dist_from_center > _r] = np.nan

    im = ax_occ.imshow(
        H_smooth_sec, origin="lower", aspect="equal", cmap="hot",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax_occ)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Time (s)", fontsize=style.base_tick_labelsize)
    cbar.ax.tick_params(labelsize=style.base_tick_labelsize - 1)
    ax_occ.set_xlim(arena_xlim)
    ax_occ.set_ylim(arena_ylim)
    ax_occ.set_xlabel("x (cm)")
    ax_occ.set_ylabel("y (cm)")
    ax_occ.set_title("Occupancy")
    style.apply_to_axes(ax_occ, sub_size, "cm")

    # --- Shared UMAP limits ---
    all_pts = np.vstack([aligned_real, aligned_shuf])
    pad = 0.05 * (all_pts.max() - all_pts.min())
    umap_lim = [all_pts.min() - pad, all_pts.max() + pad]

    # --- A3: UMAP (real) (bottom-left) ---
    ax_umap = fig.add_subplot(inner[1, 0], label="A_umap")
    ax_umap.scatter(aligned_real[:, 0], aligned_real[:, 1],
                    c=colors_ds, s=8, alpha=0.6, edgecolors="none",
                    rasterized=True)
    ax_umap.set_xlim(umap_lim)
    ax_umap.set_ylim(umap_lim)
    ax_umap.set_aspect("equal", adjustable="box")
    ax_umap.set_anchor("N")  # pin to top of cell so top aligns with B_loco
    ax_umap.set_xlabel("x (aligned)")
    ax_umap.set_ylabel("y (aligned)")
    ax_umap.set_title("UMAP (real)")
    style.apply_to_axes(ax_umap, sub_size, "cm")

    # Colormap inset
    _add_colorstamp_inset(ax_umap, cstamp, x_min, x_max, y_min, y_max, _cx, _cy, _r)

    # --- A4: UMAP (shuffled) (bottom-right) ---
    ax_umaps = fig.add_subplot(inner[1, 1], label="A_umaps",
                               sharex=ax_umap, sharey=ax_umap)
    ax_umaps.scatter(aligned_shuf[:, 0], aligned_shuf[:, 1],
                     c=colors_ds, s=8, alpha=0.6, edgecolors="none",
                     rasterized=True)
    ax_umaps.set_xlim(umap_lim)
    ax_umaps.set_ylim(umap_lim)
    ax_umaps.set_aspect("equal", adjustable="box")
    ax_umaps.set_anchor("N")  # pin to top of cell so top aligns with B_loco
    ax_umaps.set_xlabel("x (aligned)")
    ax_umaps.set_ylabel("y (aligned)")
    ax_umaps.set_title("UMAP (shuffled)")
    style.apply_to_axes(ax_umaps, sub_size, "cm")

    _add_colorstamp_inset(ax_umaps, cstamp, x_min, x_max, y_min, y_max, _cx, _cy, _r)

    print(f"  [Panel A] {n_neurons} selective neurons, "
          f"Procrustes: real={disp_real:.4f}, shuf={disp_shuf:.4f}")

    return {"A_traj": ax_traj, "A_occ": ax_occ,
            "A_umap": ax_umap, "A_umaps": ax_umaps}


def plot_on_axes(ax_traj, ax_occ, ax_umap, ax_umaps, fig, style, sub_sizes):
    """Plot panel A content on pre-created axes (master grid mode).

    Parameters
    ----------
    ax_traj, ax_occ, ax_umap, ax_umaps : Axes
    fig : Figure
    style : StylePreset
    sub_sizes : dict mapping axes name to (w_cm, h_cm)
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import Circle as mplCircle
    import colorstamps
    from driada.experiment import load_exp_from_pickle, save_exp_to_pickle
    from load_synchronized_experiments import load_experiment_from_npz
    from driada.dim_reduction.manifold_metrics import procrustes_analysis
    from pathlib import Path

    print("  [Panel A] Loading data...")

    exp = _load_experiment()

    calcium = exp.calcium.scdata
    x = exp.dynamic_features["x"].data
    y = exp.dynamic_features["y"].data
    n_neurons, n_frames = calcium.shape

    # Filter to place + spatial-selective neurons
    results_intense = np.load(
        os.path.join(INTENSE_DIR, f"{SESSION}_results.npz"), allow_pickle=True)
    sig_json = json.loads(str(results_intense["_significance_json"][0]))
    SPATIAL_FEATURES = ["place_quad", "place", "walls", "centertrue"]
    place_spatial_idx = []
    for i in range(n_neurons):
        cidx_str = str(i)
        if cidx_str not in sig_json:
            continue
        for fname in SPATIAL_FEATURES:
            if fname not in sig_json[cidx_str]:
                continue
            entry = sig_json[cidx_str][fname]
            if entry.get("stage2", entry.get("stage1", False)):
                place_spatial_idx.append(i)
                break
    calcium = calcium[place_spatial_idx]
    n_neurons = len(place_spatial_idx)

    # UMAP embeddings
    UMAP_CACHE = os.path.join(_DATA_DIR, f"{SESSION}_umap_cache.npz")
    if os.path.exists(UMAP_CACHE):
        cache = np.load(UMAP_CACHE)
        coords_real = cache["coords_real"]
        coords_shuf = cache["coords_shuf"]
    else:
        from driada.information.info_base import MultiTimeSeries
        calcium_ds = calcium[:, ::DS]
        mts = MultiTimeSeries(calcium_ds, discrete=False)
        emb_real = mts.get_embedding(method="umap", dim=2, n_neighbors=100, min_dist=0.9)
        coords_real = emb_real.coords
        rng = np.random.default_rng(42)
        calcium_shuffled = calcium.copy()
        for i in range(calcium_shuffled.shape[0]):
            shift = rng.integers(1000, calcium_shuffled.shape[1] - 1000)
            calcium_shuffled[i] = np.roll(calcium_shuffled[i], shift)
        calcium_shuf_ds = calcium_shuffled[:, ::DS]
        mts_shuf = MultiTimeSeries(calcium_shuf_ds, discrete=False)
        emb_shuf = mts_shuf.get_embedding(method="umap", dim=2, n_neighbors=100, min_dist=0.9)
        coords_shuf = emb_shuf.coords
        np.savez_compressed(UMAP_CACHE, coords_real=coords_real, coords_shuf=coords_shuf)

    x_ds, y_ds = x[::DS], y[::DS]
    positions = np.column_stack([x_ds, y_ds])
    aligned_real, disp_real, _ = procrustes_analysis(positions, coords_real.T, scaling=True, reflection=True)
    aligned_shuf, disp_shuf, _ = procrustes_analysis(positions, coords_shuf.T, scaling=True, reflection=True)

    # Coloring
    cstamp = colorstamps.get_cmap("hsv")
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    _cx, _cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    _r = min(x_max - x_min, y_max - y_min) / 2
    colors_ds, _ = colorstamps.apply_stamp(x_ds, y_ds, cstamp,
        vmin_0=x_min, vmax_0=x_max, vmin_1=y_min, vmax_1=y_max)
    colors_full, _ = colorstamps.apply_stamp(x, y, cstamp,
        vmin_0=x_min, vmax_0=x_max, vmin_1=y_min, vmax_1=y_max)

    arena_xlim = [x_min - 3, x_max + 3]
    arena_ylim = [y_min - 3, y_max + 3]

    # --- Trajectory ---
    ax_traj.plot(x, y, lw=0.5, color="0.75", alpha=0.5, rasterized=True, zorder=1)
    ax_traj.scatter(x, y, c=colors_full, s=4, alpha=0.7, edgecolors="none", rasterized=True, zorder=2)
    ax_traj.set_xlim(arena_xlim); ax_traj.set_ylim(arena_ylim)
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("x (cm)"); ax_traj.set_ylabel("y (cm)")
    ax_traj.set_title("Trajectory")
    style.apply_to_axes(ax_traj, sub_sizes["A_traj"], "cm")

    # --- Occupancy ---
    nbins = 30
    valid = np.isfinite(x) & np.isfinite(y)
    H, xedges, yedges = np.histogram2d(x[valid], y[valid], bins=nbins)
    H_smooth = gaussian_filter(H.T, sigma=1) / FPS
    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2
    xxb, yyb = np.meshgrid(xc, yc)
    H_smooth[np.sqrt((xxb - _cx)**2 + (yyb - _cy)**2) > _r] = np.nan
    im = ax_occ.imshow(H_smooth, origin="lower", aspect="equal", cmap="hot",
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset_axes
    cax = _inset_axes(ax_occ, width="100%", height="4%", loc="lower center",
                      bbox_to_anchor=(0, -0.38, 1, 1),
                      bbox_transform=ax_occ.transAxes, borderpad=0)
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Time (s)", fontsize=max(style.base_label_size - 2, 4))
    cbar.ax.tick_params(labelsize=max(style.base_tick_labelsize - 2, 4))
    ax_occ.set_xlim(arena_xlim); ax_occ.set_ylim(arena_ylim)
    ax_occ.set_xlabel("x (cm)"); ax_occ.set_ylabel("y (cm)")
    ax_occ.set_title("Occupancy")
    style.apply_to_axes(ax_occ, sub_sizes["A_occ"], "cm")

    # --- UMAP limits ---
    all_pts = np.vstack([aligned_real, aligned_shuf])
    pad = 0.05 * (all_pts.max() - all_pts.min())
    umap_lim = [all_pts.min() - pad, all_pts.max() + pad]

    # --- UMAP real ---
    ax_umap.scatter(aligned_real[:, 0], aligned_real[:, 1],
                    c=colors_ds, s=4, alpha=0.6, edgecolors="none", rasterized=True)
    ax_umap.set_xlim(umap_lim); ax_umap.set_ylim(umap_lim)
    ax_umap.set_aspect("equal", adjustable="box")
    ax_umap.set_anchor("N")
    ax_umap.set_xlabel("x (aligned)"); ax_umap.set_ylabel("y (aligned)")
    ax_umap.set_title("UMAP (real)")
    style.apply_to_axes(ax_umap, sub_sizes["A_umap"], "cm")
    _add_colorstamp_inset(ax_umap, cstamp, x_min, x_max, y_min, y_max, _cx, _cy, _r)

    # --- UMAP shuffled ---
    ax_umaps.scatter(aligned_shuf[:, 0], aligned_shuf[:, 1],
                     c=colors_ds, s=4, alpha=0.6, edgecolors="none", rasterized=True)
    ax_umaps.set_xlim(umap_lim); ax_umaps.set_ylim(umap_lim)
    ax_umaps.set_aspect("equal", adjustable="box")
    ax_umaps.set_anchor("N")
    ax_umaps.set_xlabel("x (aligned)"); ax_umaps.set_ylabel("y (aligned)")
    ax_umaps.set_title("UMAP (shuffled)")
    style.apply_to_axes(ax_umaps, sub_sizes["A_umaps"], "cm")
    _add_colorstamp_inset(ax_umaps, cstamp, x_min, x_max, y_min, y_max, _cx, _cy, _r)

    print(f"  [Panel A] {n_neurons} selective neurons, "
          f"Procrustes: real={disp_real:.4f}, shuf={disp_shuf:.4f}")


def _add_colorstamp_inset(ax, cstamp, x_min, x_max, y_min, y_max, cx, cy, r):
    """Add a circular 2D colormap inset to the upper-right of an axes."""
    import colorstamps
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import Circle as mplCircle

    _ng = 80
    _xg = np.linspace(x_min, x_max, _ng)
    _yg = np.linspace(y_min, y_max, _ng)
    _xxg, _yyg = np.meshgrid(_xg, _yg)
    _colors_grid, _ = colorstamps.apply_stamp(
        _xxg.ravel(), _yyg.ravel(), cstamp,
        vmin_0=x_min, vmax_0=x_max, vmin_1=y_min, vmax_1=y_max,
    )
    _dist = np.sqrt((_xxg.ravel() - cx)**2 + (_yyg.ravel() - cy)**2)
    _mask = _dist <= r

    ax_stamp = inset_axes(ax, width="16%", height="16%", loc="upper right",
                          borderpad=0.2)
    ax_stamp.scatter(_xxg.ravel()[_mask], _yyg.ravel()[_mask],
                     c=_colors_grid[_mask], s=1.5, edgecolors="none",
                     rasterized=True)
    ax_stamp.set_xlim(x_min, x_max)
    ax_stamp.set_ylim(y_min, y_max)
    ax_stamp.set_aspect("equal")
    ax_stamp.set_xticks([])
    ax_stamp.set_yticks([])
    for spine in ax_stamp.spines.values():
        spine.set_visible(False)
    ax_stamp.patch.set_visible(False)
    # No border circle — clean look


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
    create_panel_A(fig, gs[0], style, (W, H))
    fig.savefig(elife_paths.out_path("panel_a_standalone.png"), dpi=300,
                facecolor="white", edgecolor="none")
    print("Saved standalone panel_a")
