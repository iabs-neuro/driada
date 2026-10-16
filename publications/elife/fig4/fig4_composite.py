"""
Figure 4 — Scale-bridging: from single-neuron selectivity to population
manifolds and functional networks.

Layout (18 × 24 cm, 8 rows × 7 columns)
─────────────────────────────────────────
  Row 0:  [Isomap ALL (30%)]  [Session metrics | multisession summary]  ← SQUARE emb
  Row 1:  gap
  Row 2:  [Isomap FILT (30%)] [Embedding selectivity (67%)]  ← SQUARE emb
  Row 3:  gap
  Row 4-5: [Real network (left)]  [Shuffled network (right)]
  Row 6:  gap
  Row 7:  [Spectrum real (75%)]   [Spectrum shuffled (75%)]

Columns use the symmetric pattern of Fig 3 [1, g, 1, G, 1, g, 1].
Top rows use SubplotSpec for 38:62 embedding/chart split.
Network spectra centered at 75% via SubplotSpec [1,6,1].
"""

import os
import sys
import json
import time
import warnings
import zlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import elife_paths

from panel_c_network import (
    load_network_data, draw_real_network, draw_shuffled_network,
    draw_spectrum, draw_network_legend, COLOR_REAL, COLOR_SHUF,
    FEATURE_GROUPS, SEL_COLORS, SEL_DRAW_ORDER,
)

from driada.utils.publication.style import StylePreset
from driada.experiment import load_exp_from_aligned_data
from driada.dim_reduction.data import MVData
from driada.dim_reduction.dr_base import merge_params_with_defaults
from driada.dim_reduction.manifold_metrics import procrustes_analysis
from scipy.stats import spearmanr

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
SESSION = "FOF_F28_1D"
FPS = 30
DS = 5          # temporal downsampling
NN = 40         # Isomap n_neighbors
DIM = 2         # embedding dimensions

SYNC_DIR = elife_paths.SYNC_DIR
INTENSE_DIR = elife_paths.INTENSE_DIR
CACHE_PATH = os.path.join(elife_paths.CACHE_DIR, f"fig4_ab_cache_nn{NN}.npz")
# Per-session results of the multisession control analysis
MULTISESSION_CSV = os.path.join(elife_paths.CACHE_DIR, "fig4_multisession_sessions.csv")

# Controls and reverse analysis, as in the multisession analysis
N_RANDOM = 10       # random subsets of the filtered-set size
N_SHUF = 3          # circular-shift shuffles per neuron set
N_SHUF1, N_SHUF2 = 100, 10000   # two-stage embedding selectivity
PVAL_THR = 0.05
SEED = 42

# Figure geometry
W_CM, H_CM = 18.0, 22.0   # total height 22 cm (caption still fits)
DPI = 600
LEFT, RIGHT, TOP, BOTTOM = 0.08, 0.96, 0.96, 0.05  # room for spectra x-label

# Column / gap ratios (symmetric pattern of Fig 3)
g = 0.15    # inner gap within each half
G = 0.30    # main gap between halves

# Embedding / chart split (within SubplotSpec)
EMB_FRAC = 0.304  # 20% smaller than 0.38 to shrink top rows

# by_type representations name the place feature place_quad; a session where
# the representation was not applied keeps the plain name
SPATIAL_FEATURES = ["place_quad", "place", "walls", "centertrue"]

OUT_STEM = elife_paths.out_path("fig4")


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _get_place_spatial_idx(n_neurons, sig_json):
    """Return indices of neurons significantly selective to any spatial feature."""
    idx = []
    for i in range(n_neurons):
        cidx = str(i)
        if cidx not in sig_json:
            continue
        for fname in SPATIAL_FEATURES:
            if fname not in sig_json[cidx]:
                continue
            entry = sig_json[cidx][fname]
            if entry.get("stage2", entry.get("stage1", False)):
                idx.append(i)
                break
    return idx


def _get_primary_selectivity(neuron_idx, sig_json, stats_json):
    """Return the selectivity group with highest MI among significant features."""
    cidx = str(neuron_idx)
    if cidx not in sig_json:
        return "non-sel."
    best_group, best_mi = None, -np.inf
    for gname, fnames in FEATURE_GROUPS.items():
        for fname in fnames:
            if fname not in sig_json[cidx]:
                continue
            entry = sig_json[cidx][fname]
            is_sig = entry.get("stage2", entry.get("stage1", False))
            if not is_sig:
                continue
            mi = stats_json[cidx][fname]["me"]
            if mi > best_mi:
                best_mi = mi
                best_group = gname
    return best_group if best_group else "non-sel."


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

# Panel A/B computation mirrors the multisession control analysis
# (same loading, graph, metrics, random subsets, shuffles and seeds), so the
# example-session values in the figure are the FOF_F28_1D row of the
# multisession table rather than a separate estimate.
METRIC_KEYS = ("dimratio", "r2", "r2cv", "dcorr")


def _embed(cal):
    """Isomap coordinates (DIM x T_kept) and the indices of the kept frames.

    The k-NN graph uses the library default (union symmetrisation); frames
    outside the graph, if any, are dropped from the coordinates."""
    prm = merge_params_with_defaults("isomap", {"e_params": {"dim": DIM},
                                                "g_params": {"nn": NN}})
    emb = MVData(cal, downsampling=1).get_embedding(
        e_params=prm["e_params"], g_params=prm["g_params"], m_params=prm["m_params"])
    lost = getattr(emb.graph, "lost_nodes", set()) or set()
    kept = np.array(sorted(set(range(cal.shape[1])) - set(lost)))
    assert emb.coords.shape[1] == len(kept), (emb.coords.shape, len(kept))
    return emb.coords, kept


def _metrics(coords, kept, x, y):
    """Dimensionality ratio, in-sample and held-out position R^2, distance correlation."""
    from sklearn.linear_model import LinearRegression
    from scipy.spatial.distance import pdist

    pos = np.c_[x[kept], y[kept]]
    c = coords.T
    v = np.var(c, axis=0)
    r2_in = LinearRegression().fit(c, pos).score(c, pos)
    # Held-out R^2 on 5 contiguous time blocks: neighbouring frames are
    # strongly autocorrelated, so random folds would leak.
    folds = np.array_split(np.arange(len(pos)), 5)
    ss_res = ss_tot = 0.0
    for te in folds:
        tr = np.setdiff1d(np.arange(len(pos)), te)
        pred = LinearRegression().fit(c[tr], pos[tr]).predict(c[te])
        ss_res += ((pos[te] - pred) ** 2).sum()
        ss_tot += ((pos[te] - pos[tr].mean(0)) ** 2).sum()
    rng = np.random.RandomState(SEED)
    sub = rng.choice(len(pos), min(2000, len(pos)), replace=False)
    dcorr = spearmanr(pdist(c[sub]), pdist(pos[sub]))[0]
    return np.array([v.min() / v.max(), r2_in, 1 - ss_res / ss_tot, dcorr])


def _roll_shuffle(cal, rng):
    """Shift each neuron independently: breaks neuron-behavior coupling, keeps single-cell statistics."""
    T = cal.shape[1]
    lo = max(1, T // 10)
    return np.stack([np.roll(row, rng.integers(lo, T - lo)) for row in cal])


def compute_ab_cache(sig_json):
    """Embeddings, controls and reverse analysis for panels A and B; saved to CACHE_PATH."""
    from driada.intense import compute_embedding_selectivity
    from driada.intense.correction import get_multicomp_correction_thr

    print("[Fig4] Loading experiment...")
    data = dict(np.load(os.path.join(SYNC_DIR, f"{SESSION}_aligned.npz")))
    track, animal, day = SESSION.split("_")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp = load_exp_from_aligned_data(
            data_source="FOF", exp_params={"track": track, "animal_id": animal, "session": day},
            data=data, force_continuous=["x", "y", "speed", "bodydirection", "headdirection"],
            static_features={"fps": FPS, "t_rise_sec": 0.05, "t_off_sec": 1.5}, verbose=False,
            seed=SEED)
    cal = exp.calcium.scdata[:, ::DS]
    x_full = exp.dynamic_features["x"].data
    y_full = exp.dynamic_features["y"].data
    x, y = x_full[::DS], y_full[::DS]
    n, T = cal.shape
    filt = _get_place_spatial_idx(n, sig_json)
    rng = np.random.default_rng(SEED + zlib.crc32(SESSION.encode()) % 10000)

    print(f"[Fig4] Computing Isomap embeddings (n_neighbors={NN})...")
    c_all, k_all = _embed(cal)
    c_f, k_f = _embed(cal[filt])
    m_all = _metrics(c_all, k_all, x, y)
    m_filt = _metrics(c_f, k_f, x, y)
    print("[Fig4] Random subsets and shuffled controls...")
    m_rand = np.array([_metrics(*_embed(cal[rng.choice(n, len(filt), replace=False)]), x, y)
                       for _ in range(N_RANDOM)])
    m_shuf_all, m_shuf_filt = [], []
    for _ in range(N_SHUF):
        m_shuf_all.append(_metrics(*_embed(_roll_shuffle(cal, rng)), x, y))
        m_shuf_filt.append(_metrics(*_embed(_roll_shuffle(cal[filt], rng)), x, y))

    print(f"[Fig4] Embedding selectivity ({N_SHUF1}/{N_SHUF2} shuffles)...")
    # INTENSE needs a value for every frame: frames the graph dropped (none
    # expected with the union graph) take the nearest kept frame.
    idx = np.searchsorted(k_f, k_f[np.clip(np.searchsorted(k_f, np.arange(T)), 0, len(k_f) - 1)])
    exp.store_embedding(c_f[:, idx].T, method_name="isomap", data_type="calcium",
                        metadata={"ds": DS, "n_components": DIM, "n_neighbors": NN})
    res = compute_embedding_selectivity(
        exp, embedding_methods=["isomap"], mode="two_stage",
        n_shuffles_stage1=N_SHUF1, n_shuffles_stage2=N_SHUF2,
        multicomp_correction=None, pval_thr=PVAL_THR, find_optimal_delays=True,
        remove_anti_selective=False, verbose=False, seed=SEED)["isomap"]
    comp_sel = [sorted(int(i) for i in res["component_selectivity"][ci]) for ci in range(DIM)]
    pvals = [s.get("pval") for nd in res["stats"].values() for s in nd.values()
             if isinstance(s, dict) and s.get("pval") is not None]
    holm_thr = get_multicomp_correction_thr(PVAL_THR, mode="holm", all_pvals=pvals)
    inf_raw = sorted(set(comp_sel[0]) | set(comp_sel[1]))
    inf_holm = sorted({int(nid) for nid, nd in res["stats"].items() for s in nd.values()
                       if isinstance(s, dict) and s.get("pval") is not None
                       and s["pval"] < holm_thr and int(nid) in inf_raw})

    np.savez_compressed(
        CACHE_PATH, coords_all=c_all, kept_all=k_all, coords_filt=c_f, kept_filt=k_f,
        filt_idx=np.array(filt), x_ds=x, y_ds=y,
        xy_range=np.array([np.nanmin(x_full), np.nanmax(x_full), np.nanmin(y_full), np.nanmax(y_full)]),
        metric_keys=np.array(METRIC_KEYS), m_all=m_all, m_filt=m_filt, m_rand=m_rand,
        m_shuf_all=np.array(m_shuf_all), m_shuf_filt=np.array(m_shuf_filt),
        comp_sel_0=np.array(comp_sel[0], int), comp_sel_1=np.array(comp_sel[1], int),
        inf_raw=np.array(inf_raw, int), inf_holm=np.array(inf_holm, int), holm_thr=holm_thr,
        n_neurons=n, n_shuffles=np.array([N_SHUF1, N_SHUF2]), seed=SEED)


def load_multisession(path):
    """Per-session held-out R^2 and distance correlation from the multisession table."""
    import csv
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    cols = ["r2cv_all", "r2cv_rand_mean", "r2cv_filt", "dcorr_all", "dcorr_rand_mean", "dcorr_filt"]
    out = {c: np.array([float(r[c]) for r in rows]) for c in cols}
    out["session"] = [r["session"] for r in rows]
    out["mouse"] = [r["mouse"] for r in rows]
    return out


def load_data():
    """Load cached panel A/B results (computing them if absent) and network data."""
    t0 = time.perf_counter()

    results = np.load(
        os.path.join(INTENSE_DIR, f"{SESSION}_results.npz"), allow_pickle=True,
    )
    sig_json = json.loads(str(results["_significance_json"][0]))
    stats_json = json.loads(str(results["_stats_json"][0]))

    if not os.path.exists(CACHE_PATH):
        compute_ab_cache(sig_json)
    print(f"[Fig4] Loading {os.path.basename(CACHE_PATH)}")
    cache = np.load(CACHE_PATH)
    n_neurons = int(cache["n_neurons"])
    selectivities = [_get_primary_selectivity(i, sig_json, stats_json)
                     for i in range(n_neurons)]
    x_ds, y_ds = cache["x_ds"], cache["y_ds"]
    x_min, x_max, y_min, y_max = cache["xy_range"]

    # --- Procrustes alignment to physical position (kept frames only) ---
    aligned = {}
    for tag in ("all", "filt"):
        kept = cache[f"kept_{tag}"]
        positions = np.column_stack([x_ds[kept], y_ds[kept]])
        a, _, _ = procrustes_analysis(positions, cache[f"coords_{tag}"].T,
                                      scaling=True, reflection=True)
        aligned[tag] = (a.T, kept)

    keys = [str(k) for k in cache["metric_keys"]]
    metrics = {g: dict(zip(keys, cache[f"m_{g}"])) for g in ("all", "filt")}
    for g in ("rand", "shuf_all", "shuf_filt"):
        arr = cache[f"m_{g}"]
        metrics[g] = {k: arr[:, i] for i, k in enumerate(keys)}
    for k in ("r2cv", "dcorr"):
        print(f"  {k}: all={metrics['all'][k]:.3f} filt={metrics['filt'][k]:.3f} "
              f"rand={metrics['rand'][k].mean():.3f}+-{metrics['rand'][k].std(ddof=1):.3f} "
              f"shuf_all={metrics['shuf_all'][k].mean():.3f} "
              f"shuf_filt={metrics['shuf_filt'][k].mean():.3f}")

    # --- Pie chart data: per-dimension selectivity composition ---
    GROUP_ORDER = ["place", "spatial", "HD", "locomotion", "speed", "rest", "rear", "non-sel."]
    comp_sel = [cache["comp_sel_0"], cache["comp_sel_1"]]
    dim_compositions = []
    for ci in range(DIM):
        counts = {}
        for nid in comp_sel[ci]:
            sel = selectivities[nid]
            counts[sel] = counts.get(sel, 0) + 1
        dim_compositions.append(counts)
        print(f"    Dim {ci + 1} ({len(comp_sel[ci])} neurons): "
              + ", ".join(f"{g}={counts.get(g, 0)}" for g in GROUP_ORDER if counts.get(g, 0) > 0))
    inf_raw, inf_holm = cache["inf_raw"], cache["inf_holm"]
    nonsel = lambda ids: sum(selectivities[i] == "non-sel." for i in ids)
    print(f"  [Embed sel] informative: {len(inf_raw)} (non-sel. {nonsel(inf_raw)}), "
          f"Holm: {len(inf_holm)} (non-sel. {nonsel(inf_holm)}), "
          f"base non-sel. {nonsel(range(n_neurons))}/{n_neurons}, "
          f"shuffles {cache['n_shuffles'].tolist()}")

    # --- Bivariate position coloring ---
    import colorstamps
    cstamp = colorstamps.get_cmap("hsv")
    colors_ds, stamp = colorstamps.apply_stamp(
        x_ds, y_ds, cstamp,
        vmin_0=x_min, vmax_0=x_max, vmin_1=y_min, vmax_1=y_max,
    )
    cx_arena = (x_min + x_max) / 2
    cy_arena = (y_min + y_max) / 2
    r_arena = max(x_max - x_min, y_max - y_min) / 2

    print("[Fig4] Loading network data...")
    net_data = load_network_data()

    elapsed = time.perf_counter() - t0
    print(f"[Fig4] Data loaded in {elapsed:.1f}s")

    return {
        "aligned_all": aligned["all"][0],
        "aligned_filt": aligned["filt"][0],
        "colors_all": colors_ds[aligned["all"][1]],
        "colors_filt": colors_ds[aligned["filt"][1]],
        "stamp": stamp,
        "cstamp": cstamp,
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "cx_arena": cx_arena, "cy_arena": cy_arena, "r_arena": r_arena,
        "n_all": n_neurons,
        "n_filt": len(cache["filt_idx"]),
        "metrics": metrics,
        "multisession": load_multisession(MULTISESSION_CSV),
        "dim_compositions": dim_compositions,
        "comp_counts": [len(comp_sel[ci]) for ci in range(DIM)],
        "GROUP_ORDER": GROUP_ORDER,
        "net_data": net_data,
    }


# ──────────────────────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────────────────────

DIM_COLORS = ["#348ABD", "#E24A33"]   # Dim 1 blue, Dim 2 red
GROUP_LABELS = {
    "place": "plc", "spatial": "spt", "HD": "HD",
    "locomotion": "loc", "speed": "spd", "rest": "rst", "rear": "rer",
    "non-sel.": "n-s",
}


def draw_embedding(ax, aligned, colors, title, n_label, style, panel_size_cm,
                   cstamp_data=None):
    """Draw a Procrustes-aligned Isomap embedding colored by 2D position.

    Uses manual equal-range limits instead of set_aspect('equal') to avoid
    shrinking the axes box within the (already square) grid cell."""
    ax.scatter(aligned[0], aligned[1], c=colors, s=1.6, alpha=0.8,
               rasterized=True)
    # Manual equal scaling: set xlim/ylim to same range centered on data
    # The grid cell is algebraically square, so equal limits = correct aspect
    xmin, xmax = aligned[0].min(), aligned[0].max()
    ymin, ymax = aligned[1].min(), aligned[1].max()
    max_range = max(xmax - xmin, ymax - ymin)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    pad = max_range * 0.05
    ax.set_xlim(cx - max_range / 2 - pad, cx + max_range / 2 + pad)
    ax.set_ylim(cy - max_range / 2 - pad, cy + max_range / 2 + pad)
    ax.set_anchor("N")
    ax.set_title(f"{title} (n={n_label})", pad=4)
    ax.axis("off")
    style.apply_to_axes(ax, panel_size_cm, "cm")

    # Circular colorstamp inset
    if cstamp_data is not None:
        _add_colorstamp_inset(ax, **cstamp_data)


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


COLOR_ALL, COLOR_FILT, COLOR_RAND = "#BBBBBB", "#2CA02C", "#A8D5A8"


def draw_quality_metrics(ax, data, style, panel_size_cm):
    """Held-out position R^2 and distance correlation in the example session.

    Random subsets (same size as the filtered set) are shown as mean +- SD
    with individual subsets as dots; shuffles as the mean of the controls."""
    m = data["metrics"]
    metric_keys = ["r2cv", "dcorr"]
    metric_names = ["Held-out\nposition R$^2$", "Distance\ncorrelation"]
    positions = np.arange(len(metric_keys))
    bar_w = 0.16
    groups = [
        ("all", COLOR_ALL, f"All (n={data['n_all']})", False),
        ("rand", COLOR_RAND, f"Random (n={data['n_filt']}, {N_RANDOM}×)", False),
        ("filt", COLOR_FILT, f"Filtered (n={data['n_filt']})", False),
        ("shuf_all", COLOR_ALL, "Shuffled all", True),
        ("shuf_filt", COLOR_FILT, "Shuffled filt.", True),
    ]
    rng = np.random.RandomState(0)  # horizontal jitter of the subset dots only
    for gi, (key, color, label, is_shuf) in enumerate(groups):
        xs = positions + (gi - 2) * bar_w
        vals = [np.mean(m[key][k]) for k in metric_keys]
        ax.bar(xs, vals, bar_w, color=color, edgecolor="black", linewidth=0.4,
               alpha=0.4 if is_shuf else 1.0, hatch="//" if is_shuf else None,
               label=label)
        if key == "rand":
            sds = [np.std(m[key][k], ddof=1) for k in metric_keys]
            ax.errorbar(xs, vals, yerr=sds, fmt="none", ecolor="black",
                        elinewidth=0.6, capsize=1.5, capthick=0.6)
            for xi, k in zip(xs, metric_keys):
                ax.scatter(xi + rng.uniform(-0.04, 0.04, len(m[key][k])), m[key][k],
                           s=2, color="black", zorder=3, linewidths=0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Score")
    ax.set_ylim(-0.25, 1.25)  # headroom so the legend clears the tallest bar
    ax.set_title("Example session")
    ax.legend(loc="upper center", ncol=2, framealpha=0.7, columnspacing=0.8,
              handlelength=1.2)
    style.apply_to_axes(ax, panel_size_cm, "cm")  # AFTER title+legend exist


def draw_multisession_summary(ax, data, style, panel_size_cm):
    """Held-out position R^2 per session: all neurons, random subsets, filtered."""
    ms = data["multisession"]
    cols = ["r2cv_all", "r2cv_rand_mean", "r2cv_filt"]
    vals = np.column_stack([ms[c] for c in cols])
    xs = np.arange(3)
    for row in vals:
        ax.plot(xs, row, color="#888888", linewidth=0.4, alpha=0.6, zorder=1)
    for xi, color in zip(xs, [COLOR_ALL, COLOR_RAND, COLOR_FILT]):
        ax.scatter(np.full(len(vals), xi), vals[:, xi], s=4, color=color,
                   edgecolors="black", linewidths=0.2, zorder=2)
    ax.plot(xs, vals.mean(0), color="black", linewidth=1.0, marker="o",
            markersize=2.5, zorder=3)
    ax.axhline(0, color="black", linewidth=0.5)
    n = len(vals)
    n_rand = int((vals[:, 2] > vals[:, 1]).sum())
    n_all = int((vals[:, 2] > vals[:, 0]).sum())
    ax.text(0.97, 0.03, f"Filt. > rand.: {n_rand}/{n}\nFilt. > all: {n_all}/{n}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=style.base_legend_fontsize)
    ax.set_xticks(xs)
    ax.set_xticklabels(["All", "Rand.", "Filt."])
    ax.set_xlim(-0.4, 2.4)
    # room below the lowest session for the two-line count annotation
    lo, hi = vals.min(), vals.max()
    ax.set_ylim(lo - 0.28 * (hi - lo), hi + 0.05 * (hi - lo))
    ax.set_ylabel("Held-out position R$^2$")
    ax.set_title(f"{n} sessions, {len(set(ms['mouse']))} mice")
    style.apply_to_axes(ax, panel_size_cm, "cm")


def draw_embed_selectivity_pies(axes, data, style, panel_size_cm):
    """Draw two pie charts showing behavioral selectivity composition per Isomap dim.

    Returns list of (wedges, labels) for shared legend creation.
    """
    group_order = data["GROUP_ORDER"]
    compositions = data["dim_compositions"]
    comp_counts = data["comp_counts"]

    GROUP_LABELS_PIE = {
        "place": "place", "spatial": "spatial", "HD": "HD",
        "locomotion": "loco.", "speed": "speed", "rest": "rest",
        "rear": "rear", "non-sel.": "non-sel.",
    }

    # Use style-derived sizes
    pct_fontsize = style.base_legend_fontsize       # 6
    autopct_fontsize = pct_fontsize - 1             # 5

    all_wedge_info = []
    for ci, ax in enumerate(axes):
        comp = compositions[ci]
        labels = []
        sizes = []
        colors = []
        for g in group_order:
            n = comp.get(g, 0)
            if n > 0:
                labels.append(f"{GROUP_LABELS_PIE.get(g, g)} ({n})")
                sizes.append(n)
                colors.append(SEL_COLORS.get(g, "#CCCCCC"))

        wedges, texts, autotexts = ax.pie(
            sizes, labels=None, colors=colors, autopct="%1.0f%%",
            pctdistance=0.75, startangle=90, counterclock=False,
            wedgeprops={"edgecolor": "white", "linewidth": 0.5},
            textprops={"fontsize": pct_fontsize},
        )
        for at in autotexts:
            at.set_fontsize(autopct_fontsize)
            if float(at.get_text().replace("%", "")) < 4:
                at.set_text("")

        ax.set_anchor("N")  # pin to top, align with B_emb
        ax.set_title(f"Dim {ci+1} (n={comp_counts[ci]})", pad=2)
        style.apply_to_axes(ax, panel_size_cm, "cm")  # AFTER title exists
        all_wedge_info.append((wedges, labels))

    return all_wedge_info


def draw_spectra_overlay(ax, net_data, style, panel_size_cm):
    """Draw overlaid spectral density histograms (real vs shuffled)."""
    bins = np.linspace(0, 2, 50)
    ax.hist(net_data["spec_real"], bins=bins, color=COLOR_REAL,
            alpha=0.6, edgecolor="white", linewidth=0.3, label="Real")
    ax.hist(net_data["spec_shuf"], bins=bins, color=COLOR_SHUF,
            alpha=0.6, edgecolor="white", linewidth=0.3, label="Shuffled")
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 2)
    ax.set_title("Network spectra")
    ax.legend(loc="upper right", framealpha=0.7)
    style.apply_to_axes(ax, panel_size_cm, "cm")  # AFTER title+legend exist


# ──────────────────────────────────────────────────────────────
# Verification (from figure-alignment skill)
# ──────────────────────────────────────────────────────────────

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
    for name, info in sorted(axes_info.items()):
        sq = " [SQUARE]" if abs(info["aspect"] - 1.0) < 0.02 else ""
        print(f"  {name:16s}: ({info['w_cm']:.1f} x {info['h_cm']:.1f} cm) "
              f"aspect={info['aspect']:.3f}{sq}")
    if expected is None:
        return axes_info
    # Check squareness
    for name in expected.get("square", []):
        if name in axes_info:
            a = axes_info[name]["aspect"]
            if abs(a - 1.0) > 0.02:
                issues.append(f"SQUARE FAIL: {name} aspect={a:.3f} (need 1.0)")
    for row in expected.get("rows", []):
        present = [n for n in row if n in axes_info]
        if len(present) < 2:
            continue
        tops = [axes_info[n]["y1"] for n in present]
        bots = [axes_info[n]["y0"] for n in present]
        if max(tops) - min(tops) > tolerance:
            issues.append(f"ROW TOP MISALIGNED: {present} spread={max(tops)-min(tops):.4f}")
        if max(bots) - min(bots) > tolerance:
            issues.append(f"ROW BOT MISALIGNED: {present} spread={max(bots)-min(bots):.4f}")
    for col in expected.get("cols", []):
        present = [n for n in col if n in axes_info]
        if len(present) < 2:
            continue
        lefts = [axes_info[n]["x0"] for n in present]
        if max(lefts) - min(lefts) > tolerance:
            issues.append(f"COL LEFT MISALIGNED: {present} spread={max(lefts)-min(lefts):.4f}")
    if issues:
        print("VERIFICATION FAILED:")
        for issue in issues:
            print(f"  ! {issue}")
    else:
        print("VERIFICATION PASSED")
    return axes_info


def add_panel_labels(fig, label_map, fontsize=12, x_offset=-0.02, y_offset=0.01):
    """Add panel labels aligned by grid row and column."""
    from collections import defaultdict
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
    row_groups = defaultdict(list)
    for label, d in info.items():
        row_groups[d["row_top"]].append(label)
    for members in row_groups.values():
        if len(members) > 1:
            shared_y = max(info[l]["y"] for l in members)
            for l in members:
                info[l]["y"] = shared_y
    col_groups = defaultdict(list)
    for label, d in info.items():
        col_groups[d["col_left"]].append(label)
    for members in col_groups.values():
        if len(members) > 1:
            shared_x = min(info[l]["x"] for l in members)
            for l in members:
                info[l]["x"] = shared_x
    for label, d in info.items():
        fig.text(d["x"], d["y"], label, fontsize=fontsize,
                 fontweight="bold", va="bottom", ha="left",
                 transform=fig.transFigure)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    data = load_data()

    style = StylePreset.publication_default().copy(
        base_spine_width=1.2,
        base_tick_width=1.2,
        base_tick_length=3,
        base_tick_pad=1.5,
        base_tick_labelsize=7,
        base_label_size=8,
        base_title_size=9,
        base_legend_fontsize=6,
        base_annotation_fontsize=7,
        tight_layout=False,
    )

    fig = plt.figure(figsize=(W_CM / 2.54, H_CM / 2.54), dpi=DPI)

    # ── GridSpec: 10 rows × 7 columns ────────────────────────
    col_ratios = [1, g, 1, G, 1, g, 1]
    sum_col = sum(col_ratios)

    W_content = W_CM * (RIGHT - LEFT)
    H_content = H_CM * (TOP - BOTTOM)

    # Embedding width (38% of full content width) — must be square
    emb_w_cm = W_content * EMB_FRAC

    # Non-square row heights (cm → ratio units via H_content)
    other = {
        'gap_01': 5.2,         # white stripe A->B (~2.0 cm at H_CM=22)
        'gap_23': 5.7,         # white stripe B->C (~2.2 cm; holds shared legend)
        'net_top': 5.5,
        'net_bot': 5.5,
        'gap_net_spec': 1.5,   # gap between networks and spectra
        'spectra': 3.5,
    }
    non_R_sum = sum(other.values())

    # Algebraic square: R / (2R + non_R_sum) * H_content = emb_w_cm
    R = emb_w_cm * non_R_sum / (H_content - 2 * emb_w_cm)

    row_ratios = [
        R,                      # row 0: Isomap all + metrics
        other['gap_01'],        # row 1: gap
        R,                      # row 2: Isomap filt + embed sel
        other['gap_23'],        # row 3: gap
        other['net_top'],       # row 4: networks top
        other['net_bot'],       # row 5: networks bot
        other['gap_net_spec'],  # row 6: gap
        other['spectra'],       # row 7: spectra
    ]
    sum_row = sum(row_ratios)

    # Verify square computation
    sq_h = R / sum_row * H_content
    print(f"Embedding: {emb_w_cm:.2f} x {sq_h:.2f} cm (aspect {sq_h/emb_w_cm:.4f})")

    gs = GridSpec(8, 7, figure=fig,
                  width_ratios=col_ratios,
                  height_ratios=row_ratios,
                  left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM,
                  wspace=0, hspace=0)

    # ── Helper: compute physical sizes ────────────────────────
    half_w = W_content * (1 + g + 1) / sum_col
    net_h = H_content * (other['net_top'] + other['net_bot']) / sum_row
    chart_w = W_content * (1 - EMB_FRAC - 0.06)
    spec_w = half_w * 6 / 8   # 75% of half-width
    spec_h = H_content * other['spectra'] / sum_row

    # ── Row 0: Isomap ALL (38%) + Quality metrics (62%) ──────
    inner_r0 = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[0, 0:7],
        width_ratios=[EMB_FRAC, 0.06, 1 - EMB_FRAC - 0.06], wspace=0,
    )
    ax_A_emb = fig.add_subplot(inner_r0[0, 0], label="A_emb")
    # Chart area split: example-session metrics | multisession summary
    met_frac, sum_gap, sum_frac = 0.58, 0.14, 0.28
    inner_r0c = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=inner_r0[0, 2],
        width_ratios=[met_frac, sum_gap, sum_frac], wspace=0,
    )
    ax_A_met = fig.add_subplot(inner_r0c[0, 0], label="A_met")
    ax_A_sum = fig.add_subplot(inner_r0c[0, 2], label="A_sum")

    cstamp_data = {
        "cstamp": data["cstamp"],
        "x_min": data["x_min"], "x_max": data["x_max"],
        "y_min": data["y_min"], "y_max": data["y_max"],
        "cx": data["cx_arena"], "cy": data["cy_arena"], "r": data["r_arena"],
    }
    draw_embedding(ax_A_emb, data["aligned_all"], data["colors_all"],
                   "All neurons", data["n_all"], style, (emb_w_cm, sq_h),
                   cstamp_data=cstamp_data)
    draw_quality_metrics(ax_A_met, data, style, (chart_w * met_frac, sq_h))
    draw_multisession_summary(ax_A_sum, data, style, (chart_w * sum_frac, sq_h))

    # ── Row 2: Isomap FILTERED (EMB_FRAC) + Two pie charts with legend gap
    chart_frac = 1 - EMB_FRAC - 0.06
    leg_gap = 0.12
    pie_frac = (chart_frac - leg_gap) / 2
    inner_r2 = GridSpecFromSubplotSpec(
        1, 5, subplot_spec=gs[2, 0:7],
        width_ratios=[EMB_FRAC, 0.06, pie_frac, leg_gap, pie_frac], wspace=0,
    )
    ax_B_emb = fig.add_subplot(inner_r2[0, 0], label="B_emb")
    ax_B_pie0 = fig.add_subplot(inner_r2[0, 2], label="B_pie0")
    ax_B_pie1 = fig.add_subplot(inner_r2[0, 4], label="B_pie1")

    draw_embedding(ax_B_emb, data["aligned_filt"], data["colors_filt"],
                   "Filtered", data["n_filt"], style, (emb_w_cm, sq_h),
                   cstamp_data=cstamp_data)
    pie_w = chart_w * pie_frac / chart_frac
    draw_embed_selectivity_pies([ax_B_pie0, ax_B_pie1], data, style,
                                (pie_w, sq_h))
    # Per-pie count legend removed: a single enlarged selectivity legend
    # (drawn below) is shared by the pies (B) and the networks (C).

    # ── Rows 4-5: Networks (left/right halves) ───────────────
    ax_C_real = fig.add_subplot(gs[4:6, 0:3], label="C_real")
    ax_C_shuf = fig.add_subplot(gs[4:6, 4:7], label="C_shuf")

    draw_real_network(ax_C_real, data["net_data"])
    draw_shuffled_network(ax_C_shuf, data["net_data"])

    # Single selectivity legend, shared by B (pies) and C (networks).
    # Anchor at the gap midpoint, then raise by half the legend height by
    # using loc="lower center" (places that anchor at the legend's bottom).
    # Font matches the figure subtitle size (base_title_size).
    fig.canvas.draw()
    b_bottom = min(ax_B_emb.get_position().y0,
                   ax_B_pie0.get_position().y0,
                   ax_B_pie1.get_position().y0)
    c_top = ax_C_real.get_position().y1
    leg_y = (b_bottom + c_top) / 2
    draw_network_legend(fig, bbox_anchor=(0.5, leg_y), loc="lower center",
                        fontsize=style.base_title_size, ncol=4)

    # ── Row 7: Spectra — centered at 75% via SubplotSpec [1,6,1]
    inner_sp_l = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[7, 0:3],
        width_ratios=[1, 6, 1], wspace=0,
    )
    ax_sp_real = fig.add_subplot(inner_sp_l[0, 1], label="C_spec_real")
    draw_spectrum(ax_sp_real, data["net_data"]["spec_real"], COLOR_REAL,
                  "Spectral density (real)", style, (spec_w, spec_h))

    inner_sp_r = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[7, 4:7],
        width_ratios=[1, 6, 1], wspace=0,
    )
    ax_sp_shuf = fig.add_subplot(inner_sp_r[0, 1], label="C_spec_shuf")
    draw_spectrum(ax_sp_shuf, data["net_data"]["spec_shuf"], COLOR_SHUF,
                  "Spectral density (shuffled)", style, (spec_w, spec_h))

    # Match y-axis range
    ymax = max(ax_sp_real.get_ylim()[1], ax_sp_shuf.get_ylim()[1])
    ax_sp_real.set_ylim(0, ymax)
    ax_sp_shuf.set_ylim(0, ymax)

    # ── Panel labels ─────────────────────────────────────────
    add_panel_labels(fig, {
        "A": "A_emb", "B": "B_emb", "C": "C_real",
    })

    # ── Verification ─────────────────────────────────────────
    print("\n=== Verification ===")
    verify_figure(fig, expected={
        "rows": [["A_emb", "A_met", "A_sum"], ["C_real", "C_shuf"]],
        "cols": [["A_emb", "B_emb"]],
        "square": ["A_emb", "B_emb"],
    })
    # Verify pie top alignment with B_emb separately (bottom differs due to aspect)
    fig.canvas.draw()
    b_emb_top = ax_B_emb.get_position().y0 + ax_B_emb.get_position().height
    p0_top = ax_B_pie0.get_position().y0 + ax_B_pie0.get_position().height
    p1_top = ax_B_pie1.get_position().y0 + ax_B_pie1.get_position().height
    print(f"  B row top alignment: B_emb={b_emb_top:.4f} "
          f"pie0={p0_top:.4f} pie1={p1_top:.4f}")

    # ── Save ─────────────────────────────────────────────────
    out_png = OUT_STEM + ".png"
    out_pdf = OUT_STEM + ".pdf"
    fig.savefig(out_png, dpi=DPI, facecolor="white", edgecolor="none")
    fig.savefig(out_pdf, dpi=DPI, facecolor="white", edgecolor="none")
    print(f"\nSaved {out_png}")
    print(f"Saved {out_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
