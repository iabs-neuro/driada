#!/usr/bin/env python3
"""
Basic Recurrence Analysis Example
==================================

Demonstrates DRIADA's core recurrence workflow on dynamical systems with
increasing complexity:

1. **Sine wave** (periodic) -- regular diagonal stripes in RP
2. **Lorenz attractor** (chaotic) -- highest DET/LAM (dense attractor lobes), high ENTR
3. **White noise** (stochastic) -- low DET/LAM, no recurrence structure
4. **Non-stationary** (sine-Lorenz-sine) -- regime transitions visible in
   windowed DET

The example produces five figures that illustrate:
- Embedding parameter selection (TDMI + FNN diagnostics)
- Recurrence plots with marginal time series
- 3D delay-embedded Lorenz attractor
- Grouped RQA comparison bar chart
- Windowed RQA on a non-stationary signal

This is a self-contained example that runs without external data files.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3D projection)

from driada.recurrence import (
    takens_embedding,
    estimate_tau,
    estimate_embedding_dim,
    RecurrenceGraph,
    plot_recurrence,
    compute_rqa,
)
from driada.information.info_base import get_tdmi

# =============================================================================
# SIGNAL COLORS (consistent throughout all figures)
# =============================================================================
COLOR_SINE = "#1f77b4"
COLOR_LORENZ = "#d62728"
COLOR_NOISE = "#7f7f7f"
COLOR_NONSTAT = "#9467bd"

OUTPUT_DIR = os.path.dirname(__file__)


# =============================================================================
# HELPERS
# =============================================================================
def _lorenz_series(n, dt=0.01, seed=42):
    """Generate x-component of Lorenz attractor via RK4 integration.

    Parameters: sigma=10, rho=28, beta=8/3 (classic chaotic regime).
    Returns n equally-spaced samples after a transient warm-up.
    """
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    rng = np.random.default_rng(seed)

    def lorenz(state):
        x, y, z = state
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    # Start near the attractor with small perturbation
    state = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.01, 3)

    # Warm-up: discard transient
    warmup = 2000
    for _ in range(warmup):
        k1 = lorenz(state)
        k2 = lorenz(state + dt / 2 * k1)
        k3 = lorenz(state + dt / 2 * k2)
        k4 = lorenz(state + dt * k3)
        state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # Collect samples
    x_series = np.empty(n)
    for i in range(n):
        k1 = lorenz(state)
        k2 = lorenz(state + dt / 2 * k1)
        k3 = lorenz(state + dt / 2 * k2)
        k4 = lorenz(state + dt * k3)
        state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x_series[i] = state[0]

    return x_series


def _compute_fnn_fractions(data, tau, max_dim=10, r_tol=10.0, a_tol=2.0):
    """Compute FNN fraction for each candidate embedding dimension.

    Replicates the logic from ``estimate_embedding_dim`` but collects the
    FNN fraction at every dimension instead of returning early.

    Parameters
    ----------
    data : array-like, 1D
        Time series.
    tau : int
        Time delay for embedding.
    max_dim : int
        Maximum dimension to evaluate.

    Returns
    -------
    dims : list of int
        Dimensions tested (2 to max_dim).
    fractions : list of float
        FNN fraction at each dimension.
    """
    from scipy.spatial import cKDTree

    data = np.asarray(data, dtype=float).ravel()
    attractor_size = np.std(data)
    dist_tol = attractor_size * 1e-8

    dims, fractions = [], []
    for m in range(2, max_dim + 1):
        try:
            emb_m = takens_embedding(data, tau, m).T
            emb_m1 = takens_embedding(data, tau, m + 1).T
        except ValueError:
            break  # series too short for this dimension

        n_m1 = emb_m1.shape[0]
        emb_m_trimmed = emb_m[:n_m1]

        tree = cKDTree(emb_m_trimmed)
        dists, indices = tree.query(emb_m_trimmed, k=2)

        nn_dists_m = dists[:, 1]
        nn_indices = indices[:, 1]

        nn_dists_m1 = np.linalg.norm(emb_m1 - emb_m1[nn_indices], axis=1)

        valid = nn_dists_m > dist_tol
        if not np.any(valid):
            dims.append(m)
            fractions.append(0.0)
            continue

        ratio = np.zeros(n_m1)
        ratio[valid] = np.abs(nn_dists_m1[valid] - nn_dists_m[valid]) / nn_dists_m[valid]

        criterion1 = ratio > r_tol
        criterion2 = (nn_dists_m1 / attractor_size) > a_tol
        fnn_fraction = np.sum(criterion1 | criterion2) / n_m1

        dims.append(m)
        fractions.append(fnn_fraction)

    return dims, fractions


def _windowed_det(adj_csr, window_size, step):
    """Compute DET in sliding windows along the diagonal of the RP.

    Parameters
    ----------
    adj_csr : sparse matrix
        Recurrence matrix (CSR format preferred for fast slicing).
    window_size : int
        Width of the sliding window in time indices.
    step : int
        Step size between consecutive windows.

    Returns
    -------
    positions : ndarray
        Center positions of each window.
    values : ndarray
        DET value computed within each window.
    """
    n = adj_csr.shape[0]
    positions, values = [], []
    for start in range(0, n - window_size, step):
        end = start + window_size
        sub = adj_csr[start:end, start:end]
        if sub.nnz > 0:
            rqa = compute_rqa(sub)
            values.append(rqa["DET"])
        else:
            values.append(0.0)
        positions.append((start + end) / 2)
    return np.array(positions), np.array(values)


# =============================================================================
# FIGURE BUILDERS
# =============================================================================
def _fig_embedding(signals, signal_colors, params, max_shift=80, max_dim=10):
    """Fig 1: TDMI curves (top row) and FNN fractions (bottom row).

    Shows the diagnostic process for selecting embedding parameters tau and
    dim for each signal type.
    """
    names = list(signals.keys())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for col, name in enumerate(names):
        sig = signals[name]
        tau, dim = params[name]
        color = signal_colors[name]

        # --- TDMI curve ---
        ax_tdmi = axes[0, col]
        tdmi = get_tdmi(sig, min_shift=1, max_shift=max_shift + 1, estimator="gcmi")
        shifts = np.arange(1, len(tdmi) + 1)
        ax_tdmi.plot(shifts, tdmi, color=color, linewidth=1.5)
        ax_tdmi.axvline(tau, color="k", linestyle="--", linewidth=1, alpha=0.7,
                        label=f"tau = {tau}")
        ax_tdmi.set_title(name, fontsize=11)
        ax_tdmi.set_xlabel("Lag (samples)")
        ax_tdmi.legend(fontsize=9, loc="upper right")
        if col == 0:
            ax_tdmi.set_ylabel("TDMI (bits)")

        # --- FNN curve ---
        ax_fnn = axes[1, col]
        fnn_dims, fnn_fracs = _compute_fnn_fractions(sig, tau, max_dim=max_dim)
        ax_fnn.plot(fnn_dims, fnn_fracs, "o-", color=color, linewidth=1.5,
                    markersize=5)
        ax_fnn.axhline(0.05, color="k", linestyle="--", linewidth=1, alpha=0.7,
                       label="5% threshold")
        ax_fnn.axvline(dim, color="k", linestyle=":", linewidth=1, alpha=0.5,
                       label=f"dim = {dim}")
        ax_fnn.set_xlabel("Embedding dimension")
        ax_fnn.set_ylim(-0.02, max(0.3, max(fnn_fracs) * 1.15) if fnn_fracs else 0.3)
        ax_fnn.legend(fontsize=9, loc="upper right")
        if col == 0:
            ax_fnn.set_ylabel("FNN fraction")

    fig.suptitle("Embedding parameter selection: TDMI and FNN diagnostics",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _fig_recurrence_plots(signals, signal_colors, graphs):
    """Fig 2: Time series (top) and recurrence plots (bottom), sharing x-axes."""
    names = list(signals.keys())
    n_sig = len(names)

    fig, axes = plt.subplots(2, n_sig, figsize=(5 * n_sig, 8),
                             gridspec_kw={"height_ratios": [1, 3]})

    for col, name in enumerate(names):
        sig = signals[name]
        rg = graphs[name]
        color = signal_colors[name]
        n_emb = rg.n  # number of embedded points

        # --- Time series (top) ---
        ax_ts = axes[0, col]
        ax_ts.plot(np.arange(n_emb), sig[:n_emb], color=color, linewidth=0.6)
        ax_ts.set_title(name, fontsize=11)
        ax_ts.set_xlim(0, n_emb - 1)
        ax_ts.tick_params(labelbottom=False)
        if col == 0:
            ax_ts.set_ylabel("Amplitude")

        # --- Recurrence plot (bottom) ---
        ax_rp = axes[1, col]
        plot_recurrence(rg, ax=ax_rp, markersize=0.3, color="k")
        ax_rp.set_title("")
        ax_rp.set_xlim(0, n_emb - 1)
        ax_rp.set_ylim(n_emb - 1, 0)
        ax_rp.set_xlabel("Time index")
        if col == 0:
            ax_rp.set_ylabel("Time index")

    fig.suptitle("Recurrence plots with marginal time series",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _fig_3d_embedding(lorenz_x, tau, dim):
    """Fig 3: 3D delay-embedded Lorenz attractor colored by time."""
    emb = takens_embedding(lorenz_x, tau, dim)  # shape (dim, N_emb)
    n_emb = emb.shape[1]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Use first three embedding dimensions
    xs, ys, zs = emb[0], emb[1], emb[2]
    time_color = np.arange(n_emb)

    sc = ax.scatter(xs, ys, zs, c=time_color, cmap="inferno", s=2, alpha=0.7)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Time index", fontsize=10)

    ax.set_xlabel(r"$x(t)$", fontsize=10)
    ax.set_ylabel(r"$x(t + \tau)$", fontsize=10)
    ax.set_zlabel(r"$x(t + 2\tau)$", fontsize=10)
    ax.set_title(f"3D delay embedding of Lorenz attractor "
                 f"(tau={tau}, dim={dim})", fontsize=12)
    ax.view_init(elev=25, azim=135)

    fig.tight_layout()
    return fig


def _fig_rqa_bars(graphs, signal_colors):
    """Fig 4: Grouped bar chart comparing DET, LAM, ENTR across signals."""
    names = list(graphs.keys())
    measures = ["DET", "LAM", "ENTR"]
    n_measures = len(measures)
    n_signals = len(names)

    rqa_data = {}
    for name, rg in graphs.items():
        rqa_data[name] = rg.rqa()

    x = np.arange(n_measures)
    width = 0.8 / n_signals

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(names):
        vals = [rqa_data[name][m] for m in measures]
        offset = (i - (n_signals - 1) / 2) * width
        ax.bar(x + offset, vals, width * 0.9,
               label=name, color=signal_colors[name], edgecolor="white",
               linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(measures, fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("RQA measures comparison", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def _fig_windowed(nonstat_signal, segment_len, rg, signal_color):
    """Fig 5: Non-stationary signal with windowed DET showing regime changes.

    Top panel: time series with regime-colored background.
    Bottom panel: windowed DET.
    """
    n_total = len(nonstat_signal)
    n_emb = rg.n

    fig, (ax_ts, ax_det) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                         gridspec_kw={"height_ratios": [1, 1.2]})

    # --- Top: time series with regime shading ---
    t = np.arange(n_emb)
    ax_ts.plot(t, nonstat_signal[:n_emb], color=signal_color, linewidth=0.5)

    # Shade regime regions (approximate: boundaries shift slightly due to embedding)
    regime_colors = [COLOR_SINE, COLOR_LORENZ, COLOR_SINE]
    regime_labels = ["Sine", "Lorenz", "Sine"]
    for i in range(3):
        start = i * segment_len
        end = min((i + 1) * segment_len, n_emb)
        if start < n_emb:
            ax_ts.axvspan(start, end, alpha=0.12, color=regime_colors[i])
            mid = (start + min(end, n_emb)) / 2
            ax_ts.text(mid, ax_ts.get_ylim()[0] if i > 0 else 0, regime_labels[i],
                       ha="center", va="bottom", fontsize=9, color=regime_colors[i],
                       fontweight="bold")

    ax_ts.set_ylabel("Amplitude")
    ax_ts.set_title("Non-stationary signal: sine - Lorenz - sine", fontsize=12)

    # --- Bottom: windowed DET ---
    adj_csr = rg.adj.tocsr()
    window_size = 150
    step = 10
    positions, det_values = _windowed_det(adj_csr, window_size, step)

    ax_det.plot(positions, det_values, color=signal_color, linewidth=1.5)
    ax_det.fill_between(positions, det_values, alpha=0.2, color=signal_color)

    # Shade same regime regions
    for i in range(3):
        start = i * segment_len
        end = min((i + 1) * segment_len, n_emb)
        if start < n_emb:
            ax_det.axvspan(start, end, alpha=0.08, color=regime_colors[i])

    ax_det.set_xlabel("Time index")
    ax_det.set_ylabel("Windowed DET")
    ax_det.set_ylim(-0.02, 1.05)
    ax_det.set_title("Windowed determinism (DET) tracks regime changes", fontsize=12)
    ax_det.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("DRIADA recurrence analysis - basic example")
    print("=" * 60)

    rng = np.random.default_rng(42)
    N = 800

    # =========================================================================
    # Step 1: Generate signals
    # =========================================================================
    print("\n[1] Generating signals...")

    t = np.arange(N)
    sine = np.sin(2 * np.pi * t / 50) + rng.normal(0, 0.05, N)
    lorenz_x = _lorenz_series(N, dt=0.02, seed=42)
    noise = rng.normal(size=N)

    signals = {
        "Sine (periodic)": sine,
        "Lorenz (chaotic)": lorenz_x,
        "Noise (stochastic)": noise,
    }
    signal_colors = {
        "Sine (periodic)": COLOR_SINE,
        "Lorenz (chaotic)": COLOR_LORENZ,
        "Noise (stochastic)": COLOR_NOISE,
    }
    print(f"  {N} points each: sine, Lorenz x-component, white noise")

    # Non-stationary signal: sine | Lorenz | sine (3 x 800 = 2400 points)
    seg_len = N
    sine_seg1 = np.sin(2 * np.pi * np.arange(seg_len) / 50) + rng.normal(0, 0.05, seg_len)
    lorenz_seg = _lorenz_series(seg_len, dt=0.02, seed=99)
    sine_seg2 = np.sin(2 * np.pi * np.arange(seg_len) / 50) + rng.normal(0, 0.05, seg_len)

    # Normalize each segment to similar amplitude range before concatenation
    lorenz_seg_norm = (lorenz_seg - lorenz_seg.mean()) / lorenz_seg.std()
    nonstat_signal = np.concatenate([sine_seg1, lorenz_seg_norm, sine_seg2])
    print(f"  Non-stationary signal: {len(nonstat_signal)} points "
          f"(3 x {seg_len}: sine-Lorenz-sine)")

    # =========================================================================
    # Step 2: Estimate embedding parameters
    # =========================================================================
    print("\n[2] Estimating embedding parameters...")
    print(f"  {'Signal':<22} {'tau':>5} {'dim':>5}")
    print(f"  {'-' * 32}")

    params = {}
    for name, sig in signals.items():
        tau = estimate_tau(sig, max_shift=80)
        dim = estimate_embedding_dim(sig, tau=tau, max_dim=10)
        params[name] = (tau, dim)
        print(f"  {name:<22} {tau:>5} {dim:>5}")

    # =========================================================================
    # Step 3: Build recurrence graphs
    # =========================================================================
    print("\n[3] Building recurrence graphs (kNN, k=5)...")

    graphs = {}
    for name, sig in signals.items():
        tau, dim = params[name]
        emb = takens_embedding(sig, tau=tau, m=dim)
        theiler = tau * (dim - 1) + 1
        rg = RecurrenceGraph(emb, method="knn", k=5, theiler_window=theiler)
        graphs[name] = rg
        print(f"  {name:<22} {rg.n} embedded points, {rg.adj.nnz} recurrence entries")

    # Non-stationary signal: fixed embedding (tau=12, dim=5)
    nonstat_tau, nonstat_dim = 12, 5
    nonstat_emb = takens_embedding(nonstat_signal, tau=nonstat_tau, m=nonstat_dim)
    nonstat_theiler = nonstat_tau * (nonstat_dim - 1) + 1
    nonstat_rg = RecurrenceGraph(nonstat_emb, method="knn", k=5,
                                 theiler_window=nonstat_theiler)
    print(f"  {'Non-stationary':<22} {nonstat_rg.n} embedded points, "
          f"{nonstat_rg.adj.nnz} recurrence entries (tau={nonstat_tau}, dim={nonstat_dim})")

    # =========================================================================
    # Step 4: RQA comparison (console)
    # =========================================================================
    print("\n[4] RQA comparison:")

    measures_list = ["RR", "DET", "LAM", "L_mean", "ENTR"]
    header = f"  {'Signal':<22}" + "".join(f"{m:>10}" for m in measures_list)
    print(header)
    print(f"  {'-' * (22 + 10 * len(measures_list))}")

    for name, rg in graphs.items():
        rqa = rg.rqa()
        row = f"  {name:<22}" + "".join(f"{rqa[m]:>10.4f}" for m in measures_list)
        print(row)

    # =========================================================================
    # Step 5: Generate figures
    # =========================================================================
    print("\n[5] Generating figures...")

    # --- Fig 1: Embedding diagnostics (TDMI + FNN) ---
    fig1 = _fig_embedding(signals, signal_colors, params)
    path1 = os.path.join(OUTPUT_DIR, "recurrence_basic_embedding.png")
    fig1.savefig(path1, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path1}")
    plt.close(fig1)

    # --- Fig 2: Recurrence plots with marginal time series ---
    fig2 = _fig_recurrence_plots(signals, signal_colors, graphs)
    path2 = os.path.join(OUTPUT_DIR, "recurrence_basic_rp.png")
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path2}")
    plt.close(fig2)

    # --- Fig 3: 3D delay embedding of Lorenz ---
    lorenz_tau, lorenz_dim = params["Lorenz (chaotic)"]
    fig3 = _fig_3d_embedding(lorenz_x, lorenz_tau, max(lorenz_dim, 3))
    path3 = os.path.join(OUTPUT_DIR, "recurrence_basic_3d.png")
    fig3.savefig(path3, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path3}")
    plt.close(fig3)

    # --- Fig 4: RQA grouped bar chart ---
    fig4 = _fig_rqa_bars(graphs, signal_colors)
    path4 = os.path.join(OUTPUT_DIR, "recurrence_basic_rqa.png")
    fig4.savefig(path4, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path4}")
    plt.close(fig4)

    # --- Fig 5: Windowed RQA on non-stationary signal ---
    fig5 = _fig_windowed(nonstat_signal, seg_len, nonstat_rg, COLOR_NONSTAT)
    path5 = os.path.join(OUTPUT_DIR, "recurrence_basic_windowed.png")
    fig5.savefig(path5, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path5}")
    plt.close(fig5)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("BASIC EXAMPLE COMPLETE")
    print("=" * 60)
    print("\nExpected patterns:")
    print("  - Sine: regular diagonal stripes (periodic recurrence)")
    print("  - Lorenz: highest DET/LAM (dense attractor lobes), highest ENTR")
    print("  - Noise: lowest DET/LAM (no temporal structure)")
    print("  - Non-stationary: windowed DET high in sine regimes, lower in Lorenz")
    print("\nNext steps:")
    print("  - Try recurrence_population/ for neural data with VG, OPN, and NoN")
    print("  - Explore epsilon-ball recurrence: method='eps', epsilon=<value>")
    print("  - Adjust k or epsilon to study recurrence density effects")


if __name__ == "__main__":
    main()
