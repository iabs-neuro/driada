#!/usr/bin/env python3
"""
Analyzing RNN activations with DRIADA
======================================

This example demonstrates that DRIADA works with any neural-like activity
data, not just calcium imaging. We simulate a random recurrent neural
network (RNN) driven by structured behavioral inputs and then run the
full DRIADA analysis pipeline:

1. Simulate a driven random RNN (pure numpy, no ML framework)
2. Load activations into a DRIADA Experiment
3. INTENSE: detect which units encode which task variables
4. Dimensionality reduction: visualize the population manifold
5. Cell-cell network: map functional connectivity between units

The RNN is a continuous-time network with random fixed weights, driven by
position, speed, head direction, trial type, and event inputs. Because the
input projection matrix is random, each hidden unit receives a unique
linear combination of inputs; after ReLU, the unit's firing encodes a
nonlinear mixture. INTENSE can recover which inputs each unit is
selective to.
"""

import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp

import driada
from driada.dim_reduction import MVData
from driada.experiment import load_exp_from_aligned_data
from driada.intense import compute_cell_cell_significance
from driada.network import Network

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # RNN architecture
    "n_units": 64,
    "tau": 0.2,             # membrane time constant (s)
    "g_rec": 1.2,           # recurrent gain (near edge of chaos)
    "w_in": 0.8,            # input projection scale
    "noise_sigma": 0.05,    # injected noise
    # Recording
    "duration": 600,        # seconds (10 min)
    "fps": 20,
    # Calcium-like smoothing of activations
    "tau_smooth": 1.0,      # exponential smoothing time constant (s)
    "obs_noise": 0.01,      # observation noise
    # INTENSE parameters
    "n_shuffles_stage1": 100,
    "n_shuffles_stage2": 10000,
    "pval_thr": 0.001,
    "ds": 5,
    # Cell-cell parameters
    "cc_n_shuffles_stage2": 10000,
    "cc_pval_thr": 0.01,
    # Reproducibility
    "seed": 42,
}


# =============================================================================
# BEHAVIORAL INPUT GENERATION
# =============================================================================
def generate_behavioral_inputs(n_frames, fps, rng):
    """Generate structured behavioral inputs for the RNN.

    Returns dict of 1D arrays, each of length n_frames.
    """
    dt = 1.0 / fps

    # Smooth random walk for x, y position on [0, 1]
    step_std = 0.3 * np.sqrt(dt)
    x = np.empty(n_frames)
    y = np.empty(n_frames)
    x[0], y[0] = 0.5, 0.5
    dx_raw = rng.normal(0, step_std, n_frames)
    dy_raw = rng.normal(0, step_std, n_frames)
    for t in range(1, n_frames):
        x[t] = x[t - 1] + dx_raw[t]
        y[t] = y[t - 1] + dy_raw[t]
        # Reflecting boundaries
        if x[t] < 0:
            x[t] = -x[t]
        elif x[t] > 1:
            x[t] = 2.0 - x[t]
        if y[t] < 0:
            y[t] = -y[t]
        elif y[t] > 1:
            y[t] = 2.0 - y[t]

    # Speed and head direction derived from trajectory
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    speed = np.sqrt(dx**2 + dy**2) * fps
    head_direction = np.arctan2(dy, dx) % (2 * np.pi)  # [0, 2pi)

    # Trial type: block-structured categorical (0, 1, 2)
    trial_type = np.zeros(n_frames, dtype=int)
    t = 0
    while t < n_frames:
        label = rng.integers(0, 3)
        block_len = int(rng.exponential(7.0) * fps)  # ~7s mean block
        block_len = max(block_len, int(fps))          # at least 1s
        trial_type[t : t + block_len] = label
        t += block_len

    # Event: sparse binary signal
    event = (rng.random(n_frames) < 0.03).astype(float)  # ~3% on-rate

    return {
        "x": x,
        "y": y,
        "speed": speed,
        "head_direction": head_direction,
        "trial_type": trial_type,
        "event": event,
    }


# =============================================================================
# RNN SIMULATION
# =============================================================================
def simulate_rnn(inputs, config, rng):
    """Simulate a driven random RNN and return smoothed activations.

    The network follows continuous-time dynamics discretized with Euler:
        x[t+1] = x[t] + (dt/tau)*(-x[t] + W_rec @ relu(x[t]) + W_in @ u[t] + noise)

    Returns activations array of shape (n_units, n_frames).
    """
    n_units = config["n_units"]
    tau = config["tau"]
    g = config["g_rec"]
    w_in = config["w_in"]
    sigma = config["noise_sigma"]
    fps = config["fps"]
    dt = 1.0 / fps

    # Stack input channels into (n_input, n_frames) matrix
    input_names = ["x", "y", "speed", "head_direction", "trial_type", "event"]
    u = np.stack([inputs[k].astype(float) for k in input_names], axis=0)
    n_input, n_frames = u.shape

    # Random fixed weights
    W_rec = rng.normal(0, g / np.sqrt(n_units), (n_units, n_units))
    W_in = rng.normal(0, w_in / np.sqrt(n_input), (n_units, n_input))

    # Euler integration
    state = np.zeros(n_units)
    raw = np.empty((n_units, n_frames))
    for t in range(n_frames):
        r = np.maximum(state, 0)  # ReLU
        raw[:, t] = r
        noise = rng.normal(0, sigma, n_units)
        state += (dt / tau) * (-state + W_rec @ r + W_in @ u[:, t] + noise)

    # Safety: check for NaN / explosion
    if np.any(np.isnan(raw)) or np.max(raw) > 1e6:
        raise RuntimeError("RNN activations diverged -- try reducing g_rec")

    # Exponential smoothing (mimics slow indicator dynamics)
    alpha = dt / config["tau_smooth"]
    activations = np.empty_like(raw)
    activations[:, 0] = raw[:, 0]
    for t in range(1, n_frames):
        activations[:, t] = (1 - alpha) * activations[:, t - 1] + alpha * raw[:, t]

    # Add observation noise
    activations += rng.normal(0, config["obs_noise"], activations.shape)
    activations = np.maximum(activations, 0)  # keep non-negative

    return activations


# =============================================================================
# EXPERIMENT CONSTRUCTION
# =============================================================================
def build_experiment(activations, inputs, config):
    """Wrap RNN activations and inputs into a DRIADA Experiment."""

    data = {"activations": activations, **inputs}

    exp = load_exp_from_aligned_data(
        data_source="RNN",
        exp_params={"name": "random_rnn"},
        data=data,
        feature_types={"head_direction": "circular", "speed": "linear"},
        aggregate_features={("x", "y"): "position_2d"},
        static_features={"fps": float(config["fps"])},
        create_circular_2d=True,
        verbose=True,
    )
    return exp


# =============================================================================
# INTENSE SELECTIVITY
# =============================================================================
def run_intense(exp, config):
    """Run INTENSE selectivity analysis."""
    stats, significance, info, results = driada.compute_cell_feat_significance(
        exp,
        mode="two_stage",
        n_shuffles_stage1=config["n_shuffles_stage1"],
        n_shuffles_stage2=config["n_shuffles_stage2"],
        pval_thr=config["pval_thr"],
        ds=config["ds"],
        verbose=True,
    )
    significant_neurons = exp.get_significant_neurons()

    # Per-feature summary
    feat_counts = {}
    for feats in significant_neurons.values():
        for f in feats:
            feat_counts[f] = feat_counts.get(f, 0) + 1
    n_mixed = sum(1 for feats in significant_neurons.values() if len(feats) > 1)

    print(f"\n  Selective units: {len(significant_neurons)} / {exp.n_cells}")
    for feat, cnt in sorted(feat_counts.items(), key=lambda x: -x[1]):
        print(f"    {feat}: {cnt} units")
    print(f"  Mixed selectivity (>1 feature): {n_mixed} units")

    return significant_neurons, info


# =============================================================================
# DIMENSIONALITY REDUCTION
# =============================================================================
def run_dr(exp):
    """Compute PCA embedding of the population activity."""
    mvdata = MVData(exp.calcium.data)  # (n_units, n_frames)
    emb = mvdata.get_embedding(method="pca")
    print(f"  PCA embedding: {emb.coords.shape}")
    return emb


# =============================================================================
# CELL-CELL NETWORK
# =============================================================================
def run_network(exp, config):
    """Compute cell-cell functional network."""
    sim_mat, sig_mat, pval_mat, cell_ids, cc_info = compute_cell_cell_significance(
        exp,
        data_type="calcium",
        ds=config["ds"],
        n_shuffles_stage1=config["n_shuffles_stage1"],
        n_shuffles_stage2=config["cc_n_shuffles_stage2"],
        pval_thr=config["cc_pval_thr"],
        multicomp_correction="holm",
        verbose=True,
    )

    n_sig = int(np.sum(np.triu(sig_mat, k=1)))
    n_pairs = len(cell_ids) * (len(cell_ids) - 1) // 2
    print(f"\n  Significant pairs: {n_sig} / {n_pairs}")

    # Build weighted network
    weighted = sp.csr_matrix(sim_mat * sig_mat)
    net = Network(adj=weighted, preprocessing="giant_cc", name="RNN functional network")
    print(f"  Network: {net.n} nodes, {net.graph.number_of_edges()} edges")

    return sim_mat, sig_mat, net


# =============================================================================
# VISUALIZATION
# =============================================================================
def create_visualization(
    exp, inputs, activations, significant_neurons, emb, sim_mat, sig_mat, net,
    output_dir,
):
    """Create 3x3 summary figure."""
    fig = plt.figure(figsize=(18, 14))
    n_frames = activations.shape[1]
    fps = CONFIG["fps"]

    # Show first 50 seconds
    show_frames = min(50 * fps, n_frames)
    t_sec = np.arange(show_frames) / fps

    # ---- Row 1: Data overview ----

    # (1,1) Input signals
    ax = fig.add_subplot(3, 3, 1)
    signals = [
        ("x", inputs["x"][:show_frames]),
        ("speed", inputs["speed"][:show_frames]),
        ("HD", inputs["head_direction"][:show_frames] / (2 * np.pi)),
        ("trial", inputs["trial_type"][:show_frames].astype(float) / 2),
    ]
    for i, (label, sig) in enumerate(signals):
        ax.plot(t_sec, sig + i * 1.2, lw=0.5)
        ax.text(-1, i * 1.2 + 0.4, label, fontsize=7, ha="right")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_title("Input signals")

    # (1,2) RNN activity raster
    ax = fig.add_subplot(3, 3, 2)
    # Sort units by mean rate for readability
    order = np.argsort(activations.mean(axis=1))
    ax.imshow(
        activations[order, :show_frames],
        aspect="auto", cmap="inferno",
        extent=[0, show_frames / fps, 0, activations.shape[0]],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Unit (sorted)")
    ax.set_title("RNN activations")

    # (1,3) INTENSE selectivity heatmap
    ax = fig.add_subplot(3, 3, 3)
    feat_names = [
        f for f in exp.dynamic_features
        if f not in ("x", "y", "head_direction")
    ]
    n_units = exp.n_cells
    mi_matrix = np.zeros((n_units, len(feat_names)))
    for uid, feats in significant_neurons.items():
        idx = int(uid)
        for fname in feats:
            if fname in feat_names:
                pair_stats = exp.get_neuron_feature_pair_stats(uid, fname)
                col = feat_names.index(fname)
                mi_matrix[idx, col] = pair_stats.get("me", 0)
    im = ax.imshow(mi_matrix, aspect="auto", cmap="viridis")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Unit")
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=7)
    ax.set_title("Selectivity (MI, significant only)")
    plt.colorbar(im, ax=ax, fraction=0.046, label="MI (bits)")

    # ---- Row 2: PCA embedding colored by different variables ----
    coords = emb.coords  # (2, n_frames)
    ds = 10  # thin for scatter speed
    x_pc = coords[0, ::ds]
    y_pc = coords[1, ::ds]

    color_vars = [
        ("x position", inputs["x"][::ds]),
        ("head direction", inputs["head_direction"][::ds]),
        ("trial type", inputs["trial_type"][::ds].astype(float)),
    ]
    cmaps = ["viridis", "twilight", "Set1"]
    for i, (label, cvar) in enumerate(color_vars):
        ax = fig.add_subplot(3, 3, 4 + i)
        sc = ax.scatter(x_pc, y_pc, c=cvar, cmap=cmaps[i], s=1, alpha=0.3, rasterized=True)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title(f"PCA colored by {label}")
        plt.colorbar(sc, ax=ax, fraction=0.046)

    # ---- Row 3: Network analysis ----

    # (3,1) Similarity matrix
    ax = fig.add_subplot(3, 3, 7)
    im = ax.imshow(sim_mat, cmap="hot", aspect="auto")
    ax.set_xlabel("Unit")
    ax.set_ylabel("Unit")
    ax.set_title("Cell-cell similarity (MI)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (3,2) Network graph
    ax = fig.add_subplot(3, 3, 8)
    if net.graph.number_of_edges() > 0:
        pos = nx.spring_layout(net.graph, seed=CONFIG["seed"])
        nx.draw_networkx_nodes(net.graph, pos, ax=ax, node_size=30, node_color="steelblue")
        nx.draw_networkx_edges(net.graph, pos, ax=ax, alpha=0.2, width=0.5)
        ax.set_title(f"Functional network ({net.n} nodes, {net.graph.number_of_edges()} edges)")
    else:
        ax.text(0.5, 0.5, "No significant edges", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Functional network")
    ax.axis("off")

    # (3,3) Summary text
    ax = fig.add_subplot(3, 3, 9)
    ax.axis("off")
    n_selective = len(significant_neurons)
    n_mixed = sum(1 for feats in significant_neurons.values() if len(feats) > 1)
    n_sig_pairs = int(np.sum(np.triu(sig_mat, k=1)))
    total_pairs = exp.n_cells * (exp.n_cells - 1) // 2
    density = n_sig_pairs / total_pairs if total_pairs > 0 else 0

    text = (
        f"RNN: {CONFIG['n_units']} units, g={CONFIG['g_rec']}\n"
        f"Recording: {CONFIG['duration']}s at {CONFIG['fps']} Hz\n\n"
        f"INTENSE selectivity:\n"
        f"  Selective units: {n_selective}/{exp.n_cells}\n"
        f"  Mixed selectivity: {n_mixed}\n\n"
        f"Functional network:\n"
        f"  Significant pairs: {n_sig_pairs}/{total_pairs}\n"
        f"  Density: {density:.3f}\n"
        f"  Nodes in GCC: {net.n}"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "rnn_activations.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("DRIADA -- Analyzing RNN activations")
    print("=" * 70)

    output_dir = os.path.dirname(__file__)
    rng = np.random.default_rng(CONFIG["seed"])
    n_frames = CONFIG["duration"] * CONFIG["fps"]
    t0 = time.time()

    # [1] Generate behavioral inputs
    print("\n[1] GENERATING BEHAVIORAL INPUTS")
    print("-" * 40)
    inputs = generate_behavioral_inputs(n_frames, CONFIG["fps"], rng)
    print(f"  Frames: {n_frames}, features: {list(inputs.keys())}")

    # [2] Simulate RNN
    print("\n[2] SIMULATING RNN")
    print("-" * 40)
    activations = simulate_rnn(inputs, CONFIG, rng)
    mean_rate = activations.mean()
    frac_active = (activations > 0).mean()
    print(f"  Activations: {activations.shape}")
    print(f"  Mean rate: {mean_rate:.3f}, fraction active: {frac_active:.2f}")

    # [3] Load into DRIADA
    print("\n[3] LOADING INTO DRIADA")
    print("-" * 40)
    exp = build_experiment(activations, inputs, CONFIG)
    print(f"  Experiment: {exp.n_cells} units, {exp.n_frames} frames")
    print(f"  Features: {list(exp.dynamic_features.keys())}")

    # [4] INTENSE selectivity analysis
    print("\n[4] INTENSE SELECTIVITY ANALYSIS")
    print("-" * 40)
    significant_neurons, info = run_intense(exp, CONFIG)

    # [5] Dimensionality reduction
    print("\n[5] DIMENSIONALITY REDUCTION")
    print("-" * 40)
    emb = run_dr(exp)

    # [6] Cell-cell functional network
    print("\n[6] CELL-CELL FUNCTIONAL NETWORK")
    print("-" * 40)
    sim_mat, sig_mat, net = run_network(exp, CONFIG)

    # [7] Visualization
    print("\n[7] CREATING VISUALIZATION")
    print("-" * 40)
    create_visualization(
        exp, inputs, activations, significant_neurons, emb,
        sim_mat, sig_mat, net, output_dir,
    )

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"COMPLETE  ({elapsed:.0f}s)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
