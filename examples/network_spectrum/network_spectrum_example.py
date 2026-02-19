"""
Example: Spectral Analysis of Neural Functional Networks

This example demonstrates the spectral analysis toolkit on a Network object:
1. Eigendecomposition of adjacency and normalized Laplacian matrices
2. Inverse Participation Ratio (IPR) -- eigenvector localization
3. Eigenvalue spacing ratios (z-values) and localization signatures
4. Thermodynamic entropy, free entropy, and Renyi q-entropy
5. Estrada communicability and bipartivity index
6. Gromov hyperbolicity (tree-likeness)
7. Laplacian Eigenmaps (LEM) embedding
8. Null model comparison (degree-preserving randomization)

Uses a synthetic modular network built with INTENSE cell-cell significance.

The normalized Laplacian L_norm = I - D^{-1/2} A D^{-1/2} is used throughout,
as its spectrum is bounded in [0, 2] regardless of network size or degree
distribution, making it suitable for cross-network comparison.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

# DRIADA imports
from driada.network import Network
from driada.intense import compute_cell_cell_significance


def create_modular_experiment(duration=300, seed=42):
    """Create synthetic experiment with hierarchical modular structure.

    Reuses the same design as the network_analysis example:
    120 neurons in 6 functional groups (3 single-feature + 3 dual-feature).
    """
    from driada.experiment.synthetic import generate_tuned_selectivity_exp

    population = [
        {"name": "event_0_cells", "count": 30, "features": ["event_0"]},
        {"name": "event_1_cells", "count": 30, "features": ["event_1"]},
        {"name": "event_2_cells", "count": 30, "features": ["event_2"]},
        {
            "name": "event_0_or_1_cells",
            "count": 10,
            "features": ["event_0", "event_1"],
            "combination": "or",
        },
        {
            "name": "event_0_or_2_cells",
            "count": 10,
            "features": ["event_0", "event_2"],
            "combination": "or",
        },
        {
            "name": "event_1_or_2_cells",
            "count": 10,
            "features": ["event_1", "event_2"],
            "combination": "or",
        },
    ]

    exp = generate_tuned_selectivity_exp(
        population=population,
        n_discrete_features=3,
        duration=duration,
        fps=20.0,
        baseline_rate=0.05,
        peak_rate=2.0,
        decay_time=2.0,
        calcium_noise=0.02,
        seed=seed,
        verbose=True,
    )

    return exp


def build_network(exp, pval_thr=0.01):
    """Compute cell-cell significance and build a Network object."""
    print("=" * 60)
    print("Computing cell-cell functional connectivity")
    print("=" * 60)

    sim_mat, sig_mat, pval_mat, cells, info = compute_cell_cell_significance(
        exp,
        data_type="calcium",
        ds=5,
        n_shuffles_stage1=100,
        n_shuffles_stage2=10000,
        pval_thr=pval_thr,
        multicomp_correction="holm",
        verbose=True,
    )

    # Extract the giant connected component so spectral methods
    # operate on a single connected graph (exactly one zero eigenvalue).
    sig_sparse = sp.csr_matrix(sig_mat)
    net = Network(
        adj=sig_sparse,
        preprocessing="giant_cc",
        name="Neural Functional Network",
    )

    n_edges = net.graph.number_of_edges()
    print(f"\nNetwork: {net.n} nodes, {n_edges} edges")
    return net


def spectral_decomposition(net):
    """Section 1: Eigendecomposition of adjacency and normalized Laplacian."""
    print("\n" + "=" * 60)
    print("1. Eigendecomposition")
    print("=" * 60)

    # --- Adjacency spectrum ---
    # The adjacency eigenvalues reflect community structure: isolated clusters
    # produce near-degenerate eigenvalue groups. The spectral radius (largest
    # |lambda|) scales with the mean degree for random graphs.
    adj_spectrum = net.get_spectrum("adj")
    print(f"\nAdjacency matrix:")
    print(f"  Spectral radius (max |lambda|): {np.max(np.abs(adj_spectrum)):.3f}")
    print(f"  Min eigenvalue: {np.min(np.real(adj_spectrum)):.3f}")
    print(f"  Max eigenvalue: {np.max(np.real(adj_spectrum)):.3f}")

    # --- Normalized Laplacian spectrum ---
    # L_norm = I - D^{-1/2} A D^{-1/2}
    # Eigenvalues are bounded in [0, 2] regardless of network size or degree,
    # making them suitable for comparing networks of different sizes.
    nlap_spectrum = net.get_spectrum("nlap")
    sorted_nlap = np.sort(np.real(nlap_spectrum))

    # Each zero eigenvalue corresponds to one connected component.
    # Since we extracted the giant connected component, expect exactly 1.
    n_zero = int(np.sum(np.abs(sorted_nlap) < 1e-6))
    n_components = n_zero

    print(f"\nNormalized Laplacian (L = I - D^(-1/2) A D^(-1/2)):")
    print(f"  Eigenvalue range: [0, 2]")
    print(f"  Smallest eigenvalue: {sorted_nlap[0]:.6f}")
    print(f"  Zero eigenvalues: {n_zero}  (= number of connected components)")
    if n_components == 1:
        print(f"  Graph is connected (single component)")
    else:
        print(f"  WARNING: graph has {n_components} disconnected components")

    # The Fiedler value (first non-zero eigenvalue) is the algebraic
    # connectivity of the normalized Laplacian. Larger values indicate
    # a graph that is harder to disconnect by removing edges.
    if len(sorted_nlap) > n_zero:
        fiedler = sorted_nlap[n_zero]
        print(f"  Fiedler value (lambda_{n_zero + 1}): {fiedler:.4f}")
        print(f"    (Algebraic connectivity -- larger = harder to disconnect)")
    print(f"  Spectral gap: {sorted_nlap[n_zero] - sorted_nlap[0]:.4f}")
    print(f"  Largest eigenvalue: {sorted_nlap[-1]:.4f}")

    adj_vecs = net.get_eigenvectors("adj")
    print(f"\nAdjacency eigenvectors: {adj_vecs.shape}  (N x N matrix)")

    return adj_spectrum, nlap_spectrum, n_zero


def ipr_analysis(net):
    """Section 2: Inverse Participation Ratio -- eigenvector localization.

    IPR measures how many nodes participate in each eigenvector:
      - IPR = 1/N: eigenvector is fully delocalized (uniform over all nodes)
      - IPR = 1:   eigenvector is fully localized on a single node
    High IPR eigenvectors indicate hub nodes or tightly-knit subgraphs.
    """
    print("\n" + "=" * 60)
    print("2. IPR analysis (eigenvector localization)")
    print("=" * 60)

    # IPR = sum(|v_i|^4) for each eigenvector v.
    # For a vector uniformly spread over N nodes, IPR = 1/N.
    # For a vector concentrated on one node, IPR = 1.
    ipr_adj = net.get_ipr("adj")
    ipr_nlap = net.get_ipr("nlap")

    delocalized_bound = 1.0 / net.n

    print(f"\nAdjacency IPR:")
    print(f"  Mean: {np.mean(ipr_adj):.4f}")
    print(f"  Min:  {np.min(ipr_adj):.4f}  (delocalized bound 1/N = {delocalized_bound:.4f})")
    print(f"  Max:  {np.max(ipr_adj):.4f}")

    print(f"\nNormalized Laplacian IPR:")
    print(f"  Mean: {np.mean(ipr_nlap):.4f}")
    print(f"  Min:  {np.min(ipr_nlap):.4f}")
    print(f"  Max:  {np.max(ipr_nlap):.4f}")

    return ipr_adj, ipr_nlap


def spacing_ratios_and_localization(net):
    """Section 3: Complex spacing ratios and localization signatures.

    Complex spacing ratios z_i = (lambda_nn - lambda_i) / (lambda_nnn - lambda_i)
    where lambda_nn and lambda_nnn are the nearest and next-nearest neighbor
    eigenvalues. These ratios characterize level statistics without requiring
    eigenvalue unfolding (Atas et al. 2013, Sa et al. 2020).

    |z| is bounded in [0, 1] since the nearest neighbor is always closer
    than the next-nearest. The phase arg(z) encodes the angular arrangement
    of neighbors around each eigenvalue.

    Zero eigenvalues are excluded because they are trivial (one per connected
    component) and would distort the nearest-neighbor structure.
    """
    from driada.network.matrix_utils import turn_to_partially_directed

    print("\n" + "=" * 60)
    print("3. Spacing ratios and localization signatures")
    print("=" * 60)

    # --- Undirected (Hermitian) case ---
    # For symmetric matrices all eigenvalues are real, so z-values collapse
    # to the real line and arg(z) is either 0 or pi.

    # get_z_values returns a dict {eigenvalue: z_value} using nearest-neighbor
    # search in the complex plane (k-d tree). Duplicate eigenvalues are removed.
    z_dict = net.get_z_values("nlap")

    # Discard the z-value for the zero eigenvalue (one per connected component).
    # It is trivial and would distort 1/|z|^2 statistics.
    z_filtered = {k: v for k, v in z_dict.items() if np.abs(k) > 1e-6}
    n_discarded = len(z_dict) - len(z_filtered)
    z_list = np.array(list(z_filtered.values()))
    z_mags = np.abs(z_list)

    print(f"\n  Total eigenvalues with z-values: {len(z_dict)}")
    print(f"  Discarded (zero eigenvalues): {n_discarded}")
    print(f"  Used for analysis: {len(z_filtered)}")

    print(f"\nComplex spacing ratios z = (lambda_nn - lambda) / (lambda_nnn - lambda):")
    print(f"  Count: {len(z_list)}")
    print(f"  Mean |z|: {np.mean(z_mags):.4f}  (bounded in [0, 1])")
    print(f"  Std |z|:  {np.std(z_mags):.4f}")

    # Localization signatures: <cos(arg(z))> measures phase coherence of
    # spacing ratios; <1/|z|^2> amplifies cases where nearest and next-nearest
    # distances differ strongly. Zero z-values (from degenerate eigenvalues)
    # are filtered internally to avoid singularities.
    mean_inv_r2, mean_cos_phi = net.localization_signatures("nlap")
    print(f"\nLocalization signatures (undirected, normalized Laplacian):")
    print(f"  <cos(arg(z))>: {mean_cos_phi:.4f}")
    print(f"  <1/|z|^2>:     {mean_inv_r2:.4f}")

    # --- Directed (non-Hermitian) case ---
    # Randomly orienting edges breaks symmetry, giving complex eigenvalues
    # and z-values that spread across the complex plane. This demonstrates
    # the full capability of the complex spacing ratio framework.
    # In practice, directed networks arise from causal or effective
    # connectivity (e.g. Granger causality, transfer entropy).
    dir_adj = turn_to_partially_directed(net.adj, directed=1.0)
    dir_net = Network(adj=dir_adj, preprocessing=None, name="Directed variant")

    z_dict_dir = dir_net.get_z_values("adj")
    z_list_dir = np.array(list(z_dict_dir.values()))
    z_mags_dir = np.abs(z_list_dir)

    print(f"\nDirected variant (randomly oriented edges):")
    print(f"  Eigenvalues: {len(z_dict_dir)} (complex)")
    print(f"  Mean |z|: {np.mean(z_mags_dir):.4f}")
    print(f"  Std |z|:  {np.std(z_mags_dir):.4f}")

    mean_inv_r2_d, mean_cos_phi_d = dir_net.localization_signatures("adj")
    print(f"\nLocalization signatures (directed adjacency):")
    print(f"  <cos(arg(z))>: {mean_cos_phi_d:.4f}")
    print(f"  <1/|z|^2>:     {mean_inv_r2_d:.4f}")

    return z_filtered, z_list_dir


def thermodynamic_analysis(net):
    """Section 4: Thermodynamic entropy, free entropy, Renyi q-entropy.

    These treat the normalized Laplacian eigenvalues as energy levels of a
    quantum system at temperature t. At low t, only the lowest eigenvalues
    contribute (local structure dominates). At high t, all eigenvalues
    contribute equally (global structure).
    """
    print("\n" + "=" * 60)
    print("4. Thermodynamic entropy analysis")
    print("=" * 60)

    # Temperature sweep: low t probes local structure, high t probes global.
    tlist = np.logspace(-2, 2, 50)

    # Von Neumann entropy S(t) = -sum_i p_i log2(p_i)
    # where p_i = exp(-lambda_i / t) / Z is the Boltzmann distribution
    # over normalized Laplacian eigenvalues.
    vn_entropy = net.calculate_thermodynamic_entropy(tlist, norm=True)
    print(f"\nVon Neumann entropy S(t) [normalized Laplacian]:")
    print(f"  At t=0.01: {vn_entropy[0]:.3f} bits")
    print(f"  At t=1.00: {vn_entropy[len(tlist) // 2]:.3f} bits")
    print(f"  At t=100:  {vn_entropy[-1]:.3f} bits")
    print(f"  Max entropy: {np.max(vn_entropy):.3f} bits"
          f"  (upper bound = log2(N) = {np.log2(net.n):.2f})")

    # Free entropy F(t) = log2(Z) where Z = sum_i exp(-lambda_i / t).
    free_ent = net.calculate_free_entropy(tlist, norm=True)
    print(f"\nFree entropy F(t) = log2(Z):")
    print(f"  At t=0.01: {free_ent[0]:.3f}")
    print(f"  At t=100:  {free_ent[-1]:.3f}")

    # Renyi q-entropy: S_q(t) = log2(sum_i p_i^q) / (1 - q).
    # At q=2, this is related to the purity of the Gibbs state.
    q_ent = net.calculate_q_entropy(q=2, tlist=tlist, norm=True)
    print(f"\nRenyi 2-entropy S_2(t):")
    print(f"  At t=0.01: {q_ent[0]:.3f} bits")
    print(f"  At t=100:  {q_ent[-1]:.3f} bits")

    return tlist, vn_entropy, free_ent, q_ent


def communicability_and_geometry(net):
    """Section 5: Estrada communicability, bipartivity, Gromov hyperbolicity.

    These metrics characterize global network topology:
    - Communicability counts weighted walks of all lengths between node pairs
    - Bipartivity measures how close the network is to having two groups
      with edges only between (not within) groups
    - Gromov hyperbolicity measures tree-likeness of the shortest-path metric
    """
    print("\n" + "=" * 60)
    print("5. Communicability and network geometry")
    print("=" * 60)

    # Estrada communicability: EE = sum_i exp(lambda_i) where lambda_i are
    # adjacency eigenvalues. Counts walks of all lengths, weighted by 1/k!.
    comm = net.calculate_estrada_communicability()
    print(f"\nEstrada communicability index: {comm:.4g}")

    # Bipartivity index: ratio of even-length to total weighted walks.
    # Uses both exp(lambda) and exp(-lambda) of the adjacency spectrum.
    bipartivity = net.get_estrada_bipartivity_index()
    print(f"Estrada bipartivity index: {bipartivity:.4f}")
    print(f"  (1.0 = perfectly bipartite, 0.0 = far from bipartite)")

    # Gromov hyperbolicity: for every 4-point set, measures how far the
    # shortest-path metric deviates from a tree metric.
    # delta = 0 means the network is a tree; larger values indicate cycles.
    hyp = net.calculate_gromov_hyperbolicity(num_samples=50000)
    print(f"\nGromov hyperbolicity (mean delta): {hyp:.3f}")
    print(f"  (0 = tree-like, higher = more cycle-rich)")

    return comm, bipartivity, hyp


def lem_embedding(net):
    """Section 6: Laplacian Eigenmaps embedding.

    LEM embeds nodes into R^d using the smallest non-zero eigenvectors of the
    normalized Laplacian. Nodes connected by edges are placed nearby in the
    embedding space. This is equivalent to minimizing
    sum_{ij} A_{ij} ||y_i - y_j||^2.
    """
    print("\n" + "=" * 60)
    print("6. Laplacian Eigenmaps embedding")
    print("=" * 60)

    dim = 3
    # LEM internally uses the normalized Laplacian and selects the dim
    # smallest non-zero eigenvectors as embedding coordinates.
    net.construct_lem_embedding(dim)

    # Access stored embedding (shape: dim x n_nodes)
    if hasattr(net.lem_emb, "toarray"):
        emb_data = np.real(net.lem_emb.toarray())
    else:
        emb_data = np.real(np.asarray(net.lem_emb))

    print(f"\nLEM embedding ({dim}D):")
    print(f"  Shape: {emb_data.shape}")
    print(f"  Dim 1 range: [{emb_data[0].min():.3f}, {emb_data[0].max():.3f}]")
    print(f"  Dim 2 range: [{emb_data[1].min():.3f}, {emb_data[1].max():.3f}]")
    print(f"  Dim 3 range: [{emb_data[2].min():.3f}, {emb_data[2].max():.3f}]")

    return emb_data


def null_model_comparison(net, real_comm, real_bipart, n_replicates=10):
    """Section 7: Null model comparison for spectral properties.

    Degree-preserving randomization (edge-swap Markov chain) generates null
    networks with the same degree sequence but randomized topology. Comparing
    real vs null spectral properties reveals structure beyond degree heterogeneity.
    """
    print("\n" + "=" * 60)
    print("7. Null model comparison (degree-preserving randomization)")
    print("=" * 60)

    # Real network properties (communicability and bipartivity passed from section 5)
    real_ipr = np.mean(net.get_ipr("nlap"))

    nlap_spectrum = np.sort(np.real(net.get_spectrum("nlap")))
    # First eigenvalue above zero is the Fiedler value
    real_fiedler = nlap_spectrum[nlap_spectrum > 1e-6][0]

    # Generate degree-preserving random networks via edge swaps
    null_comm = []
    null_bipart = []
    null_ipr = []
    null_fiedler = []

    for _ in range(n_replicates):
        rand_net = net.randomize(rmode="adj_iom")
        null_comm.append(rand_net.calculate_estrada_communicability())
        null_bipart.append(rand_net.get_estrada_bipartivity_index())
        null_ipr.append(np.mean(rand_net.get_ipr("nlap")))
        rand_nlap = np.sort(np.real(rand_net.get_spectrum("nlap")))
        nonzero = rand_nlap[rand_nlap > 1e-6]
        null_fiedler.append(nonzero[0] if len(nonzero) > 0 else 0.0)

    print(f"\n  {'Metric':<28s} {'Real':>12s} {'Null (mean +/- std)':>25s}")
    print(f"  {'-' * 67}")

    rows = [
        ("Communicability", real_comm, null_comm),
        ("Bipartivity", real_bipart, null_bipart),
        ("Mean nlap IPR", real_ipr, null_ipr),
        ("Fiedler value", real_fiedler, null_fiedler),
    ]
    for name, real_val, null_vals in rows:
        null_mean = np.mean(null_vals)
        null_std = np.std(null_vals)
        print(f"  {name:<28s} {real_val:>12.4g} {null_mean:>12.4g} +/- {null_std:.4g}")

    print(f"\n  Replicates: {n_replicates}")

    return null_comm


def visualize_results(
    net, adj_spectrum, ipr_nlap, tlist, vn_entropy, z_directed, emb_data,
    real_comm, null_comm,
):
    """Create 2x3 summary figure."""
    from scipy.stats import gaussian_kde

    print("\n" + "=" * 60)
    print("Creating summary figure")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # 1. Adjacency spectrum histogram
    ax = axes[0, 0]
    real_spec = np.real(adj_spectrum)
    nbins = int(np.ceil(np.log2(len(real_spec)))) + 1
    ax.hist(real_spec, bins=nbins, edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Count")
    ax.set_title("Adjacency spectrum")
    ax.grid(True, alpha=0.3)

    # 2. Normalized Laplacian IPR (sorted by magnitude)
    ax = axes[0, 1]
    ax.plot(np.sort(ipr_nlap), "o-", markersize=2, linewidth=0.8)
    ax.axhline(1.0 / net.n, color="r", linestyle="--", label=f"1/N = {1.0/net.n:.4f}")
    ax.set_xlabel("Eigenvector index (sorted)")
    ax.set_ylabel("IPR")
    ax.set_title("Normalized Laplacian IPR")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Thermodynamic entropy curve
    ax = axes[0, 2]
    ax.semilogx(tlist, vn_entropy, "b-", linewidth=2)
    ax.set_xlabel("Temperature (t)")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Von Neumann entropy S(t)")
    ax.grid(True, alpha=0.3)

    # 4. Complex spacing ratio density (directed network).
    # Randomly orienting edges breaks the Hermitian symmetry, giving
    # complex eigenvalues whose z-values spread across the unit disk.
    # The density pattern is a fingerprint of the spectral universality class.
    ax = axes[1, 0]
    zr, zi = np.real(z_directed), np.imag(z_directed)
    xy = np.vstack([zr, zi])
    kde = gaussian_kde(xy, bw_method=0.25)
    grid_n = 200
    pad = 0.15
    xmin, xmax = zr.min() - pad, zr.max() + pad
    ymin, ymax = zi.min() - pad, zi.max() + pad
    xg = np.linspace(xmin, xmax, grid_n)
    yg = np.linspace(ymin, ymax, grid_n)
    Xg, Yg = np.meshgrid(xg, yg)
    Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(grid_n, grid_n)
    ax.pcolormesh(Xg, Yg, Z, shading="auto", cmap="inferno")
    # Unit circle for reference (|z| <= 1)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "w--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_title("Complex spacing ratios (directed)")
    ax.set_aspect("equal")

    # 5. LEM embedding (first 2 dims)
    ax = axes[1, 1]
    ax.scatter(emb_data[0], emb_data[1], s=15, alpha=0.7, c="steelblue", edgecolors="none")
    ax.set_xlabel("LEM dim 1")
    ax.set_ylabel("LEM dim 2")
    ax.set_title("Laplacian Eigenmaps")
    ax.grid(True, alpha=0.3)

    # 6. Real vs null communicability
    ax = axes[1, 2]
    ax.hist(null_comm, bins=8, edgecolor="black", linewidth=0.5, alpha=0.6,
            color="gray", label="Null model")
    ax.axvline(real_comm, color="red", linewidth=2, label=f"Real = {real_comm:.1f}")
    ax.set_xlabel("Communicability")
    ax.set_ylabel("Count")
    ax.set_title("Real vs null model")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Run complete spectral analysis example."""

    # Build network from synthetic data
    print("Creating synthetic experiment with modular structure...")
    exp = create_modular_experiment(duration=300)
    net = build_network(exp, pval_thr=0.001)

    # Spectral analysis sections
    adj_spectrum, nlap_spectrum, n_zero = spectral_decomposition(net)
    ipr_adj, ipr_nlap = ipr_analysis(net)
    z_filtered, z_directed = spacing_ratios_and_localization(net)
    tlist, vn_entropy, free_ent, q_ent = thermodynamic_analysis(net)
    comm, bipartivity, hyp = communicability_and_geometry(net)
    emb_data = lem_embedding(net)
    null_comm = null_model_comparison(net, comm, bipartivity)

    # Visualization
    fig = visualize_results(
        net, adj_spectrum, ipr_nlap, tlist, vn_entropy, z_directed, emb_data,
        comm, null_comm,
    )
    output_dir = os.path.dirname(__file__)
    plt.savefig(
        os.path.join(output_dir, "network_spectral_analysis.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print("\n" + "=" * 60)
    print("[OK] Spectral analysis complete!")
    print("=" * 60)

    return net


if __name__ == "__main__":
    net = main()
