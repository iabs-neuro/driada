"""
Fig 5 data -- toroidal CANN with nonlinear manifold structure.

Simulates a 200-unit continuous attractor neural network (CANN) on a 2D
torus driven by two circular variables (OU random walks), then runs
DRIADA advanced analyses:
  - INTENSE selectivity detection
  - Disentanglement (PCA + AE with correlation loss)
  - Representational similarity analysis (RSA)
  - Network of Networks (NoN) via per-neuron recurrence graphs

Key design: the toroidal manifold topology requires 2 linear dimensions
per circular variable (cos, sin), so PCA-3 provably cannot represent the
full structure.  An autoencoder can learn nonlinear angle extraction
(atan2) and represent each angle in 1 dimension (Chung & Abbott, 2023).

Usage:
    python publications/elife/fig5/fig5_data.py
    python publications/elife/fig5/fig5_data.py --simulation-only

The second form stops after the simulation and writes fig5_rnn.npz alone: the
simulation takes seconds and is seeded, while the other cache files take the
whole pipeline.

Outputs: fig5_*.npz in the cache directory (see elife_paths.py)
"""

import os
import sys
import time

# Limit MKL threading overhead for FFT-heavy INTENSE
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import elife_paths

# ── Module constants ──────────────────────────────────────────────
CACHE_DIR = elife_paths.CACHE_DIR
SEED = 42
DURATION = 600          # seconds
FPS = 20.0
DS = 5                  # downsampling for INTENSE / recurrence
N_UNITS = 200
TAU = 0.2               # membrane time constant (s)
TAU_SMOOTH = 0.5        # exponential smoothing time constant (s)
OBS_NOISE = 0.01        # observation noise
NOISE_SIGMA = 0.05      # injected noise

# CANN connectivity (von Mises profile on torus)
J_EXC = 8.0             # excitatory strength
J_INH = 0.8             # global inhibition
KAPPA_CONN = 2.0        # connectivity width (von Mises concentration)

# Input tuning
KAPPA_IN = 4.0          # input tuning sharpness (von Mises concentration)

# Latent variable dynamics -- switching between positions on circle
N_POSITIONS1 = 3        # number of θ₁ attractor positions
N_POSITIONS2 = 3        # number of θ₂ attractor positions
BLOCK_MEAN1 = 10.0      # mean θ₁ block duration (s)
BLOCK_MEAN2 = 6.0       # mean θ₂ block duration (s)
TAU_TRANSITION = 0.3    # smooth angular transition time constant (s)
N_RSA_BINS = 3          # bins per angle for RSA condition labels


# =====================================================================
# 1. LATENT VARIABLE GENERATION (TOROIDAL INPUTS)
# =====================================================================
def generate_cann_inputs(n_frames, fps, rng):
    """Generate latent variables for the 2D toroidal CANN.

    Two independent circular variables that switch between discrete
    positions on [0, 2pi) with smooth transitions. The toroidal topology
    means PCA-3 needs 2 linear dims per circle (cos, sin) while an AE
    can learn nonlinear angle extraction (atan2) in 1 dim per circle.

    Returns dict with:
      theta1, theta2 : continuous circular variables (for DRIADA features)
      context, stimulus_type : position indices (for RSA conditions)
    """
    dt = 1.0 / fps
    alpha_tr = dt / TAU_TRANSITION

    def _switching_angle(n, n_pos, block_mean):
        """Angle switching between n_pos positions with smooth transitions."""
        positions = np.linspace(0, 2 * np.pi, n_pos, endpoint=False)
        switch_prob = dt / block_mean
        idx = np.zeros(n, dtype=int)
        for t in range(1, n):
            if rng.random() < switch_prob:
                idx[t] = (idx[t - 1] + rng.integers(1, n_pos)) % n_pos
            else:
                idx[t] = idx[t - 1]
        theta = np.zeros(n)
        theta[0] = positions[idx[0]]
        for t in range(1, n):
            target = positions[idx[t]]
            diff = (target - theta[t - 1] + np.pi) % (2 * np.pi) - np.pi
            theta[t] = (theta[t - 1] + alpha_tr * diff) % (2 * np.pi)
        return theta, idx

    theta1, ctx_idx = _switching_angle(n_frames, N_POSITIONS1, BLOCK_MEAN1)
    theta2, stim_idx = _switching_angle(n_frames, N_POSITIONS2, BLOCK_MEAN2)

    return {
        "theta1": theta1,
        "theta2": theta2,
        # Position indices for RSA condition labels
        "context": ctx_idx,
        "stimulus_type": stim_idx,
    }


# =====================================================================
# 2. CANN SIMULATION
# =====================================================================
def simulate_cann(inputs, rng):
    """Simulate a 2D toroidal CANN with structured connectivity.

    Each neuron has a preferred position (phi1, phi2) on the torus.
    Connectivity is translation-invariant von Mises; external input
    positions a bump at (theta1, theta2) with amplitude A.

    Returns (activations, phi1, phi2) where phi are preferred positions.
    """
    dt = 1.0 / FPS
    n_frames = len(inputs["theta1"])

    # Assign preferred positions on the torus
    phi1 = rng.uniform(0, 2 * np.pi, N_UNITS)
    phi2 = rng.uniform(0, 2 * np.pi, N_UNITS)

    # Structured connectivity: local excitation + global inhibition
    dp1 = phi1[:, None] - phi1[None, :]  # (N, N)
    dp2 = phi2[:, None] - phi2[None, :]
    W_rec = (J_EXC / N_UNITS) * np.exp(
        KAPPA_CONN * (np.cos(dp1) + np.cos(dp2) - 2)
    ) - J_INH / N_UNITS

    theta1 = inputs["theta1"]
    theta2 = inputs["theta2"]

    state = np.zeros(N_UNITS)
    raw = np.empty((N_UNITS, n_frames))

    for t in range(n_frames):
        r = np.maximum(state, 0)  # ReLU
        raw[:, t] = r

        # External input: von Mises bump centred at (theta1, theta2)
        d1 = theta1[t] - phi1
        d2 = theta2[t] - phi2
        I_ext = np.exp(
            KAPPA_IN * (np.cos(d1) + np.cos(d2) - 2)
        )

        noise = rng.normal(0, NOISE_SIGMA, N_UNITS)
        state += (dt / TAU) * (-state + W_rec @ r + I_ext + noise)

    if np.any(np.isnan(raw)) or np.max(raw) > 1e6:
        raise RuntimeError("CANN diverged -- adjust J_EXC / J_INH")

    # Exponential smoothing (mimics slow indicator dynamics)
    alpha = dt / TAU_SMOOTH
    activations = np.empty_like(raw)
    activations[:, 0] = raw[:, 0]
    for t in range(1, n_frames):
        activations[:, t] = (1 - alpha) * activations[:, t - 1] + alpha * raw[:, t]

    # Add observation noise, clamp non-negative
    activations += rng.normal(0, OBS_NOISE, activations.shape)
    return np.maximum(activations, 0), phi1, phi2


# =====================================================================
# 3. EXPERIMENT CONSTRUCTION
# =====================================================================
def build_experiment(activations, inputs):
    """Wrap CANN activations and inputs into a DRIADA Experiment.

    Registers theta1 and theta2 as circular features (auto-creates _2d
    cos/sin pairs for INTENSE).
    """
    from driada.experiment import load_exp_from_aligned_data

    data = {
        "activations": activations,
        "theta1": inputs["theta1"],
        "theta2": inputs["theta2"],
    }

    exp = load_exp_from_aligned_data(
        data_source="CANN",
        exp_params={"name": "toroidal_cann"},
        data=data,
        feature_types={
            "theta1": "circular",
            "theta2": "circular",
        },
        static_features={"fps": float(FPS)},
        verbose=True,
        seed=SEED,
    )
    return exp


# =====================================================================
# 4. INTENSE SELECTIVITY
# =====================================================================
def run_intense(exp):
    """Run INTENSE selectivity detection on the RNN experiment.

    Returns
    -------
    mi_matrix : ndarray (n_neurons, n_features)
    sig_matrix : ndarray (n_neurons, n_features)
    feat_ids : list of str
    selectivity_labels : list of str
    """
    from driada.intense.pipelines import compute_cell_feat_significance

    print("\n" + "=" * 60)
    print("Running INTENSE selectivity detection...")

    stats, significance, info, intense_res, _ = compute_cell_feat_significance(
        exp,
        data_type="calcium",
        mi_estimator="gcmi",
        mode="two_stage",
        n_shuffles_stage1=100,
        n_shuffles_stage2=10000,
        ds=DS,
        pval_thr=0.001,
        multicomp_correction="holm",
        verbose=True,
        seed=SEED,
    )

    # Extract matrices from info dict
    sig_tables = info["stage_2_significance"]
    sig_matrix = sig_tables["stage2"].astype(float)

    # MI values: me_total2 shape (n_neurons, n_features, n_shuffles+1)
    # Index 0 is the real data MI
    mi_matrix = info["me_total2"][:, :, 0]

    # Feature names in column order: the features INTENSE actually tested
    # (circular features are replaced by their _2d versions), not all
    # dynamic features of the experiment
    feat_ids = list(next(iter(significance.values())).keys())
    if sig_matrix.shape[1] != len(feat_ids):
        raise ValueError(f"{sig_matrix.shape[1]} columns but features {feat_ids}")
    for i, cell in enumerate(significance):
        for j, f in enumerate(feat_ids):
            if bool(sig_matrix[i, j]) != bool(significance[cell][f].get("stage2", False)):
                raise ValueError(f"column order mismatch at neuron {cell}, feature {f}")

    # Per-neuron selectivity labels
    n_neurons = mi_matrix.shape[0]
    selectivity_labels = []
    for i in range(n_neurons):
        sig_feats = [feat_ids[j] for j in range(len(feat_ids))
                     if sig_matrix[i, j] > 0]
        if len(sig_feats) == 0:
            selectivity_labels.append("none")
        elif len(sig_feats) == 1:
            selectivity_labels.append(sig_feats[0])
        else:
            selectivity_labels.append("mixed")

    n_selective = sum(1 for lbl in selectivity_labels if lbl != "none")
    n_mixed = sum(1 for lbl in selectivity_labels if lbl == "mixed")
    print(f"\nShape: {mi_matrix.shape} (neurons x features)")
    print(f"Features: {feat_ids}")
    print(f"Significant pairs: {int(np.sum(sig_matrix))}")
    print(f"Selective units: {n_selective}/{n_neurons}, mixed: {n_mixed}")

    return mi_matrix, sig_matrix, feat_ids, selectivity_labels


# =====================================================================
# 5. DISENTANGLEMENT: PCA + beta-VAE
# =====================================================================
def _circ_lin_r2(x, theta):
    """R^2 of linear fit: x = a*cos(theta) + b*sin(theta) + c."""
    X = np.column_stack([np.cos(theta), np.sin(theta), np.ones_like(theta)])
    beta = np.linalg.lstsq(X, x, rcond=None)[0]
    pred = X @ beta
    ss_res = np.sum((x - pred) ** 2)
    ss_tot = np.sum((x - x.mean()) ** 2)
    return max(0.0, 1.0 - ss_res / ss_tot)


def _lin_r2(x, y):
    """R^2 of simple linear fit: x = a*y + b."""
    X = np.column_stack([y, np.ones_like(y)])
    beta = np.linalg.lstsq(X, x, rcond=None)[0]
    pred = X @ beta
    ss_res = np.sum((x - pred) ** 2)
    ss_tot = np.sum((x - x.mean()) ** 2)
    return max(0.0, 1.0 - ss_res / ss_tot)


def _compute_separation_metrics(coords, inputs):
    """Compute disentanglement metrics for an embedding.

    Uses circular-linear R^2 for the two circular variables (theta1, theta2).
    This correctly captures the toroidal topology that PCA-3 cannot represent
    (each circle needs 2 linear dims, but AE needs only 1 via atan2).

    Parameters
    ----------
    coords : ndarray (n_dims, n_samples)
    inputs : dict with theta1, theta2 (continuous circular latent vars)

    Returns
    -------
    dict with keys:
        r2_theta1, r2_theta2 : best-dim circular-linear R^2 per variable
        disentanglement : DCI-like score (0=entangled, 1=factored)
        total_r2 : greedy-assigned total R^2 (no dim reuse)
        dim_pair_theta1, dim_pair_theta2 : (x, y) tuples for visualization
    """
    from scipy.stats import entropy

    theta1 = inputs["theta1"]
    theta2 = inputs["theta2"]
    ndim = coords.shape[0]

    # Build R^2 matrix (n_dims x 2 variables)
    r2_mat = np.zeros((ndim, 2))
    for d in range(ndim):
        r2_mat[d, 0] = _circ_lin_r2(coords[d], theta1)
        r2_mat[d, 1] = _circ_lin_r2(coords[d], theta2)

    # Disentanglement score (DCI-like): low entropy = specialized dims
    dim_entropies = []
    for d in range(ndim):
        total = r2_mat[d].sum()
        if total < 0.01:
            dim_entropies.append(np.log(2))
            continue
        p = r2_mat[d] / total
        dim_entropies.append(entropy(p))
    D = 1.0 - np.mean(dim_entropies) / np.log(2)

    # Best-dim R^2 per variable
    best_per_var = r2_mat.max(axis=0)

    # Greedy dim-variable assignment (no reuse)
    used_dims, assigned = set(), {}
    for _ in range(min(ndim, 2)):
        best_val, best_dv = -1, None
        for d in range(ndim):
            if d in used_dims:
                continue
            for v in range(2):
                if v in assigned:
                    continue
                if r2_mat[d, v] > best_val:
                    best_val = r2_mat[d, v]
                    best_dv = (d, v)
        if best_dv:
            d, v = best_dv
            used_dims.add(d)
            assigned[v] = (d, best_val)

    total_r2 = sum(val for _, val in assigned.values())

    # Best dim pairs for visualization
    def _best_dim_pair(r2_col):
        y_dim = int(np.argmax(r2_col))
        remaining_idx = [i for i in range(ndim) if i != y_dim]
        x_dim = (remaining_idx[int(np.argmax(r2_col[remaining_idx]))]
                 if remaining_idx else 0)
        return (x_dim, y_dim)

    return dict(
        r2_theta1=best_per_var[0],
        r2_theta2=best_per_var[1],
        disentanglement=D,
        total_r2=total_r2,
        dim_pair_theta1=_best_dim_pair(r2_mat[:, 0]),
        dim_pair_theta2=_best_dim_pair(r2_mat[:, 1]),
    )


def _derive_component_feature_mapping(selectivity_labels, emb_sel_results, method):
    """Map each embedding component to the behavioral feature it encodes.

    For each component, find which behavioral feature's selective neurons
    overlap most with the component's selective neurons.

    Parameters
    ----------
    selectivity_labels : list of str
        Per-neuron selectivity labels from INTENSE (e.g. 'context', 'none', 'mixed').
    emb_sel_results : dict
        Results for one method from compute_embedding_selectivity.
    method : str
        Embedding method name (e.g. 'pca', 'flexible_ae').

    Returns
    -------
    dict : {comp_idx: (feature_name, overlap_fraction)}
    """
    component_selectivity = emb_sel_results["component_selectivity"]
    # Get unique behavioral feature names (excluding 'none' and 'mixed')
    feat_names = sorted(set(selectivity_labels) - {"none", "mixed"})

    # Build per-feature neuron sets
    feat_neurons = {}
    for feat in feat_names:
        feat_neurons[feat] = {i for i, lbl in enumerate(selectivity_labels)
                              if lbl == feat}

    mapping = {}
    for comp_idx, comp_neuron_list in component_selectivity.items():
        comp_set = set(comp_neuron_list)
        if len(comp_set) == 0:
            mapping[comp_idx] = ("none", 0.0)
            continue
        best_feat, best_overlap = "none", 0.0
        for feat, fneurons in feat_neurons.items():
            overlap = len(comp_set & fneurons) / len(comp_set) if comp_set else 0.0
            if overlap > best_overlap:
                best_overlap = overlap
                best_feat = feat
        mapping[comp_idx] = (best_feat, best_overlap)

    return mapping


def run_disentanglement(exp, inputs, selectivity_labels):
    """Run PCA, AE (recon-only), and AE+corr (3D each) via DRIADA workflow.

    Uses exp.create_embedding, compute_embedding_selectivity, and
    get_functional_organization instead of direct MVData calls.

    Returns
    -------
    pca_coords : ndarray (3, n_samples)
    ae_recon_coords : ndarray (3, n_samples)
    ae_coords : ndarray (3, n_samples)
    metrics : dict  -- separation metrics for all methods
    dim_pairs : dict -- per-method, per-variable best dim pairs
    func_org_summary : dict -- functional organization summaries per method
    """
    from driada.intense.pipelines import compute_embedding_selectivity
    from driada.integration.manifold_analysis import get_functional_organization

    print("\n" + "=" * 60)
    print("Running disentanglement analysis (DRIADA-native)...")

    # ── 1. Create embeddings on the Experiment ──
    print("  PCA (3D)...")
    pca_emb = exp.create_embedding("pca", n_components=3)
    pca_coords = pca_emb.T  # (3, n_frames) for cache compat
    print(f"    PCA embedding: {pca_coords.shape}")

    print("  AE recon-only (3D, no disentanglement loss)...")
    ae_recon_emb = exp.create_embedding(
        "flexible_ae",
        n_components=3,
        architecture="ae",
        inter_dim=128,
        loss_components=[
            {"name": "reconstruction", "weight": 1.0, "loss_type": "mse"},
        ],
        epochs=200,
        lr=1e-3,
        batch_size=512,
        verbose=True,
        seed=SEED,
    )
    ae_recon_coords = ae_recon_emb.T  # (3, n_frames)
    print(f"    AE recon embedding: {ae_recon_coords.shape}")

    print("  AE + correlation loss (3D, corr_weight=1.0)...")
    ae_emb = exp.create_embedding(
        "flexible_ae",
        n_components=3,
        architecture="ae",
        inter_dim=128,
        loss_components=[
            {"name": "reconstruction", "weight": 1.0, "loss_type": "mse"},
            {"name": "correlation", "weight": 1.0},
        ],
        epochs=200,
        lr=1e-3,
        batch_size=512,
        verbose=True,
        seed=SEED,
    )
    ae_coords = ae_emb.T  # (3, n_frames)
    print(f"    AE+corr embedding: {ae_coords.shape}")

    # ── 2. Embedding selectivity via INTENSE ──
    print("  Running embedding selectivity (INTENSE)...")
    emb_sel = compute_embedding_selectivity(
        exp,
        embedding_methods=["pca", "flexible_ae"],
        mode="two_stage",
        n_shuffles_stage1=100,
        n_shuffles_stage2=1000,
        ds=DS,
        pval_thr=0.01,
        seed=SEED,
        verbose=True,
    )

    # ── 3. Functional organization ──
    print("  Computing functional organization...")
    func_org_summary = {}
    for method in ["pca", "flexible_ae"]:
        if method in emb_sel:
            intense_res = emb_sel[method]["intense_results"]
            func_org = get_functional_organization(
                exp, method, intense_results=intense_res,
            )
            func_org_summary[method] = func_org
            n_part = func_org.get("n_participating_neurons", 0)
            n_clust = len(func_org.get("functional_clusters", []))
            print(f"    {method}: {n_part} participating neurons, "
                  f"{n_clust} functional clusters")

    # ── 4. Component → feature mapping ──
    print("  Deriving component-feature mapping...")
    comp_feat_maps = {}
    for method in ["pca", "flexible_ae"]:
        if method in emb_sel:
            cfm = _derive_component_feature_mapping(
                selectivity_labels, emb_sel[method], method,
            )
            comp_feat_maps[method] = cfm
            for comp_idx, (feat, overlap) in cfm.items():
                print(f"    {method} comp{comp_idx} -> {feat} "
                      f"(overlap={overlap:.2f})")

    # ── 5. Disentanglement metrics + dim pairs ──
    print("  Computing disentanglement metrics (circular-linear R^2)...")
    m_pca = _compute_separation_metrics(pca_coords, inputs)
    m_ae_recon = _compute_separation_metrics(ae_recon_coords, inputs)
    m_ae = _compute_separation_metrics(ae_coords, inputs)

    # Scalar metrics for cache
    metrics = {}
    for prefix, m in [("pca", m_pca), ("ae_recon", m_ae_recon), ("ae", m_ae)]:
        for k, v in m.items():
            if not k.startswith("dim_pair_"):
                metrics[f"{prefix}_{k}"] = v

    # Dim pairs per method per variable
    dim_pairs = {
        "pca": {
            "theta1": m_pca["dim_pair_theta1"],
            "theta2": m_pca["dim_pair_theta2"],
        },
        "ae_recon": {
            "theta1": m_ae_recon["dim_pair_theta1"],
            "theta2": m_ae_recon["dim_pair_theta2"],
        },
        "ae": {
            "theta1": m_ae["dim_pair_theta1"],
            "theta2": m_ae["dim_pair_theta2"],
        },
    }

    for label, m in [("PCA", m_pca), ("AE-recon", m_ae_recon), ("AE+corr", m_ae)]:
        print(f"    {label:>8} -- R2(t1)={m['r2_theta1']:.3f}, "
              f"R2(t2)={m['r2_theta2']:.3f}, D={m['disentanglement']:.3f}")

    for method_short, method_pairs in dim_pairs.items():
        for var, (x, y) in method_pairs.items():
            print(f"    {method_short} {var}: x=dim{x}, y=dim{y}")

    return pca_coords, ae_recon_coords, ae_coords, metrics, dim_pairs, \
        func_org_summary


# =====================================================================
# 6. RSA: REPRESENTATIONAL SIMILARITY ANALYSIS
# =====================================================================
def run_rsa(exp, inputs):
    """Compute RDM over context × stimulus conditions (9 conditions).

    Returns
    -------
    rdm : ndarray (9, 9)
    unique_labels : ndarray of str
    """
    from driada.rsa import compute_rdm_from_timeseries_labels

    print("\n" + "=" * 60)
    print("Running RSA...")

    calcium = exp.calcium.data
    context = inputs["context"]
    stimulus_type = inputs["stimulus_type"]

    ctx_names = ["A", "B", "C"]
    stim_names = [f"S{i}" for i in range(N_RSA_BINS)]

    labels = np.array([
        f"{ctx_names[context[t]]}_{stim_names[stimulus_type[t]]}"
        for t in range(len(context))
    ])

    rdm, unique_labels = compute_rdm_from_timeseries_labels(
        calcium, labels, metric="correlation",
    )

    print(f"  RDM shape: {rdm.shape}")
    print(f"  Condition labels: {list(unique_labels)}")

    return rdm, unique_labels


# =====================================================================
# 7. NETWORK OF NETWORKS (NoN): RECURRENCE + JACCARD + LOUVAIN
# =====================================================================
def run_non(exp):
    """Per-neuron recurrence graphs -> Jaccard similarity -> Louvain communities.

    Returns
    -------
    jaccard : ndarray (n_neurons, n_neurons)
    community_labels : ndarray (n_neurons,)
    """
    from driada.recurrence import (
        estimate_tau,
        estimate_embedding_dim,
        takens_embedding,
        RecurrenceGraph,
        pairwise_jaccard_sparse,
    )
    from joblib import Parallel, delayed
    import networkx.algorithms.community as nx_comm

    print("\n" + "=" * 60)
    print("Running Network of Networks (recurrence)...")

    calcium = exp.calcium.data  # (N_UNITS, n_frames)

    # Downsample calcium by DS
    calcium_ds = calcium[:, ::DS]
    n_neurons, n_samples = calcium_ds.shape
    print(f"  Downsampled: {calcium_ds.shape} (ds={DS})")

    def _build_rg(neuron_ts):
        """Build recurrence graph for a single neuron."""
        tau = estimate_tau(neuron_ts, max_shift=50)
        m = estimate_embedding_dim(neuron_ts, tau, max_dim=8)
        emb = takens_embedding(neuron_ts, tau, m)
        rg = RecurrenceGraph(emb, method="knn", k=20)
        return rg

    print(f"  Building {n_neurons} recurrence graphs (joblib threading)...")
    t0 = time.time()
    rg_list = Parallel(n_jobs=-1, backend="threading")(
        delayed(_build_rg)(calcium_ds[i]) for i in range(n_neurons)
    )
    print(f"  Recurrence graphs built in {time.time() - t0:.1f}s")

    # Trim all adjacency matrices to the minimum size
    import scipy.sparse as sp
    sizes = [rg.adj.shape[0] for rg in rg_list]
    min_size = min(sizes)
    print(f"  Embedded sizes: min={min_size}, max={max(sizes)}")

    adj_list = []
    for rg in rg_list:
        adj = rg.adj
        if adj.shape[0] > min_size:
            adj = adj[:min_size, :min_size]
        adj_list.append(adj)

    print("  Computing pairwise Jaccard similarity...")
    t0 = time.time()
    # Graphs are already trimmed to a common size above; newer DRIADA also
    # returns the mask of kept graphs
    jaccard, _ = pairwise_jaccard_sparse(adj_list, trim=None)
    print(f"  Jaccard computed in {time.time() - t0:.1f}s, shape: {jaccard.shape}")

    # Threshold at 85th percentile to build a denser network
    upper_tri = jaccard[np.triu_indices(n_neurons, k=1)]
    threshold = np.percentile(upper_tri, 85)
    adj_dense = (jaccard >= threshold).astype(float)
    np.fill_diagonal(adj_dense, 0)
    adj_non = sp.csr_matrix(adj_dense)

    from driada.network import Network
    net = Network(adj=adj_non, preprocessing="giant_cc", create_nx_graph=True)
    print(f"  NoN: {net.n} nodes (giant CC), {net.adj.nnz} edges")

    # Louvain community detection
    communities = nx_comm.louvain_communities(net.graph, seed=SEED)
    communities = sorted(communities, key=len, reverse=True)
    n_modules = len(communities)

    # Map back to full neuron indices
    community_labels = np.full(n_neurons, -1, dtype=int)
    for mod_idx, comm in enumerate(communities):
        for node in comm:
            community_labels[node] = mod_idx

    n_assigned = np.sum(community_labels >= 0)
    print(f"  Communities: {n_modules} (assigned {n_assigned}/{n_neurons} neurons)")
    print(f"  Sizes: {', '.join(str(len(c)) for c in communities)}")

    # Spring layout for graph visualization (spectral init for stability)
    import networkx as nx
    pos_init = nx.spectral_layout(net.graph)
    pos_dict = nx.spring_layout(net.graph, pos=pos_init, seed=SEED,
                                k=4.0/np.sqrt(net.n), iterations=200)
    graph_nodes = np.array(sorted(pos_dict.keys()))
    graph_pos = np.array([pos_dict[n] for n in graph_nodes])
    edges_arr = np.array(list(net.graph.edges()))

    return jaccard, community_labels, graph_nodes, graph_pos, edges_arr


# =====================================================================
# 8. CACHE SAVE + MAIN
# =====================================================================
def save_rnn_cache(activations, phi1, phi2, inputs):
    """Save the simulated activity and the latent variables."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(
        os.path.join(CACHE_DIR, "fig5_rnn.npz"),
        activations=activations,
        phi1=phi1,
        phi2=phi2,
        theta1=inputs["theta1"],
        theta2=inputs["theta2"],
        context=inputs["context"],
        stimulus_type=inputs["stimulus_type"],
        n_units=N_UNITS,
        duration=DURATION,
        fps=FPS,
    )


def save_cache(activations, phi1, phi2, inputs, mi_matrix, sig_matrix,
               feat_ids, selectivity_labels, pca_coords, ae_recon_coords,
               ae_coords, dis_metrics, dim_pairs, func_org_summary,
               rdm, rdm_labels, jaccard, community_labels,
               graph_nodes, graph_pos, graph_edges):
    """Save all results to NPZ cache files."""
    save_rnn_cache(activations, phi1, phi2, inputs)

    # INTENSE results
    np.savez(
        os.path.join(CACHE_DIR, "fig5_intense.npz"),
        mi_mat=mi_matrix,
        sig_mat=sig_matrix,
        feat_ids=np.array(feat_ids, dtype=object),
        selectivity_labels=np.array(selectivity_labels, dtype=object),
    )

    # Disentanglement results (PCA + AE-recon + AE+corr + metrics + dim pairs)
    vae_data = dict(
        pca_coords=pca_coords,
        ae_recon_coords=ae_recon_coords,
        vae_coords=ae_coords,
        **dis_metrics,
    )
    # Add dim pairs as int arrays
    for method_short, method_pairs in dim_pairs.items():
        for var_name, (x, y) in method_pairs.items():
            vae_data[f"dim_pair_{method_short}_{var_name}"] = np.array([x, y])

    # Add component-feature maps as string arrays
    for method, cfm in func_org_summary.items():
        method_short = "pca" if method == "pca" else "ae"
        comp_spec = cfm.get("component_specialization", {})
        entries = []
        for comp_idx in sorted(comp_spec.keys()):
            n_sel = comp_spec[comp_idx]["n_selective_neurons"]
            entries.append(f"comp{comp_idx}:{n_sel}sel")
        vae_data[f"comp_feat_map_{method_short}"] = np.array(entries, dtype=object)

    np.savez(os.path.join(CACHE_DIR, "fig5_vae.npz"), **vae_data)

    # RSA results
    np.savez(
        os.path.join(CACHE_DIR, "fig5_rsa.npz"),
        rdm=rdm,
        rdm_labels=rdm_labels,
    )

    # Network of Networks results
    np.savez(
        os.path.join(CACHE_DIR, "fig5_non.npz"),
        jaccard=jaccard,
        community_labels=community_labels,
        graph_nodes=graph_nodes,
        graph_pos=graph_pos,
        graph_edges=graph_edges,
    )

    print(f"\nAll cache files saved to {CACHE_DIR}/")


def main():
    """Run full Fig 5 data pipeline."""
    print("=" * 70)
    print("Fig 5 -- Toroidal CANN analysis pipeline")
    print("=" * 70)

    rng = np.random.default_rng(SEED)
    n_frames = int(DURATION * FPS)
    t_total = time.time()

    # [1] Generate latent variables
    print("\n[1] GENERATING CANN INPUTS (toroidal latent variables)")
    print("-" * 40)
    inputs = generate_cann_inputs(n_frames, FPS, rng)
    ctx_counts = np.bincount(inputs["context"], minlength=N_RSA_BINS)
    stim_counts = np.bincount(inputs["stimulus_type"], minlength=N_RSA_BINS)
    print(f"  Frames: {n_frames}")
    print(f"  theta1 range: [{inputs['theta1'].min():.2f}, {inputs['theta1'].max():.2f}]")
    print(f"  theta2 range: [{inputs['theta2'].min():.2f}, {inputs['theta2'].max():.2f}]")
    print(f"  RSA bin counts (theta1): {', '.join(f'{c}={n}' for c, n in enumerate(ctx_counts))}")
    print(f"  RSA bin counts (theta2): {', '.join(f'{c}={n}' for c, n in enumerate(stim_counts))}")

    # [2] Simulate CANN
    print("\n[2] SIMULATING TOROIDAL CANN")
    print("-" * 40)
    activations, phi1, phi2 = simulate_cann(inputs, rng)
    print(f"  Activations: {activations.shape}")
    print(f"  Mean rate: {activations.mean():.3f}, "
          f"fraction active: {(activations > 0).mean():.2f}")
    if "--simulation-only" in sys.argv[1:]:
        save_rnn_cache(activations, phi1, phi2, inputs)
        print(f"\nSaved fig5_rnn.npz to {CACHE_DIR}/")
        return

    # [3] Load into DRIADA (circular features → auto _2d conversion)
    print("\n[3] LOADING INTO DRIADA EXPERIMENT")
    print("-" * 40)
    exp = build_experiment(activations, inputs)
    print(f"  Experiment: {exp.n_cells} units, {exp.n_frames} frames")
    print(f"  Features: {list(exp.dynamic_features.keys())}")

    # [4] INTENSE selectivity
    mi_matrix, sig_matrix, feat_ids, selectivity_labels = run_intense(exp)

    # [5] Disentanglement (PCA + AE-recon + AE+corr) -- DRIADA native workflow
    pca_coords, ae_recon_coords, ae_coords, dis_metrics, dim_pairs, func_org = \
        run_disentanglement(exp, inputs, selectivity_labels)

    # [6] RSA
    rdm, rdm_labels = run_rsa(exp, inputs)

    # [7] Network of Networks
    jaccard, community_labels, graph_nodes, graph_pos, graph_edges = run_non(exp)

    # [8] Save cache
    print("\n[8] SAVING CACHE")
    print("-" * 40)
    save_cache(
        activations, phi1, phi2, inputs,
        mi_matrix, sig_matrix, feat_ids, selectivity_labels,
        pca_coords, ae_recon_coords, ae_coords,
        dis_metrics, dim_pairs, func_org,
        rdm, rdm_labels,
        jaccard, community_labels,
        graph_nodes, graph_pos, graph_edges,
    )

    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"Fig 5 pipeline COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
