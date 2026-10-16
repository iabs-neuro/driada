"""
Cell-cell significance matrices of the functional networks in Fig 4:
real data vs shuffled control.

  1. INTENSE-based functional network from real calcium data
  2. Same pipeline applied to roll-based shuffled calcium data

Writes network_real_<session>.npz and network_shuffled_<session>.npz to the
cache directory and a plain side-by-side drawing of the two graphs to the
output directory.

Data source: FOF_F28_1D session.

Run:  python publications/elife/compute/compute_cell_cell_network.py
"""

import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import elife_paths
elife_paths.add_tools_to_path()

SYNC_DIR = elife_paths.SYNC_DIR
SESSION = "FOF_F28_1D"

from driada.utils.plot import make_beautiful
from driada.intense.pipelines import compute_cell_cell_significance
from driada.experiment.exp_build import load_exp_from_aligned_data
from load_synchronized_experiments import load_experiment_from_npz
from pathlib import Path

MB_KWARGS = dict(
    spine_width=1.5, tick_width=1.5, tick_length=4,
    tick_pad=5, tick_labelsize=9, label_size=10, title_size=11,
)

# ---------------------------------------------------------------------------
# 1. Load experiment
# ---------------------------------------------------------------------------
print("Loading experiment...")
t0 = time.perf_counter()

exp = load_experiment_from_npz(
    Path(SYNC_DIR) / f"{SESSION}_aligned.npz", verbose=True,
)
calcium = exp.calcium.data  # (n_neurons, n_frames)
n_neurons, n_frames = calcium.shape
print(f"  {n_neurons} neurons, {n_frames} frames")

# Use all neurons
cell_bunch = None

# ---------------------------------------------------------------------------
# 2. Compute INTENSE cell-cell significance on REAL data
# ---------------------------------------------------------------------------
print("\nComputing INTENSE cell-cell significance (real data)...")
t1 = time.perf_counter()

sim_matrix, sig_matrix, pval_matrix, cell_ids, info = compute_cell_cell_significance(
    exp,
    cell_bunch=cell_bunch,
    data_type='calcium',
    ds=5,
    n_shuffles_stage1=100,
    n_shuffles_stage2=1000,
    pval_thr=0.001,
    multicomp_correction=None,
    verbose=True,
    seed=42,
)
t_real = time.perf_counter() - t1
print(f"  Real data INTENSE took {t_real:.1f}s")

# Save real results to disk immediately
os.makedirs(elife_paths.CACHE_DIR, exist_ok=True)
out_real_npz = os.path.join(elife_paths.CACHE_DIR, f"network_real_{SESSION}.npz")
np.savez_compressed(out_real_npz,
    sim_matrix=sim_matrix, sig_matrix=sig_matrix,
    pval_matrix=pval_matrix, cell_ids=cell_ids)
print(f"  Saved real matrices -> {out_real_npz}")

# Build networkx graph from binary significance matrix
G_real = nx.from_numpy_array(np.triu(sig_matrix, k=1))
isolates_real = list(nx.isolates(G_real))
G_real.remove_nodes_from(isolates_real)


def community_layout(G, seed=42, spread=3.0):
    """Spring layout with community centers placed on a circle for separation."""
    comms = nx.community.louvain_communities(G, seed=seed, resolution=1.0)
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, len(comms), endpoint=False)
    rng.shuffle(angles)
    centers = {i: spread * np.array([np.cos(a), np.sin(a)])
               for i, a in enumerate(angles)}
    pos = {}
    for cid, members in enumerate(comms):
        sub = G.subgraph(members)
        sub_pos = nx.spring_layout(
            sub, seed=seed + cid,
            k=1.5 / np.sqrt(max(len(members), 1)),
            iterations=200,
        )
        for node, xy in sub_pos.items():
            pos[node] = xy + centers[cid]
    pos = nx.spring_layout(G, pos=pos, seed=seed, k=0.3, iterations=15)
    return pos


pos_real = community_layout(G_real, seed=42, spread=3.0)
print(f"  Real network: {G_real.number_of_nodes()} nodes, "
      f"{G_real.number_of_edges()} edges  "
      f"(removed {len(isolates_real)} isolates)")

# ---------------------------------------------------------------------------
# 3. Build shuffled experiment and compute INTENSE on it
# ---------------------------------------------------------------------------
print("\nBuilding shuffled experiment...")
t2 = time.perf_counter()

# Get shuffled calcium and static features BEFORE freeing memory
calcium_shuf = exp.get_multicell_shuffled_calcium(method='roll_based')
static_features = {}
if hasattr(exp, 'static_features') and 'fps' in exp.static_features:
    static_features['fps'] = exp.static_features['fps']
else:
    static_features['fps'] = 20.0

# Free the real experiment to reclaim memory
import gc
del exp, calcium, sim_matrix, pval_matrix, cell_ids, info
gc.collect()

# Load aligned data to reconstruct a shuffled experiment
aligned = np.load(os.path.join(SYNC_DIR, f"{SESSION}_aligned.npz"), allow_pickle=True)
data_dict = {key: aligned[key] for key in aligned.keys()
             if not key.startswith('_')}
data_dict['Calcium'] = calcium_shuf  # replace with shuffled calcium
data_dict.pop('calcium', None)

exp_shuf = load_exp_from_aligned_data(
    data_source='IABS',
    exp_params={'track': 'FOF', 'animal_id': 'F28', 'session': '1D_shuffled'},
    data=data_dict,
    static_features=static_features,
    verbose=False,
)
del aligned, data_dict, calcium_shuf
gc.collect()

print(f"  Shuffled experiment: {len(exp_shuf.neurons)} neurons, "
      f"{exp_shuf.calcium.data.shape[1]} frames")

print("\nComputing INTENSE cell-cell significance (shuffled data)...")
t3 = time.perf_counter()

sim_shuf, sig_shuf, pval_shuf, cell_ids_shuf, info_shuf = compute_cell_cell_significance(
    exp_shuf,
    cell_bunch=cell_bunch,
    data_type='calcium',
    ds=5,
    n_shuffles_stage1=100,
    n_shuffles_stage2=1000,
    pval_thr=0.001,
    multicomp_correction=None,
    verbose=True,
    seed=42,
)
t_shuf = time.perf_counter() - t3
print(f"  Shuffled data INTENSE took {t_shuf:.1f}s")

# Save shuffled results to disk
out_shuf_npz = os.path.join(elife_paths.CACHE_DIR, f"network_shuffled_{SESSION}.npz")
np.savez_compressed(out_shuf_npz,
    sim_matrix=sim_shuf, sig_matrix=sig_shuf,
    pval_matrix=pval_shuf, cell_ids=cell_ids_shuf)
print(f"  Saved shuffled matrices -> {out_shuf_npz}")

# Build networkx graph from binary significance matrix
G_shuf = nx.from_numpy_array(np.triu(sig_shuf, k=1))
isolates_shuf = list(nx.isolates(G_shuf))
G_shuf.remove_nodes_from(isolates_shuf)
pos_shuf = community_layout(G_shuf, seed=42, spread=3.0)
print(f"  Shuffled network: {G_shuf.number_of_nodes()} nodes, "
      f"{G_shuf.number_of_edges()} edges  "
      f"(removed {len(isolates_shuf)} isolates)")

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
print("\nGenerating figure...")

COLOR_REAL = "#4C72B0"      # steelblue
COLOR_SHUFFLED = "#DD8452"  # muted orange

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# ---- Left: real network ---------------------------------------------------
nx.draw_networkx_edges(G_real, pos_real, ax=ax1, alpha=0.08, width=0.3,
                       edge_color="gray")
nx.draw_networkx_nodes(G_real, pos_real, ax=ax1,
                       node_size=12, node_color=COLOR_REAL,
                       alpha=0.85, linewidths=0)
ax1.set_title("Functional network (real)", fontsize=11, fontweight="bold")
ax1.axis("off")

# ---- Right: shuffled network ----------------------------------------------
nx.draw_networkx_edges(G_shuf, pos_shuf, ax=ax2, alpha=0.08, width=0.3,
                       edge_color="gray")
nx.draw_networkx_nodes(G_shuf, pos_shuf, ax=ax2,
                       node_size=12, node_color=COLOR_SHUFFLED,
                       alpha=0.85, linewidths=0)
ax2.set_title("Functional network (shuffled)", fontsize=11, fontweight="bold")
ax2.axis("off")

fig.tight_layout()

# ---------------------------------------------------------------------------
# 5. Print summary statistics
# ---------------------------------------------------------------------------
print("\n--- Network statistics ---")
print(f"  Real:     {G_real.number_of_nodes()} nodes, "
      f"{G_real.number_of_edges()} edges  "
      f"(pval_thr=0.001, no correction)")
print(f"  Shuffled: {G_shuf.number_of_nodes()} nodes, "
      f"{G_shuf.number_of_edges()} edges  "
      f"(pval_thr=0.001, no correction)")

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
out_png = elife_paths.out_path("network_real_vs_shuffled.png")
out_pdf = elife_paths.out_path("network_real_vs_shuffled.pdf")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.close(fig)

total_time = time.perf_counter() - t0
print(f"\nSaved {out_png}")
print(f"Saved {out_pdf}")
print(f"Total time: {total_time:.1f}s")
