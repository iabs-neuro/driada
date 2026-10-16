"""
Compute PROTEST (Procrustes permutation test) for Fig 3A UMAP embedding.

Tests whether the Procrustes alignment between the UMAP embedding
and physical arena positions is significantly better than chance.

Run:  python publications/elife/compute/compute_procrustes_protest.py
"""

import os
import sys
import numpy as np
from scipy.spatial import procrustes as scipy_procrustes

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import elife_paths

# ── Paths ────────────────────────────────────────────────
SESSION = "FOF_F28_1D"
ALIGNED_PATH = os.path.join(elife_paths.SYNC_DIR, f"{SESSION}_aligned.npz")
UMAP_CACHE = os.path.join(elife_paths.CACHE_DIR, f"{SESSION}_umap_cache.npz")
DS = 5  # temporal downsampling (must match fig3/panel_a.py)

# ── Parameters ───────────────────────────────────────────
N_PERMUTATIONS = 999
SEED = 42


def main():
    # Load data
    print(f"Loading {SESSION}...")
    aligned = np.load(ALIGNED_PATH)
    cache = np.load(UMAP_CACHE)

    coords_real = cache["coords_real"]  # (2, n_samples)
    coords_shuf = cache["coords_shuf"]  # (2, n_samples)

    x = aligned["x"].ravel()[::DS]
    y = aligned["y"].ravel()[::DS]
    positions = np.column_stack([x, y])  # (n_samples, 2)

    print(f"  Samples: {len(positions)}, DS={DS}")

    # Normalized Procrustes disparity (scipy, in [0, 1])
    _, _, disp_real = scipy_procrustes(positions, coords_real.T)
    _, _, disp_shuf = scipy_procrustes(positions, coords_shuf.T)
    print(f"\nNormalized Procrustes disparity:")
    print(f"  Real embedding:    {disp_real:.6f}")
    print(f"  Shuffled embedding: {disp_shuf:.6f}")

    # PROTEST: permutation test
    print(f"\nRunning PROTEST ({N_PERMUTATIONS} permutations)...")
    rng = np.random.RandomState(SEED)
    null_disparities = np.zeros(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        perm_idx = rng.permutation(len(positions))
        _, _, d = scipy_procrustes(positions[perm_idx], coords_real.T)
        null_disparities[i] = d

    # p-value: fraction of null disparities <= observed
    p_value = (np.sum(null_disparities <= disp_real) + 1) / (N_PERMUTATIONS + 1)

    print(f"\nResults:")
    print(f"  Observed disparity: {disp_real:.6f}")
    print(f"  Null mean +/- std:  {null_disparities.mean():.6f} +/- {null_disparities.std():.6f}")
    print(f"  Null range:         [{null_disparities.min():.6f}, {null_disparities.max():.6f}]")
    print(f"  p-value:            {p_value:.4f}")
    print(f"  Significant at 0.001: {'YES' if p_value < 0.001 else 'NO'}")


if __name__ == "__main__":
    main()
