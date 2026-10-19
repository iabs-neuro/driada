"""
Fig 2C statistic — circular agreement between the Isomap angle and head direction.

Reads fig2_isomap.npz written by fig2_data.py and prints rho_circ with its
circular-shift p-value, as described in Methods (synthetic data).

Usage:
    python publications/elife/fig2/fig2_circ_stats.py
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import elife_paths

SEED = 42
N_SHIFTS = 1000
N_ROT = 360  # rotation grid for the alignment


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def align(emb_angle, hd):
    """Rotation and reflection of emb_angle that minimize the mean absolute angular error."""
    best = (np.inf, None)
    rots = np.linspace(0, 2 * np.pi, N_ROT, endpoint=False)
    for sign in (1, -1):
        err = np.abs(wrap(sign * emb_angle[None, :] + rots[:, None] - hd[None, :])).mean(axis=1)
        k = int(np.argmin(err))
        if err[k] < best[0]:
            best = (err[k], sign * emb_angle + rots[k])
    return best[1]


def rho_circ(a, b):
    """Modulus of the complex correlation between mean-centered unit phasors."""
    za = np.exp(1j * a)
    zb = np.exp(1j * b)
    za = za - za.mean()
    zb = zb - zb.mean()
    return np.abs(np.sum(za * np.conj(zb))) / np.sqrt(np.sum(np.abs(za) ** 2) * np.sum(np.abs(zb) ** 2))


def main():
    d = np.load(os.path.join(elife_paths.CACHE_DIR, "fig2_isomap.npz"))
    coords, hd = d["coords"], d["hd_downsampled"]
    centered = coords - coords.mean(axis=0)
    emb_angle = np.arctan2(centered[:, 1], centered[:, 0])

    observed = rho_circ(align(emb_angle, hd), hd)

    rng = np.random.default_rng(SEED)
    n = len(hd)
    null = np.empty(N_SHIFTS)
    for i in range(N_SHIFTS):
        hd_shift = np.roll(hd, rng.integers(1, n))
        null[i] = rho_circ(align(emb_angle, hd_shift), hd_shift)
    k = int(np.sum(null >= observed))
    p = (k + 1) / (N_SHIFTS + 1)

    print(f"rho_circ = {observed:.3f}, p = {p:.4f} ({k} of {N_SHIFTS} shifts >= observed)")


if __name__ == "__main__":
    main()
