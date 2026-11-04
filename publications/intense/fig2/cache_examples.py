"""Example caches of Figure 2A, 2B, 2D and the per-session map similarity of 2E.

Built from the earlier export of the sessions ([data] nof_legacy of
config.toml) with the functions of fig2_loader.py; the figure reads the caches.
In that export the calcium matrix has one extra leading row, so the neurons
below are rows 59 (2A), 930 (2B) and 96 (2D) of the synchronized files. The
per-session metrics skip row 0 of the export and are therefore aligned with
the synchronized files and the INTENSE results.

usage: python fig2/cache_examples.py [--metrics]
    without --metrics: fig2a_data.npz, fig2b_data.npz, fig2d_data.npz (minutes)
    with --metrics: also fig2e_<session>_metrics.npz for all 64 sessions (wavelet event
    reconstruction of every neuron; hours)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import intense_paths as P  # noqa: E402

P.use_driada(tools=False)
import fig2_loader as L  # noqa: E402

LEGACY = {"a": ("H31", "1D", 60), "b": ("H31", "1D", 931), "d": ("H01_1D", 97)}


def main():
    m, d, n = LEGACY["a"]
    L.save_fig2a_cache(*L.compute_fig2a_data(animal_id=m, day=d, neuron_index=n))
    m, d, n = LEGACY["b"]
    L.save_fig2b_cache(*L.compute_fig2b_data(animal_id=m, day=d, cell_ind=n))
    s, n = LEGACY["d"]
    L.save_fig2d_cache(*L.compute_fig2d_data(session=s, neuron_idx=n))
    if "--metrics" in sys.argv:
        for s in L.SESSION_LIST:
            L.save_fig2e_session_cache(s, L.compute_fig2e_session_data(s))
    print("[OK]", P.CACHE_DIR)


if __name__ == "__main__":
    main()
