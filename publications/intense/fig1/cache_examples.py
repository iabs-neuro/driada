"""Example caches of Figure 1A and 1B, built from the earlier export of the sessions.

The figure reads the caches; this script is only needed to rebuild them and
requires [data] nof_legacy of config.toml. In that export the calcium matrix
has one extra leading row: its rows 36-38 of H01 1D are rows 35-37 of the
synchronized files, the labels used in the figure.

usage: python fig1/cache_examples.py  ->  <cache>/fig1a_data.npz, fig1b_data.npz
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

P.use_driada(tools=False)
from legacy_loader import STATIC_FEATURES, load_nof_experiment  # noqa: E402

N_FRAMES = 5000
LEGACY_ROWS = [36, 37, 38]   # = synchronized rows 35, 36, 37


def main():
    exp = load_nof_experiment(animal_id="H01", day="1D", verbose=False)
    data = {f"neuron_{i}_ca": exp.neurons[i].ca.scdata[:N_FRAMES].astype(np.float32) for i in LEGACY_ROWS}
    data["neuron_indices"] = np.array(LEGACY_ROWS)
    data["fps"] = STATIC_FEATURES["fps"]
    np.savez_compressed(P.cache_path("fig1a_data.npz"), **data)

    exp = load_nof_experiment(animal_id="H01", day="3D", verbose=False)
    np.savez_compressed(P.cache_path("fig1b_data.npz"),
                        **{k: getattr(exp, k).data[:N_FRAMES].astype(np.float32)
                           for k in ("headdirection", "object4", "x", "y")}, fps=STATIC_FEATURES["fps"])
    print("[OK]", P.CACHE_DIR)


if __name__ == "__main__":
    main()
