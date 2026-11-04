# -*- coding: utf-8 -*-
"""Generator, populations and scoring of the synthetic benchmark of Supplementary Section S1.

Recordings come from the generator of tuned cells of the DRIADA release named in config.toml
(``generate_tuned_selectivity_exp``): a random-walk trajectory from which head direction and speed
are derived, one binary event, 900 s at 20 Hz. SNR is the ratio of the peak firing rate to the
baseline rate of 0.1 Hz at fixed calcium noise; the default peak of 2 Hz is SNR 20.

Importing the module puts the configured DRIADA release first on sys.path and checks its version
(intense_paths.use_driada).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys; sys.dont_write_bytecode = True
from pathlib import Path

PAPER =Path(__file__).resolve().parents[1]
if str(PAPER) not in sys.path:
    sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402

driada = P.use_driada(tools=False)
from driada.experiment.synthetic import generate_tuned_selectivity_exp  # noqa: E402

CANON = ("head_direction", "speed", "event_0")

# the 20 seeds of Supplementary Table 3; the other parts take the first 10 or 15 of them
SEEDS20 = [42, 123, 7, 11, 19, 23, 31, 44, 57, 64, 73, 88, 91, 101, 113, 127, 139, 151, 163, 177]

# standard population: 5 cells of each selective type and 10 nonselective cells
POP_STD = [
    {"name": "hd_cells", "count": 5, "features": ["head_direction"]},
    {"name": "speed_cells", "count": 5, "features": ["speed"]},
    {"name": "event_cells", "count": 5, "features": ["event_0"]},
    {"name": "mixed_hd_speed_lin", "count": 5, "features": ["head_direction", "speed"], "combination": "weighted_sum"},
    {"name": "mixed_speed_event_lin", "count": 5, "features": ["speed", "event_0"], "combination": "weighted_sum"},
    {"name": "mixed_hd_speed_and", "count": 5, "features": ["head_direction", "speed"], "combination": "and"},
    {"name": "mixed_speed_event_and", "count": 5, "features": ["speed", "event_0"], "combination": "and"},
    {"name": "nonselective", "count": 10, "features": []},
]
TRUE_STD = {
    "hd_cells": {"head_direction"}, "speed_cells": {"speed"}, "event_cells": {"event_0"},
    "mixed_hd_speed_lin": {"head_direction", "speed"}, "mixed_speed_event_lin": {"speed", "event_0"},
    "mixed_hd_speed_and": {"head_direction", "speed"}, "mixed_speed_event_and": {"speed", "event_0"},
    "nonselective": set(),
}


def gen(population, seed, peak_rate=2.0, duration=900.0):
    return generate_tuned_selectivity_exp(
        population=population, duration=duration, fps=20.0, seed=seed,
        n_discrete_features=1, baseline_rate=0.1, peak_rate=peak_rate,
        decay_time=2.0, calcium_noise=0.05, verbose=False, reconstruct_spikes="wavelet",
    )


def group_map(population):
    out = {}; i = 0
    for g in population:
        for _ in range(g["count"]):
            out[i] = g["name"]; i += 1
    return out


def prf1(det, true):
    if not det and not true:
        return 1.0
    tp = len(det & true); fp = len(det - true); fn = len(true - det)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0
