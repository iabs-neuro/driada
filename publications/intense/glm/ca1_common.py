"""Helpers of the CA1 scripts of Section S1.4: saved INTENSE results and the acceptance criterion of the paper.

Importing the module does not load DRIADA; the scripts that run INTENSE call
intense_paths.use_driada() themselves.
"""
import json
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
if str(PAPER) not in sys.path:
    sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402

# tested feature of the type-based representation -> feature of Supplementary Table 4
CA1_TABLE4 = {"place_quad": "position", "headdirection_harm2": "head_direction", "speed_quad": "speed"}


def session_fps(name):
    """Frame rate of a CA1 session ('NOF_H01_1D') from _metadata of its file in [data] nof_aligned."""
    with np.load(P.NOF_ALIGNED / f"{name}_aligned.npz", allow_pickle=True) as d:
        return float(d["_metadata"].item()["fps"])


def load_results_npz(path):
    """Stats, significance and optimal delays of a saved INTENSE result (save_intense_results layout)."""
    with np.load(path, allow_pickle=True) as d:
        stats = json.loads(str(d["_stats_json"][0]))
        sig = json.loads(str(d["_significance_json"][0]))
        params = json.loads(str(d["_params_json"][0]))
    return stats, sig, params


def paper_accepts(st, sg):
    """Acceptance criterion of the paper for one neuron-feature pair.

    The pair must be significant after the second stage and the removal of anti-selective
    pairs (the tables of the run keep only such pairs), with p below [analysis] pval_thr, an
    excess of MI over the mean of the second-stage shuffles above [analysis] mi_excess_thr and an
    activity ratio above 1 where it is defined.
    """
    if not sg or not sg.get("stage2", False):
        return False
    pval, mex, ratio = st.get("pval"), st.get("me_excess"), st.get("signal_ratio")
    return (pval is not None and pval < P.PVAL_THR and mex is not None and mex > P.MI_EXCESS_THR
            and (ratio is None or (isinstance(ratio, float) and np.isnan(ratio)) or ratio > 1))


def significant_only(st, sg):
    """The same pair without the information threshold (significance and activity ratio only)."""
    if not sg or not sg.get("stage2", False):
        return False
    pval, ratio = st.get("pval"), st.get("signal_ratio")
    return (pval is not None and pval < P.PVAL_THR
            and (ratio is None or (isinstance(ratio, float) and np.isnan(ratio)) or ratio > 1))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False,
                               default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                    encoding="utf-8")
