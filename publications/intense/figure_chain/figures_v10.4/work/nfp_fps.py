"""plot_neuron_feature_pair of DRIADA 1.2.2 with the time axis at the frame rate of the session.

The library function draws time as frames / 20 (intense/visual.py) and shades a behaviour
period from its first sample to its last sample + 1 in the units of the axis, i.e. one second
too long when the axis is in seconds. The library is not changed: this module takes the source
of the library function, replaces exactly these two places and runs the result with the globals
of the library module, so everything else is the library code itself.

    time axis:  frames / 20       ->  frames / fps of the session
    shading:    [t_first, t_last + 1]  ->  [t_first, t_last + ds / fps]  (exact_shading=True)
"""
import inspect
import re

import numpy as np
from driada.intense import visual as _V

_SRC = inspect.getsource(_V.plot_neuron_feature_pair)
_OLD_AXIS = "xvals = np.arange(ind1, ind2)[::ds] / 20.0"
assert _SRC.count(_OLD_AXIS) == 1, "time axis line of the library function not found"
assert _SRC.count("plot_shadowed_groups(") == 1, "shading call of the library function not found"
_cache = {}


def shade_runs(ax, xvals, binary, ds, fps, color="gray", alpha=0.3, label="shadowed",
               ymin=0.0, ymax=1.0):
    """Shade each run of ones exactly over its frames: [t_first, t_last + ds / fps]."""
    b = (np.asarray(binary) == 1).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.r_[0, b, 0]) != 0)
    labelled = False
    for a, e in zip(edges[0::2], edges[1::2]):
        ax.axvspan(xvals[a], xvals[e - 1] + ds / fps, ymin=ymin, ymax=ymax, alpha=alpha,
                   color=color, label=label if not labelled else "")
        labelled = True
    return ax


def _build(fps, exact_shading):
    src = _SRC.replace(_OLD_AXIS, "xvals = np.arange(ind1, ind2)[::ds] / _FPS")
    if exact_shading:
        src = src.replace("plot_shadowed_groups(", "_shade_runs_ds(ds, ")
    src = src.replace("def plot_neuron_feature_pair(", "def _plot(", 1)
    ns = dict(vars(_V))
    ns["_FPS"] = float(fps)
    ns["_shade_runs_ds"] = lambda ds, ax, xvals, binary, color="gray", alpha=0.3, label="shadowed": \
        shade_runs(ax, xvals, binary, ds, float(fps), color=color, alpha=alpha, label=label)
    exec(compile(re.sub(r"^\s*@.*\n", "", src), "<nfp_fps>", "exec"), ns)
    return ns["_plot"]


def plot_neuron_feature_pair_fps(exp, cell_id, featname, *, fps, exact_shading=True, **kwargs):
    """plot_neuron_feature_pair of DRIADA 1.2.2, time in seconds at ``fps``."""
    key = (float(fps), bool(exact_shading))
    if key not in _cache:
        _cache[key] = _build(*key)
    return _cache[key](exp, cell_id, featname, **kwargs)
