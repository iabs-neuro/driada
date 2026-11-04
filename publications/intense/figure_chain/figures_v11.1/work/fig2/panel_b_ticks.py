"""Colour bar of Fig. 2b: where calcium values lie on the colour scale as the panel computes it.

usage: python -X utf8 -B figure_chain/figures_v11.1/work/fig2/panel_b_ticks.py
       ->  figures_v11.1/work/fig2/cache/fig2b_cbar_ticks.json under [figures] output_dir/figure_chain

The colour of a trajectory segment is neur_ds of the panel cache, made by
data_loader.compute_fig2b_data: rescale(log(ca + 1e-10 + 0.05)), ca = calcium with negative values set to 0, of row 930 of H31 1D
in the synchronized file (neuron 931 of the earlier export, which has one extra leading row and lacks the
last frame), min-max over the whole trace, then every tenth frame. So the colour runs along log(ca + 0.05),
and a calcium value c sits on the colour bar at
    (log(c + 1e-10 + 0.05) - L0) / (L1 - L0),   L0, L1 = min, max of log(ca + 1e-10 + 0.05) over the trace.
This script rebuilds neur_ds from the calcium (it must equal the cache), and writes L0, L1 and the
colour-bar positions of the chosen calcium values. Nothing else is read or written.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # figure_chain/
import _paths_fig as _F  # noqa: E402  (puts the paper folder on sys.path)

CACHE_B = _F.SAVED / "fig2b_data_v104_onsets.npz"
SESSION = _F.V.get("nof_v1") / "NOF_H31_1D_aligned.npz"
ROW = 930                    # neuron 931 of the earlier export, its leading row dropped
N_FRAMES = 11846             # of the earlier export: the synchronized file has one frame more
OFFSET = 1e-10 + 0.05        # compute_fig2b_data
TICKS = (0, 1, 10, 100)      # calcium values marked on the colour bar (older export units)
OUT = _F.out_dir(HERE / "cache") / "fig2b_cbar_ticks.json"


def main():
    if OUT.exists():
        sys.exit(f"{OUT} exists; not overwriting")
    c = np.load(CACHE_B)
    neur_ds = np.asarray(c["neur_ds"], float)
    raw = np.asarray(np.load(SESSION, allow_pickle=True)["Calcium"], float)[ROW][:N_FRAMES]
    # DRIADA clips negative calcium to 0 before ca.data is used (Neuron.calcium_preprocessing)
    ca = np.clip(raw, 0, None)
    lg = np.log(ca + OFFSET)
    l0, l1 = float(lg.min()), float(lg.max())
    rebuilt = ((lg - l0) / (l1 - l0))[::10][:-1]
    diff = float(np.abs(rebuilt - neur_ds).max()) if rebuilt.shape == neur_ds.shape else None
    assert diff is not None and diff < 1e-6, f"neur_ds not reproduced: max abs diff {diff}"
    vmin, vmax = float(neur_ds.min()), float(neur_ds.max())
    pos = [(float(np.log(t + OFFSET)) - l0) / (l1 - l0) for t in TICKS]
    # the bar spans the colours drawn, [min, max] of neur_ds; calcium 0 lies 1e-10 below its bottom
    # (the bottom is calcium 4e-11), so a tick that close to an end is put on that end
    shown = [(t, min(max(p, vmin), vmax)) for t, p in zip(TICKS, pos) if vmin - 1e-6 <= p <= vmax + 1e-6]
    res = dict(
        source=SESSION.name, row=ROW, offset=OFFSET, n_frames=int(len(ca)),
        raw_min=float(raw.min()), raw_frames_below_0=int((raw < 0).sum()),
        ca_min=float(ca.min()), ca_max=float(ca.max()), ca_median=float(np.median(ca)),
        ca_percentiles_5_50_95_99=[float(v) for v in np.percentile(ca, [5, 50, 95, 99])],
        log_min=l0, log_max=l1, neur_ds_rebuilt_max_abs_diff=diff,
        colour_range_of_panel=[vmin, vmax],
        colour_range_in_calcium=[float(np.exp(l0 + vmin * (l1 - l0)) - OFFSET),
                                 float(np.exp(l0 + vmax * (l1 - l0)) - OFFSET)],
        ticks_calcium=[t for t, _ in shown], ticks_position=[p for _, p in shown],
        ticks_position_unclamped=pos,
        ticks_dropped_outside_range=[t for t, p in zip(TICKS, pos) if not vmin - 1e-6 <= p <= vmax + 1e-6],
        old_linear_ticks_in_calcium={f"{v:.1f}": float(np.exp(l0 + v * (l1 - l0)) - OFFSET)
                                     for v in (0, 0.2, 0.4, 0.6, 0.8)},
    )
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
