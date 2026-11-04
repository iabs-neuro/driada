"""Data of Fig. 2b: one star at the onset of every detected event (events of DRIADA 1.2.2 at the session
rate), on the trajectory and with the colour of figures_v10.4/work/cache/fig2b_data_v104.npz
(panel_bd_v104.py).

Needs the earlier export ([data] nof_legacy), which is not part of the deposit; the file this script writes
is saved in figure_chain/saved_inputs, where Fig. 2b reads it (README.md).

usage: python fig2b_onsets.py   # from this folder -> figures_v10.4/work/cache/fig2b_data_v104_onsets.npz
                                # under [figures] output_dir/figure_chain
"""
import sys

import numpy as np

from common_v104 import EVENTS, load_older

import _paths_fig as _F  # noqa: E402  (figure_chain/ is put on sys.path by common_v104)

FIG_CACHE = _F.out_dir(_F.CHAIN / "figures_v10.4" / "work" / "cache")
OUT = FIG_CACHE / "fig2b_data_v104_onsets.npz"


def main():
    if OUT.exists():
        sys.exit(f"{OUT} exists; not overwriting")
    base = np.load(FIG_CACHE / "fig2b_data_v104.npz")
    _, xo, yo = load_older("H31_1D")
    e = np.load(EVENTS / "events_NOF_H31_1D_true.npz")
    onsets = np.sort(e["run_st"][e["run_cell"] == 930])
    onsets = onsets[onsets < len(xo)]
    np.savez(OUT, x_ds=base["x_ds"], y_ds=base["y_ds"], neur_ds=base["neur_ds"],
             spike_x=xo[onsets], spike_y=yo[onsets])
    print("onsets", len(onsets))


if __name__ == "__main__":
    main()
