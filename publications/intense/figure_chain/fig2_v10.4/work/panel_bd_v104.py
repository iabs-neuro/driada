"""Data of Fig. 2d and, when the earlier export is there, of Fig. 2b.

2d (H01 1D, neuron 97 of the earlier export = row 96 of the synchronized file): calcium and event
maps as data_loader.compute_fig2d_data builds them, smoothed with one bin as the maps behind 2e, from the
synchronized file with events at the session rate.

2b (H31 1D, neuron 931 of the earlier export = row 930 of the synchronized file): trajectory and
colour are those of the cache of fig2/cache_examples.py (they do not depend on the frame rate); the
stars are every tenth frame inside the detected events, as data_loader.compute_fig2b_data takes them
(np.where(sp != 0)[0][::10]), with the events of DRIADA 1.2.2 at the rate of the session, placed on the
same coordinates as the trajectory (earlier export; it equals the synchronized file without its last
frame in calcium, frame for frame). Fig. 2b itself reads figure_chain/saved_inputs/fig2b_data_v104_onsets.npz
(fig2b_onsets.py, README.md).

usage: python panel_bd_v104.py   # from this folder -> figures_v10.4/work/cache/fig2d_data_v104.npz,
                                 # fig2b_data_v104.npz under [figures] output_dir/figure_chain
"""
import json

import numpy as np

from common_v104 import (VERIFY, activity_map_fast, ca_scdata, calculate_act_maps_sim,
                         event_trains, grid_index, load_older, load_sync, threshold_signal)

import _paths_fig as _F  # noqa: E402  (figure_chain/ is put on sys.path by common_v104)
OLD_CACHE = _F.P.CACHE_DIR            # fig2/cache_examples.py
FIG_CACHE = _F.out_dir(_F.CHAIN / "figures_v10.4" / "work" / "cache")
B_SESSION, B_ROW = "H31_1D", 930      # neuron 931 of the older export
D_SESSION, D_ROW = "H01_1D", 96       # neuron 97 of the older export


def stars(session, row, clock, x_old, y_old):
    ca_old, _, _ = load_older(session)
    sp, fps = event_trains(session, clock, ca_old.shape[0], ca_old.shape[1])
    spinds = np.where(sp[row] != 0)[0]
    return x_old[spinds][::10], y_old[spinds][::10], fps, int(len(spinds))


def maps(ca, x, y, sp, row, sigma):
    flat = grid_index(x, y, 8)
    ca_sig = threshold_signal(ca_scdata(ca[row], seed=(1, row)), 95, mode="percentile")
    return activity_map_fast(flat, ca_sig, 8, sigma), activity_map_fast(flat, sp[row], 8, sigma)


def panel_b():
    """Stars of 2b on the trajectory of the earlier export."""
    b_old = np.load(OLD_CACHE / "fig2b_data.npz")
    _, xo, yo = load_older(B_SESSION)
    assert np.array_equal(b_old["x_ds"], xo[::10]) and np.array_equal(b_old["y_ds"], yo[::10])
    sx, sy, fps, nfr = stars(B_SESSION, B_ROW, "true", xo, yo)
    out_b = FIG_CACHE / "fig2b_data_v104.npz"
    if out_b.exists():
        raise FileExistsError(out_b)
    np.savez(out_b, x_ds=b_old["x_ds"], y_ds=b_old["y_ds"], neur_ds=b_old["neur_ds"], spike_x=sx, spike_y=sy)
    return dict(fps=fps, event_frames_true=nfr, n_stars_true=int(len(sx)))


def panel_d():
    ca, x, y, fps_d = load_sync(D_SESSION)
    sp, fps = event_trains(D_SESSION, "true", ca.shape[0], ca.shape[1])
    cm, sm = maps(ca, x, y, sp, D_ROW, 1.0)
    cm05, sm05 = maps(ca, x, y, sp, D_ROW, 0.5)
    out_d = FIG_CACHE / "fig2d_data_v104.npz"
    if out_d.exists():
        raise FileExistsError(out_d)
    np.savez_compressed(out_d, ca_map=cm, sp_map=sm)
    ssim_new = np.load(_F.V.get("fig2_ssim_cache") / f"fig2e_{D_SESSION}_metrics_true.npz")["ssim_vals"][:, 2]
    return dict(fps=fps, smoothing_bins=1.0, ssim_example_1bin=float(calculate_act_maps_sim(cm, sm)[2]),
                ssim_example_0_5bin=float(calculate_act_maps_sim(cm05, sm05)[2]),
                ssim_example_in_2e_cache=float(ssim_new[D_ROW]),
                ssim_percentile_in_session=float((ssim_new < ssim_new[D_ROW]).mean() * 100),
                colour_scale_max=float(max(cm.max(), sm.max())))


def main():
    VERIFY.mkdir(exist_ok=True)
    res = {"2d": panel_d()}
    if _F.P.legacy_file("H31", "1D").exists() and (OLD_CACHE / "fig2b_data.npz").exists():
        res["2b"] = panel_b()
    else:
        print("[SKIPPED] panel b: needs the earlier export ([data] nof_legacy); the figure reads its saved input")
    (VERIFY / "panel_bd_final.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
