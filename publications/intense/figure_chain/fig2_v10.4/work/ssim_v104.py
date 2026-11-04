"""Per-neuron similarity of calcium and event activity maps (Fig. 2e) for the 64 NOF sessions.

The maps (8 x 8 grid, calcium above its 95th percentile, Gaussian smoothing of one bin) are built from the
synchronized files with the events of DRIADA 1.2.2 cached by the classic method (classic/detect_events.py),
at the frame rate of the session.

usage (from this folder):
       python ssim_v104.py check     # vectorised map == original loop; scdata == DRIADA Neuron
       python ssim_v104.py sync      # -> cache/fig2e_<session>_metrics_true.npz under
                                     #    [figures] output_dir/figure_chain/fig2_v10.4/work
"""
import json
import sys
import time

import numpy as np

from common_v104 import (CACHE, SESSION_LIST, VERIFY, activity_map_fast, activity_map_loop,
                         ca_scdata, calculate_act_maps_sim, event_trains, grid_index,
                         load_sync, session_fps, threshold_signal)

THR, N_GRID, SIGMA = 95, 8, 1.0


def session_metrics(ca, x, y, sp, sidx):
    flat = grid_index(x, y, N_GRID)
    out = np.full((ca.shape[0], 3), np.nan)
    for c in range(ca.shape[0]):
        ca_sig = threshold_signal(ca_scdata(ca[c], seed=(sidx, c)), THR, mode="percentile")
        ca_map = activity_map_fast(flat, ca_sig, N_GRID, SIGMA)
        sp_map = activity_map_fast(flat, sp[c], N_GRID, SIGMA)
        out[c] = calculate_act_maps_sim(ca_map, sp_map)
    return out


def check():
    """The vectorised map equals the loop of data_loader on real neurons; scdata equals DRIADA's."""
    import warnings
    from common_v104 import CACHE as _  # noqa: F401
    res = {}
    s = "H01_1D"
    ca, x, y, fps = load_sync(s)
    sp, _ = event_trains(s, "true", ca.shape[0], ca.shape[1])
    flat = grid_index(x, y, N_GRID)
    n_eq = 0
    for c in range(0, ca.shape[0], 10):
        sig = threshold_signal(ca_scdata(ca[c], seed=(0, c)), THR)
        for a, sm in ((sig, SIGMA), (sp[c], SIGMA), (sig, 0.5), (sp[c], None)):
            m1 = activity_map_fast(flat, a, N_GRID, sm)
            m2 = activity_map_loop(x, y, a, N_GRID, 1, sm)
            n_eq += int(np.array_equal(m1, m2))
    res["maps_equal_fast_vs_loop"] = f"{n_eq} of {4 * len(range(0, ca.shape[0], 10))}"
    # scdata of DRIADA 1.2.2 (Neuron.calcium_preprocessing + TimeSeries.scdata) without the noise
    import _paths_fig as _F  # figure_chain/ is on sys.path (common_v104)
    sys.path.insert(0, str(_F.P.DRIADA_SRC))
    import driada
    from driada.experiment.neuron import Neuron
    from driada.information.info_base import TimeSeries
    assert driada.__version__ == "1.2.2", driada.__version__
    worst = 0.0
    for c in range(0, ca.shape[0], 10):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = TimeSeries(Neuron.calcium_preprocessing(ca[c], seed=(0, c)), discrete=False)
        worst = max(worst, float(np.abs(ts.scdata - ca_scdata(ca[c], seed=(0, c))).max()))
    res["scdata_max_abs_diff_vs_driada"] = worst
    (VERIFY / "ssim_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def sync():
    fps_tab = session_fps()
    CACHE.mkdir(exist_ok=True)
    log = {}
    for i, s in enumerate(SESSION_LIST):
        t0 = time.time()
        ca, x, y, fps = load_sync(s)
        assert abs(fps - fps_tab[f"NOF_{s}"]) < 1e-9, (s, fps)
        assert np.isfinite(x).all() and np.isfinite(y).all(), s
        n, T = ca.shape
        out = {}
        for clock in ("true",):
            p = CACHE / f"fig2e_{s}_metrics_{clock}.npz"
            if p.exists():
                raise FileExistsError(p)
            sp, efps = event_trains(s, clock, n, T)
            assert abs(efps - fps) < 1e-9, (s, clock, efps)
            m = session_metrics(ca, x, y, sp, i)
            np.savez_compressed(p, ssim_vals=m, fps_events=efps, n_event_frames=sp.sum(1))
            out[clock] = dict(fps=efps, mean_ssim=float(np.nanmean(m[:, 2])))
        log[s] = dict(n=int(n), frames=int(T), **out, seconds=round(time.time() - t0, 1))
        print(s, log[s], flush=True)
    (VERIFY / "ssim_sync_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    VERIFY.mkdir(exist_ok=True)
    {"check": check, "sync": sync}[sys.argv[1]]()
