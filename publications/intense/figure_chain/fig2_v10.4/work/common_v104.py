"""Shared paths and functions of the Fig. 2 v10.4 work.

Activity maps and their similarity are copied from fig2_src/data_loader.py (compute_activity_map,
calculate_act_maps_sim, threshold_signal); the binning is vectorised and checked against the
original loop (same arithmetic, same summation order, so the maps are equal bit for bit).
Calcium is prepared as DRIADA 1.2.2 prepares it for a neuron (negative values to 0, noise of
1e-8, min-max scaling; Neuron.calcium_preprocessing and TimeSeries.scdata), without building an
Experiment: the build takes minutes per session and is not needed for the maps.
"""
import csv
from pathlib import Path

import numpy as np

import sys  # noqa: E402
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # figure_chain/
import _paths_fig as _F  # noqa: E402

V2 = _F.P.NOF_ALIGNED
OLDEXP = _F.P.NOF_LEGACY
EVENTS = _F.V.get("classic_dir") / "events"
CLASSIC_B = _F.V.get("classic_dir") / "classic"
FPS_TABLE = _F.V.get("fps_table")
RESULTS_NEW = _F.V.get("stage2_results")
DB_NEW = _F.P.DB_PRE
CACHE = _F.out_dir(HERE) / "cache"
VERIFY = _F.out_dir(HERE) / "verify"

# order of the classic .mat file and of fig2_data_v8.py
SESSION_LIST = [f"{m}_{d}D" for d in range(1, 5) for m in
                ("H01", "H02", "H03", "H06", "H07", "H08", "H09", "H14", "H23", "H26", "H27", "H31",
                 "H32", "H33", "H36", "H39")]


def session_fps():
    with open(FPS_TABLE, newline="") as f:
        return {r["session"]: float(r["fps"]) for r in csv.DictReader(f)}


def load_sync(session):
    """Calcium (cells x frames), x, y and the frame rate of a synchronized v2 file."""
    d = np.load(V2 / f"NOF_{session}_aligned.npz", allow_pickle=True)
    meta = d["_metadata"].item()
    return np.asarray(d["Calcium"]), np.asarray(d["x"], float), np.asarray(d["y"], float), float(meta["fps"])


def load_older(session):
    """Calcium without the leading row, x, y of the older export the v7-v10.2 caches were built from."""
    d = np.load(OLDEXP / f"NOF_{session} syn data.npz")
    return np.asarray(d["calcium"])[1:], np.asarray(d["x"], float), np.asarray(d["y"], float)


def event_trains(session, clock, n_cells, n_frames):
    """Binary event trains (neuron.sp of DRIADA 1.2.2) from the runs cached by detect_events.py."""
    e = np.load(EVENTS / f"events_NOF_{session}_{clock}.npz")
    assert int(e["n_cells"]) == n_cells, (session, clock, int(e["n_cells"]), n_cells)
    sp = np.zeros((n_cells, int(e["n_frames"])), dtype=float)
    for c, a, b in zip(e["run_cell"], e["run_st"], e["run_en"]):
        sp[c, a:b] = 1.0
    if n_frames < sp.shape[1]:
        sp = sp[:, :n_frames]
    elif n_frames > sp.shape[1]:
        # older export one or two frames longer than the synchronized file (control only)
        sp = np.pad(sp, ((0, 0), (0, n_frames - sp.shape[1])))
    assert sp.shape[1] == n_frames, (session, clock, sp.shape, n_frames)
    return sp, float(e["fps"])


def ca_scdata(trace, seed):
    """neuron.ca.scdata of DRIADA 1.2.2: clip at 0, add uniform noise of 1e-8, min-max scale."""
    ca = np.asarray(trace, dtype=np.float64).copy()
    ca[ca < 0] = 0
    ca += np.random.default_rng(seed).random(len(ca)) * 1e-08
    lo, hi = ca.min(), ca.max()
    return (ca - lo) / (hi - lo) if hi > lo else ca - lo


def threshold_signal(signal, thr, mode="percentile"):
    if mode == "simple":
        return np.where(signal > thr, signal, 0)
    elif mode == "percentile":
        return np.where(signal > np.percentile(signal, thr), signal, 0)
    elif mode == "mad":
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        return np.where(signal > median + thr * mad, signal, 0)
    return signal


def grid_index(x, y, n=8):
    """Flat bin of every frame as compute_activity_map assigns it (-1 = frame left out)."""
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    ax = ((x - min_x) / ((max_x - min_x) / n)).astype(int)
    ay = ((y - min_y) / ((max_y - min_y) / n)).astype(int)
    ok = (ax >= 0) & (ax < n) & (ay >= 0) & (ay < n)
    flat = np.where(ok, (n - ay - 1) * n + ax, -1)
    return flat


def activity_map_fast(flat, activity, n=8, smooth_sigma=1.0):
    from scipy.ndimage import gaussian_filter
    ok = flat >= 0
    occ = np.bincount(flat[ok], minlength=n * n).reshape(n, n)
    act = np.bincount(flat[ok], weights=np.asarray(activity, float)[ok], minlength=n * n).reshape(n, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_map = np.where(occ > 0, act / occ, 0)
    if smooth_sigma is not None and smooth_sigma > 0:
        rate_map = gaussian_filter(rate_map, sigma=smooth_sigma)
    return rate_map


def activity_map_loop(x_coords, y_coords, activity, n=8, ds=1, smooth_sigma=1.0):
    """compute_activity_map of fig2_src/data_loader.py, unchanged (reference for the check)."""
    from scipy.ndimage import gaussian_filter
    x_coords, y_coords, activity = x_coords[::ds], y_coords[::ds], activity[::ds]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    area_width = (max_x - min_x) / n
    area_height = (max_y - min_y) / n
    occupancy_grid = np.zeros((n, n), dtype=int)
    activity_grid = np.zeros((n, n), dtype=float)
    for x, y, act in zip(x_coords, y_coords, activity):
        area_x = int((x - min_x) / area_width)
        area_y = int((y - min_y) / area_height)
        if 0 <= area_x < n and 0 <= area_y < n:
            occupancy_grid[n - area_y - 1, area_x] += 1
            activity_grid[n - area_y - 1, area_x] += act
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_map = np.where(occupancy_grid > 0, activity_grid / occupancy_grid, 0)
    if smooth_sigma is not None and smooth_sigma > 0:
        rate_map = gaussian_filter(rate_map, sigma=smooth_sigma)
    return rate_map


def calculate_act_maps_sim(activity_map1, activity_map2):
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

    def normalize_image(img):
        img = np.array(img, dtype=float)
        rng = np.max(img) - np.min(img)
        if rng == 0:
            return np.zeros_like(img)
        return (img - np.min(img)) / rng

    map1 = normalize_image(activity_map1)
    map2 = normalize_image(activity_map2)
    with np.errstate(divide="ignore"):
        mse_value = mean_squared_error(map1, map2)
        psnr_value = peak_signal_noise_ratio(map1, map2, data_range=1.0)
    ssim_value = ssim(map1, map2, data_range=1.0)
    return mse_value, psnr_value, ssim_value
