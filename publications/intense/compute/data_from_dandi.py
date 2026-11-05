"""The EOF sessions of Dandiset 002047 as the npz files the scripts of this folder read.

The Dandiset labels the open-field sessions of the paper EOF: session ``EOF_H01_S1`` is
``NOF_H01_1D`` here (S<k> is day <k>D). It also holds other paradigms, and every session has
two NWB files, of which ``_ophys.nwb`` only describes the raw movie. The script reads
``sub-<mouse>_ses-EOF-<mouse>-S<k>_behavior+image+ophys.nwb`` and nothing else. The analysis
reads one npz file per session (``NOF_<mouse>_<day>_aligned.npz`` in [v11] nof_v1) and one file
of zone masks per session (``<session>_zones_cm.npz`` in [data] zone_masks/NOF); the script
writes both:

* ``Calcium`` (neurons x frames, float64): the columns of ``processing/ophys/Fluorescence/C``
  of the ROIs marked ``accepted`` in ``PlaneSegmentation``.
* ``x``, ``y``, ``speed``, ``bodydirection``, ``headdirection`` from
  ``processing/behavior/Position/xy`` and ``BehavioralTimeSeries``; every other variable from
  the episodes of its ``TimeIntervals`` table in ``processing/behavior``: 1 on the behavior
  frames with start <= t < stop, 0 elsewhere. All of them go to the calcium frames by the rule
  the analysed files were made with (IABS data synchronizer 1.1.0, mode "2 timelines", see
  ``read_session`` and ``to_calcium_frames``). So a discrete variable takes values between 0
  and 1 at the edges of an episode, as in the analysis. The members are written in the order of
  the files the paper was computed from (``SESSION_KEYS``): DRIADA builds the features in that
  order, and the seeded tie-breaking noise and shuffles follow it.
* the zone masks, rebuilt from the outlines in ``processing/behavior/arena_geometry``
  (see ``read_zones``). DRIADA and synthetic/make_synth.py read ``zone_names``, ``masks`` and
  ``grid_step_cm``.

The arrays cannot be bit-identical to the files the paper was computed from: NWB holds the
traces in float32 where the analysed files hold float64, and it holds the timestamps rounded,
so an interpolation between two behavior frames lands slightly elsewhere. ``--check`` therefore
compares within explicit tolerances (``TOLERANCE``, ``MASK_MIN_AGREEMENT``). They were chosen so
that these two causes pass and anything else, such as a frame shift, another preset of the
zones or another set of ROIs, does not; a session beyond them is to be looked at, not to get a
wider tolerance.

The frame rate of each session is not taken from NWB: compute/make_v2_data.py adds the rate of
fps_by_session.csv to every file, and it needs all 64 sessions.

usage: python compute/data_from_dandi.py <dandiset folder> [--check DIR_SESSIONS DIR_MASKS]
       (the NWB files are looked for at any depth below the folder; with --check nothing is
       written, and every array is compared with the file of the same name in the two folders)
"""
import argparse
import math
import re
import sys
from pathlib import Path

import h5py
import numpy as np
from skimage.draw import line, polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402
import paths_v11 as V  # noqa: E402

SESSION_FILES = "sub-*_ses-EOF-*_behavior+image+ophys.nwb"
N_SESSIONS = 64
CONTINUOUS = ["speed", "bodydirection", "headdirection"]
DISCRETE = ["rest", "walk", "locomotion", "freezing", "rear", "corners", "walls", "center", "object1",
            "object2", "object3", "object4", "objects"]
SESSION_KEYS = ["Calcium", "x", "y", *CONTINUOUS, *DISCRETE]
ZONE_NAMES = [*(f"ArenaCornerRealOut{k}" for k in range(1, 5)), "ArenaCornersAllRealOut",
              *(f"ArenaWallRealOut{k}" for k in range(1, 5)), "ArenaWallsAllRealOut", "Center",
              *(f"Object{k}RealOut" for k in range(1, 5)), "ObjectAllRealOut", "ArenaReal"]
GRID_STEP_CM = 0.1
BEHAVIOR = "processing/behavior"
OPHYS = "processing/ophys"

# Largest accepted absolute difference from the analysed file, in the units of the array. Rounded
# timestamps move a value by a small share of the step between two behavior frames, so the limit
# of an array follows the size of its steps; the directions jump by a full turn where they wrap.
TOLERANCE = {"Calcium": 1e-3, "x": 1e-3, "y": 1e-3, "speed": 5e-3, "bodydirection": 5e-3,
             "headdirection": 5e-3, **{key: 1e-3 for key in DISCRETE}}
# The same for the corners and the centres of the objects, in cm.
CM_TOLERANCE = 1e-3
# Smallest accepted share of the cells of one zone mask equal to the analysed mask.
MASK_MIN_AGREEMENT = 0.999


def session_name(session_id):
    """``'EOF_H01_S1'`` -> ``'NOF_H01_1D'``."""
    match = re.fullmatch(r"EOF_(H\d+)_S(\d+)", session_id)
    if match is None:
        raise SystemExit(f"session id {session_id!r} is not of the form EOF_<mouse>_S<k>")
    return f"NOF_{match[1]}_{match[2]}D"


def to_calcium_frames(values, tb, tc):
    """Values on the behavior frames at the calcium frames, as the synchronizer computes them.

    Only the behavior frames within the span of the calcium frames are used, and a calcium frame
    beyond them takes the nearest one, so that the first and the last calcium frame normally hold
    a behavior value instead of a value interpolated with a frame outside that span.

    Parameters
    ----------
    values, tb : np.ndarray
        Values on the behavior frames and their timestamps.
    tc : np.ndarray
        Timestamps of the calcium frames kept, all within the span of ``tb``.

    Returns
    -------
    np.ndarray
        Values at the calcium frames, float64.
    """
    inside = (tb >= tc[0]) & (tb <= tc[-1])
    return np.interp(tc, tb[inside], values[inside])


def episode_state(table, tb):
    """1 on the behavior frames inside an episode (start <= t < stop), 0 elsewhere."""
    state = np.zeros(tb.size)
    for first, last in zip(np.searchsorted(tb, table["start_time"][()]), np.searchsorted(tb, table["stop_time"][()])):
        state[first:last] = 1.0
    return state


def fill_outline(x, y, frame_shape):
    """Pixels of a zone from its outline.

    Parameters
    ----------
    x, y : np.ndarray
        Outline in pixels counted from 0, its parts separated by NaN.
    frame_shape : tuple of int
        Height and width of the video frame.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``frame_shape``.
    """
    mask = np.zeros(frame_shape, bool)
    cuts = np.flatnonzero(np.isnan(x))
    for xs, ys in zip(np.split(x, cuts), np.split(y, cuts)):
        xs, ys = xs[~np.isnan(xs)].astype(int), ys[~np.isnan(ys)].astype(int)
        if xs.size == 0:
            continue
        mask[polygon(ys, xs, frame_shape)] = True
        # The outline runs through the boundary pixels of the zone, which a fill may leave out;
        # a part can be a single pixel.
        for k in range(xs.size):
            mask[line(ys[k - 1], xs[k - 1], ys[k], xs[k])] = True
    return mask


def read_zones(table, name):
    """Arrays of the zone-mask file of a session from its ``arena_geometry`` table.

    The rasters follow the zone masks the paper read: a cell (ix, iy) has its centre at
    (ix * step, iy * step) cm and takes the mask of the video at round(centre * scale) in MATLAB
    numbering, from 1, while the outlines of the table count pixels from 0. The scale and the frame
    size are those of the description of the table.

    Parameters
    ----------
    table : h5py.Group
        ``processing/behavior/arena_geometry``.
    name : str
        Session name, e.g. ``'NOF_H01_1D'``.

    Returns
    -------
    dict
        ``masks``, ``zone_names``, ``grid_step_cm``, ``corners_cm``, ``object_centers_cm``,
        ``session``.
    """
    description = table.attrs["description"]
    scale_x, scale_y = map(float, re.search(r"([\d.]+) px per cm along x, ([\d.]+) px per cm along y",
                                            description).groups())
    width, height = map(int, re.search(r"Video (\d+) x (\d+) px", description).groups())
    ends = table["x_px_index"][()]
    outlines = {element.decode(): (start, end)
                for element, start, end in zip(table["element"][()], np.r_[0, ends[:-1]], ends)}
    xs, ys = table["x_px"][()], table["y_px"][()]

    def pixels(element):
        # The table leaves out a zone with an empty mask.
        if element not in outlines:
            print(f"  [WARNING] {name}: no outline of {element}, its mask is empty", flush=True)
            return np.zeros((height, width), bool)
        start, end = outlines[element]
        return fill_outline(xs[start:end], ys[start:end], (height, width))

    def point(element):
        start, _ = outlines[element]
        return xs[start] / scale_x, ys[start] / scale_y

    def grid(size, scale):
        # The grid of the masks of the paper ends with the first cell past the frame, whose
        # pixels in MATLAB numbering reach size + 0.5.
        cells = np.arange(math.ceil((size + 0.5) / (scale * GRID_STEP_CM)) + 1)
        return np.floor(cells * GRID_STEP_CM * scale + 0.5).astype(int) - 1

    ix, iy = grid(width, scale_x), grid(height, scale_y)
    keep_x, keep_y = (ix >= 0) & (ix < width), (iy >= 0) & (iy < height)
    masks = np.zeros((len(ZONE_NAMES), iy.size, ix.size), np.uint8)
    for k, element in enumerate(ZONE_NAMES):
        masks[k][np.ix_(keep_y, keep_x)] = pixels(element)[np.ix_(iy[keep_y], ix[keep_x])]
    centers = []
    for k in range(1, 5):
        rows, cols = np.nonzero(pixels(f"Object{k}Real"))
        # Centroids in MATLAB numbering, as the corner points are stored.
        centers.append(((cols.mean() + 1) / scale_x, (rows.mean() + 1) / scale_y) if rows.size else (np.nan, np.nan))
    return {"masks": masks,
            "zone_names": np.array(ZONE_NAMES),
            "grid_step_cm": np.float64(GRID_STEP_CM),
            "corners_cm": np.array([point(f"ArenaCorner{k}") for k in range(1, 5)], dtype=float),
            "object_centers_cm": np.array(centers, dtype=float),
            "session": np.array(name)}


def read_session(path):
    """Session name, arrays of the session file and arrays of the zone-mask file of one NWB file.

    Parameters
    ----------
    path : Path
        ``_behavior+image+ophys.nwb`` file of one EOF session.

    Returns
    -------
    name : str
        Session name, e.g. ``'NOF_H01_1D'``.
    session : dict
        ``{key: array}`` in the order of ``SESSION_KEYS``, float64.
    zones : dict
        See ``read_zones``.
    """
    with h5py.File(path, "r") as h:
        name = session_name(h["general/session_id"][()].decode())
        beh, ophys = h[BEHAVIOR], h[OPHYS]
        position = beh["Position/xy"]
        tb = position["timestamps"][()]
        series = {"x": position["data"][:, 0], "y": position["data"][:, 1]}
        for key in CONTINUOUS:
            if not np.array_equal(beh[f"BehavioralTimeSeries/{key}/timestamps"][()], tb):
                raise SystemExit(f"{path}: {key} has a timeline of its own, the session files assume one")
            series[key] = beh[f"BehavioralTimeSeries/{key}/data"][()]
        for key in DISCRETE:
            # A variable that never occurs may have no table at all.
            if key not in beh:
                print(f"  [WARNING] {name}: no table of {key}, it is 0 on all frames", flush=True)
                series[key] = np.zeros(tb.size)
            else:
                series[key] = episode_state(beh[key], tb)

        traces = ophys["Fluorescence/C"]
        tc = traces["timestamps"][()]
        if tc[0] < tb[0]:
            print(f"  [WARNING] {name}: calcium starts before behavior; the synchronizer kept the "
                  "first frames of the traces against the later part of the timeline", flush=True)
        # The calcium frames outside the span of the behavior timeline go. The synchronizer cuts
        # the traces by the number of frames kept, from the end, whichever end the frames left.
        tc = tc[(tc >= tb[0]) & (tc <= tb[-1])]
        accepted = ophys["ImageSegmentation/PlaneSegmentation/accepted"][()][traces["rois"][()]]
        session = {"Calcium": traces["data"][:tc.size, :][:, accepted].T.astype(np.float64)}
        session.update({key: to_calcium_frames(series[key], tb, tc) for key in SESSION_KEYS[1:]})
        zones = read_zones(beh["arena_geometry"], name)
    return name, session, zones


def check(name, session, zones, dir_sessions, dir_masks):
    """Print the differences from the files of a session in two folders; return what is beyond tolerance.

    Parameters
    ----------
    name : str
        Session name.
    session, zones : dict
        Arrays from ``read_session``.
    dir_sessions, dir_masks : str or Path
        Folders with ``<name>_aligned.npz`` and ``<name>_zones_cm.npz``.

    Returns
    -------
    list of str
        One entry per array beyond its tolerance.
    """
    bad = []

    def compare(key, ours, theirs, limit):
        ours, theirs = np.asarray(ours), np.asarray(theirs)
        if ours.shape != theirs.shape or ours.dtype != theirs.dtype:
            print(f"  {key:24s} {str(ours.shape):>16s} [FAILED] analysed: {theirs.shape}, {theirs.dtype}; here {ours.dtype}")
            bad.append(key)
            return
        diff = np.abs(ours - theirs)
        ok = np.all(diff <= limit)
        print(f"  {key:24s} {str(ours.shape):>16s} {np.median(diff):12.3e} {diff.max():12.3e} {limit:10.1e}  "
              f"{'[OK]' if ok else '[FAILED]'}")
        if not ok:
            bad.append(key)

    print(f"{name}\n  {'array':24s} {'shape':>16s} {'median diff':>12s} {'max diff':>12s} {'tolerance':>10s}")
    with np.load(Path(dir_sessions) / f"{name}_aligned.npz", allow_pickle=True) as ref:
        keys = [k for k in ref.files if k not in ("_sync_info", "_metadata")]
        if keys != SESSION_KEYS:
            print(f"  [FAILED] members of the analysed file: {keys}")
            bad.append("members")
        for key in SESSION_KEYS:
            compare(key, session[key], ref[key], TOLERANCE[key])
    with np.load(Path(dir_masks) / f"{name}_zones_cm.npz", allow_pickle=False) as ref:
        compare("grid_step_cm", zones["grid_step_cm"], ref["grid_step_cm"], 0.0)
        compare("corners_cm", zones["corners_cm"], ref["corners_cm"], CM_TOLERANCE)
        compare("object_centers_cm", zones["object_centers_cm"], ref["object_centers_cm"], CM_TOLERANCE)
        by_name = dict(zip([str(s) for s in ref["zone_names"]], ref["masks"]))
    print(f"  {'zone mask':24s} {'shape':>16s} {'cells differ':>12s} {'agree':>12s} {'at least':>10s}")
    for zone, ours in zip(ZONE_NAMES, zones["masks"]):
        theirs = by_name.get(zone)
        if theirs is None or ours.shape != theirs.shape:
            print(f"  {zone:24s} {str(ours.shape):>16s} [FAILED] analysed: "
                  f"{None if theirs is None else theirs.shape}")
            bad.append(f"mask {zone}")
            continue
        differ = int(np.count_nonzero(ours != theirs))
        agree = 1 - differ / ours.size
        ok = agree >= MASK_MIN_AGREEMENT
        print(f"  {zone:24s} {str(ours.shape):>16s} {differ:12d} {agree:12.6f} {MASK_MIN_AGREEMENT:10.3f}  "
              f"{'[OK]' if ok else '[FAILED]'}")
        if not ok:
            bad.append(f"mask {zone}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dandiset", type=Path)
    ap.add_argument("--check", nargs=2, metavar=("DIR_SESSIONS", "DIR_MASKS"))
    args = ap.parse_args()
    files = sorted(args.dandiset.rglob(SESSION_FILES))
    if not files:
        raise SystemExit(f"no files {SESSION_FILES} under {args.dandiset}")
    out_sessions = V.get("nof_v1")
    out_masks = P.ZONE_MASKS / "NOF"
    if not args.check:
        out_sessions.mkdir(parents=True, exist_ok=True)
        out_masks.mkdir(parents=True, exist_ok=True)
    paper_sessions = V.fps_table()
    problems, done = [], 0
    for f in files:
        name, session, zones = read_session(f)
        if name[len("NOF_"):] not in paper_sessions:
            print(name, "[SKIPPED] not a session of the paper (fps_by_session.csv)", flush=True)
            continue
        done += 1
        if args.check:
            problems += [f"{name}: {b}" for b in check(name, session, zones, *args.check)]
            continue
        for path, arrays in ((out_sessions / f"{name}_aligned.npz", session),
                             (out_masks / f"{name}_zones_cm.npz", zones)):
            if path.exists():
                raise SystemExit(f"{path} exists; this script does not overwrite")
            np.savez(path, **arrays)
        print(name, "->", out_sessions, flush=True)
    if done < N_SESSIONS:
        print(f"[WARNING] {done} of the {N_SESSIONS} sessions of the paper found")
    if problems:
        raise SystemExit(f"[FAILED] {len(problems)} arrays beyond tolerance: {problems}")
    print(f"[OK] {done} sessions")


if __name__ == "__main__":
    main()
