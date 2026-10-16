"""Compare every packed NWB file with the synchronized file it was made from.

The check shares no code with the exporter. Each NWB file is read twice:

* directly with ``h5py``: the Calcium array and every behavioural variable
  must equal the arrays of the synchronized file, element by element;
* with ``load_exp_from_nwb`` of the library: what a user of the library gets.
  The library preprocesses traces on load (negative values clipped to zero,
  tie-breaking noise of the order of 1e-8 added) and drops variables that are
  constant over a session, so its traces are compared both with the source
  and with the source after the same clipping, and the dropped variables are
  listed.

Subject and session metadata are compared with ``subjects.csv`` and
``sessions.csv``.

Run:  python publications/elife/dandi/check_roundtrip.py [--out <folder>] [--report <csv>] [session ...]
"""
import argparse
import csv
import glob
import os
import sys
import warnings

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import elife_paths

FPS = 30.0


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _maxdiff(a, b):
    """Largest absolute difference; infinity when the shapes differ."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def read_direct(path):
    """Read data and metadata of an NWB file with h5py alone."""
    with h5py.File(path, "r") as f:
        calcium = f["processing/ophys/Fluorescence/Calcium"]
        data = {"Calcium": calcium["data"][:].T}
        rates = {"Calcium": float(calcium["starting_time"].attrs["rate"])}
        behavior = f["processing/behavior"]
        position = behavior["Position/SpatialSeries"]
        data["x"], data["y"] = position["data"][:, 0], position["data"][:, 1]
        rates["Position"] = float(position["starting_time"].attrs["rate"])
        for name in behavior:
            if name != "Position":
                data[name] = behavior[name]["data"][:]
                rates[name] = float(behavior[name]["starting_time"].attrs["rate"])
        subject = f["general/subject"]
        info = {
            "session_id": _text(f["general/session_id"][()]),
            "start": _text(f["session_start_time"][()]),
            "subject_id": _text(subject["subject_id"][()]),
            "sex": _text(subject["sex"][()]),
            "species": _text(subject["species"][()]),
            "strain": _text(subject["strain"][()]),
            "age": _text(subject["age"][()]),
            "date_of_birth": _text(subject["date_of_birth"][()]),
        }
    return data, rates, info


def check_session(npz_path, nwb_path, sessions, subjects, with_library=True):
    """Compare one NWB file with its synchronized file; return a result row."""
    session = os.path.basename(npz_path)[: -len("_aligned.npz")]
    source = dict(np.load(npz_path, allow_pickle=False))
    data, rates, info = read_direct(nwb_path)

    row = {"session": session, "file": os.path.basename(nwb_path)}
    problems = []

    if set(data) != set(source):
        problems.append(f"variables differ: only in NWB {sorted(set(data) - set(source))}, "
                        f"only in source {sorted(set(source) - set(data))}")
    row["n_variables"] = len(set(source) - {"Calcium"})
    row["direct_calcium_maxdiff"] = _maxdiff(data["Calcium"], source["Calcium"])
    behavior_diffs = {k: _maxdiff(data[k], source[k]) for k in source if k != "Calcium" and k in data}
    row["direct_behavior_maxdiff"] = max(behavior_diffs.values())
    # Equal values could still differ in type; the reverse converter promises
    # the same arrays, so the types are held to the source as well.
    bad_types = [k for k in source if k in data and data[k].dtype != source[k].dtype]
    if row["direct_calcium_maxdiff"] != 0 or row["direct_behavior_maxdiff"] != 0:
        problems.append("stored values differ from the source")
    if bad_types:
        problems.append(f"dtype differs: {bad_types}")
    if any(r != FPS for r in rates.values()):
        problems.append(f"rate is not {FPS}: {rates}")

    ses, subj = sessions[session], subjects[sessions[session]["subject_id"]]
    expected = {
        "session_id": session,
        "start": ses["start_local_from_miniscope_ticks"] + "+03:00",
        "subject_id": ses["subject_id"],
        "sex": subj["sex"],
        "species": subj["species"],
        "strain": subj["strain"],
        "age": f"P{ses['age_days']}D",
    }
    for key, value in expected.items():
        if info[key] != value:
            problems.append(f"{key}: file has {info[key]!r}, table has {value!r}")
    if not info["date_of_birth"].startswith(subj["date_of_birth"]):
        problems.append(f"date_of_birth: file has {info['date_of_birth']!r}, table has {subj['date_of_birth']!r}")

    if with_library:
        from driada.experiment.nwb import load_exp_from_nwb

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp = load_exp_from_nwb(nwb_path)
        lib_calcium = np.asarray(exp.calcium.data)
        row["library_calcium_maxdiff_vs_source"] = _maxdiff(lib_calcium, source["Calcium"])
        row["library_calcium_maxdiff_vs_clipped_source"] = _maxdiff(
            lib_calcium, np.clip(source["Calcium"], 0, None))
        returned = [k for k in source if k != "Calcium" and k in exp.dynamic_features]
        row["library_behavior_maxdiff"] = max(
            _maxdiff(exp.dynamic_features[k].data, source[k]) for k in returned)
        row["library_dropped_variables"] = " ".join(
            k for k in source if k != "Calcium" and k not in exp.dynamic_features)
        if exp.n_cells != source["Calcium"].shape[0] or exp.n_frames != source["Calcium"].shape[1]:
            problems.append("library returns another number of neurons or frames")
        if float(exp.fps) != FPS:
            problems.append(f"library returns fps {exp.fps}")
        if row["library_behavior_maxdiff"] != 0:
            problems.append("behaviour returned by the library differs from the source")
        # The noise the library adds on load is below 1e-8 by construction;
        # the margin covers rounding of the sum.
        if row["library_calcium_maxdiff_vs_clipped_source"] > 1e-8 * (1 + 1e-4):
            problems.append("traces returned by the library differ from the clipped source by more than its noise")

    row["problems"] = "; ".join(problems)
    return row


def main():
    default_out = os.environ.get("DRIADA_ELIFE_DANDI",
                                 os.path.join(elife_paths.DATA_ROOT, "dandi_nwb"))
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sessions", nargs="*", help="sessions to check (default: all synchronized files)")
    parser.add_argument("--out", default=default_out, help="folder with the packed NWB files")
    parser.add_argument("--subjects", default=os.path.join(HERE, "subjects.csv"))
    parser.add_argument("--sessions-table", default=os.path.join(HERE, "sessions.csv"))
    parser.add_argument("--report", help="write the per-file results to this CSV")
    parser.add_argument("--no-library", action="store_true", help="skip the read through the library")
    args = parser.parse_args()

    with open(args.subjects, newline="", encoding="utf-8") as fh:
        subjects = {r["subject_id"]: r for r in csv.DictReader(fh)}
    with open(args.sessions_table, newline="", encoding="utf-8") as fh:
        sessions = {r["session"]: r for r in csv.DictReader(fh)}

    by_session = {}
    for path in glob.glob(os.path.join(args.out, "**", "*.nwb"), recursive=True):
        with h5py.File(path, "r") as f:
            by_session[_text(f["general/session_id"][()])] = path

    if args.sessions:
        npz_paths = [os.path.join(elife_paths.SYNC_DIR, f"{s}_aligned.npz") for s in args.sessions]
    else:
        npz_paths = sorted(glob.glob(os.path.join(elife_paths.SYNC_DIR, "*_aligned.npz")))

    rows = []
    for npz_path in npz_paths:
        session = os.path.basename(npz_path)[: -len("_aligned.npz")]
        if session not in by_session:
            rows.append({"session": session, "problems": "no NWB file for this session"})
        else:
            rows.append(check_session(npz_path, by_session[session], sessions, subjects,
                                      with_library=not args.no_library))
        row = rows[-1]
        tag = "[FAILED]" if row["problems"] else "[OK]"
        print(f"{tag} {session}: direct calcium {row.get('direct_calcium_maxdiff')}, "
              f"direct behaviour {row.get('direct_behavior_maxdiff')}, "
              f"library calcium vs clipped source {row.get('library_calcium_maxdiff_vs_clipped_source')}, "
              f"library dropped [{row.get('library_dropped_variables', '')}] {row['problems']}", flush=True)

    extra = sorted(set(by_session) - {r["session"] for r in rows}) if not args.sessions else []
    if extra:
        print(f"[FAILED] NWB files without a synchronized file: {' '.join(extra)}")

    if args.report:
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "session", k == "problems", k))
        with open(args.report, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    failed = [r for r in rows if r["problems"]]

    def worst(key):
        values = [r[key] for r in rows if key in r]
        return max(values) if values else "not measured"

    print(f"files compared: {len(rows)}, with problems: {len(failed)}")
    print(f"largest difference, direct read: calcium {worst('direct_calcium_maxdiff')}, "
          f"behaviour {worst('direct_behavior_maxdiff')}")
    print(f"largest difference, through the library: calcium vs source "
          f"{worst('library_calcium_maxdiff_vs_source')}, calcium vs clipped source "
          f"{worst('library_calcium_maxdiff_vs_clipped_source')}, behaviour {worst('library_behavior_maxdiff')}")
    return 1 if failed or extra else 0


if __name__ == "__main__":
    sys.exit(main())
