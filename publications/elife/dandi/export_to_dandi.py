"""Pack the synchronized FOF sessions into NWB files laid out for the DANDI archive.

For every ``synchronized/<session>_aligned.npz`` the experiment is built with
the loader settings of the paper and saved with ``save_exp_to_nwb`` of the
library. The file the library writes is then completed, because on its own it
is not what the archive needs and not what the paper analysed:

* the library saves the traces after its own preprocessing (negative values
  clipped to zero, tie-breaking noise added), so the Calcium series is replaced
  by the array of the synchronized file;
* a behavioural variable that is constant over a session is dropped when the
  experiment is built, so it is added back from the synchronized file;
* all behavioural series come out as SpatialSeries with one common unit; each
  gets its own type, unit and description (see ``fof_metadata.py``);
* the centre columns of the ROI table hold only NaN (the synchronized files
  carry no footprints) and are removed;
* subject, experimenters, institution, keywords and the description of the
  experiment are added, data are gzip-compressed.

Nothing is uploaded. Output layout, as ``dandi organize`` makes it::

    <out>/sub-<mouse>/sub-<mouse>_ses-<session>_behavior+ophys.nwb

Run:  python publications/elife/dandi/export_to_dandi.py [session ...]
      without arguments every file of the synchronized folder is packed;
      a session is named as in the file, e.g. FOF_F28_1D

Inputs are found through ``elife_paths`` (``DRIADA_ELIFE_DATA``); the output
folder is ``--out``, else ``DRIADA_ELIFE_DANDI``, else ``<data root>/dandi_nwb``.
"""
import argparse
import csv
import glob
import os
import sys
import tempfile
import time
import uuid
import warnings
from datetime import date, datetime, timedelta, timezone

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import elife_paths
import fof_metadata as meta

elife_paths.add_tools_to_path()

FPS = 30.0
SEED = 42
# Moscow has kept UTC+3 without daylight saving since 2014; a fixed offset
# avoids depending on the time zone database, which Windows does not ship.
MOSCOW = timezone(timedelta(hours=3), "MSK")
GZIP = dict(compression="gzip", compression_opts=4, shuffle=True)
ID_NAMESPACE = "https://github.com/iabs-neuro/driada/datasets/FOF/"


def read_table(path, key):
    """Read a CSV file into a dict of rows keyed by one column."""
    with open(path, newline="", encoding="utf-8") as fh:
        return {row[key]: row for row in csv.DictReader(fh)}


def dandi_relpath(subject_id, session_id):
    """Relative path of a session file in the layout of ``dandi organize``."""
    ses = "".join(c if c.isalnum() else "-" for c in session_id)
    return os.path.join(f"sub-{subject_id}", f"sub-{subject_id}_ses-{ses}_behavior+ophys.nwb")


def session_facts(session, sessions, subjects):
    """Collect start time, subject and age of a session from the two tables.

    Raises
    ------
    ValueError
        If the age in days of the sessions table disagrees with the date of
        birth and the day of recording.
    """
    row = sessions[session]
    subj = subjects[row["subject_id"]]
    start = datetime.fromisoformat(row["start_local_from_miniscope_ticks"]).replace(tzinfo=MOSCOW)
    born = date.fromisoformat(subj["date_of_birth"])
    age_days = int(row["age_days"])
    if (start.date() - born).days != age_days:
        raise ValueError(f"{session}: age {age_days} d disagrees with birth {born} and start {start}")
    return {
        "session": session,
        "subject_id": row["subject_id"],
        "day": row["day"],
        "start": start,
        "age_days": age_days,
        "date_of_birth": datetime(born.year, born.month, born.day, tzinfo=MOSCOW),
        "sex": subj["sex"],
        "species": subj["species"],
        "strain": subj["strain"],
    }


def build_experiment(session, source):
    """Build the Experiment of a session the way the paper does.

    Frame rate, feature types and seed are those of the INTENSE run of the
    paper. The (x, y) aggregate of that run is left out: it is a derived,
    noise-perturbed copy of x and y and would be stored as one more variable.
    """
    from driada.experiment.exp_build import load_exp_from_aligned_data
    from driada.utils.naming import parse_iabs_filename
    from selectivity_dynamics import get_experiment_config

    parsed = parse_iabs_filename(session)
    data = dict(source)
    # The saver takes the session name, and from it the animal, out of the
    # experiment metadata; the synchronized files carry none.
    data["_metadata"] = {"session_name": session}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_exp_from_aligned_data(
            "IABS",
            {k: parsed[k] for k in ("track", "animal_id", "session")},
            data,
            feature_types=get_experiment_config(parsed["track"])["feature_types"],
            static_features={"fps": FPS},
            verbose=False,
            seed=SEED,
        )


def _replace_dataset(group, name, data):
    """Rewrite a dataset compressed, keeping its attributes."""
    attrs = dict(group[name].attrs)
    del group[name]
    ds = group.create_dataset(name, data=data, **GZIP)
    for key, value in attrs.items():
        ds.attrs[key] = value
    return ds


def _set_text(group, name, text):
    """Replace a scalar text dataset."""
    if name in group:
        del group[name]
    group.create_dataset(name, data=text, dtype=h5py.string_dtype())


def complete_library_file(path, source):
    """Correct, in place, what the library saver cannot be told to write.

    Parameters
    ----------
    path : str
        NWB file written by ``save_exp_to_nwb``.
    source : dict
        Arrays of the synchronized file.

    Returns
    -------
    list of str
        Behavioural variables of the synchronized file that are absent from
        the NWB file (dropped as constant when the experiment was built).
    """
    with h5py.File(path, "r+") as f:
        f["general/devices"][meta.DEVICE_NAME].attrs["description"] = meta.DEVICE_DESCRIPTION

        calcium = f["processing/ophys/Fluorescence/Calcium"]
        if calcium["data"].shape != source["Calcium"].T.shape:
            raise ValueError(f"{path}: Calcium shape differs from the synchronized file")
        _replace_dataset(calcium, "data", source["Calcium"].T)
        calcium.attrs["description"] = meta.CALCIUM_DESCRIPTION

        ps = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]
        for col in ("center_x", "center_y"):
            if not np.all(np.isnan(ps[col][:])):
                raise ValueError(f"{path}: column {col} holds values, refusing to drop it")
            del ps[col]
        ps.attrs["colnames"] = np.array(["image_mask"], dtype=h5py.string_dtype())
        ps.attrs["description"] = meta.ROI_TABLE_DESCRIPTION
        ps["image_mask"].attrs["description"] = meta.IMAGE_MASK_DESCRIPTION
        _replace_dataset(ps, "image_mask", ps["image_mask"][:])

        beh = f["processing/behavior"]
        position = beh["Position/SpatialSeries"]
        position.attrs["description"] = meta.POSITION_DESCRIPTION
        _replace_dataset(position, "data", position["data"][:])

        for name in beh:
            if name == "Position":
                continue
            # An unknown variable must stop the export rather than go out with
            # a placeholder unit.
            kind, unit, description, frame = meta.BEHAVIOR_SERIES[name]
            series = beh[name]
            series.attrs["neurodata_type"] = kind
            series.attrs["description"] = description
            if frame is None:
                del series["reference_frame"]
            else:
                _set_text(series, "reference_frame", frame)
            _replace_dataset(series, "data", series["data"][:]).attrs["unit"] = unit

        stored = set(beh) - {"Position"} | {"x", "y"}
    return sorted(set(source) - stored - {"Calcium"})


def finish_file(tmp_path, final_path, facts, source, missing):
    """Add the metadata and the dropped variables, write the compact final file.

    The file is exported rather than appended to: rewriting datasets in place
    leaves the old uncompressed blocks inside the HDF5 file, and an export
    copies only what is in use.
    """
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from pynwb import NWBHDF5IO, TimeSeries
    from pynwb.file import Subject

    with NWBHDF5IO(tmp_path, "r") as src_io:
        nwb = src_io.read()
        nwb.subject = Subject(
            subject_id=facts["subject_id"],
            species=facts["species"],
            sex=facts["sex"],
            age=f"P{facts['age_days']}D",
            date_of_birth=facts["date_of_birth"],
            strain=facts["strain"],
            description=meta.SUBJECT_DESCRIPTION,
        )
        nwb.experimenter = tuple(meta.EXPERIMENTERS)
        nwb.institution = meta.INSTITUTION
        if meta.LAB:
            nwb.lab = meta.LAB
        nwb.keywords = tuple(meta.KEYWORDS)
        nwb.experiment_description = meta.EXPERIMENT_DESCRIPTION
        nwb.virus = meta.VIRUS
        nwb.surgery = meta.SURGERY

        for name in missing:
            kind, unit, description, _ = meta.BEHAVIOR_SERIES[name]
            if kind != "TimeSeries":
                raise ValueError(f"{name}: only plain series are expected to be constant")
            nwb.processing["behavior"].add(TimeSeries(
                name=name,
                data=H5DataIO(np.asarray(source[name]), **GZIP),
                unit=unit,
                rate=FPS,
                description=description + " Constant over this session.",
            ))

        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        with NWBHDF5IO(final_path, "w") as out_io:
            out_io.export(src_io=src_io, nwbfile=nwb, write_args={"link_data": False})


def export_session(npz_path, out_dir, sessions, subjects, overwrite=False):
    """Pack one synchronized file; return a summary row."""
    from driada.experiment.nwb import save_exp_to_nwb

    session = os.path.basename(npz_path)[: -len("_aligned.npz")]
    facts = session_facts(session, sessions, subjects)
    final_path = os.path.join(out_dir, dandi_relpath(facts["subject_id"], session))
    if os.path.exists(final_path) and not overwrite:
        raise FileExistsError(f"{final_path} exists; pass --overwrite to replace it")

    source = dict(np.load(npz_path, allow_pickle=False))
    exp = build_experiment(session, source)

    with tempfile.TemporaryDirectory() as tmp:
        save_exp_to_nwb(
            exp, tmp,
            session_description=meta.SESSION_DESCRIPTION.format(
                day_number=facts["day"].rstrip("D"), subject_id=facts["subject_id"]),
            # Derived from the session name, so repacking gives the same identifier.
            identifier=f"{session}-{uuid.uuid5(uuid.NAMESPACE_URL, ID_NAMESPACE + session)}",
            session_start_time=facts["start"],
            protocol=meta.PROTOCOL,
            device_name=meta.DEVICE_NAME,
            imaging_plane_desc=meta.IMAGING_PLANE_DESCRIPTION,
            indicator=meta.INDICATOR,
            location=meta.LOCATION,
            pos_reference_frame=meta.POSITION_REFERENCE_FRAME,
            pos_unit=meta.POSITION_UNIT,
        )
        tmp_path = os.path.join(tmp, f"{session}.nwb")
        if not os.path.exists(tmp_path):
            raise FileNotFoundError(f"{session}: the saver wrote {os.listdir(tmp)}, not {session}.nwb")
        missing = complete_library_file(tmp_path, source)
        if os.path.exists(final_path):
            os.remove(final_path)
        finish_file(tmp_path, final_path, facts, source, missing)

    return {
        "session": session,
        "file": os.path.relpath(final_path, out_dir).replace(os.sep, "/"),
        "n_neurons": source["Calcium"].shape[0],
        "n_frames": source["Calcium"].shape[1],
        "start": facts["start"].isoformat(),
        "age": f"P{facts['age_days']}D",
        "sex": facts["sex"],
        "constant_variables_added": " ".join(missing),
        "size_mb": round(os.path.getsize(final_path) / 1e6, 1),
    }


def main():
    default_out = os.environ.get("DRIADA_ELIFE_DANDI",
                                 os.path.join(elife_paths.DATA_ROOT, "dandi_nwb"))
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sessions", nargs="*", help="sessions to pack, e.g. FOF_F28_1D (default: all)")
    parser.add_argument("--out", default=default_out, help="output folder of the NWB files")
    parser.add_argument("--subjects", default=os.path.join(HERE, "subjects.csv"))
    parser.add_argument("--sessions-table", default=os.path.join(HERE, "sessions.csv"))
    parser.add_argument("--manifest", help="write a CSV summary of the packed files to this path")
    parser.add_argument("--overwrite", action="store_true", help="replace NWB files that already exist")
    args = parser.parse_args()

    import driada
    print(f"[INFO] driada {driada.__version__} from {os.path.dirname(driada.__file__)}", flush=True)

    subjects = read_table(args.subjects, "subject_id")
    sessions = read_table(args.sessions_table, "session")
    if args.sessions:
        paths = [os.path.join(elife_paths.SYNC_DIR, f"{s}_aligned.npz") for s in args.sessions]
    else:
        paths = sorted(glob.glob(os.path.join(elife_paths.SYNC_DIR, "*_aligned.npz")))
    if not paths:
        print(f"[FAILED] no synchronized files in {elife_paths.SYNC_DIR}")
        return 1

    rows, failed = [], []
    for i, path in enumerate(paths, 1):
        t0 = time.time()
        try:
            row = export_session(path, args.out, sessions, subjects, overwrite=args.overwrite)
        except Exception as exc:
            failed.append(os.path.basename(path))
            print(f"[FAILED] {os.path.basename(path)}: {type(exc).__name__}: {exc}", flush=True)
            continue
        rows.append(row)
        print(f"[OK] {i}/{len(paths)} {row['file']}  {row['n_neurons']} neurons x {row['n_frames']} frames  "
              f"{row['size_mb']} MB  {time.time() - t0:.0f} s"
              + (f"  added constant: {row['constant_variables_added']}" if row["constant_variables_added"] else ""),
              flush=True)

    # The summary is kept out of the output folder by default: everything in
    # that folder is taken for a part of the dataset by the archive client.
    if rows and args.manifest:
        with open(args.manifest, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"[INFO] {len(rows)} files, {sum(r['size_mb'] for r in rows):.0f} MB in {args.out}")
    if failed:
        print(f"[FAILED] {len(failed)} sessions: {' '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
