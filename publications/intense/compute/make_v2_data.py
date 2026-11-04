"""SynchronizedData26_v2: the v1 sessions with their frame rate in ``_metadata``.

Each ``NOF_*_aligned.npz`` of v1 is copied byte for byte and one member is
appended to the archive: ``_metadata`` = {'fps': ..., 'fps_source': ...} from
fps_by_session.csv, a pickled 0-d object array like the ``_sync_info`` the
files already carry. The members of v1 are not rewritten, so every array of v2
is the same bytes as in v1. DRIADA 1.2.2 reads ``_metadata['fps']``
(tools/load_synchronized_experiments.get_npz_metadata) and sets it as exp.fps.

The script never overwrites a v2 file: a session already in v2 is only
verified. The verification compares, for every member of v1, the sha256 of
its stored bytes and of the loaded array (dtype, shape and data) with v2,
checks that v2 has exactly the members of v1 plus ``_metadata``, and that
``_metadata`` holds the rate of the table. The hashes go to v2_hashes.csv.

usage: python compute/make_v2_data.py [--verify-only]
"""
import csv
import hashlib
import io
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402
import paths_v11 as V  # noqa: E402

V1 = V.get("nof_v1")
V2 = P.NOF_ALIGNED
FPS_TABLE = V.get("fps_table")
HASHES = P.NUMBERS_DIR / "v2_hashes.csv"
META = "_metadata.npy"


def read_table():
    with open(FPS_TABLE, newline="") as f:
        return {r["session"]: {"fps": float(r["fps"]), "fps_source": r["source"]}
                for r in csv.DictReader(f)}


def metadata_member(meta):
    """Bytes of a .npy file holding ``meta`` as a 0-d object array, as np.savez writes it."""
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.array(meta, dtype=object), allow_pickle=True)
    return buf.getvalue()


def write_v2(session, meta):
    src, dst = V1 / f"{session}_aligned.npz", V2 / f"{session}_aligned.npz"
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    with zipfile.ZipFile(tmp, "a") as z:
        if META in z.namelist():
            raise RuntimeError(f"{src} already has {META}")
        info = zipfile.ZipInfo(META, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        z.writestr(info, metadata_member(meta))
    os.replace(tmp, dst)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def array_sha(a):
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes() if a.dtype != object else repr(a.item()).encode())
    return h.hexdigest()


def verify(session, meta):
    """Rows of the hash table for one session and a list of the differences found."""
    p1, p2 = V1 / f"{session}_aligned.npz", V2 / f"{session}_aligned.npz"
    rows, bad = [], []
    with zipfile.ZipFile(p1) as z1, zipfile.ZipFile(p2) as z2:
        n1, n2 = z1.namelist(), z2.namelist()
        if n2 != n1 + [META]:
            bad.append(f"members {n2} != v1 members + {META}")
        with np.load(p1, allow_pickle=True) as d1, np.load(p2, allow_pickle=True) as d2:
            for name in n1:
                key = name[:-len(".npy")]
                b1, b2 = sha(z1.read(name)), sha(z2.read(name))
                a1, a2 = array_sha(d1[key]), array_sha(d2[key])
                ok = b1 == b2 and a1 == a2
                if not ok:
                    bad.append(f"{key} differs")
                rows.append([session, key, b1, b2, a1, a2, ok])
            got = d2["_metadata"].item()
            if got != meta:
                bad.append(f"_metadata {got} != {meta}")
            rows.append([session, "_metadata", "", sha(z2.read(META)), "", repr(got), got == meta])
    return rows, bad


def main():
    verify_only = "--verify-only" in sys.argv[1:]
    table = read_table()
    sessions = sorted(p.name[:-len("_aligned.npz")] for p in V1.glob("NOF_*_aligned.npz"))
    assert len(sessions) == 64 and set(sessions) == set(table), set(sessions) ^ set(table)
    V2.mkdir(parents=True, exist_ok=True)
    written = 0
    for s in sessions:
        if not (V2 / f"{s}_aligned.npz").exists():
            if verify_only:
                raise SystemExit(f"[FAILED] {s} is not in v2")
            write_v2(s, table[s])
            written += 1
    rows, problems = [], []
    for s in sessions:
        r, bad = verify(s, table[s])
        rows += r
        problems += [f"{s}: {b}" for b in bad]
    HASHES.parent.mkdir(parents=True, exist_ok=True)
    with open(HASHES, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "array", "sha256_stored_v1", "sha256_stored_v2",
                    "sha256_array_v1", "sha256_array_v2", "equal"])
        w.writerows(rows)
    extra = sorted(p.name for p in V2.iterdir() if p.name not in {f"{s}_aligned.npz" for s in sessions})
    if extra:
        problems.append(f"unexpected files in v2: {extra}")
    n_arrays = sum(r[1] != "_metadata" for r in rows)
    if problems:
        print("[FAILED]", *problems, sep="\n  ")
        sys.exit(1)
    print(f"[OK] {written} files written, {len(sessions)} verified: {n_arrays} arrays identical to v1, "
          f"_metadata matches fps_by_session.csv in every file -> {HASHES}")


if __name__ == "__main__":
    main()
