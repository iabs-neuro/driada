"""One half of the false-discovery calibration on the data with their own frame rate.

ed_calibration/run_part.py of the paper, run unchanged, except for the partner
of the "cross" null. calib_null.null_data pairs the calcium of mouse i with the
behaviour of mouse i+1 on the same day and asserts that the two files have the
same frame rate. With the rates in _metadata that holds for 9 of the 16 cross
units only: 5 partners are recorded at the other rate (20 against 30 frames/s)
and 2 differ in the second decimal (19.76 against 19.8). Here the partner is
the first mouse after mouse i, in the order of calib_null.MICE, whose session of
the same day is in the same rate group (rates within 0.5 frames/s); the
experiment takes the rate of the calcium session, as calib_null does. Units
whose partner changes are printed. Everything else - the shift null, the
synthetic null (generated at 20 frames/s, as in the paper), the INTENSE call,
the output files - is calib_null.py and run_part.py.

Each part runs INTENSE without parallelization (calib_null.run_intense), so a
part uses one core. Besides the halves A and B of run_part.py, "k/N" runs the
k-th of N parts (k = 0..N-1), balanced by number of neurons in the same way as
run_part.plan, so that N parts can run side by side on N cores. Results are one
file per unit and existing files are skipped, so any split gives the same files.

usage (from the paper folder, as ed_calibration/calib_run_part_fps.py):
       python calib_run_part_fps.py A|B          (two halves, as run_part.py)
       python calib_run_part_fps.py k/N          (k-th of N parts)
       python calib_run_part_fps.py --partners   (only print the cross pairs)
"""
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER / "ed_calibration"))
import calib_null as cn  # noqa: E402  (checks the DRIADA snapshot on import)

SAME_RATE = 0.5  # frames/s: sessions closer than this are in the same rate group


def rate(session):
    fps = cn.get_npz_metadata(cn.DATA / f"{session}_aligned.npz")["fps"]
    if fps is None:
        raise SystemExit(f"{session} has no _metadata['fps']: this script needs the v2 data")
    return float(fps)


def partner(session):
    """Behaviour session of the cross null: next mouse, same day, same rate group."""
    _, mouse, day = session.split("_")
    i, own = cn.MICE.index(mouse), rate(session)
    for k in range(1, len(cn.MICE)):
        other = f"NOF_{cn.MICE[(i + k) % len(cn.MICE)]}_{day}"
        if abs(rate(other) - own) < SAME_RATE:
            return other
    raise RuntimeError(f"no session of day {day} at the rate of {session}")


_orig_null_data = cn.null_data


def null_data_same_rate(null, session, rng_seed):
    if null != "cross":
        return _orig_null_data(null, session, rng_seed)
    path = cn.DATA / f"{session}_aligned.npz"
    data = cn.load_npz(path)
    fps = rate(session)
    beh_keys = [k for k in data if k != "Calcium"]
    other_session = partner(session)
    odata = cn.load_npz(cn.DATA / f"{other_session}_aligned.npz")
    n = min(data["Calcium"].shape[1], odata["Calcium"].shape[1])
    data = {"Calcium": data["Calcium"][:, :n]}
    for k in beh_keys:
        data[k] = odata[k][:n]
    info = {"beh_session": other_session, "n_frames": n, "beh_fps": rate(other_session)}
    return data, fps, path, info


def print_partners():
    cross = [f"NOF_{m}_{k % 4 + 1}D" for k, m in enumerate(cn.MICE)]
    fails, changed, unequal = 0, 0, 0
    for s in cross:
        _, mouse, day = s.split("_")
        old = f"NOF_{cn.MICE[(cn.MICE.index(mouse) + 1) % len(cn.MICE)]}_{day}"
        new = partner(s)
        old_ok = rate(old) == rate(s)
        fails += not old_ok
        changed += new != old
        unequal += rate(new) != rate(s)
        note = "same partner" if new == old else f"partner was {old} ({rate(old)})"
        print(f"  {s} ({rate(s)}) -> {new} ({rate(new)}, difference {rate(new) - rate(s):+.2f}); "
              f"{note}; calib_null assert (exact equality) with the old partner: {'passes' if old_ok else 'FAILS'}")
    print(f"{len(cross)} cross units: the exact-equality assert of calib_null.py:131 fails for {fails}; "
          f"{changed} partners change; {unequal} new pairs still differ in rate (all within {SAME_RATE} frames/s)")


def split_tasks(n):
    """The tasks of run_part.plan split greedily by neurons into n parts (n = 2 gives A, B)."""
    import run_part
    shift = [f"NOF_{m}_{d}D" for m in cn.MICE for d in range(1, 5)]
    cross = [f"NOF_{m}_{j % 4 + 1}D" for j, m in enumerate(cn.MICE)]
    tasks = [("shift", s, run_part.n_cells(s)) for s in shift] + \
            [("cross", s, run_part.n_cells(s)) for s in cross] + [("synth", j, 300) for j in range(8)]
    parts, load = [[] for _ in range(n)], [0] * n
    for t in sorted(tasks, key=lambda t: -t[2]):
        p = load.index(min(load))
        parts[p].append(t)
        load[p] += t[2]
    return parts, load


def run_k_of_n(k, n):
    """The k-th of n parts."""
    parts, load = split_tasks(n)
    out = cn.P.CALIBRATION_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    print(f"part {k}/{n}: {len(parts[k])} tasks, {load[k]} neurons", flush=True)
    for null, unit, _ in parts[k]:
        if null == "synth":
            cn.run_synth(unit, out)
        else:
            cn.run_real(null, unit, out)
    print(f"[OK] part {k}/{n} done", flush=True)


def main():
    cn.null_data = null_data_same_rate
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--partners":
        print_partners()
    elif "/" in arg:
        k, n = map(int, arg.split("/"))
        run_k_of_n(k, n)
    else:
        import run_part
        run_part.main()


if __name__ == "__main__":
    main()
