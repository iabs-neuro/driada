"""Wavelet events of the example neurons of Fig. 1a and Fig. 2a at the frame rate of the session.

The detector is the one DRIADA 1.2.2 runs when an experiment is built with
reconstruct_spikes='wavelet' (Experiment._reconstruct_spikes ->
spike_reconstruction.reconstruct_spikes(method='wavelet')): every raw calcium row becomes a
TimeSeries and is processed on its own, so running it on the plotted rows only gives the events
of a full-session build. This is checked against the event cache that the classic method built from
whole sessions (classic/detect_events.py), when that cache exists.

Drawn events are the runs of the binary event train that DRIADA makes from the detector output
(events_to_ts_array: each event lasts 0.5-2.5 s), the same train that Fig. 2b-e and the classic
method use. The raw detector intervals are saved too.

Neurons: the panel scripts name rows of an earlier export of the sessions whose calcium matrix has one
extra leading row, so row k there is row k - 1 of the synchronized file.

usage: python -X utf8 -B figure_chain/figures_v10.4/work/events_wvt.py
       -> figures_v10.4/work/events/events_<session>.npz, events_check.json
          under [figures] output_dir/figure_chain
"""
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # figure_chain/
import _paths_fig as _F  # noqa: E402  (puts the paper folder on sys.path)

driada = _F.P.use_driada()
from driada.experiment.spike_reconstruction import reconstruct_spikes  # noqa: E402
from driada.information.info_base import MultiTimeSeries, TimeSeries  # noqa: E402

V2 = _F.P.NOF_ALIGNED
FPS_TABLE = _F.V.get("fps_table")
CLASSIC = _F.V.get("classic_dir") / "events"
OUT = _F.out_dir(HERE) / "events"

# session -> rows of the earlier export used by the panel scripts
TARGETS = {
    "NOF_H01_1D": [36, 37, 38],   # Fig. 1a (fig1a_calcium_traces.py, cache keys)
    "NOF_H31_1D": [60],           # Fig. 2a (fig2 data_loader.compute_fig2a_data)
}


def runs_of(sp):
    """Start (inclusive) and end (exclusive) frames of the runs of sp > 0."""
    on = np.flatnonzero(np.diff(np.r_[0, (np.asarray(sp) > 0).astype(np.int8), 0]) != 0)
    return on[0::2], on[1::2]


def detect(rows, fps):
    mts = MultiTimeSeries([TimeSeries(r.astype(float), discrete=False) for r in rows],
                          allow_zero_columns=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp, meta = reconstruct_spikes(mts, method="wavelet", fps=float(fps))
    return np.asarray(sp.data), meta


def main():
    with open(FPS_TABLE, newline="") as f:
        table = {r["session"]: (float(r["fps"]), r["source"]) for r in csv.DictReader(f)}
    OUT.mkdir(exist_ok=True)
    report = {"driada": driada.__version__}
    for session, old_rows in TARGETS.items():
        z = np.load(V2 / f"{session}_aligned.npz", allow_pickle=True)
        meta = z["_metadata"].item()
        fps = float(meta["fps"])
        if abs(fps - table[session][0]) > 1e-9:
            raise SystemExit(f"{session}: _metadata fps {fps} != table {table[session][0]}")
        ca = np.asarray(z["Calcium"])
        rows = [r_old - 1 for r_old in old_rows]
        sp_true, meta_true = detect(ca[rows], fps)
        sp_old, meta_old = detect(ca[rows], 20.0)
        out = {"session": session, "fps": fps, "fps_source": meta.get("fps_source", ""),
               "rows_aligned": np.array(rows), "rows_older_export": np.array(old_rows),
               "n_frames": ca.shape[1]}
        rep = {"fps": fps, "fps_source": meta.get("fps_source", ""), "n_frames": int(ca.shape[1]),
               "older_export_row_to_aligned_row": dict(zip(map(str, old_rows), rows)), "neurons": {}}
        for k, r in enumerate(rows):
            for tag, sp, md in (("true", sp_true, meta_true), ("old20", sp_old, meta_old)):
                st, en = runs_of(sp[k])
                out[f"n{r}_{tag}_sp"] = sp[k].astype(np.int8)
                out[f"n{r}_{tag}_run_st"], out[f"n{r}_{tag}_run_en"] = st, en
                out[f"n{r}_{tag}_ev_st"] = np.asarray(md["start_events"][k], int)
                out[f"n{r}_{tag}_ev_en"] = np.asarray(md["end_events"][k], int)
            st, en = out[f"n{r}_true_run_st"], out[f"n{r}_true_run_en"]
            in5000 = st < 5000
            rep["neurons"][str(r)] = {
                "events_true": int(len(out[f"n{r}_true_ev_st"])), "runs_true": int(len(st)),
                "runs_true_in_first_5000_frames": int(in5000.sum()),
                "events_at_20fps": int(len(out[f"n{r}_old20_ev_st"])),
                "runs_at_20fps": int(len(out[f"n{r}_old20_run_st"])),
                "same_runs_as_at_20fps": bool(np.array_equal(out[f"n{r}_old20_sp"], out[f"n{r}_true_sp"])),
            }
        # whole-session cache of the classic method, built with the same detector
        cache = CLASSIC / f"events_{session}_true.npz"
        if cache.exists():
            c = np.load(cache)
            chk = {"cache": cache.name, "cache_fps": float(c["fps"])}
            for r in rows:
                me, mr = c["ev_cell"] == r, c["run_cell"] == r
                chk[str(r)] = {
                    "events_equal": bool(np.array_equal(c["ev_st"][me], out[f"n{r}_true_ev_st"])
                                         and np.array_equal(c["ev_en"][me], out[f"n{r}_true_ev_en"])),
                    "runs_equal": bool(np.array_equal(c["run_st"][mr], out[f"n{r}_true_run_st"])
                                       and np.array_equal(c["run_en"][mr], out[f"n{r}_true_run_en"])),
                }
            rep["check_against_whole_session_cache"] = chk
        else:
            rep["check_against_whole_session_cache"] = f"no cache {cache.name} (not built yet)"
        np.savez_compressed(OUT / f"events_{session}.npz", **out)
        report[session] = rep
        print(session, json.dumps(rep, indent=1), flush=True)
    (OUT / "events_check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
