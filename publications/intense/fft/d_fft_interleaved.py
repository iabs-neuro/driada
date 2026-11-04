"""FFT-cache speed-up for one neuron-feature pair, with the two engines interleaved call by call in one
process. fft/d_fft_core.py imports cycles() and call() of this file.

Same pair, procedure and settings as d_fft_timing2.py (synthetic benchmark of Fig. 4, SNR 8,
realization 0, the first ground-truth pair the saved run detected; delay search, 100 / 10,000
shuffles, p < 0.001, ds = 5, type-based representation, seed 42; one thread, no parallelization).
On a shared machine the load changes within minutes, so a ratio of two calls made minutes apart in
different processes mixes the speed-up with the change of load. Here one process first makes one
call with each engine (compilation of the numerical kernels on first use), then ROUNDS rounds of
two calls, one with the cache (engine='auto') and one without (engine='loop'), the order alternating
between rounds. For every call it records wall time and the CPU cycles of the process
(QueryProcessCycleTime of Windows, which counts cycles exactly, unlike the CPU time sampled at clock
ticks); the ratio of a round is loop / auto for each of the two. The machine load over every call
(psutil.cpu_percent) is recorded beside it. Windows only (QueryProcessCycleTime of kernel32).

usage: python fft/d_fft_interleaved.py c|d [ROUNDS]  ->  <[fft] out_dir>/fft_interleaved_<type>.json
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import ctypes  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from ctypes import wintypes  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d_fft_timing2 as T  # noqa: E402  (loads DRIADA 1.2.2 through intense_paths)

import numpy as np  # noqa: E402
import psutil  # noqa: E402

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.GetCurrentProcess.restype = wintypes.HANDLE
_k32.QueryProcessCycleTime.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_ulonglong)]
_k32.QueryProcessCycleTime.restype = wintypes.BOOL


def cycles():
    c = ctypes.c_ulonglong()
    if not _k32.QueryProcessCycleTime(_k32.GetCurrentProcess(), ctypes.byref(c)):
        raise OSError(ctypes.get_last_error())
    return c.value


def call(rb, d, engine, ftype):
    psutil.cpu_percent(interval=None)
    c0 = cycles()
    seconds, cpu, feat, st, sg, timings = T.one_call(rb, d, engine, ftype)
    cyc = cycles() - c0
    return dict(engine=engine, seconds=round(seconds, 4), gcycles=round(cyc / 1e9, 3),
                machine_cpu_load_pct=psutil.cpu_percent(interval=None), stage1=bool(sg.get("stage1", False)),
                stage2=bool(sg.get("stage2", False)), pval=st.get("pval"), opt_delay=st.get("opt_delay"),
                tested=str(feat), timings={k: round(v, 4) for k, v in timings.items() if isinstance(v, (int, float))},
                fft_type_counts={str(k): v for k, v in (timings.get("fft_type_counts") or {}).items()})


def main():
    ftype = sys.argv[1]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    rb = T.bench()
    d = np.load(T.OUT / f"pair2_{ftype}.npz")
    first = [call(rb, d, e, ftype) for e in ("auto", "loop")]
    for r in first:
        print(json.dumps(dict(phase="first", **{k: r[k] for k in ("engine", "seconds", "gcycles", "stage2", "pval")})),
              flush=True)
    rows = []
    for k in range(rounds):
        order = ("auto", "loop") if k % 2 == 0 else ("loop", "auto")
        rec = {e: call(rb, d, e, ftype) for e in order}
        rows.append(dict(round=k, order=list(order), **rec))
        print(json.dumps(dict(round=k, **{e: {x: rec[e][x] for x in ("seconds", "gcycles", "machine_cpu_load_pct",
                                                                    "stage2", "pval")} for e in order})), flush=True)
    ratio = {m: [round(r["loop"][m] / r["auto"][m], 1) for r in rows if r["auto"][m] > 0] for m in ("seconds", "gcycles")}
    calls = [r[e] for r in rows for e in ("auto", "loop")] + first
    dec = {e: sorted({(c["stage1"], c["stage2"]) for c in calls if c["engine"] == e}) for e in ("auto", "loop")}
    summary = dict(
        ftype=ftype, tested=first[0]["tested"], neuron=int(d["neuron"]), feature_index=int(d["feature_index"]),
        frames=int(d["calcium"].shape[1]), rounds=rounds,
        median={e: {m: statistics.median(r[e][m] for r in rows) for m in ("seconds", "gcycles")} for e in ("auto", "loop")},
        ratio_per_round=ratio,
        ratio_median={m: statistics.median(v) for m, v in ratio.items()},
        ratio_min={m: min(v) for m, v in ratio.items()}, ratio_max={m: max(v) for m, v in ratio.items()},
        first_call={c["engine"]: {m: c[m] for m in ("seconds", "gcycles")} for c in first},
        decisions={e: [list(x) for x in v] for e, v in dec.items()},
        same_decision=dec["auto"] == dec["loop"] and len(dec["auto"]) == 1,
        pvals={e: sorted(c["pval"] for c in calls if c["engine"] == e) for e in ("auto", "loop")},
        opt_delays={e: sorted({c["opt_delay"] for c in calls if c["engine"] == e}) for e in ("auto", "loop")},
        fft_type_counts={c["engine"]: c["fft_type_counts"] for c in first},
        machine_cpu_load_pct=dict(min=min(c["machine_cpu_load_pct"] for c in calls),
                                  max=max(c["machine_cpu_load_pct"] for c in calls)),
        cpu_count=psutil.cpu_count())
    T.write_json(T.OUT / f"fft_interleaved_{ftype}.json", dict(summary=summary, first=first, rounds=rows))
    print("SUMMARY", json.dumps({k: summary[k] for k in ("ftype", "median", "ratio_median", "ratio_min", "ratio_max",
                                                         "same_decision", "machine_cpu_load_pct")}), flush=True)


if __name__ == "__main__":
    main()
