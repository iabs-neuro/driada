"""FFT-cache speed-up for one neuron-feature pair at two levels, engines interleaved call by call in one
process: the "130 to 220 times faster" of Methods (FFT-accelerated permutation testing) is
summary.ratio_median.core_gcycles of fft_core_c.json (continuous feature) and fft_core_d.json
(discrete feature), the median over the rounds of the ratio of CPU cycles without and with the cache.

As d_fft_interleaved.py (same pair, procedure, settings, interleaving and cycle counting), and in
addition the significance procedure itself is timed apart from the preparation around it:
compute_cell_feat_significance prepares the data (representations, copula transform, downsampling,
the experiment's tables) and then calls compute_me_stats, which builds the FFT cache, searches the
delay and runs both stages of shuffles with their null distributions and p-values. For the duration
of the run the name compute_me_stats in driada.intense.pipelines is wrapped by a function that
measures wall time and CPU cycles of the process around the library's own function and passes its
arguments and results through unchanged.

Run after fft/d_fft_timing2.py prepare; Windows only, as d_fft_interleaved.py.
usage: python fft/d_fft_core.py c|d [ROUNDS]  ->  <[fft] out_dir>/fft_core_<type>.json  (ROUNDS: 5)
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d_fft_interleaved as I  # noqa: E402  (sets one thread, loads DRIADA 1.2.2)
import d_fft_timing2 as T  # noqa: E402

import numpy as np  # noqa: E402
import driada.intense.pipelines as PL  # noqa: E402

_CORE = PL.compute_me_stats
_LAST = {}


def _timed_core(*args, **kwargs):
    t0, c0 = time.perf_counter(), I.cycles()
    out = _CORE(*args, **kwargs)
    _LAST.update(seconds=time.perf_counter() - t0, gcycles=(I.cycles() - c0) / 1e9, calls=_LAST.get("calls", 0) + 1)
    return out


PL.compute_me_stats = _timed_core


def call(rb, d, engine, ftype):
    _LAST.clear()
    r = I.call(rb, d, engine, ftype)
    if _LAST.get("calls") != 1:
        raise RuntimeError(f"compute_me_stats was called {_LAST.get('calls')} times in one procedure")
    r["core_seconds"], r["core_gcycles"] = round(_LAST["seconds"], 4), round(_LAST["gcycles"], 4)
    return r


def main():
    ftype = sys.argv[1]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    rb = T.bench()
    d = np.load(T.OUT / f"pair2_{ftype}.npz")
    first = [call(rb, d, e, ftype) for e in ("auto", "loop")]
    rows = []
    for k in range(rounds):
        order = ("auto", "loop") if k % 2 == 0 else ("loop", "auto")
        rec = {e: call(rb, d, e, ftype) for e in order}
        rows.append(dict(round=k, order=list(order), **rec))
        print(json.dumps(dict(round=k, **{e: {x: rec[e][x] for x in ("seconds", "gcycles", "core_seconds", "core_gcycles",
                                                                    "machine_cpu_load_pct", "stage2", "pval")}
                                          for e in order})), flush=True)
    metrics = ("seconds", "gcycles", "core_seconds", "core_gcycles")
    ratio = {m: [round(r["loop"][m] / r["auto"][m], 1) for r in rows if r["auto"][m] > 0] for m in metrics}
    calls = [r[e] for r in rows for e in ("auto", "loop")] + first
    dec = {e: sorted({(c["stage1"], c["stage2"]) for c in calls if c["engine"] == e}) for e in ("auto", "loop")}
    summary = dict(
        ftype=ftype, tested=first[0]["tested"], neuron=int(d["neuron"]), feature_index=int(d["feature_index"]),
        frames=int(d["calcium"].shape[1]), rounds=rounds,
        median={e: {m: statistics.median(r[e][m] for r in rows) for m in metrics} for e in ("auto", "loop")},
        ratio_per_round=ratio, ratio_median={m: statistics.median(v) for m, v in ratio.items()},
        ratio_min={m: min(v) for m, v in ratio.items()}, ratio_max={m: max(v) for m, v in ratio.items()},
        share_of_call_in_core={e: round(statistics.median(r[e]["core_gcycles"] / r[e]["gcycles"] for r in rows), 3)
                               for e in ("auto", "loop")},
        first_call={c["engine"]: {m: c[m] for m in metrics} for c in first},
        decisions={e: [list(x) for x in v] for e, v in dec.items()},
        same_decision=dec["auto"] == dec["loop"] and len(dec["auto"]) == 1,
        pvals={e: sorted(c["pval"] for c in calls if c["engine"] == e) for e in ("auto", "loop")},
        opt_delays={e: sorted({c["opt_delay"] for c in calls if c["engine"] == e}) for e in ("auto", "loop")},
        fft_type_counts={c["engine"]: c["fft_type_counts"] for c in first},
        machine_cpu_load_pct=dict(min=min(c["machine_cpu_load_pct"] for c in calls),
                                  max=max(c["machine_cpu_load_pct"] for c in calls)))
    T.write_json(T.OUT / f"fft_core_{ftype}.json", dict(summary=summary, first=first, rounds=rows))
    print("SUMMARY", json.dumps({k: summary[k] for k in ("ftype", "median", "ratio_median", "ratio_min", "ratio_max",
                                                         "share_of_call_in_core", "same_decision")}), flush=True)


if __name__ == "__main__":
    main()
