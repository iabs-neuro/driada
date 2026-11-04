"""Wall time of the full INTENSE procedure for one neuron-feature pair, with and without the FFT cache
(Methods, FFT-accelerated permutation testing): the chosen pairs, and one process that times the
procedure three times. The ratio of the paper comes from fft/d_fft_core.py, which uses this file.

The pair comes from the synthetic benchmark of Fig. 4 (synthetic/run_benchmark.py: its data
generator and settings, DRIADA 1.2.2), realization 0, SNR 8, no skipped events. For each feature
type (continuous fBm, discrete event) it is the first ground-truth pair, in neuron and then feature
order, that the saved benchmark run detected (<[v11] saved_synthetic>/full/raw/<type>/
snr8_skip0.0_r0.npz, intense__stage2), so that the procedure runs through both stages. The procedure
is the one the benchmark runs on every pair: delay search within +-2 s, two stages of 100 / 10,000
shuffles, p < 0.001 without correction, ds = 5, type-based representation, seed 42. "With the cache"
is engine='auto', the library default (the profile records the type of cache built); "without" is
engine='loop', which computes MI anew for every shift. One process, one thread (BLAS and OpenMP
limited to one thread, enable_parallelization=False).

A fresh process runs the procedure three times, each time on a newly built experiment (the FFT
cache is built inside the call and not kept between calls). The first call includes the one-time
costs of the process, such as compiling the numerical kernels on first use; the later calls are
the cost of one pair inside a running analysis. CPU time of the process is recorded next to wall time: on a shared
machine their ratio shows how much of the wall time the process actually ran.

usage: python fft/d_fft_timing2.py prepare             -> <[fft] out_dir>/pair2_c.npz, pair2_d.npz
       python fft/d_fft_timing2.py run auto|loop c|d   (prints one JSON line per call)
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import intense_paths as P  # noqa: E402
import paths_v11 as V  # noqa: E402

driada = P.use_driada(tools=False)
print(f"driada {driada.__version__} from {Path(driada.__file__).resolve()}", flush=True)

import numpy as np  # noqa: E402

OUT = P.get("fft", "out_dir")
SAVED = V.get("saved_synthetic") / "full" / "raw"
SNR, SKIP, REAL = 8, 0.0, 0
CALLS = 3


def bench():
    sys.path.insert(0, str(PAPER / "synthetic"))
    import run_benchmark as rb  # checks the DRIADA release itself
    return rb


def write_json(path, obj):
    """Write ``obj`` as indented UTF-8 JSON, creating the parent folder.

    Parameters
    ----------
    path : str or pathlib.Path
        File to write.
    obj : object
        JSON-serialisable object; NumPy scalars are written as Python numbers and any
        other unknown object as its string.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False,
                               default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                    encoding="utf-8")


def prepare():
    rb = bench()
    OUT.mkdir(parents=True, exist_ok=True)
    for ftype in ("c", "d"):
        saved = np.load(SAVED / ftype / f"snr{SNR}_skip{SKIP}_r{REAL}.npz")
        feats, calcium, gt, data_seed = rb.make_data(ftype, SNR, SKIP, REAL)
        if not np.array_equal(gt, saved["gt"]):
            raise SystemExit(f"[FAILED] ground truth of {ftype} differs from the saved run")
        i, j = map(int, np.argwhere(saved["gt"] & saved["intense__stage2"])[0])
        np.savez(OUT / f"pair2_{ftype}.npz", calcium=calcium[i:i + 1].astype(np.float64),
                 feature=np.asarray(feats[j]), neuron=i, feature_index=j, data_seed=data_seed,
                 name=rb.feature_names(ftype)[j], saved_pval=float(saved["intense__pval"][i, j]),
                 saved_delay=int(saved["intense__delay"][i, j]))
        print(json.dumps(dict(ftype=ftype, neuron=i, feature=rb.feature_names(ftype)[j],
                              frames=int(calcium.shape[1]), data_seed=int(data_seed),
                              saved_pval=float(saved["intense__pval"][i, j]),
                              saved_delay=int(saved["intense__delay"][i, j]))), flush=True)


def one_call(rb, d, engine, ftype):
    from driada.experiment.exp_base import Experiment
    from driada.information.info_base import TimeSeries
    name = str(d["name"])
    exp = Experiment(signature=f"fig4_synthetic_{ftype}", calcium=d["calcium"], spikes=None,
                     static_features={"fps": rb.FPS, "t_rise_sec": rb.T_RISE, "t_off_sec": rb.DECAY_TIME},
                     dynamic_features={name: TimeSeries(data=d["feature"], discrete=(ftype == "d"), name=name)},
                     exp_identificators={"type": "fig4_benchmark"}, verbose=False, optimize_kinetics=False,
                     reconstruct_spikes=None)
    settings = dict(rb.PAPER, engine=engine)
    t0, c0 = time.perf_counter(), time.process_time()
    stats, sig, info, _, _ = driada.compute_cell_feat_significance(
        exp, metric="mi", metric_distr_type=rb.DISTR["mi"], find_optimal_delays=True,
        with_disentanglement=False, remove_anti_selective=False, use_precomputed_stats=False,
        save_computed_stats=False, enable_parallelization=False, verbose=False, profile=True, **settings)
    seconds, cpu = time.perf_counter() - t0, time.process_time() - c0
    cell = next(iter(stats))
    feat = next(iter(stats[cell]))
    return seconds, cpu, feat, stats[cell][feat], sig[cell][feat], info.get("timings", {})


def run(engine, ftype):
    rb = bench()
    d = np.load(OUT / f"pair2_{ftype}.npz")
    for call in range(1, CALLS + 1):
        seconds, cpu, feat, st, sg, timings = one_call(rb, d, engine, ftype)
        print(json.dumps(dict(engine=engine, ftype=ftype, call=call, tested=str(feat), neuron=int(d["neuron"]),
                              feature_index=int(d["feature_index"]), frames=int(d["calcium"].shape[1]),
                              seconds=round(seconds, 4), cpu_seconds=round(cpu, 4),
                              stage1=bool(sg.get("stage1", False)), stage2=bool(sg.get("stage2", False)),
                              pval=st.get("pval"), me=st.get("me"), me_excess=st.get("me_excess"),
                              opt_delay=st.get("opt_delay"),
                              fft_type_counts={str(k): v for k, v in (timings.get("fft_type_counts") or {}).items()},
                              timings={k: round(v, 4) for k, v in timings.items() if isinstance(v, (int, float))},
                              driada=driada.__version__), default=float), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        run(sys.argv[2], sys.argv[3])
