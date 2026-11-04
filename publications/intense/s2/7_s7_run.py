"""The pipeline of the paper on the synthetic sessions of s7/ (s2/7_s7_make.py): synthetic/run_synth.py
imported unchanged, with its folders pointed at <[s2] out_dir>/s7, exactly as s2/s2_synth_fps_16.py
runs the synthetic neurons of the S2 categories: DRIADA 1.2.2, by_type, place_field zone rule,
p < 0.001, delay search, disentanglement in the orders base and thrfirst (thrfirst is the paper
order). Sessions with an existing pkl are skipped; by default the 16 sessions of sessions_16.csv.
N_WORKERS (environment, optional) caps the worker processes of INTENSE as in s2/s2_synth_fps_16.py
(the time changes, not the result).
usage: python s2/7_s7_run.py [session ...]
"""
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
if os.environ.get("N_WORKERS"):
    import multiprocessing
    multiprocessing.cpu_count = lambda: int(os.environ["N_WORKERS"])
PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER / "synthetic"))
sys.path.insert(1, str(PAPER))
import run_synth as R  # noqa: E402
import intense_paths as P  # noqa: E402
import pandas as pd  # noqa: E402

OUT = P.get("s2", "out_dir") / "s7"
SESSIONS = list(pd.read_csv(P.get("s2", "out_dir") / "sessions_16.csv").session)

if __name__ == "__main__":
    R.HERE, R.OUT = OUT, OUT / "results"
    assert R.CONFIG["pval_thr"] == R.PVAL_THR
    R.driada.set_parallel_backend("loky")
    for name in sys.argv[1:] or SESSIONS:
        if (R.OUT / f"{name}.pkl").exists():
            print(f"[SKIP] {name}", flush=True)
            continue
        t0 = time.time()
        R.ria.process_single_experiment(OUT / "synth_npz" / f"{name}_aligned.npz", R.CONFIG,
                                        output_dir=R.OUT, plot=False, use_filters=True,
                                        zone_rule=R.P.ZONE_RULE, zone_masks_dir=R.P.ZONE_MASKS)
        print(f"[TIME] {name}: {time.time() - t0:.0f} s", flush=True)
