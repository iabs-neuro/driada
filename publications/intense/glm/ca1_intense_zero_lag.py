"""INTENSE on the 64 CA1 sessions with the delay search switched off (Section S1.4, Supplementary Table 4).

The paper run (compute/run_intense_nof_threshold_first.py) calls run_intense_analysis of the tool
tools/selectivity_dynamics, which passes find_optimal_delays=True to the library with no setting to
change it. This script calls the same function on the same experiments, with the settings of the
paper run (DRIADA release, data with the frame rate of each session, type-based representations,
seed, 100 / 10,000 shuffles, p < pval_thr without correction, ds = 5, removal of anti-selective
pairs; all from config.toml and the tool's defaults), and for the duration of the run replaces
driada.compute_cell_feat_significance by a wrapper that sets find_optimal_delays=False (--delays off)
or leaves it as it is (--delays on); every other argument is the tool's. Disentanglement is not run:
Supplementary Table 4 counts pairs before it, and the neuron-feature statistics do not depend on it.
Results are saved with the function the paper run used (driada.intense.io.save_results), one
<session>_results.npz per session; glm/ca1_numbers.py counts them.

--delays on repeats the second stage of the paper run; its statistics then equal those of
[v11] stage2_results, which shows that the wrapper changes nothing but the delay search.

usage: python glm/ca1_intense_zero_lag.py --delays off|on [--shard k/N] [--n-jobs J] [--out DIR] [session ...]
       default DIR: <[glm] ca1_intense_dir>/delays_<off|on>, results in DIR/results; finished sessions
       are skipped, so an interrupted part is restarted with the same command
"""
import argparse
import gc
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import intense_paths as P  # noqa: E402
import ca1_common as C  # noqa: E402

driada = P.use_driada()

from selectivity_dynamics import (  # noqa: E402
    DEFAULT_CONFIG,
    get_experiment_config,
    load_experiment_from_npz,
    run_intense_analysis,
)
from driada.intense.io import save_results as save_intense_results  # noqa: E402

_LIBRARY_CALL = driada.compute_cell_feat_significance
DELAY_SEARCH = {"on": True, "off": False}


def wrap(delay_search, n_jobs):
    def compute_cell_feat_significance(*args, **kwargs):
        if kwargs.get("find_optimal_delays") is not True:
            raise RuntimeError("the tool no longer passes find_optimal_delays=True; check the wrapper")
        kwargs["find_optimal_delays"] = delay_search
        if n_jobs is not None:
            # number of worker processes only: the shuffles are drawn in advance for the whole grid
            # of pairs, so the result does not depend on it (checked with --delays on)
            kwargs["n_jobs"] = n_jobs
        return _LIBRARY_CALL(*args, **kwargs)
    return compute_cell_feat_significance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--delays", choices=sorted(DELAY_SEARCH), required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    a = ap.parse_args()
    out = a.out or (P.get("glm", "ca1_intense_dir") / f"delays_{a.delays}")
    res_dir = out / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    driada.compute_cell_feat_significance = wrap(DELAY_SEARCH[a.delays], a.n_jobs)
    driada.set_parallel_backend("loky")

    config = {
        "n_shuffles_stage1": DEFAULT_CONFIG["n_shuffles_stage1"],
        "n_shuffles_stage2": DEFAULT_CONFIG["n_shuffles_stage2"],
        "pval_thr": DEFAULT_CONFIG["pval_thr"],
        "multicomp_correction": None,
        "ds": DEFAULT_CONFIG["ds"],
        "engine": "auto",
        "metric": "mi",
        "with_disentanglement": False,
        "remove_anti_selective": True,
        "representation": P.REPRESENTATION,
        "seed": P.SEED,
    }
    assert config["pval_thr"] == P.PVAL_THR and P.REPRESENTATION == "by_type"
    k, n = (int(v) for v in a.shard.split("/"))
    names = a.sessions or sorted(p.name[:-len("_aligned.npz")] for p in P.NOF_ALIGNED.glob("NOF_*_aligned.npz"))
    names = names[k::n] if not a.sessions else names
    exp_config = get_experiment_config("NOF")
    print(f"delay search {a.delays}; part {k}/{n}: {len(names)} sessions -> {res_dir}", flush=True)
    for i, name in enumerate(names, 1):
        target = res_dir / f"{name}_results.npz"
        if target.exists():
            print(f"[{i}/{len(names)}] SKIPPED (exists): {name}", flush=True)
            continue
        t0 = time.time()
        exp = load_experiment_from_npz(P.NOF_ALIGNED / f"{name}_aligned.npz",
                                       agg_features=exp_config["aggregate_features"],
                                       feature_types=exp_config.get("feature_types"),
                                       seed=config["seed"], verbose=False)
        fps = float(exp.fps)
        if abs(fps - C.session_fps(name)) > 1e-9:
            raise SystemExit(f"[FAILED] {name}: exp.fps {fps} differs from _metadata")
        stats, significance, info, results, _, _ = run_intense_analysis(
            exp, config, exp_config["skip_for_intense"])
        tmp = res_dir / f"{name}_results.tmp.npz"
        save_intense_results(results, tmp)
        tmp.replace(target)
        delays = info["optimal_delays"]
        n_sig = sum(bool(s.get("stage2")) for c in significance.values() for s in c.values())
        print(f"[{i}/{len(names)}] {name}: {exp.n_cells} neurons at {fps:.3f} frames/s, "
              f"{n_sig} significant pairs, nonzero delays {int((delays != 0).sum())}, "
              f"{time.time() - t0:.0f} s", flush=True)
        del exp, stats, significance, info, results
        gc.collect()
    print(f"[OK] part {k}/{n} finished", flush=True)


if __name__ == "__main__":
    main()
