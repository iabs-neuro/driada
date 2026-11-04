"""The INTENSE pipeline of the paper on the synthetic sessions, disentanglement in two orders.

The pipeline is process_single_experiment of tools/run_intense_analysis.py of DRIADA
1.2.2 with the settings of the paper (by_type, place_field zone rule with the zone
masks of the session, seed 42, p < 0.001 per pair, delay search). Disentanglement
runs twice on the same INTENSE result:

* base: as in the paper, every pair that passed INTENSE enters disentanglement and
  the 0.04-bit threshold is applied afterwards;
* thrfirst: only pairs that also pass the 0.04-bit threshold enter it.

The filter chain of the paper (priority rules, NOF object rule, place_field zone
rule) is rebuilt from the same functions with a recorder around each step, so that
every decision is attributed to the step that made it.

Output: results/<session>.pkl with the INTENSE statistics, the significant pairs,
the per-neuron disentanglement of both orders, the step of every filter decision
and the statistics of the final labels (build_disentangled_stats).

usage: python run_synth.py [session ...]   (sessions with an existing pkl are skipped)
"""
import os
import pickle
import sys
import time
from pathlib import Path

# Limit BLAS threads as run_intense_analysis.py does; numpy is imported below.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

driada = P.use_driada()
print("driada", driada.__version__, driada.__file__, flush=True)

import driada.intense.pipelines as pipelines  # noqa: E402
import run_intense_analysis as ria  # noqa: E402
from driada.intense.representations import get_representation_sources  # noqa: E402
from selectivity_dynamics import filters as F  # noqa: E402
from selectivity_dynamics.analysis import build_disentangled_stats  # noqa: E402

PVAL_THR, MI_THR = P.PVAL_THR, P.MI_EXCESS_THR
OUT = HERE / "results"
_orig_disentangle = pipelines.disentangle_all_selectivities
STATE = {"order": None, "tags": {}, "dropped": {}}
STAT_KEYS = ("me", "me_excess", "pval", "opt_delay", "signal_ratio", "merged_from", "component_stats")


def passes(s):
    """Paper criterion of a pair from INTENSE statistics."""
    if not s:
        return False
    sr, me_excess, pval = s.get("signal_ratio"), s.get("me_excess"), s.get("pval")
    return (pval is not None and pval < PVAL_THR and me_excess is not None
            and me_excess > MI_THR and (sr is None or sr > 1))


def recorded(step, func):
    """Run a filter step and note, in source names, which decisions and merges it made."""
    def g(neuron_selectivities, pair_decisions, renames, **kwargs):
        before = {nid: dict(d) for nid, d in pair_decisions.items()}
        sels_before = {nid: list(s) for nid, s in neuron_selectivities.items()}
        ren_before = {nid: dict(r) for nid, r in renames.items()}
        func(neuron_selectivities, pair_decisions, renames, **kwargs)
        tags = STATE["tags"].setdefault(STATE["order"], {})
        for nid, dec in pair_decisions.items():
            t = tags.setdefault(nid, {"pairs": {}, "merges": {}, "removed": {}})
            for k, v in dec.items():
                if before[nid].get(k, None) != v:
                    t["pairs"][k] = (step, v)
            for new, old in renames[nid].items():
                if new not in ren_before[nid]:
                    t["merges"][new] = (step, tuple(old))
            merged_parts = {f for old in renames[nid].values() for f in old}
            for f in set(sels_before[nid]) - set(neuron_selectivities[nid]) - merged_parts:
                t["removed"][f] = step
    return g


def paper_filter(exp_type, zone_rule=F.DEFAULT_ZONE_RULE):
    """get_filter_for_experiment of the release, each step wrapped by the recorder."""
    assert exp_type == "NOF" and zone_rule == P.ZONE_RULE
    cfg = F.EXPERIMENT_CONFIGS[exp_type]
    return F.compose_filters(recorded("rule", F.build_priority_filter(F.GENERAL_PRIORITY_RULES)),
                             recorded("object_rule", cfg["specific_filter"]),
                             recorded("zone", F.ZONE_RULES[zone_rule]))


def threshold_first(pre_filter):
    def filt(neuron_selectivities, pair_decisions, renames, cell_feat_stats=None, **kwargs):
        dropped = {}
        for nid, sels in neuron_selectivities.items():
            keep = [f for f in sels if passes(cell_feat_stats[nid].get(f))]
            if len(keep) < len(sels):
                dropped[nid] = [f for f in sels if f not in keep]
            sels[:] = keep
        STATE["dropped"] = dropped
        if pre_filter is not None:
            pre_filter(neuron_selectivities=neuron_selectivities, pair_decisions=pair_decisions,
                       renames=renames, cell_feat_stats=cell_feat_stats, **kwargs)
    return filt


def disentangle_two_orders(*args, **kwargs):
    STATE["tags"] = {}
    STATE["order"] = "base"
    STATE["base"] = _orig_disentangle(*args, **kwargs)
    STATE["order"] = "thrfirst"
    kwargs = dict(kwargs)
    kwargs["pre_filter_func"] = threshold_first(kwargs.get("pre_filter_func"))
    return _orig_disentangle(*args, **kwargs)


def slim(stats):
    return {nid: {f: {k: s[k] for k in STAT_KEYS if k in s} for f, s in feats.items()}
            for nid, feats in stats.items()}


def save_compact(exp_name, exp, stats, significance, info, results, disent_results, output_dir):
    base = STATE.pop("base")
    orders = {"base": base, "thrfirst": disent_results}
    out = {
        "session": exp_name,
        "sources": dict(get_representation_sources(exp)),
        "feature_names": list(disent_results["feature_names"]),
        "feat_feat_significance": disent_results["feat_feat_significance"],
        "stats": slim(stats),
        "significant": {nid: list(f) for nid, f in exp.get_significant_neurons().items()},
        "thr_dropped": STATE["dropped"],
        "tags": STATE["tags"],
        "orders": {},
    }
    for name, res in orders.items():
        d_stats, _ = build_disentangled_stats(stats, significance, res, exp)
        out["orders"][name] = {"per_neuron_disent": res["per_neuron_disent"],
                               "final_stats": slim(d_stats)}
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{exp_name}.pkl", "wb") as f:
        pickle.dump(out, f)


pipelines.disentangle_all_selectivities = disentangle_two_orders
ria.get_filter_for_experiment = paper_filter
ria.save_all_results = save_compact

CONFIG = {
    "n_shuffles_stage1": ria.DEFAULT_CONFIG["n_shuffles_stage1"],
    "n_shuffles_stage2": ria.DEFAULT_CONFIG["n_shuffles_stage2"],
    "pval_thr": ria.DEFAULT_CONFIG["pval_thr"],
    "multicomp_correction": None,
    "ds": ria.DEFAULT_CONFIG["ds"],
    "engine": "auto",
    "metric": "mi",
    "with_disentanglement": True,
    "remove_anti_selective": True,
    "representation": P.REPRESENTATION,
    "seed": P.SEED,
}

if __name__ == "__main__":
    assert CONFIG["pval_thr"] == PVAL_THR
    driada.set_parallel_backend("loky")
    for name in sys.argv[1:]:
        if (OUT / f"{name}.pkl").exists():
            print(f"{name}: done, skipped", flush=True)
            continue
        t0 = time.time()
        ria.process_single_experiment(HERE / "synth_npz" / f"{name}_aligned.npz", CONFIG,
                                      output_dir=OUT, plot=False, use_filters=True,
                                      zone_rule=P.ZONE_RULE, zone_masks_dir=P.ZONE_MASKS)
        print(f"[TIME] {name}: {time.time() - t0:.0f} s", flush=True)
