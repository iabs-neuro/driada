"""INTENSE on the NOF sessions with disentanglement restricted to accepted pairs.

The run of the paper: the steps of tools/run_intense_analysis.py of the DRIADA
release of config.toml (GCMI, type-based representation, place_field zone rule,
seed 42), with one difference in the order of the steps. A neuron enters
disentanglement only with the pairs that pass the whole acceptance criterion of
the paper (p < pval_thr, excess MI > mi_excess_thr, activity ratio > 1); pairs
that pass the significance test but carry less excess information are left out
before the priority rules, the zone rule and the conditional-MI test. Without
this, a pair below the information threshold could take part in a decision and
remove an accepted label of the neuron.

The restriction is passed to the library as the first part of the pre-filter of
disentangle_all_selectivities, the hook the release provides for such filters;
the library and the tool are not modified.

Results go to [results] intense_root (or --output-dir), <intense_root>/<representation>/:
results, tables, tables_disentangled, disentanglement, where compute/build_db.py and the
figure scripts read them. Interrupted runs continue where they stopped. About 2-8 hours
for 64 sessions.

usage: python compute/run_intense_nof_threshold_first.py [--output-dir DIR] [session ...]
       session: e.g. NOF_H01_2D; all sessions of [data] nof_aligned by default
"""
import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

driada = P.use_driada()
print("driada", driada.__version__, driada.__file__, flush=True)

from selectivity_dynamics import (  # noqa: E402
    DEFAULT_CONFIG,
    extract_filter_data,
    find_zone_mask,
    get_exp_name,
    get_experiment_config,
    get_filter_for_experiment,
    load_experiment_from_npz,
    run_intense_analysis,
    save_all_results,
)


def accepted(stats):
    """Acceptance criterion of a neuron-feature pair, from its INTENSE statistics."""
    if not stats:
        return False
    pval, me_excess, ratio = stats.get("pval"), stats.get("me_excess"), stats.get("signal_ratio")
    return (pval is not None and pval < P.PVAL_THR
            and me_excess is not None and me_excess > P.MI_EXCESS_THR
            and (ratio is None or ratio > 1))


def accepted_pairs_first(filter_func):
    """Pre-filter that keeps only accepted pairs, then applies ``filter_func``."""
    def pre_filter(neuron_selectivities, pair_decisions, renames, cell_feat_stats=None, **kwargs):
        left_out = 0
        for nid, sels in neuron_selectivities.items():
            keep = [f for f in sels if accepted(cell_feat_stats[nid].get(f))]
            left_out += len(sels) - len(keep)
            sels[:] = keep
        print(f"  {left_out} significant pairs below the acceptance criterion left out of "
              "disentanglement", flush=True)
        if filter_func is not None:
            filter_func(neuron_selectivities=neuron_selectivities, pair_decisions=pair_decisions,
                        renames=renames, cell_feat_stats=cell_feat_stats, **kwargs)
    return pre_filter


def process_session(npz_path, config, output_dir):
    """One session as process_single_experiment of the tool, with the pre-filter above."""
    exp_name = get_exp_name(npz_path)
    exp_type = exp_name.split("_")[0]
    exp_config = get_experiment_config(exp_type)
    exp = load_experiment_from_npz(npz_path, agg_features=exp_config["aggregate_features"],
                                   feature_types=exp_config.get("feature_types"), seed=config["seed"])
    filter_kwargs = {
        "mi_ratio_threshold": 1.5,
        "place_feat_name": exp_config["place_feat_name"],
        "discrete_place_features": exp_config["discrete_place_features"],
        "feature_renaming": exp_config["feature_renaming"],
        "correspondence_threshold": 0.4,
    }
    if exp_config["discrete_place_features"]:
        mask = find_zone_mask(P.ZONE_MASKS, exp_name) if P.ZONE_MASKS else None
        filter_kwargs.update(extract_filter_data(
            exp, discrete_place_features=exp_config["discrete_place_features"],
            place_feat_name=exp_config["place_feat_name"], zone_mask_path=mask))
    pre_filter = accepted_pairs_first(get_filter_for_experiment(exp_type, zone_rule=P.ZONE_RULE))
    stats, significance, info, results, disent_results, _ = run_intense_analysis(
        exp, config, exp_config["skip_for_intense"], pre_filter_func=pre_filter,
        post_filter_func=exp_config.get("post_filter"), filter_kwargs=filter_kwargs)
    save_all_results(exp_name, exp, stats, significance, info, results, disent_results, output_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.output_dir is None:
        args.output_dir = P.INTENSE_ROOT

    driada.set_parallel_backend("loky")
    # the settings of the command line of tools/run_intense_analysis.py used for the paper
    config = {
        "n_shuffles_stage1": DEFAULT_CONFIG["n_shuffles_stage1"],
        "n_shuffles_stage2": DEFAULT_CONFIG["n_shuffles_stage2"],
        "pval_thr": DEFAULT_CONFIG["pval_thr"],
        "multicomp_correction": None,
        "ds": DEFAULT_CONFIG["ds"],
        "engine": "auto",
        "metric": "mi",
        "with_disentanglement": True,
        "remove_anti_selective": True,
        "representation": P.REPRESENTATION,
        "seed": P.SEED,
    }
    if config["pval_thr"] != P.PVAL_THR:
        raise RuntimeError(f"tool p-value threshold {config['pval_thr']} differs from config.toml {P.PVAL_THR}")
    out = args.output_dir / P.REPRESENTATION if P.REPRESENTATION == "by_type" else args.output_dir
    names = args.sessions or sorted(p.name[:-len("_aligned.npz")] for p in P.NOF_ALIGNED.glob("NOF_*_aligned.npz"))
    for i, name in enumerate(names, 1):
        if (out / "results" / f"{name}_results.npz").exists():
            print(f"[{i}/{len(names)}] SKIPPED (exists): {name}", flush=True)
            continue
        print(f"[{i}/{len(names)}] {name}", flush=True)
        process_session(P.NOF_ALIGNED / f"{name}_aligned.npz", config, out)
        gc.collect()
    print("[OK]", len(names), "sessions in", out)


if __name__ == "__main__":
    main()
