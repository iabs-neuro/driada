"""The paper run, with the interaction information of every pair tested by conditional MI saved.

compute/run_intense_nof_threshold_first.py is imported and its main() run
unchanged; for the duration of the run four names of DRIADA 1.2.2 are wrapped
in this process, and the library is not modified:

* disentanglement.get_mi, conditional_mi and cmi_ggg record the values they
  return when called from the pair test;
* disentanglement._disentangle_pair_with_precomputed calls the original and
  records, for the pair it decided, MI(neuron; X), MI(neuron; Y), MI(X; Y),
  MI(neuron; X | Y), MI(neuron; Y | X) and the interaction information
  I_av = mean(MI(n;X|Y) - MI(n;X), MI(n;Y|X) - MI(n;Y)) exactly as the library
  computes it, together with the decision the original returned;
* disentanglement._process_neuron_disentanglement names the neuron;
* pipelines.disentangle_all_selectivities runs with n_jobs=1, so that the
  neurons are processed in this process and the wrappers see them (worker
  processes would import the unwrapped module). The decisions do not depend
  on n_jobs.

Only the pairs that reach the conditional-MI test get a row: pairs decided by
the priority rules or the zone rule, and pairs of behavioural variables that
are not related (kept without the test), compute no interaction information.
Rows go to <output>/by_type/interaction/<session>_interaction.csv. The
per-neuron decisions of disentanglement (which step decided each pair) are
pickled to <output>/by_type/per_neuron/<session>_per_neuron.pkl; the paper
script itself does not save them.
Everything else is written by the paper script, as without the wrapper.

Because the neurons are disentangled one after another, the run is slower than
the paper script alone; --shard k/N takes every N-th session from the k-th
(k = 0..N-1, sessions of [data] nof_aligned sorted by name), so that N processes
can run side by side on disjoint sessions into the same output folder.

usage: python compute/run_intense_interaction.py [--output-dir DIR] [--shard k/N | session ...]
"""
import csv
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
SCRIPT = PAPER / "compute" / "run_intense_nof_threshold_first.py"

_spec = importlib.util.spec_from_file_location("paper_run_tf", SCRIPT)
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)          # imports DRIADA of the configuration; main() is not run

import driada.intense.disentanglement as dis  # noqa: E402
import driada.intense.pipelines as pipelines  # noqa: E402

FIELDS = ["neuron", "feat_x", "feat_y", "mi_nx", "mi_ny", "mi_xy", "cmi_nx_given_y",
          "cmi_ny_given_x", "ii_1", "ii_2", "ii_av", "decision", "delay_x_ds", "delay_y_ds", "ds"]
CAPTURE = []
ROWS = []
CTX = {}

_orig = {name: getattr(dis, name) for name in
         ("get_mi", "conditional_mi", "cmi_ggg", "_disentangle_pair_with_precomputed",
          "_process_neuron_disentanglement")}
_orig_all = pipelines.disentangle_all_selectivities
_orig_process_session = T.process_session
_orig_save = T.save_all_results


def save_with_per_neuron(exp_name, exp, stats, significance, info, results, disent_results, output_dir):
    _orig_save(exp_name, exp, stats, significance, info, results, disent_results, output_dir)
    folder = Path(output_dir) / "per_neuron"
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{exp_name}_per_neuron.pkl", "wb") as f:
        pickle.dump({"per_neuron_disent": disent_results["per_neuron_disent"],
                     "feature_names": list(disent_results["feature_names"])}, f)


def _recording(kind, func):
    def wrapped(*args, **kwargs):
        value = func(*args, **kwargs)
        CAPTURE.append((kind, float(value)))
        return value
    return wrapped


def pair_recorded(ts1, ts2, ts3, mi12=None, mi13=None, mi23=None, **kwargs):
    CAPTURE.clear()
    decision = _orig["_disentangle_pair_with_precomputed"](
        ts1, ts2, ts3, mi12=mi12, mi13=mi13, mi23=mi23, **kwargs)
    mis = iter(v for k, v in CAPTURE if k == "mi")
    cmis = [v for k, v in CAPTURE if k == "cmi"]
    m12 = mi12 if mi12 is not None else next(mis)
    m13 = mi13 if mi13 is not None else next(mis)
    m23 = mi23 if mi23 is not None else next(mis)
    if len(cmis) != 2:
        raise RuntimeError(f"expected two conditional MI values, got {cmis}")
    c123, c132 = cmis
    ii1, ii2 = c123 - m12, c132 - m13
    ROWS.append({"neuron": CTX.get("neuron"), "feat_x": ts2.name, "feat_y": ts3.name,
                 "mi_nx": m12, "mi_ny": m13, "mi_xy": m23, "cmi_nx_given_y": c123,
                 "cmi_ny_given_x": c132, "ii_1": ii1, "ii_2": ii2, "ii_av": float(np.mean([ii1, ii2])),
                 "decision": decision, "delay_x_ds": kwargs.get("delay_feat1", 0),
                 "delay_y_ds": kwargs.get("delay_feat2", 0), "ds": kwargs.get("ds", 1)})
    return decision


def neuron_named(*args, **kwargs):
    CTX["neuron"] = kwargs.get("neuron_id", args[0] if args else None)
    return _orig["_process_neuron_disentanglement"](*args, **kwargs)


def disentangle_in_process(*args, **kwargs):
    kwargs["n_jobs"] = 1
    return _orig_all(*args, **kwargs)


def process_session_saving(npz_path, config, output_dir):
    ROWS.clear()
    _orig_process_session(npz_path, config, output_dir)
    name = T.get_exp_name(npz_path)
    folder = Path(output_dir) / "interaction"
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"{name}_interaction.csv.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(ROWS)
    tmp.replace(folder / f"{name}_interaction.csv")
    print(f"  interaction information of {len(ROWS)} pairs tested by conditional MI -> "
          f"{folder / (name + '_interaction.csv')}", flush=True)


def install():
    dis.get_mi = _recording("mi", _orig["get_mi"])
    dis.conditional_mi = _recording("cmi", _orig["conditional_mi"])
    dis.cmi_ggg = _recording("cmi", _orig["cmi_ggg"])
    dis._disentangle_pair_with_precomputed = pair_recorded
    dis._process_neuron_disentanglement = neuron_named
    pipelines.disentangle_all_selectivities = disentangle_in_process
    T.process_session = process_session_saving
    T.save_all_results = save_with_per_neuron


def shard_sessions(argv):
    """Replace --shard k/N by the session names of that shard."""
    if "--shard" not in argv:
        return argv
    i = argv.index("--shard")
    k, n = map(int, argv[i + 1].split("/"))
    names = sorted(p.name[:-len("_aligned.npz")] for p in T.P.NOF_ALIGNED.glob("NOF_*_aligned.npz"))
    return argv[:i] + argv[i + 2:] + names[k::n]


if __name__ == "__main__":
    install()
    sys.argv = [str(SCRIPT), *shard_sessions(sys.argv[1:])]
    T.main()
