"""Disentanglement of the CA1 analysis with conditional MI at the optimal delays of the pair.

The numbers of the paper (868 multi-selective neuron-sessions after disentanglement, the
interaction-information paragraph, Fig. 3c) come from this wrapper with --cmi aligned
([v11] cmi_delays of config.toml); DRIADA 1.2.2 alone gives 872.

The paper run is run as it is: compute/run_intense_interaction.py with its wrappers
(interaction information of every pair tested by conditional MI, per-neuron decisions),
which runs compute/run_intense_nof_threshold_first.py on DRIADA 1.2.2. Neither the paper
script nor the library is modified. By default (as for the paper) the second stage is not
recomputed: for the duration of the run two library functions are replaced by what the
paper run saved for the session ([v11] stage2_results/<session>_results.npz). With --full
the second stage is computed and only the conditional MI is changed:

* pipelines.compute_me_stats, the shuffle test of the neuron-feature pairs, returns the
  saved statistics, significance and optimal delays (numpy scalar types restored);
* pipelines.compute_feat_feat_significance returns the saved MI and significance
  matrices of the behavioural variables.

Everything after them runs unchanged: removal of anti-selective pairs, the acceptance
pre-filter, priority rules, zone rule, the conditional-MI test, tables and per-neuron
decisions.

--cmi asis     conditional MI as in DRIADA 1.2.2.
--cmi aligned  in the branch of _disentangle_pair_with_precomputed that calls
               conditional_mi (a pair with a discrete variable), the two behavioural
               variables are taken rolled by the optimal delays of the pair, in the
               convention of the branch of two continuous variables and of the MI that
               INTENSE recorded at those delays: downsample by ds, then np.roll forward by
               the delay in downsampled samples along time. The neuron is not shifted.
               (get_mi(x, y, shift=d) follows it for a one-dimensional y only; for a
               MultiTimeSeries y it rolls x instead, i.e. y backwards - not the reference.)
               The library function conditional_mi computes the value on these series
               (with ds=1, the series being downsampled already).
--no-weak-link the decision function is a copy of _disentangle_pair_with_precomputed
               without the criterion MI(A;X) < |II| (step 4 of the brief); otherwise the
               library function decides.

Besides the outputs of the paper run (by_type/interaction, per_neuron, tables, ...), every
pair tested by conditional MI gets a row in <output>/by_type/pairs_exact/<session>.csv with
the exact values, the branch of conditional_mi, the delays applied and the branch of the
decision rule that decided it (recomputed from the same values and checked against the
decision of the run).

usage: python compute/run_disent_pair_delays.py [--cmi asis|aligned] [--full] [--no-weak-link]
                                               [--output-dir DIR] [--shard k/N | session ...]
       (defaults: --cmi from [v11] cmi_delays, --output-dir [results] intense_root)
"""
import argparse
import csv
import importlib.util
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault("INTENSE_PAPER_CONFIG", str(HERE.parent / "config.toml"))

_spec = importlib.util.spec_from_file_location("run_intense_interaction", HERE / "run_intense_interaction.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)          # imports the paper script and DRIADA of the configuration
T, dis, pipelines = R.T, R.dis, R.pipelines

sys.path.insert(0, str(HERE.parent))
import paths_v11 as V  # noqa: E402

SAVED = V.get("stage2_results") if V.V11.get("stage2_results") else None

from driada.information.gcmi import mi_gg, mi_model_gd  # noqa: E402
from driada.information.info_base import _downsample_time_axis  # noqa: E402
from driada.intense.io import load_results  # noqa: E402

CUR = {}          # session being replayed and its saved results
SHIFT = {}        # id(behavioural series) -> delay (downsampled samples) during one pair test
PAIR = {}         # how the conditional MI of the current pair was computed
EXACT = []        # rows of pairs_exact for the current session
OPTS = {}

EXACT_FIELDS = ["session", "neuron", "feat_x", "feat_y", "x_discrete", "y_discrete", "cmi_path",
                "delay_x_ds", "delay_y_ds", "shift_x_applied", "shift_y_applied", "ds",
                "mi_nx", "mi_ny", "cmi_nx_given_y", "cmi_ny_given_x", "ii_1", "ii_2", "ii_av",
                "decision", "branch",
                "chk_mi_nx_roll", "chk_mi_nx_opposite", "chk_mi_ny_roll", "chk_mi_ny_opposite",
                "chk_cmi_xy_shifted", "chk_cmi_yx_shifted", "chk_cmi_xy_unshifted", "chk_cmi_yx_unshifted"]


# ---------------------------------------------------------------- saved second stage
def _typed_stats(entry):
    """A saved stats entry with the numpy scalar types the run had (JSON keeps values only)."""
    out = dict(entry)
    for key in ("rval", "pval", "pre_rval", "pre_pval"):
        if out.get(key) is not None:
            out[key] = np.float64(out[key])
    if out.get("me") is not None:
        me = out["me"]
        # MI tables of the shuffle test are float32; a value not representable in float32
        # would mean another origin, and is kept as float64 (counted in CUR["me_not_f32"]).
        if float(np.float32(me)) == me:
            out["me"] = np.float32(me)
        else:
            out["me"] = np.float64(me)
            CUR["me_not_f32"] = CUR.get("me_not_f32", 0) + 1
    if out.get("opt_delay") is not None:
        out["opt_delay"] = np.int64(out["opt_delay"])
    if isinstance(out.get("data_hash"), list):
        out["data_hash"] = tuple(out["data_hash"])
    return out


def saved_me_stats(signals, feats, names1=None, names2=None, **kwargs):
    """compute_me_stats of the neuron-feature pairs, answered from the saved results."""
    res = CUR["saved"]
    st, sg = res.stats, res.significance
    if sorted(st) != sorted(str(int(c)) for c in names1):
        raise RuntimeError(f"{CUR['name']}: saved neurons differ from the run")
    delays = np.asarray(res.info["optimal_delays"])
    if delays.shape != (len(names1), len(names2)):
        raise RuntimeError(f"{CUR['name']}: optimal_delays {delays.shape} vs {len(names1)}x{len(names2)}")
    stats = {c: {f: _typed_stats(st[str(int(c))][f]) for f in names2} for c in names1}
    sig = {c: {f: {k: (np.bool_(v) if isinstance(v, bool) else v)
                   for k, v in sg[str(int(c))][f].items()} for f in names2} for c in names1}
    info = {"optimal_delays": delays, "timings": {"total": 0.0}}
    CUR["n_me_stats_calls"] = CUR.get("n_me_stats_calls", 0) + 1
    return stats, sig, info


def saved_feat_feat(exp, feat_bunch=None, **kwargs):
    """compute_feat_feat_significance, answered from the saved results."""
    d = CUR["saved"].disentanglement
    if [str(f) for f in feat_bunch] != list(d["feature_names"]):
        raise RuntimeError(f"{CUR['name']}: features {feat_bunch} vs saved {d['feature_names']}")
    CUR["n_feat_feat_calls"] = CUR.get("n_feat_feat_calls", 0) + 1
    return (np.array(d["feat_feat_similarity"], dtype=float), np.array(d["feat_feat_significance"], dtype=float),
            None, list(feat_bunch), {})


# ---------------------------------------------------------------- conditional MI at the delays
class _Shifted:
    """What conditional_mi reads of a series, downsampled and rolled by ``shift`` along time."""

    def __init__(self, ts, ds, shift):
        def prep(a):
            if a is None:
                return None
            a = _downsample_time_axis(a, ds)
            return np.roll(a, shift, axis=-1) if shift else a
        self.discrete = ts.discrete
        self.data = prep(ts.data)
        self.int_data = prep(getattr(ts, "int_data", None)) if ts.discrete else None
        self.copula_normal_data = None if ts.discrete else prep(ts.copula_normal_data)


def conditional_mi_at_delays(ts1, ts2, ts3, ds=1, k=5):
    """conditional_mi of DRIADA 1.2.2, the two behavioural variables at their delays."""
    path = "C" + ("D" if ts2.discrete else "C") + ("D" if ts3.discrete else "C")
    PAIR.setdefault("paths", []).append(path)
    s2, s3 = SHIFT.get(id(ts2)), SHIFT.get(id(ts3))
    if not OPTS["aligned"] or s2 is None or s3 is None:
        PAIR.setdefault("shifts", []).append((0, 0))
        value = R._orig["conditional_mi"](ts1, ts2, ts3, ds=ds, k=k)
    else:
        PAIR.setdefault("shifts", []).append((s2, s3))
        value = R._orig["conditional_mi"](_Shifted(ts1, ds, 0), _Shifted(ts2, ds, s2),
                                          _Shifted(ts3, ds, s3), ds=1, k=k)
    # the value as returned (its numpy type decides the precision of the rule's arithmetic)
    PAIR.setdefault("values", []).append(value)
    return value


def cmi_ggg_counted(*args, **kwargs):
    PAIR.setdefault("paths", []).append("CCC_copula")
    value = R._orig["cmi_ggg"](*args, **kwargs)
    PAIR.setdefault("values", []).append(value)
    return value


# ---------------------------------------------------------------- checks of the shift convention
def mi_rolled(neur, feat, ds, shift):
    """MI(neuron; feature rolled by ``shift`` after downsampling), estimators of get_mi (GCMI)."""
    nc = np.ascontiguousarray(_downsample_time_axis(neur.copula_normal_data, ds))
    if feat.discrete:
        y = np.ascontiguousarray(np.roll(_downsample_time_axis(feat.int_data, ds), shift))
        mi = mi_model_gd(nc, y, Ym=int(np.max(y) + 1), biascorrect=True, demeaned=True)
    else:
        fc = np.ascontiguousarray(np.roll(_downsample_time_axis(feat.copula_normal_data, ds), shift, axis=-1))
        mi = mi_gg(fc, nc, True, True)
    return max(float(mi), 0.0)


def convention_checks(ts1, ts2, ts3, ds, dx, dy):
    """Values that test the shift convention; computed aside, nothing of the run sees them."""
    state = np.random.get_state()        # leave the global generator as the run would have it
    try:
        out = {"chk_mi_nx_roll": mi_rolled(ts1, ts2, ds, dx), "chk_mi_nx_opposite": mi_rolled(ts1, ts2, ds, -dx),
               "chk_mi_ny_roll": mi_rolled(ts1, ts3, ds, dy), "chk_mi_ny_opposite": mi_rolled(ts1, ts3, ds, -dy)}
        cmi = R._orig["conditional_mi"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["chk_cmi_xy_shifted"] = float(cmi(_Shifted(ts1, ds, 0), _Shifted(ts2, ds, dx), _Shifted(ts3, ds, dy), ds=1))
            out["chk_cmi_yx_shifted"] = float(cmi(_Shifted(ts1, ds, 0), _Shifted(ts3, ds, dy), _Shifted(ts2, ds, dx), ds=1))
            out["chk_cmi_xy_unshifted"] = float(cmi(_Shifted(ts1, ds, 0), _Shifted(ts2, ds, 0), _Shifted(ts3, ds, 0), ds=1))
            out["chk_cmi_yx_unshifted"] = float(cmi(_Shifted(ts1, ds, 0), _Shifted(ts3, ds, 0), _Shifted(ts2, ds, 0), ds=1))
    finally:
        np.random.set_state(state)
    return {k: repr(v) for k, v in out.items()}


# ---------------------------------------------------------------- decision rule
def decide(mi12, mi13, cmi123, cmi132, weak_link=True):
    """The rule of _disentangle_pair_with_precomputed (1.2.2), with the branch that decided."""
    I_av = np.mean([cmi123 - mi12, cmi132 - mi13])
    if I_av < 0:
        if weak_link:
            c1 = mi12 < np.abs(I_av) and not cmi132 < np.abs(I_av)
            c2 = mi13 < np.abs(I_av) and not cmi123 < np.abs(I_av)
            if c1 and not c2:
                return 1, "weak_link"
            if c2 and not c1:
                return 0, "weak_link"
        if mi12 > dis.MI_EPSILON and mi13 > dis.MI_EPSILON:
            r2, r3 = cmi123 / mi12, cmi132 / mi13
            if r2 < dis.REDUNDANCY_CMI_RATIO and r3 >= dis.REDUNDANCY_CMI_RATIO:
                return 1, "residual"
            if r3 < dis.REDUNDANCY_CMI_RATIO and r2 >= dis.REDUNDANCY_CMI_RATIO:
                return 0, "residual"
            if r2 < dis.REDUNDANCY_CMI_RATIO and r3 < dis.REDUNDANCY_CMI_RATIO:
                if mi12 > mi13:
                    return 0, "residual_both"
                if mi13 > mi12:
                    return 1, "residual_both"
        return 0.5, "redundancy_kept"
    if mi13 < dis.MI_EPSILON and cmi123 > cmi132:
        return 0, "synergy_negligible"
    if mi12 < dis.MI_EPSILON and cmi132 > cmi123:
        return 1, "synergy_negligible"
    if mi13 >= dis.MI_EPSILON and mi12 / mi13 > dis.DOMINANCE_RATIO_THRESHOLD and cmi123 > cmi132:
        return 0, "synergy_dominant"
    if mi12 >= dis.MI_EPSILON and mi13 / mi12 > dis.DOMINANCE_RATIO_THRESHOLD and cmi132 > cmi123:
        return 1, "synergy_dominant"
    return 0.5, "synergy_kept"


def pair_without_weak_link(ts1, ts2, ts3, mi12=None, mi13=None, mi23=None, ts1_copnorm=None,
                           ts2_copnorm=None, ts3_copnorm=None, verbose=False, ds=1,
                           delay_feat1=0, delay_feat2=0):
    """_disentangle_pair_with_precomputed of 1.2.2 without the criterion MI(A;X) < |II|.

    The MI and conditional MI are computed exactly as in the library function (through
    the module, so that the wrappers above see them); the decision is decide(...,
    weak_link=False).
    """
    if mi12 is None:
        mi12 = dis.get_mi(ts1, ts2, ds=ds, shift=delay_feat1)
    if mi13 is None:
        mi13 = dis.get_mi(ts1, ts3, ds=ds, shift=delay_feat2)
    if mi23 is None:
        mi23 = dis.get_mi(ts2, ts3, ds=ds)
    if ts1_copnorm is not None and ts2_copnorm is not None and ts3_copnorm is not None:
        ts2_cp = np.roll(ts2_copnorm, delay_feat1, axis=-1) if delay_feat1 else ts2_copnorm
        ts3_cp = np.roll(ts3_copnorm, delay_feat2, axis=-1) if delay_feat2 else ts3_copnorm
        cmi123 = dis.cmi_ggg(ts1_copnorm, ts2_cp, ts3_cp, biascorrect=True, demeaned=True)
        cmi132 = dis.cmi_ggg(ts1_copnorm, ts3_cp, ts2_cp, biascorrect=True, demeaned=True)
    else:
        cmi123 = dis.conditional_mi(ts1, ts2, ts3, ds=ds)
        cmi132 = dis.conditional_mi(ts1, ts3, ts2, ds=ds)
    return decide(mi12, mi13, cmi123, cmi132, weak_link=False)[0]


def pair_in_context(ts1, ts2, ts3, **kwargs):
    """The pair test of the run (R.pair_recorded) with the delays known to conditional MI."""
    SHIFT.clear()
    PAIR.clear()
    SHIFT[id(ts2)] = int(kwargs.get("delay_feat1", 0))
    SHIFT[id(ts3)] = int(kwargs.get("delay_feat2", 0))
    try:
        decision = R.pair_recorded(ts1, ts2, ts3, **kwargs)
    finally:
        SHIFT.clear()
    row = R.ROWS[-1]
    values = PAIR.get("values", [])
    if len(values) != 2:
        raise RuntimeError(f"expected two conditional MI values, got {values}")
    mi12, mi13 = row["mi_nx"], row["mi_ny"]        # the objects the rule used (float32 from the stats)
    cmi123, cmi132 = values
    again, branch = decide(mi12, mi13, cmi123, cmi132, weak_link=not OPTS["no_weak_link"])
    if again != decision:
        raise RuntimeError(f"rule replica {again} != decision {decision}: {row}")
    paths = sorted(set(PAIR.get("paths", [])))
    if paths == ["CCC_copula"]:                    # the library rolls the copulas itself
        sx, sy = int(row["delay_x_ds"]), int(row["delay_y_ds"])
    else:
        sx, sy = PAIR.get("shifts", [(0, 0)])[0]
    ii1, ii2 = cmi123 - mi12, cmi132 - mi13
    ds, dx, dy = int(row["ds"]), int(row["delay_x_ds"]), int(row["delay_y_ds"])
    EXACT.append({
        "session": CUR["name"], "neuron": int(row["neuron"]), "feat_x": row["feat_x"], "feat_y": row["feat_y"],
        "x_discrete": bool(ts2.discrete), "y_discrete": bool(ts3.discrete), "cmi_path": "+".join(paths),
        "delay_x_ds": dx, "delay_y_ds": dy, "shift_x_applied": sx, "shift_y_applied": sy, "ds": ds,
        "mi_nx": repr(float(mi12)), "mi_ny": repr(float(mi13)),
        "cmi_nx_given_y": repr(float(cmi123)), "cmi_ny_given_x": repr(float(cmi132)),
        "ii_1": repr(float(ii1)), "ii_2": repr(float(ii2)), "ii_av": repr(float(np.mean([ii1, ii2]))),
        "decision": decision, "branch": branch,
        **convention_checks(ts1, ts2, ts3, ds, dx, dy)})
    return decision


def replay_session(npz_path, config, output_dir):
    name = T.get_exp_name(npz_path)
    t0 = time.time()
    CUR.clear()
    CUR.update(name=name, saved=load_results(SAVED / f"{name}_results.npz"))
    EXACT.clear()
    PROCESS_SESSION(npz_path, config, output_dir)
    if CUR.get("n_me_stats_calls") != 1 or CUR.get("n_feat_feat_calls") != 1:
        raise RuntimeError(f"{name}: saved results used {CUR.get('n_me_stats_calls')} / "
                           f"{CUR.get('n_feat_feat_calls')} times, expected once each")
    folder = Path(output_dir) / "pairs_exact"
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"{name}.csv.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXACT_FIELDS)
        w.writeheader()
        w.writerows(EXACT)
    tmp.replace(folder / f"{name}.csv")
    print(f"[replay] {name}: {len(EXACT)} pairs tested by conditional MI, "
          f"me not float32: {CUR.get('me_not_f32', 0)}, {time.time() - t0:.1f} s", flush=True)


def full_session(npz_path, config, output_dir):
    """The paper run of one session with the second stage computed; pairs_exact written as in the replay."""
    name = T.get_exp_name(npz_path)
    t0 = time.time()
    CUR.clear()
    CUR.update(name=name)
    EXACT.clear()
    PROCESS_SESSION(npz_path, config, output_dir)
    folder = Path(output_dir) / "pairs_exact"
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"{name}.csv.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXACT_FIELDS)
        w.writeheader()
        w.writerows(EXACT)
    tmp.replace(folder / f"{name}.csv")
    print(f"[full] {name}: {len(EXACT)} pairs tested by conditional MI, {time.time() - t0:.1f} s", flush=True)


def install(cmi, no_weak_link, full=False):
    OPTS.update(aligned=(cmi == "aligned"), no_weak_link=no_weak_link)
    R.install()                                   # the wrappers of the paper run
    global PROCESS_SESSION
    PROCESS_SESSION = T.process_session           # R.process_session_saving
    if full:                                      # second stage computed, not read from SAVED
        T.process_session = full_session
    else:
        if SAVED is None:
            raise SystemExit("[v11] stage2_results is not set: give --full or name the saved results")
        T.process_session = replay_session
        pipelines.compute_me_stats = saved_me_stats
        pipelines.compute_feat_feat_significance = saved_feat_feat
    dis.conditional_mi = R._recording("cmi", conditional_mi_at_delays)
    dis.cmi_ggg = R._recording("cmi", cmi_ggg_counted)
    if no_weak_link:
        R._orig["_disentangle_pair_with_precomputed"] = pair_without_weak_link
    dis._disentangle_pair_with_precomputed = pair_in_context


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cmi", choices=("asis", "aligned"), default=V.CMI_DELAYS)
    ap.add_argument("--no-weak-link", action="store_true")
    ap.add_argument("--full", action="store_true", help="compute the second stage instead of reading it")
    ap.add_argument("--output-dir", type=Path, default=V.P.INTENSE_ROOT)
    ap.add_argument("--shard", default=None)
    ap.add_argument("sessions", nargs="*")
    args = ap.parse_args()
    install(args.cmi, args.no_weak_link, args.full)
    print(f"[{'full' if args.full else 'replay'}] cmi={args.cmi} no_weak_link={args.no_weak_link} "
          f"-> {args.output_dir}", flush=True)
    sessions = R.shard_sessions(["--shard", args.shard]) if args.shard else args.sessions
    sys.argv = [str(R.SCRIPT), "--output-dir", str(args.output_dir), *sessions]
    T.main()


if __name__ == "__main__":
    main()
