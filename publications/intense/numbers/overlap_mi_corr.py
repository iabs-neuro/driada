"""Which neuron-feature pairs mutual information and linear correlation detect, and how many of
each other's strongest pairs they share (manuscript v10.6, Results "Mutual information detects
associations invisible to linear correlation" and Methods "Comparison with linear correlation").

Reads only the stored stage-2 statistics of the frame-rate run (results/NOF_*_results.npz):
  mi       [v11] stage2_results               MI on the type-based representations
  pearson  [results] pearson_dir/results      |r| on the raw variables
  rmult    [v11] rmult_dir/results            multiple correlation R on the type-based
                                                       representations of INTENSE

Criterion, the same for every measure:
  significant  = rank p < 0.001, p = (number of stage-2 shuffles at or above the observed value + 1)
                 / (number of shuffles + 1); the number of shuffles of a session is the last
                 dimension of its stored stage-2 table minus one (the observed value);
  anti-selectivity filter as in the paper (signal_ratio > 1 where it is defined; circular
                 variables have none); no disentanglement; position (place, x, y) left out.
  INTENSE dependencies   = significant MI pairs with excess MI (observed minus shuffle mean) > 0.04 bits;
  strongest corr. pairs  = as many significant correlation pairs as there are INTENSE dependencies,
                 those with the largest effect on the MI scale of a Gaussian pair,
                 f(R) - f(mean shuffled R), f(R) = -1/2 log2(1 - R^2).

usage: python numbers/overlap_mi_corr.py   writes <numbers_dir>/overlap_mi_corr.json
"""
import json
import re
from pathlib import Path

import numpy as np

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402
import paths_v11 as V  # noqa: E402

RUNS = {'mi': V.get('stage2_results'),
        'pearson': P.PEARSON_DIR / 'results',
        'rmult': V.get('rmult_dir') / 'results'}
POSITION = {'place', 'x', 'y'}
DIRECTION = {'headdirection', 'bodydirection'}
P_THR = 0.001
EXCESS_THR = 0.04


def variable(feature):
    """Behavioral variable of a stored feature name (the representation suffix removed)."""
    return re.sub(r'_(harm2|quad|2d)$', '', feature)


def group(var):
    return 'direction' if var in DIRECTION else ('speed' if var == 'speed' else 'discrete')


def mi_scale(r):
    r = np.clip(np.asarray(r, dtype=float), 0.0, 1.0 - 1e-12)
    return -0.5 * np.log2(1.0 - r ** 2)


def load(run):
    """{(session, neuron, variable): (significant, effect)} for the pairs that reached stage 2."""
    pairs = {}
    files = sorted(RUNS[run].glob('NOF_*_results.npz'))
    for path in files:
        session = path.name[len('NOF_'):-len('_results.npz')]
        z = np.load(path, allow_pickle=True)
        n_sh = int(z['info_me_total2_shape'][2]) - 1
        raw = z['_stats_json']
        stats = json.loads(str(raw.item() if raw.shape == () else raw[0]))
        for neuron, features in stats.items():
            for feature, s in features.items():
                var = variable(feature)
                if var in POSITION or s.get('rval') is None:
                    continue
                n_below = int(round(s['rval'] * (n_sh + 1))) - 1
                p_rank = (n_sh - n_below + 1) / (n_sh + 1)
                ratio = s.get('signal_ratio')
                significant = p_rank < P_THR and (ratio is None or ratio > 1)
                if run == 'mi':
                    effect = float(s['me_excess'])
                else:
                    effect = float(mi_scale(s['me']) - mi_scale(s['me_null']))
                pairs[(session, neuron, var)] = (significant, effect)
    return pairs, len(files)


def strongest(pairs, n):
    sig = [(effect, key) for key, (ok, effect) in pairs.items() if ok]
    sig.sort(key=lambda t: (-t[0], t[1]))
    return {key for _, key in sig[:n]}


def share(sub, found):
    return [len(sub & found), len(sub), round(len(sub & found) / len(sub), 4) if sub else None]


def main():
    data = {run: load(run) for run in RUNS}
    sessions = {run: n for run, (_, n) in data.items()}
    pairs = {run: p for run, (p, _) in data.items()}
    sig = {run: {k for k, (ok, _) in p.items() if ok} for run, p in pairs.items()}
    deps = {k for k in sig['mi'] if pairs['mi'][k][1] > EXCESS_THR}
    n = len(deps)
    res = {'sessions': sessions,
           'significant_pairs': {run: len(s) for run, s in sig.items()},
           'intense_dependencies': n}
    for corr in ('pearson', 'rmult'):
        top = strongest(pairs[corr], n)
        by_group = {}
        for g in ('direction', 'speed', 'discrete'):
            dg = {k for k in deps if group(k[2]) == g}
            tg = {k for k in top if group(k[2]) == g}
            by_group[g] = {'dependencies_found_by_corr': share(dg, sig[corr]),
                           'strongest_corr_found_by_mi': share(tg, sig['mi'])}
        missed = deps - sig[corr]
        res[corr] = {
            'dependencies_found_by_corr': share(deps, sig[corr]),
            'strongest_corr_found_by_mi': share(top, sig['mi']),
            'by_group': by_group,
            'missed_dependencies_on_direction': share(missed, {k for k in missed if group(k[2]) == 'direction'}),
            'significant_both': len(sig['mi'] & sig[corr]),
            'significant_mi_only': len(sig['mi'] - sig[corr]),
            'significant_corr_only': len(sig[corr] - sig['mi']),
            'ratio_significant_mi_to_corr': round(len(sig['mi']) / len(sig[corr]), 4),
        }
        robust = {}
        for m in (1000, 2500, 10000):
            dm = strongest(pairs['mi'], m)
            tm = strongest(pairs[corr], m)
            robust[m] = [round(len(dm & sig[corr]) / m, 3), round(len(tm & sig['mi']) / m, 3)]
        res[corr]['top_m_mi_found_by_corr__top_m_corr_found_by_mi'] = robust
    out = P.NUMBERS_DIR / 'overlap_mi_corr.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1))
    print('[OK] wrote', out)


if __name__ == '__main__':
    main()
