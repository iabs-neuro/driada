"""Numbers of the interaction-information paragraph of the paper (Results, mixed selectivity).

Reads the pairs that disentanglement decided by conditional MI in the recompute with the
true frame rate of each session, one file per session, written by
compute/run_disent_pair_delays.py (or compute/run_intense_interaction.py):

    <intense_root>/by_type/interaction/<session>_interaction.csv

Columns used: mi_nx, mi_ny (MI of the neuron with each variable at its delay),
cmi_nx_given_y, cmi_ny_given_x (conditional MI), ii_av (interaction information, the mean
of the two estimates, as the library computes it) and decision (0: only feat_x kept,
1: only feat_y kept, 0.5: both kept).

The same pairs are checked against the per-neuron decisions of disentanglement
(per_neuron/<session>_per_neuron.pkl, source 'standard' = conditional MI), and the
multi-selective counts against recount_v8.json of the same configuration.

Writes <numbers_dir>/ii_numbers.json; refuses to overwrite it.
"""
import csv
import glob
import json
import os
import pickle
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import intense_paths as P  # noqa: E402

ROOT = P.INTENSE_RUN
OUT = P.NUMBERS_DIR / 'ii_numbers.json'

files = sorted(glob.glob(str(ROOT / 'interaction' / '*_interaction.csv')))
if len(files) != 64:
    sys.exit('[FAILED] expected 64 interaction files, found %d' % len(files))

rows = []
for f in files:
    session = os.path.basename(f)[:-len('_interaction.csv')]
    with open(f, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            rec = {'session': session, 'neuron': r['neuron'], 'x': r['feat_x'], 'y': r['feat_y']}
            for k in ('mi_nx', 'mi_ny', 'cmi_nx_given_y', 'cmi_ny_given_x', 'ii_av', 'decision'):
                rec[k] = float(r[k])
            rows.append(rec)

keys = {(r['session'], r['neuron'], tuple(sorted((r['x'], r['y'])))) for r in rows}
if len(keys) != len(rows):
    sys.exit('[FAILED] duplicated pairs in the interaction files')

# Cross-check with the per-neuron decisions of disentanglement
source = {}
standard = set()
multi_before = multi_after = 0
for f in sorted(glob.glob(str(ROOT / 'per_neuron' / '*_per_neuron.pkl'))):
    session = os.path.basename(f)[:-len('_per_neuron.pkl')]
    with open(f, 'rb') as fh:
        d = pickle.load(fh)
    for nid, v in d['per_neuron_disent'].items():
        multi_before += 1
        multi_after += len(v['final_sels']) >= 2
        for pair, res in v['pairs'].items():
            source[res['source']] = source.get(res['source'], 0) + 1
            if res['source'] == 'standard':
                standard.add((session, str(nid), tuple(sorted(pair))))
if standard != keys:
    sys.exit('[FAILED] interaction rows differ from the pairs decided by conditional MI')
recount = P.NUMBERS_DIR / 'recount_v8.json'
if recount.exists():
    rc = json.loads(recount.read_text(encoding='utf-8'))
    if (multi_before, multi_after) != (rc['pre']['multi'], rc['post']['multi']):
        sys.exit('[FAILED] multi-selective counts %d -> %d, recount_v8.json has %d -> %d'
                 % (multi_before, multi_after, rc['pre']['multi'], rc['post']['multi']))
else:
    print('[WARN] %s not found: the multi-selective counts (%d -> %d) are not checked; '
          'run numbers/recount_v8.py first' % (recount, multi_before, multi_after))

n = len(rows)
neg = sum(r['ii_av'] < 0 for r in rows)
both = sum(r['decision'] == 0.5 for r in rows)
residuals = [r['cmi_nx_given_y'] / r['mi_nx'] for r in rows] + \
            [r['cmi_ny_given_x'] / r['mi_ny'] for r in rows]


def kind(name):
    if name.startswith('place'):
        return 'position'
    if 'direction' in name:
        return 'direction'
    if name.startswith('speed'):
        return 'speed'
    return 'discrete'


by_kind = {}
for r in rows:
    k = ' + '.join(sorted((kind(r['x']), kind(r['y']))))
    e = by_kind.setdefault(k, {'pairs': 0, 'ii_negative': 0, 'both_kept': 0})
    e['pairs'] += 1
    e['ii_negative'] += r['ii_av'] < 0
    e['both_kept'] += r['decision'] == 0.5

result = {
    'source_dir': str(ROOT),
    'sessions_with_pairs': len({r['session'] for r in rows}),
    'neuron_sessions_with_pairs': len({(r['session'], r['neuron']) for r in rows}),
    'pairs_conditional_mi': n,
    'pairs_by_step_in_multi_selective_neurons': source,
    'multi_selective_before_after': [multi_before, multi_after],
    'ii_negative': neg,
    'ii_negative_pct': round(100 * neg / n, 1),
    'both_kept': both,
    'both_kept_pct': round(100 * both / n, 1),
    # redundant pairs (II < 0) in which both selectivities were kept ('including most' in Results)
    'ii_negative_both_kept': sum(r['ii_av'] < 0 and r['decision'] == 0.5 for r in rows),
    'one_kept': n - both,
    'residual_fraction_median': round(statistics.median(residuals), 3),
    'residual_fraction_at_least_half_pct': round(100 * sum(x >= 0.5 for x in residuals) / len(residuals), 1),
    'by_variable_kind': by_kind,
}
if OUT.exists():
    sys.exit('[FAILED] %s exists; results are never overwritten' % OUT.name)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding='utf-8')
print(json.dumps(result, indent=1, ensure_ascii=False))
print('[OK] wrote', OUT.name)
