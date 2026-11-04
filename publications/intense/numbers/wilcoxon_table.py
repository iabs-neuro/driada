"""Supplementary Table 7, the Wilcoxon signed-rank tests: comparison, n, W, p.

Each test is recomputed with the same SciPy call that gave the p-value printed in the paper:
- Fig. 3c over sessions and over mice: numbers/fig3c_by_mouse.py (scipy.stats.wilcoxon with its
  defaults, two-sided), on <[figures] cache_dir>/fig3c_per_session.csv of fig3/fig3_data.py;
- Supplementary Section S2 per mouse: s2/16_significance.py (scipy.stats.wilcoxon(diff,
  alternative="greater")), on <[s2] out_dir>/data/s8_significance_permouse.csv.
W is the statistic SciPy returns: for a two-sided test the smaller of the rank sums of the
positive and the negative differences, for the one-sided test the rank sum of the positive ones.
Writes <[results] numbers_dir>/wilcoxon_table.csv and wilcoxon_table.json.

usage: python numbers/wilcoxon_table.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import scipy
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

OUT = P.NUMBERS_DIR
FIG3C = P.CACHE_DIR / 'fig3c_per_session.csv'
PERMOUSE = P.get('s2', 'out_dir') / 'data' / 's8_significance_permouse.csv'

PAIRS = [('pct_1', 'pct_2', 'one versus two variables'),
         ('pct_1', 'pct_3', 'one versus three or more variables'),
         ('pct_2', 'pct_3', 'two versus three or more variables'),
         ('pct_neg', 'pct_zero', 'negative versus near-zero delays'),
         ('pct_neg', 'pct_pos', 'negative versus positive delays'),
         ('pct_zero', 'pct_pos', 'near-zero versus positive delays')]

rows = []
d = pd.read_csv(FIG3C)
m = d.groupby('mouse')[['pct_1', 'pct_2', 'pct_3', 'pct_neg', 'pct_zero', 'pct_pos']].mean()
for level, data in (('sessions', d), ('mice', m)):
    for a, b, label in PAIRS:
        r = wilcoxon(data[a], data[b])
        nonzero = int(((data[a] - data[b]) != 0).sum())
        rows.append({'group': 'Fig. 3c', 'level': level, 'comparison': label, 'n': len(data),
                     'n_nonzero': nonzero, 'alternative': 'two-sided',
                     'W': float(r.statistic), 'p': float(r.pvalue)})

pm = pd.read_csv(PERMOUSE)
r = wilcoxon(pm['diff'], alternative='greater')
rows.append({'group': 'S2', 'level': 'mice', 'n': len(pm), 'n_nonzero': int((pm['diff'] != 0).sum()),
             'comparison': 'observed versus expected share of multiply selective neurons',
             'alternative': 'one-sided (observed greater)', 'W': float(r.statistic), 'p': float(r.pvalue)})

t = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
t.to_csv(OUT / 'wilcoxon_table.csv', index=False)
(OUT / 'wilcoxon_table.json').write_text(json.dumps({'scipy': scipy.__version__, 'rows': rows}, indent=1),
                                         encoding='utf-8')
with pd.option_context('display.width', 200, 'display.max_columns', 20):
    print(t.to_string(index=False))
print('scipy', scipy.__version__)
