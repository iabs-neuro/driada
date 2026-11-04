"""Numbers behind Figure 3 (C, D/E example, F, A/B checks) from the seeded DRIADA 1.2.2 NOF recompute.

usage: python fig3/fig3_data.py  ->  <cache>/fig3_numbers.json, fig3f_matrix.npz, fig3c_per_session.csv,
       fig3de_candidates.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

HERE = P.CACHE_DIR          # outputs go to the cache folder
HERE.mkdir(parents=True, exist_ok=True)
POST, PRE = P.DB_POST, P.DB_PRE
RUN = P.INTENSE_RUN         # <intense_root>/by_type for representation='by_type'
RAW = P.NOF_ALIGNED
AGG = {'any place', 'any object'}


def selective(df):
    df = df[~df.feature.isin(AGG)]
    ok = (df.pval < P.PVAL_THR) & (df.me_excess > P.MI_EXCESS_THR) & (df.signal_ratio.isna() | (df.signal_ratio > 1))
    return df[ok]


def n_cells():
    out = {}
    for f in sorted(RAW.glob('NOF_*_aligned.npz')):
        _, m, s, _ = f.stem.split('_')
        with np.load(f, allow_pickle=True) as z:
            out[(m, s)] = z['Calcium'].shape[0]
    return out


def main():
    post, pre = selective(pd.read_csv(POST)), selective(pd.read_csv(PRE))
    ncell = n_cells()
    res = {'criterion': 'pval<0.001 & me_excess>0.04 & (signal_ratio NaN or >1); aggregates excluded',
           'n_sessions': len(ncell), 'n_neuron_sessions': int(sum(ncell.values()))}

    # ---- 3C
    rec = []
    for (m, s), n in ncell.items():
        x = post[(post.mouse == m) & (post.session == s)]
        k = x.groupby('neuron_idx').feature.nunique()
        rec.append(dict(mouse=m, session=s, n=n,
                        pct_1=100 * (k == 1).sum() / n, pct_2=100 * (k == 2).sum() / n,
                        pct_3=100 * (k >= 3).sum() / n,
                        pct_neg=100 * (x.delay_sign == -1).sum() / n,
                        pct_zero=100 * (x.delay_sign == 0).sum() / n,
                        pct_pos=100 * (x.delay_sign == 1).sum() / n))
    df = pd.DataFrame(rec)
    df.to_csv(HERE / 'fig3c_per_session.csv', index=False)
    c = {}
    for col in ['pct_1', 'pct_2', 'pct_3', 'pct_neg', 'pct_zero', 'pct_pos']:
        c[col] = {'mean': df[col].mean(), 'sem': df[col].std() / np.sqrt(len(df))}
    tests = {}
    for grp in (['pct_1', 'pct_2', 'pct_3'], ['pct_neg', 'pct_zero', 'pct_pos']):
        for i in range(3):
            for j in range(i + 1, 3):
                tests[f'{grp[i]}_vs_{grp[j]}'] = float(wilcoxon(df[grp[i]], df[grp[j]]).pvalue)
    res['fig3C'] = {'definition': 'per mouse-session % of recorded neurons (after disentanglement); '
                                  'mean +- SEM over 64; two-sided Wilcoxon signed-rank',
                    'stats': c, 'wilcoxon_p': tests,
                    'pooled_pairs_delay_share_pct': (post.delay_sign.value_counts(normalize=True) * 100)
                    .round(2).to_dict(), 'n_pairs_post': int(len(post))}

    # ---- 3A / 3B checks
    def labels(m, s, n, table):
        x = table[(table.mouse == m) & (table.session == s) & (table.neuron_idx == n)]
        return x[['feature', 'me_excess', 'pval', 'opt_delay', 'signal_ratio']].to_dict('records')
    # example neurons of panels A, B, D-E (rows of the synchronized files)
    res['examples'] = {key: {'mouse': m, 'session': s, 'neuron_idx': n,
                             'labels_after_disentanglement': labels(m, s, n, post)}
                       for key, (m, s, n) in {'A_object': ('H02', '3D', 312), 'B_rear': ('H31', '2D', 35),
                                              'DE_hd_place': ('H36', '3D', 158)}.items()}

    # ---- 3D/E candidates: head direction + place after disentanglement
    g = post.pivot_table(index=['mouse', 'session', 'neuron_idx'], columns='feature', values='me_excess')
    cand = g[g['headdirection_2d'].notna() & g['place'].notna()].copy()
    cand['score'] = np.minimum(cand['headdirection_2d'], cand['place'])
    cand['n_feat'] = cand.drop(columns='score').notna().sum(axis=1)
    cand = cand.sort_values('score', ascending=False)
    cand.head(30).to_csv(HERE / 'fig3de_candidates.csv')
    res['fig3DE_n_candidates_hd_and_place'] = int(len(cand))
    res['fig3DE_n_candidates_hd_place_speed'] = int(cand['speed'].notna().sum()) if 'speed' in cand else 0

    # ---- 3F + co-significance
    per = []
    for f in sorted((RUN / 'results').glob('NOF_*_results.npz')):
        with np.load(f, allow_pickle=True) as z:
            nm = json.loads(str(z['_disentanglement_json'][0]))['feature_names']
            P.check_representation(nm)
            per.append((f.stem, nm, z['disent_feat_feat_similarity'], z['disent_feat_feat_significance']))
    # feature sets differ between sessions (e.g. missing objects): align on the union, NaN if absent
    names = list(per[0][1])
    for _, nm, _, _ in per:
        names += [n for n in nm if n not in names]
    ix = {n: i for i, n in enumerate(names)}
    k = len(names)
    mi = np.full((len(per), k, k), np.nan)
    sig = np.full((len(per), k, k), np.nan)
    for t, (stem, nm, m, sg) in enumerate(per):
        j = [ix[n] for n in nm]
        mi[t][np.ix_(j, j)] = m
        sig[t][np.ix_(j, j)] = sg
        cs = pd.read_csv(RUN / 'disentanglement' / f"{stem.replace('_results', '')}_feat_feat_sig.csv", index_col=0)
        assert list(cs.columns) == nm and np.allclose(cs.values, sg), stem
    frac = np.nanmean(sig, axis=0)
    nsess = np.sum(~np.isnan(sig), axis=0)
    mimean = np.nanmean(mi, axis=0)
    np.savez(HERE / 'fig3f_matrix.npz', names=np.array(names), mi_mean=mimean, mi_all=mi,
             sig_frac=frac, n_sessions=nsess)
    res['feature_sessions'] = {n: int(nsess[ix[n], ix[n]]) for n in names}

    def rng(group):
        v = [frac[ix[a], ix[b]] for i, a in enumerate(group) for b in group[i + 1:]]
        return {'min': float(min(v)), 'max': float(max(v)),
                'pairs': {f'{a}-{b}': float(frac[ix[a], ix[b]]) for i, a in enumerate(group) for b in group[i + 1:]}}
    res['cosig_locomotion'] = rng(['speed_quad', 'rest', 'walk', 'locomotion'])
    res['cosig_zones'] = rng(['corners', 'walls', 'center'])
    res['cosig_speed_locomotion'] = float(frac[ix['speed_quad'], ix['locomotion']])
    res['cosig_body_head'] = float(frac[ix['bodydirection_harm2'], ix['headdirection_harm2']])
    off = ~np.eye(len(names), dtype=bool)
    mo = np.where(off, mimean, -1)
    res['fig3F'] = {'mi_mean_offdiag_max': float(mo.max()),
                    'mi_mean_offdiag_max_pair': [names[i] for i in np.unravel_index(mo.argmax(), mo.shape)],
                    'mi_session_max': float(np.nanmax(mi[:, off])),
                    'symmetric': bool(np.allclose(np.nan_to_num(mi), np.nan_to_num(np.transpose(mi, (0, 2, 1))), atol=1e-6))}
    json.dump(res, open(HERE / 'fig3_numbers.json', 'w'), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float)[:6000])


if __name__ == '__main__':
    main()
