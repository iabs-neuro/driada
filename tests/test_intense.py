from src.driada.intense.intense_base import compute_me_stats
from src.driada.information.info_base import TimeSeries, MultiTimeSeries
from src.driada.utils.data import retrieve_relevant_from_nested_dict
import numpy as np


def create_correlated_ts(n=100,
                         binarize_first_half=False,
                         binarize_second_half=False,
                         T=10000,
                         noise_scale=0.2):

    np.random.seed(42)
    C = np.zeros((n,n))
    C[1, n-1] = 0.9
    C[2, n-2] = 0.8
    C[5, n-5] = 0.7
    C = (C + C.T)
    np.fill_diagonal(C, 1)
    signals = np.random.multivariate_normal(np.zeros(n),
                                            C,
                                            size=T,
                                            check_valid='raise').T

    # cutting coherency windows, setting to 0 outside them
    w=100
    starts = np.random.choice(np.arange(w, T-w), size=10)
    nnz_time_inds = []
    for st in starts:
        nnz_time_inds.extend([st + _ for _ in range(w)])

    cropped_signals = np.zeros((n, T))
    cropped_signals[:, np.array(nnz_time_inds)] = signals[:, np.array(nnz_time_inds)]

    # add noise to remove coinciding values
    small_noise = np.random.multivariate_normal(np.zeros(n),
                                                np.eye(n),
                                                size=T,
                                                check_valid='raise').T*noise_scale

    cropped_signals += small_noise

    tslist1 = [TimeSeries(sig, discrete=False) for sig in cropped_signals[:n // 2,:]]
    tslist2 = [TimeSeries(sig, discrete=False) for sig in cropped_signals[n // 2:, :]]

    if binarize_first_half:
        tslist1 = [binarize_ts(ts, 'av') for ts in tslist1]

    if binarize_second_half:
        tslist2 = [binarize_ts(ts, 'av') for ts in tslist2]

    for ts in tslist1:
        ts.shuffle_mask[:50] = 0
    for ts in tslist2:
        ts.shuffle_mask[:50] = 0

    return tslist1, tslist2


def binarize_ts(ts, thr='av'):
    if not ts.discrete:
        if thr == 'av':
            thr = np.mean(ts.data)
        bin_data = np.zeros(len(ts.data))
        bin_data[ts.data >= thr] = 1

    else:
        raise ValueError('binarize_ts called on discrete TimeSeries')

    return TimeSeries(bin_data, discrete=True)


def test_stage1():
    n=40
    k = n // 2  # num of ts in one block
    tslist1, tslist2 = create_correlated_ts(n)
    computed_stats, computed_significance, info = compute_me_stats(tslist1,
                                                                 tslist2,
                                                                 mode='stage1',
                                                                 n_shuffles_stage1=100,
                                                                 joint_distr=False,
                                                                 metric_distr_type='gamma',
                                                                 noise_ampl=1e-3,
                                                                 ds=1,
                                                                 topk1=1,
                                                                 verbose=True)

    rel_stats_pairs = retrieve_relevant_from_nested_dict(computed_stats, 'pre_rval', 1)
    rel_sig_pairs = retrieve_relevant_from_nested_dict(computed_significance, 'stage1', True)
    assert rel_sig_pairs == rel_stats_pairs


def test_two_stage():
    n=20
    k = n//2 # num of ts in one block

    tslist1, tslist2 = create_correlated_ts(n)
    computed_stats, computed_significance, info = compute_me_stats(tslist1,
                                                                     tslist2,
                                                                     mode='two_stage',
                                                                     n_shuffles_stage1=100,
                                                                     n_shuffles_stage2=1000,
                                                                     joint_distr=False,
                                                                     metric_distr_type='gamma',
                                                                     noise_ampl=1e-3,
                                                                     ds=1,
                                                                     topk1=1,
                                                                     topk2=5,
                                                                     multicomp_correction='holm',
                                                                     pval_thr=0.01,
                                                                     verbose=True)

    rel_sig_pairs = retrieve_relevant_from_nested_dict(computed_significance,
                                                       'stage2',
                                                       True,
                                                       allow_missing_keys=True)

    assert rel_sig_pairs == [(1, k-1), (2, k-2), (5, k-5)] # retrieve correlated signals


def aggregate_two_ts(ts1, ts2):
    # add small noise to break degeneracy
    mod_lts1 = TimeSeries(ts1.data + np.random.random(size=len(ts1.data)) * 1e-6)
    mod_lts2 = TimeSeries(ts2.data + np.random.random(size=len(ts2.data)) * 1e-6)
    mts = MultiTimeSeries([mod_lts1, mod_lts2])  # add last two TS into a single 2-d MTS
    return mts

def test_mixed_dimensions():
    n=20
    k = n // 2  # num of ts in one block

    tslist1, tslist2 = create_correlated_ts(n)
    lts1, lts2 = tslist2[-2:]
    mts = aggregate_two_ts(lts1, lts2)

    # we expect the correlation between this multi-ts (index k) and ts with indices 1,2
    mod_tslist2 = tslist2 + [mts]

    computed_stats, computed_significance, info = compute_me_stats(tslist1,
                                                                    mod_tslist2,
                                                                     mode='two_stage',
                                                                     n_shuffles_stage1=100,
                                                                     n_shuffles_stage2=1000,
                                                                     joint_distr=False,
                                                                     allow_mixed_dimensions=True,
                                                                     metric_distr_type='gamma',
                                                                     noise_ampl=1e-3,
                                                                     ds=1,
                                                                     topk1=1,
                                                                     topk2=5,
                                                                     multicomp_correction='holm',
                                                                     pval_thr=0.01,
                                                                     verbose=True)

    rel_sig_pairs = retrieve_relevant_from_nested_dict(computed_significance,
                                                       'stage2',
                                                       True,
                                                       allow_missing_keys=True)

    # index n is a multi-ts
    assert set(rel_sig_pairs) == set([(1, k-1), (2, k-2), (5, k-5), (1, k), (2, k)]) # retrieve correlated signals


def test_mirror():
    # test INTENSE of a TimeSeries and MultiTimeSeries set with itself
    n=20
    k = n // 2  # num of ts in one block

    tslist1, tslist2 = create_correlated_ts(n)
    lts1, lts2 = tslist2[-2:]
    mts1 = aggregate_two_ts(lts1, lts2)
    fts1, fts2 = tslist2[:2]
    mts2 = aggregate_two_ts(fts1, fts2)

    mod_tslist2 = tslist2 + [mts1, mts2]
    # we expect the correlation between mts1 (index k) and ts with indices k-2, k-1
    # we expect the correlation between mts2 (index k+1) and ts with indices 0, 1

    computed_stats, computed_significance, info = compute_me_stats(mod_tslist2,
                                                                    mod_tslist2,
                                                                     mode='two_stage',
                                                                     n_shuffles_stage1=100,
                                                                     n_shuffles_stage2=1000,
                                                                     joint_distr=False,
                                                                     allow_mixed_dimensions=True,
                                                                     metric_distr_type='gamma',
                                                                     noise_ampl=1e-3,
                                                                     ds=1,
                                                                     topk1=1,
                                                                     topk2=5,
                                                                     multicomp_correction='holm',
                                                                     pval_thr=0.01,
                                                                     enable_parallelization=0,
                                                                     seed=1,
                                                                     verbose=True)

    rel_sig_pairs = retrieve_relevant_from_nested_dict(computed_significance,
                                                       'stage2',
                                                       True,
                                                       allow_missing_keys=True)

    print(set(rel_sig_pairs))
    assert set(rel_sig_pairs) == set([(k, k-1), (k-2, k), (k+1, 1), (k, k-2), (0, k+1),\
                                      (k-1, k), (1, k+1), (k+1, 0)]) # retrieve correlated signals


def test_two_stage_corr():
    n=20
    k = n // 2  # num of ts in one block

    tslist1, tslist2 = create_correlated_ts(n, noise_scale=0.2)
    computed_stats, computed_significance, info = compute_me_stats(tslist1,
                                                                   tslist2,
                                                                   metric='spearmanr',
                                                                   mode='two_stage',
                                                                   n_shuffles_stage1=100,
                                                                   n_shuffles_stage2=1000,
                                                                   joint_distr=False,
                                                                   metric_distr_type='norm',
                                                                   noise_ampl=1e-4,
                                                                   ds=1,
                                                                   topk1=1,
                                                                   topk2=5,
                                                                   multicomp_correction='holm',
                                                                   pval_thr=0.01,
                                                                   verbose=True,
                                                                   enable_parallelization=False)

    rel_sig_pairs = retrieve_relevant_from_nested_dict(computed_significance,
                                                       'stage2',
                                                       True,
                                                       allow_missing_keys=True)

    # retrieve correlated signals, false positives are likely
    assert set([(1, k-1), (2, k-2), (5, k-5)]).issubset(set(rel_sig_pairs))


def test_two_stage_avsignal():
    n=20
    k = n // 2  # num of ts in one block

    tslist1, tslist2 = create_correlated_ts(n, noise_scale=0.01, binarize_second_half=True)
    computed_stats, computed_significance, info = compute_me_stats(tslist1,
                                                                   tslist2,
                                                                   metric='av',
                                                                   mode='two_stage',
                                                                   n_shuffles_stage1=100,
                                                                   n_shuffles_stage2=1000,
                                                                   joint_distr=False,
                                                                   metric_distr_type='norm',
                                                                   noise_ampl=1e-4,
                                                                   ds=1,
                                                                   topk1=1,
                                                                   topk2=5,
                                                                   multicomp_correction='holm',
                                                                   pval_thr=0.1,
                                                                   verbose=True,
                                                                   enable_parallelization=False)

    rel_sig_pairs = retrieve_relevant_from_nested_dict(computed_significance,
                                                       'stage2',
                                                       True,
                                                       allow_missing_keys=True)

    print(rel_sig_pairs)
    # retrieve correlated signals, false positives are likely
    #assert set([(1, k-1), (2, k-2), (5, k-5)]).issubset(set(rel_sig_pairs))