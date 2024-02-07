import pandas as pd

from ..intens.intens_base import compute_mi_stats
from ..intens.stats import *
from ..information.info_base import TimeSeries
from ..utils.data import retrieve_relevant_from_nested_dict
import pytest
import numpy as np
import matplotlib.pyplot as plt


def create_correlated_ts(n=100, T=10000):
    np.random.seed(42)
    C = np.zeros((n,n))
    C[1, 99] = 0.9
    C[2, 98] = 0.8
    C[5, 95] = 0.7
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

    cropped_signals = np.zeros((n,T))
    cropped_signals[:, np.array(nnz_time_inds)] = signals[:, np.array(nnz_time_inds)]

    # add noise to remove coinciding values
    small_noise = np.random.multivariate_normal(np.zeros(n),
                                                np.eye(n),
                                                size=T,
                                                check_valid='raise').T*0.2

    cropped_signals += small_noise

    tslist1 = [TimeSeries(sig, discrete=False) for sig in cropped_signals[:n // 2,:]]
    tslist2 = [TimeSeries(sig, discrete=False) for sig in cropped_signals[n // 2:, :]]

    return tslist1, tslist2


def test_stage1():
    tslist1, tslist2 = create_correlated_ts()
    min_shifts = np.full(len(tslist1), fill_value=50)
    computed_stats, computed_significance = compute_mi_stats(tslist1,
                                                             tslist2,
                                                             mode='stage1',
                                                             n_shuffles_stage1=100,
                                                             joint_distr=False,
                                                             mi_distr_type='gamma',
                                                             noise_ampl=1e-3,
                                                             min_shifts=min_shifts,
                                                             ds=1,
                                                             topk1=1,
                                                             verbose=True)

    rel_stats_pairs = retrieve_relevant_from_nested_dict(computed_stats, 'pre_rval', 1)
    rel_sig_pairs = retrieve_relevant_from_nested_dict(computed_significance, 'stage1', True)
    assert rel_sig_pairs == rel_stats_pairs


def test_two_stage():
    tslist1, tslist2 = create_correlated_ts()
    min_shifts = np.full(len(tslist1), fill_value=50)
    computed_stats, computed_significance = compute_mi_stats(tslist1,
                                                             tslist2,
                                                             mode='two_stage',
                                                             n_shuffles_stage1=100,
                                                             n_shuffles_stage2=1000,
                                                             joint_distr=False,
                                                             mi_distr_type='gamma',
                                                             noise_ampl=1e-3,
                                                             min_shifts=min_shifts,
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

    assert rel_sig_pairs == [(1, 49), (2, 48), (5, 45)] # retrieve correlated signals
