# -*- coding: utf-8 -*-
"""Data cleaning and the forward-selection GLM of the MEC cohort analysis (used by run_xyfix.py).

  clean_series(a)          interpolate over NaN/inf in a behavioural series (tracking dropouts)
  bin_position(x, y)       7 x 7 one-hot spatial bins over the range of x and y
  bin_hd(h)                24 one-hot bins of head direction over [0, 2 pi)
  bin_speed(v)             10 one-hot quantile bins of speed
  glm_select(blocks, y)    Hardcastle-style forward selection over the feature blocks: Ridge
                           (alpha = 1) cross-validated R^2 on 10 contiguous blocks, a feature is
                           added when its per-fold gain passes a one-sided Wilcoxon test at
                           0.05 with Bonferroni correction over the remaining candidates
                           (glm_variants.py); seed orders the folds.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "glm"))
from glm_variants import cv_r2, _forward, _wilcoxon_p  # noqa: E402


def bin_position(x, y, n=7):
    xn = (x - x.min()) / (x.max() - x.min() + 1e-12); yn = (y - y.min()) / (y.max() - y.min() + 1e-12)
    xi = np.clip((xn * n).astype(int), 0, n - 1); yi = np.clip((yn * n).astype(int), 0, n - 1)
    T = len(x); X = np.zeros((T, n * n), np.float32); X[np.arange(T), yi * n + xi] = 1.0; return X


def bin_hd(h, n=24):
    th = np.mod(h, 2 * np.pi); e = np.linspace(0, 2 * np.pi, n + 1)
    idx = np.clip(np.searchsorted(e, th, side="right") - 1, 0, n - 1)
    T = len(h); X = np.zeros((T, n), np.float32); X[np.arange(T), idx] = 1.0; return X


def bin_speed(v, n=10):
    e = np.quantile(v, np.linspace(0, 1, n + 1)); e = np.maximum.accumulate(e + np.arange(len(e)) * 1e-9)
    idx = np.clip(np.searchsorted(e, v, side="right") - 1, 0, n - 1)
    T = len(v); X = np.zeros((T, n), np.float32); X[np.arange(T), idx] = 1.0; return X


def glm_select(blocks, y, seed):
    def sc(sel):
        X = np.hstack([blocks[f] for f in sel]).astype(np.float32) if sel else np.zeros((len(y), 0), np.float32)
        return cv_r2(X, y, 10, 1.0, seed)
    return set(_forward(sc, list(blocks.keys()), _wilcoxon_p, alpha=0.05, mc="bonferroni")["selected"])


def clean_series(a):
    """Interpolate over NaN/inf (tracking dropouts). Returns finite array."""
    a = np.asarray(a, float).copy()
    good = np.isfinite(a)
    if good.all():
        return a
    if good.sum() < 2:
        return np.nan_to_num(a)
    idx = np.arange(len(a))
    a[~good] = np.interp(idx[~good], idx[good], a[good])
    return a
