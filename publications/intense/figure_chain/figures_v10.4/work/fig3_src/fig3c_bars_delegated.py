"""Figure 3C: Bar charts — n_sel distribution and delay sign distribution.

Delegation-ready: works standalone or embedded in fig3_delegated.py.
Data: seeded DRIADA 1.2.2 NOF recompute, after disentanglement (fig3c_per_session.csv).
Uses StylePreset for consistent sizing across standalone and embedded modes.
Includes significance brackets via starbars (Wilcoxon signed-rank test).
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from scipy.stats import wilcoxon

import _v122  # noqa: E402,F401  DRIADA 1.2.2 snapshot first on sys.path
from driada.utils.publication import StylePreset


# Per mouse-session percentages written by fig3_numbers.py from the seeded 1.2.2 NOF
# database after disentanglement (pval < 0.001, excess MI > 0.04 bits, activity ratio > 1).
_PER_SESSION_CSV = Path(__file__).parent / 'fig3c_per_session.csv'

# V7_BRACKETS: True draws the significance brackets by starbars exactly as in the v7 script
# (adjacent comparisons share a level); 'stacked' draws the same starbars brackets one call per
# comparison, narrowest first, so that each has its own level; False: the stacked brackets of
# v8-v10.2. YLABEL: label of the left y axis.
V7_BRACKETS = False
YLABEL = '% of recorded neurons'


def load_nsel_delay_stats(per_session_csv=None):
    """Mean and SEM over mouse-sessions of the n_sel and delay-sign percentages.

    Returns
    -------
    nsel_means, nsel_sems : ndarray of shape (3,)
        % of recorded neurons with 1, 2, 3+ selective features.
    delay_means, delay_sems : ndarray of shape (3,)
        Selective pairs with delay sign -, 0, + as % of recorded neurons.
    df : pd.DataFrame
        Per-mouse-session records for statistical testing.
    """
    df = pd.read_csv(per_session_csv or _PER_SESSION_CSV)
    sqrt_n = np.sqrt(len(df))
    nsel = df[['pct_1', 'pct_2', 'pct_3']]
    delay = df[['pct_neg', 'pct_zero', 'pct_pos']]
    return (nsel.mean().values, nsel.std().values / sqrt_n,
            delay.mean().values, delay.std().values / sqrt_n, df)


def _pairwise_wilcoxon(df, columns, x_positions):
    """Run pairwise Wilcoxon signed-rank tests, return starbars annotations."""
    annotations = []
    n = len(columns)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = df[columns[i]].values, df[columns[j]].values
            stat, p = wilcoxon(a, b, alternative='two-sided')
            annotations.append((x_positions[i], x_positions[j], p))
    return annotations


def _stars(p):
    for thr, txt in ((1e-4, '****'), (1e-3, '***'), (1e-2, '**'), (0.05, '*')):
        if p < thr:
            return txt
    return None


def _draw_brackets(ax, annotations, tops, fontsize, lw=0.8):
    """Stacked significance brackets, one level per bracket, so labels never overlap."""
    ymax = max(tops)
    step = 0.13 * ymax
    level = ymax + 0.06 * ymax
    shown = [(x1, x2, p) for x1, x2, p in sorted(annotations, key=lambda a: abs(a[1] - a[0]))
             if _stars(p)]
    for x1, x2, p in shown:
        ax.plot([x1, x1, x2, x2], [level - 0.03 * ymax, level, level, level - 0.03 * ymax],
                color='black', lw=lw)
        ax.text((x1 + x2) / 2, level, _stars(p), ha='center', va='bottom', fontsize=fontsize)
        level += step
    ax.set_ylim(0, level + 0.02 * ymax)


def create_fig3c(
    fig=None,
    style: Optional[StylePreset] = None,
    panel_size: Optional[Tuple[float, float]] = None,
    nsel_means=None,
    nsel_sems=None,
    delay_means=None,
    delay_sems=None,
    df=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create panel C: n_sel distribution + delay sign distribution.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or SubFigure, optional
        If None, creates standalone figure. Otherwise embeds into given subfigure.
    style : StylePreset, optional
        Styling preset.
    panel_size : tuple of (width_cm, height_cm), optional
        Physical panel size in cm. Default: 20.32 x 15.24 (8x6 inches).
    nsel_means, nsel_sems : ndarray of shape (3,), optional
        Pre-computed stats. Loaded from database if None.
    delay_means, delay_sems : ndarray of shape (3,), optional
        Pre-computed stats. Loaded from database if None.
    df : pd.DataFrame, optional
        Raw per-mouse-session data for significance testing.
    """
    if style is None:
        style = StylePreset.from_make_beautiful().copy(
            base_tick_labelsize=16,
            base_label_size=18,
            base_line_width=1.5,
            base_spine_width=1.0,
        )
    if panel_size is None:
        panel_size = (20.32, 15.24)  # 8x6 inches in cm

    # Load data if not provided
    if nsel_means is None:
        nsel_means, nsel_sems, delay_means, delay_sems, df = load_nsel_delay_stats()

    standalone = (fig is None)

    if standalone:
        fig, axes = plt.subplots(
            1, 2,
            figsize=(panel_size[0] / 2.54, panel_size[1] / 2.54), dpi=300)
    else:
        axes = fig.subplots(1, 2)

    tick_fs = style.base_tick_labelsize
    label_fs = style.base_label_size
    lw = style.get_line_width(panel_size, 'cm')
    spine_lw = style.base_spine_width
    star_fs = style.base_legend_fontsize

    # --- Left: n_sel distribution ---
    ax = axes[0]
    colors_nsel = ['#4daf4a', '#377eb8', '#4daf4a']
    ax.bar([1, 2, 3], nsel_means, color=colors_nsel,
           edgecolor='black', linewidth=0.3, width=0.75)
    ax.errorbar([1, 2, 3], nsel_means, yerr=nsel_sems,
                fmt='none', ecolor='black', capsize=3, linewidth=lw * 0.4,
                capthick=lw * 0.3)

    ax.set_xlabel('# of features', fontsize=label_fs)
    ax.set_ylabel(YLABEL, fontsize=label_fs)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['1', '2', '3+'])
    ax.tick_params(axis='both', labelsize=tick_fs,
                   length=style.base_tick_length, pad=style.base_tick_pad)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_lw)

    # Significance brackets (n_sel panel)
    if df is not None:
        nsel_annot = _pairwise_wilcoxon(
            df, ['pct_1', 'pct_2', 'pct_3'], [1, 2, 3])
        if V7_BRACKETS == 'stacked':
            import starbars
            for one in sorted(nsel_annot, key=lambda a: (abs(a[1] - a[0]), a[0])):
                starbars.draw_annotation([one], ax=ax, ns_show=False,
                                         fontsize=star_fs, line_width=0.8)
        elif V7_BRACKETS:
            import starbars
            starbars.draw_annotation(nsel_annot, ax=ax, ns_show=False,
                                     fontsize=star_fs, line_width=0.8)
        else:
            _draw_brackets(ax, nsel_annot, nsel_means + nsel_sems, star_fs)

    # --- Right: delay sign distribution ---
    ax = axes[1]
    colors_delay = ['#4daf4a', '#377eb8', '#ffbf00']
    ax.bar([0, 1, 2], delay_means, color=colors_delay,
           edgecolor='black', linewidth=0.3, width=0.75)
    ax.errorbar([0, 1, 2], delay_means, yerr=delay_sems,
                fmt='none', ecolor='black', capsize=3, linewidth=lw * 0.4,
                capthick=lw * 0.3)

    ax.set_xlabel('delay sign', fontsize=label_fs)
    ax.set_ylabel('')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['\u2212', '0', '+'], fontsize=tick_fs)
    ax.tick_params(axis='both', labelsize=tick_fs,
                   length=style.base_tick_length, pad=style.base_tick_pad)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_lw)

    # Significance brackets (delay sign panel)
    if df is not None:
        delay_annot = _pairwise_wilcoxon(
            df, ['pct_neg', 'pct_zero', 'pct_pos'], [0, 1, 2])
        if V7_BRACKETS == 'stacked':
            import starbars
            for one in sorted(delay_annot, key=lambda a: (abs(a[1] - a[0]), a[0])):
                starbars.draw_annotation([one], ax=ax, ns_show=False,
                                         fontsize=star_fs, line_width=0.8)
        elif V7_BRACKETS:
            import starbars
            starbars.draw_annotation(delay_annot, ax=ax, ns_show=False,
                                     fontsize=star_fs, line_width=0.8)
        else:
            _draw_brackets(ax, delay_annot, delay_means + delay_sems, star_fs)

    fig.patch.set_facecolor('white')

    if standalone:
        fig.tight_layout(w_pad=3)
    else:
        fig.subplots_adjust(left=0.22, right=0.95, top=0.92, bottom=0.25, wspace=0.80)

    return fig, axes


if __name__ == '__main__':
    output_dir = Path(__file__).parent
    fig, axes = create_fig3c()
    out_path = output_dir / 'fig3c_bars_standalone.png'
    fig.savefig(out_path, dpi=300, facecolor='white', edgecolor='none',
                bbox_inches='tight')
    print(f"Saved: {out_path}")
