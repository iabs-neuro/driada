"""Figure 3F: Feature-feature MI heatmap (behavioral similarity matrix).

Delegation-ready: works standalone or embedded in fig3_delegated.py.
Data: behaviour-behaviour GCMI (bits, zero delay) of the seeded DRIADA 1.2.2 NOF recompute, mean over
the 64 sessions (fig3f_matrix.npz from fig3_numbers.py). Pairs significant (p < 0.01, Methods Step 3)
in fewer than half of the sessions in which both features exist are masked grey.
Uses StylePreset for consistent sizing across standalone and embedded modes.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from typing import Optional, Tuple

import _v122  # noqa: E402  DRIADA 1.2.2 snapshot first on sys.path
from driada.utils.publication import StylePreset


# Display name corrections for feature labels
FEATURE_NAME_CORRECTIONS = {
    'bodydirection_harm2': 'body dir.',
    'headdirection_harm2': 'head dir.',
    'place_quad': 'position',
    'speed_quad': 'speed',
    'object1': 'object 1',
    'object2': 'object 2',
    'object3': 'object 3',
    'object4': 'object 4',
    'objects': 'all objects',
    'xy': 'position',
}

_DEFAULT_MATRIX_NPZ = Path(__file__).parent / 'fig3f_matrix.npz'
_MIN_SIG_FRACTION = 0.5


def _correct_names(feat_names):
    """Apply display name corrections."""
    return [FEATURE_NAME_CORRECTIONS.get(f, f) for f in feat_names]


def load_feat_feat_data(matrix_npz=None):
    """Mean feature-feature MI over sessions, with pairs rarely significant set to NaN."""
    z = np.load(matrix_npz or _DEFAULT_MATRIX_NPZ, allow_pickle=True)
    fmi = np.array(z['mi_mean'], dtype=float)
    fmi[z['sig_frac'] < _MIN_SIG_FRACTION] = np.nan
    return fmi, [str(n) for n in z['names']]


def create_fig3f(
    fig=None,
    style: Optional[StylePreset] = None,
    panel_size: Optional[Tuple[float, float]] = None,
    fmi=None,
    feat_names=None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create panel F: feature-feature MI heatmap.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or SubFigure, optional
        If None, creates standalone figure. Otherwise embeds into given subfigure.
    style : StylePreset, optional
        Styling preset. Standalone default: from_make_beautiful() with overrides.
    panel_size : tuple of (width_cm, height_cm), optional
        Physical panel size in cm. Default: 25.40 x 20.32 (10x8 inches).
    fmi : np.ndarray, optional
        Feature-feature MI matrix. Loaded from default source if None.
    feat_names : list of str, optional
        Feature names. Loaded from default source if None.
    """
    if style is None:
        style = StylePreset.from_make_beautiful().copy(
            base_tick_labelsize=16,   # feature name tick labels
            base_label_size=18,       # colorbar label
            base_line_width=2.5,      # grid linewidths between cells
            base_tick_pad=4,          # tick-label spacing
        )
    if panel_size is None:
        panel_size = (25.40, 20.32)  # 10x8 inches in cm

    # Load real data if not provided
    if fmi is None:
        fmi, feat_names = load_feat_feat_data()

    standalone = (fig is None)

    if standalone:
        fig, ax = plt.subplots(
            figsize=(panel_size[0] / 2.54, panel_size[1] / 2.54), dpi=300)
    else:
        ax = fig.subplots()

    # Derive sizes from StylePreset
    tick_fs = style.base_tick_labelsize
    cbar_label_fs = style.base_label_size
    cbar_tick_fs = style.base_tick_labelsize
    linewidths = style.get_line_width(panel_size, 'cm')

    cor_feats = _correct_names(feat_names)

    from matplotlib.colors import PowerNorm
    fmi_max = np.nanmax(fmi)
    df_crossdata = pd.DataFrame(fmi, columns=cor_feats, index=cor_feats)
    # Diagonal and rarely significant pairs (NaN) are rendered grey
    mask = np.eye(len(cor_feats), dtype=bool) | np.isnan(fmi)

    # Square-root colour scale: values span two orders of magnitude
    sns.heatmap(
        df_crossdata,
        mask=mask,
        ax=ax,
        norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=fmi_max),
        cmap='Blues',
        square=True,
        linewidths=linewidths,
        linecolor='white',
        cbar=False,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cbar = fig.colorbar(ax.collections[0], cax=cax)
    ticks = [t for t in (0, 0.05, 0.2, 0.5, 1.0) if t <= fmi_max]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{t:g}' for t in ticks])
    cbar.ax.tick_params(labelsize=cbar_tick_fs)
    cbar.set_label('mutual information, bits', size=cbar_label_fs, labelpad=style.base_tick_pad)

    # Grid and background
    ax.grid(False)
    ax.set_facecolor('lightgray')

    # Force identical labels on both axes (prevents auto-thinning at small sizes)
    ax.set_xticks(np.arange(len(cor_feats)) + 0.5)
    ax.set_yticks(np.arange(len(cor_feats)) + 0.5)
    ax.set_xticklabels(cor_feats, rotation=90, ha='center', fontsize=tick_fs)
    ax.set_yticklabels(cor_feats, rotation=0, fontsize=tick_fs)
    ax.tick_params(pad=style.base_tick_pad)

    fig.patch.set_facecolor('white')

    if standalone:
        fig.tight_layout()
    else:
        # Shorter labels allow tighter margins → larger plot
        fig.subplots_adjust(left=0.18, right=0.93, top=0.97, bottom=0.22)

    return fig, ax


if __name__ == '__main__':
    output_dir = Path(__file__).parent
    fig, ax = create_fig3f()
    out_path = output_dir / 'fig3f_heatmap_standalone.png'
    fig.savefig(out_path, dpi=300, facecolor='white', edgecolor='none',
                bbox_inches='tight')
    print(f"Saved: {out_path}")
