"""Plotting functions for INTENSE analysis visualization.

Contains:
- plot_disentanglement: Plot disentanglement heatmap
"""

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import driada


def plot_disentanglement(disent_results, output_base=None):
    """Plot disentanglement heatmap.

    Parameters
    ----------
    disent_results : dict
        Disentanglement results from INTENSE analysis
    output_base : str, optional
        Base path for saving the plot. If provided, saves as PNG.
    """
    import matplotlib.pyplot as plt

    disent_matrix = disent_results.get('disent_matrix')
    count_matrix = disent_results.get('count_matrix')
    feat_names = disent_results.get('feature_names')

    if disent_matrix is None or count_matrix is None:
        print("No disentanglement matrices to plot")
        return

    print(f"\n{'='*60}")
    print("PLOTTING DISENTANGLEMENT HEATMAP")
    print('='*60)

    fig, ax = driada.intense.plot_disentanglement_heatmap(
        disent_matrix,
        count_matrix,
        feat_names,
        title="Feature Disentanglement Analysis",
        figsize=(12, 10),
    )

    if output_base:
        # Save next to output file or with default name
        plot_path = output_base.replace('.json', '_disentanglement.png') if output_base.endswith('.json') else f"{output_base}_disentanglement.png"
    else:
        plot_path = "disentanglement_heatmap.png"

    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.show()
