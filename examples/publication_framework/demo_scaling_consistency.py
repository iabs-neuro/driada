#!/usr/bin/env python
"""Demonstrate the key feature: consistent physical sizing across different panel sizes.

This example shows that fonts, line widths, and other visual elements maintain
the same PHYSICAL size (in cm/inches) regardless of panel dimensions. This is
the core value proposition of the publication framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from driada.utils.publication import PanelLayout, StylePreset, PanelLabeler


def demo_scaling_consistency():
    """
    Show 3 panels of different sizes (4×4, 8×8, 12×12 cm) all with the SAME
    physical font sizes and line widths. When printed, all labels should look
    identical in size despite different panel dimensions.
    """
    print("Creating scaling consistency demonstration...")

    # Generate same data for all panels
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create layout with three different sized panels
    layout = PanelLayout(
        units='cm',
        dpi=300,
        spacing={'wspace': 1.5, 'hspace': 1.5}
    )

    # Panel A: Small (4×4 cm)
    layout.add_panel('A', size=(4, 4), position=(0, 0))

    # Panel B: Medium (8×8 cm)
    layout.add_panel('B', size=(8, 8), position=(0, 1))

    # Panel C: Large (12×12 cm)
    layout.add_panel('C', size=(12, 12), position=(1, 0), col_span=2)

    layout.set_grid(rows=2, cols=2)

    # Create figure with auto-scaling style
    style = StylePreset.nature_journal()
    fig, axes = layout.create_figure(style=style)

    # Plot the SAME data in all three panels
    for name, ax in axes.items():
        ax.plot(x, y, 'b-', linewidth=1.5, label='sin(x)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Panel {name}')
        ax.legend(loc='upper right')
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.2, 1.2)

    # Add panel labels
    labeler = PanelLabeler(fontsize_pt=12, location='top_left')
    labeler.add_labels_to_dict(axes, dpi=layout.dpi)

    # Add annotation explaining the feature
    fig.text(
        0.5, 0.02,
        'Key Feature: All labels, fonts, and line widths have the SAME physical size (in cm)\n'
        'across all three panels despite different panel dimensions.',
        ha='center',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )

    # Save with tight layout
    output_path = Path('examples/publication_framework/scaling_consistency_demo.pdf')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=layout.dpi, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    print("  -> When printed, measure the font sizes in all three panels - they should be identical!")
    plt.close(fig)


def demo_with_vs_without_scaling():
    """
    Compare figures WITH and WITHOUT auto-scaling to show the problem we're solving.
    """
    print("\nCreating comparison: with vs without auto-scaling...")

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Figure 1: WITHOUT auto-scaling (old way - inconsistent)
    layout1 = PanelLayout(units='cm', dpi=300, spacing={'wspace': 1.5})
    layout1.add_panel('A', size=(4, 4), position=(0, 0))
    layout1.add_panel('B', size=(12, 12), position=(0, 1))
    layout1.set_grid(rows=1, cols=2)

    # Create WITHOUT style preset (no auto-scaling)
    fig1, axes1 = layout1.create_figure(style=None)

    # Apply manual styling WITHOUT scaling
    from driada.utils.plot import make_beautiful
    for name, ax in axes1.items():
        ax.plot(x, y, 'r-', linewidth=2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Panel {name}')
        # Use make_beautiful WITHOUT panel_size (no auto-scaling)
        make_beautiful(ax, spine_width=1.5, tick_labelsize=8, label_size=10,
                      title_size=10, legend_fontsize=8, lowercase_labels=False,
                      tight_layout=False)

    labeler = PanelLabeler(fontsize_pt=12, location='top_left')
    labeler.add_labels_to_dict(axes1, dpi=layout1.dpi)

    fig1.text(0.5, 0.02, 'WITHOUT auto-scaling: Fonts look smaller in large panel',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    output1 = Path('examples/publication_framework/comparison_without_scaling.pdf')
    fig1.savefig(output1, dpi=layout1.dpi, bbox_inches='tight')
    print(f"  Saved: {output1}")
    plt.close(fig1)

    # Figure 2: WITH auto-scaling (new way - consistent!)
    layout2 = PanelLayout(units='cm', dpi=300, spacing={'wspace': 1.5})
    layout2.add_panel('A', size=(4, 4), position=(0, 0))
    layout2.add_panel('B', size=(12, 12), position=(0, 1))
    layout2.set_grid(rows=1, cols=2)

    style = StylePreset.nature_journal()
    fig2, axes2 = layout2.create_figure(style=style)

    for name, ax in axes2.items():
        ax.plot(x, y, 'g-', linewidth=1.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Panel {name}')

    labeler.add_labels_to_dict(axes2, dpi=layout2.dpi)

    fig2.text(0.5, 0.02, 'WITH auto-scaling: Fonts have SAME physical size in both panels',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    output2 = Path('examples/publication_framework/comparison_with_scaling.pdf')
    fig2.savefig(output2, dpi=layout2.dpi, bbox_inches='tight')
    print(f"  Saved: {output2}")
    plt.close(fig2)

    print("\n  Compare the two PDFs side-by-side to see the difference!")


def demo_realistic_multi_panel():
    """
    Realistic example: publication figure with mixed panel sizes.
    """
    print("\nCreating realistic multi-panel figure...")

    np.random.seed(42)

    # Create layout - typical paper figure
    layout = PanelLayout(units='cm', dpi=300, spacing={'wspace': 1.2, 'hspace': 1.2})
    layout.add_panel('A', size=(5, 5), position=(0, 0))
    layout.add_panel('B', size=(5, 5), position=(0, 1))
    layout.add_panel('C', size=(5, 5), position=(0, 2))
    layout.add_panel('D', size=(16.2, 5), position=(1, 0), col_span=3)
    layout.set_grid(rows=2, cols=3)

    style = StylePreset.nature_journal()
    fig, axes = layout.create_figure(style=style)

    # Panel A: Scatter
    axes['A'].scatter(np.random.randn(100), np.random.randn(100), alpha=0.6, s=20)
    axes['A'].set_xlabel('Feature 1')
    axes['A'].set_ylabel('Feature 2')
    axes['A'].set_title('Clustering')

    # Panel B: Bar plot
    categories = ['A', 'B', 'C', 'D']
    values = [3.2, 5.1, 4.3, 6.8]
    axes['B'].bar(categories, values, color='steelblue')
    axes['B'].set_xlabel('Category')
    axes['B'].set_ylabel('Value')
    axes['B'].set_title('Measurements')

    # Panel C: Heatmap
    data = np.random.randn(10, 10)
    im = axes['C'].imshow(data, cmap='RdBu_r', aspect='auto')
    axes['C'].set_xlabel('X')
    axes['C'].set_ylabel('Y')
    axes['C'].set_title('Correlation')
    plt.colorbar(im, ax=axes['C'], fraction=0.046)

    # Panel D: Time series (wide panel)
    t = np.linspace(0, 20, 200)
    for i in range(4):
        axes['D'].plot(t, np.sin(t + i*np.pi/2) * np.exp(-t/10),
                      label=f'Condition {i+1}', linewidth=1.5)
    axes['D'].set_xlabel('Time (s)')
    axes['D'].set_ylabel('Response')
    axes['D'].set_title('Time Course')
    axes['D'].legend(loc='upper right', fontsize=6, ncol=4)

    # Add panel labels
    labeler = PanelLabeler(fontsize_pt=14, location='top_left',
                          offset=(-0.15, 1.08))
    labeler.add_labels_to_dict(axes, dpi=layout.dpi)

    output = Path('examples/publication_framework/realistic_multi_panel.pdf')
    fig.savefig(output, dpi=layout.dpi, bbox_inches='tight')
    print(f"  Saved: {output}")
    print("  -> Notice all fonts/labels have consistent physical size across all panels!")
    plt.close(fig)


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("Scaling Consistency Demonstrations")
    print("=" * 70)
    print()

    demo_scaling_consistency()
    demo_with_vs_without_scaling()
    demo_realistic_multi_panel()

    print()
    print("=" * 70)
    print("Done! Check the PDFs to see consistent physical sizing.")
    print("=" * 70)


if __name__ == '__main__':
    main()
