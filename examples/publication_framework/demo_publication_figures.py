#!/usr/bin/env python
"""Demonstration of the publication-ready figure framework.

This script shows various features of the driada.utils.publication module:
- Creating multi-panel figures with precise physical dimensions
- Auto-scaling fonts and line widths based on panel size
- Adding panel labels (A, B, C, ...)
- Using different style presets
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the publication framework
from driada.utils.publication import (
    PanelLayout,
    StylePreset,
    PanelLabeler,
    ExternalPanel
)


def demo_simple_two_panel():
    """Example 1: Simple 2-panel figure with auto-scaling."""
    print("Creating Example 1: Simple 2-panel figure...")

    # Generate sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # Create layout
    layout = PanelLayout(
        units='cm',
        dpi=300,
        spacing={'wspace': 1.5}  # 1.5cm horizontal gap
    )
    layout.add_panel('A', size=(8, 6))
    layout.add_panel('B', size=(8, 6))
    layout.set_grid(rows=1, cols=2)

    # Create figure with Nature journal styling
    style = StylePreset.nature_journal()
    fig, axes = layout.create_figure(style=style)

    # Plot data
    axes['A'].plot(x, y1, 'b-', linewidth=1.5)
    axes['A'].set_xlabel('Time (s)')
    axes['A'].set_ylabel('Signal A')
    axes['A'].set_title('Sine Wave')

    axes['B'].plot(x, y2, 'r-', linewidth=1.5)
    axes['B'].set_xlabel('Time (s)')
    axes['B'].set_ylabel('Signal B')
    axes['B'].set_title('Cosine Wave')

    # Add panel labels
    labeler = PanelLabeler(fontsize_pt=12, location='top_left')
    labeler.add_labels_to_dict(axes, dpi=layout.dpi)

    # Save
    output_path = Path('examples/publication_framework/example1_simple_two_panel.pdf')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=layout.dpi, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


def demo_complex_layout():
    """Example 2: Complex multi-panel layout with different sizes."""
    print("Creating Example 2: Complex multi-panel layout...")

    # Generate sample data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    data_2d = np.random.randn(20, 20)

    # Create layout with custom positioning
    layout = PanelLayout(
        units='cm',
        dpi=300,
        spacing={'wspace': 1.0, 'hspace': 1.0}
    )
    layout.add_panel('A', size=(6, 6), position=(0, 0))
    layout.add_panel('B', size=(6, 6), position=(0, 1))
    layout.add_panel('C', size=(13, 5), position=(1, 0), col_span=2)
    layout.set_grid(rows=2, cols=2)

    # Create figure with auto-scaling
    style = StylePreset.nature_journal()
    fig, axes = layout.create_figure(style=style)

    # Panel A: Scatter plot
    axes['A'].scatter(np.random.randn(50), np.random.randn(50), alpha=0.5)
    axes['A'].set_xlabel('X')
    axes['A'].set_ylabel('Y')
    axes['A'].set_title('Scatter')

    # Panel B: Heatmap
    im = axes['B'].imshow(data_2d, cmap='viridis', aspect='auto')
    axes['B'].set_xlabel('Column')
    axes['B'].set_ylabel('Row')
    axes['B'].set_title('Heatmap')
    plt.colorbar(im, ax=axes['B'])

    # Panel C: Time series (wide panel)
    for i in range(5):
        axes['C'].plot(x, np.sin(x + i * 0.5), label=f'Series {i+1}')
    axes['C'].set_xlabel('Time (s)')
    axes['C'].set_ylabel('Amplitude')
    axes['C'].set_title('Time Series')
    axes['C'].legend(loc='upper right', fontsize=6)

    # Add panel labels
    labeler = PanelLabeler(fontsize_pt=10, location='top_left')
    labeler.add_labels_to_dict(axes, dpi=layout.dpi)

    # Save
    output_path = Path('examples/publication_framework/example2_complex_layout.pdf')
    fig.savefig(output_path, dpi=layout.dpi, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


def demo_manual_styling():
    """Example 3: Manual styling with make_beautiful()."""
    print("Creating Example 3: Manual styling with make_beautiful()...")

    from driada.utils.plot import make_beautiful

    # Generate sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create layout without auto-styling
    layout = PanelLayout(units='cm', dpi=300)
    layout.add_panel('A', size=(12, 8))
    layout.set_grid(rows=1, cols=1)

    # Create figure without style preset
    fig, axes = layout.create_figure(style=None)

    # Plot data
    axes['A'].plot(x, y, 'b-', linewidth=2)
    axes['A'].set_xlabel('Time (s)')
    axes['A'].set_ylabel('Amplitude')
    axes['A'].set_title('Manual Styling Example')

    # Apply make_beautiful with auto-scaling
    panel_size = layout.get_panel_size('A')
    make_beautiful(
        axes['A'],
        panel_size=panel_size,
        panel_units=layout.units,
        reference_size=(8, 8),
        lowercase_labels=False  # Keep original case
    )

    # Save
    output_path = Path('examples/publication_framework/example3_manual_styling.pdf')
    fig.savefig(output_path, dpi=layout.dpi, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


def demo_external_panel():
    """Example 4: Including external plot placeholder."""
    print("Creating Example 4: External panel placeholder...")

    # Generate sample data for Python panel
    x = np.linspace(0, 10, 100)
    y = np.exp(-x/3) * np.sin(2*x)

    # Create layout
    layout = PanelLayout(units='cm', dpi=300, spacing={'wspace': 1.5})
    layout.add_panel('A', size=(8, 6))
    layout.add_panel('B', size=(8, 6))
    layout.set_grid(rows=1, cols=2)

    # Create figure
    style = StylePreset.nature_journal()
    fig, axes = layout.create_figure(style=style)

    # Panel A: Python plot
    axes['A'].plot(x, y, 'g-', linewidth=1.5)
    axes['A'].set_xlabel('Time (s)')
    axes['A'].set_ylabel('Amplitude')
    axes['A'].set_title('Python Plot')

    # Panel B: External plot placeholder
    ExternalPanel.create_placeholder(
        axes['B'],
        text='External\nR/MATLAB Plot\nGoes Here',
        fontsize=12
    )

    # Add panel labels
    labeler = PanelLabeler(fontsize_pt=12, location='top_left')
    labeler.add_labels_to_dict(axes, dpi=layout.dpi)

    # Save
    output_path = Path('examples/publication_framework/example4_external_placeholder.pdf')
    fig.savefig(output_path, dpi=layout.dpi, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


def demo_units_comparison():
    """Example 5: Comparing cm and inches units."""
    print("Creating Example 5: Units comparison...")

    x = np.linspace(0, 10, 50)
    y = np.sin(x)

    # Create two figures with same physical size but different units

    # Figure 1: Using cm
    layout_cm = PanelLayout(units='cm', dpi=300)
    layout_cm.add_panel('A', size=(8, 6))
    layout_cm.set_grid(rows=1, cols=1)

    style = StylePreset.nature_journal()
    fig1, axes1 = layout_cm.create_figure(style=style)
    axes1['A'].plot(x, y, 'b-')
    axes1['A'].set_xlabel('X (cm units)')
    axes1['A'].set_ylabel('Y')
    axes1['A'].set_title('8 × 6 cm panel')

    output_path1 = Path('examples/publication_framework/example5a_cm_units.pdf')
    fig1.savefig(output_path1, dpi=layout_cm.dpi, bbox_inches='tight')
    print(f"  Saved: {output_path1}")
    plt.close(fig1)

    # Figure 2: Using inches (equivalent size)
    layout_inches = PanelLayout(units='inches', dpi=300)
    layout_inches.add_panel('A', size=(3.15, 2.36))  # ~8×6 cm
    layout_inches.set_grid(rows=1, cols=1)

    fig2, axes2 = layout_inches.create_figure(style=style)
    axes2['A'].plot(x, y, 'r-')
    axes2['A'].set_xlabel('X (inch units)')
    axes2['A'].set_ylabel('Y')
    axes2['A'].set_title('3.15 × 2.36 inch panel')

    output_path2 = Path('examples/publication_framework/example5b_inch_units.pdf')
    fig2.savefig(output_path2, dpi=layout_inches.dpi, bbox_inches='tight')
    print(f"  Saved: {output_path2}")
    plt.close(fig2)

    print("  Note: Both figures should have identical physical size when printed!")


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("Publication Figure Framework Demonstrations")
    print("=" * 70)
    print()

    demo_simple_two_panel()
    print()

    demo_complex_layout()
    print()

    demo_manual_styling()
    print()

    demo_external_panel()
    print()

    demo_units_comparison()
    print()

    print("=" * 70)
    print("All examples completed!")
    print("Check examples/publication_framework/ for output files")
    print("=" * 70)


if __name__ == '__main__':
    main()
