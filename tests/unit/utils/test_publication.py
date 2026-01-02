"""Tests for publication-ready figure framework."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from driada.utils.publication import (
    PanelLayout,
    StylePreset,
    PanelLabeler,
    ExternalPanel,
    to_inches,
    from_inches,
)


class TestUnitConversion:
    """Test unit conversion utilities."""

    def test_to_inches_from_cm(self):
        """Test conversion from cm to inches."""
        assert to_inches(2.54, 'cm') == pytest.approx(1.0)
        assert to_inches((2.54, 5.08), 'cm') == pytest.approx((1.0, 2.0))

    def test_to_inches_from_inches(self):
        """Test conversion from inches to inches (identity)."""
        assert to_inches(5.0, 'inches') == 5.0
        assert to_inches((3.0, 4.0), 'inches') == (3.0, 4.0)

    def test_from_inches_to_cm(self):
        """Test conversion from inches to cm."""
        assert from_inches(1.0, 'cm') == pytest.approx(2.54)
        assert from_inches((1.0, 2.0), 'cm') == pytest.approx((2.54, 5.08))

    def test_invalid_units(self):
        """Test that invalid units raise ValueError."""
        with pytest.raises(ValueError, match="Unknown units"):
            to_inches(1.0, 'meters')
        with pytest.raises(ValueError, match="Unknown units"):
            from_inches(1.0, 'feet')


class TestPanelLayout:
    """Test PanelLayout class."""

    def test_basic_panel_addition(self):
        """Test adding panels to layout."""
        layout = PanelLayout(units='cm', dpi=300)
        layout.add_panel('A', size=(8, 6))
        layout.add_panel('B', size=(8, 6))

        assert 'A' in layout.panels
        assert 'B' in layout.panels
        assert layout.panels['A'].size == (8, 6)

    def test_duplicate_panel_name(self):
        """Test that duplicate panel names raise ValueError."""
        layout = PanelLayout()
        layout.add_panel('A', size=(8, 6))

        with pytest.raises(ValueError, match="already exists"):
            layout.add_panel('A', size=(10, 10))

    def test_negative_panel_size(self):
        """Test that negative sizes raise ValueError."""
        layout = PanelLayout()

        with pytest.raises(ValueError, match="must be positive"):
            layout.add_panel('A', size=(-8, 6))

        with pytest.raises(ValueError, match="must be positive"):
            layout.add_panel('B', size=(8, -6))

    def test_invalid_span(self):
        """Test that invalid spans raise ValueError."""
        layout = PanelLayout()

        with pytest.raises(ValueError, match="must be at least 1"):
            layout.add_panel('A', size=(8, 6), row_span=0)

        with pytest.raises(ValueError, match="must be at least 1"):
            layout.add_panel('B', size=(8, 6), col_span=-1)

    def test_get_panel_size(self):
        """Test getting panel size."""
        layout = PanelLayout(units='cm')
        layout.add_panel('A', size=(8, 6))

        assert layout.get_panel_size('A') == (8, 6)

    def test_get_nonexistent_panel(self):
        """Test that getting nonexistent panel raises ValueError."""
        layout = PanelLayout()

        with pytest.raises(ValueError, match="not found"):
            layout.get_panel_size('Z')

    def test_position_out_of_bounds(self):
        """Test that panels positioned outside grid raise ValueError."""
        layout = PanelLayout(units='cm', dpi=300)
        layout.set_grid(rows=2, cols=2)
        layout.add_panel('A', size=(8, 6), position=(0, 0))
        layout.add_panel('B', size=(8, 6), position=(3, 0))  # Row 3 doesn't exist

        with pytest.raises(ValueError, match="invalid row"):
            layout.create_figure()

    def test_spanning_exceeds_grid(self):
        """Test that spanning panels exceeding grid raise ValueError."""
        layout = PanelLayout(units='cm', dpi=300)
        layout.set_grid(rows=2, cols=2)
        layout.add_panel('A', size=(8, 6), position=(0, 0), col_span=3)  # Would need 3 cols

        with pytest.raises(ValueError, match="exceeds grid columns"):
            layout.create_figure()

    def test_overlapping_panels(self):
        """Test that overlapping panels raise ValueError."""
        layout = PanelLayout(units='cm', dpi=300)
        layout.set_grid(rows=2, cols=2)
        layout.add_panel('A', size=(8, 6), position=(0, 0))
        layout.add_panel('B', size=(8, 6), position=(0, 0))  # Same position

        with pytest.raises(ValueError, match="overlaps"):
            layout.create_figure()

    def test_spanning_overlap(self):
        """Test that spanning panels creating overlap raise ValueError."""
        layout = PanelLayout(units='cm', dpi=300)
        layout.set_grid(rows=2, cols=2)
        layout.add_panel('A', size=(16, 6), position=(0, 0), col_span=2)
        layout.add_panel('B', size=(8, 6), position=(0, 1))  # Overlaps with A's span

        with pytest.raises(ValueError, match="overlaps"):
            layout.create_figure()

    def test_simple_figure_creation(self):
        """Test creating a simple 2-panel figure."""
        layout = PanelLayout(units='cm', dpi=300)
        layout.add_panel('A', size=(8, 6), position=(0, 0))
        layout.add_panel('B', size=(8, 6), position=(0, 1))
        layout.set_grid(rows=1, cols=2)

        fig, axes = layout.create_figure()

        assert isinstance(fig, plt.Figure)
        assert 'A' in axes
        assert 'B' in axes
        assert len(axes) == 2

        plt.close(fig)

    def test_spanning_panel(self):
        """Test creating figure with spanning panel."""
        layout = PanelLayout(units='cm', dpi=300)
        layout.add_panel('A', size=(5, 5), position=(0, 0))
        layout.add_panel('B', size=(5, 5), position=(0, 1))
        layout.add_panel('C', size=(11, 5), position=(1, 0), col_span=2)
        layout.set_grid(rows=2, cols=2)

        fig, axes = layout.create_figure()

        assert len(axes) == 3
        assert all(name in axes for name in ['A', 'B', 'C'])

        plt.close(fig)


class TestStylePreset:
    """Test StylePreset class."""

    def test_fixed_scaling_mode(self):
        """Test that fixed scaling always returns scale factor of 1.0."""
        style = StylePreset(scaling_mode='fixed')

        # Different panel sizes should all get scale=1.0
        assert style.calculate_scale_factor((4, 4), 'cm') == 1.0
        assert style.calculate_scale_factor((8, 8), 'cm') == 1.0
        assert style.calculate_scale_factor((12, 12), 'cm') == 1.0

    def test_area_scaling_mode(self):
        """Test that area scaling scales by sqrt of area ratio."""
        style = StylePreset(
            reference_size=(8, 8),
            reference_units='cm',
            scaling_mode='area'
        )

        # Same size as reference
        assert style.calculate_scale_factor((8, 8), 'cm') == pytest.approx(1.0)

        # Half the area (4×4 vs 8×8): scale = sqrt(16/64) = 0.5
        assert style.calculate_scale_factor((4, 4), 'cm') == pytest.approx(0.5)

        # Double the linear dimensions (16×16 vs 8×8): scale = sqrt(256/64) = 2.0
        assert style.calculate_scale_factor((16, 16), 'cm') == pytest.approx(2.0)

    def test_nature_journal_preset(self):
        """Test Nature journal preset uses fixed scaling by default."""
        style = StylePreset.nature_journal()

        assert style.name == 'nature'
        assert style.scaling_mode == 'fixed'
        assert style.base_spine_width == 1.5
        assert style.base_label_size == 10

    def test_nature_journal_with_area_scaling(self):
        """Test Nature journal preset with area scaling."""
        style = StylePreset.nature_journal(scaling_mode='area')

        assert style.scaling_mode == 'area'
        # Different sizes should have different scale factors
        assert style.calculate_scale_factor((4, 4), 'cm') != 1.0

    def test_fixed_size_preset(self):
        """Test fixed_size preset."""
        style = StylePreset.fixed_size()

        assert style.scaling_mode == 'fixed'
        assert style.calculate_scale_factor((100, 100), 'cm') == 1.0

    def test_apply_to_axes(self):
        """Test applying style to axes."""
        fig, ax = plt.subplots()
        style = StylePreset.nature_journal()

        # Should not raise
        style.apply_to_axes(ax, (8, 6), 'cm')

        # Check that styling was applied
        assert ax.spines['top'].get_linewidth() == 0.0
        assert ax.spines['right'].get_linewidth() == 0.0
        assert ax.spines['bottom'].get_linewidth() > 0.0

        plt.close(fig)

    def test_copy_preset(self):
        """Test copying preset with modifications."""
        original = StylePreset.nature_journal()
        modified = original.copy(base_label_size=12, scaling_mode='area')

        assert modified.base_label_size == 12
        assert modified.scaling_mode == 'area'
        assert original.base_label_size == 10  # Original unchanged
        assert original.scaling_mode == 'fixed'  # Original unchanged


class TestPanelLabeler:
    """Test PanelLabeler class."""

    def test_add_label(self):
        """Test adding label to axes."""
        fig, ax = plt.subplots()
        labeler = PanelLabeler(fontsize_pt=12, location='top_left')

        labeler.add_label(ax, 'A', dpi=300)

        # Check that text was added
        texts = ax.texts
        assert len(texts) == 1
        assert texts[0].get_text() == 'A'

        plt.close(fig)

    def test_add_labels_to_dict(self):
        """Test adding labels to multiple axes."""
        layout = PanelLayout(units='cm', dpi=300)
        layout.add_panel('A', size=(8, 6), position=(0, 0))
        layout.add_panel('B', size=(8, 6), position=(0, 1))
        layout.set_grid(rows=1, cols=2)

        fig, axes = layout.create_figure()
        labeler = PanelLabeler()
        labeler.add_labels_to_dict(axes, dpi=300)

        # Both axes should have labels
        assert len(axes['A'].texts) == 1
        assert len(axes['B'].texts) == 1
        assert axes['A'].texts[0].get_text() == 'A'
        assert axes['B'].texts[0].get_text() == 'B'

        plt.close(fig)


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_multi_panel_with_fixed_sizing(self):
        """Test that different sized panels get same physical font sizes."""
        layout = PanelLayout(units='cm', dpi=300, spacing={'wspace': 1.0})
        layout.add_panel('A', size=(4, 4), position=(0, 0))
        layout.add_panel('B', size=(12, 12), position=(0, 1))
        layout.set_grid(rows=1, cols=2)

        style = StylePreset.nature_journal()  # Uses fixed scaling
        fig, axes = layout.create_figure(style=style)

        # Add some content
        for ax in axes.values():
            ax.plot([1, 2, 3], [1, 2, 3])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

        # With fixed scaling, label sizes should be identical
        label_size_a = axes['A'].xaxis.label.get_size()
        label_size_b = axes['B'].xaxis.label.get_size()

        assert label_size_a == label_size_b

        plt.close(fig)

    def test_complex_layout_with_spanning(self):
        """Test complex layout with spanning panels."""
        layout = PanelLayout(units='cm', dpi=300, spacing={'wspace': 1.0, 'hspace': 1.0})
        layout.add_panel('A', size=(5, 5), position=(0, 0))
        layout.add_panel('B', size=(5, 5), position=(0, 1))
        layout.add_panel('C', size=(5, 5), position=(0, 2))
        layout.add_panel('D', size=(16, 5), position=(1, 0), col_span=3)
        layout.set_grid(rows=2, cols=3)

        style = StylePreset.nature_journal()
        fig, axes = layout.create_figure(style=style)

        assert len(axes) == 4

        # Add content to all panels
        for name, ax in axes.items():
            ax.plot([1, 2], [1, 2])
            ax.set_title(f'Panel {name}')

        plt.close(fig)

    def test_units_cm_vs_inches(self):
        """Test that cm and inch units produce equivalent figures."""
        # 8 cm = 8/2.54 ≈ 3.15 inches, 6 cm = 6/2.54 ≈ 2.36 inches
        layout_cm = PanelLayout(units='cm', dpi=300)
        layout_cm.add_panel('A', size=(8, 6))
        layout_cm.set_grid(rows=1, cols=1)

        # Create equivalent layout in inches (8cm / 2.54 ≈ 3.15 inches)
        layout_in = PanelLayout(units='inches', dpi=300)
        layout_in.add_panel('A', size=(8.0 / 2.54, 6.0 / 2.54))
        layout_in.set_grid(rows=1, cols=1)

        fig_cm, _ = layout_cm.create_figure()
        fig_in, _ = layout_in.create_figure()

        # Figure sizes should be approximately equal (both in inches internally)
        assert fig_cm.get_figwidth() == pytest.approx(fig_in.get_figwidth(), rel=0.01)
        assert fig_cm.get_figheight() == pytest.approx(fig_in.get_figheight(), rel=0.01)

        plt.close(fig_cm)
        plt.close(fig_in)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
