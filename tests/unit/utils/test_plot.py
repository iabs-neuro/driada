"""Tests for plotting utilities."""

import numpy as np
import matplotlib.pyplot as plt
from driada.utils.plot import make_beautiful, create_default_figure, plot_mat


class TestMakeBeautiful:
    """Test the make_beautiful function."""

    def test_make_beautiful_basic(self):
        """Test basic axis styling."""
        fig, ax = plt.subplots()
        styled_ax = make_beautiful(ax)

        # Check spine visibility
        assert ax.spines["bottom"].get_linewidth() == 4
        assert ax.spines["left"].get_linewidth() == 4
        assert ax.spines["top"].get_linewidth() == 0.0
        assert ax.spines["right"].get_linewidth() == 0.0

        # Check tick parameters
        tick_params = ax.xaxis._major_tick_kw
        assert tick_params["tick1On"] == True  # 'in' direction

        # Check that same axis is returned
        assert styled_ax is ax

        plt.close(fig)

    def test_make_beautiful_custom_params(self):
        """Test styling with custom parameters."""
        fig, ax = plt.subplots()
        make_beautiful(
            ax,
            spine_width=2,
            tick_width=3,
            tick_length=10,
            tick_labelsize=20,
            label_size=25,
        )

        # Check custom spine width
        assert ax.spines["bottom"].get_linewidth() == 2
        assert ax.spines["left"].get_linewidth() == 2

        # Check label sizes
        assert ax.xaxis.label.get_size() == 25
        assert ax.yaxis.label.get_size() == 25

        plt.close(fig)

    def test_make_beautiful_with_title(self):
        """Test title styling."""
        fig, ax = plt.subplots()
        ax.set_title("Test Title")
        make_beautiful(ax, title_size=35)

        assert ax.title.get_size() == 35

        plt.close(fig)

    def test_make_beautiful_with_legend(self):
        """Test legend styling."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="Test")
        ax.legend()
        make_beautiful(ax, legend_fontsize=20)

        # Check legend font size was applied
        legend_texts = ax.legend_.get_texts()
        assert len(legend_texts) > 0
        assert legend_texts[0].get_fontsize() == 20

        # Check legend frame is off by default
        assert ax.legend_.get_frame_on() == False

        plt.close(fig)

    def test_make_beautiful_legend_frame(self):
        """Test legend frame customization."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="Test")
        ax.legend()

        # Test with frame on
        make_beautiful(ax, legend_frameon=True)
        assert ax.legend_.get_frame_on() == True

        plt.close(fig)

        # Test with frame off (default)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="Test")
        ax.legend()
        make_beautiful(ax)
        assert ax.legend_.get_frame_on() == False

        plt.close(fig)

    def test_make_beautiful_legend_location(self):
        """Test legend location options."""
        # Test 'above' positioning
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="Line 1")
        ax.plot([0, 1], [1, 0], label="Line 2")
        ax.legend()
        make_beautiful(ax, legend_loc='above')

        # Check that legend was repositioned
        bbox = ax.legend_.get_bbox_to_anchor()
        assert bbox is not None
        # Legend should be positioned above (y > 1)

        plt.close(fig)

        # Test 'below' positioning
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="Line 1")
        ax.plot([0, 1], [1, 0], label="Line 2")
        ax.legend()
        make_beautiful(ax, legend_loc='below')

        # Check that legend was repositioned
        bbox = ax.legend_.get_bbox_to_anchor()
        assert bbox is not None
        # Legend should be positioned below (y < 0)

        plt.close(fig)

        # Test standard matplotlib location
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="Test")
        ax.legend()
        make_beautiful(ax, legend_loc='upper left')

        # Legend should be in upper left
        assert ax.legend_._loc == 2  # 2 is the code for 'upper left'

        plt.close(fig)

        # Test 'auto' (default)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="Test")
        ax.legend()
        make_beautiful(ax, legend_loc='auto')

        # Should use matplotlib's automatic placement
        assert ax.legend_ is not None

        plt.close(fig)

    def test_make_beautiful_tight_layout(self):
        """Test tight layout for both axes."""
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])

        # Test with tight layout (default)
        make_beautiful(ax)
        x_margins, y_margins = ax.margins()
        assert x_margins == 0
        assert y_margins == 0

        plt.close(fig)

        # Test without tight layout
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
        original_x_margins, original_y_margins = ax.margins()
        make_beautiful(ax, tight_layout=False)
        # Margins should remain unchanged
        assert ax.margins()[0] == original_x_margins
        assert ax.margins()[1] == original_y_margins

        plt.close(fig)

    def test_make_beautiful_remove_origin_tick(self):
        """Test removing ticks at origin."""
        fig, ax = plt.subplots()
        ax.plot([-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2])

        # Test with origin tick removal
        make_beautiful(ax, remove_origin_tick=True)

        xticks = ax.get_xticks()
        yticks = ax.get_yticks()

        assert 0.0 not in xticks
        assert 0.0 not in yticks

        plt.close(fig)

        # Test without origin tick removal (default)
        fig, ax = plt.subplots()
        ax.plot([-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2])
        make_beautiful(ax, remove_origin_tick=False)

        # May or may not have 0 depending on matplotlib's automatic tick placement
        # Just verify no error occurs

        plt.close(fig)

    def test_make_beautiful_dpi(self):
        """Test DPI setting."""
        fig, ax = plt.subplots()
        original_dpi = fig.get_dpi()

        make_beautiful(ax, dpi=150)
        assert fig.get_dpi() == 150

        plt.close(fig)


class TestCreateDefaultFigure:
    """Test the create_default_figure function."""

    def test_create_default_figure_basic(self):
        """Test basic figure creation."""
        fig, ax = create_default_figure()

        # Check figure size
        assert fig.get_size_inches()[0] == 16
        assert fig.get_size_inches()[1] == 12

        # Check that styling was applied
        assert ax.spines["bottom"].get_linewidth() == 4
        assert ax.spines["top"].get_linewidth() == 0.0

        plt.close(fig)

    def test_create_default_figure_custom_size(self):
        """Test figure with custom size."""
        fig, ax = create_default_figure(figsize=(10, 8))

        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 8

        plt.close(fig)

    def test_create_default_figure_style_kwargs(self):
        """Test passing style kwargs."""
        fig, ax = create_default_figure(spine_width=6, tick_labelsize=30)

        assert ax.spines["bottom"].get_linewidth() == 6

        plt.close(fig)

    def test_create_default_figure_subplots(self):
        """Test creating figure with multiple subplots."""
        fig, axes = create_default_figure(nrows=2, ncols=2)

        # Check we have 2x2 array of axes
        assert axes.shape == (2, 2)

        # Check all axes are styled
        for ax in axes.flat:
            assert ax.spines["bottom"].get_linewidth() == 4
            assert ax.spines["top"].get_linewidth() == 0.0

        plt.close(fig)

    def test_create_default_figure_subplots_no_squeeze(self):
        """Test subplot creation without squeezing."""
        fig, axes = create_default_figure(nrows=1, ncols=2, squeeze=False)

        # Should be 2D array even with single row
        assert axes.shape == (1, 2)

        plt.close(fig)

    def test_create_default_figure_shared_axes(self):
        """Test creating subplots with shared axes."""
        fig, axes = create_default_figure(nrows=2, ncols=1, sharex=True)

        # Verify axes are shared
        assert axes[0].get_shared_x_axes().joined(axes[0], axes[1])

        plt.close(fig)


class TestPlotMat:
    """Test the plot_mat function."""

    def test_plot_mat_basic(self):
        """Test basic matrix plotting."""
        mat = np.random.rand(10, 10)
        fig, ax = plot_mat(mat)

        # Check figure was created
        assert fig is not None
        assert isinstance(ax, plt.Axes)

        # Check image was plotted
        assert len(ax.images) == 1

        # Check colorbar was added
        assert len(fig.axes) > 1  # Main ax + colorbar ax

        plt.close(fig)

    def test_plot_mat_existing_axis(self):
        """Test plotting on existing axis."""
        fig, ax = plt.subplots()
        mat = np.random.rand(5, 5)

        returned_fig, returned_ax = plot_mat(mat, ax=ax)

        # Should return None for fig when ax is provided
        assert returned_fig is None
        assert returned_ax is ax

        # Check image was plotted
        assert len(ax.images) == 1

        plt.close(fig)

    def test_plot_mat_no_colorbar(self):
        """Test plotting without colorbar."""
        mat = np.random.rand(10, 10)
        fig, ax = plot_mat(mat, with_cbar=False)

        # Should only have one axis (no colorbar)
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_plot_mat_custom_params(self):
        """Test plotting with custom parameters."""
        mat = np.random.rand(10, 10)
        fig, ax = plot_mat(
            mat, figsize=(8, 8), cmap="plasma", aspect="equal", vmin=0, vmax=0.5
        )

        # Check figure size
        assert fig.get_size_inches()[0] == 8
        assert fig.get_size_inches()[1] == 8

        # Check image properties
        img = ax.images[0]
        assert img.get_cmap().name == "plasma"
        assert img.get_array().min() >= 0  # Data might be clipped
        assert img.get_clim() == (0, 0.5)

        plt.close(fig)

    def test_plot_mat_imshow_kwargs(self):
        """Test passing additional imshow kwargs."""
        mat = np.random.rand(10, 10)
        fig, ax = plot_mat(mat, interpolation="nearest", alpha=0.8)

        img = ax.images[0]
        assert img.get_interpolation() == "nearest"
        assert img.get_alpha() == 0.8

        plt.close(fig)


class TestIntegration:
    """Integration tests for plotting utilities."""

    def test_full_workflow(self):
        """Test complete workflow of creating and styling figures."""
        # Create figure with subplots
        fig, axes = create_default_figure(
            nrows=2,
            ncols=2,
            figsize=(12, 10),
            spine_width=3,
            tick_labelsize=18,
            dpi=100,
        )

        # Plot different data in each subplot
        for i, ax in enumerate(axes.flat):
            data = np.random.rand(20, 20) * (i + 1)
            plot_mat(data, ax=ax, with_cbar=False)
            ax.set_title(f"Subplot {i+1}")

        # Verify figure properties
        assert fig.get_dpi() == 100
        assert fig.get_size_inches()[0] == 12
        assert fig.get_size_inches()[1] == 10

        # Verify all axes are styled and have data
        for ax in axes.flat:
            assert ax.spines["bottom"].get_linewidth() == 3
            assert len(ax.images) == 1
            # Title size is set when make_beautiful is called, not when set_title is called
            # So we need to re-apply styling after setting title
            make_beautiful(ax, spine_width=3, title_size=30)
            assert ax.title.get_size() == 30

        plt.close(fig)

    def test_single_vs_multiple_axes(self):
        """Test that single and multiple axes behave consistently."""
        # Single axis
        fig1, ax1 = create_default_figure()
        assert isinstance(ax1, plt.Axes)

        # Multiple axes with squeeze
        fig2, axes2 = create_default_figure(nrows=1, ncols=1)
        assert isinstance(axes2, plt.Axes)  # Should be squeezed to single axis

        # Multiple axes without squeeze
        fig3, axes3 = create_default_figure(nrows=1, ncols=1, squeeze=False)
        assert isinstance(axes3, np.ndarray)
        assert axes3.shape == (1, 1)

        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
