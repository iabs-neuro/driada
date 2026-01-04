"""Complete mocked visual tests with proper return value simulation."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

# Import functions to test
from driada.intense.visual import (
    plot_pc_activity,
    plot_neuron_feature_density,
    plot_neuron_feature_pair,
    plot_disentanglement_heatmap,
    plot_disentanglement_summary,
    plot_selectivity_heatmap,
)


@pytest.fixture
def mock_experiment():
    """Create a minimal mock experiment with all required attributes."""
    exp = SimpleNamespace()
    exp.n_frames = 100
    exp.n_cells = 5
    exp.signature = "MockExp"

    # Create position data
    exp.x = SimpleNamespace(data=np.random.randn(100), scdata=np.random.randn(100))
    exp.y = SimpleNamespace(data=np.random.randn(100), scdata=np.random.randn(100))
    exp.speed = SimpleNamespace(
        data=np.abs(np.random.randn(100)),
        scdata=np.abs(np.random.randn(100)),
        is_binary=False,
        discrete=False,
    )

    # Create neurons with proper structure
    exp.neurons = []
    for i in range(5):
        neuron = SimpleNamespace()
        # Ensure positive values for log operations
        neuron.ca = SimpleNamespace(
            data=np.abs(np.random.randn(100)) + 0.1,
            scdata=np.abs(np.random.randn(100)) + 0.1,
        )
        neuron.sp = SimpleNamespace(
            data=np.random.randint(0, 2, 100),
            scdata=np.random.randint(0, 2, 100).astype(float),
        )
        exp.neurons.append(neuron)

    # Add stats table
    exp.stats_table = {("x", "y"): []}
    for i in range(5):
        exp.stats_table[("x", "y")].append({"pval": 0.01, "rel_mi_beh": 0.5})

    # Add dynamic features for selectivity heatmap
    exp.dynamic_features = {f"feat_{i}": None for i in range(4)}

    return exp


class TestVisualFunctionsCompletelyMocked:
    """Test visual functions with complete mocking of all dependencies."""

    def test_plot_pc_activity(self, mock_experiment):
        """Test place cell activity plotting."""
        with (
            patch("driada.intense.visual.create_default_figure") as mock_create_fig,
            patch("driada.intense.visual.rescale") as mock_rescale,
        ):

            # Setup mocks
            fig = MagicMock()
            ax = MagicMock()
            mock_create_fig.return_value = (fig, ax)
            mock_rescale.return_value = np.random.rand(100)

            # Mock axis methods
            ax.scatter.return_value = MagicMock()
            ax.set_xlabel.return_value = None
            ax.set_ylabel.return_value = None
            ax.set_title.return_value = None

            # Run function
            result = plot_pc_activity(mock_experiment, 0)

            # Verify
            assert result is ax
            assert ax.scatter.call_count == 2  # Path and spikes
            assert ax.set_xlabel.called
            assert ax.set_ylabel.called
            assert ax.set_title.called

    def test_plot_neuron_feature_density_continuous(self, mock_experiment):
        """Test neuron feature density plotting for continuous features."""
        with (
            patch("matplotlib.pyplot.subplots") as mock_subplots,
            patch("driada.intense.visual.gaussian_kde") as mock_kde,
        ):

            # Setup mocks
            fig = MagicMock()
            ax = MagicMock()
            mock_subplots.return_value = (fig, ax)

            # Mock KDE
            kde_instance = MagicMock()
            # KDE returns flattened values, needs 100x100 = 10000 values
            kde_instance.return_value = np.ones(10000)
            mock_kde.return_value = kde_instance

            # Mock axis methods
            ax.pcolormesh.return_value = MagicMock()
            ax.set_title.return_value = None
            ax.set_xlabel.return_value = None
            ax.set_ylabel.return_value = None

            # Run function
            result = plot_neuron_feature_density(
                mock_experiment, "calcium", 0, "speed", ind1=0, ind2=50
            )

            # Verify
            assert result is ax
            assert mock_kde.called
            assert ax.pcolormesh.called

    def test_plot_neuron_feature_density_binary(self, mock_experiment):
        """Test neuron feature density plotting for binary features."""
        with (
            patch("matplotlib.pyplot.subplots") as mock_subplots,
            patch("seaborn.kdeplot") as mock_kdeplot,
        ):

            # Setup mocks
            fig = MagicMock()
            ax = MagicMock()
            mock_subplots.return_value = (fig, ax)
            mock_kdeplot.return_value = ax

            # Add binary feature
            mock_experiment.binary_feat = SimpleNamespace(
                data=np.random.randint(0, 2, 100),
                scdata=np.random.randint(0, 2, 100).astype(float),
                is_binary=True,
                discrete=True,
            )

            # Run function
            result = plot_neuron_feature_density(
                mock_experiment, "calcium", 0, "binary_feat"
            )

            # Verify
            assert result is ax
            assert mock_kdeplot.call_count == 2  # Called for each binary value
            ax.legend.assert_called()
            ax.set_xlabel.assert_called()
            ax.set_ylabel.assert_called()

    def test_plot_neuron_feature_pair(self, mock_experiment):
        """Test neuron feature pair plotting."""
        with (
            patch("matplotlib.pyplot.subplots") as mock_subplots,
            patch("driada.intense.visual.make_beautiful") as mock_beautiful,
        ):

            # Setup figure with two subplots
            fig = MagicMock()
            ax1 = MagicMock()
            ax2 = MagicMock()
            mock_subplots.return_value = (fig, [ax1, ax2])
            mock_beautiful.side_effect = lambda x, **kwargs: x  # Just return the axis

            # Mock axis methods
            for ax in [ax1, ax2]:
                ax.plot.return_value = [MagicMock()]
                ax.set_xlim.return_value = None
                ax.set_ylim.return_value = None
                ax.set_xlabel.return_value = None
                ax.set_ylabel.return_value = None
                ax.legend.return_value = MagicMock()

            # Run function
            result = plot_neuron_feature_pair(mock_experiment, 0, "speed")

            # Verify
            assert result is fig
            assert mock_subplots.called
            assert ax1.plot.call_count >= 2  # Feature and neural data

    def test_plot_disentanglement_heatmap(self, mock_experiment):
        """Test disentanglement heatmap plotting."""
        with (
            patch("matplotlib.pyplot.subplots") as mock_subplots,
            patch("seaborn.heatmap") as mock_heatmap,
        ):

            # Setup mocks
            fig = MagicMock()
            ax = MagicMock()
            mock_subplots.return_value = (fig, ax)

            # Create test data
            n_features = 4
            disent_matrix = np.random.rand(n_features, n_features) * 20
            count_matrix = np.random.randint(10, 50, (n_features, n_features))
            feat_names = [f"feat{i}" for i in range(n_features)]

            # Run function
            result_fig, result_ax = plot_disentanglement_heatmap(
                disent_matrix, count_matrix, feat_names
            )

            # Verify
            assert result_fig is fig
            assert result_ax is ax
            assert mock_heatmap.called

    def test_plot_disentanglement_summary(self, mock_experiment):
        """Test disentanglement summary plotting."""
        with (
            patch("matplotlib.pyplot.figure") as mock_figure,
            patch("seaborn.heatmap") as mock_heatmap,
        ):

            # Setup mocks
            fig = MagicMock()
            mock_figure.return_value = fig

            # Mock subplots
            ax = MagicMock()
            fig.add_subplot.return_value = ax

            # Mock axis methods
            ax.bar.return_value = MagicMock()
            ax.errorbar.return_value = MagicMock()
            ax.imshow.return_value = MagicMock()
            ax.set_xlim.return_value = None
            ax.set_ylim.return_value = None
            ax.set_xlabel.return_value = None
            ax.set_ylabel.return_value = None
            ax.set_title.return_value = None
            ax.set_xticks.return_value = None
            ax.set_xticklabels.return_value = None

            # Create test data
            n_features = 4
            disent_matrix = np.random.rand(n_features, n_features) * 20
            count_matrix = np.random.randint(10, 50, (n_features, n_features))
            feat_names = [f"feat{i}" for i in range(n_features)]

            # Run function
            result = plot_disentanglement_summary(
                disent_matrix, count_matrix, feat_names
            )

            # Verify
            assert result is fig
            assert fig.add_subplot.call_count >= 3  # Should create multiple subplots
            assert mock_heatmap.called

    def test_plot_selectivity_heatmap(self, mock_experiment):
        """Test selectivity heatmap plotting."""
        with patch("matplotlib.pyplot.subplots") as mock_subplots:

            # Setup mocks
            fig = MagicMock()
            ax = MagicMock()
            mock_subplots.return_value = (fig, ax)

            # Mock imshow
            im = MagicMock()
            ax.imshow.return_value = im
            ax.set_xticks.return_value = None
            ax.set_xticklabels.return_value = None
            ax.set_yticks.return_value = None
            ax.set_yticklabels.return_value = None
            ax.set_xlabel.return_value = None
            ax.set_ylabel.return_value = None
            ax.tick_params.return_value = None
            ax.get_xticklabels.return_value = [MagicMock() for _ in range(4)]

            # Create stats for mock experiment
            def get_stats(cell_id, feat_name, mode="calcium"):
                return {"me": 0.3, "pval": 0.01, "pre_rval": 0.3}

            mock_experiment.get_neuron_feature_pair_stats = get_stats

            # Define significant neurons
            significant_neurons = {0: ["feat_0"], 2: ["feat_1", "feat_2"]}

            # Run function
            result_fig, result_ax, stats = plot_selectivity_heatmap(
                mock_experiment, significant_neurons
            )

            # Verify
            assert result_fig is fig
            assert result_ax is ax
            assert ax.imshow.called
            assert stats["n_selective"] == 2
            assert stats["n_pairs"] == 3
            assert len(stats["metric_values"]) == 3

    def test_edge_cases(self, mock_experiment):
        """Test edge cases without errors."""
        with (
            patch("driada.intense.visual.create_default_figure") as mock_create_fig,
            patch("driada.intense.visual.rescale") as mock_rescale,
        ):

            # Setup mocks
            fig = MagicMock()
            ax = MagicMock()
            mock_create_fig.return_value = (fig, ax)
            mock_rescale.return_value = np.random.rand(100)
            ax.scatter.return_value = MagicMock()

            # Test with None pval
            mock_experiment.stats_table[("x", "y")][0]["pval"] = None
            result = plot_pc_activity(mock_experiment, 0)
            assert result is ax

        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            fig = MagicMock()
            ax = MagicMock()
            mock_subplots.return_value = (fig, ax)
            ax.imshow.return_value = MagicMock()
            ax.get_xticklabels.return_value = []

            # Test with empty significant neurons
            mock_experiment.get_neuron_feature_pair_stats = lambda *args, **kwargs: {
                "me": 0.1,
                "pval": 0.1,
            }
            result_fig, result_ax, stats = plot_selectivity_heatmap(mock_experiment, {})
            assert stats["n_selective"] == 0
            assert stats["n_pairs"] == 0

    def test_spikes_with_binary_raises_error(self, mock_experiment):
        """Test that spikes with binary features raises NotImplementedError."""
        # Add binary feature
        mock_experiment.binary_feat = SimpleNamespace(
            data=np.random.randint(0, 2, 100),
            scdata=np.random.randint(0, 2, 100).astype(float),
            is_binary=True,
            discrete=True,
        )

        # Test that spikes with binary features raises NotImplementedError
        with pytest.raises(
            NotImplementedError, match="Binary feature density plot for spike data"
        ):
            plot_neuron_feature_density(mock_experiment, "spikes", 0, "binary_feat")


class TestPerformanceWithCompleteMocking:
    """Verify that completely mocked tests are fast."""

    def test_all_functions_are_fast(self, mock_experiment):
        """All mocked operations should be very fast."""
        import time

        # Test plot_pc_activity
        with (
            patch("driada.intense.visual.create_default_figure") as mock_create_fig,
            patch("driada.intense.visual.rescale") as mock_rescale,
        ):

            fig = MagicMock()
            ax = MagicMock()
            mock_create_fig.return_value = (fig, ax)
            mock_rescale.return_value = np.random.rand(100)
            ax.scatter.return_value = MagicMock()

            start = time.time()
            result = plot_pc_activity(mock_experiment, 0)
            duration = time.time() - start
            assert duration < 0.1, f"plot_pc_activity took {duration:.3f}s"
            assert result is not None

        # Test plot_neuron_feature_density
        with (
            patch("matplotlib.pyplot.subplots") as mock_subplots,
            patch("driada.intense.visual.gaussian_kde") as mock_kde,
        ):

            fig = MagicMock()
            ax = MagicMock()
            mock_subplots.return_value = (fig, ax)
            kde_instance = MagicMock()
            kde_instance.return_value = np.ones(10000)  # 100x100 flattened
            mock_kde.return_value = kde_instance
            ax.pcolormesh.return_value = MagicMock()

            start = time.time()
            result = plot_neuron_feature_density(mock_experiment, "calcium", 0, "speed")
            duration = time.time() - start
            assert duration < 0.1, f"plot_neuron_feature_density took {duration:.3f}s"
            assert result is not None

        # Test plot_neuron_feature_pair
        with (
            patch("matplotlib.pyplot.subplots") as mock_subplots,
            patch("driada.intense.visual.make_beautiful") as mock_beautiful,
        ):

            fig = MagicMock()
            ax1 = MagicMock()
            ax2 = MagicMock()
            mock_subplots.return_value = (fig, [ax1, ax2])
            mock_beautiful.side_effect = lambda x, **kwargs: x

            for ax in [ax1, ax2]:
                ax.plot.return_value = [MagicMock()]

            start = time.time()
            result = plot_neuron_feature_pair(mock_experiment, 0, "speed")
            duration = time.time() - start
            assert duration < 1.0, f"plot_neuron_feature_pair took {duration:.3f}s"
            assert result is not None
