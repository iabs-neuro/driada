"""Configuration for visualization tests - optimized for speed."""

import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch

# Disable interactive mode globally
plt.ioff()


@pytest.fixture(scope="module", autouse=True)
def mock_matplotlib():
    """Mock matplotlib to avoid actual figure creation."""
    with (
        patch("matplotlib.pyplot.figure") as mock_figure,
        patch("matplotlib.pyplot.subplots") as mock_subplots,
        patch("matplotlib.pyplot.show") as mock_show,
        patch("matplotlib.pyplot.savefig") as mock_savefig,
    ):

        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()

        # Configure mock returns
        mock_figure.return_value = mock_fig
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_fig.add_subplot.return_value = mock_ax

        # Mock common ax methods
        mock_ax.plot.return_value = [MagicMock()]
        mock_ax.scatter.return_value = MagicMock()
        mock_ax.imshow.return_value = MagicMock()
        mock_ax.contourf.return_value = MagicMock()
        mock_ax.set_xlim.return_value = None
        mock_ax.set_ylim.return_value = None
        mock_ax.set_xlabel.return_value = None
        mock_ax.set_ylabel.return_value = None
        mock_ax.set_title.return_value = None
        mock_ax.legend.return_value = MagicMock()
        mock_ax.get_xlim.return_value = (0, 1)
        mock_ax.get_ylim.return_value = (0, 1)

        yield {
            "figure": mock_figure,
            "subplots": mock_subplots,
            "fig": mock_fig,
            "ax": mock_ax,
        }


@pytest.fixture(scope="module")
def small_visual_experiment():
    """Create a minimal mock experiment for all visual tests."""
    from types import SimpleNamespace

    # Create minimal mock experiment - no real generation!
    exp = SimpleNamespace()
    exp.n_frames = 100  # Only 100 frames
    exp.n_cells = 5
    exp.neurons = []

    # Create mock neurons
    for i in range(5):
        neuron = SimpleNamespace()
        neuron.ca = SimpleNamespace(
            data=np.random.randn(100), scdata=np.random.randn(100)
        )
        neuron.sp = SimpleNamespace(data=np.random.randint(0, 2, 100))
        exp.neurons.append(neuron)

    exp.signature = "TestExp"
    exp.dynamic_features = {}

    return exp


@pytest.fixture
def visual_experiment(small_visual_experiment):
    """Prepare the cached experiment for visual testing."""
    from types import SimpleNamespace

    exp = small_visual_experiment

    # Add minimal required attributes if missing
    T_actual = exp.n_frames
    x_data = np.cumsum(np.random.randn(T_actual) * 0.1)
    y_data = np.cumsum(np.random.randn(T_actual) * 0.1)
    speed_data = np.abs(np.random.randn(T_actual)) * 0.5

    exp.x = SimpleNamespace(data=x_data, scdata=x_data)
    exp.y = SimpleNamespace(data=y_data, scdata=y_data)
    exp.speed = SimpleNamespace(
        data=speed_data, scdata=speed_data, is_binary=False, discrete=False
    )

    if not hasattr(exp, "signature"):
        exp.signature = "TestExp"

    # Minimal stats table
    exp.stats_table = {("x", "y"): []}
    for i in range(5):
        exp.stats_table[("x", "y")].append(
            {"pval": 0.01 * (i + 1), "rel_mi_beh": 0.5 / (i + 1)}
        )

    return exp
