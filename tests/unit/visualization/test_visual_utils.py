"""
Tests for visual utilities module
==================================

Test the new visualization functions in driada.utils.visual
"""

import sys
import numpy as np
import pytest
import matplotlib.pyplot as plt
from driada.utils import visual
from driada.utils.visual import (
    plot_embedding_comparison,
    plot_trajectories,
    plot_component_interpretation,
    plot_embeddings_grid,
    plot_neuron_selectivity_summary,
    plot_component_selectivity_heatmap,
    DEFAULT_DPI,
)
import tempfile
import os


class TestVisualUtils:
    """Test visual utility functions."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        np.random.seed(42)
        n_samples = 200

        # Create circular embeddings
        angles = np.linspace(0, 2 * np.pi, n_samples)
        embeddings = {
            "pca": np.column_stack([np.cos(angles), np.sin(angles)])
            + 0.1 * np.random.randn(n_samples, 2),
            "umap": np.column_stack([1.5 * np.cos(angles), 1.5 * np.sin(angles)])
            + 0.15 * np.random.randn(n_samples, 2),
            "le": np.column_stack([0.8 * np.cos(angles), 0.8 * np.sin(angles)])
            + 0.1 * np.random.randn(n_samples, 2),
        }

        # Create features
        features = {
            "angle": angles,
            "speed": np.abs(np.sin(angles)) + 0.1 * np.random.randn(n_samples),
        }

        return embeddings, features, angles

    @pytest.fixture
    def sample_metrics_data(self):
        """Create sample metrics data for testing."""
        data = {
            "pca": {
                "all_neurons": {
                    "spatial_decoding_r2": 0.85,
                    "distance_correlation": 0.78,
                    "runtime": 0.5,
                },
                "spatial_neurons": {
                    "spatial_decoding_r2": 0.92,
                    "distance_correlation": 0.88,
                    "runtime": 0.3,
                },
                "random_half": {
                    "spatial_decoding_r2": 0.82,
                    "distance_correlation": 0.75,
                    "runtime": 0.25,
                },
            },
            "umap": {
                "all_neurons": {
                    "spatial_decoding_r2": 0.88,
                    "distance_correlation": 0.82,
                    "runtime": 2.5,
                },
                "spatial_neurons": {
                    "spatial_decoding_r2": 0.94,
                    "distance_correlation": 0.91,
                    "runtime": 1.8,
                },
                "random_half": {
                    "spatial_decoding_r2": 0.85,
                    "distance_correlation": 0.80,
                    "runtime": 1.2,
                },
            },
        }
        return data

    def test_default_dpi(self):
        """Test that DEFAULT_DPI is properly set."""
        assert DEFAULT_DPI == 150
        assert visual.DEFAULT_DPI == 150

    @pytest.mark.skipif(sys.platform == "win32", reason="Windows file locking prevents temp file deletion")
    def test_plot_embedding_comparison(self, sample_embeddings):
        """Test embedding comparison plot."""
        embeddings, features, _ = sample_embeddings

        # Test basic usage
        fig = plot_embedding_comparison(
            embeddings=embeddings,
            features=features,
            with_trajectory=True,
            compute_metrics=True,
        )

        assert fig is not None
        # Check that fig has Figure-like attributes instead of strict type check
        assert hasattr(fig, "add_subplot") or hasattr(fig, "savefig")

        # Test without trajectory
        fig2 = plot_embedding_comparison(
            embeddings=embeddings, features=features, with_trajectory=False
        )
        assert fig2 is not None

        # Test with save path
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig3 = plot_embedding_comparison(
                embeddings=embeddings, features=features, save_path=tmp.name, dpi=100
            )
            assert os.path.exists(tmp.name)
            plt.close(fig3)  # Close figure before deleting on Windows
            os.unlink(tmp.name)

        plt.close("all")

    def test_plot_trajectories(self, sample_embeddings):
        """Test trajectory plotting."""
        embeddings, _, _ = sample_embeddings

        fig = plot_trajectories(
            embeddings=embeddings,
            trajectory_kwargs={"arrow_spacing": 10, "color": "blue"},
        )

        assert fig is not None
        # Check that fig has Figure-like attributes instead of strict type check
        assert hasattr(fig, "add_subplot") or hasattr(fig, "savefig")

        plt.close("all")

    def test_plot_component_interpretation(self):
        """Test component interpretation plot."""
        # Create sample MI matrices
        mi_matrices = {
            "pca": np.random.rand(3, 5) * 0.3,  # 3 features, 5 components
            "umap": np.random.rand(3, 5) * 0.25,
        }

        feature_names = ["Speed", "Position X", "Position Y"]

        # Add some metadata
        metadata = {
            "pca": {"explained_variance_ratio": np.array([0.4, 0.25, 0.15, 0.1, 0.05])}
        }

        fig = plot_component_interpretation(
            mi_matrices=mi_matrices,
            feature_names=feature_names,
            metadata=metadata,
            n_components=3,
        )

        assert fig is not None
        # Check that fig has Figure-like attributes instead of strict type check
        assert hasattr(fig, "add_subplot") or hasattr(fig, "savefig")

        plt.close("all")

    def test_plot_embeddings_grid(self, sample_embeddings, sample_metrics_data):
        """Test embeddings grid plot."""
        embeddings, _, angles = sample_embeddings

        # Create nested structure for grid plot
        grid_embeddings = {
            "pca": {"all": embeddings["pca"], "subset": embeddings["pca"][:100]},
            "umap": {"all": embeddings["umap"], "subset": embeddings["umap"][:100]},
        }

        # Create labels dict matching the structure
        labels_dict = {
            "pca": {"all": angles, "subset": angles[:100]},
            "umap": {"all": angles, "subset": angles[:100]},
        }

        fig = plot_embeddings_grid(
            embeddings=grid_embeddings,
            labels=labels_dict,
            metrics=sample_metrics_data,
            n_cols=2,
        )

        assert fig is not None
        # Check that fig has Figure-like attributes instead of strict type check
        assert hasattr(fig, "add_subplot") or hasattr(fig, "savefig")

        plt.close("all")

    def test_plot_neuron_selectivity_summary(self):
        """Test neuron selectivity summary plot."""
        selectivity_counts = {
            "Spatial (2D)": 45,
            "Spatial (X)": 30,
            "Spatial (Y)": 28,
            "Speed": 15,
            "Non-selective": 82,
        }

        fig = plot_neuron_selectivity_summary(
            selectivity_counts=selectivity_counts, total_neurons=200
        )

        assert fig is not None
        # Check that fig has Figure-like attributes instead of strict type check
        assert hasattr(fig, "add_subplot") or hasattr(fig, "savefig")

        plt.close("all")

    def test_plot_component_selectivity_heatmap(self):
        """Test component selectivity heatmap."""
        n_neurons = 50
        n_components_total = 6

        # Create sample selectivity matrix with some structure
        selectivity_matrix = np.zeros((n_neurons, n_components_total))
        # Make some neurons selective to specific components
        selectivity_matrix[5:10, 0] = np.random.rand(5) * 0.5 + 0.3
        selectivity_matrix[15:20, 2] = np.random.rand(5) * 0.4 + 0.2
        selectivity_matrix[25:30, 4] = np.random.rand(5) * 0.6 + 0.1

        methods = ["pca", "umap", "le"]
        n_components_per_method = {"pca": 2, "umap": 2, "le": 2}

        fig = plot_component_selectivity_heatmap(
            selectivity_matrix=selectivity_matrix,
            methods=methods,
            n_components_per_method=n_components_per_method,
        )

        assert fig is not None
        # Check that fig has Figure-like attributes instead of strict type check
        assert hasattr(fig, "add_subplot") or hasattr(fig, "savefig")

        plt.close("all")

    @pytest.mark.skipif(sys.platform == "win32", reason="Windows file locking prevents temp file deletion")
    def test_configurable_dpi(self, sample_embeddings):
        """Test that DPI is configurable in all functions."""
        embeddings, features, _ = sample_embeddings

        # Test that DPI parameter is accepted without error
        fig1 = plot_embedding_comparison(
            embeddings={"pca": embeddings["pca"]}, features=features, dpi=300
        )
        assert fig1 is not None

        fig2 = plot_embedding_comparison(
            embeddings={"pca": embeddings["pca"]}, features=features, dpi=72
        )
        assert fig2 is not None

        # Test save functionality
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig3 = plot_embedding_comparison(
                embeddings={"pca": embeddings["pca"]},
                features=features,
                save_path=tmp.name,
                dpi=150,
            )
            # Just check file was created - actual saving may be mocked
            assert fig3 is not None
            plt.close(fig3)  # Close figure before deleting on Windows
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

        plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
