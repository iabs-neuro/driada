"""
Tests for RSA visualization functions.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from driada.rsa.visual import plot_rdm, plot_rdm_comparison


class TestPlotRDM:
    """Test RDM plotting function."""
    
    def test_plot_rdm_basic(self):
        """Test basic RDM plotting without errors."""
        rdm = np.random.rand(10, 10)
        rdm = (rdm + rdm.T) / 2  # Make symmetric
        np.fill_diagonal(rdm, 0)
        
        # Should not raise any errors
        fig = plot_rdm(rdm)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_rdm_with_labels(self):
        """Test RDM plotting with custom labels."""
        rdm = np.random.rand(5, 5)
        rdm = (rdm + rdm.T) / 2  # Make symmetric
        np.fill_diagonal(rdm, 0)
        
        labels = ["A", "B", "C", "D", "E"]
        
        # Should not raise errors with labels
        fig = plot_rdm(rdm, labels=labels, title="Test RDM")
        assert fig is not None
        plt.close(fig)
    
    def test_plot_rdm_with_values(self):
        """Test RDM plotting with values shown."""
        # Small RDM for value display
        rdm = np.array([
            [0.0, 0.5, 0.8],
            [0.5, 0.0, 0.3],
            [0.8, 0.3, 0.0],
        ])
        
        # Should not raise errors with show_values
        fig = plot_rdm(rdm, show_values=True)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_rdm_no_dendrogram(self):
        """Test RDM plotting without dendrogram."""
        rdm = np.eye(4)
        
        # Should not raise errors without dendrogram
        fig = plot_rdm(rdm, dendrogram_ratio=0)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_rdm_existing_axes(self):
        """Test plotting on existing axes."""
        rdm = np.random.rand(4, 4)
        rdm = (rdm + rdm.T) / 2
        np.fill_diagonal(rdm, 0)
        
        # Create figure and axes
        fig, ax = plt.subplots()
        
        # Plot on existing axes
        returned_fig = plot_rdm(rdm, ax=ax)
        assert returned_fig is fig
        plt.close(fig)
    
    def test_plot_rdm_custom_colormap(self):
        """Test custom colormap."""
        rdm = np.random.rand(5, 5)
        rdm = (rdm + rdm.T) / 2
        np.fill_diagonal(rdm, 0)
        
        # Should work with custom colormap
        fig = plot_rdm(rdm, cmap="viridis", cbar_label="Custom Distance")
        assert fig is not None
        plt.close(fig)
    
    def test_plot_rdm_edge_cases(self):
        """Test edge cases."""
        # All zeros
        rdm = np.zeros((5, 5))
        fig = plot_rdm(rdm)
        assert fig is not None
        plt.close(fig)
        
        # All ones (except diagonal)
        rdm = np.ones((5, 5))
        np.fill_diagonal(rdm, 0)
        fig = plot_rdm(rdm)
        assert fig is not None
        plt.close(fig)


class TestPlotRDMComparison:
    """Test RDM comparison plotting."""
    
    def test_plot_comparison_basic(self):
        """Test basic comparison plotting."""
        # Create three RDMs
        rdms = [
            np.random.rand(5, 5),
            np.random.rand(5, 5),
            np.random.rand(5, 5),
        ]
        
        # Make symmetric
        for i in range(3):
            rdms[i] = (rdms[i] + rdms[i].T) / 2
            np.fill_diagonal(rdms[i], 0)
        
        # Should not raise errors
        fig = plot_rdm_comparison(rdms)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_comparison_with_titles(self):
        """Test comparison with custom titles."""
        rdms = [np.eye(4), np.ones((4, 4)) - np.eye(4)]
        titles = ["Identity", "Anti-Identity"]
        
        fig = plot_rdm_comparison(rdms, titles=titles)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_comparison_with_labels(self):
        """Test comparison with item labels."""
        rdms = [
            np.array([[0, 0.5], [0.5, 0]]),
            np.array([[0, 1], [1, 0]]),
        ]
        labels = ["Item1", "Item2"]
        
        fig = plot_rdm_comparison(rdms, labels=labels)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_comparison_single_rdm(self):
        """Test with single RDM."""
        rdm = np.random.rand(3, 3)
        rdm = (rdm + rdm.T) / 2
        np.fill_diagonal(rdm, 0)
        
        fig = plot_rdm_comparison([rdm])
        assert fig is not None
        plt.close(fig)
    
    def test_plot_comparison_different_sizes_error(self):
        """Test that different sized RDMs raise error."""
        rdms = [
            np.zeros((3, 3)),
            np.zeros((4, 4)),  # Different size
        ]
        
        with pytest.raises(ValueError, match="All RDMs must have the same shape"):
            plot_rdm_comparison(rdms)
    
    def test_plot_comparison_custom_figure_size(self):
        """Test custom figure size."""
        rdms = [np.eye(5) for _ in range(4)]
        
        # Should handle figsize parameter
        fig = plot_rdm_comparison(rdms, figsize=(12, 8))
        assert fig is not None
        plt.close(fig)
    
    def test_plot_comparison_small_rdms(self):
        """Test comparison with small RDMs."""
        # Small RDMs 
        rdms = [
            np.array([[0, 0.5], [0.5, 0]]),
            np.array([[0, 1], [1, 0]]),
        ]
        
        fig = plot_rdm_comparison(rdms)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_comparison_custom_colormap(self):
        """Test custom colormap in comparison."""
        rdms = [np.random.rand(4, 4) for _ in range(2)]
        for rdm in rdms:
            rdm[:] = (rdm + rdm.T) / 2
            np.fill_diagonal(rdm, 0)
        
        fig = plot_rdm_comparison(rdms, cmap="coolwarm")
        assert fig is not None
        plt.close(fig)


class TestPlotIntegration:
    """Test integration with actual RSA computations."""
    
    def test_plot_from_correlation_rdm(self):
        """Test plotting RDM from correlation distance computation."""
        from driada.rsa.core import compute_rdm
        
        # Create patterns
        patterns = np.random.randn(10, 50)  # 10 conditions, 50 features
        
        # Compute RDM
        rdm = compute_rdm(patterns, metric="correlation")
        
        # Should plot without errors
        fig = plot_rdm(rdm, title="Correlation RDM")
        assert fig is not None
        plt.close(fig)
    
    def test_plot_multiple_metrics(self):
        """Test plotting RDMs from different metrics."""
        from driada.rsa.core import compute_rdm
        
        patterns = np.random.randn(8, 20)
        
        rdms = []
        titles = []
        for metric in ["euclidean", "correlation", "manhattan"]:
            rdm = compute_rdm(patterns, metric=metric)
            rdms.append(rdm)
            titles.append(metric.capitalize())
        
        fig = plot_rdm_comparison(rdms, titles=titles)
        assert fig is not None
        plt.close(fig)