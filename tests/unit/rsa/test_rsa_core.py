"""
Tests for core RSA functions.
"""

import pytest
import numpy as np
from driada.rsa import core
from driada.dim_reduction.data import MVData
from driada.rsa.core import compute_rdm_unified, rsa_compare
import driada


class TestComputeRDM:
    """Test RDM computation functions."""

    def test_compute_rdm_basic(self):
        """Test basic RDM computation."""
        # Create simple patterns
        patterns = np.array(
            [
                [1, 0, 0],  # Pattern 1
                [0, 1, 0],  # Pattern 2
                [0, 0, 1],  # Pattern 3
                [1, 0, 0],  # Pattern 4 (same as 1)
            ]
        )

        rdm = core.compute_rdm(patterns, metric="correlation")

        # Check shape
        assert rdm.shape == (4, 4)

        # Check diagonal is zero
        assert np.allclose(np.diag(rdm), 0)

        # Check symmetry
        assert np.allclose(rdm, rdm.T)

        # Check that identical patterns have zero distance
        assert np.isclose(rdm[0, 3], 0)  # Patterns 1 and 4 are identical

        # Check that orthogonal patterns have high distance
        assert rdm[0, 1] > 0.9  # Patterns 1 and 2 are orthogonal

    def test_compute_rdm_with_mvdata(self):
        """Test RDM computation with MVData input."""
        # Create simple patterns
        patterns = np.array(
            [
                [1, 0, 0],  # Pattern 1
                [0, 1, 0],  # Pattern 2
                [0, 0, 1],  # Pattern 3
                [1, 0, 0],  # Pattern 4 (same as 1)
            ]
        )

        # Create MVData object (transpose because MVData expects n_features x n_items)
        mvdata = MVData(patterns.T)

        rdm = core.compute_rdm(mvdata, metric="correlation")

        # Check shape
        assert rdm.shape == (4, 4)

        # Check diagonal is zero
        assert np.allclose(np.diag(rdm), 0)

        # Check that identical patterns have zero distance
        assert np.isclose(rdm[0, 3], 0)  # Patterns 1 and 4 are identical

    def test_compute_rdm_euclidean(self):
        """Test RDM with Euclidean distance."""
        patterns = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
            ]
        )

        rdm = core.compute_rdm(patterns, metric="euclidean")

        # Check known distances
        assert np.isclose(rdm[0, 1], 1.0)  # Distance from (0,0) to (1,0)
        assert np.isclose(rdm[0, 3], np.sqrt(2))  # Distance from (0,0) to (1,1)

    def test_compute_rdm_from_timeseries_labels(self):
        """Test RDM computation from labeled time series."""
        # Create data with 3 features, 12 timepoints
        data = np.array(
            [
                [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1],  # Feature 1
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # Feature 2
                [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],  # Feature 3
            ]
        )

        # Labels for each timepoint
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0])

        rdm, unique_labels = core.compute_rdm_from_timeseries_labels(
            data, labels, metric="euclidean"
        )

        # Check output
        assert rdm.shape == (3, 3)  # 3 unique labels
        assert np.array_equal(unique_labels, [0, 1, 2])

        # Check that same conditions averaged correctly
        # Condition 0 appears twice, should be averaged
        assert np.allclose(np.diag(rdm), 0)

    def test_compute_rdm_from_trials(self):
        """Test RDM computation from trial structure."""
        # Create data
        data = np.random.randn(5, 100)  # 5 features, 100 timepoints

        # Define trials
        trial_starts = np.array([0, 20, 40, 60, 80])
        trial_labels = np.array(["A", "B", "A", "C", "B"])

        rdm, unique_labels = core.compute_rdm_from_trials(
            data, trial_starts, trial_labels, trial_duration=20
        )

        # Check output
        assert rdm.shape == (3, 3)  # 3 unique labels (A, B, C)
        assert set(unique_labels) == {"A", "B", "C"}
        assert np.allclose(np.diag(rdm), 0)


class TestCompareRDMs:
    """Test RDM comparison functions."""

    def test_compare_rdms_identical(self):
        """Test comparing identical RDMs."""
        rdm = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])

        similarity = core.compare_rdms(rdm, rdm)
        assert similarity == 1.0

    def test_compare_rdms_different(self):
        """Test comparing different RDMs."""
        rdm1 = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])

        rdm2 = np.array([[0, 3, 1], [3, 0, 2], [1, 2, 0]])

        # Spearman correlation
        sim_spearman = core.compare_rdms(rdm1, rdm2, method="spearman")
        assert -1 <= sim_spearman <= 1

        # Pearson correlation
        sim_pearson = core.compare_rdms(rdm1, rdm2, method="pearson")
        assert -1 <= sim_pearson <= 1

        # Cosine similarity
        sim_cosine = core.compare_rdms(rdm1, rdm2, method="cosine")
        assert -1 <= sim_cosine <= 1

    def test_compare_rdms_wrong_shape(self):
        """Test error when comparing RDMs of different shapes."""
        rdm1 = np.zeros((3, 3))
        rdm2 = np.zeros((4, 4))

        with pytest.raises(ValueError, match="same shape"):
            core.compare_rdms(rdm1, rdm2)


class TestBootstrap:
    """Test bootstrap functions."""

    def test_bootstrap_rdm_comparison(self):
        """Test bootstrap significance testing."""
        # Create two datasets with known structure
        n_features = 10
        n_timepoints = 100

        # Create correlated data
        data1 = np.random.randn(n_features, n_timepoints)
        data2 = data1 + 0.5 * np.random.randn(n_features, n_timepoints)

        # Create labels
        labels = np.repeat([0, 1, 2, 3], 25)

        # Run bootstrap
        results = core.bootstrap_rdm_comparison(
            data1,
            data2,
            labels,
            labels,
            n_bootstrap=50,  # Small for testing
            random_state=42,
        )

        # Check results structure
        assert "observed" in results
        assert "bootstrap_distribution" in results
        assert "p_value" in results
        assert "ci_lower" in results
        assert "ci_upper" in results

        # Check bootstrap distribution
        assert len(results["bootstrap_distribution"]) == 50
        assert results["ci_lower"] <= results["observed"] <= results["ci_upper"]


class TestUnifiedAPI:
    """Test the unified compute_rdm API."""

    def test_unified_with_numpy_patterns(self):
        """Test unified API with pre-averaged numpy patterns."""
        patterns = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )

        rdm, labels = compute_rdm_unified(patterns)

        assert rdm.shape == (3, 3)
        assert labels is None  # No labels when using direct patterns
        assert np.allclose(np.diag(rdm), 0)

    def test_unified_with_numpy_timeseries(self):
        """Test unified API with numpy time series and labels."""
        data = np.array(
            [
                [1, 1, 1, 2, 2, 2, 3, 3, 3],  # Feature 1
                [0, 0, 0, 1, 1, 1, 0, 0, 0],  # Feature 2
            ]
        )
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        rdm, unique_labels = compute_rdm_unified(data, labels)

        assert rdm.shape == (3, 3)
        assert np.array_equal(unique_labels, [0, 1, 2])
        assert np.allclose(np.diag(rdm), 0)

    def test_unified_with_mvdata(self):
        """Test unified API with MVData object."""
        patterns = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
        mvdata = MVData(patterns.T)

        rdm, labels = compute_rdm_unified(mvdata)

        assert rdm.shape == (3, 3)
        assert labels is None
        assert np.allclose(np.diag(rdm), 0)

    def test_unified_with_trial_structure(self):
        """Test unified API with trial structure."""
        data = np.random.randn(5, 100)
        trial_info = {
            "trial_starts": [0, 20, 40, 60, 80],
            "trial_labels": ["A", "B", "A", "C", "B"],
        }

        rdm, labels = compute_rdm_unified(data, trial_info)

        assert rdm.shape == (3, 3)  # 3 unique labels
        assert set(labels) == {"A", "B", "C"}
        assert np.allclose(np.diag(rdm), 0)


class TestJITFunctions:
    """Test JIT-compiled functions when enabled."""

    def test_jit_euclidean_distance(self):
        """Test JIT euclidean distance computation."""
        from driada.utils.jit import is_jit_enabled

        if not is_jit_enabled():
            pytest.skip("JIT compilation is disabled")

        # Create patterns
        patterns = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
            ]
        )

        # Compute RDM with euclidean metric (should use JIT)
        rdm = core.compute_rdm(patterns, metric="euclidean")

        # Check known distances
        assert np.isclose(rdm[0, 1], 1.0)  # Distance from (0,0) to (1,0)
        assert np.isclose(rdm[0, 3], np.sqrt(2))  # Distance from (0,0) to (1,1)
        assert np.allclose(np.diag(rdm), 0)

    def test_jit_manhattan_distance(self):
        """Test JIT manhattan distance computation."""
        from driada.utils.jit import is_jit_enabled

        if not is_jit_enabled():
            pytest.skip("JIT compilation is disabled")

        # Create patterns
        patterns = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
            ]
        )

        # Compute RDM with manhattan metric (should use JIT)
        rdm = core.compute_rdm(patterns, metric="manhattan")

        # Check known distances
        assert np.isclose(rdm[0, 1], 1.0)  # Manhattan distance from (0,0) to (1,0)
        assert np.isclose(rdm[0, 3], 2.0)  # Manhattan distance from (0,0) to (1,1)
        assert np.isclose(rdm[1, 2], 2.0)  # Manhattan distance from (1,0) to (0,1)
        assert np.allclose(np.diag(rdm), 0)

    def test_jit_average_patterns(self):
        """Test JIT-compiled pattern averaging."""
        from driada.utils.jit import is_jit_enabled

        if not is_jit_enabled():
            pytest.skip("JIT compilation is disabled")

        # Create data with clear pattern
        data = np.array(
            [
                [1, 1, 2, 2, 3, 3],  # Feature 1
                [0, 0, 1, 1, 0, 0],  # Feature 2
            ]
        )
        labels = np.array([0, 0, 1, 1, 2, 2])

        # This should use JIT-compiled averaging
        rdm, unique_labels = core.compute_rdm_from_timeseries_labels(
            data, labels, metric="euclidean", average_method="mean"
        )

        assert rdm.shape == (3, 3)
        assert np.array_equal(unique_labels, [0, 1, 2])
        assert np.allclose(np.diag(rdm), 0)


class TestMVDataMetrics:
    """Test MVData with different metrics."""

    def test_mvdata_euclidean_metric(self):
        """Test MVData with euclidean metric."""
        patterns = np.random.randn(10, 20)
        mvdata = MVData(patterns.T)

        rdm = core.compute_rdm(mvdata, metric="euclidean")

        assert rdm.shape == (10, 10)
        assert np.allclose(np.diag(rdm), 0)
        assert np.all(rdm >= 0)

    def test_mvdata_cosine_metric(self):
        """Test MVData with cosine metric."""
        patterns = np.random.randn(8, 15)
        mvdata = MVData(patterns.T)

        rdm = core.compute_rdm(mvdata, metric="cosine")

        assert rdm.shape == (8, 8)
        assert np.allclose(np.diag(rdm), 0)
        assert np.all(rdm >= 0)

    def test_mvdata_cityblock_metric(self):
        """Test MVData with cityblock metric (manhattan)."""
        patterns = np.random.randn(5, 10)
        mvdata = MVData(patterns.T)

        # MVData uses 'cityblock' instead of 'manhattan'
        rdm = core.compute_rdm(mvdata, metric="cityblock")

        assert rdm.shape == (5, 5)
        assert np.allclose(np.diag(rdm), 0)
        assert np.all(rdm >= 0)


class TestMedianAveraging:
    """Test median averaging method."""

    def test_median_averaging_timeseries(self):
        """Test median averaging for time series."""
        # Create data with outliers
        data = np.array(
            [
                [1, 1, 10, 2, 2, 20, 3, 3, 30],  # Feature with outliers
                [0, 0, 0, 1, 1, 1, 0, 0, 0],  # Normal feature
            ]
        )
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        # Test with median (should be robust to outliers)
        rdm_median, _ = core.compute_rdm_from_timeseries_labels(
            data, labels, metric="euclidean", average_method="median"
        )

        # Test with mean (should be affected by outliers)
        rdm_mean, _ = core.compute_rdm_from_timeseries_labels(
            data, labels, metric="euclidean", average_method="mean"
        )

        # Median should give different result due to outliers
        assert not np.allclose(rdm_median, rdm_mean)
        assert rdm_median.shape == (3, 3)
        assert np.allclose(np.diag(rdm_median), 0)

    def test_median_averaging_trials(self):
        """Test median averaging for trials."""
        data = np.random.randn(5, 100)

        # Add outliers to some trials
        data[:, 10:20] *= 10  # Outliers in first trial
        data[:, 50:60] *= 10  # Outliers in third trial

        trial_starts = np.array([0, 20, 40, 60, 80])
        trial_labels = np.array(["A", "B", "A", "C", "B"])

        # Test with median
        rdm_median, _ = core.compute_rdm_from_trials(
            data, trial_starts, trial_labels, trial_duration=20, average_method="median"
        )

        # Test with mean
        rdm_mean, _ = core.compute_rdm_from_trials(
            data, trial_starts, trial_labels, trial_duration=20, average_method="mean"
        )

        # Results should differ due to outliers
        assert not np.allclose(rdm_median, rdm_mean)
        assert rdm_median.shape == (3, 3)

    def test_invalid_average_method(self):
        """Test error for invalid average method."""
        data = np.random.randn(3, 10)
        labels = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1, 1])

        with pytest.raises(ValueError, match="Unknown average method"):
            core.compute_rdm_from_timeseries_labels(
                data, labels, average_method="invalid"
            )

        # Also test for trials
        trial_starts = [0, 5]
        trial_labels = ["A", "B"]

        with pytest.raises(ValueError, match="Unknown average method"):
            core.compute_rdm_from_trials(
                data, trial_starts, trial_labels, average_method="invalid"
            )


class TestEmbeddingSupport:
    """Test RSA with Embedding objects."""

    def test_compute_rdm_unified_with_embedding(self):
        """Test unified API with Embedding object."""
        from unittest.mock import Mock

        # Create mock embedding with known structure
        coords = np.array(
            [
                [1, 0, 1, 0],  # Dim 1
                [0, 1, 0, 1],  # Dim 2
            ]
        )

        # Create mock MVData that will be returned by to_mvdata()
        mock_mvdata = MVData(coords)

        # Create mock embedding
        mock_embedding = Mock()
        mock_embedding.to_mvdata = Mock(return_value=mock_mvdata)

        # Make isinstance work
        from driada.dim_reduction.embedding import Embedding

        mock_embedding.__class__ = Embedding

        # Compute RDM
        rdm, labels = core.compute_rdm_unified(mock_embedding)

        assert rdm.shape == (4, 4)  # 4 samples
        assert labels is None  # No labels for direct embedding
        assert np.allclose(np.diag(rdm), 0)
        mock_embedding.to_mvdata.assert_called_once()

    def test_rsa_compare_with_embeddings(self):
        """Test rsa_compare with Embedding objects."""
        from unittest.mock import Mock

        # Create two embeddings with known similarity
        coords1 = np.random.randn(3, 10)  # 3D embedding, 10 samples
        coords2 = coords1 + 0.1 * np.random.randn(3, 10)  # Similar but not identical

        # Create mock MVData objects
        mock_mvdata1 = MVData(coords1)
        mock_mvdata2 = MVData(coords2)

        # Create mock embeddings
        mock_embedding1 = Mock()
        mock_embedding1.to_mvdata = Mock(return_value=mock_mvdata1)
        mock_embedding2 = Mock()
        mock_embedding2.to_mvdata = Mock(return_value=mock_mvdata2)

        # Make isinstance work
        from driada.dim_reduction.embedding import Embedding

        mock_embedding1.__class__ = Embedding
        mock_embedding2.__class__ = Embedding

        # Compare embeddings
        similarity = core.rsa_compare(mock_embedding1, mock_embedding2)

        assert isinstance(similarity, float)
        assert -1 <= similarity <= 1
        assert similarity > 0.5  # Should be similar
        mock_embedding1.to_mvdata.assert_called_once()
        mock_embedding2.to_mvdata.assert_called_once()

    def test_embedding_with_labels_error(self):
        """Test that trial structure is not supported for embeddings."""
        from unittest.mock import Mock

        coords = np.random.randn(2, 10)
        mock_mvdata = MVData(coords)

        # Create mock embedding
        mock_embedding = Mock()
        mock_embedding.to_mvdata = Mock(return_value=mock_mvdata)

        # Make isinstance work
        from driada.dim_reduction.embedding import Embedding

        mock_embedding.__class__ = Embedding

        trial_info = {"trial_starts": [0, 5], "trial_labels": ["A", "B"]}

        with pytest.raises(ValueError, match="Trial structure not supported"):
            core.compute_rdm_unified(mock_embedding, items=trial_info)


class TestRSACompare:
    """Test the simplified rsa_compare function."""

    def test_rsa_compare_arrays(self):
        """Test rsa_compare with numpy arrays."""
        # Create two datasets with some variance
        np.random.seed(42)
        patterns1 = np.random.randn(5, 10)  # 5 items, 10 features
        patterns2 = patterns1.copy()  # Identical copy

        # Identical patterns should have similarity 1.0
        similarity = rsa_compare(patterns1, patterns2)
        assert np.isclose(similarity, 1.0)

        # Different patterns should have lower similarity
        patterns3 = np.random.randn(5, 10)
        similarity = rsa_compare(patterns1, patterns3)
        assert similarity < 1.0
        assert similarity > -1.0  # Valid correlation range

    def test_rsa_compare_mvdata(self):
        """Test rsa_compare with MVData objects."""
        # Create MVData objects (remember: n_features x n_items)
        data1 = np.random.randn(5, 10)  # 5 features, 10 items
        data2 = data1 + 0.1 * np.random.randn(5, 10)  # Similar but not identical

        mvdata1 = MVData(data1)
        mvdata2 = MVData(data2)

        similarity = rsa_compare(mvdata1, mvdata2)
        assert isinstance(similarity, float)
        assert -1 <= similarity <= 1
        assert similarity > 0.5  # Should be highly similar

    def test_rsa_compare_experiments(self):
        """Test rsa_compare with Experiment objects."""
        # Create two synthetic experiments with longer duration
        exp1 = driada.generate_synthetic_exp(
            n_dfeats=0,
            n_cfeats=3,
            nneurons=10,
            duration=30,  # Longer duration for stable shuffle mask
            seed=42,
        )
        exp2 = driada.generate_synthetic_exp(
            n_dfeats=0, n_cfeats=3, nneurons=10, duration=30, seed=43
        )

        # Add stimulus labels to both
        n_timepoints = exp1.calcium.data.shape[1]
        labels = np.repeat([0, 1, 2], n_timepoints // 3 + 1)[:n_timepoints]
        exp1.dynamic_features["stimulus"] = driada.TimeSeries(labels)
        exp2.dynamic_features["stimulus"] = driada.TimeSeries(labels)

        # Compare experiments
        similarity = rsa_compare(exp1, exp2, items="stimulus")
        assert isinstance(similarity, float)
        assert -1 <= similarity <= 1

    def test_rsa_compare_experiments_no_items_error(self):
        """Test that comparing experiments without items raises error."""
        exp1 = driada.generate_synthetic_exp(
            n_dfeats=0, n_cfeats=2, nneurons=5, duration=30
        )
        exp2 = driada.generate_synthetic_exp(
            n_dfeats=0, n_cfeats=2, nneurons=5, duration=30
        )

        with pytest.raises(ValueError, match="items must be specified"):
            rsa_compare(exp1, exp2)

    def test_rsa_compare_mixed_types_error(self):
        """Test that mixing data types raises error."""
        exp = driada.generate_synthetic_exp(
            n_dfeats=0, n_cfeats=2, nneurons=5, duration=30
        )
        array = np.random.randn(5, 10)

        with pytest.raises(
            ValueError, match="Cannot compare Experiment with non-Experiment"
        ):
            rsa_compare(exp, array)

        with pytest.raises(
            ValueError, match="Cannot compare Experiment with non-Experiment"
        ):
            rsa_compare(array, exp)

    def test_rsa_compare_different_metrics(self):
        """Test rsa_compare with different distance metrics."""
        patterns1 = np.random.randn(10, 20)
        patterns2 = np.random.randn(10, 20)

        # Test different metrics
        for metric in ["correlation", "euclidean", "cosine"]:
            similarity = rsa_compare(patterns1, patterns2, metric=metric)
            assert isinstance(similarity, float)
            assert -1 <= similarity <= 1

    def test_rsa_compare_different_comparisons(self):
        """Test rsa_compare with different comparison methods."""
        patterns1 = np.random.randn(10, 20)
        patterns2 = patterns1 + 0.1 * np.random.randn(10, 20)

        # Test different comparison methods
        for comparison in ["spearman", "pearson", "kendall"]:
            similarity = rsa_compare(patterns1, patterns2, comparison=comparison)
            assert isinstance(similarity, float)
            assert -1 <= similarity <= 1
            assert similarity > 0.5  # Should be similar
