"""
Tests for core RSA functions.
"""

import pytest
import numpy as np
from driada.rsa import core
from driada.dim_reduction.data import MVData
from driada.rsa.core import compute_rdm_unified


class TestComputeRDM:
    """Test RDM computation functions."""
    
    def test_compute_rdm_basic(self):
        """Test basic RDM computation."""
        # Create simple patterns
        patterns = np.array([
            [1, 0, 0],  # Pattern 1
            [0, 1, 0],  # Pattern 2
            [0, 0, 1],  # Pattern 3
            [1, 0, 0],  # Pattern 4 (same as 1)
        ])
        
        rdm = core.compute_rdm(patterns, metric='correlation')
        
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
        patterns = np.array([
            [1, 0, 0],  # Pattern 1
            [0, 1, 0],  # Pattern 2
            [0, 0, 1],  # Pattern 3
            [1, 0, 0],  # Pattern 4 (same as 1)
        ])
        
        # Create MVData object (transpose because MVData expects n_features x n_items)
        mvdata = MVData(patterns.T)
        
        rdm = core.compute_rdm(mvdata, metric='correlation')
        
        # Check shape
        assert rdm.shape == (4, 4)
        
        # Check diagonal is zero
        assert np.allclose(np.diag(rdm), 0)
        
        # Check that identical patterns have zero distance
        assert np.isclose(rdm[0, 3], 0)  # Patterns 1 and 4 are identical
    
    def test_compute_rdm_euclidean(self):
        """Test RDM with Euclidean distance."""
        patterns = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ])
        
        rdm = core.compute_rdm(patterns, metric='euclidean')
        
        # Check known distances
        assert np.isclose(rdm[0, 1], 1.0)  # Distance from (0,0) to (1,0)
        assert np.isclose(rdm[0, 3], np.sqrt(2))  # Distance from (0,0) to (1,1)
    
    def test_compute_rdm_from_timeseries_labels(self):
        """Test RDM computation from labeled time series."""
        # Create data with 3 features, 12 timepoints
        data = np.array([
            [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1],  # Feature 1
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # Feature 2
            [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],  # Feature 3
        ])
        
        # Labels for each timepoint
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0])
        
        rdm, unique_labels = core.compute_rdm_from_timeseries_labels(
            data, labels, metric='euclidean'
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
        trial_labels = np.array(['A', 'B', 'A', 'C', 'B'])
        
        rdm, unique_labels = core.compute_rdm_from_trials(
            data, trial_starts, trial_labels,
            trial_duration=20
        )
        
        # Check output
        assert rdm.shape == (3, 3)  # 3 unique labels (A, B, C)
        assert set(unique_labels) == {'A', 'B', 'C'}
        assert np.allclose(np.diag(rdm), 0)


class TestCompareRDMs:
    """Test RDM comparison functions."""
    
    def test_compare_rdms_identical(self):
        """Test comparing identical RDMs."""
        rdm = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0]
        ])
        
        similarity = core.compare_rdms(rdm, rdm)
        assert similarity == 1.0
    
    def test_compare_rdms_different(self):
        """Test comparing different RDMs."""
        rdm1 = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0]
        ])
        
        rdm2 = np.array([
            [0, 3, 1],
            [3, 0, 2],
            [1, 2, 0]
        ])
        
        # Spearman correlation
        sim_spearman = core.compare_rdms(rdm1, rdm2, method='spearman')
        assert -1 <= sim_spearman <= 1
        
        # Pearson correlation
        sim_pearson = core.compare_rdms(rdm1, rdm2, method='pearson')
        assert -1 <= sim_pearson <= 1
        
        # Cosine similarity
        sim_cosine = core.compare_rdms(rdm1, rdm2, method='cosine')
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
            data1, data2, labels, labels,
            n_bootstrap=50,  # Small for testing
            random_state=42
        )
        
        # Check results structure
        assert 'observed' in results
        assert 'bootstrap_distribution' in results
        assert 'p_value' in results
        assert 'ci_lower' in results
        assert 'ci_upper' in results
        
        # Check bootstrap distribution
        assert len(results['bootstrap_distribution']) == 50
        assert results['ci_lower'] <= results['observed'] <= results['ci_upper']


class TestUnifiedAPI:
    """Test the unified compute_rdm API."""
    
    def test_unified_with_numpy_patterns(self):
        """Test unified API with pre-averaged numpy patterns."""
        patterns = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        
        rdm, labels = compute_rdm_unified(patterns)
        
        assert rdm.shape == (3, 3)
        assert labels is None  # No labels when using direct patterns
        assert np.allclose(np.diag(rdm), 0)
    
    def test_unified_with_numpy_timeseries(self):
        """Test unified API with numpy time series and labels."""
        data = np.array([
            [1, 1, 1, 2, 2, 2, 3, 3, 3],  # Feature 1
            [0, 0, 0, 1, 1, 1, 0, 0, 0],  # Feature 2
        ])
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        
        rdm, unique_labels = compute_rdm_unified(data, labels)
        
        assert rdm.shape == (3, 3)
        assert np.array_equal(unique_labels, [0, 1, 2])
        assert np.allclose(np.diag(rdm), 0)
    
    def test_unified_with_mvdata(self):
        """Test unified API with MVData object."""
        patterns = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        mvdata = MVData(patterns.T)
        
        rdm, labels = compute_rdm_unified(mvdata)
        
        assert rdm.shape == (3, 3)
        assert labels is None
        assert np.allclose(np.diag(rdm), 0)
    
    def test_unified_with_trial_structure(self):
        """Test unified API with trial structure."""
        data = np.random.randn(5, 100)
        trial_info = {
            'trial_starts': [0, 20, 40, 60, 80],
            'trial_labels': ['A', 'B', 'A', 'C', 'B']
        }
        
        rdm, labels = compute_rdm_unified(data, trial_info)
        
        assert rdm.shape == (3, 3)  # 3 unique labels
        assert set(labels) == {'A', 'B', 'C'}
        assert np.allclose(np.diag(rdm), 0)