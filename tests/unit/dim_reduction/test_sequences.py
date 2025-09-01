"""Tests for dimensionality reduction sequences."""

import pytest
import numpy as np
from driada.dim_reduction import MVData, dr_sequence


class TestDRSequence:
    """Test cases for dr_sequence function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample MVData for testing."""
        # 100 features, 500 samples
        data = np.random.randn(100, 500)
        return MVData(data)

    def test_single_step_sequence(self, sample_data):
        """Test sequence with a single reduction step."""
        # Single PCA step
        embedding = dr_sequence(sample_data, steps=["pca"])

        assert embedding is not None
        assert embedding.coords.shape[0] == 2  # Default dim is 2
        assert embedding.coords.shape[1] == 500  # Same number of samples

    def test_two_step_sequence(self, sample_data):
        """Test sequence with two reduction steps."""
        steps = [("pca", {"dim": 50}), ("pca", {"dim": 3})]
        embedding = dr_sequence(sample_data, steps=steps)

        assert embedding.coords.shape == (3, 500)

    def test_mixed_methods_sequence(self, sample_data):
        """Test sequence with different reduction methods."""
        steps = [
            ("pca", {"dim": 30}),
            ("lle", {"dim": 10, "n_neighbors": 15}),
            ("pca", {"dim": 2}),
        ]
        embedding = dr_sequence(sample_data, steps=steps)

        assert embedding.coords.shape[0] == 2
        # LLE may filter nodes, so we can have <= 500 points
        assert embedding.coords.shape[1] <= 500

    def test_default_params(self, sample_data):
        """Test sequence using default parameters."""
        # All use default parameters
        steps = ["pca", "pca"]
        embedding = dr_sequence(sample_data, steps=steps)

        assert embedding.coords.shape == (2, 500)

    def test_mixed_format_steps(self, sample_data):
        """Test sequence with mixed step formats."""
        steps = [
            ("pca", {"dim": 20}),  # Tuple format
            "pca",  # String format (defaults to dim=2)
        ]
        embedding = dr_sequence(sample_data, steps=steps)

        assert embedding.coords.shape == (2, 500)

    def test_empty_steps_error(self, sample_data):
        """Test that empty steps list raises error."""
        with pytest.raises(ValueError, match="At least one reduction step"):
            dr_sequence(sample_data, steps=[])

    def test_invalid_step_format(self, sample_data):
        """Test that invalid step format raises error."""
        with pytest.raises(ValueError, match="Invalid step format"):
            dr_sequence(sample_data, steps=[123])  # Invalid: not string or tuple

    def test_invalid_tuple_format(self, sample_data):
        """Test that invalid tuple format raises error."""
        with pytest.raises(ValueError, match="Invalid step format"):
            dr_sequence(
                sample_data, steps=[("pca", {"dim": 2}, "extra")]
            )  # Too many elements

    def test_preserves_labels(self):
        """Test that labels are preserved through the sequence."""
        # Create data with labels
        data = np.random.randn(50, 200)
        labels = ["sample_{}".format(i) for i in range(200)]
        mvdata = MVData(data, labels=labels)

        steps = [("pca", {"dim": 20}), ("pca", {"dim": 2})]
        embedding = dr_sequence(mvdata, steps=steps)

        # Check labels are preserved (accounting for potential node filtering)
        mvdata_final = embedding.to_mvdata()
        # Labels may be filtered if graph methods removed nodes
        assert len(mvdata_final.labels) == mvdata_final.n_points
        assert len(mvdata_final.labels) <= len(labels)

    def test_large_sequence(self, sample_data):
        """Test a longer sequence of reductions."""
        steps = [
            ("pca", {"dim": 80}),
            ("pca", {"dim": 60}),
            ("pca", {"dim": 40}),
            ("pca", {"dim": 20}),
            ("pca", {"dim": 10}),
            ("pca", {"dim": 3}),
        ]
        embedding = dr_sequence(sample_data, steps=steps)

        assert embedding.coords.shape == (3, 500)

    @pytest.mark.parametrize(
        "method,params",
        [
            ("pca", {"dim": 5}),
            ("lle", {"dim": 3, "n_neighbors": 10}),
            ("le", {"dim": 4, "n_neighbors": 20}),
        ],
    )
    def test_various_methods(self, sample_data, method, params):
        """Test sequence with various dimensionality reduction methods."""
        steps = [
            ("pca", {"dim": 50}),  # First reduce to 50
            (method, params),  # Then apply the test method
        ]
        embedding = dr_sequence(sample_data, steps=steps)

        assert embedding.coords.shape[0] == params["dim"]
        # Graph methods may filter nodes
        assert embedding.coords.shape[1] <= 500
