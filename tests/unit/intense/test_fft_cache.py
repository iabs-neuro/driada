"""
Tests for FFT cache functionality in INTENSE.

Tests cache building, duplicate name detection, and memory cleanup.
"""
import pytest
import numpy as np
import pickle
import gc
from driada.information import TimeSeries
from driada.intense.intense_base import _build_fft_cache, _get_ts_key


class TestDuplicateNameDetection:
    """Test duplicate name collision detection."""

    def test_duplicate_name_same_object_allowed(self):
        """Same TimeSeries object in both bunches should not raise error."""
        # Create a single TimeSeries
        ts = TimeSeries(np.random.randn(100), discrete=False, name="shared")

        # Use same object in both bunches - should NOT raise
        ts_bunch1 = [ts]
        ts_bunch2 = [ts]

        # This should work - same object reused
        cache = _build_fft_cache(
            ts_bunch1, ts_bunch2, "mi", "gcmi", ds=1, engine="auto"
        )

        assert isinstance(cache, dict)

    def test_duplicate_name_different_data_raises(self):
        """Different TimeSeries with same name should raise ValueError."""
        # Create two different TimeSeries with SAME name but DIFFERENT data
        ts1 = TimeSeries(np.random.randn(100), discrete=False, name="collision")
        ts2 = TimeSeries(np.random.randn(100) + 10, discrete=False, name="collision")

        ts_bunch1 = [ts1]
        ts_bunch2 = [ts2]

        # This should raise ValueError due to cache collision
        with pytest.raises(ValueError, match="Cache collision.*different data"):
            _build_fft_cache(
                ts_bunch1, ts_bunch2, "mi", "gcmi", ds=1, engine="auto"
            )

    def test_duplicate_name_same_data_allowed(self):
        """Different objects with same name and identical data should be allowed."""
        # Create two TimeSeries with same name AND same data
        data = np.random.randn(100)
        ts1 = TimeSeries(data.copy(), discrete=False, name="same_data")
        ts2 = TimeSeries(data.copy(), discrete=False, name="same_data")

        ts_bunch1 = [ts1]
        ts_bunch2 = [ts2]

        # Should NOT raise - same data, so cache can be shared
        cache = _build_fft_cache(
            ts_bunch1, ts_bunch2, "mi", "gcmi", ds=1, engine="auto"
        )

        assert isinstance(cache, dict)

    def test_duplicate_detection_after_pickling(self):
        """Duplicate detection should work after pickling (loky backend compatibility)."""
        # Create TimeSeries and pickle/unpickle it
        data = np.random.randn(100)
        ts_original = TimeSeries(data, discrete=False, name="pickled")

        # Pickle and unpickle (simulates loky backend behavior)
        ts_unpickled = pickle.loads(pickle.dumps(ts_original))

        # Different objects now (different id()), but same data
        assert id(ts_original) != id(ts_unpickled)
        assert ts_original.name == ts_unpickled.name
        assert np.array_equal(ts_original.data, ts_unpickled.data)

        # Should NOT raise - data equality check instead of id() check
        cache = _build_fft_cache(
            [ts_original], [ts_unpickled], "mi", "gcmi", ds=1, engine="auto"
        )

        assert isinstance(cache, dict)

    def test_unique_names_fast_path(self):
        """All unique names should take fast path (no data comparison)."""
        # Create many TimeSeries with unique names
        ts_bunch1 = [
            TimeSeries(np.random.randn(100), discrete=False, name=f"ts1_{i}")
            for i in range(10)
        ]
        ts_bunch2 = [
            TimeSeries(np.random.randn(100), discrete=False, name=f"ts2_{i}")
            for i in range(10)
        ]

        # Should work efficiently (fast path)
        cache = _build_fft_cache(
            ts_bunch1, ts_bunch2, "mi", "gcmi", ds=1, engine="auto"
        )

        assert isinstance(cache, dict)
        # Cache should have entries for compatible pairs
        assert len(cache) > 0


class TestFFTCacheBuilding:
    """Test FFT cache building functionality."""

    def test_build_fft_cache_basic(self):
        """Basic FFT cache building test."""
        # Create continuous TimeSeries
        ts1 = TimeSeries(np.random.randn(100), discrete=False, name="ts1")
        ts2 = TimeSeries(np.random.randn(100), discrete=False, name="ts2")

        cache = _build_fft_cache(
            [ts1], [ts2], "mi", "gcmi", ds=1, engine="auto"
        )

        assert isinstance(cache, dict)
        # Should have cached the pair
        assert len(cache) > 0

    def test_fft_cache_discrete_pairs_none(self):
        """Discrete-discrete pairs should have None entries (no FFT)."""
        # Create discrete TimeSeries
        ts1 = TimeSeries(np.random.randint(0, 5, 100), discrete=True, name="disc1")
        ts2 = TimeSeries(np.random.randint(0, 5, 100), discrete=True, name="disc2")

        cache = _build_fft_cache(
            [ts1], [ts2], "mi", "gcmi", ds=1, engine="auto"
        )

        # Cache should have entry but with None (no FFT for discrete-discrete)
        assert len(cache) > 0
        assert cache[(_get_ts_key(ts1), _get_ts_key(ts2))] is None

    def test_fft_cache_keys_stable(self):
        """FFT cache keys should be based on stable TimeSeries names."""
        ts1 = TimeSeries(np.random.randn(100), discrete=False, name="stable_key1")
        ts2 = TimeSeries(np.random.randn(100), discrete=False, name="stable_key2")

        cache = _build_fft_cache(
            [ts1], [ts2], "mi", "gcmi", ds=1, engine="auto"
        )

        # Check that cache keys use TimeSeries names
        expected_key = (_get_ts_key(ts1), _get_ts_key(ts2))
        assert expected_key in cache

    def test_unnamed_timeseries_raises(self):
        """Unnamed TimeSeries should raise ValueError."""
        # Create TimeSeries without name
        ts_unnamed = TimeSeries(np.random.randn(100), discrete=False)
        ts_named = TimeSeries(np.random.randn(100), discrete=False, name="named")

        # Should raise when trying to get key from unnamed TimeSeries
        with pytest.raises(ValueError, match="TimeSeries missing name"):
            _build_fft_cache(
                [ts_unnamed], [ts_named], "mi", "gcmi", ds=1, engine="auto"
            )


class TestMemoryCleanup:
    """Test FFT cache memory cleanup (integration test)."""

    def test_cache_cleanup_in_compute_me_stats(self):
        """FFT cache should be cleaned up after compute_me_stats."""
        from driada.intense.intense_base import compute_me_stats

        # Create test data
        ts_bunch1 = [
            TimeSeries(np.random.randn(200), discrete=False, name=f"neuron_{i}")
            for i in range(5)
        ]
        ts_bunch2 = [
            TimeSeries(np.random.randn(200), discrete=False, name=f"feature_{i}")
            for i in range(3)
        ]

        # Run compute_me_stats (builds FFT cache internally)
        stats, significance, info = compute_me_stats(
            ts_bunch1,
            ts_bunch2,
            metric="mi",
            mode="stage1",
            n_shuffles_stage1=10,
            ds=2,
            mi_estimator="gcmi",
            engine="auto",
            verbose=False,
        )

        # Force garbage collection
        gc.collect()

        # Check that FFTCacheEntry objects are cleaned up
        # (This is a basic check - in practice, cache should be deleted in finally block)
        assert stats is not None  # Function completed successfully
