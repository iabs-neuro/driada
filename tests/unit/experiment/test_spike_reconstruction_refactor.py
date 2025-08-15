"""Tests for refactored spike reconstruction module."""

import pytest
import numpy as np
from driada.experiment.spike_reconstruction import reconstruct_spikes
from driada.information.info_base import TimeSeries, MultiTimeSeries


def test_reconstruct_spikes_wavelet_method(small_experiment):
    """Test spike reconstruction using wavelet method."""
    # Use fixture
    exp = small_experiment
    calcium = exp.calcium

    # Reconstruct spikes
    spikes, metadata = reconstruct_spikes(calcium, method="wavelet", fps=exp.fps)

    # Check output types
    assert isinstance(spikes, MultiTimeSeries)
    assert isinstance(metadata, dict)

    # Check spike properties
    assert spikes.data.shape == calcium.data.shape
    assert np.all((spikes.data == 0) | (spikes.data == 1))  # Binary

    # Check metadata
    assert metadata["method"] == "wavelet"
    assert "parameters" in metadata
    assert "start_events" in metadata
    assert "end_events" in metadata
    assert "ridges" in metadata

    # Check that spikes are discrete
    assert spikes.discrete == True


def test_reconstruct_spikes_threshold_method(small_experiment):
    """Test spike reconstruction using threshold method."""
    # Use fixture
    exp = small_experiment
    calcium = exp.calcium

    # Reconstruct spikes with custom parameters
    params = {"threshold_std": 2.0, "smooth_sigma": 3, "min_spike_interval": 0.2}
    spikes, metadata = reconstruct_spikes(
        calcium, method="threshold", fps=exp.fps, params=params
    )

    # Check output types
    assert isinstance(spikes, MultiTimeSeries)
    assert isinstance(metadata, dict)

    # Check spike properties
    assert spikes.data.shape == calcium.data.shape
    assert np.all((spikes.data == 0) | (spikes.data == 1))  # Binary

    # Check metadata
    assert metadata["method"] == "threshold"
    assert metadata["parameters"]["threshold_std"] == 2.0
    assert metadata["parameters"]["smooth_sigma"] == 3
    assert metadata["parameters"]["min_spike_interval"] == 0.2
    assert "spike_times" in metadata

    # Check that some spikes were detected
    assert np.sum(spikes.data) > 0


def test_reconstruct_spikes_custom_method(small_experiment):
    """Test spike reconstruction using custom callable method."""

    # Define custom reconstruction method
    def custom_method(calcium, fps, params):
        # Simple mock implementation
        n_neurons = calcium.data.shape[0]
        n_frames = calcium.data.shape[1]

        # Create random spikes
        np.random.seed(params.get("seed", 0))
        spikes_data = (np.random.rand(n_neurons, n_frames) > 0.95).astype(float)

        spike_ts_list = [
            TimeSeries(spikes_data[i, :], discrete=True) for i in range(n_neurons)
        ]
        spikes = MultiTimeSeries(spike_ts_list, allow_zero_columns=True)

        metadata = {"method": "custom", "params": params}

        return spikes, metadata

    # Use fixture
    exp = small_experiment
    calcium = exp.calcium

    # Reconstruct spikes
    params = {"seed": 123}
    spikes, metadata = reconstruct_spikes(
        calcium, method=custom_method, fps=exp.fps, params=params
    )

    # Check output
    assert isinstance(spikes, MultiTimeSeries)
    assert spikes.data.shape == calcium.data.shape
    assert metadata["method"] == "custom"
    assert metadata["params"]["seed"] == 123


def test_reconstruct_spikes_invalid_method(small_experiment):
    """Test error handling for invalid method."""
    exp = small_experiment
    calcium = exp.calcium

    with pytest.raises(ValueError, match="Unknown method"):
        reconstruct_spikes(calcium, method="invalid_method")


def test_wavelet_vs_threshold_comparison(medium_experiment):
    """Compare wavelet and threshold methods on same data."""
    # Use fixture with more neurons
    exp = medium_experiment
    calcium = exp.calcium

    # Reconstruct with both methods
    spikes_wavelet, meta_wavelet = reconstruct_spikes(
        calcium, method="wavelet", fps=exp.fps
    )
    spikes_threshold, meta_threshold = reconstruct_spikes(
        calcium, method="threshold", fps=exp.fps
    )

    # Both should detect some spikes
    assert np.sum(spikes_wavelet.data) > 0
    assert np.sum(spikes_threshold.data) > 0

    # Compare spike trains using appropriate metric for binary data
    # Use Jaccard similarity (intersection over union) for each neuron
    jaccard_scores = []
    for i in range(spikes_wavelet.data.shape[0]):
        wavelet_spikes = spikes_wavelet.data[i, :].astype(bool)
        threshold_spikes = spikes_threshold.data[i, :].astype(bool)

        intersection = np.sum(wavelet_spikes & threshold_spikes)
        union = np.sum(wavelet_spikes | threshold_spikes)

        if union > 0:
            jaccard = intersection / union
            jaccard_scores.append(jaccard)

    # Average Jaccard score should be positive but not perfect
    # (methods detect some common events but also differ)
    if jaccard_scores:
        avg_jaccard = np.mean(jaccard_scores)
        assert 0 < avg_jaccard < 0.9  # Some overlap but not identical


def test_experiment_integration(spike_reconstruction_experiment):
    """Test that Experiment class properly uses new reconstruction."""
    # Use fixture with spike reconstruction
    exp = spike_reconstruction_experiment

    # Check that spikes were reconstructed
    assert hasattr(exp, "spikes")
    assert isinstance(exp.spikes, MultiTimeSeries)
    assert hasattr(exp, "_reconstruction_metadata")

    # Check metadata
    assert exp._reconstruction_metadata["method"] == "wavelet"

    # Test with threshold method by creating experiment directly
    from driada.experiment import Experiment

    calcium = np.random.randn(3, 1000)
    exp2 = Experiment(
        "Test",
        calcium,
        None,  # spikes
        {},  # exp_identificators
        {"fps": 20.0},  # static_features
        {"test_feat": np.random.randn(1000)},  # dynamic_features
        reconstruct_spikes="threshold",
    )

    assert exp2._reconstruction_metadata["method"] == "threshold"


def test_parameter_propagation(small_experiment):
    """Test that parameters are properly propagated to reconstruction methods."""
    exp = small_experiment
    calcium = exp.calcium

    # Test wavelet parameters
    wavelet_params = {"sigma": 10, "eps": 20, "max_scale_thr": 10}
    spikes_w, meta_w = reconstruct_spikes(
        calcium, method="wavelet", fps=exp.fps, params=wavelet_params
    )

    assert meta_w["parameters"]["sigma"] == 10
    assert meta_w["parameters"]["eps"] == 20
    assert meta_w["parameters"]["max_scale_thr"] == 10

    # Test threshold parameters
    threshold_params = {
        "threshold_std": 3.0,
        "smooth_sigma": 5,
        "min_spike_interval": 0.5,
    }
    spikes_t, meta_t = reconstruct_spikes(
        calcium, method="threshold", fps=exp.fps, params=threshold_params
    )

    assert meta_t["parameters"]["threshold_std"] == 3.0
    assert meta_t["parameters"]["smooth_sigma"] == 5
    assert meta_t["parameters"]["min_spike_interval"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
