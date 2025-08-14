"""Tests for spike reconstruction from calcium signals."""

import pytest
import numpy as np
from driada.experiment.synthetic import generate_synthetic_exp
from driada.experiment.wavelet_event_detection import (
    WVT_EVENT_DETECTION_PARAMS,
    extract_wvt_events,
    events_to_ts_array,
)
from driada.information.info_base import TimeSeries


def test_events_to_ts_array_basic():
    """Test basic functionality of events_to_ts_array."""
    # Create simple event data
    length = 1000
    fps = 20

    # Single neuron with two events
    st_ev_inds = [[100, 500]]
    end_ev_inds = [[120, 540]]

    spikes = events_to_ts_array(length, st_ev_inds, end_ev_inds, fps)

    assert spikes.shape == (1, length)
    assert np.sum(spikes[0, 100:120]) > 0  # First event
    assert np.sum(spikes[0, 500:540]) > 0  # Second event
    assert np.sum(spikes[0, :100]) == 0  # Before first event
    assert np.sum(spikes[0, 121:499]) == 0  # Between events


def test_events_to_ts_array_multiple_neurons():
    """Test events_to_ts_array with multiple neurons."""
    length = 1000
    fps = 20

    # Three neurons with different events
    st_ev_inds = [[100, 300], [200], [150, 450, 750]]
    end_ev_inds = [[120, 340], [250], [170, 490, 790]]

    spikes = events_to_ts_array(length, st_ev_inds, end_ev_inds, fps)

    assert spikes.shape == (3, length)
    # Check neuron 0
    assert np.sum(spikes[0, 100:120]) > 0
    assert np.sum(spikes[0, 300:340]) > 0
    # Check neuron 1
    assert np.sum(spikes[1, 200:250]) > 0
    # Check neuron 2
    assert np.sum(spikes[2, 150:170]) > 0
    assert np.sum(spikes[2, 450:490]) > 0
    assert np.sum(spikes[2, 750:790]) > 0


def test_events_to_ts_array_edge_cases():
    """Test edge cases for events_to_ts_array."""
    length = 1000
    fps = 20

    # Empty events
    st_ev_inds = [[], []]
    end_ev_inds = [[], []]

    spikes = events_to_ts_array(length, st_ev_inds, end_ev_inds, fps)
    assert spikes.shape == (2, length)
    assert np.sum(spikes) == 0  # No spikes

    # Very short event (less than min duration)
    st_ev_inds = [[100]]
    end_ev_inds = [[101]]  # 1 frame = 0.05s < 0.5s min

    spikes = events_to_ts_array(length, st_ev_inds, end_ev_inds, fps)
    # Should be extended to minimum duration
    assert np.sum(spikes[0]) >= int(0.5 * fps)  # At least 0.5s

    # Very long event (more than max duration)
    st_ev_inds = [[100]]
    end_ev_inds = [[200]]  # 100 frames = 5s > 2.5s max

    spikes = events_to_ts_array(length, st_ev_inds, end_ev_inds, fps)
    # Should be truncated to maximum duration
    assert np.sum(spikes[0]) <= int(2.5 * fps)  # At most 2.5s


def test_wavelet_spike_reconstruction(small_experiment):
    """Test spike reconstruction using wavelet method."""
    # Use fixture for synthetic calcium data
    exp = small_experiment

    # Get calcium data
    calcium = exp.calcium.data
    fps = exp.fps

    # Set up wavelet parameters
    wvt_kwargs = WVT_EVENT_DETECTION_PARAMS.copy()
    wvt_kwargs["fps"] = fps

    # Extract events
    st_ev_inds, end_ev_inds, all_ridges = extract_wvt_events(calcium, wvt_kwargs)

    # Convert to spike array
    spikes = events_to_ts_array(calcium.shape[1], st_ev_inds, end_ev_inds, fps)

    # Basic checks
    assert spikes.shape == calcium.shape
    assert spikes.dtype == np.float64
    assert np.all((spikes == 0) | (spikes == 1))  # Binary

    # Check that different neurons have different spike patterns
    if calcium.shape[0] > 1:
        # Not all neurons should have identical spikes
        assert not np.all(
            [np.array_equal(spikes[0], spikes[i]) for i in range(1, calcium.shape[0])]
        )


def test_experiment_with_spike_reconstruction(spike_reconstruction_experiment):
    """Test creating experiment with spike reconstruction."""
    # Use fixture for experiment with spike reconstruction
    exp = spike_reconstruction_experiment

    # Check that spikes were created
    assert hasattr(exp, "spikes")
    assert exp.spikes is not None
    assert exp.spikes.data.shape == exp.calcium.data.shape

    # Check that neurons have spike TimeSeries
    for neuron in exp.neurons:
        assert neuron.sp is not None
        assert isinstance(neuron.sp, TimeSeries)
        assert neuron.sp.discrete == True

    # Check that spikes are not all zero
    assert np.sum(exp.spikes.data) > 0

    # Check that different neurons have different spike patterns
    spike_sums = [np.sum(neuron.sp.data) for neuron in exp.neurons]
    assert len(set(spike_sums)) > 1  # Not all identical


def test_spike_reconstruction_reproducibility():
    """Test that spike reconstruction is reproducible with same seed."""
    # Create two experiments with same parameters
    exp1 = generate_synthetic_exp(
        n_dfeats=1, n_cfeats=0, nneurons=3, seed=42, with_spikes=True
    )
    exp2 = generate_synthetic_exp(
        n_dfeats=1, n_cfeats=0, nneurons=3, seed=42, with_spikes=True
    )

    # Check that calcium signals are identical
    assert np.allclose(exp1.calcium.data, exp2.calcium.data)

    # Check that reconstructed spikes are identical
    for i in range(3):
        assert np.array_equal(exp1.neurons[i].sp.data, exp2.neurons[i].sp.data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
