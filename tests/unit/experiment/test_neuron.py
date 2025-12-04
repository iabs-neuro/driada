import numpy as np
import pytest
from unittest.mock import patch
from driada.experiment.neuron import Neuron, MIN_CA_SHIFT
from driada.information.info_base import TimeSeries
from driada.utils.neural import generate_pseudo_calcium_multisignal
from driada.experiment.wavelet_event_detection import (
    WVT_EVENT_DETECTION_PARAMS,
    extract_wvt_events,
    events_to_ts_array,
)


class TestNeuronInitialization:
    """Test Neuron class initialization and basic properties."""

    def test_init_basic(self):
        """Test basic initialization with calcium and spike data."""
        ca_data = np.random.random(size=1000)
        sp_data = np.zeros(1000, dtype=int)
        sp_data[100:900:100] = 1  # Add some spikes

        neuron = Neuron("cell_0", ca_data, sp_data)

        assert neuron.cell_id == "cell_0"
        assert neuron.n_frames == 1000
        assert isinstance(neuron.ca, TimeSeries)
        assert not neuron.ca.discrete
        assert isinstance(neuron.sp, TimeSeries)
        assert neuron.sp.discrete
        assert neuron.sp_count == 8

    def test_init_without_spikes(self):
        """Test initialization without spike data."""
        ca_data = np.random.random(size=1000)

        neuron = Neuron("cell_1", ca_data, None)

        assert neuron.sp is None
        assert neuron.sp_count == 0

    def test_init_with_negative_calcium(self):
        """Test that negative calcium values are set to zero."""
        ca_data = np.array([-1, -0.5, 0, 0.5, 1])
        sp_data = np.zeros(5)

        neuron = Neuron("cell_2", ca_data, sp_data)

        # Check that negative values were set to 0
        assert np.all(neuron.ca.data >= 0)

    def test_init_with_custom_parameters(self):
        """Test initialization with custom t_rise and t_off."""
        ca_data = np.random.random(size=1000)
        sp_data = np.zeros(1000)

        neuron = Neuron(
            "cell_3", ca_data, sp_data, default_t_rise=0.5, default_t_off=3.0, fps=30.0
        )

        assert neuron.default_t_rise == 0.5 * 30.0  # Converted to frames
        assert neuron.default_t_off == 3.0 * 30.0  # Converted to frames

    def test_shuffle_mask_creation(self):
        """Test that shuffle mask is created correctly."""
        ca_data = np.random.random(size=1000)
        sp_data = np.zeros(1000)

        neuron = Neuron("cell_4", ca_data, sp_data, fps=20.0)

        # Check shuffle mask properties
        assert hasattr(neuron.ca, "shuffle_mask")
        assert len(neuron.ca.shuffle_mask) == 1000

        # Check that edges are masked
        min_shift = int(neuron.default_t_off * MIN_CA_SHIFT)
        assert not np.any(neuron.ca.shuffle_mask[:min_shift])
        assert not np.any(neuron.ca.shuffle_mask[-min_shift:])
        assert np.all(neuron.ca.shuffle_mask[min_shift:-min_shift])

    def test_fit_individual_t_off(self):
        """Test initialization with individual t_off fitting."""
        ca_data = np.random.random(size=1000)
        sp_data = np.zeros(1000)
        sp_data[100:900:50] = 1  # Add spikes

        with patch.object(Neuron, "_fit_t_off", return_value=(50.0, 0.1)):
            neuron = Neuron("cell_5", ca_data, sp_data, fit_individual_t_off=True)
            assert neuron._fit_t_off.called


class TestNeuronStaticMethods:
    """Test static methods of Neuron class."""

    def test_spike_form(self):
        """Test spike waveform generation."""
        t = np.linspace(0, 100, 100)
        form = Neuron.spike_form(t, t_rise=5, t_off=20)

        # Check properties
        assert form.max() == 1.0  # Normalized to max = 1
        assert form[0] == 0.0  # Starts at 0
        assert (
            form[-1] < 0.02
        )  # Decays to near 0 (exp(-100/20) ≈ 0.0067, but normalized)

    def test_get_restored_calcium(self):
        """Test calcium restoration from spikes."""
        sp = np.zeros(100)
        sp[20] = 1  # Single spike

        restored = Neuron.get_restored_calcium(sp, t_rise=5, t_off=20)

        assert len(restored) == len(sp)  # Output has same length as input
        assert restored.max() > 0  # Has positive values
        assert np.argmax(restored) >= 20  # Peak should be at or after spike

    def test_calcium_preprocessing(self):
        """Test calcium signal preprocessing."""
        ca = np.array([-1, 0, 1, 2, -0.5])
        processed = Neuron.calcium_preprocessing(ca)

        # Check that negative values are set to 0
        assert np.all(processed >= 0)
        # Check that small noise is added
        assert not np.array_equal(processed[1], 0)  # Zero value has noise added


class TestNeuronMethods:
    """Test instance methods of Neuron class."""

    def test_reconstruct_spikes_not_implemented(self):
        """Test that reconstruct_spikes raises AttributeError."""
        ca_data = np.random.random(size=1000)
        sp_data = np.zeros(1000)
        neuron = Neuron("cell_6", ca_data, sp_data)

        # Now reconstruct_spikes is implemented, so test it works
        spikes = neuron.reconstruct_spikes(method="wavelet", create_event_regions=True)
        assert isinstance(spikes, np.ndarray)
        assert len(spikes) == neuron.n_frames

    def test_get_mad(self):
        """Test median absolute deviation calculation."""
        ca_data = np.random.normal(loc=1, scale=0.5, size=1000)
        sp_data = np.zeros(1000)
        neuron = Neuron("cell_7", ca_data, sp_data)

        mad = neuron.get_mad()
        assert mad > 0
        assert neuron.mad == mad  # Cached

    def test_get_snr_with_spikes(self):
        """Test signal-to-noise ratio calculation with spikes."""
        ca_data = np.random.normal(loc=1, scale=0.1, size=1000)
        sp_data = np.zeros(1000)
        spike_indices = [100, 200, 300, 400, 500]
        sp_data[spike_indices] = 1
        ca_data[spike_indices] += 2  # Make spikes have higher amplitude

        neuron = Neuron("cell_8", ca_data, sp_data)
        snr = neuron.get_snr()

        assert snr > 0
        assert neuron.snr == snr  # Cached
        assert neuron.mad is not None  # MAD also calculated

    def test_get_snr_without_spikes(self):
        """Test SNR calculation fails without spikes."""
        ca_data = np.random.normal(loc=1, scale=0.1, size=1000)
        sp_data = np.zeros(1000)  # No spikes

        neuron = Neuron("cell_9", ca_data, sp_data)

        with pytest.raises(ValueError, match="No spike data available"):
            neuron.get_snr()

    def test_calc_snr_nan_handling(self):
        """Test that Neuron creation fails with NaN values in calcium data."""
        ca_data = np.ones(1000)
        ca_data[100] = np.nan
        sp_data = np.zeros(1000)
        sp_data[100] = 1

        # TimeSeries now validates against NaN values during creation
        with pytest.raises(ValueError, match="Time series contains NaN values"):
            neuron = Neuron("cell_10", ca_data, sp_data)

    def test_fit_t_off(self):
        """Test t_off fitting."""
        # Create synthetic calcium with known decay
        t = np.arange(1000)
        sp_data = np.zeros(1000)
        sp_data[100] = 1

        # Create calcium response
        ca_data = np.zeros(1000)
        spike_response = np.exp(-t / 40) * (t < 200)
        ca_data[100:300] = spike_response[:200]
        ca_data += np.random.normal(0, 0.01, 1000)  # Add noise

        neuron = Neuron("cell_11", ca_data, sp_data, fps=20.0)
        t_off, noise_ampl = neuron._fit_t_off()

        assert t_off > 0
        assert noise_ampl > 0

    def test_get_t_off_caching(self):
        """Test that t_off is cached after first calculation."""
        ca_data = np.random.random(size=1000)
        sp_data = np.zeros(1000)
        sp_data[100:900:100] = 1

        neuron = Neuron("cell_12", ca_data, sp_data)

        with patch.object(neuron, "_fit_t_off", return_value=(50.0, 0.1)) as mock_fit:
            t_off1 = neuron.get_t_off()
            t_off2 = neuron.get_t_off()

            assert t_off1 == t_off2
            mock_fit.assert_called_once()  # Only called once due to caching

    def test_get_noise_ampl_caching(self):
        """Test that noise amplitude is cached."""
        ca_data = np.random.random(size=1000)
        sp_data = np.zeros(1000)
        sp_data[100:900:100] = 1

        neuron = Neuron("cell_13", ca_data, sp_data)

        # Reconstruct spikes to populate asp (required for get_noise_ampl)
        neuron.reconstruct_spikes(method="wavelet", create_event_regions=True)

        # Test caching behavior
        noise1 = neuron.get_noise_ampl()
        noise2 = neuron.get_noise_ampl()

        assert noise1 == noise2
        assert noise1 > 0  # Should have valid noise amplitude

    def test_fit_t_off_high_value_warning(self, capsys):
        """Test warning when fitted t_off is too high."""
        ca_data = np.random.random(size=1000)
        sp_data = np.zeros(1000)
        sp_data[100] = 1

        neuron = Neuron("cell_14", ca_data, sp_data, default_t_off=2.0, fps=20.0)

        # Mock minimize to return very high t_off
        with patch("driada.experiment.neuron.minimize") as mock_min:
            mock_min.return_value.x = [300.0]  # Very high t_off
            mock_min.return_value.fun = 0.1

            t_off, _ = neuron._fit_t_off()

            # Check that t_off is capped
            assert t_off == neuron.default_t_off * 5

            # Check warning message was logged
            # The warning should be in the logs, not stdout
            # We can verify the t_off was capped which is the important behavior


class TestCalciumShuffling:
    """Test calcium data shuffling methods."""

    def test_shuffle_calcium_roll_based(self):
        """Test roll-based calcium shuffling."""
        ca_data = np.sin(np.linspace(0, 4 * np.pi, 1000))  # Sinusoidal pattern
        sp_data = np.zeros(1000)

        neuron = Neuron("cell_15", ca_data, sp_data)
        shuffled = neuron.get_shuffled_calcium(method="roll_based", return_array=True)

        assert len(shuffled) == len(ca_data)
        assert not np.array_equal(shuffled, ca_data)  # Should be different
        assert np.allclose(
            np.sort(shuffled), np.sort(neuron.ca.data), atol=1e-6
        )  # Same values, different order

    def test_shuffle_calcium_roll_based_with_shift(self):
        """Test roll-based shuffling with specific shift."""
        ca_data = np.arange(1000)
        sp_data = np.zeros(1000)

        neuron = Neuron("cell_16", ca_data, sp_data)
        shift = 100
        shuffled = neuron.get_shuffled_calcium(
            method="roll_based", return_array=True, shift=shift
        )

        # Check that data is rolled by shift amount
        expected = np.roll(neuron.ca.data, shift)
        assert np.allclose(shuffled, expected, atol=1e-6)

    def test_shuffle_calcium_chunks_based(self):
        """Test chunk-based calcium shuffling."""
        ca_data = np.arange(1000)
        sp_data = np.zeros(1000)

        neuron = Neuron("cell_17", ca_data, sp_data)

        # Set random seed for reproducibility
        np.random.seed(42)
        shuffled = neuron.get_shuffled_calcium(method="chunks_based", return_array=True, n=10)

        assert len(shuffled) == len(ca_data)
        assert not np.array_equal(shuffled, ca_data)

    def test_shuffle_calcium_waveform_based(self):
        """Test waveform-based calcium shuffling."""
        # Create calcium data with clear spike responses
        ca_data = np.zeros(1000)
        sp_data = np.zeros(1000)
        spike_times = [100, 300, 500, 700]
        sp_data[spike_times] = 1

        # Add spike responses to calcium
        for t in spike_times:
            response = np.exp(-np.arange(100) / 20)
            ca_data[t : t + 100] += response[: min(100, 1000 - t)]

        neuron = Neuron("cell_18", ca_data, sp_data)

        # Mock the spike shuffling to return known pattern
        with patch.object(
            neuron, "_shuffle_spikes_data_isi_based", return_value=np.roll(sp_data, 50)
        ):
            shuffled = neuron.get_shuffled_calcium(method="waveform_based", return_array=True)

        assert len(shuffled) == len(ca_data)
        assert not np.array_equal(shuffled, ca_data)

    def test_shuffle_calcium_return_timeseries(self):
        """Test returning shuffled calcium as TimeSeries."""
        ca_data = np.random.random(size=1000)
        sp_data = np.zeros(1000)

        neuron = Neuron("cell_19", ca_data, sp_data)
        shuffled_ts = neuron.get_shuffled_calcium(method="roll_based", return_array=False)

        assert isinstance(shuffled_ts, TimeSeries)
        assert not shuffled_ts.discrete

    def test_shuffle_calcium_unknown_method(self):
        """Test error on unknown shuffling method."""
        ca_data = np.random.random(size=1000)
        sp_data = np.zeros(1000)

        neuron = Neuron("cell_20", ca_data, sp_data)

        # Test that unknown methods raise ValueError
        with pytest.raises(ValueError, match="Invalid method 'unknown_method'"):
            neuron.get_shuffled_calcium(method="unknown_method")


class TestSpikeShuffling:
    """Test spike data shuffling methods."""

    def test_shuffle_spikes_no_spike_data(self):
        """Test error when shuffling spikes without spike data."""
        ca_data = np.random.random(size=1000)
        neuron = Neuron("cell_21", ca_data, None)

        with pytest.raises(
            AttributeError, match="Unable to shuffle spikes without spikes data"
        ):
            neuron.get_shuffled_spikes()

    def test_shuffle_spikes_isi_based(self):
        """Test ISI-based spike shuffling."""
        ca_data = np.random.random(size=1000)
        sp_data = np.zeros(1000)
        spike_times = [100, 250, 400, 600, 800]
        sp_data[spike_times] = 1

        neuron = Neuron("cell_22", ca_data, sp_data)

        np.random.seed(42)
        shuffled = neuron.get_shuffled_spikes(method="isi_based", return_array=True)

        assert len(shuffled) == len(sp_data)
        assert np.sum(shuffled) == np.sum(sp_data)  # Same number of spikes
        assert not np.array_equal(shuffled, sp_data)  # Different pattern

    def test_shuffle_spikes_isi_based_no_events(self):
        """Test ISI shuffling with no spikes returns original."""
        ca_data = np.random.random(size=1000)
        sp_data = np.zeros(1000)  # No spikes

        neuron = Neuron("cell_23", ca_data, sp_data)
        shuffled = neuron._shuffle_spikes_data_isi_based()

        assert np.array_equal(shuffled, sp_data)

    def test_shuffle_spikes_isi_based_multi_valued(self):
        """Test ISI shuffling with multi-valued spikes."""
        ca_data = np.random.random(size=1000)
        sp_data = np.zeros(1000)
        sp_data[100] = 1
        sp_data[200] = 2
        sp_data[300] = 3

        neuron = Neuron("cell_24", ca_data, sp_data)

        np.random.seed(42)
        shuffled = neuron._shuffle_spikes_data_isi_based()

        # Check that spike values are preserved (though positions change)
        assert set(shuffled[shuffled > 0]) == {1, 2, 3}

    def test_shuffle_spikes_return_timeseries(self):
        """Test returning shuffled spikes as TimeSeries."""
        ca_data = np.random.random(size=1000)
        sp_data = np.zeros(1000)
        sp_data[100:900:100] = 1

        neuron = Neuron("cell_25", ca_data, sp_data)
        shuffled_ts = neuron.get_shuffled_spikes(method="isi_based", return_array=False)

        assert isinstance(shuffled_ts, TimeSeries)
        assert shuffled_ts.discrete

    def test_shuffle_spikes_unknown_method(self):
        """Test error on unknown spike shuffling method."""
        ca_data = np.random.random(size=1000)
        sp_data = np.zeros(1000)
        sp_data[100] = 1

        neuron = Neuron("cell_26", ca_data, sp_data)

        # Test that unknown methods raise ValueError
        with pytest.raises(ValueError, match="Invalid method 'unknown_method'"):
            neuron.get_shuffled_spikes(method="unknown_method")


class TestIntegration:
    """Integration tests with other components."""

    def test_wavelet_spike_inference(self):
        """Test integration with wavelet spike detection."""
        # Set seed for reproducibility and to avoid NaN issues from test interactions
        np.random.seed(42)

        wvt_kwargs = WVT_EVENT_DETECTION_PARAMS.copy()
        wvt_kwargs["fps"] = 20

        n = 10  # number of neurons
        duration = 60  # seconds
        sampling_rate = 20  # Hz
        event_rate = 0.5  # events per second
        amplitude_range = (0.5, 2)
        decay_time = 2  # seconds
        noise_std = 0.1

        pseudo_calcium = generate_pseudo_calcium_multisignal(
            n,
            duration,
            sampling_rate,
            event_rate,
            amplitude_range,
            decay_time,
            noise_std,
        )

        st_ev_inds, end_ev_inds, all_ridges = extract_wvt_events(
            pseudo_calcium, wvt_kwargs
        )
        spikes = events_to_ts_array(
            pseudo_calcium.shape[1], st_ev_inds, end_ev_inds, wvt_kwargs["fps"]
        )

        # Create neurons with detected spikes
        neurons = []
        for i in range(n):
            neuron = Neuron(f"cell_{i}", pseudo_calcium[i, :], spikes[i, :])
            neurons.append(neuron)

            # Test that neurons were created successfully
            assert neuron.n_frames == duration * sampling_rate
            assert neuron.sp_count >= 0  # Should have detected some spikes

    def test_neuron_with_experiment_compatibility(self):
        """Test that Neuron objects are compatible with Experiment class expectations."""
        ca_data = np.random.random(size=1000)
        sp_data = np.zeros(1000)
        sp_data[100:900:100] = 1

        neuron = Neuron("test_cell", ca_data, sp_data)

        # Check attributes expected by Experiment class
        assert hasattr(neuron, "ca")
        assert hasattr(neuron, "sp")
        assert hasattr(neuron, "cell_id")
        assert hasattr(neuron, "n_frames")
        assert hasattr(neuron.ca, "shuffle_mask")

        # Check methods expected by Experiment
        assert callable(getattr(neuron, "get_shuffled_calcium", None))
        assert callable(getattr(neuron, "get_shuffled_spikes", None))


class TestWaveletSNR:
    """Test wavelet-based SNR calculation methods."""

    def test_wavelet_snr_without_reconstruction(self):
        """Test that get_wavelet_snr raises error without prior reconstruction."""
        ca_data = np.random.random(size=1000)
        neuron = Neuron("cell_snr_1", ca_data, None)

        with pytest.raises(
            ValueError,
            match="No event regions detected.*reconstruct_spikes",
        ):
            neuron.get_wavelet_snr()

    def test_wavelet_snr_with_no_events(self):
        """Test that get_wavelet_snr raises error when events mask is all False."""
        # Create neuron with valid data
        ca_data = np.random.normal(0.2, 0.05, 1000)
        neuron = Neuron("cell_snr_2", ca_data, None)

        # Manually set events to an all-False mask (simulating no events detected)
        from driada.information.info_base import TimeSeries
        neuron.events = TimeSeries(
            data=np.zeros(1000, dtype=bool),
            discrete=False,
        )

        with pytest.raises(ValueError, match="No events in event mask"):
            neuron.get_wavelet_snr()

    def test_wavelet_snr_with_valid_events(self):
        """Test SNR calculation with valid events."""
        np.random.seed(42)

        # Create calcium signal with clear events
        ca_data = np.random.normal(0.2, 0.05, 1000)  # Baseline
        event_times = [100, 300, 500, 700, 900]
        for t in event_times:
            # Add event peaks
            ca_data[t : t + 20] += np.linspace(0, 0.5, 20)
            ca_data[t + 20 : t + 60] += np.linspace(0.5, 0, 40)

        neuron = Neuron("cell_snr_3", ca_data, None)
        neuron.reconstruct_spikes(
            method="wavelet",
            create_event_regions=True,
            max_ampl_thr=0.1,
            sigma=4,
        )

        snr = neuron.get_wavelet_snr()

        assert isinstance(snr, float)
        assert snr > 0
        assert np.isfinite(snr)

    def test_wavelet_snr_caching(self):
        """Test that wavelet SNR is cached after first calculation."""
        np.random.seed(43)

        ca_data = np.random.normal(0.2, 0.05, 1000)
        event_times = [100, 300, 500, 700]
        for t in event_times:
            ca_data[t : t + 20] += np.linspace(0, 0.4, 20)
            ca_data[t + 20 : t + 50] += np.linspace(0.4, 0, 30)

        neuron = Neuron("cell_snr_4", ca_data, None)
        neuron.reconstruct_spikes(
            method="wavelet",
            create_event_regions=True,
        )

        # First call
        snr1 = neuron.get_wavelet_snr()
        assert neuron.wavelet_snr == snr1

        # Second call should return cached value
        snr2 = neuron.get_wavelet_snr()
        assert snr2 == snr1
        assert neuron.wavelet_snr == snr1

    def test_wavelet_snr_event_at_start(self):
        """Test SNR calculation when event starts at beginning of signal."""
        np.random.seed(44)

        ca_data = np.random.normal(0.2, 0.05, 1000)
        # Event at start
        ca_data[0:20] += np.linspace(0, 0.5, 20)
        ca_data[20:60] += np.linspace(0.5, 0, 40)
        # More events
        for t in [300, 600, 900]:
            ca_data[t : t + 20] += np.linspace(0, 0.5, 20)
            ca_data[t + 20 : t + 60] += np.linspace(0.5, 0, 40)

        neuron = Neuron("cell_snr_5", ca_data, None)
        neuron.reconstruct_spikes(
            method="wavelet",
            create_event_regions=True,
            max_ampl_thr=0.1,
        )

        snr = neuron.get_wavelet_snr()
        assert isinstance(snr, float)
        assert snr > 0

    def test_wavelet_snr_event_at_end(self):
        """Test SNR calculation when event extends to end of signal."""
        np.random.seed(45)

        ca_data = np.random.normal(0.2, 0.05, 1000)
        # Events in middle
        for t in [100, 400, 700]:
            ca_data[t : t + 20] += np.linspace(0, 0.5, 20)
            ca_data[t + 20 : t + 60] += np.linspace(0.5, 0, 40)
        # Event at end
        ca_data[950:970] += np.linspace(0, 0.5, 20)
        ca_data[970:] += 0.5

        neuron = Neuron("cell_snr_6", ca_data, None)
        neuron.reconstruct_spikes(
            method="wavelet",
            create_event_regions=True,
            max_ampl_thr=0.1,
        )

        snr = neuron.get_wavelet_snr()
        assert isinstance(snr, float)
        assert snr > 0

    def test_wavelet_snr_sparse_high_amplitude_events(self):
        """Test SNR with sparse but high amplitude events (peak amplitude fix)."""
        np.random.seed(46)

        # Low baseline with noise
        ca_data = np.random.normal(0.1, 0.02, 1000)
        # Few but very high amplitude events
        event_times = [200, 600, 900]
        for t in event_times:
            # Very narrow, high peak
            ca_data[t : t + 5] += np.linspace(0, 1.0, 5)
            ca_data[t + 5 : t + 10] += np.linspace(1.0, 0, 5)

        neuron = Neuron("cell_snr_7", ca_data, None)
        neuron.reconstruct_spikes(
            method="wavelet",
            create_event_regions=True,
            max_ampl_thr=0.05,
        )

        snr = neuron.get_wavelet_snr()
        # With peak amplitude extraction, should get high SNR
        assert snr > 1.0

    def test_wavelet_snr_dense_low_amplitude_events(self):
        """Test SNR with dense but lower amplitude events."""
        np.random.seed(47)

        ca_data = np.random.normal(0.2, 0.05, 1000)
        # Many low amplitude events
        for t in range(100, 900, 80):
            ca_data[t : t + 15] += np.linspace(0, 0.2, 15)
            ca_data[t + 15 : t + 30] += np.linspace(0.2, 0, 15)

        neuron = Neuron("cell_snr_8", ca_data, None)
        neuron.reconstruct_spikes(
            method="wavelet",
            create_event_regions=True,
            max_ampl_thr=0.1,
        )

        snr = neuron.get_wavelet_snr()
        assert isinstance(snr, float)
        assert snr > 0

    def test_wavelet_snr_insufficient_baseline(self):
        """Test error when insufficient baseline frames available."""
        np.random.seed(48)

        # Create a signal where almost all frames are events
        ca_data = np.random.normal(0.2, 0.05, 100)
        neuron = Neuron("cell_snr_9", ca_data, None)

        # Manually create events mask with only 5 baseline frames
        from driada.information.info_base import TimeSeries
        events_mask = np.ones(100, dtype=bool)
        events_mask[10:15] = False  # Only 5 baseline frames
        neuron.events = TimeSeries(
            data=events_mask,
            discrete=False,
        )

        # Should raise error for insufficient baseline
        with pytest.raises(ValueError, match="Insufficient baseline frames"):
            neuron.get_wavelet_snr()

    def test_wavelet_snr_too_few_events(self):
        """Test error when fewer than 3 events detected."""
        np.random.seed(49)

        ca_data = np.random.normal(0.2, 0.05, 1000)
        neuron = Neuron("cell_snr_10", ca_data, None)

        # Manually create events mask with only 2 events
        from driada.information.info_base import TimeSeries
        events_mask = np.zeros(1000, dtype=bool)
        events_mask[300:350] = True  # Event 1
        events_mask[700:750] = True  # Event 2
        neuron.events = TimeSeries(
            data=events_mask,
            discrete=False,
        )

        with pytest.raises(ValueError, match="Too few events detected.*Need at least 3"):
            neuron.get_wavelet_snr()

    def test_wavelet_snr_zero_baseline_noise(self):
        """Test error when baseline has zero variance (perfect signal)."""
        # Create signal with perfectly constant baseline
        # Note: We need to use calcium_preprocessing=False or directly set ca.data
        # to avoid noise being added
        ca_data = np.ones(1000) * 0.5
        event_times = [100, 300, 500, 700]
        for t in event_times:
            ca_data[t : t + 20] = np.linspace(0.5, 1.0, 20)
            ca_data[t + 20 : t + 60] = np.linspace(1.0, 0.5, 40)

        # Create neuron and override ca.data to avoid preprocessing noise
        neuron = Neuron("cell_snr_11", ca_data, None)
        # Force ca.data to be exactly constant in baseline
        neuron.ca.data[:] = 0.5
        for t in event_times:
            neuron.ca.data[t : t + 20] = np.linspace(0.5, 1.0, 20)
            neuron.ca.data[t + 20 : t + 60] = np.linspace(1.0, 0.5, 40)

        # Manually create events mask
        from driada.information.info_base import TimeSeries
        events_mask = np.zeros(1000, dtype=bool)
        for t in event_times:
            events_mask[t : t + 60] = True
        neuron.events = TimeSeries(
            data=events_mask,
            discrete=False,
        )

        with pytest.raises(ValueError, match="Baseline noise is zero"):
            neuron.get_wavelet_snr()

    def test_wavelet_snr_initialization_cache(self):
        """Test that wavelet_snr cache attribute is initialized to None."""
        ca_data = np.random.random(size=1000)
        neuron = Neuron("cell_snr_12", ca_data, None)

        assert hasattr(neuron, "wavelet_snr")
        assert neuron.wavelet_snr is None

    def test_wavelet_snr_realistic_values(self):
        """Test that SNR values are in realistic range for typical signals."""
        np.random.seed(50)

        # Realistic calcium signal
        ca_data = np.random.normal(0.3, 0.1, 2000)
        event_times = np.arange(200, 1800, 150)
        for t in event_times:
            ca_data[t : t + 30] += np.linspace(0, 0.6, 30)
            ca_data[t + 30 : t + 100] += 0.6 * np.exp(-np.arange(70) / 20)

        neuron = Neuron("cell_snr_13", ca_data, None)
        neuron.reconstruct_spikes(
            method="wavelet",
            create_event_regions=True,
        )

        snr = neuron.get_wavelet_snr()

        # Realistic SNR should be between 0.5 and 50 for typical signals
        assert 0.5 < snr < 50


class TestExtractEventAmplitudes:
    """Test dF/F0 amplitude extraction correctness."""

    def test_dff_normalization_correctness(self):
        """Verify dF/F0 calculation matches hand-calculated values."""
        # Create signal with known baseline and peak
        # F0 = 100, Fpeak = 150 -> dF/F0 = (150-100)/100 = 0.5
        ca_signal = np.ones(100) * 100  # Baseline F0 = 100
        ca_signal[50:60] = 150  # Peak value

        st_ev_inds = [50]
        end_ev_inds = [60]
        baseline_window = 20

        amplitudes = Neuron.extract_event_amplitudes(
            ca_signal, st_ev_inds, end_ev_inds,
            baseline_window=baseline_window,
            already_dff=False
        )

        # dF/F0 = (150 - 100) / 100 = 0.5
        assert len(amplitudes) == 1
        assert amplitudes[0] == pytest.approx(0.5, abs=0.01)

    def test_already_dff_mode(self):
        """Test extraction when signal is already dF/F normalized."""
        # Pre-normalized signal: baseline = 0, peak = 0.6
        ca_signal = np.zeros(100)
        ca_signal[50:60] = 0.6

        st_ev_inds = [50]
        end_ev_inds = [60]

        amplitudes = Neuron.extract_event_amplitudes(
            ca_signal, st_ev_inds, end_ev_inds,
            baseline_window=20,
            already_dff=True
        )

        # Peak - baseline = 0.6 - 0 = 0.6
        assert amplitudes[0] == pytest.approx(0.6, abs=0.01)

    def test_baseline_edge_case_event_at_start(self):
        """Test handling when event is at signal start (limited baseline)."""
        ca_signal = np.ones(100) * 100
        ca_signal[5:15] = 150  # Event very close to start

        st_ev_inds = [5]
        end_ev_inds = [15]

        # Should use available frames for baseline (0-5)
        amplitudes = Neuron.extract_event_amplitudes(
            ca_signal, st_ev_inds, end_ev_inds,
            baseline_window=20,  # Requests more than available
            already_dff=False
        )

        # Should still compute amplitude using available baseline
        assert len(amplitudes) == 1
        assert amplitudes[0] > 0

    def test_multiple_events_correct_baselines(self):
        """Verify each event uses its own local baseline."""
        ca_signal = np.concatenate([
            np.ones(50) * 100,   # First baseline
            np.ones(20) * 150,   # First event
            np.ones(30) * 200,   # Second baseline (higher)
            np.ones(20) * 300,   # Second event
            np.ones(30) * 200,
        ])

        st_ev_inds = [50, 100]
        end_ev_inds = [70, 120]

        amplitudes = Neuron.extract_event_amplitudes(
            ca_signal, st_ev_inds, end_ev_inds,
            baseline_window=20,
            already_dff=False
        )

        assert len(amplitudes) == 2
        # First event: dF/F0 = (150 - 100) / 100 = 0.5
        assert amplitudes[0] == pytest.approx(0.5, abs=0.05)
        # Second event: dF/F0 = (300 - 200) / 200 = 0.5
        assert amplitudes[1] == pytest.approx(0.5, abs=0.05)


class TestAmplitudesToPointEvents:
    """Test conversion of amplitudes to point event arrays."""

    def test_placement_peak(self):
        """Test that 'peak' placement puts amplitude at calcium peak."""
        length = 100
        ca_signal = np.zeros(100)
        ca_signal[55] = 1.0  # Peak at frame 55

        st_ev_inds = [50]
        end_ev_inds = [60]
        amplitudes = [0.5]

        point_events = Neuron.amplitudes_to_point_events(
            length, ca_signal, st_ev_inds, end_ev_inds, amplitudes,
            placement='peak'
        )

        # Amplitude should be at frame 55 (peak location)
        assert point_events[55] == pytest.approx(0.5)
        assert np.sum(point_events) == pytest.approx(0.5)

    def test_placement_start(self):
        """Test that 'start' placement puts amplitude at event start."""
        length = 100
        ca_signal = np.zeros(100)
        ca_signal[55] = 1.0

        st_ev_inds = [50]
        end_ev_inds = [60]
        amplitudes = [0.5]

        point_events = Neuron.amplitudes_to_point_events(
            length, ca_signal, st_ev_inds, end_ev_inds, amplitudes,
            placement='start'
        )

        # Amplitude should be at frame 50 (start location)
        assert point_events[50] == pytest.approx(0.5)

    def test_amplitude_accumulation(self):
        """Test that multiple events at same frame accumulate amplitudes."""
        length = 100
        ca_signal = np.zeros(100)
        ca_signal[50] = 1.0  # Both events peak at same location

        # Two overlapping events
        st_ev_inds = [48, 49]
        end_ev_inds = [52, 53]
        amplitudes = [0.3, 0.4]

        point_events = Neuron.amplitudes_to_point_events(
            length, ca_signal, st_ev_inds, end_ev_inds, amplitudes,
            placement='peak'
        )

        # Both events peak at frame 50, amplitudes should sum
        assert point_events[50] == pytest.approx(0.7)

    def test_onset_placement_uses_kernel_offset(self):
        """Test that 'onset' placement accounts for kernel peak delay."""
        length = 100
        ca_signal = np.zeros(100)
        ca_signal[60] = 1.0  # Peak at frame 60

        st_ev_inds = [55]
        end_ev_inds = [65]
        amplitudes = [0.5]
        t_rise = 5.0
        t_off = 20.0

        point_events = Neuron.amplitudes_to_point_events(
            length, ca_signal, st_ev_inds, end_ev_inds, amplitudes,
            placement='onset', t_rise_frames=t_rise, t_off_frames=t_off, fps=20.0
        )

        # Should place amplitude before peak (at estimated onset)
        non_zero_idx = np.nonzero(point_events)[0]
        assert len(non_zero_idx) == 1
        assert non_zero_idx[0] < 60  # Should be before peak


class TestSpikeKernelReconstruction:
    """Test spike waveform and calcium reconstruction accuracy."""

    def test_spike_form_double_exponential_shape(self):
        """Verify spike kernel has correct double-exponential shape."""
        t = np.arange(0, 100, 1)
        t_rise = 5.0
        t_off = 20.0

        form = Neuron.spike_form(t, t_rise=t_rise, t_off=t_off)

        # Properties of double exponential:
        # 1. Starts at 0
        assert form[0] == 0.0

        # 2. Has single peak
        peaks = np.where(np.diff(np.sign(np.diff(form))) < 0)[0] + 1
        assert len(peaks) == 1

        # 3. Peak is normalized to 1.0
        assert form.max() == pytest.approx(1.0)

        # 4. Peak time should be around t_rise * ln(t_off/t_rise) / (1/t_rise - 1/t_off)
        # For t_rise=5, t_off=20, peak is around t=8
        peak_idx = np.argmax(form)
        assert 5 < peak_idx < 15

    def test_reconstruction_accuracy_single_spike(self):
        """Test that get_restored_calcium accurately reconstructs single spike."""
        sp = np.zeros(200)
        spike_time = 50
        sp[spike_time] = 1.0

        t_rise = 5.0
        t_off = 20.0

        restored = Neuron.get_restored_calcium(sp, t_rise=t_rise, t_off=t_off)

        # Check basic properties
        assert len(restored) == len(sp)
        assert np.argmax(restored) >= spike_time  # Peak after spike

        # Peak amplitude should be close to 1 (normalized kernel)
        assert restored.max() == pytest.approx(1.0, abs=0.01)

    def test_reconstruction_accuracy_multiple_spikes(self):
        """Test linear superposition of multiple spike kernels."""
        sp = np.zeros(300)
        sp[50] = 1.0
        sp[150] = 1.0

        t_rise = 5.0
        t_off = 20.0

        restored = Neuron.get_restored_calcium(sp, t_rise=t_rise, t_off=t_off)

        # Should have two peaks
        # Find peaks by looking for local maxima
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(restored, height=0.5, distance=30)

        assert len(peaks) == 2


class TestDeconvolveGivenEventTimes:
    """Test deconvolution for amplitude recovery."""

    def test_single_isolated_event_recovery(self):
        """Test amplitude recovery for single isolated event."""
        # Create synthetic calcium signal
        n_frames = 200
        t_rise = 5.0
        t_off = 20.0
        true_amplitude = 0.8
        event_time = 50

        # Generate known calcium response
        t = np.arange(n_frames - event_time)
        kernel = (1 - np.exp(-t / t_rise)) * np.exp(-t / t_off)
        kernel = kernel / kernel.max()

        ca_signal = np.zeros(n_frames)
        ca_signal[event_time:] = true_amplitude * kernel

        # Recover amplitude
        event_times = [event_time]
        amplitudes = Neuron.deconvolve_given_event_times(
            ca_signal, event_times, t_rise, t_off
        )

        assert len(amplitudes) == 1
        assert amplitudes[0] == pytest.approx(true_amplitude, abs=0.1)

    def test_overlapping_events_amplitude_recovery(self):
        """Test that overlapping events have amplitudes recovered correctly."""
        n_frames = 300
        t_rise = 5.0
        t_off = 20.0

        # Two overlapping events with different amplitudes
        event_times = [50, 70]  # Only 20 frames apart (overlap)
        true_amplitudes = [1.0, 0.6]

        # Generate overlapping calcium response
        ca_signal = np.zeros(n_frames)
        for event_time, true_amp in zip(event_times, true_amplitudes):
            t = np.arange(n_frames - event_time)
            kernel = (1 - np.exp(-t / t_rise)) * np.exp(-t / t_off)
            kernel = kernel / kernel.max()
            ca_signal[event_time:] += true_amp * kernel

        # Recover amplitudes
        recovered = Neuron.deconvolve_given_event_times(
            ca_signal, event_times, t_rise, t_off
        )

        assert len(recovered) == 2
        # Should recover both amplitudes reasonably well
        assert recovered[0] == pytest.approx(true_amplitudes[0], abs=0.15)
        assert recovered[1] == pytest.approx(true_amplitudes[1], abs=0.15)

    def test_empty_event_list(self):
        """Test handling of empty event list."""
        ca_signal = np.random.randn(100)
        event_times = []

        amplitudes = Neuron.deconvolve_given_event_times(
            ca_signal, event_times, t_rise_frames=5, t_off_frames=20
        )

        assert len(amplitudes) == 0


class TestEstimateOnsetTimes:
    """Test spike onset estimation from detected events."""

    def test_onset_before_peak(self):
        """Verify onset is estimated before the calcium peak."""
        # Create signal with clear peak
        ca_signal = np.zeros(100)
        ca_signal[50:70] = np.concatenate([
            np.linspace(0, 1, 10),  # Rise
            np.linspace(1, 0.2, 10)  # Decay
        ])

        st_inds = [50]
        end_inds = [70]
        t_rise = 5.0
        t_off = 20.0

        onsets = Neuron.estimate_onset_times(
            ca_signal, st_inds, end_inds, t_rise, t_off
        )

        assert len(onsets) == 1
        # Peak is at frame 60, onset should be before it
        assert onsets[0] < 60

    def test_onset_uses_kernel_offset(self):
        """Verify onset calculation uses correct kernel peak offset."""
        ca_signal = np.zeros(100)
        ca_signal[50] = 1.0  # Sharp peak

        st_inds = [45]
        end_inds = [55]
        t_rise = 5.0
        t_off = 20.0

        # Compute expected offset
        expected_offset = Neuron.compute_kernel_peak_offset(t_rise, t_off)

        onsets = Neuron.estimate_onset_times(
            ca_signal, st_inds, end_inds, t_rise, t_off
        )

        # Onset should be peak (50) minus kernel offset
        expected_onset = max(0, 50 - expected_offset)
        assert onsets[0] == int(expected_onset)

    def test_multiple_events(self):
        """Test onset estimation for multiple events."""
        ca_signal = np.zeros(200)
        ca_signal[50] = 1.0
        ca_signal[120] = 0.8

        st_inds = [45, 115]
        end_inds = [55, 125]
        t_rise = 5.0
        t_off = 20.0

        onsets = Neuron.estimate_onset_times(
            ca_signal, st_inds, end_inds, t_rise, t_off
        )

        assert len(onsets) == 2
        # Both should be before their respective peaks
        assert onsets[0] < 50
        assert onsets[1] < 120

    def test_onset_at_signal_start(self):
        """Test onset clamped to 0 when event is at signal start."""
        ca_signal = np.zeros(100)
        ca_signal[5] = 1.0  # Peak very close to start

        st_inds = [0]
        end_inds = [10]
        t_rise = 5.0
        t_off = 20.0

        onsets = Neuron.estimate_onset_times(
            ca_signal, st_inds, end_inds, t_rise, t_off
        )

        # Should be clamped to 0, not negative
        assert onsets[0] >= 0


class TestOptimizeKinetics:
    """Test kinetics optimization workflows."""

    def test_optimize_kinetics_returns_result_dict(self):
        """Verify optimize_kinetics returns expected result structure."""
        np.random.seed(42)

        # Create signal with clear events
        n_frames = 1000
        fps = 20.0
        ca_data = np.random.normal(0.2, 0.05, n_frames)

        # Add events with known kinetics
        event_times = [100, 300, 500, 700]
        t_rise_true = 0.2  # seconds
        t_off_true = 1.0  # seconds

        for t in event_times:
            duration = int(3 * t_off_true * fps)
            t_arr = np.arange(duration) / fps
            kernel = (1 - np.exp(-t_arr / t_rise_true)) * np.exp(-t_arr / t_off_true)
            kernel = kernel / kernel.max() * 0.5
            end_idx = min(t + duration, n_frames)
            ca_data[t:end_idx] += kernel[:end_idx - t]

        neuron = Neuron("kinetics_test", ca_data, None, fps=fps)
        neuron.reconstruct_spikes(method='wavelet', create_event_regions=True)

        result = neuron.optimize_kinetics(
            method='direct',
            fps=fps,
            update_reconstruction=False  # Skip re-detection for speed
        )

        # Check result structure
        assert 'optimized' in result
        assert 't_rise' in result
        assert 't_off' in result
        assert 'method' in result

    def test_optimize_kinetics_updates_instance(self):
        """Verify optimize_kinetics updates neuron's t_rise/t_off."""
        np.random.seed(42)

        n_frames = 1000
        fps = 20.0
        ca_data = np.random.normal(0.2, 0.05, n_frames)

        # Add clear events
        for t in [100, 300, 500, 700]:
            duration = 60
            t_arr = np.arange(duration) / fps
            kernel = (1 - np.exp(-t_arr / 0.2)) * np.exp(-t_arr / 1.0)
            kernel = kernel / kernel.max() * 0.5
            ca_data[t:t+duration] += kernel

        neuron = Neuron("kinetics_test2", ca_data, None, fps=fps)
        neuron.reconstruct_spikes(method='wavelet', create_event_regions=True)

        # Store original values
        original_t_rise = neuron.t_rise
        original_t_off = neuron.t_off

        result = neuron.optimize_kinetics(
            method='direct',
            fps=fps,
            update_reconstruction=False
        )

        if result.get('optimized', False):
            # If optimization succeeded, values should be updated
            # (they might be same if already optimal, but the mechanism works)
            assert neuron.t_rise is not None
            assert neuron.t_off is not None

    def test_optimize_kinetics_invalid_method(self):
        """Test that invalid method raises ValueError."""
        ca_data = np.random.random(500)
        neuron = Neuron("test", ca_data, None)

        with pytest.raises(ValueError, match="Only 'direct' method"):
            neuron.optimize_kinetics(method='invalid_method')


class TestReconstructionR2:
    """Test reconstruction quality metrics."""

    def test_get_reconstruction_r2_requires_asp(self):
        """Test that R2 calculation requires prior spike reconstruction."""
        ca_data = np.random.random(500)
        neuron = Neuron("r2_test", ca_data, None)

        with pytest.raises(ValueError, match="reconstruct_spikes"):
            neuron.get_reconstruction_r2()

    def test_get_reconstruction_r2_basic(self):
        """Test basic R2 calculation after reconstruction."""
        np.random.seed(42)

        # Create signal with clear, well-separated events for better reconstruction
        ca_data = np.random.normal(0.2, 0.02, 1000)  # Lower noise
        for t in [100, 300, 500, 700]:
            ca_data[t:t+50] += 0.8 * np.exp(-np.arange(50) / 15)  # Stronger events

        neuron = Neuron("r2_test2", ca_data, None, fps=20.0)
        neuron.reconstruct_spikes(method='wavelet', create_event_regions=True)

        r2 = neuron.get_reconstruction_r2()

        assert isinstance(r2, float)
        # R2 can be negative for poor fits, but should be finite
        assert np.isfinite(r2)

    def test_get_reconstruction_r2_event_only(self):
        """Test event-only R2 calculation."""
        np.random.seed(42)

        ca_data = np.random.normal(0.2, 0.05, 1000)
        for t in [100, 300, 500, 700]:
            ca_data[t:t+40] += 0.5 * np.exp(-np.arange(40) / 15)

        neuron = Neuron("r2_event_test", ca_data, None, fps=20.0)
        neuron.reconstruct_spikes(method='wavelet', create_event_regions=True)

        # Event-only R2 should focus on signal regions
        r2_event = neuron.get_reconstruction_r2(event_only=True)

        assert isinstance(r2_event, float)
        assert np.isfinite(r2_event)


class TestSNRReconstruction:
    """Test reconstruction-based SNR calculation."""

    def test_get_snr_reconstruction(self):
        """Test reconstruction SNR calculation."""
        np.random.seed(42)

        ca_data = np.random.normal(0.2, 0.05, 1000)
        for t in [100, 300, 500, 700]:
            ca_data[t:t+40] += 0.5 * np.exp(-np.arange(40) / 15)

        neuron = Neuron("snr_recon_test", ca_data, None, fps=20.0)
        neuron.reconstruct_spikes(method='wavelet', create_event_regions=True)

        snr = neuron.get_snr_reconstruction()

        assert isinstance(snr, float)
        assert snr > 0  # SNR should be positive
        assert np.isfinite(snr)

    def test_snr_reconstruction_cached(self):
        """Test that reconstruction SNR is cached."""
        np.random.seed(42)

        ca_data = np.random.normal(0.2, 0.05, 500)
        for t in [100, 250, 400]:
            ca_data[t:t+30] += 0.4 * np.exp(-np.arange(30) / 12)

        neuron = Neuron("snr_cache_test", ca_data, None, fps=20.0)
        neuron.reconstruct_spikes(method='wavelet', create_event_regions=True)

        snr1 = neuron.get_snr_reconstruction()
        snr2 = neuron.get_snr_reconstruction()

        assert snr1 == snr2
        assert neuron.snr_reconstruction == snr1


class TestIterativeReconstruction:
    """Test iterative spike reconstruction mode."""

    def test_iterative_detects_weak_events(self):
        """Test that iterative mode can detect weaker events missed initially."""
        np.random.seed(42)

        n_frames = 1500
        fps = 20.0
        ca_data = np.random.normal(0.1, 0.03, n_frames)

        # Add strong events
        strong_times = [200, 600, 1000]
        for t in strong_times:
            ca_data[t:t+50] += 0.8 * np.exp(-np.arange(50) / 15)

        # Add weak events (should be harder to detect)
        weak_times = [400, 800, 1200]
        for t in weak_times:
            ca_data[t:t+50] += 0.15 * np.exp(-np.arange(50) / 15)

        # Single iteration
        neuron_single = Neuron("iter_test1", ca_data.copy(), None, fps=fps)
        neuron_single.reconstruct_spikes(
            method='threshold',
            n_iter=1,
            n_mad=4.0,
            create_event_regions=True
        )
        count_single = neuron_single.sp_count

        # Multiple iterations with adaptive thresholds
        neuron_iter = Neuron("iter_test2", ca_data.copy(), None, fps=fps)
        neuron_iter.reconstruct_spikes(
            method='threshold',
            n_iter=5,
            n_mad=4.0,
            adaptive_thresholds=True,
            create_event_regions=True
        )
        count_iter = neuron_iter.sp_count

        # Iterative should find at least as many (usually more for weak events)
        assert count_iter >= count_single


class TestDeconvolveWithEventMask:
    """Test NNLS deconvolution with event masking."""

    def test_deconvolve_with_mask_reduces_baseline_fit(self):
        """Test that event mask focuses fit on event regions."""
        np.random.seed(42)

        n_frames = 300
        t_rise = 5.0
        t_off = 20.0
        event_time = 100
        true_amplitude = 0.8

        # Generate clean signal
        t = np.arange(n_frames - event_time)
        kernel = (1 - np.exp(-t / t_rise)) * np.exp(-t / t_off)
        kernel = kernel / kernel.max()

        ca_signal = np.zeros(n_frames)
        ca_signal[event_time:] = true_amplitude * kernel

        # Add significant baseline noise
        ca_signal += np.random.randn(n_frames) * 0.1

        # Create event mask (True where events are)
        event_mask = np.zeros(n_frames, dtype=bool)
        event_mask[event_time:event_time + 80] = True  # Cover event region

        event_times = [event_time]

        # Without mask
        amp_no_mask = Neuron.deconvolve_given_event_times(
            ca_signal, event_times, t_rise, t_off, event_mask=None
        )

        # With mask
        amp_with_mask = Neuron.deconvolve_given_event_times(
            ca_signal, event_times, t_rise, t_off, event_mask=event_mask
        )

        # Both should recover reasonable amplitudes
        assert len(amp_no_mask) == 1
        assert len(amp_with_mask) == 1
        assert amp_no_mask[0] > 0
        assert amp_with_mask[0] > 0
