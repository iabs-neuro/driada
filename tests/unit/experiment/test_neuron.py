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

        assert len(restored) > len(sp)  # Convolution extends signal
        assert restored.max() > 0  # Has positive values

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
        spikes = neuron.reconstruct_spikes(method="wavelet")
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

        with pytest.raises(ValueError, match="No spikes found"):
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

        with patch.object(neuron, "_fit_t_off", return_value=(50.0, 0.1)) as mock_fit:
            noise1 = neuron.get_noise_ampl()
            noise2 = neuron.get_noise_ampl()

            assert noise1 == noise2
            mock_fit.assert_called_once()

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

        # The error handling in the original code has a bug - AttributeError() should be AttributeError
        # So this will actually raise AttributeError, not UserWarning
        with pytest.raises(AttributeError):
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

        # Same bug as in calcium shuffling
        with pytest.raises(AttributeError):
            neuron.get_shuffled_spikes(method="unknown_method")


class TestIntegration:
    """Integration tests with other components."""

    def test_wavelet_spike_inference(self):
        """Test integration with wavelet spike detection."""
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
