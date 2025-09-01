"""Comprehensive tests for exp_base.py module."""

import pytest
import numpy as np
import warnings
from unittest.mock import patch

from driada.experiment.exp_base import (
    Experiment,
    check_dynamic_features,
    STATS_VARS,
    SIGNIFICANCE_VARS,
)
from driada.information.info_base import TimeSeries, MultiTimeSeries
from driada.experiment.neuron import Neuron


class TestDynamicFeatureValidation:
    """Test dynamic feature checking and validation."""

    def test_check_dynamic_features_empty(self):
        """Test validation with empty features."""
        check_dynamic_features({})  # Should not raise

    def test_check_dynamic_features_valid_timeseries(self):
        """Test validation with valid TimeSeries objects."""
        features = {
            "feat1": TimeSeries(np.random.rand(100)),
            "feat2": TimeSeries(np.random.rand(100)),
        }
        # Should not raise
        check_dynamic_features(features)

    def test_check_dynamic_features_valid_arrays(self):
        """Test validation with numpy arrays."""
        features = {"feat1": np.random.rand(100), "feat2": np.random.rand(100)}
        # Should not raise
        check_dynamic_features(features)

    def test_check_dynamic_features_valid_multitimeseries(self):
        """Test validation with MultiTimeSeries objects."""
        ts1 = TimeSeries(np.random.rand(100))
        ts2 = TimeSeries(np.random.rand(100))
        features = {
            "feat1": MultiTimeSeries([ts1, ts2]),
            "feat2": MultiTimeSeries([ts1, ts2]),
        }
        # Should not raise
        check_dynamic_features(features)

    def test_check_dynamic_features_mismatched_lengths(self):
        """Test validation fails with mismatched feature lengths."""
        features = {
            "feat1": TimeSeries(np.random.rand(100)),
            "feat2": TimeSeries(np.random.rand(50)),  # Different length
        }
        with pytest.raises(ValueError, match="Dynamic features have different lengths"):
            check_dynamic_features(features)

    def test_check_dynamic_features_2d_arrays(self):
        """Test validation with 2D numpy arrays."""
        features = {
            "feat1": np.random.rand(3, 100),  # 3 components, 100 timepoints
            "feat2": np.random.rand(2, 100),  # 2 components, 100 timepoints
        }
        # Should not raise - both have 100 timepoints
        check_dynamic_features(features)

    def test_check_dynamic_features_mixed_types(self):
        """Test validation with mixed feature types."""
        features = {
            "feat1": TimeSeries(np.random.rand(100)),
            "feat2": np.random.rand(100),
            "feat3": MultiTimeSeries([TimeSeries(np.random.rand(100))]),
        }
        # Should not raise
        check_dynamic_features(features)

    def test_check_dynamic_features_unsupported_type(self):
        """Test validation with unsupported type."""
        features = {
            "feat1": TimeSeries(np.random.rand(100)),
            "feat2": "invalid_string",  # Not a valid feature type
        }
        with pytest.raises(TypeError, match="unsupported type"):
            check_dynamic_features(features)


class TestExperimentInitialization:
    """Test Experiment class initialization."""

    def test_init_no_calcium_data(self):
        """Test initialization fails without calcium data."""
        with pytest.raises(ValueError, match="Calcium data is required"):
            Experiment(
                signature="test",
                calcium=None,
                spikes=None,
                exp_identificators={},
                static_features={},
                dynamic_features={},
            )

    def test_init_minimal_valid(self):
        """Test minimal valid initialization."""
        calcium = np.random.rand(5, 500)  # Increased to 500 frames

        exp = Experiment(
            signature="test",
            calcium=calcium,
            spikes=None,
            exp_identificators={},
            static_features={"fps": 20.0},
            dynamic_features={},
        )

        assert exp.n_cells == 5
        assert exp.n_frames == 500
        assert exp.signature == "test"
        assert hasattr(exp, "calcium")
        assert hasattr(exp, "spikes")
        assert len(exp.neurons) == 5

    def test_init_with_spikes(self):
        """Test initialization with spike data."""
        calcium = np.random.rand(3, 500)
        spikes = np.random.randint(0, 2, (3, 500))

        exp = Experiment(
            signature="test",
            calcium=calcium,
            spikes=spikes,
            exp_identificators={},
            static_features={"fps": 20.0},
            dynamic_features={},
        )

        assert exp.n_cells == 3
        assert exp.n_frames == 500
        # Check spikes were properly integrated
        assert not np.allclose(exp.spikes.data, 0)

    def test_init_with_spike_reconstruction(self):
        """Test initialization with spike reconstruction."""
        calcium = np.random.rand(5, 500)
        features = {"feat1": TimeSeries(np.random.rand(500))}

        with patch.object(Experiment, "_reconstruct_spikes") as mock_reconstruct:
            # Mock the reconstruction to return appropriate spike data
            mock_spikes = np.random.randint(0, 2, (5, 500))
            mock_reconstruct.return_value = mock_spikes

            with warnings.catch_warnings(record=True) as w:
                exp = Experiment(
                    signature="test",
                    calcium=calcium,
                    spikes=None,
                    exp_identificators={"exp_id": "test_exp"},
                    static_features={"fps": 20.0},
                    dynamic_features=features,
                    reconstruct_spikes="wavelet",
                )

                # Check reconstruction was called
                mock_reconstruct.assert_called_once()
                assert hasattr(exp, "spike_reconstruction_method")
                assert exp.spike_reconstruction_method == "wavelet"

    def test_init_with_bad_frames_mask(self):
        """Test initialization with bad frames mask."""
        calcium = np.random.rand(5, 500)
        spikes = np.zeros((5, 500))
        features = {"feat1": TimeSeries(np.random.rand(500))}
        bad_mask = np.zeros(500, dtype=bool)
        bad_mask[10:20] = True  # Mark frames 10-20 as bad

        exp = Experiment(
            signature="test",
            calcium=calcium,
            spikes=spikes,
            exp_identificators={},
            static_features={"fps": 20.0},
            dynamic_features=features,
            bad_frames_mask=bad_mask,
        )

        assert exp.n_frames == 490  # 500 - 10 bad frames
        assert exp.filtered_flag == True
        assert hasattr(exp, "bad_frames_mask")
        assert np.array_equal(exp.bad_frames_mask, bad_mask)

    def test_init_dynamic_feature_conversion(self):
        """Test automatic conversion of numpy arrays to TimeSeries."""
        calcium = np.random.rand(3, 500)
        spikes = np.zeros((3, 500))

        # Mix of different feature types
        features = {
            "1d_array": np.random.rand(500),
            "2d_array": np.random.rand(2, 500),
            "timeseries": TimeSeries(np.random.rand(500)),
            "multitimeseries": MultiTimeSeries([TimeSeries(np.random.rand(500))]),
        }

        exp = Experiment(
            signature="test",
            calcium=calcium,
            spikes=spikes,
            exp_identificators={},
            static_features={"fps": 20.0},
            dynamic_features=features,
        )

        # Check conversions
        assert isinstance(exp.dynamic_features["1d_array"], TimeSeries)
        assert isinstance(exp.dynamic_features["2d_array"], MultiTimeSeries)
        assert isinstance(exp.dynamic_features["timeseries"], TimeSeries)
        assert isinstance(exp.dynamic_features["multitimeseries"], MultiTimeSeries)

    def test_init_exp_identificators(self):
        """Test that exp_identificators are set as attributes."""
        calcium = np.random.rand(2, 500)
        exp_ids = {"mouse_id": "M123", "session": "S456", "date": "2024-01-01"}

        exp = Experiment(
            signature="test",
            calcium=calcium,
            spikes=None,
            exp_identificators=exp_ids,
            static_features={"fps": 20.0},
            dynamic_features={},
        )

        assert exp.mouse_id == "M123"
        assert exp.session == "S456"
        assert exp.date == "2024-01-01"

    def test_checkpoint_transposed_data(self):
        """Test checkpoint detects transposed data."""
        # calcium shape is (n_cells, n_frames)
        # If n_cells > n_frames, it's likely transposed
        # Use 1000 cells and 500 frames (transposed from typical 500 cells, 1000 frames)
        calcium = np.random.rand(1000, 500)
        features = {"feat": TimeSeries(np.random.rand(500))}

        with pytest.raises(ValueError, match="Data appears to be transposed"):
            Experiment(
                signature="test",
                calcium=calcium,
                spikes=None,
                exp_identificators={},
                static_features={"fps": 20.0},
                dynamic_features=features,
            )

    def test_checkpoint_feature_shape_mismatch(self):
        """Test checkpoint detects feature shape mismatches."""
        calcium = np.random.rand(5, 500)
        # Feature with wrong shape
        features = {"feat": TimeSeries(np.random.rand(50))}  # Should be 500

        with pytest.raises(ValueError, match="inappropriate shape"):
            Experiment(
                signature="test",
                calcium=calcium,
                spikes=None,
                exp_identificators={},
                static_features={"fps": 20.0},
                dynamic_features=features,
            )

    def test_feature_attribute_conflict(self):
        """Test that feature names conflicting with protected attributes are rejected."""
        calcium = np.random.rand(2, 500)
        features = {
            "neurons": TimeSeries(np.random.rand(500)),  # Conflicts with self.neurons
            "valid_feat": TimeSeries(np.random.rand(500)),
        }

        with pytest.raises(
            ValueError,
            match="Dynamic feature names conflict with protected attributes.*neurons",
        ):
            Experiment(
                signature="test",
                calcium=calcium,
                spikes=None,
                exp_identificators={},
                static_features={"fps": 20.0},
                dynamic_features=features,
            )

    def test_static_feature_conflict(self):
        """Test static features with conflicting names get prefixed."""
        calcium = np.random.rand(2, 500)
        static_features = {
            "fps": 20.0,
            "calcium": "this_conflicts",  # Conflicts with self.calcium
        }

        with warnings.catch_warnings(record=True) as w:
            exp = Experiment(
                signature="test",
                calcium=calcium,
                spikes=None,
                exp_identificators={},
                static_features=static_features,
                dynamic_features={},
            )

            # Check warning was issued
            assert any(
                "conflicts with protected attribute" in str(warning.message)
                for warning in w
            )
            # Should be accessible with underscore
            assert hasattr(exp, "_calcium")
            assert exp._calcium == "this_conflicts"
            # Original calcium should be untouched
            assert isinstance(exp.calcium, MultiTimeSeries)

    def test_init_with_3d_feature(self):
        """Test initialization fails with 3D feature array."""
        calcium = np.random.rand(2, 500)
        features = {"feat3d": np.random.rand(2, 3, 500)}  # 3D array

        with pytest.raises(ValueError, match="unsupported dimensionality: 3D"):
            Experiment(
                signature="test",
                calcium=calcium,
                spikes=None,
                exp_identificators={},
                static_features={"fps": 20.0},
                dynamic_features=features,
            )


class TestExperimentMethods:
    """Test various Experiment methods."""

    @pytest.fixture
    def basic_experiment(self):
        """Create a basic experiment for testing."""
        np.random.seed(42)  # For reproducibility
        calcium = np.random.rand(5, 500)
        spikes = np.random.randint(0, 2, (5, 500))
        features = {
            "feat1": TimeSeries(np.random.rand(500)),
            "feat2": TimeSeries(np.random.rand(500)),
        }

        # Create experiment with proper shuffle mask to avoid issues
        exp = Experiment(
            signature="test",
            calcium=calcium,
            spikes=spikes,
            exp_identificators={"exp_id": "test"},
            static_features={"fps": 20.0, "t_rise_sec": 0.25, "t_off_sec": 2.0},
            dynamic_features=features,
        )

        # Override the calcium shuffle mask to ensure tests work
        if hasattr(exp.calcium, "shuffle_mask"):
            # Create a less restrictive mask for testing
            exp.calcium.shuffle_mask = np.ones(500, dtype=bool)
            exp.calcium.shuffle_mask[:10] = False  # Only mask first 10 frames
            exp.calcium.shuffle_mask[-10:] = False  # And last 10 frames

        return exp

    def test_check_ds(self, basic_experiment, capsys):
        """Test downsampling check."""
        # Normal ds - no warning
        basic_experiment.check_ds(1)
        captured = capsys.readouterr()
        assert "too high" not in captured.err

        # High ds - should warn
        basic_experiment.check_ds(10)
        captured = capsys.readouterr()
        assert "too high" in captured.err

    def test_process_cbunch(self, basic_experiment):
        """Test cell bunch processing."""
        # Single int
        assert basic_experiment._process_cbunch(0) == [0]

        # None - all cells
        assert basic_experiment._process_cbunch(None) == [0, 1, 2, 3, 4]

        # List
        assert basic_experiment._process_cbunch([1, 3]) == [1, 3]

    def test_process_fbunch(self, basic_experiment):
        """Test feature bunch processing."""
        # Single string
        assert basic_experiment._process_fbunch("feat1") == ["feat1"]

        # None - all features
        assert set(basic_experiment._process_fbunch(None)) == {"feat1", "feat2"}

        # List
        assert basic_experiment._process_fbunch(["feat1"]) == ["feat1"]

    def test_process_fbunch_multifeatures(self, basic_experiment):
        """Test processing of multifeatures."""
        # Enable multifeatures
        multi = basic_experiment._process_fbunch(
            [("feat1", "feat2")], allow_multifeatures=True
        )
        assert multi == [("feat1", "feat2")]

        # Disable multifeatures - should raise
        with pytest.raises(ValueError, match="Multifeature detected"):
            basic_experiment._process_fbunch(
                [("feat1", "feat2")], allow_multifeatures=False
            )

    def test_process_sbunch(self, basic_experiment):
        """Test stats bunch processing."""
        # Single string
        assert basic_experiment._process_sbunch("me") == ["me"]

        # None - all stats
        assert basic_experiment._process_sbunch(None) == STATS_VARS

        # Significance mode
        assert (
            basic_experiment._process_sbunch(None, significance_mode=True)
            == SIGNIFICANCE_VARS
        )

    def test_build_pair_hash(self, basic_experiment):
        """Test hash building for cell-feature pairs."""
        # Single feature
        hash1 = basic_experiment._build_pair_hash(0, "feat1")
        assert isinstance(hash1, tuple)
        assert len(hash1) == 2

        # Multifeature
        hash2 = basic_experiment._build_pair_hash(0, ("feat1", "feat2"))
        assert isinstance(hash2, tuple)
        assert len(hash2) == 3  # cell + 2 features

    def test_get_feature_entropy(self, basic_experiment):
        """Test feature entropy calculation."""
        # Single feature - with warning since it's continuous
        with warnings.catch_warnings(record=True) as w:
            entropy = basic_experiment.get_feature_entropy("feat1")
            assert isinstance(entropy, float)
            # Continuous entropy can be negative
            assert any("continuous" in str(warning.message) for warning in w)

        # Multifeature - test with 2 features
        with warnings.catch_warnings(record=True) as w:
            multi_entropy = basic_experiment.get_feature_entropy(("feat1", "feat2"))
            assert isinstance(multi_entropy, float)
            # Should warn about continuous components
            assert any("continuous" in str(warning.message) for warning in w)

    def test_stats_table_initialization(self, basic_experiment):
        """Test stats tables are properly initialized."""
        # Verify stats_tables exists and is initialized
        assert hasattr(basic_experiment, "stats_tables")
        assert isinstance(basic_experiment.stats_tables, dict)

        # Verify significance_tables exists
        assert hasattr(basic_experiment, "significance_tables")
        assert isinstance(basic_experiment.significance_tables, dict)

        # Initially empty until _set_selectivity_tables is called
        assert len(basic_experiment.stats_tables) == 0
        assert basic_experiment.selectivity_tables_initialized == False

    def test_hash_methods(self, basic_experiment):
        """Test hash computation methods."""
        # Test _build_pair_hash for single feature
        pair_hash = basic_experiment._build_pair_hash(0, "feat1")
        assert isinstance(pair_hash, tuple)
        assert len(pair_hash) == 2
        assert all(isinstance(h, str) for h in pair_hash)

        # Same pair should produce same hash
        pair_hash2 = basic_experiment._build_pair_hash(0, "feat1")
        assert pair_hash == pair_hash2

        # Different cell should produce different hash
        pair_hash3 = basic_experiment._build_pair_hash(1, "feat1")
        assert pair_hash != pair_hash3

        # Different feature should produce different hash
        pair_hash4 = basic_experiment._build_pair_hash(0, "feat2")
        assert pair_hash != pair_hash4

        # Test multifeature hash
        multi_hash = basic_experiment._build_pair_hash(0, ("feat1", "feat2"))
        assert isinstance(multi_hash, tuple)
        assert len(multi_hash) == 3  # cell + 2 features

    def test_spikes_detection(self, basic_experiment):
        """Test spike data presence."""
        # Create experiment with no spikes and disable reconstruction
        calcium = np.random.rand(3, 500)
        no_spikes = np.zeros((3, 500))

        exp_no_spikes = Experiment(
            signature="test",
            calcium=calcium,
            spikes=no_spikes,
            exp_identificators={},
            static_features={"fps": 20.0},
            dynamic_features={},
            reconstruct_spikes=None,  # Disable spike reconstruction
        )

        # Check if spikes are all zeros (when reconstruction is disabled)
        assert not np.any(exp_no_spikes.spikes.data)

        # basic_experiment has spikes
        assert np.any(basic_experiment.spikes.data)

    def test_trim_data_with_bad_frames(self, basic_experiment):
        """Test data trimming with bad frames mask."""
        # Create test data
        calcium = np.random.rand(3, 500)
        spikes = np.random.randint(0, 2, (3, 500))
        features = {
            "feat1": TimeSeries(np.random.rand(500)),
            "feat2": np.random.rand(2, 500),  # 2D array
        }
        bad_mask = np.zeros(500, dtype=bool)
        bad_mask[100:110] = True  # Mark 10 frames as bad

        # Test trimming
        f_calcium, f_spikes, f_features = basic_experiment._trim_data(
            calcium, spikes, features, bad_mask
        )

        # Check shapes
        assert f_calcium.shape == (3, 490)
        assert f_spikes.shape == (3, 490)
        assert isinstance(f_features["feat1"], TimeSeries)
        assert f_features["feat1"].data.shape == (490,)
        assert isinstance(f_features["feat2"], TimeSeries) or f_features[
            "feat2"
        ].shape == (2, 490)

    def test_data_hashes(self, basic_experiment):
        """Test data hash storage."""
        # Check that _data_hashes was built
        assert hasattr(basic_experiment, "_data_hashes")
        assert "calcium" in basic_experiment._data_hashes
        assert "spikes" in basic_experiment._data_hashes

        # Check structure
        for feat in basic_experiment.dynamic_features:
            assert feat in basic_experiment._data_hashes["calcium"]
            assert feat in basic_experiment._data_hashes["spikes"]

            # Each feature should have hashes for all cells
            assert (
                len(basic_experiment._data_hashes["calcium"][feat])
                == basic_experiment.n_cells
            )
            assert (
                len(basic_experiment._data_hashes["spikes"][feat])
                == basic_experiment.n_cells
            )

    def test_multicell_shuffled_data(self, basic_experiment):
        """Test multi-cell shuffle methods."""
        # Test calcium shuffling
        shuffled_ca = basic_experiment.get_multicell_shuffled_calcium(
            cbunch=[0, 1], method="roll_based"
        )
        assert shuffled_ca.shape == (2, basic_experiment.n_frames)
        # Should be different from original (with high probability)
        orig_ca = basic_experiment.calcium.data[[0, 1], :]
        assert not np.allclose(shuffled_ca, orig_ca)

        # Test invalid method
        with pytest.raises(ValueError, match="Invalid shuffling method 'invalid_method'"):
            basic_experiment.get_multicell_shuffled_calcium(
                cbunch=[0], method="invalid_method"
            )

        # Test spike shuffling
        if np.any(basic_experiment.spikes.data):
            shuffled_sp = basic_experiment.get_multicell_shuffled_spikes(
                cbunch=[0, 1], method="isi_based"
            )
            assert shuffled_sp.shape == (2, basic_experiment.n_frames)

    def test_neuron_list_creation(self, basic_experiment):
        """Test that neurons are properly created."""
        assert len(basic_experiment.neurons) == basic_experiment.n_cells

        # Check each neuron
        for i, neuron in enumerate(basic_experiment.neurons):
            assert isinstance(neuron, Neuron)
            assert neuron.cell_id == str(i)  # cell_id is string
            # Verify neuron has correct data shape
            assert neuron.ca.data.shape[0] == basic_experiment.n_frames
            if hasattr(neuron, "sp") and neuron.sp is not None:
                assert neuron.sp.data.shape[0] == basic_experiment.n_frames

    def test_set_selectivity_tables(self, basic_experiment):
        """Test selectivity tables initialization."""
        # Initialize tables for calcium mode
        basic_experiment._set_selectivity_tables("calcium")

        assert "calcium" in basic_experiment.stats_tables
        assert "calcium" in basic_experiment.significance_tables
        assert basic_experiment.selectivity_tables_initialized == True

        # Check structure
        for feat in basic_experiment.dynamic_features:
            assert feat in basic_experiment.stats_tables["calcium"]
            for cell_id in range(basic_experiment.n_cells):
                assert cell_id in basic_experiment.stats_tables["calcium"][feat]
                # Check default values
                stats = basic_experiment.stats_tables["calcium"][feat][cell_id]
                assert all(stats[key] is None for key in STATS_VARS)

    def test_update_neuron_feature_pair_stats(self, basic_experiment):
        """Test updating neuron-feature pair statistics."""
        # Initialize tables first
        basic_experiment._set_selectivity_tables("calcium")

        # Update stats for a specific cell-feature pair
        test_stats = {"data_hash": "test_hash", "me": 0.5, "pval": 0.01}

        basic_experiment.update_neuron_feature_pair_stats(
            test_stats, cell_id=0, feat_id="feat1", mode="calcium"
        )

        # Verify update
        stored_stats = basic_experiment.stats_tables["calcium"]["feat1"][0]
        assert stored_stats["data_hash"] == "test_hash"
        assert stored_stats["me"] == 0.5
        assert stored_stats["pval"] == 0.01

    def test_get_stats_slice(self, basic_experiment):
        """Test retrieving statistics slice."""
        # Initialize and populate some stats
        basic_experiment._set_selectivity_tables("calcium")

        # Add some test data
        basic_experiment.stats_tables["calcium"]["feat1"][0]["me"] = 0.5
        basic_experiment.stats_tables["calcium"]["feat1"][1]["me"] = 0.7
        basic_experiment.stats_tables["calcium"]["feat2"][0]["me"] = 0.3

        # Get slice for specific cells and features
        slice_data = basic_experiment.get_stats_slice(
            cbunch=[0, 1], fbunch=["feat1"], sbunch=["me"]
        )

        assert "feat1" in slice_data
        assert 0 in slice_data["feat1"]
        assert 1 in slice_data["feat1"]
        assert slice_data["feat1"][0]["me"] == 0.5
        assert slice_data["feat1"][1]["me"] == 0.7
        assert "feat2" not in slice_data  # Not requested

    def test_get_feature_entropy_multitimeseries(self, basic_experiment):
        """Test entropy calculation for MultiTimeSeries features."""
        # Create a MultiTimeSeries feature
        ts1 = TimeSeries(np.random.rand(500), discrete=False)
        ts2 = TimeSeries(np.random.rand(500), discrete=False)
        mts = MultiTimeSeries([ts1, ts2])
        basic_experiment.dynamic_features["multi_feat"] = mts

        # Test entropy calculation with warning
        with warnings.catch_warnings(record=True) as w:
            entropy = basic_experiment.get_feature_entropy("multi_feat")
            assert isinstance(entropy, float)
            # Should warn about continuous components
            assert any("continuous components" in str(warning.message) for warning in w)

    def test_get_feature_entropy_errors(self, basic_experiment):
        """Test error cases for get_feature_entropy."""
        # Test with invalid feature ID type
        with pytest.raises(TypeError, match="feat_id must be str or tuple"):
            basic_experiment.get_feature_entropy(123)

        # Test with tuple of wrong length
        with pytest.raises(ValueError, match="exactly 2 variables"):
            basic_experiment.get_feature_entropy(("feat1", "feat2", "feat3"))

    def test_store_and_get_embedding(self, basic_experiment):
        """Test embedding storage and retrieval."""
        # Create test embedding
        embedding = np.random.rand(500, 3)  # 500 timepoints, 3 components
        metadata = {"method": "pca", "n_components": 3}

        # Store embedding
        basic_experiment.store_embedding(
            embedding, "pca", data_type="calcium", metadata=metadata
        )

        # Retrieve embedding
        result = basic_experiment.get_embedding("pca", data_type="calcium")
        assert "data" in result
        assert "metadata" in result
        np.testing.assert_array_equal(result["data"], embedding)
        assert result["metadata"]["method"] == "pca"

        # Test invalid data type
        with pytest.raises(ValueError, match="data_type must be"):
            basic_experiment.store_embedding(embedding, "pca", data_type="invalid")

        # Test missing embedding
        with pytest.raises(KeyError, match="No embedding found"):
            basic_experiment.get_embedding("umap", data_type="calcium")

    def test_get_significant_neurons(self, basic_experiment):
        """Test getting significant neurons."""
        # Initialize tables
        basic_experiment._set_selectivity_tables("calcium")

        # Mark some neurons as significant
        basic_experiment.significance_tables["calcium"]["feat1"][0]["stage2"] = True
        basic_experiment.significance_tables["calcium"]["feat1"][1]["stage2"] = True
        basic_experiment.significance_tables["calcium"]["feat2"][0]["stage2"] = True

        # Get significant neurons
        sig_neurons = basic_experiment.get_significant_neurons(min_nspec=2)

        # Only neuron 0 has 2 significant features
        assert 0 in sig_neurons
        assert 1 not in sig_neurons
        assert set(sig_neurons[0]) == {"feat1", "feat2"}
        
    def test_get_significant_neurons_with_override(self, basic_experiment):
        """Test getting significant neurons with override_intense_significance."""
        # Initialize tables
        basic_experiment._set_selectivity_tables("calcium")
        
        # Add p-values to stats tables for all neurons
        basic_experiment.stats_tables["calcium"]["feat1"][0]["pval"] = 0.001
        basic_experiment.stats_tables["calcium"]["feat1"][1]["pval"] = 0.02
        basic_experiment.stats_tables["calcium"]["feat1"][2]["pval"] = 0.08
        basic_experiment.stats_tables["calcium"]["feat1"][3]["pval"] = 0.5
        basic_experiment.stats_tables["calcium"]["feat1"][4]["pval"] = 0.7
        basic_experiment.stats_tables["calcium"]["feat2"][0]["pval"] = 0.03
        basic_experiment.stats_tables["calcium"]["feat2"][1]["pval"] = 0.15
        basic_experiment.stats_tables["calcium"]["feat2"][2]["pval"] = 0.9
        basic_experiment.stats_tables["calcium"]["feat2"][3]["pval"] = 0.8
        basic_experiment.stats_tables["calcium"]["feat2"][4]["pval"] = 0.6
        
        # Mark all as NOT significant in the original INTENSE analysis
        for feat in ["feat1", "feat2"]:
            for cell in range(5):  # 5 neurons in basic_experiment
                basic_experiment.significance_tables["calcium"][feat][cell]["stage2"] = False
        
        # Test 1: Use original INTENSE significance (all False)
        sig_neurons = basic_experiment.get_significant_neurons(
            override_intense_significance=False
        )
        assert len(sig_neurons) == 0
        
        # Test 2: Override with pval_thr=0.05, no correction
        sig_neurons = basic_experiment.get_significant_neurons(
            override_intense_significance=True,
            pval_thr=0.05,
            multicomp_correction=None
        )
        # Neurons with p < 0.05: neuron 0 (feat1, feat2), neuron 1 (feat1)
        assert 0 in sig_neurons
        assert 1 in sig_neurons
        assert 2 not in sig_neurons
        assert set(sig_neurons[0]) == {"feat1", "feat2"}
        assert set(sig_neurons[1]) == {"feat1"}
        
        # Test 3: Override with stricter threshold
        sig_neurons = basic_experiment.get_significant_neurons(
            override_intense_significance=True,
            pval_thr=0.01,
            multicomp_correction=None
        )
        # Only neuron 0 with feat1 (p=0.001)
        assert 0 in sig_neurons
        assert 1 not in sig_neurons
        assert set(sig_neurons[0]) == {"feat1"}
        
        # Test 4: With Bonferroni correction (10 tests total: 5 neurons × 2 features)
        sig_neurons = basic_experiment.get_significant_neurons(
            override_intense_significance=True,
            pval_thr=0.05,
            multicomp_correction="bonferroni"
        )
        # Bonferroni threshold = 0.05/10 = 0.005
        # Only neuron 0 with feat1 (p=0.001) passes
        assert 0 in sig_neurons
        assert 1 not in sig_neurons
        assert set(sig_neurons[0]) == {"feat1"}
        
        # Test 5: Update significance tables
        sig_neurons = basic_experiment.get_significant_neurons(
            override_intense_significance=True,
            pval_thr=0.05,
            multicomp_correction=None,
            significance_update=True
        )
        
        # Check that significance tables were updated
        assert basic_experiment.significance_tables["calcium"]["feat1"][0]["stage2"] == True
        assert basic_experiment.significance_tables["calcium"]["feat1"][1]["stage2"] == True
        assert basic_experiment.significance_tables["calcium"]["feat1"][2]["stage2"] == False
        assert basic_experiment.significance_tables["calcium"]["feat2"][0]["stage2"] == True
        assert basic_experiment.significance_tables["calcium"]["feat2"][1]["stage2"] == False
        
        # Check additional fields were added
        assert basic_experiment.significance_tables["calcium"]["feat1"][0]["pval_thr"] == 0.05
        assert basic_experiment.significance_tables["calcium"]["feat1"][0]["multicomp_correction"] is None
        assert basic_experiment.significance_tables["calcium"]["feat1"][0]["corrected_pval_thr"] == 0.05

    def test_multicell_shuffled_spikes_error(self, basic_experiment):
        """Test error handling for spike shuffling."""
        # Create experiment with zero spikes
        calcium = np.random.rand(3, 500)
        zero_spikes = np.zeros((3, 500))

        exp = Experiment(
            signature="test",
            calcium=calcium,
            spikes=zero_spikes,
            exp_identificators={},
            static_features={"fps": 20.0},
            dynamic_features={},
            reconstruct_spikes=None,
        )

        # Should raise error when trying to shuffle zero spikes
        with pytest.raises(AttributeError, match="Unable to shuffle spikes"):
            exp.get_multicell_shuffled_spikes()

    def test_get_neuron_feature_pair_stats(self, basic_experiment):
        """Test retrieving neuron-feature pair stats."""
        # Initialize tables
        basic_experiment._set_selectivity_tables("calcium")

        # Add test data with proper hash
        cell_id = 0
        feat_id = "feat1"
        pair_hash = basic_experiment._build_pair_hash(cell_id, feat_id)
        basic_experiment.stats_tables["calcium"][feat_id][cell_id][
            "data_hash"
        ] = pair_hash
        basic_experiment.stats_tables["calcium"][feat_id][cell_id]["me"] = 0.5

        # Retrieve stats
        stats = basic_experiment.get_neuron_feature_pair_stats(cell_id, feat_id)
        assert stats is not None
        assert stats["me"] == 0.5

    def test_get_significance_slice(self, basic_experiment):
        """Test retrieving significance slice."""
        # Initialize tables
        basic_experiment._set_selectivity_tables("calcium")

        # Add test significance data
        basic_experiment.significance_tables["calcium"]["feat1"][0]["stage1"] = True
        basic_experiment.significance_tables["calcium"]["feat1"][0]["stage2"] = True

        # Get significance slice
        sig_slice = basic_experiment.get_significance_slice(
            cbunch=[0], fbunch=["feat1"], sbunch=["stage1", "stage2"]
        )

        assert sig_slice["feat1"][0]["stage1"] == True
        assert sig_slice["feat1"][0]["stage2"] == True

    def test_update_neuron_feature_pair_significance(self, basic_experiment):
        """Test updating neuron-feature pair significance."""
        # Initialize tables
        basic_experiment._set_selectivity_tables("calcium")

        # Set up data hash first
        cell_id = 0
        feat_id = "feat1"
        pair_hash = basic_experiment._build_pair_hash(cell_id, feat_id)
        basic_experiment.stats_tables["calcium"][feat_id][cell_id][
            "data_hash"
        ] = pair_hash

        # Update significance
        sig_data = {"stage1": True, "stage2": False, "shuffles1": 1000}
        basic_experiment.update_neuron_feature_pair_significance(
            sig_data, cell_id, feat_id
        )

        # Verify
        stored_sig = basic_experiment.significance_tables["calcium"][feat_id][cell_id]
        assert stored_sig["stage1"] == True
        assert stored_sig["stage2"] == False
        assert stored_sig["shuffles1"] == 1000

    def test_add_multifeature_to_stats(self, basic_experiment):
        """Test adding multifeature to stats tables."""
        # Initialize tables
        basic_experiment._set_selectivity_tables("calcium")

        # Add multifeature
        multifeature = ("feat1", "feat2")
        basic_experiment._add_multifeature_to_stats(multifeature)

        # Check it was added
        ordered_feat = tuple(sorted(multifeature))
        assert ordered_feat in basic_experiment.stats_tables["calcium"]
        assert ordered_feat in basic_experiment.significance_tables["calcium"]

        # Check structure
        for cell_id in range(basic_experiment.n_cells):
            assert cell_id in basic_experiment.stats_tables["calcium"][ordered_feat]

    def test_compute_rdm(self, basic_experiment):
        """Test RDM computation with caching."""
        # Add a categorical feature for RDM
        conditions = np.array([0, 0, 1, 1, 2, 2] * 83 + [0, 0])  # Total 500
        basic_experiment.dynamic_features["conditions"] = TimeSeries(
            conditions, discrete=True
        )

        # Mock the compute_experiment_rdm function
        with patch("driada.rsa.integration.compute_experiment_rdm") as mock_compute:
            mock_rdm = np.random.rand(3, 3)
            mock_labels = np.array([0, 1, 2])
            mock_compute.return_value = (mock_rdm, mock_labels)

            # First call - should compute
            rdm1, labels1 = basic_experiment.compute_rdm("conditions")
            assert mock_compute.call_count == 1
            np.testing.assert_array_equal(rdm1, mock_rdm)

            # Second call - should use cache
            rdm2, labels2 = basic_experiment.compute_rdm("conditions")
            assert mock_compute.call_count == 1  # Still 1, used cache
            np.testing.assert_array_equal(rdm2, mock_rdm)

            # Clear cache and call again
            basic_experiment.clear_rdm_cache()
            rdm3, labels3 = basic_experiment.compute_rdm("conditions")
            assert mock_compute.call_count == 2  # Computed again

    def test_process_fbunch_with_stats_table(self, basic_experiment):
        """Test _process_fbunch when stats table exists."""
        # Initialize stats tables with a multifeature
        basic_experiment._set_selectivity_tables("calcium")
        multifeature = ("feat1", "feat2")
        basic_experiment._add_multifeature_to_stats(multifeature)

        # Process fbunch with None - should include multifeature
        feat_ids = basic_experiment._process_fbunch(None, allow_multifeatures=True)

        # Should contain both single features and multifeature
        assert "feat1" in feat_ids
        assert "feat2" in feat_ids
        assert tuple(sorted(multifeature)) in feat_ids

    def test_embedding_shape_validation(self, basic_experiment):
        """Test embedding shape validation with downsampling."""
        # Test with downsampling
        ds = 2
        expected_frames = basic_experiment.n_frames // ds
        embedding = np.random.rand(expected_frames, 3)

        # Store with downsampling metadata
        basic_experiment.store_embedding(embedding, "pca_ds2", metadata={"ds": ds})

        # Test with wrong shape
        wrong_embedding = np.random.rand(100, 3)  # Wrong number of frames
        with pytest.raises(ValueError, match="Embedding timepoints"):
            basic_experiment.store_embedding(wrong_embedding, "wrong_shape")

    def test_check_stats_relevance_multifeature(self, basic_experiment):
        """Test stats relevance checking for multifeatures."""
        # Initialize tables
        basic_experiment._set_selectivity_tables("calcium")

        # Add multifeature to stats
        multifeature = ["feat2", "feat1"]  # Intentionally unordered
        basic_experiment._add_multifeature_to_stats(multifeature)

        # Add hash for the multifeature
        ordered_feat = ("feat1", "feat2")
        basic_experiment._add_multifeature_to_data_hashes(multifeature)

        # Set data hash to match
        cell_id = 0
        pair_hash = basic_experiment._data_hashes["calcium"][ordered_feat][cell_id]
        basic_experiment.stats_tables["calcium"][ordered_feat][cell_id][
            "data_hash"
        ] = pair_hash

        # Check relevance - should pass
        assert basic_experiment._check_stats_relevance(cell_id, multifeature) == True

    def test_update_stats_force_update(self, basic_experiment):
        """Test force updating stats when data hash mismatch."""
        # Initialize tables
        basic_experiment._set_selectivity_tables("calcium")

        # Set mismatched hash
        cell_id = 0
        feat_id = "feat1"
        basic_experiment.stats_tables["calcium"][feat_id][cell_id][
            "data_hash"
        ] = "old_hash"

        # Try to update without force - should print warning
        new_stats = {"me": 0.8}
        with patch("builtins.print") as mock_print:
            basic_experiment.update_neuron_feature_pair_stats(
                new_stats, cell_id, feat_id, force_update=False
            )
            mock_print.assert_called()  # Should print warning

        # Stats should not be updated
        assert basic_experiment.stats_tables["calcium"][feat_id][cell_id]["me"] is None

        # Now force update
        basic_experiment.update_neuron_feature_pair_stats(
            new_stats, cell_id, feat_id, force_update=True
        )

        # Stats should be updated
        assert basic_experiment.stats_tables["calcium"][feat_id][cell_id]["me"] == 0.8

    def test_check_ds_no_fps(self, basic_experiment):
        """Test check_ds when fps is not set."""
        # Remove fps attribute
        if hasattr(basic_experiment, "fps"):
            delattr(basic_experiment, "fps")

        # Should raise error
        with pytest.raises(ValueError, match="fps not set"):
            basic_experiment.check_ds(1)

    def test_trim_data_already_filtered(self, basic_experiment):
        """Test trim_data when already filtered."""
        # Set filtered flag
        basic_experiment.filtered_flag = True

        # Try to filter again without force
        calcium = np.random.rand(3, 500)
        spikes = np.random.rand(3, 500)
        features = {"feat": TimeSeries(np.random.rand(500))}
        bad_mask = np.zeros(500, dtype=bool)
        bad_mask[10:20] = True

        with pytest.raises(AttributeError, match="Data is already filtered"):
            basic_experiment._trim_data(calcium, spikes, features, bad_mask)

    def test_add_multifeature_already_exists(self, basic_experiment):
        """Test adding multifeature that already exists."""
        # Initialize tables
        basic_experiment._set_selectivity_tables("calcium")

        # Add multifeature first time
        multifeature = ("feat1", "feat2")
        basic_experiment._add_multifeature_to_stats(multifeature)

        # Add same multifeature again - should not print anything
        with patch("builtins.print") as mock_print:
            basic_experiment._add_multifeature_to_stats(multifeature)
            mock_print.assert_not_called()  # Should not print

    def test_get_stats_relevance_missing_feature(self, basic_experiment):
        """Test stats relevance check for missing feature."""
        # Initialize tables
        basic_experiment._set_selectivity_tables("calcium")

        # Try to check non-existent feature
        with pytest.raises(ValueError, match="Feature .* is not present in dynamic_features"):
            basic_experiment._check_stats_relevance(0, "non_existent_feat")
